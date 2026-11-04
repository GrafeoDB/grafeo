//! Relationship expansion planning.

use grafeo_common::types::Value;
use grafeo_core::execution::operators::{IntersectExpandOperator, IntersectSide, SeekKey};

use super::reachability::ReachabilityMode;
use super::seek::constant_seek;
use super::{
    Arc, Direction, Error, ExecutionPathMode, ExpandDirection, ExpandOp, ExpandOperator, FilterOp,
    GraphStore, LogicalExpression, LogicalOperator, Operator, PathMode, Result,
    VariableLengthExpandOperator,
};
use crate::query::plan::PushdownHint;

/// A condition on every edge of the path a variable-length expand binds: the
/// edge is `hops` (each of them names it), the predicate may read the input
/// row's columns too.
pub(crate) struct PendingEdgeCondition {
    /// The path variable of the expand that checks it.
    pub(crate) path: String,
    /// The names the predicate reads the candidate edge as.
    pub(crate) hops: Vec<String>,
    /// The condition.
    pub(crate) predicate: LogicalExpression,
}

/// What a filter knows of the target of the expand below it (see
/// [`Planner::note_hop_target`](super::Planner::note_hop_target)).
pub(super) enum HopTarget {
    /// It is the node in this column of the input.
    Column(String),
    /// It holds this key, with these labels.
    Key(SeekKey, Value, grafeo_core::execution::operators::KeyLabels),
    /// It is the node in this column, and the hop below goes with it: the two
    /// intersect their lists.
    Intersect(String),
}

impl super::Planner {
    /// The hops `near` and `closing` as one intersection of the lists of `near`'s
    /// source and `target`, with the columns the two expands give.
    #[inline(never)]
    fn plan_intersection(
        &self,
        near: &ExpandOp,
        closing: &ExpandOp,
        target: &str,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input, mut columns) = self.plan_operator(&near.input)?;
        let column_of = |name: &str| {
            columns
                .iter()
                .position(|column| column == name)
                .ok_or_else(|| {
                    Error::Internal(format!("Variable '{name}' not found in input columns"))
                })
        };
        let side = |column, direction, types: &[String]| -> Result<IntersectSide> {
            Ok(IntersectSide {
                column,
                direction,
                edge_types: self
                    .stored_edge_types(types)?
                    .into_iter()
                    .map(Into::into)
                    .collect(),
            })
        };
        let direction_of = |direction: ExpandDirection| match direction {
            ExpandDirection::Outgoing => Direction::Outgoing,
            ExpandDirection::Incoming => Direction::Incoming,
            ExpandDirection::Both => Direction::Both,
        };
        let near_side = side(
            column_of(&near.from_variable)?,
            direction_of(near.direction),
            &near.edge_types,
        )?;
        // A hop that goes on from where they meet is read from the node it comes
        // back to; one that ends there, from the node it starts at.
        let ends_there = near.to_variable == target;
        let far_side = if ends_there {
            side(
                column_of(&closing.from_variable)?,
                direction_of(closing.direction),
                &closing.edge_types,
            )?
        } else {
            let back = match closing.direction {
                ExpandDirection::Outgoing => Direction::Incoming,
                ExpandDirection::Incoming => Direction::Outgoing,
                ExpandDirection::Both => Direction::Both,
            };
            side(column_of(target)?, back, &closing.edge_types)?
        };
        let operator = IntersectExpandOperator::new(
            Arc::clone(&self.store) as Arc<dyn GraphStore>,
            input,
            near_side,
            far_side,
            self.read_view(),
        )
        .with_read_counter(self.read_counter());
        let operator = if ends_there {
            operator.ending_at_the_common_node()
        } else {
            operator
        };
        columns.push(self.register_edge_column(&near.edge_variable));
        columns.push(near.to_variable.clone());
        columns.push(self.register_edge_column(&closing.edge_variable));
        columns.push(closing.to_variable.clone());
        self.record_absorbed_scan_entry("Expand", &LogicalOperator::Expand(near.clone()));
        Ok((Box::new(operator), columns))
    }

    /// Hands the expand below `filter` what the filter's hint says of its
    /// target. A key needs a literal value (parameters are filled in by now)
    /// and a statement that reads the store as planned and whose writes see no
    /// row of it while it runs: the expand finds the key once, through the
    /// indexes the hint names, and lists every edge if one has gone since.
    pub(super) fn note_hop_target(&self, filter: &FilterOp) {
        let mut below = filter.input.as_ref();
        while let LogicalOperator::Filter(check) = below {
            below = &check.input;
        }
        let LogicalOperator::Expand(expand) = below else {
            return;
        };
        let target = match &filter.pushdown_hint {
            Some(PushdownHint::ExpandInto { target }) => Some(HopTarget::Column(target.clone())),
            Some(PushdownHint::IntersectExpand { target }) => {
                Some(HopTarget::Intersect(target.clone()))
            }
            Some(PushdownHint::ExpandToKey { key, labels })
                if self.reads_the_store_as_planned()
                    && self
                        .statement_writes
                        .get()
                        .is_some_and(|writes| !writes.per_row) =>
            {
                constant_seek(&filter.predicate, &expand.to_variable, |property| {
                    property == key
                })
                .and_then(|seek| match seek.value {
                    LogicalExpression::Literal(value) => {
                        Some(HopTarget::Key(seek.key, value.clone(), labels.clone()))
                    }
                    _ => None,
                })
            }
            _ => None,
        };
        if let Some(target) = target {
            self.hop_targets
                .borrow_mut()
                .insert(expand.to_variable.clone(), target);
        }
    }

    /// Plans an expand operator.
    pub(super) fn plan_expand(
        &self,
        expand: &ExpandOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let hop_target = self.hop_targets.borrow_mut().remove(&expand.to_variable);
        if let Some(HopTarget::Intersect(target)) = &hop_target
            && let LogicalOperator::Expand(near) = expand.input.as_ref()
        {
            return self.plan_intersection(near, expand, target);
        }

        // Plan the input operator first
        let (input_op, input_columns) = self.plan_operator(&expand.input)?;
        self.expand_over(expand, hop_target, input_op, input_columns)
    }

    /// The expand `expand` over its planned input. Not part of [`plan_expand`](Self::plan_expand):
    /// that frame is on the stack once per hop of the plan (see `crate::query::limits`), and the
    /// operators built here by value would make it larger.
    #[inline(never)]
    fn expand_over(
        &self,
        expand: &ExpandOp,
        hop_target: Option<HopTarget>,
        input_op: Box<dyn Operator>,
        input_columns: Vec<String>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Find the source column index
        let source_column = input_columns
            .iter()
            .position(|c| c == &expand.from_variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Source variable '{}' not found in input columns",
                    expand.from_variable
                ))
            })?;

        // Convert expand direction
        let direction = match expand.direction {
            ExpandDirection::Outgoing => Direction::Outgoing,
            ExpandDirection::Incoming => Direction::Incoming,
            ExpandDirection::Both => Direction::Both,
        };

        let is_variable_length = expand.is_variable_length();

        // Use VariableLengthExpandOperator when multi-hop OR when a named path
        // needs path detail columns (length, nodes, edges)
        let needs_path_details = expand.path_alias.is_some();
        // A named edge variable of a variable-length pattern binds the list of
        // the path's edges. The variable a translator adds for an anonymous
        // edge with a property map gets one too: that map is checked over
        // the path's edges, never on this column.
        let binds_edge_list = is_variable_length && expand.edge_variable.is_some();

        let operator: Box<dyn Operator> = if is_variable_length || needs_path_details {
            // Use VariableLengthExpandOperator for multi-hop paths or named paths
            let min_hops = if is_variable_length {
                expand.min_hops
            } else {
                1
            };
            let max_hops = if is_variable_length {
                // Unbounded runs to the search's budget: a path in TRAIL,
                // SIMPLE or ACYCLIC mode ends on its own; a walk around a
                // cycle does not, and fails at the budget, which a path is
                // charged for each edge (#678).
                expand.max_hops.unwrap_or(u32::MAX)
            } else {
                1
            };
            let reachability = self.reachability_mode(expand);
            let exec_path_mode = match expand.path_mode {
                // A reachability search follows walks where the paths reach
                // what the walks reach (see `reaches_as_walks`); an undirected
                // TRAIL keeps its mode, which checks its source for a closed
                // trail (see `reaches_by_search`)
                _ if reachability.is_some() && super::reachability::reaches_as_walks(expand) => {
                    ExecutionPathMode::Walk
                }
                PathMode::Walk => ExecutionPathMode::Walk,
                PathMode::Trail => ExecutionPathMode::Trail,
                PathMode::Simple => ExecutionPathMode::Simple,
                PathMode::Acyclic => ExecutionPathMode::Acyclic,
            };

            let mut expand_op = VariableLengthExpandOperator::new(
                Arc::clone(&self.store) as Arc<dyn GraphStore>,
                input_op,
                source_column,
                direction,
                self.stored_edge_types(&expand.edge_types)?,
                min_hops,
                max_hops,
                self.read_view(),
            )
            .with_path_mode(exec_path_mode)
            .with_read_counter(self.read_counter())
            .with_memory_budget(self.path_search_budget);

            // If a path alias is set, enable path length and detail output
            if needs_path_details {
                expand_op = expand_op
                    .with_path_length_output()
                    .with_path_detail_output();
            }
            if binds_edge_list {
                expand_op = expand_op.with_edge_list_output();
            }
            // The conditions a filter over this expand handed over: checked
            // on each candidate edge, with the input row's columns
            let conditions: Vec<PendingEdgeCondition> = match &expand.path_alias {
                Some(path) => {
                    let mut pending = self.pending_edge_conditions.borrow_mut();
                    let (taken, kept) = std::mem::take(&mut *pending)
                        .into_iter()
                        .partition(|condition| &condition.path == path);
                    *pending = kept;
                    taken
                }
                None => Vec::new(),
            };
            if !conditions.is_empty() {
                let mut variable_columns: std::collections::HashMap<String, usize> = input_columns
                    .iter()
                    .enumerate()
                    .map(|(i, name)| (name.clone(), i))
                    .collect();
                let mut predicates = Vec::with_capacity(conditions.len());
                for condition in conditions {
                    for hop in condition.hops {
                        variable_columns.insert(hop, input_columns.len());
                    }
                    predicates.push(condition.predicate);
                }
                let predicate =
                    LogicalExpression::conjunction(predicates).expect("at least one condition");
                let predicate = grafeo_core::execution::operators::ExpressionPredicate::new(
                    self.convert_expression(&predicate)?,
                    variable_columns,
                    Arc::clone(&self.store),
                    self.read_view(),
                )
                .with_session_context(self.session_context.clone());
                expand_op = expand_op.with_edge_condition(Box::new(predicate));
            }
            match reachability {
                Some(ReachabilityMode::PerInputRow) => {
                    expand_op = expand_op.with_reachability();
                }
                Some(ReachabilityMode::AcrossInputRows) => {
                    expand_op = expand_op.with_reachability_across_rows();
                }
                None => {}
            }

            Box::new(expand_op)
        } else {
            // Use simple ExpandOperator for single-hop paths without named paths
            let mut expand_op = ExpandOperator::new(
                Arc::clone(&self.store) as Arc<dyn GraphStore>,
                input_op,
                source_column,
                direction,
                self.stored_edge_types(&expand.edge_types)?,
                self.read_view(),
            )
            .with_read_counter(self.read_counter());
            match hop_target {
                Some(HopTarget::Column(target)) => {
                    if let Some(column) = input_columns.iter().position(|name| *name == target) {
                        expand_op = expand_op.with_target_column(column);
                    }
                }
                Some(HopTarget::Key(key, value, labels)) => {
                    expand_op = expand_op.with_target_key(key, value, labels);
                }
                Some(HopTarget::Intersect(_)) | None => {}
            }
            Box::new(expand_op)
        };

        // Build output columns: [input_columns..., edge, target, (path_length)?]
        // Preserve all input columns and add edge + target to match ExpandOperator output
        let mut columns = input_columns;

        // Generate edge column name and register for EdgeResolve in RETURN
        let edge_col_name = self.register_edge_column(&expand.edge_variable);
        if binds_edge_list {
            self.edge_columns.borrow_mut().remove(&edge_col_name);
            self.entity_list_columns.borrow_mut().insert(
                edge_col_name.clone(),
                grafeo_core::execution::operators::EntityValue::Edges,
            );
        }
        if is_variable_length {
            self.group_list_variables
                .borrow_mut()
                .insert(edge_col_name.clone());
        }
        columns.push(edge_col_name);

        columns.push(expand.to_variable.clone());

        // If a path alias is set, add columns for path length, nodes, edges, and the path itself
        if let Some(ref path_alias) = expand.path_alias {
            let length_col = format!("_path_length_{}", path_alias);
            let nodes_col = format!("_path_nodes_{}", path_alias);
            let edges_col = format!("_path_edges_{}", path_alias);
            // Mark as scalar so plan_return uses Column pass-through, not NodeResolve
            self.scalar_columns.borrow_mut().insert(length_col.clone());
            self.scalar_columns.borrow_mut().insert(nodes_col.clone());
            self.scalar_columns.borrow_mut().insert(edges_col.clone());
            // The path alias itself holds a Value::Path of node and edge ids,
            // which RETURN gives as nodes and edges.
            self.set_column_entity(
                path_alias,
                Some(grafeo_core::execution::operators::EntityValue::Nested(
                    grafeo_common::types::LogicalType::Path,
                )),
            );
            columns.push(length_col);
            columns.push(nodes_col);
            columns.push(edges_col);
            columns.push(path_alias.clone());
        }

        Ok((operator, columns))
    }
}
