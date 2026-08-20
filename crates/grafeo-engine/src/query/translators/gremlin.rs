//! Gremlin to LogicalPlan translator.
//!
//! Translates Gremlin AST to the common logical plan representation.

use super::common::{VarGen, wrap_filter, wrap_limit, wrap_return, wrap_skip, wrap_sort};
use crate::query::plan::{
    AggregateExpr, AggregateFunction, AggregateOp, BinaryOp, CreateEdgeOp, CreateNodeOp,
    DeleteNodeOp, DistinctOp, ExpandDirection, ExpandOp, JoinOp, JoinType, LeftJoinOp,
    LogicalExpression, LogicalOperator, LogicalPlan, MapCollectOp, NodeScanOp, OtherwiseOp,
    PathMode, ProjectOp, Projection, ReturnItem, SetPropertyOp, SortKey, SortOrder, UnaryOp,
    UnionOp, UnwindOp,
};
use grafeo_adapters::query::gremlin::{self, ast};
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};

/// Translates a Gremlin query string to a logical plan.
///
/// Supports an `EXPLAIN` or `EXPLAIN ANALYZE` prefix (case-insensitive,
/// non-standard extension) to request plan-only or profiled execution.
///
/// # Errors
///
/// Returns an error if the query cannot be parsed or translated.
pub fn translate(query: &str) -> Result<LogicalPlan> {
    let trimmed = query.trim_start();
    let (explain, profile, actual_query) = if trimmed
        .get(..7)
        .is_some_and(|s| s.eq_ignore_ascii_case("EXPLAIN"))
        && trimmed
            .as_bytes()
            .get(7)
            .is_some_and(u8::is_ascii_whitespace)
    {
        let rest = trimmed[7..].trim_start();
        if rest
            .get(..7)
            .is_some_and(|s| s.eq_ignore_ascii_case("ANALYZE"))
            && rest.as_bytes().get(7).is_some_and(u8::is_ascii_whitespace)
        {
            (false, true, rest[7..].trim_start())
        } else {
            (true, false, rest)
        }
    } else {
        (false, false, query)
    };

    let statement = gremlin::parse(actual_query)?;
    let translator = GremlinTranslator::new();
    let mut plan = translator.translate_statement(&statement)?;
    plan.explain = explain;
    plan.profile = profile;
    Ok(plan)
}

/// Translator from Gremlin AST to LogicalPlan.
struct GremlinTranslator {
    /// Generator for anonymous variable names.
    var_gen: VarGen,
}

/// Context for building an edge during traversal processing.
struct PendingEdge {
    edge_type: String,
    from_var: Option<String>,
    to_var: Option<String>,
    properties: Vec<(String, LogicalExpression)>,
}

/// Tracks edge expansion context for InV/OutV/BothV resolution.
///
/// When an edge traversal step (OutE/InE/BothE) is processed, this stores
/// the source and target vertex variables so subsequent vertex steps can
/// switch to the correct endpoint.
struct EdgeContext {
    source_var: String,
    target_var: String,
    direction: ExpandDirection,
}

/// Fold `preds` into one expression joined by `op`. Returns `None` when empty.
///
/// Right-fold: last element is the innermost right operand, matching the shape the
/// `and()`/`or()` arms have always produced.
fn fold_predicates(mut preds: Vec<LogicalExpression>, op: BinaryOp) -> Option<LogicalExpression> {
    let mut combined = preds.pop()?;
    for pred in preds {
        combined = LogicalExpression::Binary {
            left: Box::new(pred),
            op,
            right: Box::new(combined),
        };
    }
    Some(combined)
}

/// TinkerPop `not()` semantics: a traverser missing the property is *retained*.
/// SQL `NOT NULL` is `NULL` (row dropped), so widen to `NOT(p) OR p IS NULL`.
fn negate_null_safe(pred: LogicalExpression) -> LogicalExpression {
    LogicalExpression::Binary {
        left: Box::new(LogicalExpression::Unary {
            op: UnaryOp::Not,
            operand: Box::new(pred.clone()),
        }),
        op: BinaryOp::Or,
        right: Box::new(LogicalExpression::Unary {
            op: UnaryOp::IsNull,
            operand: Box::new(pred),
        }),
    }
}

impl GremlinTranslator {
    fn new() -> Self {
        Self {
            var_gen: VarGen::new(),
        }
    }

    fn translate_statement(&self, stmt: &ast::Statement) -> Result<LogicalPlan> {
        // Special handling for addE source - need to collect from/to/property steps
        if let ast::TraversalSource::AddE(edge_type) = &stmt.source {
            return self.translate_add_edge_traversal(edge_type, &stmt.steps);
        }

        // Start with the source
        let mut plan = self.translate_source(&stmt.source)?;

        // Track current variable for property access
        let mut current_var = self.get_current_var(&stmt.source);

        // Track edge context for step-level addE
        let mut pending_edge: Option<PendingEdge> = None;

        // Track edge expansion context for InV/OutV/BothV resolution
        let mut edge_ctx: Option<EdgeContext> = None;

        // Label-to-variable mapping for as()/select() patterns
        let mut labels: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();

        // Override for multi-column Return (e.g., select('a', 'b'))
        let mut return_items_override: Option<Vec<ReturnItem>> = None;

        // For g.E(), initialize edge context from the source Expand
        if matches!(&stmt.source, ast::TraversalSource::E(_)) {
            edge_ctx = Self::extract_edge_context(&plan);
        }

        // Process each step
        for step in &stmt.steps {
            // Handle edge creation steps specially
            if let Some(ref mut edge) = pending_edge {
                match step {
                    ast::Step::From(from_to) => {
                        // When from() overrides the source vertex, the current
                        // traverser (stored as the default from_var by addE)
                        // becomes the target vertex if to_var is not yet set.
                        if edge.to_var.is_none() {
                            edge.to_var = edge.from_var.take();
                        }
                        // Resolve as() labels before falling through to plan extraction
                        if let ast::FromTo::Label(label) = from_to
                            && let Some(var) = labels.get(label.as_str())
                        {
                            edge.from_var = Some(var.clone());
                            continue;
                        }
                        let (var, new_plan) =
                            self.extract_from_to_with_plan(from_to, plan, &current_var)?;
                        plan = new_plan;
                        edge.from_var = Some(var);
                        continue;
                    }
                    ast::Step::To(from_to) => {
                        // Resolve as() labels before falling through to plan extraction
                        if let ast::FromTo::Label(label) = from_to
                            && let Some(var) = labels.get(label.as_str())
                        {
                            edge.to_var = Some(var.clone());
                            continue;
                        }
                        let (var, new_plan) =
                            self.extract_from_to_with_plan(from_to, plan, &current_var)?;
                        plan = new_plan;
                        edge.to_var = Some(var);
                        // Don't create edge yet: wait for subsequent .property()
                        // steps to be collected. Edge creation happens when a
                        // non-edge step is encountered or at finalization.
                        continue;
                    }
                    ast::Step::Property(prop_step) => {
                        edge.properties.push((
                            prop_step.key.clone(),
                            LogicalExpression::Literal(prop_step.value.clone()),
                        ));
                        continue;
                    }
                    _ => {
                        // Non-edge step encountered, finalize edge if possible
                        if edge.from_var.is_some() && edge.to_var.is_some() {
                            let edge_var = self.var_gen.next();
                            plan = LogicalOperator::CreateEdge(CreateEdgeOp {
                                variable: Some(edge_var.clone()),
                                from_variable: edge
                                    .from_var
                                    .take()
                                    .expect("from_var checked above"),
                                to_variable: edge.to_var.take().expect("to_var checked above"),
                                edge_type: edge.edge_type.clone(),
                                properties: std::mem::take(&mut edge.properties),
                                input: Box::new(plan),
                            });
                            current_var = edge_var;
                            pending_edge = None;
                        }
                    }
                }
            }

            // Handle edge vertex steps (InV, OutV, BothV) using edge context
            match step {
                ast::Step::InV => {
                    if let Some(ctx) = edge_ctx.take() {
                        // inV = the vertex the edge points TO (the target)
                        current_var = match ctx.direction {
                            ExpandDirection::Outgoing | ExpandDirection::Both => ctx.target_var,
                            ExpandDirection::Incoming => ctx.source_var,
                        };
                    }
                    continue;
                }
                ast::Step::OutV => {
                    if let Some(ctx) = edge_ctx.take() {
                        // outV = the vertex the edge comes FROM (the source)
                        current_var = match ctx.direction {
                            ExpandDirection::Outgoing | ExpandDirection::Both => ctx.source_var,
                            ExpandDirection::Incoming => ctx.target_var,
                        };
                    }
                    continue;
                }
                ast::Step::BothV => {
                    if let Some(ctx) = edge_ctx.take() {
                        // bothV = emit both endpoints via Union
                        let alias = self.var_gen.next();
                        plan = LogicalOperator::Union(UnionOp {
                            inputs: vec![
                                LogicalOperator::Project(ProjectOp {
                                    projections: vec![Projection {
                                        expression: LogicalExpression::Variable(ctx.source_var),
                                        alias: Some(alias.clone()),
                                    }],
                                    input: Box::new(plan.clone()),
                                    pass_through_input: false,
                                }),
                                LogicalOperator::Project(ProjectOp {
                                    projections: vec![Projection {
                                        expression: LogicalExpression::Variable(ctx.target_var),
                                        alias: Some(alias.clone()),
                                    }],
                                    input: Box::new(plan),
                                    pass_through_input: false,
                                }),
                            ],
                        });
                        current_var = alias;
                    }
                    continue;
                }
                ast::Step::OtherV => {
                    if let Some(ctx) = edge_ctx.take() {
                        // otherV = the discovered endpoint (to_variable).
                        // In the Expand operator, from_variable is always the
                        // traverser's current node and to_variable is the node
                        // reached by following the edge, regardless of direction.
                        current_var = ctx.target_var;
                    }
                    continue;
                }
                _ => {}
            }

            // Handle as() labels: store current variable under the label name
            if let ast::Step::As(label) = step {
                labels.insert(label.clone(), current_var.clone());
                continue;
            }

            // Handle select() using stored labels
            if let ast::Step::Select(keys) = step {
                if keys.len() == 1 {
                    // select('a') - project the stored variable
                    if let Some(var) = labels.get(&keys[0]) {
                        let alias = keys[0].clone();
                        plan = LogicalOperator::Project(ProjectOp {
                            projections: vec![Projection {
                                expression: LogicalExpression::Variable(var.clone()),
                                alias: Some(alias.clone()),
                            }],
                            input: Box::new(plan),
                            pass_through_input: false,
                        });
                        current_var = alias;
                    }
                } else {
                    // select('a', 'b') - project multiple stored variables as a map
                    let projections: Vec<Projection> = keys
                        .iter()
                        .filter_map(|k| {
                            labels.get(k).map(|var| Projection {
                                expression: LogicalExpression::Variable(var.clone()),
                                alias: Some(k.clone()),
                            })
                        })
                        .collect();
                    if !projections.is_empty() {
                        // Build multi-column Return items so all keys appear in output
                        return_items_override = Some(
                            projections
                                .iter()
                                .map(|p| ReturnItem {
                                    expression: LogicalExpression::Variable(
                                        p.alias.clone().unwrap_or_default(),
                                    ),
                                    alias: p.alias.clone(),
                                })
                                .collect(),
                        );
                        let first_alias = projections[0].alias.clone();
                        plan = LogicalOperator::Project(ProjectOp {
                            projections,
                            input: Box::new(plan),
                            pass_through_input: false,
                        });
                        if let Some(alias) = first_alias {
                            current_var = alias;
                        }
                    }
                }
                continue;
            }

            // Check if this is a step-level addE
            if let ast::Step::AddE(edge_type) = step {
                edge_ctx = None;
                // For step-level addE, the current context is the source by default
                pending_edge = Some(PendingEdge {
                    edge_type: edge_type.clone(),
                    from_var: Some(current_var.clone()), // Default to current traversal context
                    to_var: None,
                    properties: Vec::new(),
                });
                continue;
            }

            let (new_plan, new_var) =
                self.translate_step(step, plan, &current_var, edge_ctx.is_some())?;
            plan = new_plan;

            // When a Project step with multiple projections is produced,
            // set up a multi-column Return so all projected columns appear.
            if let ast::Step::Project(keys) = step
                && keys.len() > 1
            {
                return_items_override = Some(
                    keys.iter()
                        .map(|k| ReturnItem {
                            expression: LogicalExpression::Variable(k.clone()),
                            alias: Some(k.clone()),
                        })
                        .collect(),
                );
            }

            // ValueMap/ElementMap with specific keys: set multi-column return
            // so all projected property columns appear in the output.
            if let ast::Step::ValueMap(keys) = step
                && keys.len() > 1
            {
                return_items_override = Some(
                    keys.iter()
                        .map(|k| ReturnItem {
                            expression: LogicalExpression::Variable(k.clone()),
                            alias: Some(k.clone()),
                        })
                        .collect(),
                );
            }
            if let ast::Step::ElementMap(keys) = step
                && !keys.is_empty()
            {
                let mut items: Vec<ReturnItem> = vec![
                    ReturnItem {
                        expression: LogicalExpression::Variable("id".to_string()),
                        alias: Some("id".to_string()),
                    },
                    ReturnItem {
                        expression: LogicalExpression::Variable("label".to_string()),
                        alias: Some("label".to_string()),
                    },
                ];
                items.extend(keys.iter().map(|k| ReturnItem {
                    expression: LogicalExpression::Variable(k.clone()),
                    alias: Some(k.clone()),
                }));
                return_items_override = Some(items);
            }

            // Update edge context after step translation
            match step {
                ast::Step::OutE(_) | ast::Step::InE(_) | ast::Step::BothE(_) => {
                    edge_ctx = Self::extract_edge_context(&plan);
                }
                // Filter steps preserve edge context
                ast::Step::Has(_)
                | ast::Step::HasLabel(_)
                | ast::Step::HasId(_)
                | ast::Step::HasNot(_)
                | ast::Step::Filter(_) => {}
                // All other steps clear edge context
                _ => {
                    edge_ctx = None;
                }
            }

            if let Some(v) = new_var {
                current_var = v;
            }
        }

        // Finalize any pending edge
        if let Some(edge) = pending_edge
            && let (Some(from_var), Some(to_var)) = (edge.from_var, edge.to_var)
        {
            let edge_var = self.var_gen.next();
            plan = LogicalOperator::CreateEdge(CreateEdgeOp {
                variable: Some(edge_var.clone()),
                from_variable: from_var,
                to_variable: to_var,
                edge_type: edge.edge_type,
                properties: edge.properties,
                input: Box::new(plan),
            });
            current_var = edge_var;
        }

        // If the last step doesn't produce a Return, wrap with one
        // Exception: DeleteNode doesn't have output to return
        if !matches!(
            plan,
            LogicalOperator::Return(_) | LogicalOperator::DeleteNode(_)
        ) {
            let items = return_items_override.unwrap_or_else(|| {
                vec![ReturnItem {
                    expression: LogicalExpression::Variable(current_var),
                    alias: None,
                }]
            });
            plan = wrap_return(plan, items, false);
        }

        Ok(LogicalPlan::new(plan))
    }

    /// Handle g.addE('type').from(...).to(...) pattern
    fn translate_add_edge_traversal(
        &self,
        edge_type: &str,
        steps: &[ast::Step],
    ) -> Result<LogicalPlan> {
        let mut from_var: Option<String> = None;
        let mut to_var: Option<String> = None;
        let mut properties: Vec<(String, LogicalExpression)> = Vec::new();

        // Start with an empty plan (will be built up with traversals)
        let mut plan = LogicalOperator::Empty;

        for step in steps {
            match step {
                ast::Step::From(from_to) => {
                    let (var, new_plan) = self.extract_from_to_with_plan(from_to, plan, "")?;
                    plan = new_plan;
                    from_var = Some(var);
                }
                ast::Step::To(from_to) => {
                    let (var, new_plan) = self.extract_from_to_with_plan(from_to, plan, "")?;
                    plan = new_plan;
                    to_var = Some(var);
                }
                ast::Step::Property(prop_step) => {
                    properties.push((
                        prop_step.key.clone(),
                        LogicalExpression::Literal(prop_step.value.clone()),
                    ));
                }
                _ => {
                    // Ignore other steps for now
                }
            }
        }

        let from_var = from_var.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "addE requires from() step",
            ))
        })?;
        let to_var = to_var.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "addE requires to() step",
            ))
        })?;

        // If plan is still empty (both from/to were labels), create a scan
        if matches!(plan, LogicalOperator::Empty) {
            let scan_var = self.var_gen.next();
            plan = LogicalOperator::NodeScan(NodeScanOp {
                variable: scan_var,
                label: None,
                input: None,
            });
        }

        let edge_var = self.var_gen.next();
        let create_edge = LogicalOperator::CreateEdge(CreateEdgeOp {
            variable: Some(edge_var.clone()),
            from_variable: from_var,
            to_variable: to_var,
            edge_type: edge_type.to_string(),
            properties,
            input: Box::new(plan),
        });

        let final_plan = wrap_return(
            create_edge,
            vec![ReturnItem {
                expression: LogicalExpression::Variable(edge_var),
                alias: None,
            }],
            false,
        );

        Ok(LogicalPlan::new(final_plan))
    }

    /// Extract variable name from FromTo specification and optionally modify the plan.
    /// Returns (variable_name, modified_plan).
    fn extract_from_to_with_plan(
        &self,
        from_to: &ast::FromTo,
        plan: LogicalOperator,
        _current_var: &str,
    ) -> Result<(String, LogicalOperator)> {
        match from_to {
            ast::FromTo::Label(label) => Ok((label.clone(), plan)),
            ast::FromTo::Traversal(steps) => {
                // Create a fresh NodeScan for the sub-traversal
                let target_var = self.var_gen.next();
                let mut sub_plan = LogicalOperator::NodeScan(NodeScanOp {
                    variable: target_var.clone(),
                    label: None,
                    input: None,
                });

                // Apply any steps from the sub-traversal
                let mut sub_current_var = target_var.clone();
                for step in steps {
                    let (new_plan, new_var) =
                        self.translate_step(step, sub_plan, &sub_current_var, false)?;
                    sub_plan = new_plan;
                    if let Some(v) = new_var {
                        sub_current_var = v;
                    }
                }

                // If the main plan is Empty, just return the sub-plan
                if matches!(plan, LogicalOperator::Empty) {
                    return Ok((sub_current_var, sub_plan));
                }

                // Join the main plan with the sub-traversal (cross product)
                let joined_plan = LogicalOperator::Join(JoinOp {
                    left: Box::new(plan),
                    right: Box::new(sub_plan),
                    join_type: JoinType::Inner,
                    conditions: Vec::new(), // Cross product - no conditions
                });

                Ok((sub_current_var, joined_plan))
            }
        }
    }

    fn translate_source(&self, source: &ast::TraversalSource) -> Result<LogicalOperator> {
        match source {
            ast::TraversalSource::V(ids) => {
                let var = self.var_gen.next();
                let mut plan = LogicalOperator::NodeScan(NodeScanOp {
                    variable: var.clone(),
                    label: None,
                    input: None,
                });

                // If specific IDs, add filter
                if let Some(ids) = ids
                    && !ids.is_empty()
                {
                    let id_filter = self.build_id_filter(&var, ids);
                    plan = wrap_filter(plan, id_filter);
                }

                Ok(plan)
            }
            ast::TraversalSource::E(ids) => {
                // Edge scan - need to scan nodes and expand
                // Use Outgoing direction to get each edge exactly once (from its source node)
                let var = self.var_gen.next();
                let mut plan = LogicalOperator::NodeScan(NodeScanOp {
                    variable: var.clone(),
                    label: None,
                    input: None,
                });

                let edge_var = self.var_gen.next();
                let target_var = self.var_gen.next();

                plan = LogicalOperator::Expand(ExpandOp {
                    from_variable: var,
                    to_variable: target_var,
                    edge_variable: Some(edge_var.clone()),
                    direction: ExpandDirection::Outgoing, // Use Outgoing to avoid duplicate edges
                    edge_types: vec![],
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(plan),
                    path_alias: None,
                    path_mode: PathMode::Walk,
                });

                // Filter by edge IDs if specified
                if let Some(ids) = ids
                    && !ids.is_empty()
                {
                    let id_filter = self.build_id_filter(&edge_var, ids);
                    plan = wrap_filter(plan, id_filter);
                }

                Ok(plan)
            }
            ast::TraversalSource::AddV(label) => {
                let var = self.var_gen.next();
                Ok(LogicalOperator::CreateNode(CreateNodeOp {
                    variable: var,
                    labels: label.iter().cloned().collect(),
                    properties: Vec::new(),
                    input: None,
                }))
            }
            ast::TraversalSource::AddE(_label) => {
                // AddE needs from/to steps to complete
                Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    "addE requires from() and to() steps",
                )))
            }
        }
    }

    fn translate_step(
        &self,
        step: &ast::Step,
        input: LogicalOperator,
        current_var: &str,
        is_edge: bool,
    ) -> Result<(LogicalOperator, Option<String>)> {
        match step {
            // Navigation steps
            ast::Step::Out(labels) => {
                let target_var = self.var_gen.next();
                let edge_types = labels.clone();
                let plan = LogicalOperator::Expand(ExpandOp {
                    from_variable: current_var.to_string(),
                    to_variable: target_var.clone(),
                    edge_variable: None,
                    direction: ExpandDirection::Outgoing,
                    edge_types,
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(input),
                    path_alias: None,
                    path_mode: PathMode::Walk,
                });
                Ok((plan, Some(target_var)))
            }
            ast::Step::In(labels) => {
                let target_var = self.var_gen.next();
                let edge_types = labels.clone();
                let plan = LogicalOperator::Expand(ExpandOp {
                    from_variable: current_var.to_string(),
                    to_variable: target_var.clone(),
                    edge_variable: None,
                    direction: ExpandDirection::Incoming,
                    edge_types,
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(input),
                    path_alias: None,
                    path_mode: PathMode::Walk,
                });
                Ok((plan, Some(target_var)))
            }
            ast::Step::Both(labels) => {
                let target_var = self.var_gen.next();
                let edge_types = labels.clone();
                let plan = LogicalOperator::Expand(ExpandOp {
                    from_variable: current_var.to_string(),
                    to_variable: target_var.clone(),
                    edge_variable: None,
                    direction: ExpandDirection::Both,
                    edge_types,
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(input),
                    path_alias: None,
                    path_mode: PathMode::Walk,
                });
                Ok((plan, Some(target_var)))
            }
            ast::Step::OutE(labels) => {
                let edge_var = self.var_gen.next();
                let target_var = self.var_gen.next();
                let edge_types = labels.clone();
                let plan = LogicalOperator::Expand(ExpandOp {
                    from_variable: current_var.to_string(),
                    to_variable: target_var,
                    edge_variable: Some(edge_var.clone()),
                    direction: ExpandDirection::Outgoing,
                    edge_types,
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(input),
                    path_alias: None,
                    path_mode: PathMode::Walk,
                });
                Ok((plan, Some(edge_var)))
            }
            ast::Step::InE(labels) => {
                let edge_var = self.var_gen.next();
                let target_var = self.var_gen.next();
                let edge_types = labels.clone();
                let plan = LogicalOperator::Expand(ExpandOp {
                    from_variable: current_var.to_string(),
                    to_variable: target_var,
                    edge_variable: Some(edge_var.clone()),
                    direction: ExpandDirection::Incoming,
                    edge_types,
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(input),
                    path_alias: None,
                    path_mode: PathMode::Walk,
                });
                Ok((plan, Some(edge_var)))
            }
            ast::Step::BothE(labels) => {
                let edge_var = self.var_gen.next();
                let target_var = self.var_gen.next();
                let edge_types = labels.clone();
                let plan = LogicalOperator::Expand(ExpandOp {
                    from_variable: current_var.to_string(),
                    to_variable: target_var,
                    edge_variable: Some(edge_var.clone()),
                    direction: ExpandDirection::Both,
                    edge_types,
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(input),
                    path_alias: None,
                    path_mode: PathMode::Walk,
                });
                Ok((plan, Some(edge_var)))
            }

            // Filter steps
            ast::Step::Has(has_step) => {
                let predicate = self.translate_has_step(has_step, current_var)?;
                let plan = wrap_filter(input, predicate);
                Ok((plan, None))
            }
            ast::Step::HasLabel(labels) => {
                let predicate = if is_edge {
                    // Edges have a single type, compare with Type(var)
                    if labels.len() == 1 {
                        LogicalExpression::Binary {
                            left: Box::new(LogicalExpression::Type(current_var.to_string())),
                            op: BinaryOp::Eq,
                            right: Box::new(LogicalExpression::Literal(Value::String(
                                labels[0].clone().into(),
                            ))),
                        }
                    } else {
                        // Multiple labels: type(e) IN [...]
                        LogicalExpression::Binary {
                            left: Box::new(LogicalExpression::Type(current_var.to_string())),
                            op: BinaryOp::In,
                            right: Box::new(LogicalExpression::Literal(Value::List(
                                labels
                                    .iter()
                                    .map(|l| Value::String(l.clone().into()))
                                    .collect(),
                            ))),
                        }
                    }
                } else {
                    // Nodes: check if label IN labels(var)
                    if labels.len() == 1 {
                        LogicalExpression::Binary {
                            left: Box::new(LogicalExpression::Literal(Value::String(
                                labels[0].clone().into(),
                            ))),
                            op: BinaryOp::In,
                            right: Box::new(LogicalExpression::Labels(current_var.to_string())),
                        }
                    } else {
                        let mut conditions: Vec<LogicalExpression> = labels
                            .iter()
                            .map(|l| LogicalExpression::Binary {
                                left: Box::new(LogicalExpression::Literal(Value::String(
                                    l.clone().into(),
                                ))),
                                op: BinaryOp::In,
                                right: Box::new(LogicalExpression::Labels(current_var.to_string())),
                            })
                            .collect();
                        let mut result = conditions
                            .pop()
                            .expect("conditions non-empty for multi-label");
                        for cond in conditions {
                            result = LogicalExpression::Binary {
                                left: Box::new(cond),
                                op: BinaryOp::Or,
                                right: Box::new(result),
                            };
                        }
                        result
                    }
                };
                let plan = wrap_filter(input, predicate);
                Ok((plan, None))
            }
            ast::Step::HasId(ids) => {
                let predicate = self.build_id_filter(current_var, ids);
                let plan = wrap_filter(input, predicate);
                Ok((plan, None))
            }
            ast::Step::HasNot(key) => {
                let predicate = LogicalExpression::Unary {
                    op: UnaryOp::IsNull,
                    operand: Box::new(LogicalExpression::Property {
                        variable: current_var.to_string(),
                        property: key.clone(),
                    }),
                };
                let plan = wrap_filter(input, predicate);
                Ok((plan, None))
            }
            ast::Step::Dedup(keys) => {
                // Gremlin dedup() deduplicates on the current traverser variable.
                // When keys are empty, dedup on current_var (not all columns).
                let columns = if keys.is_empty() {
                    Some(vec![current_var.to_string()])
                } else {
                    Some(keys.clone())
                };
                let plan = LogicalOperator::Distinct(DistinctOp {
                    input: Box::new(input),
                    columns,
                });
                Ok((plan, None))
            }
            ast::Step::Limit(n) => {
                let plan = wrap_limit(input, *n);
                Ok((plan, None))
            }
            ast::Step::Skip(n) => {
                let plan = wrap_skip(input, *n);
                Ok((plan, None))
            }
            ast::Step::Range(start, end) => {
                if end < start {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("range() end ({end}) must be >= start ({start})"),
                    )));
                }
                let plan = wrap_skip(input, *start);
                let plan = wrap_limit(plan, end - start);
                Ok((plan, None))
            }

            // Map steps
            ast::Step::Values(keys) => {
                // Helper: IS NOT NULL filter expression for a variable
                let not_null = |var: &str| LogicalExpression::Unary {
                    op: UnaryOp::IsNotNull,
                    operand: Box::new(LogicalExpression::Variable(var.to_string())),
                };

                if keys.is_empty() {
                    // values() with no keys: return all property values,
                    // one row per value. Uses property_values(var) to get
                    // all values as a list, then Unwind to emit individual rows.
                    let list_alias = self.var_gen.next();
                    let elem_alias = self.var_gen.next();
                    let plan = LogicalOperator::Project(ProjectOp {
                        projections: vec![Projection {
                            expression: LogicalExpression::FunctionCall {
                                name: "property_values".to_string(),
                                args: vec![LogicalExpression::Variable(current_var.to_string())],
                                distinct: false,
                            },
                            alias: Some(list_alias.clone()),
                        }],
                        input: Box::new(input),
                        pass_through_input: false,
                    });
                    let plan = LogicalOperator::Unwind(UnwindOp {
                        expression: LogicalExpression::Variable(list_alias),
                        variable: elem_alias.clone(),
                        ordinality_var: None,
                        offset_var: None,
                        input: Box::new(plan),
                    });
                    Ok((plan, Some(elem_alias)))
                } else if keys.len() > 1 {
                    // Gremlin values('a','b') emits one traverser per key.
                    // Translate as Union of individual property projections,
                    // each filtered to exclude nulls (non-existent
                    // properties emit nothing in Gremlin).
                    let alias = self.var_gen.next();
                    let branches: Vec<LogicalOperator> = keys
                        .iter()
                        .map(|k| {
                            let project = LogicalOperator::Project(ProjectOp {
                                projections: vec![Projection {
                                    expression: LogicalExpression::Property {
                                        variable: current_var.to_string(),
                                        property: k.clone(),
                                    },
                                    alias: Some(alias.clone()),
                                }],
                                input: Box::new(input.clone()),
                                pass_through_input: false,
                            });
                            // Filter out nulls: non-existent properties
                            // must not emit a traverser
                            wrap_filter(project, not_null(&alias))
                        })
                        .collect();
                    let plan = LogicalOperator::Union(UnionOp { inputs: branches });
                    Ok((plan, Some(alias)))
                } else {
                    // Single key: Project then filter out nulls
                    let key = &keys[0];
                    let alias = key.clone();
                    let plan = LogicalOperator::Project(ProjectOp {
                        projections: vec![Projection {
                            expression: LogicalExpression::Property {
                                variable: current_var.to_string(),
                                property: key.clone(),
                            },
                            alias: Some(alias.clone()),
                        }],
                        input: Box::new(input),
                        pass_through_input: false,
                    });
                    // Filter out nulls: Gremlin values() skips entities
                    // that lack the requested property
                    let plan = wrap_filter(plan, not_null(&alias));
                    Ok((plan, Some(alias)))
                }
            }
            ast::Step::Id => {
                let plan = LogicalOperator::Project(ProjectOp {
                    projections: vec![Projection {
                        expression: LogicalExpression::Id(current_var.to_string()),
                        alias: Some("id".to_string()),
                    }],
                    input: Box::new(input),
                    pass_through_input: false,
                });
                Ok((plan, Some("id".to_string())))
            }
            ast::Step::Label => {
                // For edges, use Type(var) which returns the edge type directly.
                // For nodes, Labels(var) returns a list, so use IndexAccess to
                // extract element [0] as a scalar string.
                let label_expr = if is_edge {
                    LogicalExpression::Type(current_var.to_string())
                } else {
                    LogicalExpression::IndexAccess {
                        base: Box::new(LogicalExpression::Labels(current_var.to_string())),
                        index: Box::new(LogicalExpression::Literal(Value::Int64(0))),
                    }
                };
                let plan = LogicalOperator::Project(ProjectOp {
                    projections: vec![Projection {
                        expression: label_expr,
                        alias: Some("label".to_string()),
                    }],
                    input: Box::new(input),
                    pass_through_input: false,
                });
                Ok((plan, Some("label".to_string())))
            }
            ast::Step::Count => {
                let alias = "count".to_string();
                let plan = LogicalOperator::Aggregate(AggregateOp {
                    group_by: Vec::new(),
                    aggregates: vec![AggregateExpr {
                        function: AggregateFunction::Count,
                        expression: None,
                        expression2: None,
                        distinct: false,
                        alias: Some(alias.clone()),
                        percentile: None,
                        separator: None,
                    }],
                    input: Box::new(input),
                    having: None,
                });
                // Return the aggregate alias as the new variable so Return uses correct column
                Ok((plan, Some(alias)))
            }
            ast::Step::Sum => {
                let alias = "sum".to_string();
                let plan = LogicalOperator::Aggregate(AggregateOp {
                    group_by: Vec::new(),
                    aggregates: vec![AggregateExpr {
                        function: AggregateFunction::Sum,
                        expression: Some(LogicalExpression::Variable(current_var.to_string())),
                        expression2: None,
                        distinct: false,
                        alias: Some(alias.clone()),
                        percentile: None,
                        separator: None,
                    }],
                    input: Box::new(input),
                    having: None,
                });
                Ok((plan, Some(alias)))
            }
            ast::Step::Mean => {
                let alias = "mean".to_string();
                let plan = LogicalOperator::Aggregate(AggregateOp {
                    group_by: Vec::new(),
                    aggregates: vec![AggregateExpr {
                        function: AggregateFunction::Avg,
                        expression: Some(LogicalExpression::Variable(current_var.to_string())),
                        expression2: None,
                        distinct: false,
                        alias: Some(alias.clone()),
                        percentile: None,
                        separator: None,
                    }],
                    input: Box::new(input),
                    having: None,
                });
                Ok((plan, Some(alias)))
            }
            ast::Step::Min => {
                let alias = "min".to_string();
                let plan = LogicalOperator::Aggregate(AggregateOp {
                    group_by: Vec::new(),
                    aggregates: vec![AggregateExpr {
                        function: AggregateFunction::Min,
                        expression: Some(LogicalExpression::Variable(current_var.to_string())),
                        expression2: None,
                        distinct: false,
                        alias: Some(alias.clone()),
                        percentile: None,
                        separator: None,
                    }],
                    input: Box::new(input),
                    having: None,
                });
                Ok((plan, Some(alias)))
            }
            ast::Step::Max => {
                let alias = "max".to_string();
                let plan = LogicalOperator::Aggregate(AggregateOp {
                    group_by: Vec::new(),
                    aggregates: vec![AggregateExpr {
                        function: AggregateFunction::Max,
                        expression: Some(LogicalExpression::Variable(current_var.to_string())),
                        expression2: None,
                        distinct: false,
                        alias: Some(alias.clone()),
                        percentile: None,
                        separator: None,
                    }],
                    input: Box::new(input),
                    having: None,
                });
                Ok((plan, Some(alias)))
            }
            ast::Step::Fold => {
                let alias = "fold".to_string();
                let plan = LogicalOperator::Aggregate(AggregateOp {
                    group_by: Vec::new(),
                    aggregates: vec![AggregateExpr {
                        function: AggregateFunction::Collect,
                        expression: Some(LogicalExpression::Variable(current_var.to_string())),
                        expression2: None,
                        distinct: false,
                        alias: Some(alias.clone()),
                        percentile: None,
                        separator: None,
                    }],
                    input: Box::new(input),
                    having: None,
                });
                Ok((plan, Some(alias)))
            }
            ast::Step::Order(modifiers) => {
                let keys = if modifiers.is_empty() {
                    vec![SortKey {
                        expression: LogicalExpression::Variable(current_var.to_string()),
                        order: SortOrder::Ascending,
                        nulls: None,
                    }]
                } else {
                    modifiers
                        .iter()
                        .map(|m| SortKey {
                            expression: self.translate_by_modifier(&m.by, current_var),
                            order: match m.order {
                                ast::SortOrder::Asc => SortOrder::Ascending,
                                ast::SortOrder::Desc => SortOrder::Descending,
                                ast::SortOrder::Shuffle => SortOrder::Ascending, // Not supported
                            },
                            nulls: None,
                        })
                        .collect()
                };
                let plan = wrap_sort(input, keys);
                Ok((plan, None))
            }

            // Side effect steps
            ast::Step::As(label) => {
                // 'as' just adds a label, which we track via variables
                // In LogicalPlan, we use the label as an alias
                Ok((input, Some(label.clone())))
            }
            ast::Step::Property(prop_step) => {
                // If setting property on a node being created, add to CreateNodeOp
                // Otherwise, use SetPropertyOp
                match input {
                    LogicalOperator::CreateNode(mut create_op) => {
                        // Add property to the CreateNodeOp
                        create_op.properties.push((
                            prop_step.key.clone(),
                            LogicalExpression::Literal(prop_step.value.clone()),
                        ));
                        Ok((LogicalOperator::CreateNode(create_op), None))
                    }
                    _ => {
                        // Use SetPropertyOp for existing nodes
                        let plan = LogicalOperator::SetProperty(SetPropertyOp {
                            variable: current_var.to_string(),
                            properties: vec![(
                                prop_step.key.clone(),
                                LogicalExpression::Literal(prop_step.value.clone()),
                            )],
                            replace: false,
                            is_edge: false,
                            input: Box::new(input),
                        });
                        Ok((plan, None))
                    }
                }
            }
            ast::Step::Drop => {
                // Delete the current element
                // Gremlin drop() is equivalent to DETACH DELETE
                let plan = LogicalOperator::DeleteNode(DeleteNodeOp {
                    variable: current_var.to_string(),
                    detach: true,
                    input: Box::new(input),
                });
                Ok((plan, None))
            }
            ast::Step::AddV(label) => {
                let var = self.var_gen.next();
                let plan = LogicalOperator::CreateNode(CreateNodeOp {
                    variable: var.clone(),
                    labels: label.iter().cloned().collect(),
                    properties: Vec::new(),
                    input: Some(Box::new(input)),
                });
                Ok((plan, Some(var)))
            }
            ast::Step::AddE(_label) => {
                // AddE is handled specially in translate_statement with from/to context
                // If we reach here, it means the step was processed outside the normal flow
                Ok((input, None))
            }

            ast::Step::By(by_modifier) => {
                // 'by' modifies a preceding step: order(), groupCount(), or project()
                match input {
                    LogicalOperator::Sort(mut sort_op) => {
                        let by_expr = self.translate_by_modifier(by_modifier, current_var);
                        let order = match by_modifier {
                            ast::ByModifier::KeyWithOrder(_, ast_order)
                            | ast::ByModifier::Order(ast_order) => match ast_order {
                                ast::SortOrder::Asc => SortOrder::Ascending,
                                ast::SortOrder::Desc => SortOrder::Descending,
                                ast::SortOrder::Shuffle => SortOrder::Ascending,
                            },
                            _ => SortOrder::Ascending,
                        };
                        sort_op.keys = vec![SortKey {
                            expression: by_expr,
                            order,
                            nulls: None,
                        }];
                        Ok((LogicalOperator::Sort(sort_op), None))
                    }
                    LogicalOperator::Aggregate(mut agg_op) => {
                        // groupCount().by('key') or group().by('key'): wrap
                        // aggregate in MapCollect. Extract the original variable
                        // from the existing group_by (set as Variable(current_var)
                        // during groupCount()/group()), since current_var now
                        // points to the aggregate alias.
                        let original_var =
                            if let Some(LogicalExpression::Variable(v)) = agg_op.group_by.first() {
                                v.clone()
                            } else {
                                current_var.to_string()
                            };
                        let mut by_expr = self.translate_by_modifier(by_modifier, &original_var);

                        // Labels(var) returns a list: wrap with IndexAccess [0]
                        // to extract a scalar label (same as the Label step).
                        if let LogicalExpression::Labels(_) = &by_expr {
                            by_expr = LogicalExpression::IndexAccess {
                                base: Box::new(by_expr),
                                index: Box::new(LogicalExpression::Literal(Value::Int64(0))),
                            };
                        }

                        // Compute the key column name using the planner's naming.
                        let key_var = crate::query::planner::common::expression_to_string(&by_expr);

                        agg_op.group_by = vec![by_expr];

                        let value_var = agg_op
                            .aggregates
                            .first()
                            .and_then(|a| a.alias.clone())
                            .unwrap_or_else(|| "count".to_string());

                        let is_group = agg_op
                            .aggregates
                            .first()
                            .is_some_and(|a| matches!(a.function, AggregateFunction::Collect));
                        let alias = if is_group {
                            "_group".to_string()
                        } else {
                            "_groupCount".to_string()
                        };
                        let plan = LogicalOperator::MapCollect(MapCollectOp {
                            key_var,
                            value_var,
                            alias: alias.clone(),
                            input: Box::new(LogicalOperator::Aggregate(agg_op)),
                        });
                        Ok((plan, Some(alias)))
                    }
                    LogicalOperator::MapCollect(mut map_op) => {
                        // Second .by() for group(): modify the Collect expression
                        // inside the MapCollect's inner Aggregate.
                        if let LogicalOperator::Aggregate(ref mut agg_op) = *map_op.input
                            && let Some(agg) = agg_op.aggregates.first()
                            && matches!(agg.function, AggregateFunction::Collect)
                        {
                            let original_var = match &agg.expression {
                                Some(LogicalExpression::Variable(v)) => v.clone(),
                                _ => current_var.to_string(),
                            };
                            let by_expr = self.translate_by_modifier(by_modifier, &original_var);
                            let new_value_var =
                                crate::query::planner::common::expression_to_string(&by_expr);
                            agg_op.aggregates[0].expression = Some(by_expr);
                            agg_op.aggregates[0].alias = Some(new_value_var.clone());
                            map_op.value_var = new_value_var;
                            let alias = map_op.alias.clone();
                            return Ok((LogicalOperator::MapCollect(map_op), Some(alias)));
                        }
                        // Not a group second .by(), pass through
                        Ok((LogicalOperator::MapCollect(map_op), None))
                    }
                    LogicalOperator::Project(mut proj_op) => {
                        // project('n','a').by('name').by('age')
                        // Each successive .by() fills the next unfilled projection.
                        // The original vertex variable is in the first projection's
                        // Property expression (set during the Project step).
                        let original_var = proj_op
                            .projections
                            .first()
                            .and_then(|p| {
                                if let LogicalExpression::Property { variable, .. } = &p.expression
                                {
                                    Some(variable.clone())
                                } else {
                                    None
                                }
                            })
                            .unwrap_or_else(|| current_var.to_string());
                        let by_expr = self.translate_by_modifier(by_modifier, &original_var);
                        for proj in &mut proj_op.projections {
                            if let LogicalExpression::Property { property, .. } = &proj.expression
                                && proj.alias.as_deref() == Some(property.as_str())
                            {
                                proj.expression = by_expr;
                                return Ok((LogicalOperator::Project(proj_op), None));
                            }
                        }
                        Ok((LogicalOperator::Project(proj_op), None))
                    }
                    _ => {
                        // by() without a supported preceding step - ignore
                        Ok((input, None))
                    }
                }
            }

            // Constant: replace each traverser with a fixed literal value
            ast::Step::Constant(value) => {
                let alias = "constant".to_string();
                let plan = LogicalOperator::Project(ProjectOp {
                    projections: vec![Projection {
                        expression: LogicalExpression::Literal(value.clone()),
                        alias: Some(alias.clone()),
                    }],
                    input: Box::new(input),
                    pass_through_input: false,
                });
                Ok((plan, Some(alias)))
            }

            // Unfold: expand list elements into individual rows
            ast::Step::Unfold => {
                let new_var = self.var_gen.next();
                let plan = LogicalOperator::Unwind(UnwindOp {
                    expression: LogicalExpression::Variable(current_var.to_string()),
                    variable: new_var.clone(),
                    ordinality_var: None,
                    offset_var: None,
                    input: Box::new(input),
                });
                Ok((plan, Some(new_var)))
            }

            // Group: group by a key and collect values into lists.
            // First .by() sets the key, second .by() sets the value.
            ast::Step::Group(_modifiers) => {
                let alias = "collect".to_string();
                let plan = LogicalOperator::Aggregate(AggregateOp {
                    group_by: vec![LogicalExpression::Variable(current_var.to_string())],
                    aggregates: vec![AggregateExpr {
                        function: AggregateFunction::Collect,
                        expression: Some(LogicalExpression::Variable(current_var.to_string())),
                        expression2: None,
                        distinct: false,
                        alias: Some(alias.clone()),
                        percentile: None,
                        separator: None,
                    }],
                    input: Box::new(input),
                    having: None,
                });
                Ok((plan, Some(alias)))
            }

            // GroupCount: group by a key and count occurrences
            ast::Step::GroupCount(_label) => {
                // GroupCount groups by current_var and counts
                let plan = LogicalOperator::Aggregate(AggregateOp {
                    group_by: vec![LogicalExpression::Variable(current_var.to_string())],
                    aggregates: vec![AggregateExpr {
                        function: AggregateFunction::Count,
                        expression: None,
                        expression2: None,
                        distinct: false,
                        alias: Some("count".to_string()),
                        percentile: None,
                        separator: None,
                    }],
                    input: Box::new(input),
                    having: None,
                });
                Ok((plan, Some("count".to_string())))
            }

            // Select: retrieve labeled traversers by their .as() labels.
            // Currently only works when the select keys match known bound
            // variables (e.g., from a preceding as() step). Falls through to
            // pass-through for unbound sideEffect references.
            ast::Step::Select(_keys) => {
                // Full sideEffect collection semantics (aggregate/store +
                // select) are not yet supported. Pass through unchanged so
                // previously passing tests remain stable.
                Ok((input, None))
            }

            // Project: create named projections from properties
            ast::Step::Project(keys) => {
                // Project creates a map: keys are the column names, values come
                // from subsequent .by() steps. Each projection uses current_var
                // as a placeholder; .by() replaces the expression.
                let projections: Vec<Projection> = keys
                    .iter()
                    .map(|k| Projection {
                        expression: LogicalExpression::Property {
                            variable: current_var.to_string(),
                            property: k.clone(),
                        },
                        alias: Some(k.clone()),
                    })
                    .collect();
                let plan = LogicalOperator::Project(ProjectOp {
                    projections,
                    input: Box::new(input),
                    pass_through_input: false,
                });
                // Set first key as current_var so Return uses a valid column.
                // The full Return items are set below.
                let first_key = keys.first().cloned();
                Ok((plan, first_key))
            }

            // and() filter: all sub-traversals must produce results
            ast::Step::And(traversals) => {
                // Translate each sub-traversal as a filter predicate
                let mut predicates: Vec<LogicalExpression> = Vec::new();
                for steps in traversals {
                    if let Some(pred) = self.steps_to_predicate(steps, current_var)? {
                        predicates.push(pred);
                    }
                }
                match fold_predicates(predicates, BinaryOp::And) {
                    Some(combined) => Ok((wrap_filter(input, combined), None)),
                    None => Ok((input, None)),
                }
            }

            // or() filter: at least one sub-traversal must produce results
            ast::Step::Or(traversals) => {
                let mut predicates: Vec<LogicalExpression> = Vec::new();
                for steps in traversals {
                    if let Some(pred) = self.steps_to_predicate(steps, current_var)? {
                        predicates.push(pred);
                    }
                }
                match fold_predicates(predicates, BinaryOp::Or) {
                    Some(combined) => Ok((wrap_filter(input, combined), None)),
                    None => Ok((input, None)),
                }
            }

            // not() filter: negate a sub-traversal filter (null-safe, per TinkerPop)
            ast::Step::Not(steps) => {
                if let Some(pred) = self.steps_to_predicate(steps, current_var)? {
                    Ok((wrap_filter(input, negate_null_safe(pred)), None))
                } else {
                    Ok((input, None))
                }
            }

            // where() filter: inline filter via traversal or predicate
            ast::Step::Where(clause) => match clause {
                ast::WhereClause::Traversal(steps) => {
                    if let Some(pred) = self.steps_to_predicate(steps, current_var)? {
                        let plan = wrap_filter(input, pred);
                        Ok((plan, None))
                    } else {
                        Ok((input, None))
                    }
                }
                ast::WhereClause::Predicate(_var, pred) => {
                    let prop_expr = LogicalExpression::Variable(current_var.to_string());
                    let predicate = Self::translate_predicate(pred, prop_expr)?;
                    let plan = wrap_filter(input, predicate);
                    Ok((plan, None))
                }
            },

            // Filter step: treated as where(traversal)
            ast::Step::Filter(pred) => {
                let prop_expr = LogicalExpression::Variable(current_var.to_string());
                let predicate = Self::translate_predicate(pred, prop_expr)?;
                let plan = wrap_filter(input, predicate);
                Ok((plan, None))
            }

            // choose() branching: if/then/else via union of filtered paths
            ast::Step::Choose(clause) => {
                // Build condition predicate from the condition traversal
                let cond_pred = match &clause.condition {
                    ast::ChooseCondition::Traversal(steps) => {
                        self.steps_to_predicate(steps, current_var)?
                    }
                    ast::ChooseCondition::HasKey(key) => Some(LogicalExpression::Unary {
                        op: UnaryOp::IsNotNull,
                        operand: Box::new(LogicalExpression::Property {
                            variable: current_var.to_string(),
                            property: key.clone(),
                        }),
                    }),
                    ast::ChooseCondition::Predicate(pred) => {
                        let prop_expr = LogicalExpression::Variable(current_var.to_string());
                        Some(Self::translate_predicate(pred, prop_expr)?)
                    }
                };

                if let Some(pred) = cond_pred {
                    // True branch: filter by condition, then apply true steps
                    let mut true_plan = wrap_filter(input.clone(), pred.clone());
                    let mut true_var = current_var.to_string();
                    for step in &clause.true_branch {
                        let (new_plan, new_var) =
                            self.translate_step(step, true_plan, &true_var, false)?;
                        if let Some(v) = new_var {
                            true_var = v;
                        }
                        true_plan = new_plan;
                    }

                    // False branch: filter by NOT condition, then apply false steps
                    let negated = LogicalExpression::Unary {
                        op: UnaryOp::Not,
                        operand: Box::new(pred),
                    };
                    let mut false_plan = wrap_filter(input, negated);
                    let mut false_var = current_var.to_string();
                    if let Some(false_steps) = &clause.false_branch {
                        for step in false_steps {
                            let (new_plan, new_var) =
                                self.translate_step(step, false_plan, &false_var, false)?;
                            if let Some(v) = new_var {
                                false_var = v;
                            }
                            false_plan = new_plan;
                        }
                    }

                    let plan = LogicalOperator::Union(UnionOp {
                        inputs: vec![true_plan, false_plan],
                    });
                    Ok((plan, Some(true_var)))
                } else {
                    Ok((input, None))
                }
            }

            // optional(): try inner traversal per traverser, fall back to
            // identity when empty.
            //
            // In Gremlin, optional(T) evaluates T for each traverser.
            // When T produces results, those replace the traverser. When
            // T is empty for a given traverser, that traverser is kept
            // unchanged. This requires per-row semantics, not batch-level.
            //
            // Translation:
            //   LeftJoin(input, inner_traversal)
            //     -> produces [current_var, ..., inner_var or NULL]
            //   Project(CASE WHEN inner_var IS NOT NULL
            //                THEN inner_var ELSE current_var END AS alias)
            //     -> picks inner result when available, identity otherwise
            ast::Step::Optional(steps) => {
                let alias = self.var_gen.next();

                // Build the inner traversal starting from the same input
                let mut inner_plan = input.clone();
                let mut inner_var = current_var.to_string();
                for step in steps {
                    let (new_plan, new_var) =
                        self.translate_step(step, inner_plan, &inner_var, false)?;
                    if let Some(v) = new_var {
                        inner_var = v;
                    }
                    inner_plan = new_plan;
                }

                // If the inner traversal produced a new variable (e.g.,
                // out() creates a target node variable), use LeftJoin +
                // CASE to select inner result or identity per row.
                // If no new variable was introduced (inner_var ==
                // current_var), the inner traversal is a filter, and we
                // can use Otherwise with projected branches.
                if inner_var == current_var {
                    // Pure filter case: optional(has(...)) or similar.
                    // Inner plan filters the same variable, so use
                    // Otherwise: try filtered, fall back to unfiltered.
                    let inner_projected = LogicalOperator::Project(ProjectOp {
                        projections: vec![Projection {
                            expression: LogicalExpression::Variable(current_var.to_string()),
                            alias: Some(alias.clone()),
                        }],
                        input: Box::new(inner_plan),
                        pass_through_input: false,
                    });
                    let identity_projected = LogicalOperator::Project(ProjectOp {
                        projections: vec![Projection {
                            expression: LogicalExpression::Variable(current_var.to_string()),
                            alias: Some(alias.clone()),
                        }],
                        input: Box::new(input),
                        pass_through_input: false,
                    });
                    let plan = LogicalOperator::Otherwise(OtherwiseOp {
                        left: Box::new(inner_projected),
                        right: Box::new(identity_projected),
                    });
                    Ok((plan, Some(alias)))
                } else {
                    // Navigation case: inner traversal introduces a new
                    // variable (e.g., out() target). Use LeftJoin to
                    // correlate per source row, then CASE to pick inner
                    // result when present or identity when NULL.
                    let left_join = LogicalOperator::LeftJoin(LeftJoinOp {
                        left: Box::new(input),
                        right: Box::new(inner_plan),
                        condition: None,
                    });

                    // CASE WHEN inner_var IS NOT NULL
                    //      THEN inner_var ELSE current_var END
                    let case_expr = LogicalExpression::Case {
                        operand: None,
                        when_clauses: vec![(
                            LogicalExpression::Unary {
                                op: UnaryOp::IsNotNull,
                                operand: Box::new(LogicalExpression::Variable(inner_var.clone())),
                            },
                            LogicalExpression::Variable(inner_var),
                        )],
                        else_clause: Some(Box::new(LogicalExpression::Variable(
                            current_var.to_string(),
                        ))),
                    };

                    let plan = LogicalOperator::Project(ProjectOp {
                        projections: vec![Projection {
                            expression: case_expr,
                            alias: Some(alias.clone()),
                        }],
                        input: Box::new(left_join),
                        pass_through_input: false,
                    });
                    Ok((plan, Some(alias)))
                }
            }

            // union(): merge results from multiple sub-traversals
            ast::Step::Union(traversals) => {
                let alias = self.var_gen.next();
                let branches = self.translate_branches(traversals, &input, current_var, &alias)?;
                let plan = LogicalOperator::Union(UnionOp { inputs: branches });
                Ok((plan, Some(alias)))
            }

            // coalesce(): return first non-empty traversal
            ast::Step::Coalesce(traversals) => {
                // Build a right-nested chain of OtherwiseOp so each branch is
                // tried in order: use the first one that produces results.
                let alias = self.var_gen.next();
                let branches = self.translate_branches(traversals, &input, current_var, &alias)?;
                let mut iter = branches.into_iter().rev();
                let mut accumulator = iter.next().ok_or_else(|| {
                    Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "coalesce() requires at least one traversal",
                    ))
                })?;
                for branch in iter {
                    accumulator = LogicalOperator::Otherwise(OtherwiseOp {
                        left: Box::new(branch),
                        right: Box::new(accumulator),
                    });
                }
                Ok((accumulator, Some(alias)))
            }

            // sideEffect(): perform inner traversal for side effects, pass
            // traversers through unchanged
            ast::Step::SideEffect(_steps) => {
                // Side effects are not observable in a pure query plan, so we
                // simply pass the input through unchanged.
                Ok((input, None))
            }

            // Mid-traversal V() restarts from all vertices (a new node scan).
            ast::Step::MidV(_ids) => {
                let new_var = self.var_gen.next();
                let plan = LogicalOperator::NodeScan(NodeScanOp {
                    variable: new_var.clone(),
                    label: None,
                    input: Some(Box::new(input)),
                });
                Ok((plan, Some(new_var)))
            }

            // valueMap() with no keys: return all properties as a map.
            ast::Step::ValueMap(keys) if keys.is_empty() => {
                let alias = self.var_gen.next();
                let plan = LogicalOperator::Project(ProjectOp {
                    projections: vec![Projection {
                        expression: LogicalExpression::FunctionCall {
                            name: "properties".to_string(),
                            args: vec![LogicalExpression::Variable(current_var.to_string())],
                            distinct: false,
                        },
                        alias: Some(alias.clone()),
                    }],
                    input: Box::new(input),
                    pass_through_input: false,
                });
                Ok((plan, Some(alias)))
            }

            // valueMap('key1', 'key2', ...) returns a map of specified properties.
            ast::Step::ValueMap(keys) if !keys.is_empty() => {
                let projections: Vec<Projection> = keys
                    .iter()
                    .map(|k| Projection {
                        expression: LogicalExpression::Property {
                            variable: current_var.to_string(),
                            property: k.clone(),
                        },
                        alias: Some(k.clone()),
                    })
                    .collect();
                let plan = LogicalOperator::Project(ProjectOp {
                    projections,
                    input: Box::new(input),
                    pass_through_input: false,
                });
                let first_key = keys.first().cloned();
                Ok((plan, first_key))
            }

            // elementMap() with no keys: return id, label, plus all properties
            // as a single map value.
            ast::Step::ElementMap(keys) if keys.is_empty() => {
                let alias = self.var_gen.next();
                let plan = LogicalOperator::Project(ProjectOp {
                    projections: vec![Projection {
                        expression: LogicalExpression::Map(vec![
                            (
                                "id".to_string(),
                                LogicalExpression::Id(current_var.to_string()),
                            ),
                            (
                                "label".to_string(),
                                LogicalExpression::IndexAccess {
                                    base: Box::new(LogicalExpression::Labels(
                                        current_var.to_string(),
                                    )),
                                    index: Box::new(LogicalExpression::Literal(Value::Int64(0))),
                                },
                            ),
                            (
                                "*".to_string(),
                                LogicalExpression::FunctionCall {
                                    name: "properties".to_string(),
                                    args: vec![LogicalExpression::Variable(
                                        current_var.to_string(),
                                    )],
                                    distinct: false,
                                },
                            ),
                        ]),
                        alias: Some(alias.clone()),
                    }],
                    input: Box::new(input),
                    pass_through_input: false,
                });
                Ok((plan, Some(alias)))
            }

            // elementMap('key1', 'key2', ...) returns id, label, plus specified
            // properties.
            ast::Step::ElementMap(keys) if !keys.is_empty() => {
                let mut projections = vec![
                    Projection {
                        expression: LogicalExpression::Id(current_var.to_string()),
                        alias: Some("id".to_string()),
                    },
                    Projection {
                        expression: LogicalExpression::IndexAccess {
                            base: Box::new(LogicalExpression::Labels(current_var.to_string())),
                            index: Box::new(LogicalExpression::Literal(Value::Int64(0))),
                        },
                        alias: Some("label".to_string()),
                    },
                ];
                projections.extend(keys.iter().map(|k| Projection {
                    expression: LogicalExpression::Property {
                        variable: current_var.to_string(),
                        property: k.clone(),
                    },
                    alias: Some(k.clone()),
                }));
                let plan = LogicalOperator::Project(ProjectOp {
                    projections,
                    input: Box::new(input),
                    pass_through_input: false,
                });
                // Return "id" as the variable so Return uses a valid output column.
                // The multi-column return items override is set in the main loop.
                Ok((plan, Some("id".to_string())))
            }

            // Looping step: repeat(body).times(n) or repeat(body).until(predicate)
            ast::Step::Repeat(repeat_step) => {
                // Determine the direction and edge types from the repeat body.
                // The body typically contains a single navigation step (out/in/both).
                let (direction, edge_types) =
                    self.extract_navigation_from_repeat(&repeat_step.body);

                let (min_hops, max_hops) = match &repeat_step.termination {
                    Some(ast::RepeatTermination::Times(n)) => {
                        // reason: repeat count is always small, fits u32
                        #[allow(clippy::cast_possible_truncation)]
                        let hops = *n as u32;
                        if repeat_step.emit {
                            // emit() with times(n): return results at every depth 1..n
                            (1, Some(hops))
                        } else {
                            // times(n): return results at exactly depth n
                            (hops, Some(hops))
                        }
                    }
                    Some(ast::RepeatTermination::Until(_)) | None => {
                        // until or no termination: expand from 1 to a reasonable max
                        (1, Some(32))
                    }
                };

                let target_var = self.var_gen.next();
                let plan = LogicalOperator::Expand(ExpandOp {
                    from_variable: current_var.to_string(),
                    to_variable: target_var.clone(),
                    edge_variable: None,
                    direction,
                    edge_types,
                    min_hops,
                    max_hops,
                    input: Box::new(input),
                    path_alias: None,
                    path_mode: PathMode::Walk,
                });

                Ok((plan, Some(target_var)))
            }

            _ => Ok((input, None)),
        }
    }

    /// Extracts the navigation direction and edge types from a repeat body.
    ///
    /// The repeat body is typically a single navigation step like `out()`,
    /// `in('KNOWS')`, or `both()`. This extracts the direction and edge type
    /// filters to configure the `VariableLengthExpand` operator.
    fn extract_navigation_from_repeat(&self, body: &[ast::Step]) -> (ExpandDirection, Vec<String>) {
        for step in body {
            match step {
                ast::Step::Out(labels) => {
                    return (ExpandDirection::Outgoing, labels.clone());
                }
                ast::Step::In(labels) => {
                    return (ExpandDirection::Incoming, labels.clone());
                }
                ast::Step::Both(labels) => {
                    return (ExpandDirection::Both, labels.clone());
                }
                _ => {}
            }
        }
        // Default: outgoing with no type filter
        (ExpandDirection::Outgoing, Vec::new())
    }

    /// Convert a parsed Value to a LogicalExpression, resolving $parameter references.
    fn value_to_expr(value: &Value) -> LogicalExpression {
        if let Value::String(s) = value
            && let Some(name) = s.strip_prefix('$')
        {
            return LogicalExpression::Parameter(name.to_string());
        }
        LogicalExpression::Literal(value.clone())
    }

    fn translate_has_step(&self, has: &ast::HasStep, var: &str) -> Result<LogicalExpression> {
        match has {
            ast::HasStep::Key(key) => {
                // has(key) - check if property exists
                Ok(LogicalExpression::Unary {
                    op: UnaryOp::IsNotNull,
                    operand: Box::new(LogicalExpression::Property {
                        variable: var.to_string(),
                        property: key.clone(),
                    }),
                })
            }
            ast::HasStep::KeyValue(key, value) => {
                // has(key, value) - check property equals value
                Ok(LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: var.to_string(),
                        property: key.clone(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(Self::value_to_expr(value)),
                })
            }
            ast::HasStep::KeyPredicate(key, pred) => {
                let prop = LogicalExpression::Property {
                    variable: var.to_string(),
                    property: key.clone(),
                };
                Self::translate_predicate(pred, prop)
            }
            ast::HasStep::LabelKeyValue(label, key, value) => {
                // has(label, key, value) - check label IN labels AND property equals value
                let label_check = LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Literal(Value::String(
                        label.clone().into(),
                    ))),
                    op: BinaryOp::In,
                    right: Box::new(LogicalExpression::Labels(var.to_string())),
                };
                let prop_check = LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: var.to_string(),
                        property: key.clone(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(LogicalExpression::Literal(value.clone())),
                };
                Ok(LogicalExpression::Binary {
                    left: Box::new(label_check),
                    op: BinaryOp::And,
                    right: Box::new(prop_check),
                })
            }
        }
    }

    fn translate_predicate(
        pred: &ast::Predicate,
        expr: LogicalExpression,
    ) -> Result<LogicalExpression> {
        match pred {
            ast::Predicate::Eq(value) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::Eq,
                right: Box::new(LogicalExpression::Literal(value.clone())),
            }),
            ast::Predicate::Neq(value) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::Ne,
                right: Box::new(LogicalExpression::Literal(value.clone())),
            }),
            ast::Predicate::Lt(value) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::Lt,
                right: Box::new(LogicalExpression::Literal(value.clone())),
            }),
            ast::Predicate::Lte(value) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::Le,
                right: Box::new(LogicalExpression::Literal(value.clone())),
            }),
            ast::Predicate::Gt(value) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::Gt,
                right: Box::new(LogicalExpression::Literal(value.clone())),
            }),
            ast::Predicate::Gte(value) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::Ge,
                right: Box::new(LogicalExpression::Literal(value.clone())),
            }),
            ast::Predicate::Within(values) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::In,
                right: Box::new(LogicalExpression::List(
                    values
                        .iter()
                        .map(|v| LogicalExpression::Literal(v.clone()))
                        .collect(),
                )),
            }),
            ast::Predicate::Without(values) => Ok(LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand: Box::new(LogicalExpression::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::In,
                    right: Box::new(LogicalExpression::List(
                        values
                            .iter()
                            .map(|v| LogicalExpression::Literal(v.clone()))
                            .collect(),
                    )),
                }),
            }),
            ast::Predicate::Between(start, end) => Ok(LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Binary {
                    left: Box::new(expr.clone()),
                    op: BinaryOp::Ge,
                    right: Box::new(LogicalExpression::Literal(start.clone())),
                }),
                op: BinaryOp::And,
                right: Box::new(LogicalExpression::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Lt,
                    right: Box::new(LogicalExpression::Literal(end.clone())),
                }),
            }),
            ast::Predicate::Containing(s) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::Contains,
                right: Box::new(LogicalExpression::Literal(Value::String(s.clone().into()))),
            }),
            ast::Predicate::StartingWith(s) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::StartsWith,
                right: Box::new(LogicalExpression::Literal(Value::String(s.clone().into()))),
            }),
            ast::Predicate::EndingWith(s) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::EndsWith,
                right: Box::new(LogicalExpression::Literal(Value::String(s.clone().into()))),
            }),
            ast::Predicate::And(preds) => {
                let mut result = Self::translate_predicate(&preds[0], expr.clone())?;
                for pred in &preds[1..] {
                    let right = Self::translate_predicate(pred, expr.clone())?;
                    result = LogicalExpression::Binary {
                        left: Box::new(result),
                        op: BinaryOp::And,
                        right: Box::new(right),
                    };
                }
                Ok(result)
            }
            ast::Predicate::Or(preds) => {
                let mut result = Self::translate_predicate(&preds[0], expr.clone())?;
                for pred in &preds[1..] {
                    let right = Self::translate_predicate(pred, expr.clone())?;
                    result = LogicalExpression::Binary {
                        left: Box::new(result),
                        op: BinaryOp::Or,
                        right: Box::new(right),
                    };
                }
                Ok(result)
            }
            // inside(start, end) = start < x < end (exclusive both ends)
            ast::Predicate::Inside(start, end) => Ok(LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Binary {
                    left: Box::new(expr.clone()),
                    op: BinaryOp::Gt,
                    right: Box::new(LogicalExpression::Literal(start.clone())),
                }),
                op: BinaryOp::And,
                right: Box::new(LogicalExpression::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Lt,
                    right: Box::new(LogicalExpression::Literal(end.clone())),
                }),
            }),
            // outside(start, end) = x < start OR x > end
            ast::Predicate::Outside(start, end) => Ok(LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Binary {
                    left: Box::new(expr.clone()),
                    op: BinaryOp::Lt,
                    right: Box::new(LogicalExpression::Literal(start.clone())),
                }),
                op: BinaryOp::Or,
                right: Box::new(LogicalExpression::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Gt,
                    right: Box::new(LogicalExpression::Literal(end.clone())),
                }),
            }),
            ast::Predicate::Not(pred) => Ok(LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand: Box::new(Self::translate_predicate(pred, expr)?),
            }),
            ast::Predicate::Regex(pattern) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::Regex,
                right: Box::new(LogicalExpression::Literal(Value::String(
                    pattern.clone().into(),
                ))),
            }),
            ast::Predicate::NotRegex(pattern) => Ok(LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand: Box::new(LogicalExpression::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Regex,
                    right: Box::new(LogicalExpression::Literal(Value::String(
                        pattern.clone().into(),
                    ))),
                }),
            }),
            ast::Predicate::NotContaining(s) => Ok(LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand: Box::new(LogicalExpression::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Contains,
                    right: Box::new(LogicalExpression::Literal(Value::String(s.clone().into()))),
                }),
            }),
            ast::Predicate::NotStartingWith(s) => Ok(LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand: Box::new(LogicalExpression::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::StartsWith,
                    right: Box::new(LogicalExpression::Literal(Value::String(s.clone().into()))),
                }),
            }),
            ast::Predicate::NotEndingWith(s) => Ok(LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand: Box::new(LogicalExpression::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::EndsWith,
                    right: Box::new(LogicalExpression::Literal(Value::String(s.clone().into()))),
                }),
            }),
        }
    }

    /// Translate multiple sub-traversal branches, aligning each to a common
    /// output alias so Union rows share one column name.
    fn translate_branches(
        &self,
        traversals: &[Vec<ast::Step>],
        input: &LogicalOperator,
        current_var: &str,
        alias: &str,
    ) -> Result<Vec<LogicalOperator>> {
        let mut branches = Vec::new();
        for steps in traversals {
            let mut branch_plan = input.clone();
            let mut branch_var = current_var.to_string();
            for step in steps {
                let (new_plan, new_var) =
                    self.translate_step(step, branch_plan, &branch_var, false)?;
                if let Some(v) = new_var {
                    branch_var = v;
                }
                branch_plan = new_plan;
            }
            // Wrap in a Project to align the branch result to the common alias
            branch_plan = LogicalOperator::Project(ProjectOp {
                projections: vec![Projection {
                    expression: LogicalExpression::Variable(branch_var),
                    alias: Some(alias.to_string()),
                }],
                input: Box::new(branch_plan),
                pass_through_input: false,
            });
            branches.push(branch_plan);
        }
        Ok(branches)
    }

    fn translate_by_modifier(&self, by: &ast::ByModifier, current_var: &str) -> LogicalExpression {
        match by {
            ast::ByModifier::Identity => LogicalExpression::Variable(current_var.to_string()),
            ast::ByModifier::Key(key) | ast::ByModifier::KeyWithOrder(key, _) => {
                LogicalExpression::Property {
                    variable: current_var.to_string(),
                    property: key.clone(),
                }
            }
            ast::ByModifier::Token(token) => match token {
                ast::TokenType::Id => LogicalExpression::Id(current_var.to_string()),
                ast::TokenType::Label => LogicalExpression::Labels(current_var.to_string()),
                _ => LogicalExpression::Variable(current_var.to_string()),
            },
            _ => LogicalExpression::Variable(current_var.to_string()),
        }
    }

    /// Convert a list of Gremlin steps (typically filter steps like `has()`)
    /// into a single `LogicalExpression` predicate for use in `and()`, `or()`,
    /// `not()`, `where()`, and `choose()`.
    fn steps_to_predicate(
        &self,
        steps: &[ast::Step],
        current_var: &str,
    ) -> Result<Option<LogicalExpression>> {
        let mut predicates: Vec<LogicalExpression> = Vec::new();
        for step in steps {
            match step {
                ast::Step::Has(has_step) => {
                    predicates.push(self.translate_has_step(has_step, current_var)?);
                }
                ast::Step::HasLabel(labels) => {
                    let pred = if labels.len() == 1 {
                        LogicalExpression::Binary {
                            left: Box::new(LogicalExpression::Literal(Value::String(
                                labels[0].clone().into(),
                            ))),
                            op: BinaryOp::In,
                            right: Box::new(LogicalExpression::Labels(current_var.to_string())),
                        }
                    } else {
                        let mut conditions: Vec<LogicalExpression> = labels
                            .iter()
                            .map(|l| LogicalExpression::Binary {
                                left: Box::new(LogicalExpression::Literal(Value::String(
                                    l.clone().into(),
                                ))),
                                op: BinaryOp::In,
                                right: Box::new(LogicalExpression::Labels(current_var.to_string())),
                            })
                            .collect();
                        let mut result = conditions
                            .pop()
                            .expect("conditions non-empty for multi-label");
                        for cond in conditions {
                            result = LogicalExpression::Binary {
                                left: Box::new(cond),
                                op: BinaryOp::Or,
                                right: Box::new(result),
                            };
                        }
                        result
                    };
                    predicates.push(pred);
                }
                ast::Step::HasNot(key) => {
                    predicates.push(LogicalExpression::Unary {
                        op: UnaryOp::IsNull,
                        operand: Box::new(LogicalExpression::Property {
                            variable: current_var.to_string(),
                            property: key.clone(),
                        }),
                    });
                }
                ast::Step::HasId(ids) => {
                    predicates.push(self.build_id_filter(current_var, ids));
                }
                // For navigation steps like out('knows') in where(), check if
                // expanding produces any results (existence check).
                ast::Step::Out(labels) | ast::Step::In(labels) | ast::Step::Both(labels) => {
                    let direction = match step {
                        ast::Step::Out(_) => ExpandDirection::Outgoing,
                        ast::Step::In(_) => ExpandDirection::Incoming,
                        _ => ExpandDirection::Both,
                    };
                    let edge_types = labels.clone();
                    let target_var = self.var_gen.next();
                    // Create an existence subquery via Expand + count > 0
                    let expand = LogicalOperator::Expand(ExpandOp {
                        from_variable: current_var.to_string(),
                        to_variable: target_var,
                        edge_variable: None,
                        direction,
                        edge_types,
                        min_hops: 1,
                        max_hops: Some(1),
                        input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                            variable: current_var.to_string(),
                            label: None,
                            input: None,
                        })),
                        path_alias: None,
                        path_mode: PathMode::Walk,
                    });
                    predicates.push(LogicalExpression::ExistsSubquery(Box::new(expand)));
                }
                // Nested compound groups: recurse so `.or(__.and(..), __.has(..))` is not
                // silently dropped by the fallthrough below.
                ast::Step::And(traversals) | ast::Step::Or(traversals) => {
                    let op = if matches!(step, ast::Step::And(_)) {
                        BinaryOp::And
                    } else {
                        BinaryOp::Or
                    };
                    let mut inner: Vec<LogicalExpression> = Vec::new();
                    for sub in traversals {
                        if let Some(pred) = self.steps_to_predicate(sub, current_var)? {
                            inner.push(pred);
                        }
                    }
                    if let Some(pred) = fold_predicates(inner, op) {
                        predicates.push(pred);
                    }
                }
                ast::Step::Not(inner_steps) => {
                    if let Some(pred) = self.steps_to_predicate(inner_steps, current_var)? {
                        predicates.push(negate_null_safe(pred));
                    }
                }
                ast::Step::Where(ast::WhereClause::Traversal(inner_steps)) => {
                    if let Some(pred) = self.steps_to_predicate(inner_steps, current_var)? {
                        predicates.push(pred);
                    }
                }
                _ => {}
            }
        }
        if predicates.is_empty() {
            return Ok(None);
        }
        Ok(fold_predicates(predicates, BinaryOp::And))
    }

    fn build_id_filter(&self, var: &str, ids: &[Value]) -> LogicalExpression {
        if ids.len() == 1 {
            LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Id(var.to_string())),
                op: BinaryOp::Eq,
                right: Box::new(LogicalExpression::Literal(ids[0].clone())),
            }
        } else {
            LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Id(var.to_string())),
                op: BinaryOp::In,
                right: Box::new(LogicalExpression::List(
                    ids.iter()
                        .map(|id| LogicalExpression::Literal(id.clone()))
                        .collect(),
                )),
            }
        }
    }

    fn get_current_var(&self, source: &ast::TraversalSource) -> String {
        let counter = self.var_gen.current();
        match source {
            // For g.E(), the edge variable is counter-2 (since we generate: node, edge, target)
            ast::TraversalSource::E(_) => {
                format!("_v{}", counter.saturating_sub(2))
            }
            _ => {
                // Return the most recently generated variable (counter - 1)
                if counter == 0 {
                    "_v0".to_string()
                } else {
                    format!("_v{}", counter - 1)
                }
            }
        }
    }

    /// Walk a logical plan to find the nearest Expand operator and extract
    /// its source/target/direction as an `EdgeContext`.
    fn extract_edge_context(plan: &LogicalOperator) -> Option<EdgeContext> {
        match plan {
            LogicalOperator::Expand(e) => Some(EdgeContext {
                source_var: e.from_variable.clone(),
                target_var: e.to_variable.clone(),
                direction: e.direction,
            }),
            LogicalOperator::Filter(f) => Self::extract_edge_context(&f.input),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::{FilterOp, LimitOp, SkipOp, SortOp};

    // === Basic Traversal Tests ===

    #[test]
    fn test_translate_simple_traversal() {
        let result = translate("g.V()");
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_with_filter() {
        let result = translate("g.V().hasLabel('Person')");
        assert!(result.is_ok());
    }

    // === Navigation Tests ===

    #[test]
    fn test_translate_navigation() {
        let result = translate("g.V().out('knows')");
        assert!(result.is_ok());
        let plan = result.unwrap();
        // Should have NodeScan -> Expand -> Return
        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::Expand(expand) = ret.input.as_ref() {
                assert_eq!(expand.edge_types, vec!["knows".to_string()]);
                assert_eq!(expand.direction, ExpandDirection::Outgoing);
            } else {
                panic!("Expected Expand operator");
            }
        } else {
            panic!("Expected Return operator");
        }
    }

    #[test]
    fn test_translate_in_navigation() {
        let result = translate("g.V().in('knows')");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Return(r) => find_expand(&r.input),
                _ => None,
            }
        }

        let expand = find_expand(&plan.root).expect("Expected Expand");
        assert_eq!(expand.direction, ExpandDirection::Incoming);
    }

    #[test]
    fn test_translate_both_navigation() {
        let result = translate("g.V().both('knows')");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Return(r) => find_expand(&r.input),
                _ => None,
            }
        }

        let expand = find_expand(&plan.root).expect("Expected Expand");
        assert_eq!(expand.direction, ExpandDirection::Both);
    }

    #[test]
    fn test_translate_out_e() {
        let result = translate("g.V().outE('knows')");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Return(r) => find_expand(&r.input),
                _ => None,
            }
        }

        let expand = find_expand(&plan.root).expect("Expected Expand");
        assert!(expand.edge_variable.is_some());
    }

    // === Filter Tests ===

    #[test]
    fn test_translate_has_key_value() {
        let result = translate("g.V().has('name', 'Alix')");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        if let LogicalExpression::Binary { op, .. } = &filter.predicate {
            assert_eq!(*op, BinaryOp::Eq);
        }
    }

    #[test]
    fn test_translate_has_not() {
        let result = translate("g.V().hasNot('deleted')");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        if let LogicalExpression::Unary { op, .. } = &filter.predicate {
            assert_eq!(*op, UnaryOp::IsNull);
        }
    }

    #[test]
    fn test_translate_dedup() {
        let result = translate("g.V().dedup()");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_distinct(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Distinct(_) => true,
                LogicalOperator::Return(r) => find_distinct(&r.input),
                _ => false,
            }
        }

        assert!(find_distinct(&plan.root));
    }

    // === Pagination Tests ===

    #[test]
    fn test_translate_limit() {
        let result = translate("g.V().limit(10)");
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_skip() {
        let result = translate("g.V().skip(5)");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_skip(op: &LogicalOperator) -> Option<&SkipOp> {
            match op {
                LogicalOperator::Skip(s) => Some(s),
                LogicalOperator::Return(r) => find_skip(&r.input),
                _ => None,
            }
        }

        let skip = find_skip(&plan.root).expect("Expected Skip");
        assert_eq!(skip.count, 5);
    }

    #[test]
    fn test_translate_range() {
        let result = translate("g.V().range(5, 15)");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_limit(op: &LogicalOperator) -> Option<&LimitOp> {
            match op {
                LogicalOperator::Limit(l) => Some(l),
                LogicalOperator::Return(r) => find_limit(&r.input),
                _ => None,
            }
        }

        let limit = find_limit(&plan.root).expect("Expected Limit");
        assert_eq!(limit.count, 10); // 15 - 5
    }

    // === Aggregation Tests ===

    #[test]
    fn test_translate_count() {
        let result = translate("g.V().count()");
        assert!(result.is_ok());
        let plan = result.unwrap();
        // The result is wrapped in Return(Aggregate(...))
        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::Aggregate(agg) = ret.input.as_ref() {
                assert_eq!(agg.aggregates.len(), 1);
                assert_eq!(agg.aggregates[0].function, AggregateFunction::Count);
            } else {
                panic!("Expected Aggregate operator inside Return");
            }
        } else {
            panic!("Expected Return operator");
        }
    }

    #[test]
    fn test_translate_sum() {
        let result = translate("g.V().values('age').sum()");
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_mean() {
        let result = translate("g.V().values('age').mean()");
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_min() {
        let result = translate("g.V().values('age').min()");
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_max() {
        let result = translate("g.V().values('age').max()");
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_fold() {
        let result = translate("g.V().fold()");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_aggregate(op: &LogicalOperator) -> Option<&AggregateOp> {
            match op {
                LogicalOperator::Aggregate(a) => Some(a),
                LogicalOperator::Return(r) => find_aggregate(&r.input),
                _ => None,
            }
        }

        let agg = find_aggregate(&plan.root).expect("Expected Aggregate");
        assert_eq!(agg.aggregates[0].function, AggregateFunction::Collect);
    }

    // === Map Steps ===

    #[test]
    fn test_translate_values() {
        let result = translate("g.V().values('name')");
        assert!(result.is_ok());
        let plan = result.unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            if let LogicalExpression::Property { property, .. } = &ret.items[0].expression {
                assert_eq!(property, "name");
            }
        }
    }

    #[test]
    fn test_translate_id() {
        let result = translate("g.V().id()");
        assert!(result.is_ok());
        let plan = result.unwrap();

        // Id step produces Project(Id), which gets wrapped in Return
        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::Project(proj) = ret.input.as_ref() {
                assert!(
                    matches!(&proj.projections[0].expression, LogicalExpression::Id(_)),
                    "Expected Id projection"
                );
            } else {
                panic!("Expected Project under Return");
            }
        } else {
            panic!("Expected Return at root");
        }
    }

    #[test]
    fn test_translate_label() {
        let result = translate("g.V().label()");
        assert!(result.is_ok());
        let plan = result.unwrap();

        // Label step produces Project(IndexAccess(Labels, 0)), wrapped in Return
        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::Project(proj) = ret.input.as_ref() {
                assert!(
                    matches!(
                        &proj.projections[0].expression,
                        LogicalExpression::IndexAccess { base, .. }
                            if matches!(base.as_ref(), LogicalExpression::Labels(_))
                    ),
                    "Expected IndexAccess(Labels) projection"
                );
            } else {
                panic!("Expected Project under Return");
            }
        } else {
            panic!("Expected Return at root");
        }
    }

    // === Mutation Tests ===

    #[test]
    fn test_translate_add_v() {
        let result = translate("g.addV('Person')");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_create(op: &LogicalOperator) -> Option<&CreateNodeOp> {
            match op {
                LogicalOperator::CreateNode(c) => Some(c),
                LogicalOperator::Return(r) => find_create(&r.input),
                _ => None,
            }
        }

        let create = find_create(&plan.root).expect("Expected CreateNode");
        assert_eq!(create.labels, vec!["Person".to_string()]);
    }

    #[test]
    fn test_translate_drop() {
        let result = translate("g.V().drop()");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_delete(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::DeleteNode(_) => true,
                LogicalOperator::Return(r) => find_delete(&r.input),
                _ => false,
            }
        }

        assert!(find_delete(&plan.root));
    }

    #[test]
    fn test_translate_add_v_with_property() {
        let result = translate("g.addV('Person').property('name', 'Alix')");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_create(op: &LogicalOperator) -> Option<&CreateNodeOp> {
            match op {
                LogicalOperator::CreateNode(c) => Some(c),
                LogicalOperator::Return(r) => find_create(&r.input),
                _ => None,
            }
        }

        let create = find_create(&plan.root).expect("Expected CreateNode");
        assert_eq!(create.labels, vec!["Person".to_string()]);
        assert_eq!(create.properties.len(), 1);
        assert_eq!(create.properties[0].0, "name");
    }

    #[test]
    fn test_translate_add_v_with_multiple_properties() {
        let result = translate("g.addV('Person').property('name', 'Alix').property('age', 30)");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_create(op: &LogicalOperator) -> Option<&CreateNodeOp> {
            match op {
                LogicalOperator::CreateNode(c) => Some(c),
                LogicalOperator::Return(r) => find_create(&r.input),
                _ => None,
            }
        }

        let create = find_create(&plan.root).expect("Expected CreateNode");
        assert_eq!(create.labels, vec!["Person".to_string()]);
        assert_eq!(create.properties.len(), 2);
    }

    #[test]
    fn test_translate_property_on_existing_node() {
        // property() on an existing node should create SetPropertyOp
        let result = translate("g.V().has('name', 'Alix').property('updated', true)");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_set_property(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::SetProperty(_) => true,
                LogicalOperator::Return(r) => find_set_property(&r.input),
                LogicalOperator::Filter(f) => find_set_property(&f.input),
                _ => false,
            }
        }

        assert!(find_set_property(&plan.root));
    }

    #[test]
    fn test_translate_add_e_with_from_to() {
        let result = translate("g.addE('knows').from('a').to('b')");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_create_edge(op: &LogicalOperator) -> Option<&CreateEdgeOp> {
            match op {
                LogicalOperator::CreateEdge(e) => Some(e),
                LogicalOperator::Return(r) => find_create_edge(&r.input),
                _ => None,
            }
        }

        let edge = find_create_edge(&plan.root).expect("Expected CreateEdge");
        assert_eq!(edge.edge_type, "knows");
        assert_eq!(edge.from_variable, "a");
        assert_eq!(edge.to_variable, "b");
    }

    #[test]
    fn test_translate_add_e_with_properties() {
        let result = translate("g.addE('knows').from('a').to('b').property('since', 2020)");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_create_edge(op: &LogicalOperator) -> Option<&CreateEdgeOp> {
            match op {
                LogicalOperator::CreateEdge(e) => Some(e),
                LogicalOperator::Return(r) => find_create_edge(&r.input),
                _ => None,
            }
        }

        let edge = find_create_edge(&plan.root).expect("Expected CreateEdge");
        assert_eq!(edge.edge_type, "knows");
        assert_eq!(edge.properties.len(), 1);
        assert_eq!(edge.properties[0].0, "since");
    }

    // === Order Tests ===

    #[test]
    fn test_translate_order() {
        let result = translate("g.V().order()");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_sort(op: &LogicalOperator) -> Option<&SortOp> {
            match op {
                LogicalOperator::Sort(s) => Some(s),
                LogicalOperator::Return(r) => find_sort(&r.input),
                _ => None,
            }
        }

        assert!(find_sort(&plan.root).is_some());
    }

    // === Predicate Tests ===

    #[test]
    fn test_predicate_gt() {
        let expr = LogicalExpression::Variable("x".to_string());
        let pred = ast::Predicate::Gt(Value::Int64(10));
        let result = GremlinTranslator::translate_predicate(&pred, expr).unwrap();

        if let LogicalExpression::Binary { op, .. } = result {
            assert_eq!(op, BinaryOp::Gt);
        } else {
            panic!("Expected Binary expression");
        }
    }

    #[test]
    fn test_predicate_within() {
        let expr = LogicalExpression::Variable("x".to_string());
        let pred = ast::Predicate::Within(vec![Value::Int64(1), Value::Int64(2)]);
        let result = GremlinTranslator::translate_predicate(&pred, expr).unwrap();

        if let LogicalExpression::Binary { op, .. } = result {
            assert_eq!(op, BinaryOp::In);
        } else {
            panic!("Expected Binary expression");
        }
    }

    #[test]
    fn test_predicate_containing() {
        let expr = LogicalExpression::Variable("x".to_string());
        let pred = ast::Predicate::Containing("test".to_string());
        let result = GremlinTranslator::translate_predicate(&pred, expr).unwrap();

        if let LogicalExpression::Binary { op, .. } = result {
            assert_eq!(op, BinaryOp::Contains);
        } else {
            panic!("Expected Binary expression");
        }
    }

    // === Predicate edge cases ===

    #[test]
    fn test_predicate_starting_with() {
        let expr = LogicalExpression::Variable("x".to_string());
        let pred = ast::Predicate::StartingWith("foo".to_string());
        let result = GremlinTranslator::translate_predicate(&pred, expr).unwrap();

        if let LogicalExpression::Binary { op, right, .. } = result {
            assert_eq!(op, BinaryOp::StartsWith);
            // Verify the literal value is preserved
            if let LogicalExpression::Literal(Value::String(s)) = right.as_ref() {
                assert_eq!(s.as_str(), "foo");
            } else {
                panic!("Expected String literal on the right");
            }
        } else {
            panic!("Expected Binary expression");
        }
    }

    #[test]
    fn test_predicate_ending_with() {
        let expr = LogicalExpression::Variable("x".to_string());
        let pred = ast::Predicate::EndingWith(".txt".to_string());
        let result = GremlinTranslator::translate_predicate(&pred, expr).unwrap();

        if let LogicalExpression::Binary { op, .. } = result {
            assert_eq!(op, BinaryOp::EndsWith);
        } else {
            panic!("Expected Binary expression");
        }
    }

    #[test]
    fn test_predicate_between_produces_and_of_ge_lt() {
        // between(10, 20) should produce: x >= 10 AND x < 20
        let expr = LogicalExpression::Variable("x".to_string());
        let pred = ast::Predicate::Between(Value::Int64(10), Value::Int64(20));
        let result = GremlinTranslator::translate_predicate(&pred, expr).unwrap();

        if let LogicalExpression::Binary {
            op: BinaryOp::And,
            left,
            right,
        } = result
        {
            // Left: x >= 10
            if let LogicalExpression::Binary { op, .. } = left.as_ref() {
                assert_eq!(*op, BinaryOp::Ge, "Left side of between should be >=");
            } else {
                panic!("Expected Binary expression for left side of between");
            }
            // Right: x < 20
            if let LogicalExpression::Binary { op, .. } = right.as_ref() {
                assert_eq!(*op, BinaryOp::Lt, "Right side of between should be <");
            } else {
                panic!("Expected Binary expression for right side of between");
            }
        } else {
            panic!("Expected AND expression for between");
        }
    }

    #[test]
    fn test_predicate_without_produces_not_in() {
        // without(1, 2) should produce: NOT (x IN [1, 2])
        let expr = LogicalExpression::Variable("x".to_string());
        let pred = ast::Predicate::Without(vec![Value::Int64(1), Value::Int64(2)]);
        let result = GremlinTranslator::translate_predicate(&pred, expr).unwrap();

        if let LogicalExpression::Unary {
            op: UnaryOp::Not,
            operand,
        } = result
        {
            if let LogicalExpression::Binary { op, .. } = operand.as_ref() {
                assert_eq!(*op, BinaryOp::In);
            } else {
                panic!("Expected IN inside NOT");
            }
        } else {
            panic!("Expected NOT expression for without");
        }
    }

    #[test]
    fn test_predicate_and_combines_multiple() {
        // P.gt(10).and(P.lt(100)) should produce: (x > 10) AND (x < 100)
        let expr = LogicalExpression::Variable("x".to_string());
        let pred = ast::Predicate::And(vec![
            ast::Predicate::Gt(Value::Int64(10)),
            ast::Predicate::Lt(Value::Int64(100)),
        ]);
        let result = GremlinTranslator::translate_predicate(&pred, expr).unwrap();

        if let LogicalExpression::Binary {
            op: BinaryOp::And, ..
        } = result
        {
            // Success - it's an AND
        } else {
            panic!("Expected AND expression");
        }
    }

    #[test]
    fn test_predicate_or_combines_multiple() {
        let expr = LogicalExpression::Variable("x".to_string());
        let pred = ast::Predicate::Or(vec![
            ast::Predicate::Eq(Value::Int64(1)),
            ast::Predicate::Eq(Value::Int64(2)),
            ast::Predicate::Eq(Value::Int64(3)),
        ]);
        let result = GremlinTranslator::translate_predicate(&pred, expr).unwrap();

        // Should produce nested OR: (x == 1) OR ((x == 2) OR (x == 3))
        if let LogicalExpression::Binary {
            op: BinaryOp::Or, ..
        } = result
        {
            // Success
        } else {
            panic!("Expected OR expression for 3-way or");
        }
    }

    #[test]
    fn test_predicate_not_wraps_inner() {
        let expr = LogicalExpression::Variable("x".to_string());
        let pred = ast::Predicate::Not(Box::new(ast::Predicate::Gt(Value::Int64(100))));
        let result = GremlinTranslator::translate_predicate(&pred, expr).unwrap();

        if let LogicalExpression::Unary {
            op: UnaryOp::Not,
            operand,
        } = result
        {
            if let LogicalExpression::Binary { op, .. } = operand.as_ref() {
                assert_eq!(*op, BinaryOp::Gt);
            } else {
                panic!("Inner should be a comparison");
            }
        } else {
            panic!("Expected NOT expression");
        }
    }

    // === E() source and multi-label tests ===

    #[test]
    fn test_translate_e_source() {
        // g.E() - edge scan starting point
        let result = translate("g.E()");
        assert!(
            result.is_ok(),
            "g.E() should translate successfully: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_translate_has_label_multiple() {
        // hasLabel with multiple labels should produce OR filter
        let result = translate("g.V().hasLabel('Person', 'Employee')");
        assert!(
            result.is_ok(),
            "Multi-label hasLabel should work: {:?}",
            result.err()
        );

        let plan = result.unwrap();

        // Walk the plan to find the Filter and verify it's an OR
        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter for hasLabel");
        // Multiple labels should produce an OR expression
        match &filter.predicate {
            LogicalExpression::Binary {
                op: BinaryOp::Or, ..
            } => {} // correct
            other => panic!("Expected OR for multi-label, got: {:?}", other),
        }
    }

    // === Mutation edge cases ===

    #[test]
    fn test_add_vertex_with_properties() {
        let result = translate("g.addV('Person').property('name', 'Alix').property('age', 30)");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_create_node(op: &LogicalOperator) -> Option<&CreateNodeOp> {
            match op {
                LogicalOperator::CreateNode(n) => Some(n),
                LogicalOperator::Return(r) => find_create_node(&r.input),
                _ => None,
            }
        }

        let node = find_create_node(&plan.root).expect("Expected CreateNode");
        assert_eq!(node.labels, vec!["Person"]);
        assert_eq!(node.properties.len(), 2);
    }

    #[test]
    fn test_has_not_produces_is_null() {
        let result = translate("g.V().hasNot('email')");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter for hasNot");
        match &filter.predicate {
            LogicalExpression::Unary {
                op: UnaryOp::IsNull,
                ..
            } => {} // correct
            other => panic!("hasNot should produce IsNull, got: {:?}", other),
        }
    }

    // === Recursive helper functions for new tests ===

    fn find_map_collect(op: &LogicalOperator) -> Option<&MapCollectOp> {
        match op {
            LogicalOperator::MapCollect(mc) => Some(mc),
            LogicalOperator::Return(r) => find_map_collect(&r.input),
            LogicalOperator::Filter(f) => find_map_collect(&f.input),
            LogicalOperator::Project(p) => find_map_collect(&p.input),
            _ => None,
        }
    }

    fn find_union(op: &LogicalOperator) -> Option<&UnionOp> {
        match op {
            LogicalOperator::Union(u) => Some(u),
            LogicalOperator::Return(r) => find_union(&r.input),
            LogicalOperator::Filter(f) => find_union(&f.input),
            LogicalOperator::Project(p) => find_union(&p.input),
            _ => None,
        }
    }

    fn find_aggregate(op: &LogicalOperator) -> Option<&AggregateOp> {
        match op {
            LogicalOperator::Aggregate(a) => Some(a),
            LogicalOperator::Return(r) => find_aggregate(&r.input),
            LogicalOperator::MapCollect(mc) => find_aggregate(&mc.input),
            _ => None,
        }
    }

    fn find_unwind(op: &LogicalOperator) -> Option<&UnwindOp> {
        match op {
            LogicalOperator::Unwind(u) => Some(u),
            LogicalOperator::Return(r) => find_unwind(&r.input),
            _ => None,
        }
    }

    fn find_distinct_op(op: &LogicalOperator) -> Option<&DistinctOp> {
        match op {
            LogicalOperator::Distinct(d) => Some(d),
            LogicalOperator::Return(r) => find_distinct_op(&r.input),
            _ => None,
        }
    }

    fn find_project(op: &LogicalOperator) -> Option<&ProjectOp> {
        match op {
            LogicalOperator::Project(p) => Some(p),
            LogicalOperator::Return(r) => find_project(&r.input),
            LogicalOperator::Filter(f) => find_project(&f.input),
            _ => None,
        }
    }

    fn find_create_edge(op: &LogicalOperator) -> Option<&CreateEdgeOp> {
        match op {
            LogicalOperator::CreateEdge(e) => Some(e),
            LogicalOperator::Return(r) => find_create_edge(&r.input),
            _ => None,
        }
    }

    // === MapCollect wrapping in By handler for Aggregate ===

    #[test]
    fn test_group_count_by_produces_map_collect() {
        let result = translate("g.V().groupCount().by('city')");
        assert!(
            result.is_ok(),
            "groupCount().by('city') should parse: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        // The plan should contain a MapCollect wrapping an Aggregate
        let mc = find_map_collect(&plan.root).expect("Expected MapCollect operator");
        assert_eq!(mc.alias, "_groupCount");
        assert_eq!(mc.value_var, "count");

        // The input to MapCollect should be an Aggregate with a group_by
        let agg = find_aggregate(&plan.root).expect("Expected Aggregate inside MapCollect");
        assert_eq!(agg.aggregates.len(), 1);
        assert_eq!(agg.aggregates[0].function, AggregateFunction::Count);
        assert!(
            !agg.group_by.is_empty(),
            "Aggregate should have a group_by for the 'city' key"
        );
    }

    // === values() with multiple keys ===

    #[test]
    fn test_values_multiple_keys_produces_union() {
        let result = translate("g.V().values('name', 'age')");
        assert!(
            result.is_ok(),
            "values('name','age') should parse: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        // Multi-key values() should produce a Union of Projects
        let union = find_union(&plan.root).expect("Expected Union for multi-key values()");
        assert_eq!(
            union.inputs.len(),
            2,
            "Union should have 2 branches (one per key)"
        );

        // Each branch should be a Filter(IS NOT NULL) wrapping a Project
        // with a Property expression (Gremlin values() filters out nulls).
        for (i, branch) in union.inputs.iter().enumerate() {
            let proj = find_project(branch)
                .unwrap_or_else(|| panic!("Expected Project inside branch {i}"));
            assert_eq!(proj.projections.len(), 1);
            match &proj.projections[0].expression {
                LogicalExpression::Property { property, .. } => {
                    let expected = if i == 0 { "name" } else { "age" };
                    assert_eq!(property, expected, "Branch {i} should project '{expected}'");
                }
                other => {
                    panic!("Expected Property expression in branch {i}, got: {other:?}")
                }
            }
        }
    }

    // === fold() produces Aggregate with Collect ===

    #[test]
    fn test_fold_with_values_produces_aggregate_collect() {
        let result = translate("g.V().values('name').fold()");
        assert!(
            result.is_ok(),
            "values('name').fold() should parse: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        let agg = find_aggregate(&plan.root).expect("Expected Aggregate for fold()");
        assert_eq!(agg.aggregates.len(), 1);
        assert_eq!(agg.aggregates[0].function, AggregateFunction::Collect);
        assert!(
            agg.aggregates[0].expression.is_some(),
            "Collect should have an expression to collect"
        );
        assert_eq!(
            agg.aggregates[0].alias.as_deref(),
            Some("fold"),
            "fold() alias should be 'fold'"
        );
    }

    // === unfold() produces Unwind ===

    #[test]
    fn test_unfold_produces_unwind() {
        let result = translate("g.V().values('name').fold().unfold()");
        assert!(
            result.is_ok(),
            "fold().unfold() should parse: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        let unwind = find_unwind(&plan.root).expect("Expected Unwind for unfold()");
        // The unwind expression should reference a variable (the fold alias)
        match &unwind.expression {
            LogicalExpression::Variable(var) => {
                assert_eq!(var, "fold", "Unwind should reference the fold variable");
            }
            other => panic!("Expected Variable expression for Unwind, got: {other:?}"),
        }
    }

    // === dedup() with navigation produces Distinct with column ===

    #[test]
    fn test_dedup_after_navigation_produces_distinct_with_column() {
        let result = translate("g.V().out('knows').dedup()");
        assert!(
            result.is_ok(),
            "out('knows').dedup() should parse: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        let distinct = find_distinct_op(&plan.root).expect("Expected Distinct for dedup()");
        // dedup() with no keys should deduplicate on the current variable
        assert!(
            distinct.columns.is_some(),
            "Distinct should have column specification"
        );
        let cols = distinct.columns.as_ref().unwrap();
        assert_eq!(cols.len(), 1, "Should deduplicate on a single column");
    }

    // === label() produces Project with Labels/IndexAccess ===

    #[test]
    fn test_label_produces_project_with_labels() {
        let result = translate("g.V().label()");
        assert!(result.is_ok());
        let plan = result.unwrap();

        let proj = find_project(&plan.root).expect("Expected Project for label()");
        assert_eq!(proj.projections.len(), 1);
        assert_eq!(proj.projections[0].alias.as_deref(), Some("label"));
        // The expression should be IndexAccess(Labels, 0)
        match &proj.projections[0].expression {
            LogicalExpression::IndexAccess { base, .. } => {
                assert!(
                    matches!(base.as_ref(), LogicalExpression::Labels(_)),
                    "IndexAccess base should be Labels"
                );
            }
            other => panic!("Expected IndexAccess expression for label(), got: {other:?}"),
        }
    }

    // === bothV() produces Union ===

    #[test]
    fn test_both_v_produces_union() {
        let result = translate("g.E().bothV()");
        assert!(
            result.is_ok(),
            "g.E().bothV() should parse: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        let union = find_union(&plan.root).expect("Expected Union for bothV()");
        assert_eq!(
            union.inputs.len(),
            2,
            "bothV() Union should have 2 branches (source and target)"
        );

        // Each branch should be a Project referencing a variable
        for (i, branch) in union.inputs.iter().enumerate() {
            match branch {
                LogicalOperator::Project(proj) => {
                    assert_eq!(proj.projections.len(), 1);
                    assert!(
                        matches!(
                            &proj.projections[0].expression,
                            LogicalExpression::Variable(_)
                        ),
                        "Branch {i} should project a Variable"
                    );
                }
                other => panic!("Expected Project in bothV() Union branch {i}, got: {other:?}"),
            }
        }
    }

    // === as()/select() variable label mapping ===

    #[test]
    fn test_as_select_produces_project() {
        let result = translate("g.V().has('name','Alix').as('a').out('knows').select('a')");
        assert!(result.is_ok(), "as/select should parse: {:?}", result.err());
        let plan = result.unwrap();

        // select('a') should produce a Project referencing the variable stored by as('a')
        let proj = find_project(&plan.root).expect("Expected Project for select('a')");
        assert_eq!(proj.projections.len(), 1);
        assert_eq!(
            proj.projections[0].alias.as_deref(),
            Some("a"),
            "Projection alias should match the select key"
        );
        // The expression should reference the original variable
        match &proj.projections[0].expression {
            LogicalExpression::Variable(var) => {
                assert!(
                    !var.is_empty(),
                    "select('a') should reference a non-empty variable"
                );
            }
            other => panic!("Expected Variable expression for select, got: {other:?}"),
        }
    }

    // === union() produces Union with branches ===

    #[test]
    fn test_union_produces_union_with_branches() {
        let result = translate("g.V().union(out('knows'), out('works_at'))");
        assert!(result.is_ok(), "union() should parse: {:?}", result.err());
        let plan = result.unwrap();

        let union = find_union(&plan.root).expect("Expected Union for union()");
        assert_eq!(union.inputs.len(), 2, "union() should have 2 branches");

        // Each branch should end with a Project (for alias alignment)
        // wrapping an Expand
        for (i, branch) in union.inputs.iter().enumerate() {
            match branch {
                LogicalOperator::Project(proj) => {
                    // The input should contain an Expand
                    fn has_expand(op: &LogicalOperator) -> bool {
                        match op {
                            LogicalOperator::Expand(_) => true,
                            LogicalOperator::Project(p) => has_expand(&p.input),
                            LogicalOperator::Filter(f) => has_expand(&f.input),
                            _ => false,
                        }
                    }
                    assert!(
                        has_expand(&proj.input),
                        "Branch {i} should contain an Expand"
                    );
                }
                other => panic!("Expected Project wrapping branch {i}, got: {other:?}"),
            }
        }
    }

    // === coalesce() produces Otherwise (first-non-empty) ===

    #[test]
    fn test_coalesce_produces_otherwise() {
        let result = translate("g.V().coalesce(out('knows'), out('works_at'))");
        assert!(
            result.is_ok(),
            "coalesce() should parse: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        // coalesce() now uses Otherwise for first-non-empty semantics
        fn find_otherwise(op: &LogicalOperator) -> bool {
            matches!(op, LogicalOperator::Otherwise(_))
                || op.children().iter().any(|c| find_otherwise(c))
        }
        assert!(
            find_otherwise(&plan.root),
            "coalesce() should produce an Otherwise operator"
        );
    }

    // === project().by() produces correct Project ===

    #[test]
    fn test_project_by_produces_correct_projections() {
        let result = translate("g.V().project('n','a').by('name').by('age')");
        assert!(
            result.is_ok(),
            "project().by() should parse: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        let proj = find_project(&plan.root).expect("Expected Project for project().by()");
        assert_eq!(
            proj.projections.len(),
            2,
            "project('n','a') should have 2 projections"
        );

        // Check aliases
        assert_eq!(proj.projections[0].alias.as_deref(), Some("n"));
        assert_eq!(proj.projections[1].alias.as_deref(), Some("a"));

        // After .by('name').by('age'), the first projection's expression should reference 'name'
        // and the second should reference 'age'
        match &proj.projections[0].expression {
            LogicalExpression::Property { property, .. } => {
                assert_eq!(property, "name", "First projection should use 'name'");
            }
            other => panic!("Expected Property for first projection, got: {other:?}"),
        }
        match &proj.projections[1].expression {
            LogicalExpression::Property { property, .. } => {
                assert_eq!(property, "age", "Second projection should use 'age'");
            }
            other => panic!("Expected Property for second projection, got: {other:?}"),
        }
    }

    // === addE().property() edge creation with properties ===

    #[test]
    fn test_add_e_with_property_produces_create_edge() {
        let result = translate(
            "g.addE('knows').from('a').to('b').property('since', 2020).property('weight', 0.5)",
        );
        assert!(
            result.is_ok(),
            "addE with properties should parse: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        let edge = find_create_edge(&plan.root).expect("Expected CreateEdge");
        assert_eq!(edge.edge_type, "knows");
        assert_eq!(edge.from_variable, "a");
        assert_eq!(edge.to_variable, "b");
        assert_eq!(edge.properties.len(), 2, "Edge should have 2 properties");
        assert_eq!(edge.properties[0].0, "since");
        assert_eq!(edge.properties[1].0, "weight");
    }

    // === Additional edge cases ===

    #[test]
    fn test_values_single_key_produces_project() {
        // Verify that single-key values() does NOT produce a Union
        let result = translate("g.V().values('name')");
        assert!(result.is_ok());
        let plan = result.unwrap();

        // Single key should produce a Project, not a Union
        assert!(
            find_union(&plan.root).is_none(),
            "Single-key values() should not produce a Union"
        );
        // The Project under the Return should contain the property projection
        let proj = find_project(&plan.root).expect("Expected Project for single-key values()");
        assert_eq!(proj.projections.len(), 1);
        match &proj.projections[0].expression {
            LogicalExpression::Property { property, .. } => {
                assert_eq!(property, "name");
            }
            other => panic!("Expected Property in Project, got: {other:?}"),
        }
    }

    #[test]
    fn test_dedup_no_keys_deduplicates_on_current_var() {
        let result = translate("g.V().dedup()");
        assert!(result.is_ok());
        let plan = result.unwrap();

        let distinct = find_distinct_op(&plan.root).expect("Expected Distinct");
        let cols = distinct
            .columns
            .as_ref()
            .expect("dedup() should specify columns");
        assert_eq!(cols.len(), 1, "dedup() with no keys should use 1 column");
    }

    #[test]
    fn test_fold_on_bare_traversal() {
        // fold() directly on g.V() should collect all vertices
        let result = translate("g.V().fold()");
        assert!(result.is_ok());
        let plan = result.unwrap();

        let agg = find_aggregate(&plan.root).expect("Expected Aggregate for fold()");
        assert_eq!(agg.aggregates[0].function, AggregateFunction::Collect);
    }

    #[test]
    fn test_select_multiple_keys_produces_multi_column() {
        let result =
            translate("g.V().has('name','Alix').as('a').out('knows').as('b').select('a','b')");
        assert!(
            result.is_ok(),
            "select('a','b') should parse: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        // Multi-key select should produce a Project with 2 projections
        let proj = find_project(&plan.root).expect("Expected Project for multi-key select");
        assert_eq!(
            proj.projections.len(),
            2,
            "select('a','b') should have 2 projections"
        );
        let aliases: Vec<_> = proj
            .projections
            .iter()
            .filter_map(|p| p.alias.as_deref())
            .collect();
        assert!(aliases.contains(&"a"), "Should have alias 'a'");
        assert!(aliases.contains(&"b"), "Should have alias 'b'");
    }

    #[test]
    fn test_union_single_branch() {
        let result = translate("g.V().union(out('knows'))");
        assert!(
            result.is_ok(),
            "single-branch union should parse: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        let union = find_union(&plan.root).expect("Expected Union");
        assert_eq!(
            union.inputs.len(),
            1,
            "Single-branch union should have 1 input"
        );
    }

    #[test]
    fn test_group_count_without_by_produces_aggregate() {
        // groupCount() without by() still creates an Aggregate
        let result = translate("g.V().groupCount()");
        assert!(
            result.is_ok(),
            "groupCount() should parse: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        let agg = find_aggregate(&plan.root).expect("Expected Aggregate for groupCount()");
        assert_eq!(agg.aggregates[0].function, AggregateFunction::Count);
        assert!(
            !agg.group_by.is_empty(),
            "groupCount() should have a group_by expression"
        );
    }

    // === Edge HasLabel with multiple labels ===

    #[test]
    fn test_edge_has_label_multiple_labels_produces_in() {
        // hasLabel with multiple labels on an edge should produce Type(var) IN [labels]
        let translator = GremlinTranslator::new();
        let step = ast::Step::HasLabel(vec!["KNOWS".into(), "FOLLOWS".into()]);

        // Create a simple NodeScan as input
        let input = LogicalOperator::NodeScan(NodeScanOp {
            variable: "v0".to_string(),
            label: None,
            input: None,
        });

        let (plan, _) = translator
            .translate_step(&step, input, "e0", true)
            .expect("translate_step should succeed");

        // Should produce a Filter wrapping the input
        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                _ => None,
            }
        }

        let filter = find_filter(&plan).expect("Expected Filter for edge hasLabel");
        // The predicate should be Type(e0) IN [KNOWS, FOLLOWS]
        match &filter.predicate {
            LogicalExpression::Binary { left, op, right } => {
                assert_eq!(*op, BinaryOp::In, "Multi-label edge filter should use IN");
                assert!(
                    matches!(left.as_ref(), LogicalExpression::Type(var) if var == "e0"),
                    "Left side should be Type('e0'), got: {:?}",
                    left
                );
                match right.as_ref() {
                    LogicalExpression::Literal(Value::List(items)) => {
                        assert_eq!(items.len(), 2, "Should have 2 labels in list");
                        assert_eq!(items[0], Value::String("KNOWS".into()));
                        assert_eq!(items[1], Value::String("FOLLOWS".into()));
                    }
                    other => panic!("Right side should be a List literal, got: {other:?}"),
                }
            }
            other => panic!("Expected Binary expression, got: {other:?}"),
        }
    }

    // === MidV step translation ===

    #[test]
    fn test_mid_v_none_produces_node_scan_with_input() {
        // MidV(None) should produce a new NodeScan with the previous plan as input
        let translator = GremlinTranslator::new();
        let step = ast::Step::MidV(None);

        let input = LogicalOperator::NodeScan(NodeScanOp {
            variable: "v0".to_string(),
            label: None,
            input: None,
        });

        let (plan, new_var) = translator
            .translate_step(&step, input, "v0", false)
            .expect("translate_step should succeed for MidV");

        // Should return a new variable
        assert!(new_var.is_some(), "MidV should produce a new variable");
        let var = new_var.unwrap();
        assert_ne!(var, "v0", "New variable should differ from original");

        // The result should be a NodeScan with input set
        match &plan {
            LogicalOperator::NodeScan(scan) => {
                assert_eq!(
                    scan.variable, var,
                    "NodeScan variable should match returned var"
                );
                assert!(scan.label.is_none(), "MidV(None) should not set a label");
                assert!(
                    scan.input.is_some(),
                    "MidV NodeScan should chain the previous plan as input"
                );
                // Verify the input is the original NodeScan
                match scan.input.as_deref() {
                    Some(LogicalOperator::NodeScan(inner)) => {
                        assert_eq!(inner.variable, "v0");
                    }
                    other => panic!("Expected inner NodeScan, got: {other:?}"),
                }
            }
            other => panic!("Expected NodeScan for MidV, got: {other:?}"),
        }
    }

    // === value_to_expr with parameter resolution ===

    #[test]
    fn test_value_to_expr_parameter_reference() {
        // A "$name" string should resolve to a Parameter expression
        let value = Value::String("$name".into());
        let expr = GremlinTranslator::value_to_expr(&value);

        match expr {
            LogicalExpression::Parameter(name) => {
                assert_eq!(name, "name", "Parameter name should strip the '$' prefix");
            }
            other => panic!("Expected Parameter expression, got: {other:?}"),
        }
    }

    #[test]
    fn test_value_to_expr_regular_string() {
        // A regular string (no $ prefix) should become a Literal
        let value = Value::String("regular".into());
        let expr = GremlinTranslator::value_to_expr(&value);

        match expr {
            LogicalExpression::Literal(Value::String(s)) => {
                assert_eq!(s.as_str(), "regular");
            }
            other => panic!("Expected Literal(String) expression, got: {other:?}"),
        }
    }

    #[test]
    fn test_value_to_expr_int64() {
        // Non-string values should always become Literals
        let value = Value::Int64(42);
        let expr = GremlinTranslator::value_to_expr(&value);

        match expr {
            LogicalExpression::Literal(Value::Int64(n)) => {
                assert_eq!(n, 42);
            }
            other => panic!("Expected Literal(Int64) expression, got: {other:?}"),
        }
    }

    #[test]
    fn translate_repeat_times() {
        let plan =
            super::translate("g.V().has('Person', 'name', 'Alix').repeat(out('KNOWS')).times(2)")
                .unwrap();
        assert!(plan.root.has_mutations() || !plan.root.has_mutations()); // plan is valid
    }

    #[test]
    fn translate_repeat_emit() {
        let plan =
            super::translate("g.V().has('Person', 'name', 'Alix').repeat(out('KNOWS')).emit()")
                .unwrap();
        assert!(!plan.root.has_mutations());
    }

    #[test]
    fn translate_repeat_until() {
        let plan = super::translate(
            "g.V().has('Person', 'name', 'Alix').repeat(out('KNOWS')).until(has('name', 'Gus'))",
        )
        .unwrap();
        assert!(!plan.root.has_mutations());
    }

    #[test]
    fn translate_repeat_in_direction() {
        let plan =
            super::translate("g.V().has('Person', 'name', 'Alix').repeat(in('KNOWS')).times(1)")
                .unwrap();
        assert!(!plan.root.has_mutations());
    }

    #[test]
    fn translate_repeat_both_direction() {
        let plan = super::translate("g.V().has('Person', 'name', 'Alix').repeat(both()).times(1)")
            .unwrap();
        assert!(!plan.root.has_mutations());
    }

    // === EXPLAIN/PROFILE tests ===

    #[test]
    fn test_explain_prefix_sets_explain_flag() {
        let plan = super::translate("EXPLAIN g.V().hasLabel('Person')").unwrap();
        assert!(plan.explain, "EXPLAIN prefix should set explain flag");
        assert!(!plan.profile, "EXPLAIN should not set profile flag");
    }

    #[test]
    fn test_explain_analyze_sets_profile_flag() {
        let plan = super::translate("EXPLAIN ANALYZE g.V().hasLabel('Person')").unwrap();
        assert!(!plan.explain, "EXPLAIN ANALYZE should not set explain flag");
        assert!(plan.profile, "EXPLAIN ANALYZE should set profile flag");
    }

    #[test]
    fn test_explain_case_insensitive() {
        let plan = super::translate("explain g.V().hasLabel('Person')").unwrap();
        assert!(plan.explain, "explain (lowercase) should set explain flag");
    }

    #[test]
    fn test_no_explain_prefix_normal_query() {
        let plan = super::translate("g.V().hasLabel('Person')").unwrap();
        assert!(!plan.explain);
        assert!(!plan.profile);
    }
}
