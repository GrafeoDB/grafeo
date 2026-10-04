//! Relationship expansion and factorized chain planning.

use super::{
    Arc, Direction, Error, ExecutionPathMode, ExpandDirection, ExpandOp, ExpandOperator,
    ExpandStep, GraphStoreSearch, LazyFactorizedChainOperator, LogicalOperator, Operator, PathMode,
    Result, VariableLengthExpandOperator,
};

impl super::Planner {
    /// Plans an expand operator.
    pub(super) fn plan_expand(
        &self,
        expand: &ExpandOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Plan the input operator first
        let (input_op, input_columns) = self.plan_operator(&expand.input)?;

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

        // Check if this is a variable-length path. The GQL, Cypher and SQL/PGQ
        // translators set `quantified` for one; other plans (Gremlin, built
        // in code) only set hop bounds, so those count too.
        let is_variable_length =
            expand.quantified || expand.min_hops != 1 || expand.max_hops != Some(1);

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
                expand.max_hops.unwrap_or(expand.min_hops + 100)
            } else {
                1
            };
            let exec_path_mode = match expand.path_mode {
                PathMode::Walk => ExecutionPathMode::Walk,
                PathMode::Trail => ExecutionPathMode::Trail,
                PathMode::Simple => ExecutionPathMode::Simple,
                PathMode::Acyclic => ExecutionPathMode::Acyclic,
            };

            let mut expand_op = VariableLengthExpandOperator::new(
                Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
                input_op,
                source_column,
                direction,
                expand.edge_types.clone(),
                min_hops,
                max_hops,
            )
            .with_path_mode(exec_path_mode)
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_read_only(self.read_only);

            // If a path alias is set, enable path length and detail output
            if needs_path_details {
                expand_op = expand_op
                    .with_path_length_output()
                    .with_path_detail_output();
            }
            if binds_edge_list {
                expand_op = expand_op.with_edge_list_output();
            }

            Box::new(expand_op)
        } else {
            // Use simple ExpandOperator for single-hop paths without named paths
            let expand_op = ExpandOperator::new(
                Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
                input_op,
                source_column,
                direction,
                expand.edge_types.clone(),
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_read_only(self.read_only);
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
            // The path alias itself is also a scalar column containing Value::Path
            self.scalar_columns.borrow_mut().insert(path_alias.clone());
            columns.push(length_col);
            columns.push(nodes_col);
            columns.push(edges_col);
            columns.push(path_alias.clone());
        }

        Ok((operator, columns))
    }

    /// Plans a chain of consecutive expand operations using factorized execution.
    ///
    /// This avoids the Cartesian product explosion that occurs with separate expands.
    /// For a 2-hop query with degree d, this uses O(d) memory instead of O(d^2).
    ///
    /// The chain is executed lazily at query time, not during planning. This ensures
    /// that any filters applied above the expand chain are properly respected.
    pub(super) fn plan_expand_chain(
        &self,
        op: &LogicalOperator,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let expands = Self::collect_expand_chain(op);
        if expands.is_empty() {
            return Err(Error::Internal("Empty expand chain".to_string()));
        }

        // Get the base operator (before first expand)
        let first_expand = expands[0];
        let (base_op, base_columns) = self.plan_operator(&first_expand.input)?;

        let mut columns = base_columns.clone();
        let mut steps = Vec::new();

        // Track the level-local source column for each expand
        // For the first expand, it's the column in the input (base_columns)
        // For subsequent expands, the target from the previous level is always at index 1
        // (each level adds [edge, target], so target is at index 1)
        let mut is_first = true;

        for expand in &expands {
            // Find source column for this expand
            let source_column = if is_first {
                // For first expand, find in base columns
                base_columns
                    .iter()
                    .position(|c| c == &expand.from_variable)
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "Source variable '{}' not found in base columns",
                            expand.from_variable
                        ))
                    })?
            } else {
                // For subsequent expands, the target from the previous level is at index 1
                // (each level adds [edge, target], so target is the second column)
                1
            };

            // Convert direction
            let direction = match expand.direction {
                ExpandDirection::Outgoing => Direction::Outgoing,
                ExpandDirection::Incoming => Direction::Incoming,
                ExpandDirection::Both => Direction::Both,
            };

            // Add expand step configuration
            steps.push(ExpandStep {
                source_column,
                direction,
                edge_types: expand.edge_types.clone(),
            });

            // Add edge and target columns
            let edge_col_name = self.register_edge_column(&expand.edge_variable);
            columns.push(edge_col_name);
            columns.push(expand.to_variable.clone());

            is_first = false;
        }

        // Create lazy operator that executes at query time, not planning time
        let mut lazy_op = LazyFactorizedChainOperator::new(
            Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
            base_op,
            steps,
        )
        .with_read_only(self.read_only);

        if let Some(transaction_id) = self.transaction_id {
            lazy_op = lazy_op.with_transaction_context(self.viewing_epoch, Some(transaction_id));
        } else {
            lazy_op = lazy_op.with_transaction_context(self.viewing_epoch, None);
        }

        Ok((Box::new(lazy_op), columns))
    }
}
