//! Join, union, and distinct planning.

use super::{
    ApplyOp, ApplyOperator, DistinctOp, Error, ExceptOp, HashJoinOperator, IntersectOp, JoinOp,
    JoinType, LeapfrogJoinOperator, LogicalExpression, MultiWayJoinOp, Operator, OtherwiseOp,
    PhysicalJoinType, ProjectExpr, ProjectOperator, Result, UnionOp, Value, common,
};

impl super::Planner {
    /// Plans a JOIN operator.
    ///
    /// When join conditions reference shared variables, deduplicates the output
    /// columns by projecting out the right-side copies (whose values are equal
    /// to the left-side copies due to the join condition).
    pub(super) fn plan_join(&self, join: &JoinOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (left_op, left_columns) = self.plan_operator(&join.left)?;
        let (right_op, right_columns) = self.plan_operator(&join.right)?;

        // Full column list before deduplication (HashJoin produces all columns)
        let mut all_columns = left_columns.clone();
        all_columns.extend(right_columns.clone());

        // Convert join type
        let physical_join_type = match join.join_type {
            JoinType::Inner => PhysicalJoinType::Inner,
            JoinType::Left => PhysicalJoinType::Left,
            JoinType::Right => PhysicalJoinType::Right,
            JoinType::Full => PhysicalJoinType::Full,
            JoinType::Cross => PhysicalJoinType::Cross,
            JoinType::Semi => PhysicalJoinType::Semi,
            JoinType::Anti => PhysicalJoinType::Anti,
        };

        // Build key columns from join conditions
        let (probe_keys, build_keys): (Vec<usize>, Vec<usize>) = if join.conditions.is_empty() {
            // Cross join - no keys
            (vec![], vec![])
        } else {
            join.conditions
                .iter()
                .filter_map(|cond| {
                    // Try to extract column indices from expressions
                    let left_idx = self.expression_to_column(&cond.left, &left_columns).ok()?;
                    let right_idx = self
                        .expression_to_column(&cond.right, &right_columns)
                        .ok()?;
                    Some((left_idx, right_idx))
                })
                .unzip()
        };

        let output_schema = self.derive_schema_from_columns(&all_columns);

        let join_op: Box<dyn Operator> = Box::new(HashJoinOperator::new(
            left_op,
            right_op,
            probe_keys,
            build_keys,
            physical_join_type,
            output_schema,
        ));

        // Deduplicate shared variable columns: right-side columns that also
        // appear on the left are redundant (the join guarantees equality).
        // Skip deduplication for CROSS joins (no equality guarantee).
        // For RIGHT/FULL joins, use COALESCE to prefer the non-NULL side.
        if matches!(join.join_type, JoinType::Cross) || join.conditions.is_empty() {
            return Ok((join_op, all_columns));
        }
        let needs_coalesce = matches!(join.join_type, JoinType::Right | JoinType::Full);

        let left_count = left_columns.len();
        let mut proj_exprs: Vec<ProjectExpr> = Vec::new();
        let mut deduped_columns: Vec<String> = Vec::new();
        let mut has_duplicates = false;

        for (li, col_name) in left_columns.iter().enumerate() {
            if needs_coalesce {
                if let Some(ri) = right_columns.iter().position(|c| c == col_name) {
                    proj_exprs.push(ProjectExpr::Coalesce {
                        first: li,
                        second: left_count + ri,
                    });
                    has_duplicates = true;
                } else {
                    proj_exprs.push(ProjectExpr::Column(li));
                }
            } else {
                proj_exprs.push(ProjectExpr::Column(li));
            }
            deduped_columns.push(col_name.clone());
        }

        for (ri, col_name) in right_columns.iter().enumerate() {
            if !deduped_columns.contains(col_name) {
                proj_exprs.push(ProjectExpr::Column(left_count + ri));
                deduped_columns.push(col_name.clone());
            } else {
                has_duplicates = true;
            }
        }

        if !has_duplicates {
            return Ok((join_op, deduped_columns));
        }

        let proj_schema = self.derive_schema_from_columns(&deduped_columns);
        let operator: Box<dyn Operator> =
            Box::new(ProjectOperator::new(join_op, proj_exprs, proj_schema));

        Ok((operator, deduped_columns))
    }

    /// Plans a multi-way leapfrog join (WCOJ) operator.
    ///
    /// Materializes each input into a sorted trie and uses `LeapfrogJoinOperator`
    /// for worst-case optimal intersection.
    pub(super) fn plan_multi_way_join(
        &self,
        mwj: &MultiWayJoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Plan each input, collecting operators and their column lists
        let mut input_ops: Vec<Box<dyn Operator>> = Vec::with_capacity(mwj.inputs.len());
        let mut input_columns: Vec<Vec<String>> = Vec::with_capacity(mwj.inputs.len());

        for input in &mwj.inputs {
            let (op, cols) = self.plan_operator(input)?;
            input_ops.push(op);
            input_columns.push(cols);
        }

        // For each input, find the column indices of the shared variables (join keys)
        let mut join_key_indices: Vec<Vec<usize>> = Vec::with_capacity(mwj.inputs.len());
        for cols in &input_columns {
            let mut key_indices = Vec::new();
            for shared_var in &mwj.shared_variables {
                if let Some(idx) = cols.iter().position(|c| c == shared_var) {
                    key_indices.push(idx);
                }
            }
            join_key_indices.push(key_indices);
        }

        // Build combined output columns: shared variables first (deduplicated),
        // then remaining columns from each input
        let mut output_columns: Vec<String> = mwj.shared_variables.clone();
        let mut output_column_mapping: Vec<(usize, usize)> = Vec::new();

        // Map shared variables from the first input that has them
        for shared_var in &mwj.shared_variables {
            let mut found = false;
            for (input_idx, cols) in input_columns.iter().enumerate() {
                if let Some(col_idx) = cols.iter().position(|c| c == shared_var) {
                    output_column_mapping.push((input_idx, col_idx));
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(Error::Internal(format!(
                    "Shared variable '{}' not found in any input",
                    shared_var
                )));
            }
        }

        // Add non-shared columns from each input
        for (input_idx, cols) in input_columns.iter().enumerate() {
            for (col_idx, col_name) in cols.iter().enumerate() {
                if !mwj.shared_variables.contains(col_name) {
                    output_columns.push(col_name.clone());
                    output_column_mapping.push((input_idx, col_idx));
                }
            }
        }

        let output_schema = self.derive_schema_from_columns(&output_columns);

        let operator: Box<dyn Operator> = Box::new(LeapfrogJoinOperator::new(
            input_ops,
            join_key_indices,
            output_schema,
            output_column_mapping,
        ));

        Ok((operator, output_columns))
    }

    /// Extracts a column index from an expression.
    pub(super) fn expression_to_column(
        &self,
        expr: &LogicalExpression,
        columns: &[String],
    ) -> Result<usize> {
        match expr {
            LogicalExpression::Variable(name) => columns
                .iter()
                .position(|c| c == name)
                .ok_or_else(|| Error::Internal(format!("Variable '{}' not found", name))),
            _ => Err(Error::Internal(
                "Only variables supported in join conditions".to_string(),
            )),
        }
    }

    /// Plans a UNION operator.
    ///
    /// Cypher / GQL UNION aligns branches by column POSITION (the output column
    /// names come from the first branch that reaches each position), so the
    /// output arity must be the MAXIMUM branch arity. The historical
    /// branch-0-only seed used the first branch's arity instead, which silently
    /// TRUNCATED a UNION whose later branch is WIDER: the extra trailing columns
    /// were dropped with no error, because Cypher projects inside each branch
    /// below the union, leaving no node above it to fail. Each narrower branch
    /// is now null-padded to the maximum width so the core `UnionOperator`
    /// forwards uniform-width chunks and no column is lost.
    ///
    /// This is the LPG-side counterpart to the RDF/SPARQL fix in the RDF
    /// planner, but the reconciliation differs: SPARQL UNION is over solution
    /// mappings keyed by variable NAME (so the RDF leg unions branch variable
    /// SETS), whereas Cypher/GQL UNION is POSITIONAL. The legs also fail
    /// differently -- the RDF leg crashes (X001/V001), this leg silently
    /// truncates -- so they are verified by different oracles. Cypher-level
    /// union-compatibility validation (requiring matching column names) is out
    /// of scope; this fix targets only the drop mechanism.
    pub(super) fn plan_union(&self, union: &UnionOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let mut planned = Vec::with_capacity(union.inputs.len());
        for input in &union.inputs {
            planned.push(self.plan_operator(input)?);
        }

        // Positional column domain: position i is named by the first branch
        // that reaches it; the output arity is the maximum branch arity.
        let mut unified_columns: Vec<String> = Vec::new();
        for (_, cols) in &planned {
            for (i, col) in cols.iter().enumerate() {
                if i == unified_columns.len() {
                    unified_columns.push(col.clone());
                }
            }
        }

        let unified_schema = self.derive_schema_from_columns(&unified_columns);
        let width = unified_columns.len();

        // Null-pad each narrower branch's trailing positions so the core
        // UnionOperator never forwards a short chunk. A full-width branch passes
        // through unwrapped (positional union keeps its values in place; the
        // output column names are taken from `unified_columns`).
        let inputs: Vec<Box<dyn Operator>> = planned
            .into_iter()
            .map(|(op, cols)| {
                if cols.len() == width {
                    return op;
                }
                let projections: Vec<ProjectExpr> = (0..width)
                    .map(|i| {
                        if i < cols.len() {
                            ProjectExpr::Column(i)
                        } else {
                            ProjectExpr::Constant(Value::Null)
                        }
                    })
                    .collect();
                Box::new(ProjectOperator::new(
                    op,
                    projections,
                    unified_schema.clone(),
                )) as Box<dyn Operator>
            })
            .collect();

        common::build_union(inputs, unified_columns, unified_schema)
    }

    /// Plans a DISTINCT operator.
    pub(super) fn plan_distinct(
        &self,
        distinct: &DistinctOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&distinct.input)?;
        let schema = self.derive_schema_from_columns(&columns);
        Ok(common::build_distinct(
            input_op,
            columns,
            distinct.columns.as_deref(),
            schema,
        ))
    }

    /// Plans an EXCEPT operator.
    pub(super) fn plan_except(
        &self,
        except: &ExceptOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (left_op, columns) = self.plan_operator(&except.left)?;
        let (right_op, _) = self.plan_operator(&except.right)?;
        let schema = self.derive_schema_from_columns(&columns);
        Ok(common::build_except(
            left_op, right_op, columns, except.all, schema,
        ))
    }

    /// Plans an INTERSECT operator.
    pub(super) fn plan_intersect(
        &self,
        intersect: &IntersectOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (left_op, columns) = self.plan_operator(&intersect.left)?;
        let (right_op, _) = self.plan_operator(&intersect.right)?;
        let schema = self.derive_schema_from_columns(&columns);
        Ok(common::build_intersect(
            left_op,
            right_op,
            columns,
            intersect.all,
            schema,
        ))
    }

    /// Plans an OTHERWISE operator.
    pub(super) fn plan_otherwise(
        &self,
        otherwise: &OtherwiseOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (left_op, columns) = self.plan_operator(&otherwise.left)?;
        let (right_op, _) = self.plan_operator(&otherwise.right)?;
        Ok(common::build_otherwise(left_op, right_op, columns))
    }

    /// Plans an APPLY (lateral join) operator.
    ///
    /// When `shared_variables` is non-empty, creates a correlated Apply that
    /// injects outer row values into the inner plan via [`ParameterState`].
    pub(super) fn plan_apply(&self, apply: &ApplyOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (outer_op, outer_columns) = self.plan_operator(&apply.input)?;

        if apply.shared_variables.is_empty() {
            // Uncorrelated Apply
            let (inner_op, inner_columns) = self.plan_operator(&apply.subplan)?;
            // Inner subquery RETURN materializes values (PropertyAccess, NodeResolve,
            // aggregates, etc.), so all its output columns are scalar.
            for col in &inner_columns {
                self.scalar_columns.borrow_mut().insert(col.clone());
            }
            let inner_col_count = inner_columns.len();
            let mut columns = outer_columns;
            columns.extend(inner_columns);
            let mut op = ApplyOperator::new(outer_op, inner_op);
            if apply.optional {
                op = op.with_optional(inner_col_count);
            }
            return Ok((Box::new(op), columns));
        }

        // Expand wildcard: WITH * imports all outer-scope variables
        let shared_vars = if apply.shared_variables.len() == 1 && apply.shared_variables[0] == "*" {
            outer_columns.clone()
        } else {
            apply.shared_variables.clone()
        };

        // Correlated Apply: create shared ParameterState
        let param_state = std::sync::Arc::new(
            grafeo_core::execution::operators::ParameterState::new(shared_vars.clone()),
        );

        // Find column indices for the shared variables in outer columns
        let param_col_indices: Vec<usize> = shared_vars
            .iter()
            .map(|var| outer_columns.iter().position(|c| c == var).unwrap_or(0))
            .collect();

        // Set the parameter state so the inner plan's ParameterScan can find it
        *self.correlated_param_state.borrow_mut() = Some(std::sync::Arc::clone(&param_state));

        let (inner_op, inner_columns) = self.plan_operator(&apply.subplan)?;

        // Clear the parameter state after planning the inner operator
        *self.correlated_param_state.borrow_mut() = None;

        // Inner subquery RETURN materializes values, so register as scalar
        // to prevent the outer RETURN from misinterpreting them as node IDs.
        for col in &inner_columns {
            self.scalar_columns.borrow_mut().insert(col.clone());
        }

        // Build correlated Apply
        let mut columns = outer_columns;
        let inner_col_count = inner_columns.len();
        columns.extend(inner_columns);
        let mut op =
            ApplyOperator::new_correlated(outer_op, inner_op, param_state, param_col_indices);
        if apply.optional {
            op = op.with_optional(inner_col_count);
        }
        Ok((Box::new(op), columns))
    }
}
