//! Path, node, edge, and mutation pattern translation.

#[allow(clippy::wildcard_imports)]
use super::*;

impl GqlTranslator {
    /// Translates a MERGE clause into a MergeOp.
    pub(super) fn translate_merge(
        &self,
        merge_clause: &ast::MergeClause,
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        let (variable, labels, match_properties) = match &merge_clause.pattern {
            ast::Pattern::Node(node) => {
                let var = node
                    .variable
                    .clone()
                    .unwrap_or_else(|| format!("_anon_{}", rand_id()));
                let labels = node.labels.clone();
                let props: Vec<(String, LogicalExpression)> = node
                    .properties
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                    .collect::<Result<_>>()?;
                (var, labels, props)
            }
            ast::Pattern::Path(path) if !path.edges.is_empty() => {
                return self.translate_merge_relationship(path, merge_clause, input);
            }
            ast::Pattern::Path(path) => {
                // Path with no edges is just a node pattern
                let var = path
                    .source
                    .variable
                    .clone()
                    .unwrap_or_else(|| format!("_anon_{}", rand_id()));
                let labels = path.source.labels.clone();
                let props: Vec<(String, LogicalExpression)> = path
                    .source
                    .properties
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                    .collect::<Result<_>>()?;
                (var, labels, props)
            }
            ast::Pattern::Quantified { .. }
            | ast::Pattern::Union(_)
            | ast::Pattern::MultisetUnion(_) => {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    "MERGE does not support quantified or union patterns",
                )));
            }
        };

        let on_create: Vec<(String, LogicalExpression)> = merge_clause
            .on_create
            .as_ref()
            .map(|assignments| {
                assignments
                    .iter()
                    .map(|a| Ok((a.property.clone(), self.translate_expression(&a.value)?)))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();

        let on_match: Vec<(String, LogicalExpression)> = merge_clause
            .on_match
            .as_ref()
            .map(|assignments| {
                assignments
                    .iter()
                    .map(|a| Ok((a.property.clone(), self.translate_expression(&a.value)?)))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();

        Ok(LogicalOperator::Merge(MergeOp {
            variable,
            labels,
            match_properties,
            on_create,
            on_match,
            input: Box::new(input),
        }))
    }

    /// Translates a MERGE with a relationship pattern.
    pub(super) fn translate_merge_relationship(
        &self,
        path: &ast::PathPattern,
        merge_clause: &ast::MergeClause,
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        let mut current_input = input;

        let source_variable = path.source.variable.clone().ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "MERGE relationship pattern requires a source node variable",
            ))
        })?;

        // If source node has labels or properties, emit a MergeOp for it
        if !path.source.labels.is_empty() || !path.source.properties.is_empty() {
            let node_props: Vec<(String, LogicalExpression)> = path
                .source
                .properties
                .iter()
                .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                .collect::<Result<Vec<_>>>()?;

            current_input = LogicalOperator::Merge(MergeOp {
                variable: source_variable.clone(),
                labels: path.source.labels.clone(),
                match_properties: node_props,
                on_create: Vec::new(),
                on_match: Vec::new(),
                input: Box::new(current_input),
            });
        }

        let edge = path.edges.first().ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "MERGE relationship pattern is empty",
            ))
        })?;

        let variable = edge
            .variable
            .clone()
            .unwrap_or_else(|| format!("_merge_rel_{}", rand_id()));

        let edge_type = edge.types.first().cloned().ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "MERGE relationship pattern requires an edge type",
            ))
        })?;

        let target_variable = edge.target.variable.clone().ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "MERGE relationship pattern requires a target node variable",
            ))
        })?;

        // If target node has labels or properties, emit a MergeOp for it
        if !edge.target.labels.is_empty() || !edge.target.properties.is_empty() {
            let node_props: Vec<(String, LogicalExpression)> = edge
                .target
                .properties
                .iter()
                .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                .collect::<Result<Vec<_>>>()?;

            current_input = LogicalOperator::Merge(MergeOp {
                variable: target_variable.clone(),
                labels: edge.target.labels.clone(),
                match_properties: node_props,
                on_create: Vec::new(),
                on_match: Vec::new(),
                input: Box::new(current_input),
            });
        }

        let match_properties: Vec<(String, LogicalExpression)> = edge
            .properties
            .iter()
            .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
            .collect::<Result<Vec<_>>>()?;

        let on_create: Vec<(String, LogicalExpression)> = merge_clause
            .on_create
            .as_ref()
            .map(|assignments| {
                assignments
                    .iter()
                    .map(|a| Ok((a.property.clone(), self.translate_expression(&a.value)?)))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();

        let on_match: Vec<(String, LogicalExpression)> = merge_clause
            .on_match
            .as_ref()
            .map(|assignments| {
                assignments
                    .iter()
                    .map(|a| Ok((a.property.clone(), self.translate_expression(&a.value)?)))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();

        Ok(LogicalOperator::MergeRelationship(MergeRelationshipOp {
            variable,
            source_variable,
            target_variable,
            edge_type,
            match_properties,
            on_create,
            on_match,
            input: Box::new(current_input),
        }))
    }

    /// Translates the patterns of an INSERT (or CREATE) that follows other
    /// clauses, so `plan` produces the input rows.
    pub(super) fn translate_create_patterns(
        &self,
        patterns: &[ast::Pattern],
        mut plan: LogicalOperator,
    ) -> Result<LogicalOperator> {
        let mut bound = insert_input_variables(&plan);
        for pattern in patterns {
            match pattern {
                ast::Pattern::Node(node) => {
                    let (variable, is_new) = insert_endpoint(node, &mut bound, true)?;
                    if is_new {
                        plan = LogicalOperator::CreateNode(CreateNodeOp {
                            variable,
                            labels: node.labels.clone(),
                            properties: self.insert_properties(&node.properties)?,
                            input: Some(Box::new(plan)),
                        });
                    }
                }
                ast::Pattern::Path(path) => {
                    // A bare variable of a path refers to a node of the input
                    // rows, even one `bound` does not list.
                    let (mut source_var, is_new) =
                        insert_endpoint(&path.source, &mut bound, false)?;
                    if is_new {
                        plan = LogicalOperator::CreateNode(CreateNodeOp {
                            variable: source_var.clone(),
                            labels: path.source.labels.clone(),
                            properties: self.insert_properties(&path.source.properties)?,
                            input: Some(Box::new(plan)),
                        });
                    }

                    for edge in &path.edges {
                        let (target_var, is_new) =
                            insert_endpoint(&edge.target, &mut bound, false)?;
                        if is_new {
                            plan = LogicalOperator::CreateNode(CreateNodeOp {
                                variable: target_var.clone(),
                                labels: edge.target.labels.clone(),
                                properties: self.insert_properties(&edge.target.properties)?,
                                input: Some(Box::new(plan)),
                            });
                        }

                        let (from_variable, to_variable) = match edge.direction {
                            ast::EdgeDirection::Incoming => (target_var.clone(), source_var),
                            ast::EdgeDirection::Outgoing | ast::EdgeDirection::Undirected => {
                                (source_var, target_var.clone())
                            }
                        };
                        plan = LogicalOperator::CreateEdge(CreateEdgeOp {
                            variable: edge.variable.clone(),
                            from_variable,
                            to_variable,
                            edge_type: edge.types.first().cloned().unwrap_or_default(),
                            properties: self.insert_properties(&edge.properties)?,
                            input: Box::new(plan),
                        });
                        // The next edge of the path starts at this target.
                        source_var = target_var;
                    }
                }
                ast::Pattern::Quantified { .. }
                | ast::Pattern::Union(_)
                | ast::Pattern::MultisetUnion(_) => {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "CREATE does not support quantified or union patterns",
                    )));
                }
            }
        }
        Ok(plan)
    }

    pub(super) fn translate_node_pattern(
        &self,
        node: &ast::NodePattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let variable = node
            .variable
            .clone()
            .unwrap_or_else(|| format!("_anon_{}", rand_id()));

        // Use first label from colon syntax for the NodeScan optimization,
        // or first label from IS expression if it's a simple label.
        let label = if let Some(ref label_expr) = node.label_expression {
            if let ast::LabelExpression::Label(name) = label_expr {
                Some(name.clone())
            } else {
                None
            }
        } else {
            node.labels.first().cloned()
        };

        let mut plan = LogicalOperator::NodeScan(NodeScanOp {
            variable: variable.clone(),
            label,
            input: input.map(Box::new),
        });

        // Add hasLabel filters for additional colon-syntax labels (AND semantics).
        // First label is used in NodeScan for scan-time filtering; remaining
        // labels are checked via post-scan Filter.
        if node.labels.len() > 1 {
            plan = Self::add_extra_label_filters(plan, &variable, &node.labels[1..]);
        }

        // Add label expression filter for complex IS expressions
        if let Some(ref label_expr) = node.label_expression {
            // Only add filter for non-simple expressions (simple Label already used in NodeScan)
            if !matches!(label_expr, ast::LabelExpression::Label(_)) {
                let predicate = Self::translate_label_expression(&variable, label_expr);
                plan = wrap_filter(plan, predicate);
            }
        }

        // Add filter for node pattern properties (e.g., {name: 'Alix'})
        if !node.properties.is_empty() {
            let predicate = self.build_property_predicate(&variable, &node.properties)?;
            plan = wrap_filter(plan, predicate);
        }

        // Add element pattern WHERE clause (e.g., (n WHERE n.age > 30))
        if let Some(ref where_expr) = node.where_clause {
            let predicate = self.translate_expression(where_expr)?;
            plan = wrap_filter(plan, predicate);
        }

        Ok(plan)
    }

    /// Translates a label expression into a filter predicate using hasLabel() calls.
    pub(super) fn translate_label_expression(
        variable: &str,
        expr: &ast::LabelExpression,
    ) -> LogicalExpression {
        match expr {
            ast::LabelExpression::Label(name) => LogicalExpression::FunctionCall {
                name: "hasLabel".into(),
                args: vec![
                    LogicalExpression::Variable(variable.to_string()),
                    LogicalExpression::Literal(Value::from(name.as_str())),
                ],
                distinct: false,
            },
            ast::LabelExpression::Conjunction(operands) => {
                let mut iter = operands.iter();
                let first = Self::translate_label_expression(
                    variable,
                    iter.next().expect("conjunction has at least one operand"),
                );
                iter.fold(first, |acc, op| LogicalExpression::Binary {
                    left: Box::new(acc),
                    op: BinaryOp::And,
                    right: Box::new(Self::translate_label_expression(variable, op)),
                })
            }
            ast::LabelExpression::Disjunction(operands) => {
                let mut iter = operands.iter();
                let first = Self::translate_label_expression(
                    variable,
                    iter.next().expect("disjunction has at least one operand"),
                );
                iter.fold(first, |acc, op| LogicalExpression::Binary {
                    left: Box::new(acc),
                    op: BinaryOp::Or,
                    right: Box::new(Self::translate_label_expression(variable, op)),
                })
            }
            ast::LabelExpression::Negation(inner) => LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand: Box::new(Self::translate_label_expression(variable, inner)),
            },
            ast::LabelExpression::Wildcard => LogicalExpression::Literal(Value::Bool(true)),
        }
    }

    /// Wraps a plan with AND-combined `hasLabel` filters for extra labels beyond the
    /// first (which is already used in `NodeScan` for scan-time filtering).
    fn add_extra_label_filters(
        plan: LogicalOperator,
        variable: &str,
        extra_labels: &[String],
    ) -> LogicalOperator {
        match has_all_labels(variable, extra_labels) {
            Some(predicate) => wrap_filter(plan, predicate),
            None => plan,
        }
    }

    /// Builds a predicate expression for property filters like {name: 'Alix', age: 30}.
    ///
    /// When a pattern property is null (e.g., `{key: null}`), we emit `IS NULL`
    /// instead of equality, because `null = null` is `null` (falsy) in
    /// three-valued logic but a null property pattern should match absent or
    /// null-valued properties.
    pub(super) fn build_property_predicate(
        &self,
        variable: &str,
        properties: &[(String, ast::Expression)],
    ) -> Result<LogicalExpression> {
        let predicates = properties
            .iter()
            .map(|(prop_name, prop_value)| {
                let left = LogicalExpression::Property {
                    variable: variable.to_string(),
                    property: prop_name.clone(),
                };
                let right = self.translate_expression(prop_value)?;
                if matches!(right, LogicalExpression::Literal(Value::Null)) {
                    Ok(LogicalExpression::Unary {
                        op: UnaryOp::IsNull,
                        operand: Box::new(left),
                    })
                } else {
                    Ok(LogicalExpression::Binary {
                        left: Box::new(left),
                        op: BinaryOp::Eq,
                        right: Box::new(right),
                    })
                }
            })
            .collect::<Result<Vec<_>>>()?;

        combine_with_and(predicates)
    }

    pub(super) fn translate_path_pattern_with_alias(
        &self,
        path: &ast::PathPattern,
        input: Option<LogicalOperator>,
        path_alias: Option<&str>,
        path_mode: PathMode,
    ) -> Result<LogicalOperator> {
        // Start with the source node
        let source_var = path
            .source
            .variable
            .clone()
            .unwrap_or_else(|| format!("_anon_{}", rand_id()));

        let source_label = path.source.labels.first().cloned();

        let mut plan = LogicalOperator::NodeScan(NodeScanOp {
            variable: source_var.clone(),
            label: source_label,
            input: input.map(Box::new),
        });

        // Add hasLabel filters for additional source labels (AND semantics)
        if path.source.labels.len() > 1 {
            plan = Self::add_extra_label_filters(plan, &source_var, &path.source.labels[1..]);
        }

        // Add filter for source node properties (e.g., {id: 'a'})
        if !path.source.properties.is_empty() {
            let predicate = self.build_property_predicate(&source_var, &path.source.properties)?;
            plan = wrap_filter(plan, predicate);
        }

        // Add element WHERE clause for source node
        if let Some(ref where_expr) = path.source.where_clause {
            let predicate = self.translate_expression(where_expr)?;
            plan = wrap_filter(plan, predicate);
        }

        // Process each edge in the chain
        let mut current_source = source_var;
        let edge_count = path.edges.len();

        for (idx, edge) in path.edges.iter().enumerate() {
            let target_var = edge
                .target
                .variable
                .clone()
                .unwrap_or_else(|| format!("_anon_{}", rand_id()));

            // An edge with a property map needs a variable to filter on, even
            // when the pattern leaves it anonymous: `-[:T {w: 1}]->`.
            let edge_var = edge
                .variable
                .clone()
                .or_else(|| (!edge.properties.is_empty()).then(|| format!("_anon_{}", rand_id())));
            let edge_types = edge.types.clone();

            let direction = match edge.direction {
                ast::EdgeDirection::Outgoing => ExpandDirection::Outgoing,
                ast::EdgeDirection::Incoming => ExpandDirection::Incoming,
                ast::EdgeDirection::Undirected => ExpandDirection::Both,
            };

            let edge_var_for_filter = edge_var.clone();

            // Set path_alias on the last edge of a named path
            let expand_path_alias = if idx == edge_count - 1 {
                path_alias.map(String::from)
            } else {
                None
            };

            let (min_hops, max_hops) = edge_hop_bounds(edge);

            // A quantifier (`{1,1}` too) makes the edge variable a group
            // variable: the list of the path's edges. An edge without one is a
            // single hop, so the quantified edges are the variable-length ones.
            let quantified = edge.min_hops.is_some() || edge.max_hops.is_some();
            let is_variable_length = quantified;

            // For variable-length edges with a named edge variable, auto-generate
            // a path alias if none exists, so path detail columns are available
            // for horizontal aggregation (GE09).
            let expand_path_alias = if is_variable_length
                && edge_var_for_filter.is_some()
                && expand_path_alias.is_none()
            {
                Some(format!("_auto_path_{}", rand_id()))
            } else {
                expand_path_alias
            };

            // Track group-list variables for horizontal aggregation detection
            if is_variable_length
                && let (Some(ev), Some(pa)) = (&edge_var_for_filter, &expand_path_alias)
            {
                self.group_list_variables
                    .borrow_mut()
                    .insert(ev.clone(), pa.clone());
            }

            // For questioned edges (->?), save the plan BEFORE the expand so the
            // left side of the LeftJoin contains only the required patterns.
            let pre_expand_plan = if edge.questioned {
                Some(plan.clone())
            } else {
                None
            };

            // Detect cycle pattern: (s)-[*]->(s) where source == target variable.
            // The expand must use a temporary target, then filter for equality.
            let is_cycle = target_var == current_source;
            let expand_target = if is_cycle {
                format!("_cycle_{}", rand_id())
            } else {
                target_var.clone()
            };

            let property_path = expand_path_alias.clone();
            plan = LogicalOperator::Expand(ExpandOp {
                quantified,
                from_variable: current_source,
                to_variable: expand_target.clone(),
                edge_variable: edge_var,
                direction,
                edge_types,
                min_hops,
                max_hops,
                input: Box::new(plan),
                path_alias: expand_path_alias,
                path_mode,
            });

            // For cycle patterns, enforce that expanded target == original source
            if is_cycle {
                plan = wrap_filter(
                    plan,
                    LogicalExpression::Binary {
                        left: Box::new(LogicalExpression::FunctionCall {
                            name: "id".into(),
                            args: vec![LogicalExpression::Variable(expand_target.clone())],
                            distinct: false,
                        }),
                        op: BinaryOp::Eq,
                        right: Box::new(LogicalExpression::FunctionCall {
                            name: "id".into(),
                            args: vec![LogicalExpression::Variable(target_var.clone())],
                            distinct: false,
                        }),
                    },
                );
            }

            // Add filter for edge properties
            if !edge.properties.is_empty()
                && let Some(ref ev) = edge_var_for_filter
            {
                let predicate = match property_path.filter(|_| is_variable_length) {
                    // `all(e IN edges(path) WHERE e.k = v ...)`: the edge column
                    // of a variable-length expand only holds the last hop.
                    Some(path) => {
                        let hop = format!("_anon_{}", rand_id());
                        crate::query::translators::common::every_edge_matches(
                            path,
                            hop.clone(),
                            self.build_property_predicate(&hop, &edge.properties)?,
                        )
                    }
                    None => self.build_property_predicate(ev, &edge.properties)?,
                };
                plan = wrap_filter(plan, predicate);
            }

            // Add element WHERE clause for edge
            if let Some(ref where_expr) = edge.where_clause {
                let predicate = self.translate_expression(where_expr)?;
                plan = wrap_filter(plan, predicate);
            }

            // Add filter for target node properties
            if !edge.target.properties.is_empty() {
                let predicate =
                    self.build_property_predicate(&target_var, &edge.target.properties)?;
                plan = wrap_filter(plan, predicate);
            }

            // Add filter for target node labels (colon syntax or IS expression)
            if let Some(ref label_expr) = edge.target.label_expression {
                let predicate = Self::translate_label_expression(&target_var, label_expr);
                plan = wrap_filter(plan, predicate);
            } else if !edge.target.labels.is_empty() {
                // Filter for ALL target labels (AND semantics per ISO GQL)
                for target_label in &edge.target.labels {
                    plan = wrap_filter(
                        plan,
                        LogicalExpression::FunctionCall {
                            name: "hasLabel".into(),
                            args: vec![
                                LogicalExpression::Variable(target_var.clone()),
                                LogicalExpression::Literal(Value::from(target_label.clone())),
                            ],
                            distinct: false,
                        },
                    );
                }
            }

            // Add element WHERE clause for target node
            if let Some(ref where_expr) = edge.target.where_clause {
                let predicate = self.translate_expression(where_expr)?;
                plan = wrap_filter(plan, predicate);
            }

            // Questioned edge: wrap expand + all filters in a LeftJoin so the
            // edge is optional (rows without a matching edge are preserved with nulls).
            if let Some(left) = pre_expand_plan {
                // Extract only the expand + filter portion (the right side):
                // plan currently includes the expand on top of the left plan.
                // We need to rebuild: LeftJoin(left, expand_on_left).
                // Since expand already includes left as input, we use it as-is
                // and the LeftJoin semantics handle preserving unmatched rows.
                plan = LogicalOperator::LeftJoin(LeftJoinOp {
                    left: Box::new(left),
                    right: Box::new(plan),
                    condition: None,
                });
            }

            current_source = target_var;
        }

        Ok(plan)
    }

    /// Translates the property map of an inserted node or edge.
    pub(super) fn insert_properties(
        &self,
        properties: &[(String, ast::Expression)],
    ) -> Result<Vec<(String, LogicalExpression)>> {
        properties
            .iter()
            .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
            .collect()
    }
}

/// How an INSERT treats a node pattern: returns its variable and whether it is
/// a new node.
///
/// A variable in `bound` (bound by the input rows or created earlier in this
/// INSERT) refers to its node, and INSERT cannot add labels or properties to
/// it: they used to be dropped, or a second node was created under the same
/// name. Anonymous, labeled and property-carrying endpoints are new nodes
/// (only labeled ones used to be, so `({id: 1})` failed as undefined), and so
/// is a bare variable when `bare_is_new`.
pub(super) fn insert_endpoint(
    node: &ast::NodePattern,
    bound: &mut HashSet<String>,
    bare_is_new: bool,
) -> Result<(String, bool)> {
    let Some(variable) = &node.variable else {
        return Ok((format!("_anon_{}", rand_id()), true));
    };
    let declares = !node.labels.is_empty() || !node.properties.is_empty();
    if bound.contains(variable) {
        if declares {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!(
                    "INSERT cannot add labels or properties to the bound variable '{variable}': \
                     use SET"
                ),
            )));
        }
        return Ok((variable.clone(), false));
    }
    let is_new = declares || bare_is_new;
    if is_new {
        bound.insert(variable.clone());
    }
    Ok((variable.clone(), is_new))
}

/// The variables the rows of `plan` bind: pattern variables, and the nodes
/// and edges created by an earlier INSERT clause of the same query.
fn insert_input_variables(plan: &LogicalOperator) -> HashSet<String> {
    fn collect(op: &LogicalOperator, vars: &mut HashSet<String>) {
        match op {
            LogicalOperator::CreateNode(create) => {
                vars.insert(create.variable.clone());
                if let Some(input) = &create.input {
                    collect(input, vars);
                }
            }
            LogicalOperator::CreateEdge(create) => {
                if let Some(variable) = &create.variable {
                    vars.insert(variable.clone());
                }
                collect(&create.input, vars);
            }
            LogicalOperator::Merge(merge) => {
                vars.insert(merge.variable.clone());
                collect(&merge.input, vars);
            }
            LogicalOperator::MergeRelationship(merge) => {
                vars.insert(merge.variable.clone());
                collect(&merge.input, vars);
            }
            other => crate::query::translators::common::collect_operator_variables(other, vars),
        }
    }
    let mut vars = HashSet::new();
    collect(plan, &mut vars);
    vars
}

/// The minimum and maximum number of hops (`None` = unbounded) an edge pattern
/// matches: an edge without a quantifier is exactly one hop.
pub(super) fn edge_hop_bounds(edge: &ast::EdgePattern) -> (u32, Option<u32>) {
    if edge.min_hops.is_none() && edge.max_hops.is_none() {
        (1, Some(1))
    } else {
        (edge.min_hops.unwrap_or(1), edge.max_hops)
    }
}
