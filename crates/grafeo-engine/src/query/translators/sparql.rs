//! SPARQL to LogicalPlan translator.
//!
//! Translates SPARQL 1.1 AST to the common logical plan representation.

use super::common::{
    collect_expression_variables, wrap_distinct, wrap_filter, wrap_limit, wrap_skip, wrap_sort,
};
use crate::database::ResultKind;
use crate::query::plan::Callee;
use crate::query::plan::always_bound;
use crate::query::plan::{
    AddGraphOp, AggregateExpr, AggregateFunction, AggregateOp, AntiJoinOp, BinaryOp, BindOp,
    ClearGraphOp, ConstructOp, CopyGraphOp, CreateGraphOp, DatasetNamedGraphs, DatasetRestriction,
    DeleteTripleOp, DropGraphOp, FilterOp, GroupGraph, InNamedGraphOp, InsertTripleOp,
    JoinCondition, JoinOp, JoinType, LeftJoinOp, LoadGraphOp, LogicalExpression, LogicalOperator,
    LogicalPlan, ModifyOp, MoveGraphOp, NamedGraph, NamedGraphsOp, ProjectOp, Projection, SortKey,
    SortOrder, TripleComponent, TripleScanOp, TripleTemplate, UnaryOp, UnionOp, group_graph_name,
    is_hidden_variable,
};
use grafeo_adapters::query::sparql::{self, ast};
use grafeo_common::storage::log_record::RdfGraphTarget;
use grafeo_common::types::Value;
use grafeo_common::types::rdf::Term;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// Global counter for generating unique query IDs (blank node scoping).
static QUERY_ID_COUNTER: AtomicU32 = AtomicU32::new(0);

/// Translates a SPARQL query string to a logical plan.
///
/// # Errors
///
/// Returns an error if the query cannot be parsed or translated.
pub fn translate(query: &str) -> Result<LogicalPlan> {
    translate_with_bindings(query, &HashMap::new())
}

/// Translates a SPARQL query whose variables `bindings` binds before it is
/// evaluated: a parameter `x` binds `?x` (or `$x`, the same variable) in the
/// WHERE clause of the query, as if a one-row `VALUES` for them came first
/// (the initial bindings of Jena and RDF4J). A name may start with `?` or
/// `$`; a NULL binds nothing. SPARQL has no parameter syntax of its own.
///
/// # Errors
///
/// As [`translate`].
pub fn translate_with_bindings(
    query: &str,
    bindings: &HashMap<String, Value>,
) -> Result<LogicalPlan> {
    // Check for EXPLAIN [ANALYZE] prefix (case-insensitive, non-standard extension).
    // EXPLAIN: show physical plan without executing.
    // EXPLAIN ANALYZE: execute with profiling, show actual vs estimated stats.
    // Written after the prologue (`PREFIX ... EXPLAIN SELECT ...`), it is
    // read as if it came first.
    let lifted = explain_after_prologue(query);
    let query = lifted.as_deref().unwrap_or(query);
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
            // EXPLAIN ANALYZE: execute with profiling
            (false, true, rest[7..].trim_start())
        } else {
            // EXPLAIN: show plan only
            (true, false, rest)
        }
    } else if trimmed
        .get(..7)
        .is_some_and(|s| s.eq_ignore_ascii_case("PROFILE"))
        && trimmed
            .as_bytes()
            .get(7)
            .is_some_and(u8::is_ascii_whitespace)
    {
        // PROFILE: run with profiling, as EXPLAIN ANALYZE
        (false, true, trimmed[7..].trim_start())
    } else {
        (false, false, query)
    };

    let sparql_query = sparql::parse(actual_query)?;
    let mut translator = SparqlTranslator::new();
    let mut bound: Vec<(String, Value)> = bindings
        .iter()
        .filter(|(_, value)| !matches!(value, Value::Null))
        .map(|(name, value)| {
            let name = name.strip_prefix(['?', '$']).unwrap_or(name);
            (name.to_string(), value.clone())
        })
        .collect();
    bound.sort_by(|a, b| a.0.cmp(&b.0));
    translator.initial_bindings = (!bound.is_empty()).then_some(bound);
    let plan = translator.translate_query(&sparql_query)?;
    let mut plan = crate::query::limits::check_plan_depth(plan)?;
    plan.explain = explain;
    plan.profile = profile;
    Ok(plan)
}

/// `query` with an `EXPLAIN`, `EXPLAIN ANALYZE` or `PROFILE` that follows
/// its prologue (`BASE` and `PREFIX` declarations) moved before it, or
/// `None` when there is none: users write the prologue first, and the
/// keyword is read at the start.
fn explain_after_prologue(query: &str) -> Option<String> {
    let starts_with_word = |text: &str, word: &str| {
        text.get(..word.len())
            .is_some_and(|start| start.eq_ignore_ascii_case(word))
            && text
                .as_bytes()
                .get(word.len())
                .is_some_and(u8::is_ascii_whitespace)
    };
    let mut rest = query.trim_start();
    let mut prologue_end = query.len() - rest.len();
    while starts_with_word(rest, "PREFIX") || starts_with_word(rest, "BASE") {
        // A declaration ends with its IRI's `>`.
        let close = rest.find('>')?;
        rest = rest[close + 1..].trim_start();
        prologue_end = query.len() - rest.len();
    }
    if prologue_end == query.len() - query.trim_start().len() {
        return None;
    }
    let keyword = ["EXPLAIN", "PROFILE"]
        .into_iter()
        .find(|word| starts_with_word(rest, word))?;
    let mut after = &rest[keyword.len()..];
    // `EXPLAIN ANALYZE` moves as one: the `ANALYZE` left behind the
    // prologue would start the query.
    let keyword = if keyword == "EXPLAIN" && starts_with_word(after.trim_start(), "ANALYZE") {
        after = &after.trim_start()["ANALYZE".len()..];
        "EXPLAIN ANALYZE"
    } else {
        keyword
    };
    Some(format!("{keyword} {}{after}", &query[..prologue_end]))
}

/// Translator from SPARQL AST to LogicalPlan.
struct SparqlTranslator {
    /// Prefix mappings for IRI resolution.
    prefixes: HashMap<String, String>,
    /// Base IRI for relative IRI resolution.
    base: Option<String>,
    /// Counter for generating anonymous variables.
    anon_counter: u32,
    /// Stack of active graph contexts (pushed/popped around GRAPH patterns).
    graph_context_stack: Vec<TripleComponent>,
    /// Unique ID for this query (used for blank node scoping).
    query_id: u32,
    /// Dataset restriction from FROM / FROM NAMED clauses, if any.
    dataset: Option<DatasetRestriction>,
    /// The variables the caller binds (see [`translate_with_bindings`]),
    /// taken by the first graph pattern translated: the WHERE clause of the
    /// query form.
    initial_bindings: Option<Vec<(String, Value)>>,
    /// The row of the initial bindings, which the outermost group joins
    /// first.
    group_start: Option<LogicalOperator>,
    /// The empty group's one solution inside `GRAPH ?g`, when the pattern
    /// there leaves ?g unbound: one row per named graph (see [`Self::unit`]).
    graph_unit: Option<NamedGraphsOp>,
    /// How many `EXISTS` patterns are being translated around this point: in one, both sides of a
    /// `MINUS` start from the row, so one with no variable in common may still remove a row.
    exists_depth: usize,
}

/// Whether a `GRAPH ?g` over `pattern` must run it once per graph: a sub-SELECT's projection drops
/// `?g`, so its aggregate, `ORDER BY` and `LIMIT` would run over all graphs. A nested `GRAPH` has
/// its own graph.
fn holds_a_sub_select(pattern: &ast::GraphPattern) -> bool {
    match pattern {
        ast::GraphPattern::SubSelect(_) => true,
        ast::GraphPattern::Group(patterns) | ast::GraphPattern::Union(patterns) => {
            patterns.iter().any(holds_a_sub_select)
        }
        ast::GraphPattern::Optional(inner) | ast::GraphPattern::Minus(inner) => {
            holds_a_sub_select(inner)
        }
        _ => false,
    }
}

/// What a walk of a pattern meets: a variable, wherever it is written, or a sub-SELECT.
enum Met<'a> {
    Variable(&'a str),
    SubSelect,
}

/// Whether `found` holds of something `pattern` holds at any depth: its triples' terms and paths,
/// its filters, `BIND`s, `VALUES` and nested `GRAPH`s, and the patterns of its `EXISTS` and
/// sub-selects.
fn pattern_holds(pattern: &ast::GraphPattern, found: &mut dyn FnMut(Met<'_>) -> bool) -> bool {
    use ast::GraphPattern as Pattern;
    let variable_or_iri = |graph: &ast::VariableOrIri, found: &mut dyn FnMut(Met<'_>) -> bool| matches!(graph, ast::VariableOrIri::Variable(name) if found(Met::Variable(name)));
    match pattern {
        Pattern::Basic(triples) => triples.iter().any(|triple| {
            term_holds(&triple.subject, found)
                || path_holds(&triple.predicate, found)
                || term_holds(&triple.object, found)
        }),
        Pattern::Group(patterns) | Pattern::Union(patterns) => {
            patterns.iter().any(|pattern| pattern_holds(pattern, found))
        }
        Pattern::Optional(inner) | Pattern::Minus(inner) => pattern_holds(inner, found),
        Pattern::NamedGraph { graph, pattern } => {
            variable_or_iri(graph, found) || pattern_holds(pattern, found)
        }
        Pattern::Service {
            endpoint, pattern, ..
        } => variable_or_iri(endpoint, found) || pattern_holds(pattern, found),
        Pattern::Filter(expression) => expression_holds(expression, found),
        Pattern::Bind {
            expression,
            variable,
        } => found(Met::Variable(variable)) || expression_holds(expression, found),
        Pattern::InlineData(data) => data.variables.iter().any(|name| found(Met::Variable(name))),
        Pattern::SubSelect(select) => found(Met::SubSelect) || select_holds(select, found),
    }
}

/// [`pattern_holds`] of a sub-SELECT: its pattern, its projection and its modifiers.
fn select_holds(select: &ast::SelectQuery, found: &mut dyn FnMut(Met<'_>) -> bool) -> bool {
    let modifiers = &select.solution_modifiers;
    pattern_holds(&select.where_clause, found)
        || matches!(&select.projection, ast::Projection::Variables(projected) if projected.iter().any(|projected| {
            projected.alias.as_deref().is_some_and(|alias| found(Met::Variable(alias)))
                || expression_holds(&projected.expression, found)
        }))
        || modifiers
            .group_by
            .iter()
            .flatten()
            .any(|group| match group {
                ast::GroupCondition::Variable(name) => found(Met::Variable(name)),
                ast::GroupCondition::Expression { expression, alias } => {
                    alias
                        .as_deref()
                        .is_some_and(|alias| found(Met::Variable(alias)))
                        || expression_holds(expression, found)
                }
                ast::GroupCondition::BuiltInCall(expression) => expression_holds(expression, found),
            })
        || modifiers
            .having
            .as_ref()
            .is_some_and(|having| expression_holds(having, found))
        || modifiers
            .order_by
            .iter()
            .flatten()
            .any(|order| expression_holds(&order.expression, found))
        || modifiers
            .values
            .iter()
            .any(|data| data.variables.iter().any(|name| found(Met::Variable(name))))
}

/// [`pattern_holds`] of a triple's subject or object.
fn term_holds(term: &ast::TripleTerm, found: &mut dyn FnMut(Met<'_>) -> bool) -> bool {
    match term {
        ast::TripleTerm::Variable(name) => found(Met::Variable(name)),
        ast::TripleTerm::BlankNode(ast::BlankNode::Anonymous(properties)) => properties
            .iter()
            .any(|(path, object)| path_holds(path, found) || term_holds(object, found)),
        _ => false,
    }
}

/// [`pattern_holds`] of a triple's predicate.
fn path_holds(path: &ast::PropertyPath, found: &mut dyn FnMut(Met<'_>) -> bool) -> bool {
    use ast::PropertyPath as Path;
    match path {
        Path::Variable(name) => found(Met::Variable(name)),
        Path::Inverse(inner)
        | Path::ZeroOrMore(inner)
        | Path::OneOrMore(inner)
        | Path::ZeroOrOne(inner) => path_holds(inner, found),
        Path::Sequence(paths) | Path::Alternative(paths) => {
            paths.iter().any(|path| path_holds(path, found))
        }
        Path::Predicate(_) | Path::RdfType | Path::Negation(_) => false,
    }
}

/// [`pattern_holds`] of an expression.
fn expression_holds(expression: &ast::Expression, found: &mut dyn FnMut(Met<'_>) -> bool) -> bool {
    use ast::AggregateExpression as Aggregate;
    use ast::Expression;
    match expression {
        Expression::Variable(name) | Expression::Bound(name) => found(Met::Variable(name)),
        Expression::Iri(_) | Expression::Literal(_) => false,
        Expression::Binary { left, right, .. } => {
            expression_holds(left, found) || expression_holds(right, found)
        }
        Expression::Unary { operand, .. } | Expression::Bracketed(operand) => {
            expression_holds(operand, found)
        }
        Expression::FunctionCall { arguments, .. } | Expression::Coalesce(arguments) => arguments
            .iter()
            .any(|argument| expression_holds(argument, found)),
        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            expression_holds(condition, found)
                || expression_holds(then_expression, found)
                || expression_holds(else_expression, found)
        }
        Expression::Exists(pattern) | Expression::NotExists(pattern) => {
            pattern_holds(pattern, found)
        }
        Expression::In { expression, list } | Expression::NotIn { expression, list } => {
            expression_holds(expression, found)
                || list.iter().any(|item| expression_holds(item, found))
        }
        Expression::Aggregate(aggregate) => match aggregate {
            Aggregate::Count { expression, .. } => expression
                .as_deref()
                .is_some_and(|expression| expression_holds(expression, found)),
            Aggregate::Sum { expression, .. }
            | Aggregate::Average { expression, .. }
            | Aggregate::Minimum { expression }
            | Aggregate::Maximum { expression }
            | Aggregate::Sample { expression }
            | Aggregate::GroupConcat { expression, .. } => expression_holds(expression, found),
        },
    }
}

/// Whether `pattern` names the variable `name` anywhere (see [`pattern_holds`]). A sub-SELECT's
/// own `?name` counts too, which errs only towards the hidden reading of a graph.
fn names_the_variable(pattern: &ast::GraphPattern, name: &str) -> bool {
    pattern_holds(
        pattern,
        &mut |met| matches!(met, Met::Variable(variable) if variable == name),
    )
}

/// Whether `pattern` holds a sub-SELECT anywhere, an `EXISTS`'s pattern included.
fn holds_a_sub_select_anywhere(pattern: &ast::GraphPattern) -> bool {
    pattern_holds(pattern, &mut |met| matches!(met, Met::SubSelect))
}

/// Whether a scan or path in `plan`, or in the pattern of an `EXISTS` in it, reads the graph named
/// `own`: a sub-SELECT's patterns read `GRAPH ?g`'s graph so, which an `EXISTS` may hold.
fn reads_the_graph(plan: &LogicalOperator, own: &str) -> bool {
    let reads = |graph: &Option<TripleComponent>| {
        graph.as_ref().and_then(TripleComponent::as_variable) == Some(own)
    };
    let here = match plan {
        LogicalOperator::TripleScan(scan) => reads(&scan.graph),
        LogicalOperator::PropertyPath(path) => reads(&path.graph),
        _ => false,
    };
    here || sparql_expressions(plan).into_iter().any(|expression| {
        crate::query::plan::subqueries::exists_plans(expression)
            .iter()
            .any(|exists| reads_the_graph(exists, own))
    }) || plan
        .children()
        .into_iter()
        .any(|child| reads_the_graph(child, own))
}

/// The expressions `plan`'s own operator holds, of the operators a SPARQL plan holds them in:
/// a filter, a `BIND`, a projection, a sort, an aggregation and a left join's condition. The
/// RDF planner's `map_sparql_expressions` (`planner/rdf/named_graphs.rs`) maps the same list.
fn sparql_expressions(plan: &LogicalOperator) -> Vec<&LogicalExpression> {
    match plan {
        LogicalOperator::Filter(filter) => vec![&filter.predicate],
        LogicalOperator::Bind(bind) => vec![&bind.expression],
        LogicalOperator::Project(project) => project
            .projections
            .iter()
            .map(|projection| &projection.expression)
            .collect(),
        LogicalOperator::Sort(sort) => sort.keys.iter().map(|key| &key.expression).collect(),
        LogicalOperator::Aggregate(aggregate) => {
            let arguments = aggregate
                .aggregates
                .iter()
                .flat_map(|function| [&function.expression, &function.expression2])
                .flatten();
            aggregate
                .group_by
                .iter()
                .chain(arguments)
                .chain(&aggregate.having)
                .collect()
        }
        LogicalOperator::LeftJoin(join) => join.condition.iter().collect(),
        _ => Vec::new(),
    }
}

/// Whether `pattern`'s reads find a graph the dataset does not name for themselves: its triples
/// read nothing there, and a path searches only the graphs it names, a zero-length step too. A
/// sub-SELECT need not: `COUNT(*)` of nothing is a row.
fn reads_a_triple(pattern: &ast::GraphPattern) -> bool {
    match pattern {
        ast::GraphPattern::Basic(triples) => !triples.is_empty(),
        ast::GraphPattern::Group(patterns) => patterns.iter().any(reads_a_triple),
        ast::GraphPattern::Union(patterns) => patterns.iter().all(reads_a_triple),
        _ => false,
    }
}

impl SparqlTranslator {
    fn new() -> Self {
        Self {
            // The predefined prefixes, which a query's PREFIX redefines.
            prefixes: sparql::PREDEFINED_PREFIXES
                .iter()
                .map(|(prefix, namespace)| ((*prefix).to_string(), (*namespace).to_string()))
                .collect(),
            base: None,
            anon_counter: 0,
            graph_context_stack: Vec::new(),
            query_id: QUERY_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            dataset: None,
            initial_bindings: None,
            group_start: None,
            graph_unit: None,
            exists_depth: 0,
        }
    }

    fn translate_query(&mut self, query: &ast::Query) -> Result<LogicalPlan> {
        // Process prologue
        if let Some(base) = &query.base {
            self.base = Some(base.as_str().to_string());
        }

        for prefix in &query.prefixes {
            self.prefixes
                .insert(prefix.prefix.clone(), prefix.namespace.as_str().to_string());
        }

        // Translate query form
        match &query.query_form {
            ast::QueryForm::Select(select) => self.translate_select(select),
            ast::QueryForm::Ask(ask) => self.translate_ask(ask),
            ast::QueryForm::Construct(construct) => self.translate_construct(construct),
            ast::QueryForm::Describe(describe) => self.translate_describe(describe),
            ast::QueryForm::Update(update) => self.translate_update(update),
        }
    }

    fn translate_select(&mut self, select: &ast::SelectQuery) -> Result<LogicalPlan> {
        // FROM / FROM NAMED restrict this query's WHERE clause. A subquery has
        // no dataset clause and is evaluated against its parent's dataset (the
        // query's FROM, or the WITH graph of an update), so an absent clause
        // keeps the active one. Restore the parent's dataset afterwards, also
        // when translation fails.
        let parent_dataset = self.dataset.clone();
        if let Some(dataset) = self.translate_dataset_clause(&select.dataset) {
            self.dataset = Some(dataset);
        }
        let plan = self.translate_graph_pattern(&select.where_clause);
        self.dataset = parent_dataset;
        let mut plan = plan?;

        // Check if projection contains aggregates (handles both explicit GROUP BY and implicit aggregation)
        let has_aggregates = Self::has_aggregates_in_projection(&select.projection);

        // Apply GROUP BY if present, OR create aggregate for implicit aggregation
        if has_aggregates || select.solution_modifiers.group_by.is_some() {
            let (aggregates, _group_exprs) = self.extract_aggregates_for_select(select)?;

            // Get explicit GROUP BY expressions, or empty vec for whole-dataset aggregation
            let group_by_exprs = if let Some(group_by) = &select.solution_modifiers.group_by {
                let mut keys = Vec::with_capacity(group_by.len());
                for condition in group_by {
                    // `GROUP BY (expr AS ?v)` extends the solutions with ?v and
                    // groups on it (SPARQL 1.1, 18.2.4.1), so the SELECT, the
                    // HAVING and the ORDER BY read ?v.
                    if let ast::GroupCondition::Expression {
                        expression,
                        alias: Some(alias),
                    } = condition
                    {
                        plan = LogicalOperator::Bind(BindOp {
                            expression: self.translate_expression(expression)?,
                            variable: alias.clone(),
                            input: Box::new(plan),
                        });
                        keys.push(LogicalExpression::Variable(alias.clone()));
                    } else {
                        keys.push(self.translate_group_condition(condition)?);
                    }
                }
                keys
            } else {
                // No GROUP BY means aggregate over entire dataset (empty group_by)
                Vec::new()
            };

            // Translate HAVING: each aggregate in it, wherever it stands (in a
            // function call too), reads the column of the same aggregate of the
            // SELECT, or of one computed for the HAVING only
            let mut aggregates = aggregates;
            let projected_aggregates = aggregates.len();
            // A SELECT expression that is neither an aggregate nor a variable
            // (`SUM(?x) / COUNT(?x)`, `STR(COUNT(?x))`, `STR(?group_key)`) is
            // computed after the aggregation: each aggregate in it reads its
            // column, as in HAVING. It was dropped from the result.
            let mut computed: Vec<(String, LogicalExpression)> = Vec::new();
            if let ast::Projection::Variables(vars) = &select.projection {
                for pv in vars {
                    if self.is_aggregate_expression(&pv.expression)
                        || matches!(pv.expression, ast::Expression::Variable(_))
                    {
                        continue;
                    }
                    let Some(name) = pv.alias.clone() else {
                        continue;
                    };
                    let mut expression = pv.expression.clone();
                    self.take_having_aggregates(&mut expression, &mut aggregates)?;
                    computed.push((name, self.translate_expression(&expression)?));
                }
            }
            let having_expr = if let Some(having) = &select.solution_modifiers.having {
                let mut having = having.clone();
                self.take_having_aggregates(&mut having, &mut aggregates)?;
                Some(self.translate_expression(&having)?)
            } else {
                None
            };
            let hidden_aggregates = aggregates.len() > projected_aggregates;

            plan = LogicalOperator::Aggregate(AggregateOp {
                group_by: group_by_exprs,
                aggregates,
                input: Box::new(plan),
                having: having_expr,
            });
            // The aggregates the HAVING or a SELECT expression alone computes
            // are not results; the SELECT expressions are computed here.
            if (hidden_aggregates || !computed.is_empty())
                && let ast::Projection::Variables(vars) = &select.projection
            {
                plan = LogicalOperator::Project(ProjectOp {
                    projections: vars
                        .iter()
                        .filter_map(|pv| {
                            let name = pv.alias.clone().or_else(|| match &pv.expression {
                                ast::Expression::Variable(name) => Some(name.clone()),
                                _ => None,
                            })?;
                            if let Some((_, expression)) = computed
                                .iter()
                                .find(|(computed_name, _)| *computed_name == name)
                            {
                                return Some(Projection {
                                    expression: expression.clone(),
                                    alias: Some(name),
                                });
                            }
                            Some(Projection {
                                expression: LogicalExpression::Variable(name),
                                alias: None,
                            })
                        })
                        .collect(),
                    input: Box::new(plan),
                    pass_through_input: false,
                });
            }
        }

        // A trailing VALUES joins the solutions after GROUP BY and HAVING
        // (SPARQL 1.1 sec 18.2.4.3), before the SELECT expressions and the
        // ORDER BY read them.
        if let Some(values) = &select.solution_modifiers.values {
            let data =
                self.translate_graph_pattern(&ast::GraphPattern::InlineData(values.clone()))?;
            plan = self.join_patterns(plan, data)?;
        }

        let sort_keys = match &select.solution_modifiers.order_by {
            Some(order_by) => Some(
                order_by
                    .iter()
                    .map(|oc| {
                        Ok(SortKey {
                            expression: self.translate_expression(&oc.expression)?,
                            order: match oc.direction {
                                ast::SortDirection::Ascending => SortOrder::Ascending,
                                ast::SortDirection::Descending => SortOrder::Descending,
                            },
                            nulls: None,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
            None => None,
        };

        // Apply projection before ORDER BY so that computed aliases
        // (e.g. `SELECT (CONCAT(?a, ?b) AS ?full) ... ORDER BY ?full`)
        // are available to the sort operator. SPARQL orders the solutions
        // before it projects them (SPARQL 1.1, 18.2.5), so ORDER BY may read
        // a variable the SELECT leaves out: the projection carries it to the
        // sort, and a second one leaves it out after.
        let mut projected_after_sort = None;
        if !has_aggregates {
            let mut projections = self.translate_projection(&select.projection)?;
            if !projections.is_empty() {
                let names: Vec<String> = projections
                    .iter()
                    .map(|projection| {
                        crate::query::planner::common::output_column_name(
                            projection.alias.as_deref(),
                            &projection.expression,
                        )
                    })
                    .collect();
                let mut read = HashSet::new();
                for key in sort_keys.iter().flatten() {
                    collect_expression_variables(&key.expression, &mut read);
                    // An `EXISTS` reads the whole solution by substitution (18.6), whatever the
                    // SELECT lists.
                    if crate::query::plan::has_subquery(&key.expression) {
                        read.extend(Self::collect_operator_variables(&plan));
                    }
                }
                let mut carried: Vec<&String> =
                    read.iter().filter(|name| !names.contains(name)).collect();
                carried.sort();
                if !carried.is_empty() {
                    projections.extend(carried.into_iter().map(|name| Projection {
                        expression: LogicalExpression::Variable(name.clone()),
                        alias: None,
                    }));
                    projected_after_sort = Some(names);
                }
                plan = LogicalOperator::Project(ProjectOp {
                    projections,
                    input: Box::new(plan),
                    pass_through_input: false,
                });
            }
        }

        // Apply ORDER BY (after projection so aliases are resolvable)
        if let Some(keys) = sort_keys {
            plan = wrap_sort(plan, keys);
        }
        if let Some(names) = projected_after_sort {
            plan = LogicalOperator::Project(ProjectOp {
                projections: names
                    .into_iter()
                    .map(|name| Projection {
                        expression: LogicalExpression::Variable(name),
                        alias: None,
                    })
                    .collect(),
                input: Box::new(plan),
                pass_through_input: false,
            });
        }

        // Apply DISTINCT (after projection, before OFFSET/LIMIT).
        // REDUCED is a no-op: the spec allows returning all rows unchanged.
        if matches!(select.modifier, ast::SelectModifier::Distinct) {
            plan = wrap_distinct(plan);
        }

        // Apply OFFSET
        if let Some(offset) = select.solution_modifiers.offset {
            // reason: SPARQL u64 offset fits usize on 64-bit targets
            #[allow(clippy::cast_possible_truncation)]
            let skip_n = offset as usize;
            plan = wrap_skip(plan, skip_n);
        }

        // Apply LIMIT
        if let Some(limit) = select.solution_modifiers.limit {
            // reason: SPARQL u64 limit fits usize on 64-bit targets
            #[allow(clippy::cast_possible_truncation)]
            let limit_n = limit as usize;
            plan = wrap_limit(plan, limit_n);
        }

        Ok(LogicalPlan::new(plan))
    }

    fn translate_ask(&mut self, ask: &ast::AskQuery) -> Result<LogicalPlan> {
        // Apply dataset restriction from FROM / FROM NAMED clauses
        self.dataset = self.translate_dataset_clause(&ask.dataset);

        // ASK returns true if the pattern has any matches
        let plan = self.translate_graph_pattern(&ask.where_clause)?;

        // Clear dataset after translating the WHERE clause
        self.dataset = None;

        // The first solution decides: stop there.
        let plan = wrap_limit(plan, 1);

        // One row holding one Boolean, whether there is a solution, in the
        // column the SPARQL 1.1 Query Results JSON Format names an ASK result
        // (section 4.2, "boolean").
        const SOLUTIONS: &str = "__ask_solutions";
        let counted = LogicalOperator::Aggregate(AggregateOp {
            group_by: Vec::new(),
            aggregates: vec![AggregateExpr {
                function: AggregateFunction::Count,
                expression: None,
                expression2: None,
                distinct: false,
                alias: Some(SOLUTIONS.to_string()),
                percentile: None,
                separator: None,
            }],
            input: Box::new(plan),
            having: None,
        });
        Ok(LogicalPlan::new(LogicalOperator::Project(ProjectOp {
            projections: vec![Projection {
                expression: LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Variable(SOLUTIONS.to_string())),
                    op: BinaryOp::Gt,
                    right: Box::new(LogicalExpression::Literal(Value::Int64(0))),
                },
                alias: Some("boolean".to_string()),
            }],
            input: Box::new(counted),
            pass_through_input: false,
        }))
        .with_result_kind(ResultKind::Boolean))
    }

    fn translate_construct(&mut self, construct: &ast::ConstructQuery) -> Result<LogicalPlan> {
        // Apply dataset restriction from FROM / FROM NAMED clauses
        self.dataset = self.translate_dataset_clause(&construct.dataset);

        // Evaluate the WHERE pattern to produce variable bindings
        let mut plan = self.translate_graph_pattern(&construct.where_clause)?;

        // Clear dataset after translating the WHERE clause
        self.dataset = None;

        // Apply solution modifiers to the WHERE output
        if let Some(limit) = construct.solution_modifiers.limit {
            // reason: SPARQL u64 limit fits usize on 64-bit targets
            #[allow(clippy::cast_possible_truncation)]
            let limit_n = limit as usize;
            plan = wrap_limit(plan, limit_n);
        }

        // Translate template triples: substitute variable bindings from WHERE.
        // Use translate_data_term for subject/object so that blank nodes become
        // TripleComponent::BlankNode (constants in output) rather than variables
        // that would fail to bind against the WHERE clause results.
        let mut templates = Vec::new();
        for tp in &construct.template {
            let subject = self.translate_data_term(&tp.subject)?;
            // CONSTRUCT templates use simple predicates (IRIs/variables), not paths
            let predicate = match &tp.predicate {
                ast::PropertyPath::Predicate(iri) => TripleComponent::Iri(self.resolve_iri(iri)),
                ast::PropertyPath::Variable(name) => TripleComponent::Variable(name.clone()),
                ast::PropertyPath::RdfType => TripleComponent::Iri(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                ),
                _ => continue, // Skip complex property paths in templates
            };
            let object = self.translate_data_term(&tp.object)?;
            templates.push(TripleTemplate {
                subject,
                predicate,
                object,
                graph: None,
            });
        }

        Ok(LogicalPlan::new(LogicalOperator::Construct(ConstructOp {
            templates,
            input: Box::new(plan),
        }))
        .with_result_kind(ResultKind::Graph))
    }

    fn translate_describe(&mut self, describe: &ast::DescribeQuery) -> Result<LogicalPlan> {
        // Apply dataset restriction from FROM / FROM NAMED clauses
        self.dataset = self.translate_dataset_clause(&describe.dataset);

        // DESCRIBE implements Concise Bounded Description (CBD): the
        // triples whose subject is the described resource, as a graph.
        let subject_var = format!("__describe_s{}", self.next_anon());
        let pred_var = format!("__describe_p{}", self.next_anon());
        let obj_var = format!("__describe_o{}", self.next_anon());

        let mut cbd_scans: Vec<LogicalOperator> = Vec::new();
        for resource in &describe.resources {
            let (subject, bound) = match resource {
                ast::VariableOrIri::Iri(iri) => {
                    let iri = self.resolve_iri(iri);
                    (
                        TripleComponent::Iri(iri.clone()),
                        LogicalExpression::Literal(iri_term(&iri)),
                    )
                }
                ast::VariableOrIri::Variable(name) => (
                    TripleComponent::Variable(name.clone()),
                    LogicalExpression::Variable(name.clone()),
                ),
            };
            let scan = self.make_triple_scan(
                subject,
                TripleComponent::Variable(pred_var.clone()),
                TripleComponent::Variable(obj_var.clone()),
                None,
            );
            // Every branch binds the resource to one variable, the template's subject.
            cbd_scans.push(LogicalOperator::Bind(BindOp {
                expression: bound,
                variable: subject_var.clone(),
                input: Box::new(scan),
            }));
        }

        let cbd_plan = if cbd_scans.len() == 1 {
            cbd_scans.into_iter().next().expect("single CBD scan")
        } else {
            LogicalOperator::Union(UnionOp { inputs: cbd_scans })
        };

        let plan = if let Some(where_clause) = &describe.where_clause {
            let where_plan = self.translate_graph_pattern(where_clause)?;
            self.join_patterns(where_plan, cbd_plan)?
        } else {
            cbd_plan
        };

        // Clear dataset after translating the WHERE clause
        self.dataset = None;

        Ok(LogicalPlan::new(LogicalOperator::Construct(ConstructOp {
            templates: vec![TripleTemplate {
                subject: TripleComponent::Variable(subject_var),
                predicate: TripleComponent::Variable(pred_var),
                object: TripleComponent::Variable(obj_var),
                graph: None,
            }],
            input: Box::new(plan),
        }))
        .with_result_kind(ResultKind::Graph))
    }

    // ==================== SPARQL Update Translation ====================

    fn translate_update(&mut self, update: &ast::UpdateOperation) -> Result<LogicalPlan> {
        match update {
            ast::UpdateOperation::InsertData { data } => self.translate_insert_data(data),
            ast::UpdateOperation::DeleteData { data } => self.translate_delete_data(data),
            ast::UpdateOperation::DeleteWhere { pattern } => self.translate_delete_where(pattern),
            ast::UpdateOperation::Modify {
                with_graph,
                delete_template,
                insert_template,
                using_clauses,
                where_clause,
            } => self.translate_modify(
                with_graph,
                delete_template,
                insert_template,
                using_clauses,
                where_clause,
            ),
            ast::UpdateOperation::Load {
                silent,
                source,
                destination,
            } => self.translate_load(*silent, source, destination.as_ref()),
            ast::UpdateOperation::Clear { silent, target } => self.translate_clear(*silent, target),
            ast::UpdateOperation::Drop { silent, target } => self.translate_drop(*silent, target),
            ast::UpdateOperation::Create { silent, graph } => self.translate_create(*silent, graph),
            ast::UpdateOperation::Copy {
                silent,
                source,
                destination,
            } => self.translate_copy(*silent, source, destination),
            ast::UpdateOperation::Move {
                silent,
                source,
                destination,
            } => self.translate_move(*silent, source, destination),
            ast::UpdateOperation::Add {
                silent,
                source,
                destination,
            } => self.translate_add(*silent, source, destination),
        }
    }

    fn translate_insert_data(&mut self, data: &[ast::QuadPattern]) -> Result<LogicalPlan> {
        // Build a sequence of InsertTriple operators
        let mut ops = Vec::new();
        for quad in data {
            let subject = self.translate_data_term(&quad.triple.subject)?;
            let predicate = self.translate_property_path(&quad.triple.predicate)?;
            let object = self.translate_data_term(&quad.triple.object)?;
            let graph = quad.graph.as_ref().map(|g| self.resolve_variable_or_iri(g));

            ops.push(LogicalOperator::InsertTriple(InsertTripleOp {
                subject,
                predicate,
                object,
                graph,
                input: None,
            }));
        }

        // Combine all inserts into a sequence using Union
        if ops.is_empty() {
            Ok(LogicalPlan::new(LogicalOperator::Empty))
        } else if ops.len() == 1 {
            Ok(LogicalPlan::new(
                ops.into_iter().next().expect("single-element iterator"),
            ))
        } else {
            Ok(LogicalPlan::new(LogicalOperator::Union(UnionOp {
                inputs: ops,
            })))
        }
    }

    /// Translates a triple term in a data context (INSERT DATA / DELETE DATA).
    ///
    /// Blank nodes become `TripleComponent::BlankNode` instead of variables,
    /// because data operations have no WHERE clause to bind variables against.
    fn translate_data_term(&mut self, term: &ast::TripleTerm) -> Result<TripleComponent> {
        match term {
            ast::TripleTerm::BlankNode(bnode) => {
                let label = match bnode {
                    ast::BlankNode::Labeled(label) => {
                        format!("q{}_{label}", self.query_id)
                    }
                    ast::BlankNode::Anonymous(_) => {
                        let anon = self.next_anon();
                        format!("q{}_anon{anon}", self.query_id)
                    }
                };
                Ok(TripleComponent::BlankNode(label))
            }
            // Non-blank-node terms use the standard translation
            other => self.translate_triple_term(other),
        }
    }

    fn translate_delete_data(&mut self, data: &[ast::QuadPattern]) -> Result<LogicalPlan> {
        // Build a sequence of DeleteTriple operators
        let mut ops = Vec::new();
        for quad in data {
            let subject = self.translate_triple_term(&quad.triple.subject)?;
            let predicate = self.translate_property_path(&quad.triple.predicate)?;
            let object = self.translate_triple_term(&quad.triple.object)?;
            let graph = quad.graph.as_ref().map(|g| self.resolve_variable_or_iri(g));

            ops.push(LogicalOperator::DeleteTriple(DeleteTripleOp {
                subject,
                predicate,
                object,
                graph,
                input: None,
            }));
        }

        if ops.is_empty() {
            Ok(LogicalPlan::new(LogicalOperator::Empty))
        } else if ops.len() == 1 {
            Ok(LogicalPlan::new(
                ops.into_iter().next().expect("single-element iterator"),
            ))
        } else {
            Ok(LogicalPlan::new(LogicalOperator::Union(UnionOp {
                inputs: ops,
            })))
        }
    }

    /// `DELETE WHERE { P }` is `DELETE { P } WHERE { P }` (SPARQL 1.1 Update
    /// 3.1.3.3): every solution of the whole pattern, read before any
    /// change, deletes its instance of each triple of P, a triple without
    /// variables included, and no solution deletes nothing.
    fn translate_delete_where(&mut self, pattern: &ast::GraphPattern) -> Result<LogicalPlan> {
        let where_plan = self.translate_graph_pattern(pattern)?;

        let mut delete_templates = Vec::new();
        for (triple, graph) in &Self::extract_triples_from_pattern(pattern) {
            let subject = self.translate_triple_term(&triple.subject)?;
            let predicate = self.translate_property_path(&triple.predicate)?;
            let object = self.translate_triple_term(&triple.object)?;
            // `GRAPH ?g` resolves to a "?"-prefixed name, which the executor rejects.
            let graph = graph.as_ref().map(|g| self.resolve_variable_or_iri(g));
            delete_templates.push(TripleTemplate {
                subject,
                predicate,
                object,
                graph,
            });
        }

        Ok(LogicalPlan::new(LogicalOperator::Modify(ModifyOp {
            delete_templates,
            insert_templates: Vec::new(),
            where_clause: Box::new(where_plan),
            graph: None,
        })))
    }

    /// Extracts the triple templates of a DELETE WHERE pattern, each tagged with
    /// its enclosing `GRAPH` (if any) so the delete targets that graph.
    fn extract_triples_from_pattern(
        pattern: &ast::GraphPattern,
    ) -> Vec<(ast::TriplePattern, Option<ast::VariableOrIri>)> {
        Self::extract_triples_with_graph(pattern, None)
    }

    fn extract_triples_with_graph(
        pattern: &ast::GraphPattern,
        graph: Option<&ast::VariableOrIri>,
    ) -> Vec<(ast::TriplePattern, Option<ast::VariableOrIri>)> {
        match pattern {
            ast::GraphPattern::Basic(triples) => triples
                .iter()
                .map(|t| (t.clone(), graph.cloned()))
                .collect(),
            ast::GraphPattern::Group(patterns) => patterns
                .iter()
                .flat_map(|p| Self::extract_triples_with_graph(p, graph))
                .collect(),
            ast::GraphPattern::NamedGraph {
                graph: inner_graph,
                pattern: inner_pattern,
            } => Self::extract_triples_with_graph(inner_pattern, Some(inner_graph)),
            _ => Vec::new(),
        }
    }

    fn translate_modify(
        &mut self,
        with_graph: &Option<ast::Iri>,
        delete_template: &Option<Vec<ast::QuadPattern>>,
        insert_template: &Option<Vec<ast::QuadPattern>>,
        using_clauses: &[ast::UsingClause],
        where_clause: &ast::GraphPattern,
    ) -> Result<LogicalPlan> {
        // USING / USING NAMED is parsed but not executed yet; reject it instead
        // of evaluating the WHERE clause against the wrong dataset.
        if !using_clauses.is_empty() {
            return Err(Error::from(QueryError::new(
                QueryErrorKind::Unsupported,
                "SPARQL Update USING / USING NAMED is not yet supported",
            )));
        }

        let default_graph = with_graph.as_ref().map(|g| self.resolve_iri(g));

        // WITH <g> makes <g> the default graph of the WHERE clause (SPARQL 1.1
        // Update 3.1.3). Redirect the dataset while translating WHERE, then
        // restore the previous one, also when translation fails.
        let previous_dataset = default_graph.as_ref().map(|graph| {
            self.dataset.replace(DatasetRestriction {
                default_graphs: vec![graph.clone()],
                named_graphs: DatasetNamedGraphs::OfTheStore,
            })
        });

        // Translate the WHERE clause - this will be evaluated once and shared
        let where_plan = self.translate_graph_pattern(where_clause);
        if let Some(previous) = previous_dataset {
            self.dataset = previous;
        }
        let where_plan = where_plan?;

        // Build DELETE templates
        let mut delete_templates = Vec::new();
        if let Some(delete_quads) = delete_template {
            for quad in delete_quads {
                let subject = self.translate_triple_term(&quad.triple.subject)?;
                let predicate = self.translate_property_path(&quad.triple.predicate)?;
                let object = self.translate_triple_term(&quad.triple.object)?;
                let graph = quad
                    .graph
                    .as_ref()
                    .map(|g| self.resolve_variable_or_iri(g))
                    .or_else(|| default_graph.clone());

                delete_templates.push(TripleTemplate {
                    subject,
                    predicate,
                    object,
                    graph,
                });
            }
        }

        // Build INSERT templates
        let mut insert_templates = Vec::new();
        if let Some(insert_quads) = insert_template {
            for quad in insert_quads {
                let subject = self.translate_triple_term(&quad.triple.subject)?;
                let predicate = self.translate_property_path(&quad.triple.predicate)?;
                let object = self.translate_triple_term(&quad.triple.object)?;
                let graph = quad
                    .graph
                    .as_ref()
                    .map(|g| self.resolve_variable_or_iri(g))
                    .or_else(|| default_graph.clone());

                insert_templates.push(TripleTemplate {
                    subject,
                    predicate,
                    object,
                    graph,
                });
            }
        }

        // Use ModifyOp which handles SPARQL MODIFY semantics correctly:
        // 1. Evaluate WHERE once
        // 2. Apply DELETE templates
        // 3. Apply INSERT templates (using same bindings)
        Ok(LogicalPlan::new(LogicalOperator::Modify(ModifyOp {
            delete_templates,
            insert_templates,
            where_clause: Box::new(where_plan),
            graph: default_graph,
        })))
    }

    fn translate_load(
        &mut self,
        silent: bool,
        source: &ast::Iri,
        destination: Option<&ast::Iri>,
    ) -> Result<LogicalPlan> {
        Ok(LogicalPlan::new(LogicalOperator::LoadGraph(LoadGraphOp {
            source: self.resolve_iri(source),
            destination: destination.map(|d| self.resolve_iri(d)),
            silent,
        })))
    }

    fn translate_clear(&mut self, silent: bool, target: &ast::GraphTarget) -> Result<LogicalPlan> {
        let target = self.translate_graphs_target(target);
        Ok(LogicalPlan::new(LogicalOperator::ClearGraph(
            ClearGraphOp { target, silent },
        )))
    }

    fn translate_drop(&mut self, silent: bool, target: &ast::GraphTarget) -> Result<LogicalPlan> {
        let target = self.translate_graphs_target(target);
        Ok(LogicalPlan::new(LogicalOperator::DropGraph(DropGraphOp {
            target,
            silent,
        })))
    }

    /// The graphs a `CLEAR` or `DROP` names.
    fn translate_graphs_target(&self, target: &ast::GraphTarget) -> RdfGraphTarget {
        match target {
            ast::GraphTarget::Default => RdfGraphTarget::Default,
            ast::GraphTarget::Named(iri) => RdfGraphTarget::Named(self.resolve_iri(iri)),
            ast::GraphTarget::AllNamed => RdfGraphTarget::AllNamed,
            ast::GraphTarget::All => RdfGraphTarget::All,
        }
    }

    fn translate_create(&mut self, silent: bool, graph: &ast::Iri) -> Result<LogicalPlan> {
        Ok(LogicalPlan::new(LogicalOperator::CreateGraph(
            CreateGraphOp {
                graph: self.resolve_iri(graph),
                silent,
            },
        )))
    }

    fn translate_copy(
        &mut self,
        silent: bool,
        source: &ast::GraphTarget,
        destination: &ast::GraphTarget,
    ) -> Result<LogicalPlan> {
        Ok(LogicalPlan::new(LogicalOperator::CopyGraph(CopyGraphOp {
            source: self.translate_graph_target(source)?,
            destination: self.translate_graph_target(destination)?,
            silent,
        })))
    }

    fn translate_move(
        &mut self,
        silent: bool,
        source: &ast::GraphTarget,
        destination: &ast::GraphTarget,
    ) -> Result<LogicalPlan> {
        Ok(LogicalPlan::new(LogicalOperator::MoveGraph(MoveGraphOp {
            source: self.translate_graph_target(source)?,
            destination: self.translate_graph_target(destination)?,
            silent,
        })))
    }

    fn translate_add(
        &mut self,
        silent: bool,
        source: &ast::GraphTarget,
        destination: &ast::GraphTarget,
    ) -> Result<LogicalPlan> {
        Ok(LogicalPlan::new(LogicalOperator::AddGraph(AddGraphOp {
            source: self.translate_graph_target(source)?,
            destination: self.translate_graph_target(destination)?,
            silent,
        })))
    }

    /// The one graph a `COPY`, `MOVE` or `ADD` reads or writes: `None` for
    /// the default graph.
    fn translate_graph_target(&self, target: &ast::GraphTarget) -> Result<Option<String>> {
        match target {
            ast::GraphTarget::Default => Ok(None),
            ast::GraphTarget::Named(iri) => Ok(Some(self.resolve_iri(iri))),
            ast::GraphTarget::AllNamed | ast::GraphTarget::All => {
                Err(Error::from(QueryError::new(
                    QueryErrorKind::Semantic,
                    "COPY, MOVE and ADD name one graph: DEFAULT or a graph IRI, not NAMED or ALL",
                )))
            }
        }
    }

    fn resolve_variable_or_iri(&self, var_or_iri: &ast::VariableOrIri) -> String {
        match var_or_iri {
            ast::VariableOrIri::Variable(name) => format!("?{}", name),
            ast::VariableOrIri::Iri(iri) => self.resolve_iri(iri),
        }
    }

    fn translate_projection(&mut self, projection: &ast::Projection) -> Result<Vec<Projection>> {
        match projection {
            ast::Projection::Wildcard => Ok(Vec::new()), // Empty means select all
            ast::Projection::Variables(vars) => vars
                .iter()
                .map(|pv| {
                    Ok(Projection {
                        expression: self.translate_expression(&pv.expression)?,
                        alias: pv.alias.clone(),
                    })
                })
                .collect(),
        }
    }

    /// Translates `pattern`; the outermost one (the query's WHERE clause)
    /// joins the initial bindings first, as a `VALUES` row at the start of
    /// its group would be, so the group's filters see them.
    fn translate_graph_pattern(&mut self, pattern: &ast::GraphPattern) -> Result<LogicalOperator> {
        let Some(bindings) = self.initial_bindings.take() else {
            return self.translate_graph_pattern_unbound(pattern);
        };
        let mut row = LogicalOperator::Empty;
        for (variable, value) in bindings {
            row = LogicalOperator::Bind(BindOp {
                expression: LogicalExpression::Literal(value),
                variable,
                input: Box::new(row),
            });
        }
        if matches!(pattern, ast::GraphPattern::Group(_)) {
            self.group_start = Some(row);
            return self.translate_graph_pattern_unbound(pattern);
        }
        let plan = self.translate_graph_pattern_unbound(pattern)?;
        self.join_patterns(row, plan)
    }

    /// Translates the elements of a group, with its FILTERs left out: they
    /// scope over the whole group, so they are returned for the caller to
    /// apply (a group filters its result; an OPTIONAL makes them the
    /// condition of its left join). FILTER EXISTS and NOT EXISTS are applied
    /// here.
    fn translate_group(
        &mut self,
        patterns: &[ast::GraphPattern],
    ) -> Result<(LogicalOperator, Vec<LogicalExpression>)> {
        self.translate_group_returning(patterns, false)
    }

    /// [`translate_group`](Self::translate_group), returning the FILTER
    /// EXISTS and NOT EXISTS among the filters too when `exists_too`: an
    /// OPTIONAL decides them per pair, in its left join's condition.
    fn translate_group_returning(
        &mut self,
        patterns: &[ast::GraphPattern],
        exists_too: bool,
    ) -> Result<(LogicalOperator, Vec<LogicalExpression>)> {
        // Process patterns in document order so that BIND, OPTIONAL,
        // MINUS, etc. see the variables introduced by preceding
        // patterns. FILTER and FILTER NOT EXISTS/EXISTS scope over
        // the entire group (SPARQL spec), so they are collected and
        // applied last.
        let mut filter_exprs: Vec<&ast::Expression> = Vec::new();
        let mut exists_filters: Vec<&ast::Expression> = Vec::new();

        // The initial bindings start the outermost group, so its
        // filters see them (a `VALUES` row in the group would).
        let mut plan = match self.group_start.take() {
            Some(start) => start,
            None => self.unit(),
        };

        for p in patterns {
            match p {
                ast::GraphPattern::Filter(expr) => {
                    // Collect filters for group-level application
                    match expr {
                        ast::Expression::NotExists(_) | ast::Expression::Exists(_)
                            if !exists_too =>
                        {
                            exists_filters.push(expr);
                        }
                        _ => filter_exprs.push(expr),
                    }
                }
                ast::GraphPattern::Bind {
                    expression,
                    variable,
                } => {
                    let expr = self.translate_expression(expression)?;
                    plan = LogicalOperator::Bind(BindOp {
                        expression: expr,
                        variable: variable.clone(),
                        input: Box::new(plan),
                    });
                }
                ast::GraphPattern::Optional(inner) => {
                    plan = self.translate_optional(plan, inner)?;
                }
                ast::GraphPattern::Minus(inner) => {
                    let inner_plan = self.translate_graph_pattern(inner)?;
                    // Minus(Z, P) is Z; MINUS removes a solution only through
                    // a variable both sides bind.
                    if self.exists_depth > 0
                        || (!self.is_unit(&plan)
                            && self.share_a_pattern_variable(&plan, &inner_plan))
                    {
                        plan = LogicalOperator::AntiJoin(AntiJoinOp {
                            left: Box::new(plan),
                            right: Box::new(inner_plan),
                        });
                    }
                }
                // InlineData with UNDEF entries: translate as a Union of
                // filtered copies of the base plan.
                ast::GraphPattern::InlineData(data)
                    if data
                        .values
                        .iter()
                        .any(|row| row.iter().any(Option::is_none)) =>
                {
                    if self.is_unit(&plan) {
                        let values = self.translate_graph_pattern(
                            &ast::GraphPattern::InlineData(data.clone()),
                        )?;
                        plan = self.join_patterns(plan, values)?;
                    } else {
                        let mut branches = Vec::new();
                        for row in &data.values {
                            let mut conditions: Vec<LogicalExpression> = Vec::new();
                            for (var, val) in data.variables.iter().zip(row.iter()) {
                                if let Some(dv) = val {
                                    let value = self.data_value_to_value(dv);
                                    conditions.push(LogicalExpression::Binary {
                                        left: Box::new(LogicalExpression::Variable(var.clone())),
                                        op: BinaryOp::Eq,
                                        right: Box::new(LogicalExpression::Literal(value)),
                                    });
                                }
                            }
                            if conditions.is_empty() {
                                branches.push(plan.clone());
                            } else {
                                let combined = LogicalExpression::conjunction(conditions)
                                    .expect("conditions non-empty");
                                branches.push(wrap_filter(plan.clone(), combined));
                            }
                        }
                        plan = if branches.len() == 1 {
                            branches.into_iter().next().expect("single branch")
                        } else {
                            LogicalOperator::Union(UnionOp { inputs: branches })
                        };
                    }
                }
                _ => {
                    let p_plan = self.translate_graph_pattern(p)?;
                    plan = self.join_patterns(plan, p_plan)?;
                }
            }
        }

        // Each FILTER EXISTS or NOT EXISTS filters the group on its own: the
        // planner runs it as a semi or anti join, which keeps or drops rows
        // and never adds one or a variable (a FILTER binds nothing). An
        // empty group so far is Z, one solution, which the filter keeps or
        // drops as any other.
        for expr in exists_filters {
            let predicate = self.translate_expression(expr)?;
            plan = wrap_filter(plan, predicate);
        }

        let filters = filter_exprs
            .into_iter()
            .map(|e| self.translate_expression(e))
            .collect::<Result<Vec<_>>>()?;
        Ok((plan, filters))
    }

    /// `OPTIONAL { P }` after `left`, the group so far, is LeftJoin(left, P,
    /// F) (SPARQL 1.1 sec 18.2.2.6), F the FILTERs at the top of P's group.
    /// F decides which pairs match over the merged solution, so it reads the
    /// variables bound before the OPTIONAL too, an `EXISTS` or `NOT EXISTS`
    /// among them; a FILTER with a subquery elsewhere in it stays inside P. An empty `left` is Z, the empty
    /// group's one solution: an OPTIONAL that opens a group keeps it when P
    /// has no solution.
    fn translate_optional(
        &mut self,
        left: LogicalOperator,
        inner: &ast::GraphPattern,
    ) -> Result<LogicalOperator> {
        let (right, condition) = if let ast::GraphPattern::Group(patterns) = inner {
            // Every filter is decided per pair, over the merged solution.
            let (plan, filters) = self.translate_group_returning(patterns, true)?;
            (plan, LogicalExpression::conjunction(filters))
        } else {
            (self.translate_graph_pattern(inner)?, None)
        };
        Ok(LogicalOperator::LeftJoin(LeftJoinOp {
            left: Box::new(left),
            right: Box::new(right),
            condition,
        }))
    }

    /// `GRAPH <iri> { P }`: P with its scans reading the graph, and nothing when the dataset has no
    /// such graph.
    fn translate_graph_iri(
        &mut self,
        pattern: &ast::GraphPattern,
        iri: &ast::Iri,
    ) -> Result<LogicalOperator> {
        let iri = self.resolve_iri(iri);
        self.graph_context_stack
            .push(TripleComponent::Iri(iri.clone()));
        let outer_unit = self.graph_unit.take();
        let plan = match self.translate_graph_pattern(pattern) {
            // With a triple its solutions must read, the scans find nothing in a graph the dataset
            // does not name; otherwise it must say so.
            Ok(plan) if !reads_a_triple(pattern) => {
                Ok(LogicalOperator::InNamedGraph(InNamedGraphOp {
                    graph: NamedGraph::Iri(iri),
                    dataset: self.dataset.clone(),
                    input: Box::new(plan),
                    read_as: None,
                }))
            }
            plan => plan,
        };
        self.graph_unit = outer_unit;
        self.graph_context_stack.pop();
        plan
    }

    /// `GRAPH ?name { P }`: P evaluated in each named graph, then joined with `?name` bound to that
    /// graph (sec 18.6). Where P uses `?name` only as its graph, its scans read the graph into
    /// `?name`, which is that join. Where P names `?name` itself, `?name` is free in P: its scans
    /// read the graph under a hidden name, and the join follows P.
    fn translate_graph_variable(
        &mut self,
        pattern: &ast::GraphPattern,
        name: &str,
    ) -> Result<LogicalOperator> {
        let read_as = GroupGraph {
            name: group_graph_name(name, self.graph_context_stack.len()),
            whole_pattern: names_the_variable(pattern, name),
        };
        let read_into = if read_as.whole_pattern {
            read_as.name.clone()
        } else {
            name.to_string()
        };
        self.graph_context_stack
            .push(TripleComponent::Variable(read_into.clone()));
        let outer_unit = self.graph_unit.take();
        let plan = match self.translate_graph_pattern(pattern) {
            // A solution of P that reads no graph (the empty group's, which an OPTIONAL or MINUS
            // can keep) belongs to every graph: P is translated again with that solution as one
            // row per named graph.
            Ok(plan) if !always_bound(&plan).is_none_or(|bound| bound.contains(&read_into)) => {
                self.graph_unit = Some(NamedGraphsOp {
                    variable: read_into,
                    dataset: self.dataset.clone(),
                });
                self.translate_graph_pattern(pattern)
            }
            plan => plan,
        };
        // A sub-SELECT's projection drops the graph, so its aggregate, `ORDER BY` and `LIMIT` run
        // once per graph.
        let per_graph = |plan: &LogicalOperator| {
            if read_as.whole_pattern {
                holds_a_sub_select_anywhere(pattern)
            } else {
                holds_a_sub_select(pattern) || reads_the_graph(plan, &read_as.name)
            }
        };
        let plan = match plan {
            Ok(plan) if per_graph(&plan) => Ok(LogicalOperator::InNamedGraph(InNamedGraphOp {
                graph: NamedGraph::Variable(name.to_string()),
                dataset: self.dataset.clone(),
                input: Box::new(plan),
                read_as: Some(read_as),
            })),
            Ok(plan) if read_as.whole_pattern => {
                Ok(self.joined_with_its_graph(plan, name, &read_as.name))
            }
            plan => plan,
        };
        self.graph_unit = outer_unit;
        self.graph_context_stack.pop();
        plan
    }

    /// `plan`, the pattern of `GRAPH ?name` that read its graph under `graph`, joined with `?name`
    /// bound to that graph, as sec 18.6 joins it: a solution whose own `?name` is another graph
    /// goes, and `?name` is the graph. A solution that read no graph (a branch of a UNION that holds
    /// only a nested `GRAPH`) is the pattern's in every graph, so it meets each first.
    fn joined_with_its_graph(
        &self,
        plan: LogicalOperator,
        name: &str,
        graph: &str,
    ) -> LogicalOperator {
        let plan = if always_bound(&plan).is_none_or(|bound| bound.contains(graph)) {
            plan
        } else {
            LogicalOperator::Join(JoinOp {
                left: Box::new(LogicalOperator::NamedGraphs(NamedGraphsOp {
                    variable: graph.to_string(),
                    dataset: self.dataset.clone(),
                })),
                right: Box::new(plan),
                join_type: JoinType::Inner,
                conditions: Vec::new(),
            })
        };
        let graph = LogicalExpression::Variable(graph.to_string());
        let plan = if Self::collect_operator_variables(&plan)
            .iter()
            .any(|variable| variable == name)
        {
            let own = LogicalExpression::Variable(name.to_string());
            let unbound = LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand: Box::new(LogicalExpression::SparqlCall {
                    name: "BOUND".to_string(),
                    args: vec![own.clone()],
                }),
            };
            let same = LogicalExpression::SparqlCall {
                name: "SAMETERM".to_string(),
                args: vec![own, graph.clone()],
            };
            wrap_filter(
                plan,
                LogicalExpression::Binary {
                    left: Box::new(unbound),
                    op: BinaryOp::Or,
                    right: Box::new(same),
                },
            )
        } else {
            plan
        };
        LogicalOperator::Bind(BindOp {
            expression: graph,
            variable: name.to_string(),
            input: Box::new(plan),
        })
    }

    /// The pattern of an `EXISTS` or `NOT EXISTS`.
    fn translate_exists_pattern(&mut self, pattern: &ast::GraphPattern) -> Result<LogicalOperator> {
        self.exists_depth += 1;
        let subquery = self.translate_graph_pattern(pattern);
        self.exists_depth -= 1;
        subquery
    }

    /// The empty group's one solution: Z, or inside a `GRAPH ?g` whose
    /// pattern leaves ?g unbound, one row per named graph with ?g bound.
    fn unit(&self) -> LogicalOperator {
        match &self.graph_unit {
            Some(graphs) => LogicalOperator::NamedGraphs(graphs.clone()),
            None => LogicalOperator::Empty,
        }
    }

    /// Whether `plan` is [`Self::unit`].
    fn is_unit(&self, plan: &LogicalOperator) -> bool {
        match plan {
            LogicalOperator::Empty => true,
            LogicalOperator::NamedGraphs(graphs) => self
                .graph_unit
                .as_ref()
                .is_some_and(|unit| unit.variable == graphs.variable),
            _ => false,
        }
    }

    /// Whether `left` and `right` share a variable of the pattern. Inside
    /// `GRAPH ?g`, ?g names the graph the pattern is evaluated in, not a
    /// variable of it (sec 18.6): sharing it alone is sharing none.
    fn share_a_pattern_variable(&self, left: &LogicalOperator, right: &LogicalOperator) -> bool {
        let right_variables = Self::collect_operator_variables(right);
        Self::collect_operator_variables(left).iter().any(|name| {
            right_variables.contains(name)
                && !self
                    .graph_context_stack
                    .iter()
                    .any(|graph| graph.as_variable() == Some(name.as_str()))
                && !is_hidden_variable(name)
        })
    }

    fn translate_graph_pattern_unbound(
        &mut self,
        pattern: &ast::GraphPattern,
    ) -> Result<LogicalOperator> {
        match pattern {
            ast::GraphPattern::Basic(triples) => self.translate_basic_pattern(triples),

            ast::GraphPattern::Group(patterns) => {
                let (plan, filters) = self.translate_group(patterns)?;
                Ok(match LogicalExpression::conjunction(filters) {
                    Some(predicate) => wrap_filter(plan, predicate),
                    None => plan,
                })
            }

            ast::GraphPattern::Union(alternatives) => {
                let inputs = alternatives
                    .iter()
                    .map(|p| self.translate_graph_pattern(p))
                    .collect::<Result<Vec<_>>>()?;

                Ok(LogicalOperator::Union(UnionOp { inputs }))
            }

            // These read the group around them, so the parser keeps theirs; alone they are one.
            ast::GraphPattern::Optional(_)
            | ast::GraphPattern::Minus(_)
            | ast::GraphPattern::Filter(_)
            | ast::GraphPattern::Bind { .. } => {
                let (plan, filters) = self.translate_group(std::slice::from_ref(pattern))?;
                Ok(match LogicalExpression::conjunction(filters) {
                    Some(predicate) => wrap_filter(plan, predicate),
                    None => plan,
                })
            }

            ast::GraphPattern::NamedGraph { graph, pattern } => match graph {
                ast::VariableOrIri::Variable(name) => self.translate_graph_variable(pattern, name),
                ast::VariableOrIri::Iri(iri) => self.translate_graph_iri(pattern, iri),
            },

            ast::GraphPattern::SubSelect(subquery) => {
                // Its names are its own: a `?g` in it is not the `GRAPH ?g` around it, whose graph
                // its patterns read under the group's hidden name (the one the whole pattern reads
                // when it names `?g`, and itself in a nested sub-SELECT), and its groups start with
                // no `?g` bound.
                let around = self.graph_context_stack.last().cloned();
                let depth = self.graph_context_stack.len().saturating_sub(1);
                if let Some(TripleComponent::Variable(variable)) = &around
                    && !is_hidden_variable(variable)
                    && let Some(graph) = self.graph_context_stack.last_mut()
                {
                    *graph = TripleComponent::Variable(group_graph_name(variable, depth));
                }
                let unit_around = self.graph_unit.take();
                let plan = self.translate_select(subquery);
                self.graph_unit = unit_around;
                if let Some(around) = around
                    && let Some(graph) = self.graph_context_stack.last_mut()
                {
                    *graph = around;
                }
                Ok(plan?.root)
            }

            ast::GraphPattern::Service { .. } => Err(Error::from(QueryError::new(
                QueryErrorKind::Unsupported,
                "SPARQL SERVICE (federated queries) is not yet supported",
            ))),

            ast::GraphPattern::InlineData(data) => {
                // VALUES clause: each row becomes a chain of BIND operators
                // starting from Empty, and all rows are combined with UNION.
                // No rows: no solution, with the variables defined (an
                // aggregate over them sees an empty group).
                if data.values.is_empty() {
                    let mut plan = LogicalOperator::Empty;
                    for var in &data.variables {
                        plan = LogicalOperator::Bind(BindOp {
                            expression: LogicalExpression::Literal(Value::Null),
                            variable: var.clone(),
                            input: Box::new(plan),
                        });
                    }
                    return Ok(LogicalOperator::Filter(FilterOp {
                        predicate: LogicalExpression::Literal(Value::Bool(false)),
                        input: Box::new(plan),
                        pushdown_hint: None,
                    }));
                }
                let mut branches = Vec::new();
                for row in &data.values {
                    let mut plan = self.unit();
                    for (var, val) in data.variables.iter().zip(row.iter()) {
                        if let Some(dv) = val {
                            plan = LogicalOperator::Bind(BindOp {
                                expression: self.data_value_expression(dv),
                                variable: var.clone(),
                                input: Box::new(plan),
                            });
                        }
                    }
                    branches.push(plan);
                }
                if branches.len() == 1 {
                    Ok(branches
                        .into_iter()
                        .next()
                        .expect("single-element iterator"))
                } else {
                    Ok(LogicalOperator::Union(UnionOp { inputs: branches }))
                }
            }
        }
    }

    fn translate_basic_pattern(
        &mut self,
        triples: &[ast::TriplePattern],
    ) -> Result<LogicalOperator> {
        if triples.is_empty() {
            return Ok(self.unit());
        }

        let mut plan = LogicalOperator::Empty;

        for triple in triples {
            let triple_scan = self.translate_triple_pattern(triple)?;
            plan = self.join_patterns(plan, triple_scan)?;
        }

        Ok(plan)
    }

    fn translate_triple_pattern(&mut self, triple: &ast::TriplePattern) -> Result<LogicalOperator> {
        // Handle Sequence property paths: expand into chained triple patterns
        // e.g. ?person foaf:knows/foaf:name ?name  becomes:
        //   ?person foaf:knows ?_anon0 . ?_anon0 foaf:name ?name
        if let ast::PropertyPath::Sequence(paths) = &triple.predicate {
            let subject = self.translate_triple_term(&triple.subject)?;
            let object = self.translate_triple_term(&triple.object)?;
            let graph = self.graph_context_stack.last().cloned();

            let mut current_subject = subject;
            let mut plan = LogicalOperator::Empty;

            for (i, path) in paths.iter().enumerate() {
                let next_object = if i == paths.len() - 1 {
                    object.clone()
                } else {
                    TripleComponent::Variable(format!("_:seq{}", self.next_anon()))
                };

                let step = if self.is_simple_path(path) {
                    let pred = self.translate_property_path(path)?;
                    self.make_triple_scan(
                        current_subject.clone(),
                        pred,
                        next_object.clone(),
                        graph.clone(),
                    )
                } else {
                    // Complex path (ZeroOrMore, OneOrMore, etc.): recurse
                    let sub_triple = ast::TriplePattern {
                        subject: self.triple_component_to_term(&current_subject)?,
                        predicate: path.clone(),
                        object: self.triple_component_to_term(&next_object)?,
                    };
                    self.translate_triple_pattern(&sub_triple)?
                };

                plan = self.join_patterns(plan, step)?;
                current_subject = next_object;
            }

            return Ok(plan);
        }

        // Handle Alternative property paths: translate as Union of triple scans
        if let ast::PropertyPath::Alternative(alternatives) = &triple.predicate {
            let subject = self.translate_triple_term(&triple.subject)?;
            let object = self.translate_triple_term(&triple.object)?;
            let graph = self.graph_context_stack.last().cloned();

            let mut branches = Vec::new();
            for alt_path in alternatives {
                if self.is_simple_path(alt_path) {
                    let pred = self.translate_property_path(alt_path)?;
                    branches.push(self.make_triple_scan(
                        subject.clone(),
                        pred,
                        object.clone(),
                        graph.clone(),
                    ));
                } else {
                    // An inverse, sequence or repeated branch (`^p`, `p/q`,
                    // `p+`): recurse, as a sequence step does, and keep the
                    // two endpoints, so every branch has the same columns.
                    let sub_triple = ast::TriplePattern {
                        subject: triple.subject.clone(),
                        predicate: alt_path.clone(),
                        object: triple.object.clone(),
                    };
                    let branch = self.translate_triple_pattern(&sub_triple)?;
                    branches.push(self.project_path_endpoints(&subject, &object, branch));
                }
            }

            return Ok(LogicalOperator::Union(UnionOp { inputs: branches }));
        }

        // A quantified step is one search of its endpoints: path+, path* and path?.
        match &triple.predicate {
            ast::PropertyPath::OneOrMore(inner) => {
                return self.translate_path_search(triple, inner, false, true);
            }
            ast::PropertyPath::ZeroOrMore(inner) => {
                return self.translate_path_search(triple, inner, true, true);
            }
            ast::PropertyPath::ZeroOrOne(inner) => {
                return self.translate_path_search(triple, inner, true, false);
            }
            _ => {}
        }

        // Handle Inverse (^path): swap subject and object, translate inner path
        if let ast::PropertyPath::Inverse(inner) = &triple.predicate {
            let swapped = ast::TriplePattern {
                subject: triple.object.clone(),
                predicate: *inner.clone(),
                object: triple.subject.clone(),
            };
            return self.translate_triple_pattern(&swapped);
        }

        // Handle Negation: !(iri1|^iri2) scans all triples and filters out excluded predicates
        if let ast::PropertyPath::Negation(negated_iris) = &triple.predicate {
            return self.translate_negated_property_set(triple, negated_iris);
        }

        let subject = self.translate_triple_term(&triple.subject)?;
        let predicate = self.translate_property_path(&triple.predicate)?;
        let object = self.translate_triple_term(&triple.object)?;

        Ok(self.make_triple_scan(
            subject,
            predicate,
            object,
            self.graph_context_stack.last().cloned(),
        ))
    }

    fn translate_triple_term(&mut self, term: &ast::TripleTerm) -> Result<TripleComponent> {
        match term {
            ast::TripleTerm::Variable(name) => Ok(TripleComponent::Variable(name.clone())),
            ast::TripleTerm::Iri(iri) => Ok(TripleComponent::Iri(self.resolve_iri(iri))),
            ast::TripleTerm::Literal(lit) => Ok(TripleComponent::Literal(self.literal_value(lit))),
            ast::TripleTerm::BlankNode(bnode) => {
                // Treat blank nodes as variables, scoped by query_id
                match bnode {
                    ast::BlankNode::Labeled(label) => Ok(TripleComponent::Variable(format!(
                        "_:q{}_{label}",
                        self.query_id
                    ))),
                    ast::BlankNode::Anonymous(_) => {
                        let anon = self.next_anon();
                        let var = format!("_:q{}_anon{anon}", self.query_id);
                        Ok(TripleComponent::Variable(var))
                    }
                }
            }
        }
    }

    /// Returns true if the property path is a simple predicate (IRI, variable, or rdf:type).
    fn is_simple_path(&self, path: &ast::PropertyPath) -> bool {
        matches!(
            path,
            ast::PropertyPath::Predicate(_)
                | ast::PropertyPath::Variable(_)
                | ast::PropertyPath::RdfType
        )
    }

    /// Converts a `TripleComponent` back to an AST `TripleTerm` for recursive translation.
    ///
    /// # Errors
    ///
    /// For a constant term of a kind the translator does not know.
    fn triple_component_to_term(&self, component: &TripleComponent) -> Result<ast::TripleTerm> {
        Ok(match component {
            TripleComponent::Variable(name) => ast::TripleTerm::Variable(name.clone()),
            TripleComponent::Iri(iri) => ast::TripleTerm::Iri(ast::Iri(iri.clone())),
            TripleComponent::Literal(Value::RdfTerm(term)) => match term.as_ref() {
                Term::Literal(literal) => ast::TripleTerm::Literal(ast::Literal {
                    value: literal.value().to_string(),
                    datatype: literal
                        .language()
                        .is_none()
                        .then(|| ast::Iri(literal.datatype().to_string())),
                    language: literal.language().map(str::to_string),
                }),
                Term::Iri(iri) => ast::TripleTerm::Iri(ast::Iri(iri.as_str().to_string())),
                Term::BlankNode(node) => {
                    ast::TripleTerm::BlankNode(ast::BlankNode::Labeled(node.id().to_string()))
                }
                other => {
                    return Err(grafeo_common::types::rdf::UnsupportedTermKind::of(other).into());
                }
            },
            TripleComponent::Literal(val) => ast::TripleTerm::Literal(ast::Literal {
                value: val.to_string(),
                datatype: None,
                language: None,
            }),
            TripleComponent::LangLiteral { value, lang } => {
                ast::TripleTerm::Literal(ast::Literal {
                    value: value.clone(),
                    datatype: None,
                    language: Some(lang.clone()),
                })
            }
            TripleComponent::BlankNode(label) => {
                ast::TripleTerm::BlankNode(ast::BlankNode::Labeled(label.clone()))
            }
        })
    }

    fn translate_property_path(&mut self, path: &ast::PropertyPath) -> Result<TripleComponent> {
        match path {
            ast::PropertyPath::Predicate(iri) => Ok(TripleComponent::Iri(self.resolve_iri(iri))),
            ast::PropertyPath::Variable(name) => Ok(TripleComponent::Variable(name.clone())),
            ast::PropertyPath::RdfType => Ok(TripleComponent::Iri(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            )),
            // Complex property paths are not fully supported yet
            _ => Err(Error::from(QueryError::new(
                QueryErrorKind::Unsupported,
                "Complex property paths not yet supported",
            ))),
        }
    }

    fn translate_expression(&mut self, expr: &ast::Expression) -> Result<LogicalExpression> {
        match expr {
            ast::Expression::Variable(name) => Ok(LogicalExpression::Variable(name.clone())),

            // A constant is the exact term written: an IRI, or a literal with
            // its lexical form and its datatype or language tag.
            ast::Expression::Iri(iri) => {
                Ok(LogicalExpression::Literal(iri_term(&self.resolve_iri(iri))))
            }
            ast::Expression::Literal(lit) => {
                Ok(LogicalExpression::Literal(self.literal_value(lit)))
            }

            ast::Expression::Binary {
                left,
                operator,
                right,
            } => {
                let left = self.translate_expression(left)?;
                let right = self.translate_expression(right)?;
                let op = self.translate_binary_op(*operator);
                Ok(LogicalExpression::Binary {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                })
            }

            ast::Expression::Unary { operator, operand } => {
                let operand = self.translate_expression(operand)?;
                // Unary plus is a no-op: just return the operand unchanged
                if *operator == ast::UnaryOperator::Plus {
                    return Ok(operand);
                }
                let op = self.translate_unary_op(*operator);
                Ok(LogicalExpression::Unary {
                    op,
                    operand: Box::new(operand),
                })
            }

            ast::Expression::FunctionCall {
                function,
                arguments,
            } => {
                let name = self.translate_function_name(function);
                let args = arguments
                    .iter()
                    .map(|a| self.translate_expression(a))
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::SparqlCall { name, args })
            }

            ast::Expression::Bound(var) => {
                // BOUND(?x) checks if variable is bound
                Ok(LogicalExpression::SparqlCall {
                    name: "BOUND".to_string(),
                    args: vec![LogicalExpression::Variable(var.clone())],
                })
            }

            ast::Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                let cond = self.translate_expression(condition)?;
                let then_expr = self.translate_expression(then_expression)?;
                let else_expr = self.translate_expression(else_expression)?;
                Ok(LogicalExpression::Case {
                    operand: None,
                    when_clauses: vec![(cond, then_expr)],
                    else_clause: Some(Box::new(else_expr)),
                })
            }

            ast::Expression::Coalesce(exprs) => {
                let args = exprs
                    .iter()
                    .map(|e| self.translate_expression(e))
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::SparqlCall {
                    name: "COALESCE".to_string(),
                    args,
                })
            }

            ast::Expression::Exists(pattern) => {
                // EXISTS { pattern } - check if pattern has any matches
                let subquery = self.translate_exists_pattern(pattern)?;
                Ok(LogicalExpression::ExistsSubquery(Box::new(subquery)))
            }

            ast::Expression::NotExists(pattern) => {
                // NOT EXISTS { pattern } - negate the EXISTS check
                let subquery = self.translate_exists_pattern(pattern)?;
                Ok(LogicalExpression::Unary {
                    op: UnaryOp::Not,
                    operand: Box::new(LogicalExpression::ExistsSubquery(Box::new(subquery))),
                })
            }

            ast::Expression::In { expression, list } => {
                let expr = self.translate_expression(expression)?;
                let items = list
                    .iter()
                    .map(|e| self.translate_expression(e))
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::In,
                    right: Box::new(LogicalExpression::List(items)),
                })
            }

            ast::Expression::NotIn { expression, list } => {
                let expr = self.translate_expression(expression)?;
                let items = list
                    .iter()
                    .map(|e| self.translate_expression(e))
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::Unary {
                    op: UnaryOp::Not,
                    operand: Box::new(LogicalExpression::Binary {
                        left: Box::new(expr),
                        op: BinaryOp::In,
                        right: Box::new(LogicalExpression::List(items)),
                    }),
                })
            }

            ast::Expression::Aggregate(agg) => self.translate_aggregate_expression(agg),

            ast::Expression::Bracketed(inner) => self.translate_expression(inner),
        }
    }

    fn translate_aggregate_expression(
        &mut self,
        agg: &ast::AggregateExpression,
    ) -> Result<LogicalExpression> {
        let (func_name, distinct) = match agg {
            ast::AggregateExpression::Count { distinct, .. } => ("COUNT", *distinct),
            ast::AggregateExpression::Sum { distinct, .. } => ("SUM", *distinct),
            ast::AggregateExpression::Average { distinct, .. } => ("AVG", *distinct),
            ast::AggregateExpression::Minimum { .. } => ("MIN", false),
            ast::AggregateExpression::Maximum { .. } => ("MAX", false),
            ast::AggregateExpression::Sample { .. } => ("SAMPLE", false),
            ast::AggregateExpression::GroupConcat { distinct, .. } => ("GROUP_CONCAT", *distinct),
        };

        let args = match agg {
            ast::AggregateExpression::Count { expression, .. } => {
                if let Some(expr) = expression {
                    vec![self.translate_expression(expr)?]
                } else {
                    vec![]
                }
            }
            ast::AggregateExpression::Sum { expression, .. }
            | ast::AggregateExpression::Average { expression, .. }
            | ast::AggregateExpression::Minimum { expression, .. }
            | ast::AggregateExpression::Maximum { expression, .. }
            | ast::AggregateExpression::Sample { expression, .. }
            | ast::AggregateExpression::GroupConcat { expression, .. } => {
                vec![self.translate_expression(expression)?]
            }
        };

        let aggregate = AggregateFunction::from_name(func_name).ok_or_else(|| {
            Error::Internal(format!(
                "SPARQL aggregate {func_name} has no aggregate function"
            ))
        })?;
        Ok(LogicalExpression::FunctionCall {
            callee: Callee::Aggregate(aggregate),
            name: func_name.to_string(),
            args,
            distinct,
        })
    }

    fn translate_binary_op(&self, op: ast::BinaryOperator) -> BinaryOp {
        match op {
            ast::BinaryOperator::Or => BinaryOp::Or,
            ast::BinaryOperator::And => BinaryOp::And,
            ast::BinaryOperator::Equal => BinaryOp::Eq,
            ast::BinaryOperator::NotEqual => BinaryOp::Ne,
            ast::BinaryOperator::LessThan => BinaryOp::Lt,
            ast::BinaryOperator::LessOrEqual => BinaryOp::Le,
            ast::BinaryOperator::GreaterThan => BinaryOp::Gt,
            ast::BinaryOperator::GreaterOrEqual => BinaryOp::Ge,
            ast::BinaryOperator::Add => BinaryOp::Add,
            ast::BinaryOperator::Subtract => BinaryOp::Sub,
            ast::BinaryOperator::Multiply => BinaryOp::Mul,
            ast::BinaryOperator::Divide => BinaryOp::Div,
        }
    }

    fn translate_unary_op(&self, op: ast::UnaryOperator) -> UnaryOp {
        match op {
            ast::UnaryOperator::Not => UnaryOp::Not,
            ast::UnaryOperator::Plus => {
                unreachable!("unary plus is handled as a no-op at the call site")
            }
            ast::UnaryOperator::Minus => UnaryOp::Neg,
        }
    }

    fn translate_function_name(&self, func: &ast::FunctionName) -> String {
        match func {
            ast::FunctionName::BuiltIn(builtin) => format!("{:?}", builtin).to_uppercase(),
            ast::FunctionName::Custom(iri) => self.resolve_iri(iri),
        }
    }

    fn translate_group_condition(
        &mut self,
        cond: &ast::GroupCondition,
    ) -> Result<LogicalExpression> {
        match cond {
            ast::GroupCondition::Variable(name) => Ok(LogicalExpression::Variable(name.clone())),
            ast::GroupCondition::Expression { expression, .. } => {
                self.translate_expression(expression)
            }
            ast::GroupCondition::BuiltInCall(expr) => self.translate_expression(expr),
        }
    }

    fn extract_aggregates_for_select(
        &mut self,
        select: &ast::SelectQuery,
    ) -> Result<(Vec<AggregateExpr>, Vec<LogicalExpression>)> {
        let mut aggregates = Vec::new();
        let mut group_by = Vec::new();

        if let ast::Projection::Variables(vars) = &select.projection {
            for pv in vars {
                if self.is_aggregate_expression(&pv.expression) {
                    if let Some(agg) = self.extract_aggregate(&pv.expression, &pv.alias)? {
                        aggregates.push(agg);
                    }
                } else {
                    group_by.push(self.translate_expression(&pv.expression)?);
                }
            }
        }

        Ok((aggregates, group_by))
    }

    fn is_aggregate_expression(&self, expr: &ast::Expression) -> bool {
        matches!(expr, ast::Expression::Aggregate(_))
    }

    /// Recursively checks if an expression contains any aggregate function.
    fn contains_aggregate(expr: &ast::Expression) -> bool {
        match expr {
            ast::Expression::Aggregate(_) => true,
            ast::Expression::Binary { left, right, .. } => {
                Self::contains_aggregate(left) || Self::contains_aggregate(right)
            }
            ast::Expression::Unary { operand, .. } => Self::contains_aggregate(operand),
            ast::Expression::FunctionCall { arguments, .. } => {
                arguments.iter().any(Self::contains_aggregate)
            }
            ast::Expression::Bracketed(inner) => Self::contains_aggregate(inner),
            ast::Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                Self::contains_aggregate(condition)
                    || Self::contains_aggregate(then_expression)
                    || Self::contains_aggregate(else_expression)
            }
            ast::Expression::Coalesce(exprs) => exprs.iter().any(Self::contains_aggregate),
            ast::Expression::In { expression, list }
            | ast::Expression::NotIn { expression, list } => {
                Self::contains_aggregate(expression) || list.iter().any(Self::contains_aggregate)
            }
            _ => false,
        }
    }

    /// Checks if the SELECT projection contains any aggregate expressions.
    fn has_aggregates_in_projection(projection: &ast::Projection) -> bool {
        match projection {
            ast::Projection::Wildcard => false,
            ast::Projection::Variables(vars) => vars
                .iter()
                .any(|pv| Self::contains_aggregate(&pv.expression)),
        }
    }

    fn extract_aggregate(
        &mut self,
        expr: &ast::Expression,
        alias: &Option<String>,
    ) -> Result<Option<AggregateExpr>> {
        if let ast::Expression::Aggregate(agg) = expr {
            let (func, expr_inner, distinct, separator) = match agg {
                ast::AggregateExpression::Count {
                    distinct,
                    expression,
                } => {
                    // COUNT(?expr) uses CountNonNull to skip NULLs;
                    // COUNT(*) (no expression) uses Count to count all rows.
                    let func = if expression.is_some() {
                        AggregateFunction::CountNonNull
                    } else {
                        AggregateFunction::Count
                    };
                    (
                        func,
                        expression.as_ref().map(|e| e.as_ref()),
                        *distinct,
                        None,
                    )
                }
                ast::AggregateExpression::Sum {
                    distinct,
                    expression,
                } => (
                    AggregateFunction::Sum,
                    Some(expression.as_ref()),
                    *distinct,
                    None,
                ),
                ast::AggregateExpression::Average {
                    distinct,
                    expression,
                } => (
                    AggregateFunction::Avg,
                    Some(expression.as_ref()),
                    *distinct,
                    None,
                ),
                ast::AggregateExpression::Minimum { expression } => (
                    AggregateFunction::Min,
                    Some(expression.as_ref()),
                    false,
                    None,
                ),
                ast::AggregateExpression::Maximum { expression } => (
                    AggregateFunction::Max,
                    Some(expression.as_ref()),
                    false,
                    None,
                ),
                ast::AggregateExpression::Sample { expression } => (
                    AggregateFunction::Sample,
                    Some(expression.as_ref()),
                    false,
                    None,
                ),
                ast::AggregateExpression::GroupConcat {
                    distinct,
                    expression,
                    separator,
                } => (
                    AggregateFunction::GroupConcat,
                    Some(expression.as_ref()),
                    *distinct,
                    separator.clone(),
                ),
            };

            let expression = if let Some(e) = expr_inner {
                Some(self.translate_expression(e)?)
            } else {
                None
            };

            Ok(Some(AggregateExpr {
                function: func,
                expression,
                expression2: None,
                distinct,
                alias: alias.clone(),
                percentile: None, // SPARQL doesn't support percentile functions
                separator,
            }))
        } else {
            Ok(None)
        }
    }

    /// Replaces each aggregate of a HAVING expression, wherever it stands,
    /// by a variable that reads its column: the column of the same aggregate
    /// of `aggregates` (same function, arguments, DISTINCT and separator),
    /// or of one this adds for the HAVING alone.
    fn take_having_aggregates(
        &mut self,
        expr: &mut ast::Expression,
        aggregates: &mut Vec<AggregateExpr>,
    ) -> Result<()> {
        match expr {
            ast::Expression::Aggregate(_) => {
                let Some(mut aggregate) = self.extract_aggregate(expr, &None)? else {
                    return Ok(());
                };
                let key = |aggregate: &AggregateExpr| {
                    format!(
                        "{:?}",
                        (
                            aggregate.function,
                            &aggregate.expression,
                            &aggregate.expression2,
                            aggregate.distinct,
                            &aggregate.separator,
                            aggregate.percentile,
                        )
                    )
                };
                let wanted = key(&aggregate);
                let alias = match aggregates
                    .iter()
                    .find(|existing| existing.alias.is_some() && key(existing) == wanted)
                {
                    Some(existing) => existing.alias.clone().expect("checked above"),
                    None => {
                        let alias = format!("__having_{}", self.next_anon());
                        aggregate.alias = Some(alias.clone());
                        aggregates.push(aggregate);
                        alias
                    }
                };
                *expr = ast::Expression::Variable(alias);
            }
            ast::Expression::Binary { left, right, .. } => {
                self.take_having_aggregates(left, aggregates)?;
                self.take_having_aggregates(right, aggregates)?;
            }
            ast::Expression::Unary { operand, .. } => {
                self.take_having_aggregates(operand, aggregates)?;
            }
            ast::Expression::FunctionCall { arguments, .. } => {
                for argument in arguments {
                    self.take_having_aggregates(argument, aggregates)?;
                }
            }
            ast::Expression::Bracketed(inner) => self.take_having_aggregates(inner, aggregates)?,
            ast::Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                self.take_having_aggregates(condition, aggregates)?;
                self.take_having_aggregates(then_expression, aggregates)?;
                self.take_having_aggregates(else_expression, aggregates)?;
            }
            ast::Expression::Coalesce(expressions) => {
                for expression in expressions {
                    self.take_having_aggregates(expression, aggregates)?;
                }
            }
            ast::Expression::In { expression, list }
            | ast::Expression::NotIn { expression, list } => {
                self.take_having_aggregates(expression, aggregates)?;
                for item in list {
                    self.take_having_aggregates(item, aggregates)?;
                }
            }
            // No aggregate inside: an EXISTS pattern is evaluated per group
            // row, not aggregated.
            ast::Expression::Variable(_)
            | ast::Expression::Iri(_)
            | ast::Expression::Literal(_)
            | ast::Expression::Bound(_)
            | ast::Expression::Exists(_)
            | ast::Expression::NotExists(_) => {}
        }
        Ok(())
    }

    fn join_patterns(
        &self,
        mut left: LogicalOperator,
        mut right: LogicalOperator,
    ) -> Result<LogicalOperator> {
        // The variables of each side are collected recursively: a pattern of
        // many triples joins one at a time, so stop before that is too deep.
        crate::query::limits::check_partial_plan_depth(&mut right)?;
        crate::query::limits::check_partial_plan_depth(&mut left)?;
        if matches!(left, LogicalOperator::Empty) {
            return Ok(right);
        }
        if matches!(right, LogicalOperator::Empty) {
            return Ok(left);
        }

        // Collect output variables from each side and build explicit join
        // conditions for shared variables. This enables the DPccp optimizer
        // to reorder SPARQL triple patterns by selectivity.
        let left_vars = Self::collect_operator_variables(&left);
        let right_vars = Self::collect_operator_variables(&right);

        let mut conditions = Vec::new();
        for var in &left_vars {
            if right_vars.contains(var) {
                conditions.push(JoinCondition {
                    left: LogicalExpression::Variable(var.clone()),
                    right: LogicalExpression::Variable(var.clone()),
                });
            }
        }

        Ok(LogicalOperator::Join(JoinOp {
            left: Box::new(left),
            right: Box::new(right),
            join_type: JoinType::Inner,
            conditions,
        }))
    }

    /// Collects all variable names produced by an operator subtree.
    ///
    /// Walks `TripleScan`, `Join`, `LeftJoin`, `Union`, `Filter`, `Bind`,
    /// and `Project` operators to gather the set of variable names that
    /// appear in the output.
    fn collect_operator_variables(op: &LogicalOperator) -> Vec<String> {
        let mut vars = Vec::new();
        Self::collect_variables_recursive(op, &mut vars);
        vars
    }

    fn collect_variables_recursive(op: &LogicalOperator, vars: &mut Vec<String>) {
        match op {
            LogicalOperator::InNamedGraph(graph) => {
                Self::collect_variables_recursive(&graph.input, vars);
                if let NamedGraph::Variable(variable) = &graph.graph
                    && !vars.contains(variable)
                {
                    vars.push(variable.clone());
                }
            }
            LogicalOperator::NamedGraphs(graphs) => {
                if !vars.contains(&graphs.variable) {
                    vars.push(graphs.variable.clone());
                }
            }
            LogicalOperator::PropertyPath(path) => {
                for component in [Some(&path.subject), Some(&path.object), path.graph.as_ref()]
                    .into_iter()
                    .flatten()
                {
                    if let Some(v) = component.as_variable()
                        && !vars.iter().any(|name| name == v)
                    {
                        vars.push(v.to_string());
                    }
                }
            }
            LogicalOperator::TripleScan(scan) => {
                if let TripleComponent::Variable(v) = &scan.subject
                    && !vars.contains(v)
                {
                    vars.push(v.clone());
                }
                if let TripleComponent::Variable(v) = &scan.predicate
                    && !vars.contains(v)
                {
                    vars.push(v.clone());
                }
                if let TripleComponent::Variable(v) = &scan.object
                    && !vars.contains(v)
                {
                    vars.push(v.clone());
                }
                if let Some(TripleComponent::Variable(v)) = &scan.graph
                    && !vars.contains(v)
                {
                    vars.push(v.clone());
                }
                if let Some(input) = &scan.input {
                    Self::collect_variables_recursive(input, vars);
                }
            }
            LogicalOperator::Join(join) => {
                Self::collect_variables_recursive(&join.left, vars);
                Self::collect_variables_recursive(&join.right, vars);
            }
            LogicalOperator::LeftJoin(join) => {
                Self::collect_variables_recursive(&join.left, vars);
                Self::collect_variables_recursive(&join.right, vars);
            }
            LogicalOperator::Union(union) => {
                // Union outputs variables from both branches
                for input in &union.inputs {
                    Self::collect_variables_recursive(input, vars);
                }
            }
            LogicalOperator::Filter(filter) => {
                Self::collect_variables_recursive(&filter.input, vars);
            }
            LogicalOperator::Bind(bind) => {
                Self::collect_variables_recursive(&bind.input, vars);
                if !vars.contains(&bind.variable) {
                    vars.push(bind.variable.clone());
                }
            }
            LogicalOperator::Project(proj) => {
                Self::collect_variables_recursive(&proj.input, vars);
            }
            _ => {}
        }
    }

    fn resolve_iri(&self, iri: &ast::Iri) -> String {
        let iri_str = iri.as_str();

        // Check if it's a prefixed name
        if let Some(colon_pos) = iri_str.find(':') {
            let prefix = &iri_str[..colon_pos];
            let local = &iri_str[colon_pos + 1..];

            if let Some(namespace) = self.prefixes.get(prefix) {
                return format!("{}{}", namespace, local);
            }
        }

        // Return as-is if no prefix match or already a full IRI
        iri_str.to_string()
    }

    /// The term of a literal as written: its lexical form with its language
    /// tag, its datatype (a prefixed name resolved), or neither (a simple
    /// literal), so `"5"^^xsd:int` stays an `xsd:int` and `"01"` keeps its
    /// text.
    fn literal_value(&self, lit: &ast::Literal) -> Value {
        let term = match (&lit.language, &lit.datatype) {
            (Some(lang), _) => Term::lang_literal(lit.value.as_str(), lang.as_str()),
            (None, Some(datatype)) => {
                Term::typed_literal(lit.value.as_str(), self.resolve_iri(datatype))
            }
            (None, None) => Term::literal(lit.value.as_str()),
        };
        Value::RdfTerm(Arc::new(term))
    }

    /// The expression a VALUES entry binds: the exact term written.
    fn data_value_expression(&self, dv: &ast::DataValue) -> LogicalExpression {
        LogicalExpression::Literal(self.data_value_to_value(dv))
    }

    /// The term of a VALUES entry, as [`literal_value`](Self::literal_value)
    /// reads a literal.
    fn data_value_to_value(&self, dv: &ast::DataValue) -> Value {
        match dv {
            ast::DataValue::Iri(iri) => iri_term(&self.resolve_iri(iri)),
            ast::DataValue::Literal(lit) => self.literal_value(lit),
        }
    }

    /// Translates a negated property set `!(iri1|^iri2)`.
    ///
    /// For forward IRIs: scans `?s ?p ?o` and filters out excluded predicates.
    /// For inverse IRIs: scans `?o ?p ?s` (swapped) and filters out excluded predicates.
    /// Mixed sets produce a `Union` of forward and inverse branches.
    fn translate_negated_property_set(
        &mut self,
        triple: &ast::TriplePattern,
        negated_iris: &[ast::NegatedIri],
    ) -> Result<LogicalOperator> {
        let subject = self.translate_triple_term(&triple.subject)?;
        let object = self.translate_triple_term(&triple.object)?;
        let graph = self.graph_context_stack.last().cloned();

        let forward_iris: Vec<&ast::Iri> = negated_iris
            .iter()
            .filter(|ni| !ni.inverse)
            .map(|ni| &ni.iri)
            .collect();
        let inverse_iris: Vec<&ast::Iri> = negated_iris
            .iter()
            .filter(|ni| ni.inverse)
            .map(|ni| &ni.iri)
            .collect();

        let has_forward = !forward_iris.is_empty() || inverse_iris.is_empty();
        let has_inverse = !inverse_iris.is_empty();

        let build_branch = |translator: &mut Self,
                            subj: TripleComponent,
                            obj: TripleComponent,
                            excluded: &[&ast::Iri]|
         -> Result<LogicalOperator> {
            let pred_var = format!("_:neg_pred{}", translator.next_anon());
            let scan = translator.make_triple_scan(
                subj,
                TripleComponent::Variable(pred_var.clone()),
                obj,
                graph.clone(),
            );

            if excluded.is_empty() {
                return Ok(scan);
            }

            // Build filter: _:neg_pred != iri1 AND _:neg_pred != iri2 AND ...
            let conditions: Vec<LogicalExpression> = excluded
                .iter()
                .map(|iri| LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Variable(pred_var.clone())),
                    op: BinaryOp::Ne,
                    right: Box::new(LogicalExpression::Literal(iri_term(
                        &translator.resolve_iri(iri),
                    ))),
                })
                .collect();

            let predicate = LogicalExpression::conjunction(conditions)
                .expect("excluded non-empty after is_empty check");

            Ok(wrap_filter(scan, predicate))
        };

        if has_forward && has_inverse {
            // Union of forward scan (excluding forward IRIs) and
            // inverse scan with swapped s/o (excluding inverse IRIs)
            let forward_branch =
                build_branch(self, subject.clone(), object.clone(), &forward_iris)?;
            let inverse_branch = build_branch(self, object, subject, &inverse_iris)?;
            Ok(LogicalOperator::Union(UnionOp {
                inputs: vec![forward_branch, inverse_branch],
            }))
        } else if has_inverse {
            // Only inverse exclusions: scan with swapped subject/object
            build_branch(self, object, subject, &inverse_iris)
        } else {
            // Only forward exclusions (most common case)
            build_branch(self, subject, object, &forward_iris)
        }
    }

    /// Plans one native endpoint search of `inner_path`: with its zero-length step or not, and
    /// taking it once or any number of times.
    fn translate_path_search(
        &mut self,
        triple: &ast::TriplePattern,
        inner_path: &ast::PropertyPath,
        zero_length: bool,
        repeats: bool,
    ) -> Result<LogicalOperator> {
        Ok(LogicalOperator::PropertyPath(
            crate::query::plan::PropertyPathOp {
                subject: self.translate_triple_term(&triple.subject)?,
                object: self.translate_triple_term(&triple.object)?,
                path: self.translate_path_step(inner_path)?,
                zero_length,
                repeats,
                graph: self.graph_context_stack.last().cloned(),
                dataset: self.dataset.clone(),
            },
        ))
    }

    fn translate_path_step(
        &self,
        path: &ast::PropertyPath,
    ) -> Result<grafeo_core::execution::operators::PathStep> {
        use grafeo_core::execution::operators::PathStep;
        match path {
            ast::PropertyPath::Predicate(iri) => Ok(PathStep::Predicate(self.resolve_iri(iri))),
            ast::PropertyPath::RdfType => Ok(PathStep::Predicate(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            )),
            ast::PropertyPath::Inverse(inner) => Ok(PathStep::Inverse(Box::new(
                self.translate_path_step(inner)?,
            ))),
            ast::PropertyPath::Sequence(paths) => paths
                .iter()
                .map(|p| self.translate_path_step(p))
                .collect::<Result<Vec<_>>>()
                .map(PathStep::Sequence),
            ast::PropertyPath::Alternative(paths) => paths
                .iter()
                .map(|p| self.translate_path_step(p))
                .collect::<Result<Vec<_>>>()
                .map(PathStep::Alternative),
            _ => Err(Error::from(QueryError::new(
                QueryErrorKind::Unsupported,
                "Quantified property paths support IRIs, inverse, sequence and alternative; nested quantifiers and negated sets are not supported",
            ))),
        }
    }

    /// Wraps a branch of an alternative path in a projection that outputs only the subject and
    /// object variables, stripping companion columns so that all union branches have consistent
    /// column counts.
    fn project_path_endpoints(
        &self,
        subject: &TripleComponent,
        object: &TripleComponent,
        input: LogicalOperator,
    ) -> LogicalOperator {
        let mut projections = Vec::new();
        if let TripleComponent::Variable(s) = subject {
            projections.push(Projection {
                expression: LogicalExpression::Variable(s.clone()),
                alias: Some(s.clone()),
            });
        }
        if let TripleComponent::Variable(o) = object {
            projections.push(Projection {
                expression: LogicalExpression::Variable(o.clone()),
                alias: Some(o.clone()),
            });
        }
        self.keep_the_graph(&mut projections);
        if projections.is_empty() {
            return input;
        }
        LogicalOperator::Project(ProjectOp {
            projections,
            input: Box::new(input),
            pass_through_input: false,
        })
    }

    /// Under `GRAPH ?g` a path's graph is an endpoint too: the scans below bind it, and a branch
    /// without it would give `?g` as null.
    fn keep_the_graph(&self, projections: &mut Vec<Projection>) {
        if let Some(TripleComponent::Variable(g)) = self.graph_context_stack.last()
            && !projections
                .iter()
                .any(|p| p.alias.as_deref() == Some(g.as_str()))
        {
            projections.push(Projection {
                expression: LogicalExpression::Variable(g.clone()),
                alias: Some(g.clone()),
            });
        }
    }

    /// Converts a parsed `DatasetClause` into a `DatasetRestriction` for the logical plan.
    fn translate_dataset_clause(
        &self,
        dataset: &Option<ast::DatasetClause>,
    ) -> Option<DatasetRestriction> {
        let clause = dataset.as_ref()?;

        let default_graphs: Vec<String> = clause
            .default_graphs
            .iter()
            .map(|iri| self.resolve_iri(iri))
            .collect();
        let named_graphs: Vec<String> = clause
            .named_graphs
            .iter()
            .map(|iri| self.resolve_iri(iri))
            .collect();

        // Only create a restriction if at least one FROM or FROM NAMED is present
        if default_graphs.is_empty() && named_graphs.is_empty() {
            return None;
        }

        Some(DatasetRestriction {
            default_graphs,
            named_graphs: DatasetNamedGraphs::Listed(named_graphs),
        })
    }

    /// Creates a `TripleScanOp` with the current graph context and dataset restriction.
    fn make_triple_scan(
        &self,
        subject: TripleComponent,
        predicate: TripleComponent,
        object: TripleComponent,
        graph: Option<TripleComponent>,
    ) -> LogicalOperator {
        LogicalOperator::TripleScan(TripleScanOp {
            subject,
            predicate,
            object,
            graph,
            input: None,
            dataset: self.dataset.clone(),
        })
    }

    fn next_anon(&mut self) -> u32 {
        let n = self.anon_counter;
        self.anon_counter += 1;
        n
    }
}

/// An IRI as the constant of an expression.
fn iri_term(iri: &str) -> Value {
    Value::RdfTerm(Arc::new(Term::iri(iri)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::{SkipOp, SortOp};

    // === Basic SELECT Tests ===

    #[test]
    fn test_translate_simple_select() {
        let query = "SELECT ?x WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_select_with_prefix() {
        let query = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?name
            WHERE { ?x foaf:name ?name }
        "#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_select_wildcard() {
        let query = "SELECT * WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_select_distinct() {
        let query = "SELECT DISTINCT ?x WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_distinct(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Distinct(_) => true,
                LogicalOperator::Project(p) => find_distinct(&p.input),
                _ => false,
            }
        }
        assert!(find_distinct(&plan.root));
    }

    // === Filter Tests ===

    #[test]
    fn test_translate_select_with_filter() {
        let query = "SELECT ?x WHERE { ?x ?y ?z FILTER(?z > 10) }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_filter_equality() {
        let query = r#"SELECT ?x WHERE { ?x ?y ?z FILTER(?z = "test") }"#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_filter_and() {
        let query = "SELECT ?x WHERE { ?x ?y ?z FILTER(?z > 10 && ?z < 100) }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_filter_or() {
        let query = r#"SELECT ?x WHERE { ?x ?y ?z FILTER(?z = 1 || ?z = 2) }"#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_filter_bound() {
        let query = "SELECT ?x WHERE { ?x ?y ?z FILTER(BOUND(?z)) }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    // === ASK Query Tests ===

    #[test]
    fn test_translate_ask() {
        let query = "ASK { ?x ?y ?z }";
        let plan = translate(query).unwrap();
        // One Boolean in `boolean`: whether the count of at most one solution
        // passes zero.
        let LogicalOperator::Project(project) = &plan.root else {
            panic!("ASK projects its answer: {:?}", plan.root);
        };
        assert_eq!(project.projections.len(), 1);
        assert_eq!(project.projections[0].alias.as_deref(), Some("boolean"));
        let LogicalOperator::Aggregate(count) = project.input.as_ref() else {
            panic!("ASK counts the solutions: {:?}", project.input);
        };
        assert!(count.group_by.is_empty());
        let LogicalOperator::Limit(limit) = count.input.as_ref() else {
            panic!("ASK stops at the first solution: {:?}", count.input);
        };
        assert_eq!(limit.count, 1);
    }

    // === Solution Modifiers Tests ===

    #[test]
    fn test_translate_select_with_limit() {
        let query = "SELECT ?x WHERE { ?x ?y ?z } LIMIT 10";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_select_with_offset() {
        let query = "SELECT ?x WHERE { ?x ?y ?z } OFFSET 5";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_skip(op: &LogicalOperator) -> Option<&SkipOp> {
            match op {
                LogicalOperator::Skip(s) => Some(s),
                LogicalOperator::Project(p) => find_skip(&p.input),
                _ => None,
            }
        }
        let skip = find_skip(&plan.root).expect("Expected Skip");
        assert_eq!(skip.count, 5);
    }

    #[test]
    fn test_translate_select_with_order_by() {
        let query = "SELECT ?x WHERE { ?x ?y ?z } ORDER BY ?z";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_sort(op: &LogicalOperator) -> Option<&SortOp> {
            match op {
                LogicalOperator::Sort(s) => Some(s),
                LogicalOperator::Project(p) => find_sort(&p.input),
                _ => None,
            }
        }
        assert!(find_sort(&plan.root).is_some());
    }

    #[test]
    fn test_translate_select_with_order_by_desc() {
        let query = "SELECT ?x WHERE { ?x ?y ?z } ORDER BY DESC(?z)";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_sort(op: &LogicalOperator) -> Option<&SortOp> {
            match op {
                LogicalOperator::Sort(s) => Some(s),
                LogicalOperator::Project(p) => find_sort(&p.input),
                _ => None,
            }
        }
        let sort = find_sort(&plan.root).expect("Expected Sort");
        assert_eq!(sort.keys[0].order, SortOrder::Descending);
    }

    // === Graph Pattern Tests ===

    #[test]
    fn test_translate_union() {
        let query = "SELECT ?x WHERE { { ?x ?y ?z } UNION { ?x ?a ?b } }";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_union(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Union(_) => true,
                LogicalOperator::Project(p) => find_union(&p.input),
                _ => false,
            }
        }
        assert!(find_union(&plan.root));
    }

    #[test]
    fn test_translate_optional() {
        let query = "SELECT ?x ?name WHERE { ?x ?y ?z OPTIONAL { ?x ?p ?name } }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_bind() {
        let query = "SELECT ?x ?doubled WHERE { ?x ?y ?z BIND(?z * 2 AS ?doubled) }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    // === Aggregate Tests ===

    #[test]
    fn test_translate_count() {
        let query = "SELECT (COUNT(?x) AS ?cnt) WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_group_by() {
        let query = "SELECT ?y (COUNT(?x) AS ?cnt) WHERE { ?x ?y ?z } GROUP BY ?y";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_aggregate(op: &LogicalOperator) -> Option<&AggregateOp> {
            match op {
                LogicalOperator::Aggregate(a) => Some(a),
                LogicalOperator::Project(p) => find_aggregate(&p.input),
                _ => None,
            }
        }
        let agg = find_aggregate(&plan.root).expect("Expected Aggregate");
        assert!(!agg.group_by.is_empty());
    }

    // === Expression Tests ===

    #[test]
    fn test_translate_arithmetic_expression() {
        let query = "SELECT (?x + ?y AS ?sum) WHERE { ?x ?p ?y }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_string_function() {
        let query = r#"SELECT ?x WHERE { ?x ?y ?z FILTER(CONTAINS(?z, "test")) }"#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    // === CONSTRUCT and DESCRIBE Tests ===

    #[test]
    fn test_translate_construct() {
        let query = "CONSTRUCT { ?x ?y ?z } WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_describe() {
        let query = "DESCRIBE ?x WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    // === Multiple Triple Patterns ===

    #[test]
    fn test_translate_multiple_triples() {
        let query = "SELECT ?x ?name ?age WHERE { ?x ?y ?name . ?x ?z ?age }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    // === Literal Types ===

    #[test]
    fn test_translate_literal_types() {
        let query = r#"SELECT ?x WHERE { ?x ?y 42 . ?x ?z "hello" . ?x ?w true }"#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    // === Helper Function Tests ===

    #[test]
    fn test_translator_new() {
        let translator = SparqlTranslator::new();
        // It starts with the predefined prefixes (rdf, rdfs, xsd, owl, fn).
        assert_eq!(translator.prefixes.len(), sparql::PREDEFINED_PREFIXES.len());
        for (prefix, namespace) in sparql::PREDEFINED_PREFIXES {
            assert_eq!(
                translator.prefixes.get(prefix).map(String::as_str),
                Some(namespace)
            );
        }
        assert!(translator.base.is_none());
        assert_eq!(translator.anon_counter, 0);
    }

    #[test]
    fn test_translator_next_anon() {
        let mut translator = SparqlTranslator::new();
        assert_eq!(translator.next_anon(), 0);
        assert_eq!(translator.next_anon(), 1);
        assert_eq!(translator.next_anon(), 2);
    }

    // === SPARQL Update Tests ===

    #[test]
    fn test_translate_insert_data() {
        let query = r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "value" }"#;
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn has_insert_triple(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::InsertTriple(_) => true,
                LogicalOperator::Union(u) => u.inputs.iter().any(has_insert_triple),
                _ => false,
            }
        }
        assert!(has_insert_triple(&plan.root));
    }

    #[test]
    fn test_translate_delete_data() {
        let query = r#"DELETE DATA { <http://ex.org/s> <http://ex.org/p> "value" }"#;
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn has_delete_triple(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::DeleteTriple(_) => true,
                LogicalOperator::Union(u) => u.inputs.iter().any(has_delete_triple),
                _ => false,
            }
        }
        assert!(has_delete_triple(&plan.root));
    }

    #[test]
    fn test_translate_delete_where() {
        let query = r#"DELETE WHERE { ?s <http://ex.org/p> ?o }"#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_modify_delete_insert() {
        let query = r#"
            DELETE { ?s <http://ex.org/old> ?o }
            INSERT { ?s <http://ex.org/new> ?o }
            WHERE { ?s <http://ex.org/old> ?o }
        "#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_clear_graph() {
        let query = "CLEAR DEFAULT";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::ClearGraph(_)));
    }

    #[test]
    fn test_translate_drop_graph() {
        let query = "DROP GRAPH <http://example.org/graph>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::DropGraph(_)));
    }

    #[test]
    fn test_translate_create_graph() {
        let query = "CREATE GRAPH <http://example.org/newgraph>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::CreateGraph(_)));
    }

    #[test]
    fn test_translate_copy_graph() {
        let query = "COPY DEFAULT TO <http://example.org/backup>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::CopyGraph(_)));
    }

    #[test]
    fn test_translate_move_graph() {
        let query = "MOVE <http://example.org/old> TO <http://example.org/new>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::MoveGraph(_)));
    }

    #[test]
    fn test_translate_add_graph() {
        let query = "ADD <http://example.org/source> TO <http://example.org/dest>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::AddGraph(_)));
    }

    #[test]
    fn test_translate_load_graph() {
        let query = "LOAD <http://example.org/data.ttl>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::LoadGraph(_)));
    }

    #[test]
    fn test_translate_load_into_graph() {
        let query = "LOAD <http://example.org/data.ttl> INTO GRAPH <http://example.org/target>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::LoadGraph(load) = &plan.root {
            assert!(load.destination.is_some());
        } else {
            panic!("Expected LoadGraph operator");
        }
    }

    #[test]
    fn test_translate_silent_operations() {
        let query = "DROP SILENT GRAPH <http://example.org/graph>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::DropGraph(drop) = &plan.root {
            assert!(drop.silent);
        } else {
            panic!("Expected DropGraph operator");
        }
    }

    // === BIND Expression Tests ===

    #[test]
    fn test_translate_bind_with_concat() {
        let query = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?name ?label
            WHERE {
                ?person foaf:name ?name .
                BIND (CONCAT(?name, " test") AS ?label)
            }
        "#;
        let result = translate(query);
        assert!(
            result.is_ok(),
            "BIND translation failed: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        fn find_bind(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Bind(_) => true,
                LogicalOperator::Project(p) => find_bind(&p.input),
                LogicalOperator::Filter(f) => find_bind(&f.input),
                _ => false,
            }
        }
        assert!(find_bind(&plan.root), "Expected Bind operator in plan");
    }

    // === VALUES Inline Data Tests ===

    #[test]
    fn test_translate_values_inline_data() {
        let query = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?name
            WHERE {
                VALUES ?person { <http://ex.org/alix> }
                ?person foaf:name ?name .
            }
        "#;
        let result = translate(query);
        assert!(
            result.is_ok(),
            "VALUES translation failed: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        // VALUES with a single value produces a Bind chain joined with the
        // triple pattern. Walk the tree and verify we find at least one Bind.
        fn find_bind(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Bind(_) => true,
                LogicalOperator::Project(p) => find_bind(&p.input),
                LogicalOperator::Filter(f) => find_bind(&f.input),
                LogicalOperator::Join(j) => find_bind(&j.left) || find_bind(&j.right),
                LogicalOperator::Union(u) => u.inputs.iter().any(find_bind),
                _ => false,
            }
        }
        assert!(find_bind(&plan.root), "Expected Bind from VALUES clause");
    }

    // === OneOrMore Property Path Tests ===

    #[test]
    fn test_translate_one_or_more_property_path() {
        let plan = translate("SELECT ?s ?o WHERE { ?s <http://ex.org/p>+ ?o }").unwrap();
        fn find_path(op: &LogicalOperator) -> Option<&crate::query::plan::PropertyPathOp> {
            if let LogicalOperator::PropertyPath(path) = op {
                return Some(path);
            }
            op.children().into_iter().find_map(find_path)
        }
        let path = find_path(&plan.root).expect("native repeated property path");
        assert!(!path.zero_length);
        assert!(matches!(&path.subject, TripleComponent::Variable(v) if v == "s"));
        assert!(matches!(&path.object, TripleComponent::Variable(v) if v == "o"));
        assert!(!plan.root.explain_tree().contains("Union"));
    }

    // === ZeroOrMore Property Path Tests ===

    #[test]
    fn test_translate_zero_or_more_property_path() {
        let plan = translate("SELECT ?s ?o WHERE { ?s <http://ex.org/p>* ?o }").unwrap();
        fn find_path(op: &LogicalOperator) -> Option<&crate::query::plan::PropertyPathOp> {
            if let LogicalOperator::PropertyPath(path) = op {
                return Some(path);
            }
            op.children().into_iter().find_map(find_path)
        }
        let path = find_path(&plan.root).expect("native repeated property path");
        assert!(path.zero_length);
        assert!(matches!(&path.subject, TripleComponent::Variable(v) if v == "s"));
        assert!(matches!(&path.object, TripleComponent::Variable(v) if v == "o"));
        assert!(!plan.root.explain_tree().contains("Union"));
    }

    // === Sequence Property Path Tests ===

    #[test]
    fn test_translate_sequence_property_path() {
        let query = r#"
            SELECT ?name
            WHERE {
                ?person <http://ex.org/knows>/<http://ex.org/name> ?name
            }
        "#;
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Sequence path translation failed: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        // Sequence path expands into joined triple scans. Count TripleScan operators.
        fn count_triple_scans(op: &LogicalOperator) -> usize {
            match op {
                LogicalOperator::TripleScan(_) => 1,
                LogicalOperator::Project(p) => count_triple_scans(&p.input),
                LogicalOperator::Filter(f) => count_triple_scans(&f.input),
                LogicalOperator::Join(j) => {
                    count_triple_scans(&j.left) + count_triple_scans(&j.right)
                }
                _ => 0,
            }
        }
        let scan_count = count_triple_scans(&plan.root);
        assert!(
            scan_count >= 2,
            "Sequence path should produce at least 2 joined TripleScans, got {}",
            scan_count
        );
    }

    // === Alternative Property Path Tests ===

    #[test]
    fn test_translate_alternative_property_path() {
        let query = "SELECT ?v WHERE { ?s <http://a>|<http://b> ?v }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Alternative path translation failed: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        // Alternative path produces a Union of TripleScans
        fn find_union(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Union(_) => true,
                LogicalOperator::Project(p) => find_union(&p.input),
                _ => false,
            }
        }
        assert!(
            find_union(&plan.root),
            "Expected Union for alternative property path"
        );

        fn count_union_branches(op: &LogicalOperator) -> Option<usize> {
            match op {
                LogicalOperator::Union(u) => Some(u.inputs.len()),
                LogicalOperator::Project(p) => count_union_branches(&p.input),
                _ => None,
            }
        }
        let branch_count =
            count_union_branches(&plan.root).expect("Expected Union in plan for alternative path");
        assert_eq!(
            branch_count, 2,
            "Alternative path with 2 predicates should have 2 Union branches"
        );
    }

    // === Inverse Property Path Tests ===

    #[test]
    fn test_translate_inverse_property_path() {
        let query = "SELECT ?s WHERE { ?o ^<http://ex.org/knows> ?s }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Inverse path translation failed: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        // Inverse path swaps subject and object, so we should get a TripleScan
        // where the predicate is the inner IRI.
        fn find_triple_scan(op: &LogicalOperator) -> Option<&TripleScanOp> {
            match op {
                LogicalOperator::TripleScan(ts) => Some(ts),
                LogicalOperator::Project(p) => find_triple_scan(&p.input),
                LogicalOperator::Filter(f) => find_triple_scan(&f.input),
                LogicalOperator::Join(j) => {
                    find_triple_scan(&j.left).or_else(|| find_triple_scan(&j.right))
                }
                _ => None,
            }
        }
        let scan = find_triple_scan(&plan.root).expect("Expected TripleScan for inverse path");
        // The predicate should be the IRI (not the inverse wrapper)
        assert!(
            matches!(&scan.predicate, TripleComponent::Iri(_)),
            "Expected IRI predicate in TripleScan after inverse, got {:?}",
            scan.predicate
        );
    }

    // === translate() returns Ok for various path types ===

    #[test]
    fn test_translate_ok_for_named_iri_path() {
        let query = "SELECT ?s ?o WHERE { ?s <http://ex.org/rel> ?o }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Named IRI path should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_translate_ok_for_zero_or_one_path() {
        let query = "SELECT ?s ?o WHERE { ?s <http://ex.org/rel>? ?o }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "ZeroOrOne path should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_translate_ok_for_rdf_type_shorthand() {
        let query = "SELECT ?s WHERE { ?s a <http://ex.org/Person> }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "'a' (rdf:type) shorthand should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_translate_ok_for_negated_property_set() {
        let query = "SELECT ?s ?o WHERE { ?s !<http://ex.org/skip> ?o }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Negated property set should translate: {:?}",
            result.err()
        );
    }

    // === Basic SELECT verification ===

    #[test]
    fn test_translate_basic_select_structure() {
        let query = "SELECT ?x ?y WHERE { ?x <http://ex.org/p> ?y }";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        // Top-level should be a Project
        fn find_project(op: &LogicalOperator) -> Option<&ProjectOp> {
            match op {
                LogicalOperator::Project(p) => Some(p),
                _ => None,
            }
        }
        let project = find_project(&plan.root).expect("Expected Project operator at top level");
        assert_eq!(
            project.projections.len(),
            2,
            "SELECT ?x ?y should produce 2 projections"
        );
    }

    // === OPTIONAL pattern ===

    #[test]
    fn test_translate_optional_produces_left_join() {
        let query = "SELECT ?x ?name WHERE { ?x <http://ex.org/type> ?t OPTIONAL { ?x <http://ex.org/name> ?name } }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "OPTIONAL should translate: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        fn find_left_join(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::LeftJoin(_) => true,
                LogicalOperator::Project(p) => find_left_join(&p.input),
                LogicalOperator::Filter(f) => find_left_join(&f.input),
                _ => false,
            }
        }
        assert!(
            find_left_join(&plan.root),
            "OPTIONAL should produce a LeftJoin operator in the plan"
        );
    }

    // === SERVICE clause tests (federated queries not supported) ===

    #[test]
    fn test_service_clause_returns_explicit_error() {
        let result =
            translate("SELECT ?x WHERE { SERVICE <http://example.org/sparql> { ?x ?p ?o } }");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("SERVICE"),
            "Error should mention SERVICE, got: {err}"
        );
    }

    #[test]
    fn test_service_silent_clause_also_errors() {
        let result = translate(
            "SELECT ?x WHERE { SERVICE SILENT <http://example.org/sparql> { ?x ?p ?o } }",
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("SERVICE"),
            "SILENT SERVICE should also error, got: {err}"
        );
    }

    #[test]
    fn test_service_clause_with_local_patterns_still_errors() {
        // Ensure SERVICE is not silently executed locally
        let result =
            translate("SELECT ?x ?y WHERE { ?x ?p ?y . SERVICE <http://remote/> { ?y ?q ?z } }");
        assert!(result.is_err());
    }

    #[test]
    fn test_query_without_service_still_works() {
        // Regression: ensure normal queries still translate fine
        let result = translate("SELECT ?x ?y WHERE { ?x ?p ?y }");
        assert!(result.is_ok());
    }

    // === SAMPLE aggregate tests (3G verification) ===

    #[test]
    fn test_sample_aggregate_translates() {
        let result = translate(
            "SELECT (SAMPLE(?name) AS ?sampleName) WHERE { ?x <http://example.org/name> ?name }",
        );
        assert!(
            result.is_ok(),
            "SAMPLE aggregate should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_group_concat_with_separator_translates() {
        let result = translate(
            r#"SELECT (GROUP_CONCAT(?name; separator=", ") AS ?names) WHERE { ?x <http://example.org/name> ?name }"#,
        );
        assert!(
            result.is_ok(),
            "GROUP_CONCAT with separator should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_group_concat_without_separator_translates() {
        let result = translate(
            "SELECT (GROUP_CONCAT(?name) AS ?names) WHERE { ?x <http://example.org/name> ?name }",
        );
        assert!(
            result.is_ok(),
            "GROUP_CONCAT without separator should translate: {:?}",
            result.err()
        );
    }

    // === Property path depth tests (3D verification) ===

    #[test]
    fn test_property_path_plus_translates_beyond_10() {
        // Native reachability has no translation-time depth cap.
        let result = translate("SELECT ?x ?y WHERE { ?x <http://example.org/knows>+ ?y }");
        assert!(
            result.is_ok(),
            "Property path + should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_property_path_star_translates() {
        let result = translate("SELECT ?x ?y WHERE { ?x <http://example.org/knows>* ?y }");
        assert!(
            result.is_ok(),
            "Property path * should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_property_path_optional_translates() {
        let result = translate("SELECT ?x ?y WHERE { ?x <http://example.org/knows>? ?y }");
        assert!(
            result.is_ok(),
            "Property path ? should translate: {:?}",
            result.err()
        );
    }
}
