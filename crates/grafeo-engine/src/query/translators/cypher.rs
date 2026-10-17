//! Cypher AST to Logical Plan translator.
//!
//! Translates parsed Cypher queries into the common logical plan representation
//! that can be optimized and executed.

use super::common::{
    CallImports, EdgeOccurrence, GeneratedNames, build_left_join_with_predicates, call_imports,
    check_branch_columns, collect_expression_variables, combine_with_and,
    comma_part_join_variables, comma_part_reads_earlier_rows, different_edges, edges_to_compare,
    expand_subquery_return_star, has_all_labels, is_aggregate_function, is_binary_set_function,
    no_result, optional_join, push_set_property, to_aggregate_function, with_imports,
    wrap_distinct, wrap_filter, wrap_limit, wrap_return, wrap_skip, wrap_sort,
};
use crate::query::plan::{
    AddLabelOp, AggregateExpr, AggregateFunction, AggregateOp, ApplyOp, BinaryOp, CallProcedureOp,
    CountExpr, CreateElement, CreateOp, DeleteEdgeOp, DeleteNodeOp, ExpandDirection, ExpandOp,
    JoinCondition, JoinOp, JoinType, ListPredicateKind, LoadDataFormat, LoadDataOp,
    LogicalExpression, LogicalOperator, LogicalPlan, MapProjectionEntry, MergeOp,
    MergeRelationshipOp, NodeScanOp, ParameterScanOp, PathMode, PathSelection, ProcedureYield,
    ProjectOp, Projection, RemoveLabelOp, ReturnItem, SetPropertyOp, ShortestPathEdgeCondition,
    ShortestPathOp, SortKey, SortOrder, UnaryOp, UnionOp, UnwindOp,
};
use grafeo_adapters::query::cypher::{self, ast};
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

/// Result of translating a Cypher query: either a plan or a schema DDL command.
#[non_exhaustive]
pub enum CypherTranslationResult {
    /// Regular query or mutation, produces a logical plan.
    Plan(LogicalPlan),
    /// Schema DDL (CREATE/DROP INDEX, CREATE/DROP CONSTRAINT).
    SchemaCommand(grafeo_adapters::query::gql::ast::SchemaStatement),
    /// SHOW INDEXES introspection.
    ShowIndexes,
    /// SHOW CONSTRAINTS introspection.
    ShowConstraints,
    /// SHOW CURRENT GRAPH TYPE introspection.
    ShowCurrentGraphType,
}

/// Translates a Cypher query string to a logical plan.
///
/// # Errors
///
/// Returns an error if parsing fails or the query is a schema command
/// that cannot be represented as a logical plan.
pub fn translate(query: &str) -> Result<LogicalPlan> {
    match translate_full(query)? {
        CypherTranslationResult::Plan(plan) => Ok(plan),
        _ => Err(Error::Query(QueryError::new(
            QueryErrorKind::Semantic,
            "Schema commands cannot be translated to a logical plan",
        ))),
    }
}

/// Translates a Cypher query, returning either a plan or a schema command.
///
/// # Errors
///
/// Returns an error if parsing fails or the AST contains unsupported constructs.
pub fn translate_full(query: &str) -> Result<CypherTranslationResult> {
    let statement = cypher::parse(query)?;
    let translator = CypherTranslator::new(query);
    match translator.translate_statement_full(&statement)? {
        CypherTranslationResult::Plan(plan) => Ok(CypherTranslationResult::Plan(
            crate::query::limits::check_plan_depth(plan)?,
        )),
        other => Ok(other),
    }
}

/// Cypher AST to logical plan translator.
struct CypherTranslator {
    /// Variables bound to edges (from MATCH relationship patterns or MERGE relationships).
    /// Used to set `is_edge: true` on `SetPropertyOp` when the SET target is an edge variable.
    edge_variables: RefCell<HashSet<String>>,
    /// The names made up for anonymous elements, none of them one the
    /// statement spells.
    names: GeneratedNames,
    /// Alias-to-output-column-name mapping from the most recent RETURN/WITH clause.
    /// Used by ORDER BY to resolve alias references to actual output column names.
    return_aliases: RefCell<HashMap<String, String>>,
    /// While the clauses of a CALL subquery are translated, the variables of
    /// the outer row it runs for (`None`: not known, or not in a subquery):
    /// the ones its `WITH *` imports.
    call_scope: RefCell<Option<HashSet<String>>>,
    /// What the clause just translated projects, when it is a WITH or a
    /// RETURN: what an ORDER BY right after it may read.
    sort_scope: RefCell<Option<SortScope>>,
    /// The variables the `CALL` body being translated imports, which a
    /// `WITH` in it passes on whether it names them or not.
    call_imports: CallImports,
}

/// What a WITH or RETURN projects, for the ORDER BY that follows it
/// (openCypher 9, ORDER BY): it reads what the projection returns, and the
/// variables of the projection's input as well unless the projection
/// aggregates or is DISTINCT.
struct SortScope {
    /// `WITH` or `RETURN`, for messages.
    clause: &'static str,
    /// Whether the projection aggregates.
    aggregating: bool,
    /// Whether the projection is a DISTINCT WITH, after which only its
    /// columns are in scope. After a RETURN DISTINCT the RETURN's planning
    /// adds a key that reads a dropped variable to the distinct row.
    distinct_with: bool,
    /// The items it projects, or `None` when they are not known here
    /// (`RETURN *`, a `*` whose variables are not known).
    items: Option<Vec<ast::ProjectionItem>>,
    /// For a WITH that projects its items itself (not `WITH *`), neither
    /// aggregating nor DISTINCT: the input variables a sort key reads are
    /// kept for the sort (see [`CypherTranslator::sort_keeping_input`]).
    keeps_input: bool,
}

/// The column of a projection item and the text an ORDER BY key that repeats
/// the item's expression has (the aggregate's own name for an aggregate).
struct ProjectedColumn {
    text: String,
    name: String,
}

impl CypherTranslator {
    fn new(query: &str) -> Self {
        Self {
            edge_variables: RefCell::new(HashSet::new()),
            names: GeneratedNames::new(query),
            return_aliases: RefCell::new(HashMap::new()),
            call_scope: RefCell::new(None),
            sort_scope: RefCell::new(None),
            call_imports: CallImports::default(),
        }
    }

    /// Generates a unique anonymous variable name, one the statement does
    /// not spell: a user variable named `_anon_0` stays the user's.
    fn next_anon_var(&self) -> String {
        self.names.next("_anon_")
    }

    /// Records a variable as an edge variable.
    fn register_edge_variable(&self, variable: &str) {
        self.edge_variables
            .borrow_mut()
            .insert(variable.to_string());
    }

    /// Returns true if the variable was bound to an edge.
    fn is_edge_variable(&self, variable: &str) -> bool {
        self.edge_variables.borrow().contains(variable)
    }

    fn translate_statement(&self, stmt: &ast::Statement) -> Result<LogicalPlan> {
        match stmt {
            ast::Statement::Query(query) => self.translate_query(query),
            ast::Statement::Create(create) => self.translate_create_statement(create),
            ast::Statement::Merge(merge) => self.translate_merge_statement(merge),
            ast::Statement::Delete(_) => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Standalone DELETE requires a preceding MATCH clause. Use: MATCH (n) WHERE ... DELETE n",
            ))),
            ast::Statement::Set(_) => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Standalone SET requires a preceding MATCH clause. Use: MATCH (n) WHERE ... SET n.prop = value",
            ))),
            ast::Statement::Remove(_) => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Standalone REMOVE requires a preceding MATCH clause. Use: MATCH (n) WHERE ... REMOVE n.prop",
            ))),
            ast::Statement::Union { queries, all } => {
                let inputs: Vec<LogicalOperator> = queries
                    .iter()
                    .map(|q| {
                        let plan = self.translate_query(q)?;
                        Ok(plan.root)
                    })
                    .collect::<Result<Vec<_>>>()?;
                check_branch_columns("UNION", &inputs)?;

                let union_op = LogicalOperator::Union(UnionOp { inputs });

                // UNION (not ALL) removes duplicates
                let root = if *all {
                    union_op
                } else {
                    wrap_distinct(union_op)
                };

                Ok(LogicalPlan::new(root))
            }
            ast::Statement::Explain(inner) => {
                let mut plan = self.translate_statement(inner)?;
                plan.explain = true;
                Ok(plan)
            }
            ast::Statement::Profile(inner) => {
                let mut plan = self.translate_statement(inner)?;
                plan.profile = true;
                Ok(plan)
            }
            ast::Statement::Schema(_)
            | ast::Statement::ShowIndexes
            | ast::Statement::ShowConstraints
            | ast::Statement::ShowCurrentGraphType => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Schema commands should be routed through translate_statement_full",
            ))),
        }
    }

    fn translate_statement_full(&self, stmt: &ast::Statement) -> Result<CypherTranslationResult> {
        match stmt {
            ast::Statement::Schema(schema) => {
                Ok(CypherTranslationResult::SchemaCommand(schema.clone()))
            }
            ast::Statement::ShowIndexes => Ok(CypherTranslationResult::ShowIndexes),
            ast::Statement::ShowConstraints => Ok(CypherTranslationResult::ShowConstraints),
            ast::Statement::ShowCurrentGraphType => {
                Ok(CypherTranslationResult::ShowCurrentGraphType)
            }
            other => {
                let plan = self.translate_statement(other)?;
                Ok(CypherTranslationResult::Plan(plan))
            }
        }
    }

    fn translate_query(&self, query: &ast::Query) -> Result<LogicalPlan> {
        // As in Neo4j, the rows of a CALL subquery that returns some are not
        // the result of a query: a RETURN after it says what is.
        if let Some(ast::Clause::CallSubquery { query: inner, .. }) = query.clauses.last()
            && returns_rows(inner)
        {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                concat!(
                    "Query cannot conclude with CALL (must be a RETURN clause, an update clause, ",
                    "a unit subquery call, or a procedure call with no YIELD)"
                ),
            )));
        }
        let mut plan: Option<LogicalOperator> = None;

        for clause in &query.clauses {
            plan = Some(self.translate_clause(clause, plan)?);
        }

        let root = plan.ok_or_else(|| {
            Error::Query(QueryError::new(QueryErrorKind::Semantic, "Empty query"))
        })?;
        // A query that ends with an update has no result (openCypher).
        if query.clauses.last().is_some_and(ends_without_result) {
            return Ok(LogicalPlan::new(no_result(root)));
        }
        Ok(LogicalPlan::new(root))
    }

    fn translate_clause(
        &self,
        clause: &ast::Clause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // What the clause before projected, if it was a WITH or a RETURN: an
        // ORDER BY sorts it. A WITH or RETURN sets it again; any other clause
        // (a CALL subquery whose body ends with a RETURN among them) clears it.
        let sort_scope = self.sort_scope.take();
        let plan = match clause {
            ast::Clause::Match(match_clause) => self.translate_match(match_clause, input),
            ast::Clause::OptionalMatch(match_clause) => {
                self.translate_optional_match(match_clause, input)
            }
            ast::Clause::Where(where_clause) => self.translate_where(where_clause, input),
            ast::Clause::With(with_clause) => self.translate_with(with_clause, input),
            ast::Clause::Return(return_clause) => self.translate_return(return_clause, input),
            ast::Clause::Unwind(unwind_clause) => self.translate_unwind(unwind_clause, input),
            ast::Clause::OrderBy(order_by) => {
                self.translate_order_by(order_by, input, sort_scope.as_ref())
            }
            ast::Clause::Skip(expr) => self.translate_skip(expr, input),
            ast::Clause::Limit(expr) => self.translate_limit(expr, input),
            ast::Clause::Create(create_clause) => {
                self.translate_create_clause(create_clause, input)
            }
            ast::Clause::Merge(merge_clause) => self.translate_merge(merge_clause, input),
            ast::Clause::Delete(delete_clause) => self.translate_delete(delete_clause, input),
            ast::Clause::Set(set_clause) => self.translate_set(set_clause, input),
            ast::Clause::Remove(remove_clause) => self.translate_remove(remove_clause, input),
            ast::Clause::Call(call) => self.translate_call_clause(call, input),
            ast::Clause::CallSubquery {
                query,
                scope,
                unions,
                union_all,
            } => self.translate_call_subquery(query, unions, *union_all, scope.as_deref(), input),
            ast::Clause::ForEach(foreach) => self.translate_foreach(foreach, input),
            ast::Clause::LoadCsv(load_csv) => self.translate_load_csv(load_csv),
        };
        if !matches!(clause, ast::Clause::With(_) | ast::Clause::Return(_)) {
            self.sort_scope.replace(None);
        }
        plan
    }

    fn translate_load_csv(&self, load_csv: &ast::LoadCsvClause) -> Result<LogicalOperator> {
        Ok(LogicalOperator::LoadData(LoadDataOp {
            format: LoadDataFormat::Csv,
            with_headers: load_csv.with_headers,
            path: load_csv.path.clone(),
            variable: load_csv.variable.clone(),
            field_terminator: load_csv.field_terminator,
        }))
    }

    fn translate_call_clause(
        &self,
        call: &ast::CallClause,
        _input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let arguments = call
            .arguments
            .iter()
            .map(|a| self.translate_expression(a))
            .collect::<Result<Vec<_>>>()?;

        let yield_items = call.yield_items.as_ref().map(|items| {
            items
                .iter()
                .map(|item| ProcedureYield {
                    field_name: item.field_name.clone(),
                    alias: item.alias.clone(),
                })
                .collect()
        });

        Ok(LogicalOperator::CallProcedure(CallProcedureOp {
            name: call.procedure_name.clone(),
            arguments,
            yield_items,
        }))
    }

    /// Translates `CALL { subquery }` to an Apply operator.
    ///
    /// The subquery sees the outer variables its variable scope clause names
    /// (`CALL (a, b) { ... }`, all of them for `(*)`, none for `()`), or
    /// without one, the variables its importing `WITH` names. It starts from a
    /// `ParameterScan` of them, and they are recorded in
    /// `ApplyOp.shared_variables` so the planner can wire them through
    /// `ParameterState`. Parts joined by `UNION` each import their own; the
    /// Apply imports all of them, and each part's scan names its own. The
    /// imports stay in scope for the whole part: a later `WITH` passes them
    /// on whether it names them or not (see [`CallImports`]).
    fn translate_call_subquery(
        &self,
        inner: &ast::Query,
        unions: &[ast::Query],
        union_all: bool,
        scope: Option<&[String]>,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let outer_names = match &input {
            Some(outer) => outer.bound_variables(self.call_scope.borrow().as_ref()),
            None => Some(HashSet::new()),
        };
        let mut shared_variables: Vec<String> = Vec::new();
        let mut parts = Vec::with_capacity(1 + unions.len());
        for part in std::iter::once(inner).chain(unions) {
            let (plan, imported) = self.translate_call_subquery_part(
                part,
                scope,
                input.is_some(),
                outer_names.as_ref(),
            )?;
            for name in imported {
                if !shared_variables.contains(&name) {
                    shared_variables.push(name);
                }
            }
            parts.push(plan);
        }
        if shared_variables.iter().any(|name| name == "*") {
            shared_variables = vec!["*".to_string()];
        }
        let subplan = if parts.len() == 1 {
            parts.remove(0)
        } else {
            check_branch_columns("UNION", &parts)?;
            let union = LogicalOperator::Union(UnionOp { inputs: parts });
            if union_all {
                union
            } else {
                wrap_distinct(union)
            }
        };

        // A CALL that comes first runs once, on one empty row. A body
        // without a final RETURN (a unit subquery) runs for its writes and
        // passes each row on once, as it came in (openCypher).
        Ok(LogicalOperator::Apply(ApplyOp {
            input: Box::new(input.unwrap_or(LogicalOperator::Empty)),
            subplan: Box::new(subplan),
            shared_variables,
            optional: false,
            unit: !returns_rows(inner),
        }))
    }

    /// Translates one part of a `CALL` subquery (the whole body, or one side
    /// of a `UNION` in it): its plan and the outer variables it imports.
    fn translate_call_subquery_part(
        &self,
        inner: &ast::Query,
        scope: Option<&[String]>,
        has_input: bool,
        outer_names: Option<&HashSet<String>>,
    ) -> Result<(LogicalOperator, Vec<String>)> {
        let mut shared_variables = Vec::new();
        let mut clauses_iter = inner.clauses.iter();

        match scope {
            // After a scope clause, a WITH is an ordinary WITH. With no outer
            // row, `(*)` imports nothing, and named variables are reported as
            // undefined by the binder.
            Some(names) => {
                if has_input || names.iter().any(|name| name != "*") {
                    shared_variables = names.to_vec();
                }
            }
            // Without one, an importing WITH names what the subquery sees and
            // is replaced by the ParameterScan.
            None => {
                if has_input
                    && let Some(ast::Clause::With(with_clause)) = inner.clauses.first()
                    && let Some(imported) =
                        self.importing_with(with_clause, inner.clauses.get(1))?
                {
                    shared_variables = imported;
                    clauses_iter.next();
                }
            }
        }
        let inner_plan = (!shared_variables.is_empty()).then(|| {
            LogicalOperator::ParameterScan(ParameterScanOp {
                columns: shared_variables.clone(),
            })
        });

        // Translate the remaining inner subquery clauses, which see the outer
        // row's variables through `WITH *`, and keep the imports in scope
        // after a `WITH` that leaves them out
        let enclosing = self.call_scope.replace(outer_names.cloned());
        let imports = call_imports(&shared_variables, outer_names);
        let translated = self.call_imports.within(imports, || {
            let mut translated = Ok(inner_plan);
            for clause in clauses_iter {
                translated =
                    translated.and_then(|plan| self.translate_clause(clause, plan).map(Some));
            }
            translated
        });
        self.call_scope.replace(enclosing);
        let mut inner_plan = translated?.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "CALL subquery requires at least one clause",
            ))
        })?;
        expand_subquery_return_star(&mut inner_plan, outer_names)?;
        Ok((inner_plan, shared_variables))
    }

    /// Reads the first `WITH` of a `CALL` subquery: the outer variables it
    /// imports (`*` for `WITH *`), or `None` when it names no variable and is
    /// an ordinary `WITH`. As in openCypher, an importing `WITH` only lists
    /// variables; an alias, an expression, `WHERE`, `DISTINCT`, or an
    /// `ORDER BY`, `SKIP` or `LIMIT` after it (the `next` clause) is an error
    /// (a second `WITH` can do those).
    fn importing_with(
        &self,
        with_clause: &ast::WithClause,
        next: Option<&ast::Clause>,
    ) -> Result<Option<Vec<String>>> {
        let mut imported = Vec::new();
        let mut names_variables = with_clause.is_wildcard;
        let mut only_names = true;
        if with_clause.is_wildcard {
            imported.push("*".to_string());
        }
        for item in &with_clause.items {
            match &item.expression {
                ast::Expression::Variable(name)
                    if item.alias.as_ref().is_none_or(|alias| alias == name) =>
                {
                    imported.push(name.clone());
                    names_variables = true;
                }
                expression => {
                    only_names = false;
                    let mut variables = HashSet::new();
                    collect_expression_variables(
                        &self.translate_expression(expression)?,
                        &mut variables,
                    );
                    names_variables |= !variables.is_empty();
                }
            }
        }
        if !names_variables {
            return Ok(None);
        }
        let not_allowed = if !only_names {
            Some("Aliasing or expressions are not supported.")
        } else if with_clause.where_clause.is_some() {
            Some("WHERE is not allowed.")
        } else if with_clause.distinct {
            Some("DISTINCT is not allowed.")
        } else {
            match next {
                Some(ast::Clause::OrderBy(_)) => Some("ORDER BY is not allowed."),
                Some(ast::Clause::Skip(_)) => Some("SKIP is not allowed."),
                Some(ast::Clause::Limit(_)) => Some("LIMIT is not allowed."),
                _ => None,
            }
        };
        if let Some(reason) = not_allowed {
            const IMPORTING_WITH: &str =
                "Importing WITH should consist only of simple references to outside variables.";
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!("{IMPORTING_WITH} {reason}"),
            )));
        }
        Ok(Some(imported))
    }

    /// Translates `FOREACH (var IN list | clauses)`: for each row, the list is
    /// unwound and the update clauses run once per item, and the row then goes
    /// on once, as it came in (openCypher). The updates run as a unit Apply
    /// (see [`ApplyOp::unit`]) that imports every variable of the row: an
    /// empty or null list writes nothing and keeps the row, a longer one does
    /// not repeat it, and neither `var` nor what the updates bind is a
    /// variable after the `FOREACH`.
    fn translate_foreach(
        &self,
        foreach: &ast::ForEachClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // As the first clause it runs for the one row a query starts from,
        // which has no variables to import.
        let first = input.is_none();
        let input = input.unwrap_or(LogicalOperator::Empty);

        let list_expr = self.translate_expression(&foreach.list)?;

        // Unwind the list of the row the updates run for into one row per item
        let unwind_input = if first {
            LogicalOperator::Empty
        } else {
            LogicalOperator::ParameterScan(ParameterScanOp {
                columns: vec!["*".to_string()],
            })
        };
        let unwind = LogicalOperator::Unwind(UnwindOp {
            input: Box::new(unwind_input),
            expression: list_expr,
            variable: foreach.variable.clone(),
            ordinality_var: None,
            offset_var: None,
        });

        // Chain the inner update clauses, which see the row's variables (a
        // nested FOREACH or CALL imports them through `*`)
        let outer_names = input.bound_variables(self.call_scope.borrow().as_ref());
        let enclosing = self.call_scope.replace(outer_names);
        let mut body = Ok(unwind);
        for clause in &foreach.clauses {
            body = body.and_then(|plan| self.translate_clause(clause, Some(plan)));
        }
        self.call_scope.replace(enclosing);

        Ok(LogicalOperator::Apply(ApplyOp {
            input: Box::new(input),
            subplan: Box::new(body?),
            shared_variables: if first {
                Vec::new()
            } else {
                vec!["*".to_string()]
            },
            optional: false,
            unit: true,
        }))
    }

    /// Extracts all named variables from a Cypher AST pattern.
    fn pattern_variables(pattern: &ast::Pattern) -> HashSet<String> {
        let mut vars = HashSet::new();
        match pattern {
            ast::Pattern::Node(node) => {
                if let Some(v) = &node.variable {
                    vars.insert(v.clone());
                }
            }
            ast::Pattern::Path(path) => {
                if let Some(v) = &path.start.variable {
                    vars.insert(v.clone());
                }
                for rel in &path.chain {
                    if let Some(v) = &rel.variable {
                        vars.insert(v.clone());
                    }
                    if let Some(v) = &rel.target.variable {
                        vars.insert(v.clone());
                    }
                }
            }
            ast::Pattern::NamedPath { name, pattern, .. } => {
                vars.insert(name.clone());
                vars.extend(Self::pattern_variables(pattern));
            }
        }
        vars
    }

    /// The variable of the node a pattern starts from, if it names one.
    fn pattern_start(pattern: &ast::Pattern) -> Option<&str> {
        match pattern {
            ast::Pattern::Node(node) => node.variable.as_deref(),
            ast::Pattern::Path(path) => path.start.variable.as_deref(),
            ast::Pattern::NamedPath { pattern, .. } => Self::pattern_start(pattern),
        }
    }

    /// Translates the comma-separated patterns of a MATCH (or of an OPTIONAL
    /// MATCH, a subquery's MATCH or a pattern comprehension), creating proper
    /// joins for shared variables instead of cross products, and binding each
    /// relationship once (see [`Self::unique_relationships`]).
    ///
    /// The first pattern receives `input` (to chain with prior clauses like
    /// UNWIND or an earlier MATCH). A later pattern goes on from the rows
    /// before it, or is translated on its own and joined to them on the
    /// variables it shares with them, the input's included (see
    /// [`comma_part_join_variables`]).
    fn translate_comma_patterns(
        &self,
        patterns: &[ast::Pattern],
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        if patterns.is_empty() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Empty MATCH pattern",
            )));
        }
        match self.unique_relationships(patterns) {
            Some((named, unique)) => Ok(wrap_filter(
                self.translate_pattern_parts(&named, input)?,
                unique,
            )),
            None => self.translate_pattern_parts(patterns, input),
        }
    }

    /// openCypher matches relationships isomorphically: a MATCH binds one
    /// relationship at most once, across all its patterns (openCypher 9,
    /// "Uniqueness"; TCK Match3 [15] and [16], Match4 [7]), as GQL's MATCH
    /// DIFFERENT EDGES does. The relationship patterns of `patterns` that may
    /// bind one relationship are compared once all are matched, a
    /// variable-length one by the list of its relationships; the hops of one
    /// variable-length relationship differ already, as its expand follows
    /// trails (see [`Self::translate_relationship`]).
    ///
    /// Returns the patterns with a name for each anonymous relationship that
    /// is compared and the condition, or `None` when no two relationship
    /// patterns can bind one relationship (two of different types never do).
    fn unique_relationships(
        &self,
        patterns: &[ast::Pattern],
    ) -> Option<(Vec<ast::Pattern>, LogicalExpression)> {
        let mut edges: Vec<EdgeOccurrence> = patterns
            .iter()
            .flat_map(relationship_patterns)
            .map(|rel| EdgeOccurrence {
                variable: rel.variable.clone().unwrap_or_default(),
                group: rel.length.is_some(),
                types: rel.types.clone(),
            })
            .collect();
        let groups = edges_to_compare(&edges, false);
        if groups.is_empty() {
            return None;
        }
        let mut named = patterns.to_vec();
        let mut relationships: Vec<&mut ast::RelationshipPattern> = named
            .iter_mut()
            .flat_map(relationship_patterns_mut)
            .collect();
        for &index in groups.iter().flatten() {
            if relationships[index].variable.is_none() {
                let name = self.next_anon_var();
                relationships[index].variable = Some(name.clone());
                edges[index].variable = name;
            }
        }
        let unique = different_edges(&edges, &groups)?;
        Some((named, unique))
    }

    /// Translates comma-separated patterns, creating proper joins for shared
    /// variables instead of cross products (see
    /// [`Self::translate_comma_patterns`]).
    fn translate_pattern_parts(
        &self,
        patterns: &[ast::Pattern],
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // Single pattern: fast path, no join logic needed
        if patterns.len() == 1 {
            return self.translate_pattern(&patterns[0], input);
        }

        // The variables the input binds, those of the outer row for a
        // subquery's `WITH *` included (none when they are not known here)
        let input_vars = input
            .as_ref()
            .and_then(|input| input.bound_variables(self.call_scope.borrow().as_ref()))
            .unwrap_or_default();

        // Multiple patterns: detect shared variables and create joins
        let pattern_vars: Vec<HashSet<String>> =
            patterns.iter().map(Self::pattern_variables).collect();

        let mut plan = self.translate_pattern(&patterns[0], input)?;
        let mut bound_vars = pattern_vars[0].clone();

        for (index, pattern) in patterns.iter().enumerate().skip(1) {
            let current_vars = &pattern_vars[index];
            let shared = comma_part_join_variables(
                current_vars,
                Self::pattern_start(pattern),
                &bound_vars,
                &input_vars,
            );

            // Shared variables: translate independently and inner join,
            // unless the part reads a value of the rows before it
            let right = if shared.is_empty() {
                None
            } else {
                Some(self.translate_pattern(pattern, None)?).filter(|right| {
                    !comma_part_reads_earlier_rows(right, current_vars, &bound_vars, &input_vars)
                })
            };

            if let Some(right) = right {
                let conditions = shared
                    .iter()
                    .map(|var| JoinCondition {
                        left: LogicalExpression::Variable(var.clone()),
                        right: LogicalExpression::Variable(var.clone()),
                    })
                    .collect();
                plan = LogicalOperator::Join(JoinOp {
                    left: Box::new(plan),
                    right: Box::new(right),
                    join_type: JoinType::Inner,
                    conditions,
                });
            } else {
                // Go on from the rows before: a cross product, or an expand
                // from the bound start
                plan = self.translate_pattern(pattern, Some(plan))?;
            }

            bound_vars.extend(current_vars.iter().cloned());
        }

        Ok(plan)
    }

    fn translate_match(
        &self,
        match_clause: &ast::MatchClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        self.translate_comma_patterns(&match_clause.patterns, input)
    }

    fn translate_optional_match(
        &self,
        match_clause: &ast::MatchClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // OPTIONAL MATCH uses LEFT JOIN semantics; one that comes first
        // joins one empty row, so no match is one row of nulls.
        let input = input.unwrap_or(LogicalOperator::Empty);

        // Build the right side with proper shared variable joins
        let right = self.translate_comma_patterns(&match_clause.patterns, None)?;

        Ok(optional_join(
            input,
            right,
            self.call_scope.borrow().as_ref(),
        ))
    }

    fn translate_pattern(
        &self,
        pattern: &ast::Pattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        match pattern {
            ast::Pattern::Node(node_pattern) => self.translate_node_pattern(node_pattern, input),
            ast::Pattern::Path(path_pattern) => self.translate_path_pattern(path_pattern, input),
            ast::Pattern::NamedPath {
                name,
                path_function,
                pattern,
            } => {
                // Check if this is a path function (shortestPath/allShortestPaths)
                if let Some(func) = path_function {
                    self.translate_shortest_path(name, *func, pattern, input)
                } else {
                    // Pass the path alias through to the inner pattern
                    self.translate_pattern_with_alias(pattern, input, Some(name.clone()))
                }
            }
        }
    }

    fn translate_node_pattern(
        &self,
        node: &ast::NodePattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let variable = node
            .variable
            .clone()
            .unwrap_or_else(|| self.next_anon_var());
        let label = node.labels.first().cloned();

        let mut plan = LogicalOperator::NodeScan(NodeScanOp {
            variable: variable.clone(),
            label,
            input: input.map(Box::new),
        });

        // Add hasLabel filters for additional labels (AND semantics).
        // First label is used in NodeScan for scan-time filtering; remaining
        // labels are checked via post-scan Filter.
        if let Some(predicate) = has_all_labels(&variable, extra_labels(node)) {
            plan = wrap_filter(plan, predicate);
        }

        // Add filter for inline properties (e.g., {city: 'NYC'})
        if !node.properties.is_empty() {
            let predicate = self.build_property_predicate(&variable, &node.properties)?;
            plan = wrap_filter(plan, predicate);
        }

        Ok(plan)
    }

    /// Builds a predicate expression for property filters like {name: 'Alix', city: 'NYC'}.
    ///
    /// When a pattern property is null (e.g., `{key: null}`), we emit `IS NULL`
    /// instead of equality, because `null = null` is `null` (falsy) in
    /// three-valued logic but a null property pattern should match absent or
    /// null-valued properties.
    fn build_property_predicate(
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

    fn translate_path_pattern(
        &self,
        path: &ast::PathPattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        self.translate_path_pattern_with_alias(path, input, None)
    }

    fn translate_path_pattern_with_alias(
        &self,
        path: &ast::PathPattern,
        input: Option<LogicalOperator>,
        path_alias: Option<String>,
    ) -> Result<LogicalOperator> {
        let mut plan = self.translate_node_pattern(&path.start, input)?;

        // A path variable on more than one relationship pattern binds the
        // path of all of them, which no single expand has: the hops are bound
        // one by one and the path is put together after them.
        let whole_path_alias = path_alias.clone().filter(|_| path.chain.len() > 1);
        let Some(alias) = whole_path_alias else {
            for rel in &path.chain {
                plan = self
                    .translate_relationship(rel, plan, path_alias.clone(), false)?
                    .0;
            }
            return Ok(plan);
        };

        let source = Self::get_last_variable(&plan)?;
        let mut hops = Vec::with_capacity(path.chain.len());
        for rel in &path.chain {
            // A variable-length hop has a path of its own, read for its
            // nodes and edges
            let segment = rel.length.is_some().then(|| self.next_anon_var());
            let (next, hop) = self.translate_relationship(rel, plan, segment, true)?;
            plan = next;
            hops.push(hop);
        }
        Ok(crate::query::translators::common::bind_whole_path(
            plan, &alias, &source, &hops,
        ))
    }

    /// Translates a pattern with an optional path alias.
    fn translate_pattern_with_alias(
        &self,
        pattern: &ast::Pattern,
        input: Option<LogicalOperator>,
        path_alias: Option<String>,
    ) -> Result<LogicalOperator> {
        match pattern {
            ast::Pattern::Node(node_pattern) => self.translate_node_pattern(node_pattern, input),
            ast::Pattern::Path(path_pattern) => {
                self.translate_path_pattern_with_alias(path_pattern, input, path_alias)
            }
            ast::Pattern::NamedPath {
                name,
                path_function,
                pattern: inner,
            } => {
                // Use the outer path alias if none was passed, otherwise use the inner one
                let alias = path_alias.or_else(|| Some(name.clone()));
                if let Some(func) = path_function {
                    self.translate_shortest_path(name, *func, inner, input)
                } else {
                    self.translate_pattern_with_alias(inner, input, alias)
                }
            }
        }
    }

    fn translate_shortest_path(
        &self,
        path_alias: &str,
        path_function: ast::PathFunction,
        pattern: &ast::Pattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // Extract the path pattern from the inner pattern
        let path = match pattern {
            ast::Pattern::Path(p) => p,
            ast::Pattern::Node(_) => {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    "shortestPath requires a path pattern, not a node",
                )));
            }
            ast::Pattern::NamedPath { pattern: inner, .. } => {
                // Recursively get the path pattern
                if let ast::Pattern::Path(p) = inner.as_ref() {
                    p
                } else {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "shortestPath requires a path pattern",
                    )));
                }
            }
        };

        // One relationship pattern between two node patterns: only the first
        // would be searched, and the nodes between them ignored
        let [rel] = path.chain.as_slice() else {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "shortestPath requires a pattern of exactly one relationship",
            )));
        };
        // The variables the input binds: a relationship variable among them
        // is the one relationship the path may take
        let bound = input
            .as_ref()
            .and_then(|input| input.bound_variables(self.call_scope.borrow().as_ref()))
            .unwrap_or_default();

        // Scan for the source node first
        let source_var = path
            .start
            .variable
            .clone()
            .unwrap_or_else(|| self.next_anon_var());
        let source_label = path.start.labels.first().cloned();

        let mut plan = LogicalOperator::NodeScan(NodeScanOp {
            variable: source_var.clone(),
            label: source_label,
            input: input.map(Box::new),
        });
        if let Some(predicate) = has_all_labels(&source_var, extra_labels(&path.start)) {
            plan = wrap_filter(plan, predicate);
        }

        // Apply property filters on the source node if any
        for (key, value) in &path.start.properties {
            let filter_expr = LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Property {
                    variable: source_var.clone(),
                    property: key.clone(),
                }),
                op: BinaryOp::Eq,
                right: Box::new(self.translate_expression(value)?),
            };
            plan = wrap_filter(plan, filter_expr);
        }

        // The target node of the relationship
        let target_var = rel
            .target
            .variable
            .clone()
            .unwrap_or_else(|| self.next_anon_var());
        let target_label = rel.target.labels.first().cloned();

        // Scan for target node
        plan = LogicalOperator::NodeScan(NodeScanOp {
            variable: target_var.clone(),
            label: target_label,
            input: Some(Box::new(plan)),
        });
        if let Some(predicate) = has_all_labels(&target_var, extra_labels(&rel.target)) {
            plan = wrap_filter(plan, predicate);
        }

        // Apply property filters on the target node if any
        for (key, value) in &rel.target.properties {
            let filter_expr = LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Property {
                    variable: target_var.clone(),
                    property: key.clone(),
                }),
                op: BinaryOp::Eq,
                right: Box::new(self.translate_expression(value)?),
            };
            plan = wrap_filter(plan, filter_expr);
        }

        let direction = match rel.direction {
            ast::Direction::Outgoing => ExpandDirection::Outgoing,
            ast::Direction::Incoming => ExpandDirection::Incoming,
            ast::Direction::Undirected => ExpandDirection::Both,
        };

        let edge_types = rel.types.clone();
        let selection = match path_function {
            ast::PathFunction::AllShortestPaths => PathSelection::ShortestGroups(1),
            ast::PathFunction::ShortestPath => PathSelection::Shortest(1),
        };
        // The path must fit the relationship's length: `[*]` needs at least
        // one hop, and a relationship without `*` is a single hop.
        let (min_hops, max_hops) = hop_bounds(rel);
        let quantified = rel.length.is_some();

        // The relationship variable binds the relationships of each path,
        // unless the input bound it: the path then takes that relationship.
        // Its property map and WHERE hold for every relationship of the path,
        // so the search checks them.
        let bound_edge = rel.variable.as_ref().filter(|name| bound.contains(*name));
        let candidate = match (&rel.variable, bound_edge) {
            (Some(name), None) => name.clone(),
            _ => self.next_anon_var(),
        };
        let mut conditions = Vec::new();
        if let Some(name) = bound_edge {
            conditions.push(LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Id(candidate.clone())),
                op: BinaryOp::Eq,
                right: Box::new(LogicalExpression::Id(name.clone())),
            });
        }
        if !rel.properties.is_empty() {
            conditions.push(self.build_property_predicate(&candidate, &rel.properties)?);
        }
        if let Some(where_expr) = &rel.where_clause {
            conditions.push(self.translate_expression(where_expr)?);
        }
        let edge_condition = conditions
            .into_iter()
            .reduce(|left, right| LogicalExpression::Binary {
                left: Box::new(left),
                op: BinaryOp::And,
                right: Box::new(right),
            })
            .map(|predicate| ShortestPathEdgeCondition {
                variable: candidate,
                predicate,
            });
        let edge_variable = rel.variable.clone().filter(|_| bound_edge.is_none());
        if let Some(name) = &edge_variable {
            self.register_edge_variable(name);
        }

        plan = LogicalOperator::ShortestPath(ShortestPathOp {
            input: Box::new(plan),
            source_var,
            target_var,
            edge_types,
            direction,
            path_alias: path_alias.to_string(),
            selection,
            path_mode: PathMode::Walk,
            binds_target: false,
            min_hops,
            max_hops,
            edge_variable,
            quantified,
            edge_condition,
        });

        Ok(plan)
    }

    /// Translates the relationship pattern `rel` after `input`, with the
    /// path alias `path_alias` on its expand. Returns the plan and the hop it
    /// binds; with `name_edge` an anonymous relationship gets a variable, so
    /// that the hop names its edge.
    fn translate_relationship(
        &self,
        rel: &ast::RelationshipPattern,
        input: LogicalOperator,
        path_alias: Option<String>,
        name_edge: bool,
    ) -> Result<(LogicalOperator, crate::query::translators::common::PathHop)> {
        let from_variable = Self::get_last_variable(&input)?;
        // An edge with a property map needs a variable to filter on, even when
        // the pattern leaves it anonymous: `-[:T {w: 1}]->`, `-[*1..2 {w: 1}]->`.
        let edge_variable = rel
            .variable
            .clone()
            .or_else(|| (!rel.properties.is_empty() || name_edge).then(|| self.next_anon_var()));
        if let Some(ref ev) = edge_variable {
            self.register_edge_variable(ev);
        }
        let edge_variable_for_filter = edge_variable.clone();
        let edge_types = rel.types.clone();
        let to_variable = rel
            .target
            .variable
            .clone()
            .unwrap_or_else(|| self.next_anon_var());

        let direction = match rel.direction {
            ast::Direction::Outgoing => ExpandDirection::Outgoing,
            ast::Direction::Incoming => ExpandDirection::Incoming,
            ast::Direction::Undirected => ExpandDirection::Both,
        };

        let (min_hops, max_hops) = hop_bounds(rel);

        // A property map on a variable-length edge must hold for every hop, so
        // it is checked over the path's edges; that needs a path column.
        let per_hop_properties = rel.length.is_some() && !rel.properties.is_empty();
        let path_alias = if per_hop_properties {
            path_alias.or_else(|| Some(self.next_anon_var()))
        } else {
            path_alias
        };
        let property_path = path_alias.clone();

        // Detect cycle pattern: (s)-[*]->(s) where source == target variable.
        // The expand must use a temporary target, then filter for equality.
        let is_cycle = to_variable == from_variable;
        let expand_target = if is_cycle {
            self.next_anon_var()
        } else {
            to_variable.clone()
        };

        let expand = LogicalOperator::Expand(ExpandOp {
            quantified: rel.length.is_some(),
            from_variable,
            to_variable: expand_target.clone(),
            edge_variable,
            direction,
            edge_types,
            min_hops,
            max_hops,
            input: Box::new(input),
            path_alias,
            // A variable-length relationship takes a relationship once
            // (openCypher 9, relationship uniqueness): its hops form a
            // trail, which ends on its own without the cap of a walk
            path_mode: if rel.length.is_some() {
                PathMode::Trail
            } else {
                PathMode::Walk
            },
        });

        // For cycle patterns, enforce that expanded target == original source
        let expand = if is_cycle {
            wrap_filter(
                expand,
                LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::FunctionCall {
                        name: "id".into(),
                        args: vec![LogicalExpression::Variable(expand_target)],
                        distinct: false,
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(LogicalExpression::FunctionCall {
                        name: "id".into(),
                        args: vec![LogicalExpression::Variable(to_variable.clone())],
                        distinct: false,
                    }),
                },
            )
        } else {
            expand
        };

        let mut result = match has_all_labels(&to_variable, &rel.target.labels) {
            Some(predicate) => wrap_filter(expand, predicate),
            None => expand,
        };

        // Apply property filters on the edge: -[r {since: 2020}]->
        if !rel.properties.is_empty()
            && let Some(ref ev) = edge_variable_for_filter
        {
            let predicate = match property_path.clone().filter(|_| per_hop_properties) {
                // `all(e IN edges(path) WHERE e.k = v ...)`: the edge column of a
                // variable-length expand only holds the last hop.
                Some(path) => {
                    let hop = self.next_anon_var();
                    crate::query::translators::common::every_edge_matches(
                        path,
                        hop.clone(),
                        self.build_property_predicate(&hop, &rel.properties)?,
                    )
                }
                None => self.build_property_predicate(ev, &rel.properties)?,
            };
            result = wrap_filter(result, predicate);
        }

        // Apply inline WHERE clause from relationship pattern: -[r WHERE expr]->
        if let Some(where_expr) = &rel.where_clause {
            let predicate = self.translate_expression(where_expr)?;
            result = wrap_filter(result, predicate);
        }

        // Apply property filters on the target node: ()-[r]->(o {id: "X"})
        if !rel.target.properties.is_empty() {
            let predicate = self.build_property_predicate(&to_variable, &rel.target.properties)?;
            result = wrap_filter(result, predicate);
        }

        let hop = crate::query::translators::common::PathHop {
            edge: edge_variable_for_filter.unwrap_or_default(),
            target: to_variable,
            segment: property_path.filter(|_| rel.length.is_some()),
        };
        Ok((result, hop))
    }

    fn translate_where(
        &self,
        where_clause: &ast::WhereClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "WHERE requires input",
            ))
        })?;
        let predicate = self.translate_expression(&where_clause.predicate)?;

        // When the input is a LeftJoin (from OPTIONAL MATCH), classify the
        // predicate so right-side references become join conditions rather
        // than post-filters (which would incorrectly eliminate NULL rows).
        if let LogicalOperator::LeftJoin(left_join) = input {
            let (join, post_filter) = build_left_join_with_predicates(
                left_join,
                Some(predicate),
                self.call_scope.borrow().as_ref(),
            );
            if let Some(pf) = post_filter {
                Ok(wrap_filter(join, pf))
            } else {
                Ok(join)
            }
        } else {
            Ok(wrap_filter(input, predicate))
        }
    }

    fn translate_with(
        &self,
        with_clause: &ast::WithClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // WITH can work with or without prior input (e.g., standalone WITH [1,2,3] AS nums)
        // If there's no input, use Empty which produces a single row for projection evaluation
        let input = input.unwrap_or(LogicalOperator::Empty);

        if with_clause.items.iter().any(|item| {
            item.alias.is_none() && !matches!(item.expression, ast::Expression::Variable(_))
        }) {
            return Err(super::common::unaliased_with_expression());
        }

        // In a CALL body, the imports this WITH leaves out pass on too.
        let projected: Vec<(String, bool)> = with_clause
            .items
            .iter()
            .filter_map(|item| match (&item.alias, &item.expression) {
                (Some(alias), ast::Expression::Variable(name)) => {
                    Some((alias.clone(), alias == name))
                }
                (Some(alias), _) => Some((alias.clone(), false)),
                (None, ast::Expression::Variable(name)) => Some((name.clone(), true)),
                (None, _) => None,
            })
            .collect();
        let missing = self.call_imports.left_out(&projected);

        // Check if WITH contains aggregate functions (e.g. WITH collect(n) AS people)
        let has_aggregates = with_clause
            .items
            .iter()
            .any(|item| contains_aggregate(&item.expression));

        // WITH *: all variables pass through unchanged, with the items after
        // the `*` added. With an aggregate the `*` names the grouping keys.
        let star_items;
        let items = if !with_clause.is_wildcard {
            &with_clause.items
        } else if has_aggregates {
            star_items = self.star_items("WITH", &with_clause.items, &input)?;
            &star_items
        } else {
            let mut plan = if with_clause.items.is_empty() {
                input
            } else {
                self.project_after_star("WITH", &with_clause.items, input)?
            };

            if let Some(where_clause) = &with_clause.where_clause {
                let predicate = self.translate_expression(&where_clause.predicate)?;
                plan = wrap_filter(plan, predicate);
            }

            if with_clause.distinct {
                plan = wrap_distinct(plan);
            }

            self.sort_scope.replace(Some(SortScope {
                clause: "WITH",
                aggregating: false,
                distinct_with: with_clause.distinct,
                items: None,
                keeps_input: false,
            }));
            return Ok(plan);
        };

        let mut plan = if has_aggregates {
            let (mut aggregates, mut group_by, post_return) =
                self.extract_aggregates_and_groups_from_items(items)?;
            let input =
                self.lift_aggregate_pattern_comprehensions(input, &mut aggregates, &mut group_by)?;

            let agg_op = LogicalOperator::Aggregate(AggregateOp {
                group_by,
                aggregates,
                input: Box::new(input),
                having: None,
            });

            if let Some(return_items) = post_return {
                let projections = return_items
                    .into_iter()
                    .map(|item| Projection {
                        expression: item.expression,
                        alias: item.alias,
                    })
                    .collect();
                LogicalOperator::Project(ProjectOp {
                    projections,
                    input: Box::new(agg_op),
                    pass_through_input: false,
                })
            } else {
                agg_op
            }
        } else {
            let projections: Vec<Projection> = items
                .iter()
                .map(|item| {
                    Ok(Projection {
                        expression: self.translate_expression(&item.expression)?,
                        alias: item.alias.clone(),
                    })
                })
                .collect::<Result<_>>()?;

            // Rewrite pattern comprehensions into Apply + Aggregate(Collect):
            // the ones inside an expression here, the item ones below.
            let mut projections = projections;
            let input = self.lift_nested_pattern_comprehensions(
                input,
                projections.iter_mut().map(|p| &mut p.expression),
            )?;
            let has_pattern_comp = projections.iter().any(|p| {
                matches!(
                    &p.expression,
                    LogicalExpression::PatternComprehension { .. }
                )
            });
            let (input, projections) = if has_pattern_comp {
                let items: Vec<ReturnItem> = projections
                    .into_iter()
                    .map(|p| ReturnItem {
                        expression: p.expression,
                        alias: p.alias,
                    })
                    .collect();
                let (rewritten_input, rewritten_items) =
                    self.rewrite_pattern_comprehensions(input, items)?;
                let projections = rewritten_items
                    .into_iter()
                    .map(|item| Projection {
                        expression: item.expression,
                        alias: item.alias,
                    })
                    .collect();
                (rewritten_input, projections)
            } else {
                (input, projections)
            };

            LogicalOperator::Project(ProjectOp {
                projections,
                input: Box::new(input),
                pass_through_input: false,
            })
        };
        plan = with_imports(plan, missing, has_aggregates);

        if let Some(where_clause) = &with_clause.where_clause {
            let predicate = self.translate_expression(&where_clause.predicate)?;
            plan = wrap_filter(plan, predicate);
        }

        if with_clause.distinct {
            plan = wrap_distinct(plan);
        }

        self.sort_scope.replace(Some(SortScope {
            clause: "WITH",
            aggregating: has_aggregates,
            distinct_with: with_clause.distinct,
            items: Some(items.clone()),
            keeps_input: !has_aggregates && !with_clause.distinct,
        }));
        Ok(plan)
    }

    /// The items of a projection `*` followed by `items` that aggregates
    /// (`WITH *, count(*) AS c`): the variables of `input` in name order,
    /// then `items`. `clause` is `WITH` or `RETURN`, for messages.
    fn star_items(
        &self,
        clause: &str,
        items: &[ast::ProjectionItem],
        input: &LogicalOperator,
    ) -> Result<Vec<ast::ProjectionItem>> {
        let Some(bound) = input.bound_variables(self.call_scope.borrow().as_ref()) else {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!(
                    "{clause} * with an aggregate cannot tell which variables are in scope \
                     here: name them"
                ),
            )));
        };
        let mut names: Vec<String> = bound
            .iter()
            .filter(|name| !name.starts_with('_'))
            .cloned()
            .collect();
        names.sort();
        for item in items {
            star_item_name(clause, item, Some(&bound))?;
        }
        Ok(names
            .into_iter()
            .map(|name| ast::ProjectionItem {
                expression: ast::Expression::Variable(name),
                alias: None,
                span: None,
            })
            .chain(items.iter().cloned())
            .collect())
    }

    /// `*` followed by `items` in a projection that does not aggregate
    /// (`WITH *, r.years AS y`): a projection that passes every column of
    /// `input` on and adds the items. `clause` is `WITH` or `RETURN`.
    fn project_after_star(
        &self,
        clause: &str,
        items: &[ast::ProjectionItem],
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        let bound = input.bound_variables(self.call_scope.borrow().as_ref());
        let mut projections: Vec<Projection> = Vec::with_capacity(items.len());
        for item in items {
            let expression = self.translate_expression(&item.expression)?;
            let name = star_item_name(clause, item, bound.as_ref())?.unwrap_or_else(|| {
                crate::query::planner::common::expression_to_string(&expression)
            });
            if projections
                .iter()
                .any(|projection| projection.alias.as_deref() == Some(name.as_str()))
            {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!("{clause} *, ...: the column {name} is already one of the items"),
                )));
            }
            projections.push(Projection {
                expression,
                alias: Some(name),
            });
        }
        // Pattern comprehensions collect into columns of their own, which
        // the projection passes on with the others
        let mut lifted = Vec::new();
        for projection in &mut projections {
            self.take_pattern_comprehensions(&mut projection.expression, &mut lifted);
        }
        let input = if lifted.is_empty() {
            input
        } else {
            self.rewrite_pattern_comprehensions(input, lifted)?.0
        };
        Ok(LogicalOperator::Project(ProjectOp {
            projections,
            input: Box::new(input),
            pass_through_input: true,
        }))
    }

    fn translate_unwind(
        &self,
        unwind_clause: &ast::UnwindClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // UNWIND can work with or without prior input
        // If there's no input, create an implicit single row (Empty with one result)
        let input = input.unwrap_or(LogicalOperator::Empty);

        let expression = self.translate_expression(&unwind_clause.expression)?;

        Ok(LogicalOperator::Unwind(UnwindOp {
            expression,
            variable: unwind_clause.variable.clone(),
            ordinality_var: None,
            offset_var: None,
            input: Box::new(input),
        }))
    }

    fn translate_merge_statement(&self, merge: &ast::MergeClause) -> Result<LogicalPlan> {
        let op = self.translate_merge(merge, None)?;
        Ok(LogicalPlan::new(no_result(op)))
    }

    fn translate_merge(
        &self,
        merge_clause: &ast::MergeClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.unwrap_or(LogicalOperator::Empty);
        let pattern = &merge_clause.pattern;

        // Check if this is a relationship (path) pattern
        let path = match pattern {
            ast::Pattern::Path(path) if !path.chain.is_empty() => Some(path),
            ast::Pattern::NamedPath { pattern: inner, .. } => match inner.as_ref() {
                ast::Pattern::Path(path) if !path.chain.is_empty() => Some(path),
                _ => None,
            },
            _ => None,
        };

        if let Some(path) = path {
            return self.translate_merge_relationship(path, merge_clause, input);
        }

        // Node-only MERGE
        let node = match pattern {
            ast::Pattern::Node(n) => n,
            ast::Pattern::Path(path) => &path.start,
            ast::Pattern::NamedPath { pattern: inner, .. } => match inner.as_ref() {
                ast::Pattern::Node(n) => n,
                ast::Pattern::Path(path) => &path.start,
                _ => {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "MERGE NamedPath must contain a node or path",
                    )));
                }
            },
        };

        let variable = node
            .variable
            .clone()
            .unwrap_or_else(|| self.next_anon_var());
        let labels: Vec<String> = node.labels.clone();

        let match_properties: Vec<(String, LogicalExpression)> = node
            .properties
            .iter()
            .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
            .collect::<Result<Vec<_>>>()?;

        let (on_create, on_create_labels) = self.merge_actions(
            merge_clause.on_create.as_ref(),
            "ON CREATE",
            &variable,
            true,
        )?;
        let (on_match, on_match_labels) =
            self.merge_actions(merge_clause.on_match.as_ref(), "ON MATCH", &variable, true)?;

        Ok(LogicalOperator::Merge(MergeOp {
            variable,
            labels,
            match_properties,
            on_create,
            on_match,
            on_create_labels,
            on_match_labels,
            input: Box::new(input),
        }))
    }

    fn translate_merge_relationship(
        &self,
        path: &ast::PathPattern,
        merge_clause: &ast::MergeClause,
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        // One relationship: a longer pattern would have to be matched or
        // created as a whole (openCypher 9, MERGE), and merging its first
        // relationship alone dropped the rest
        let rel = match path.chain.as_slice() {
            [rel] => rel,
            [] => {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    "MERGE relationship pattern is empty",
                )));
            }
            _ => return Err(super::common::merge_of_a_longer_path()),
        };

        // The source node, merged first when the pattern defines it
        let (source_variable, current_input) = self.merge_end_node(&path.start, input)?;

        // Extract relationship variable
        let variable = rel.variable.clone().unwrap_or_else(|| self.next_anon_var());
        self.register_edge_variable(&variable);

        // Extract relationship type
        let edge_type = rel.types.first().cloned().ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "MERGE relationship pattern requires a relationship type",
            ))
        })?;

        // The target node, merged next when the pattern defines it
        let (target_variable, current_input) = self.merge_end_node(&rel.target, current_input)?;

        // Extract relationship properties
        let match_properties: Vec<(String, LogicalExpression)> = rel
            .properties
            .iter()
            .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
            .collect::<Result<Vec<_>>>()?;

        let (on_create, _) = self.merge_actions(
            merge_clause.on_create.as_ref(),
            "ON CREATE",
            &variable,
            false,
        )?;
        let (on_match, _) =
            self.merge_actions(merge_clause.on_match.as_ref(), "ON MATCH", &variable, false)?;

        // `(a)<-[:T]-(b)` is a relationship from b to a.
        let (source_variable, target_variable) = if rel.direction == ast::Direction::Incoming {
            (target_variable, source_variable)
        } else {
            (source_variable, target_variable)
        };
        Ok(LogicalOperator::MergeRelationship(MergeRelationshipOp {
            variable,
            source_variable,
            target_variable,
            undirected: rel.direction == ast::Direction::Undirected,
            edge_type,
            match_properties,
            on_create,
            on_match,
            input: Box::new(current_input),
        }))
    }

    /// The variable of an end node of a MERGE relationship pattern and the
    /// plan that binds it (see [`super::common::merge_end_node`]): as in
    /// `MERGE (h)-[:R]->(:T {x: 3})`, a node the pattern gives a label or
    /// properties is merged on its own first.
    fn merge_end_node(
        &self,
        node: &ast::NodePattern,
        input: LogicalOperator,
    ) -> Result<(String, LogicalOperator)> {
        let match_properties = node
            .properties
            .iter()
            .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
            .collect::<Result<Vec<_>>>()?;
        super::common::merge_end_node(
            node.variable.as_deref(),
            &node.labels,
            match_properties,
            input,
            || self.next_anon_var(),
        )
    }

    /// The SET items of an `ON CREATE` or `ON MATCH` (`clause`) of a MERGE
    /// that binds `variable`: the properties they set and the labels they add
    /// (`takes_labels` is false for a relationship, which has none). The
    /// MERGE applies them to the element it binds only, so an item it cannot
    /// apply is an error instead of a write to the wrong element or none.
    fn merge_actions(
        &self,
        set_clause: Option<&ast::SetClause>,
        clause: &str,
        variable: &str,
        takes_labels: bool,
    ) -> Result<(Vec<(String, LogicalExpression)>, Vec<String>)> {
        let unsupported = |message: String| {
            Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!("MERGE ... {clause} SET {message}"),
            )))
        };
        let mut properties = Vec::new();
        let mut labels: Vec<String> = Vec::new();
        for item in set_clause.map_or(&[][..], |set| &set.items) {
            let (ast::SetItem::Property {
                variable: item_variable,
                ..
            }
            | ast::SetItem::AllProperties {
                variable: item_variable,
                ..
            }
            | ast::SetItem::MergeProperties {
                variable: item_variable,
                ..
            }
            | ast::SetItem::Labels {
                variable: item_variable,
                ..
            }) = item;
            if item_variable != variable {
                return unsupported(format!(
                    "sets {item_variable}, but can only set the element the MERGE binds"
                ));
            }
            match item {
                ast::SetItem::Property {
                    property, value, ..
                } => {
                    properties.push((property.clone(), self.translate_expression(value)?));
                }
                // `n = {...}` and `n += {...}` with a map literal set its keys.
                ast::SetItem::AllProperties {
                    properties: prop_expr,
                    ..
                }
                | ast::SetItem::MergeProperties {
                    properties: prop_expr,
                    ..
                } => {
                    let ast::Expression::Map(pairs) = prop_expr else {
                        return unsupported(format!(
                            "{item_variable} = ... and {item_variable} += ... need a map literal \
                             here, such as {{name: 'Alix'}}"
                        ));
                    };
                    for (k, v) in pairs {
                        properties.push((k.clone(), self.translate_expression(v)?));
                    }
                }
                ast::SetItem::Labels { labels: added, .. } => {
                    if !takes_labels {
                        return unsupported(format!(
                            "{item_variable}:{}: a relationship has no labels",
                            added.join(":")
                        ));
                    }
                    for label in added {
                        if !labels.contains(label) {
                            labels.push(label.clone());
                        }
                    }
                }
            }
        }
        Ok((properties, labels))
    }

    fn translate_return(
        &self,
        return_clause: &ast::ReturnClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // Standalone RETURN (e.g. RETURN 2 * 3) uses Empty as a single-row source
        let input = input.unwrap_or(LogicalOperator::Empty);

        // `RETURN *, items`: the items are added to what `*` returns, or
        // with an aggregate, `*` names the grouping keys
        let star_items;
        let (input, items) = match &return_clause.items {
            ast::ReturnItems::All => (input, None),
            ast::ReturnItems::Explicit(items) => (input, Some(items.as_slice())),
            ast::ReturnItems::AllAnd(items)
                if items
                    .iter()
                    .any(|item| contains_aggregate(&item.expression)) =>
            {
                star_items = self.star_items("RETURN", items, &input)?;
                (input, Some(star_items.as_slice()))
            }
            ast::ReturnItems::AllAnd(items) => {
                (self.project_after_star("RETURN", items, input)?, None)
            }
        };
        let aggregating = items.is_some_and(|items| {
            items
                .iter()
                .any(|item| contains_aggregate(&item.expression))
        });
        let plan = self.translate_return_items(items, return_clause.distinct, input)?;
        self.sort_scope.replace(Some(SortScope {
            clause: "RETURN",
            aggregating,
            distinct_with: false,
            items: items.map(<[ast::ProjectionItem]>::to_vec),
            keeps_input: false,
        }));
        Ok(plan)
    }

    /// Translates the items of a RETURN (`None` for `*`) after `input`.
    fn translate_return_items(
        &self,
        items: Option<&[ast::ProjectionItem]>,
        distinct: bool,
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        // Record alias-to-output-column mappings for ORDER BY alias resolution.
        // For non-aggregate RETURN, output columns use aliases directly.
        // For aggregate RETURN without post_return, group-by columns use
        // expression_to_string names, and aggregate columns use their aliases.
        self.return_aliases.borrow_mut().clear();

        // Check if RETURN contains aggregate functions
        let aggregate_items = items.filter(|items| {
            items
                .iter()
                .any(|item| contains_aggregate(&item.expression))
        });

        if let Some(items) = aggregate_items {
            // Extract aggregates and group-by expressions.
            // When a return item wraps an aggregate in a binary/unary expression
            // (e.g. `count(n) > 0 AS exists`), we decompose it into:
            //   1. An aggregate (`count(n)` with synthetic alias)
            //   2. A post-aggregate projection (`_agg_0 > 0 AS exists`)
            // With aliases (e.g. `n.city AS city`) the post-Return renames the
            // columns, which ORDER BY alias resolution and result naming need.
            let (mut aggregates, mut group_by, post_return) =
                self.extract_aggregates_and_groups_from_items(items)?;
            let input =
                self.lift_aggregate_pattern_comprehensions(input, &mut aggregates, &mut group_by)?;

            // Register aggregate output column names so ORDER BY can
            // reference them. Group-by columns use expression_to_string
            // format (e.g. "o.status"), aggregate columns use their alias.
            {
                let mut aliases = self.return_aliases.borrow_mut();
                for gb in &group_by {
                    let col = crate::query::planner::common::expression_to_string(gb);
                    aliases.insert(col.clone(), col);
                }
                for agg in &aggregates {
                    if let Some(ref alias) = agg.alias {
                        aliases.insert(alias.clone(), alias.clone());
                    }
                }
            }

            let agg_op = LogicalOperator::Aggregate(AggregateOp {
                group_by,
                aggregates,
                input: Box::new(input),
                having: None,
            });

            if let Some(return_items) = post_return {
                // Post-projection renames columns using aliases.
                // Register alias -> alias (identity) so ORDER BY resolves
                // directly since the Return outputs alias names.
                {
                    let mut aliases = self.return_aliases.borrow_mut();
                    for ri in &return_items {
                        if let Some(ref alias) = ri.alias {
                            let a = alias.clone();
                            aliases.insert(a.clone(), a);
                        }
                    }
                }
                Ok(wrap_return(agg_op, return_items, distinct))
            } else {
                Ok(agg_op)
            }
        } else {
            // Normal return without aggregates
            let items = match items {
                None => {
                    vec![ReturnItem {
                        expression: LogicalExpression::Variable("*".into()),
                        alias: None,
                    }]
                }
                Some(items) => items
                    .iter()
                    .map(|item| {
                        Ok(ReturnItem {
                            expression: self.translate_expression(&item.expression)?,
                            alias: item.alias.clone(),
                        })
                    })
                    .collect::<Result<_>>()?,
            };

            // Rewrite pattern comprehensions into Apply + Aggregate(Collect):
            // the ones inside an expression here, the item ones below.
            let mut items = items;
            let input = self.lift_nested_pattern_comprehensions(
                input,
                items.iter_mut().map(|item| &mut item.expression),
            )?;
            let has_pattern_comp = items.iter().any(|item| {
                matches!(
                    &item.expression,
                    LogicalExpression::PatternComprehension { .. }
                )
            });
            if has_pattern_comp {
                let (rewritten_input, rewritten_items) =
                    self.rewrite_pattern_comprehensions(input, items)?;
                Ok(wrap_return(rewritten_input, rewritten_items, distinct))
            } else {
                Ok(wrap_return(input, items, distinct))
            }
        }
    }

    /// Extracts aggregate and group-by expressions from RETURN items.
    ///
    /// Returns `(aggregates, group_by, post_return)`. The Aggregate operator
    /// outputs its grouping keys first, then its aggregates, named by alias or
    /// [`aggregate_column_name`](crate::query::planner::common::aggregate_column_name).
    /// `post_return` is `Some(...)` whenever that output differs from what the
    /// items ask for: an item wraps an aggregate (`count(n) > 0 AS exists`),
    /// an item has an alias, or a grouping key follows an aggregate (so
    /// columns must be reordered). A post-aggregate projection must then be
    /// chained.
    fn extract_aggregates_and_groups_from_items(
        &self,
        items: &[ast::ProjectionItem],
    ) -> Result<(
        Vec<AggregateExpr>,
        Vec<LogicalExpression>,
        Option<Vec<ReturnItem>>,
    )> {
        let mut aggregates = Vec::new();
        let mut group_by = Vec::new();
        let mut needs_post_return = false;
        let mut post_return_items = Vec::new();
        let mut agg_counter: u32 = 0;
        let mut seen_aggregate = false;

        for item in items {
            if let Some(mut agg_expr) = self.try_extract_aggregate(&item.expression, &item.alias)? {
                // Direct aggregate (e.g. `count(n) AS cnt`). An unaliased one is
                // named after its source text (`count(n)`); the name doubles as
                // the alias so the binder resolves the post-Return reference.
                let column = agg_expr.alias.clone().unwrap_or_else(|| {
                    crate::query::planner::common::aggregate_column_name(&agg_expr)
                });
                agg_expr.alias = Some(column.clone());
                aggregates.push(agg_expr);
                post_return_items.push(ReturnItem {
                    expression: LogicalExpression::Variable(column),
                    alias: item.alias.clone(),
                });
                seen_aggregate = true;
            } else if contains_aggregate(&item.expression) {
                // Wrapped aggregate (e.g. `count(n) > 0 AS exists`)
                needs_post_return = true;
                seen_aggregate = true;

                // Extract all aggregates and build a substitute expression
                // with variable references replacing each aggregate.
                let substitute = self.extract_wrapped_aggregates(
                    &item.expression,
                    &mut agg_counter,
                    &mut aggregates,
                )?;
                // Unaliased, the column is named after the written expression
                // (`sum(n.v) + 1`), not the substitute (`_agg_0 + 1`).
                let alias = item.alias.clone().or_else(|| {
                    self.translate_expression(&item.expression)
                        .ok()
                        .map(|expr| crate::query::planner::common::expression_to_string(&expr))
                });
                post_return_items.push(ReturnItem {
                    expression: substitute,
                    alias,
                });
            } else {
                // Non-aggregate expression: group-by key. Keys come first in
                // the Aggregate's output, so a key after an aggregate needs the
                // post-return to restore the item order.
                needs_post_return |= seen_aggregate;
                let expr = self.translate_expression(&item.expression)?;
                group_by.push(expr.clone());
                // In the post-return, reference the Aggregate's output column
                // by its generated name. The Aggregate already extracts
                // property values, so we reference the column, not re-evaluate.
                let col_name = crate::query::planner::common::expression_to_string(&expr);
                post_return_items.push(ReturnItem {
                    expression: LogicalExpression::Variable(col_name),
                    alias: item.alias.clone(),
                });
            }
        }

        let has_aliases = items.iter().any(|item| item.alias.is_some());
        if needs_post_return || has_aliases {
            Ok((aggregates, group_by, Some(post_return_items)))
        } else {
            Ok((aggregates, group_by, None))
        }
    }

    /// Extracts all aggregates from inside a wrapping expression, assigning
    /// each a synthetic alias (`_agg_{counter}`), and returns the expression
    /// with every aggregate replaced by a variable reference to its alias.
    ///
    /// For `count(n) > 0`:
    /// - pushes `count(n)` with alias `_agg_0` into `aggregates_out`
    /// - returns `Variable("_agg_0") > Literal(0)`
    ///
    /// For `sum(x) / count(x)`:
    /// - pushes `sum(x)` as `_agg_0` and `count(x)` as `_agg_1`
    /// - returns `Variable("_agg_0") / Variable("_agg_1")`
    fn extract_wrapped_aggregates(
        &self,
        expr: &ast::Expression,
        agg_counter: &mut u32,
        aggregates_out: &mut Vec<AggregateExpr>,
    ) -> Result<LogicalExpression> {
        match expr {
            ast::Expression::FunctionCall { name, args, .. } => {
                // Check if the function itself is an aggregate. Its column
                // gets a name the statement does not spell (a grouping key
                // named `_agg_0` stays the user's).
                while self.names.is_written(&format!("_agg_{agg_counter}")) {
                    *agg_counter += 1;
                }
                let alias = format!("_agg_{agg_counter}");
                if let Some(agg) = self.try_extract_aggregate(expr, &Some(alias.clone()))? {
                    *agg_counter += 1;
                    aggregates_out.push(agg);
                    return Ok(LogicalExpression::Variable(alias));
                }
                // Non-aggregate function wrapping aggregate arguments,
                // e.g. size(collect(DISTINCT n.v)). Recurse into each argument.
                let mut translated_args = Vec::with_capacity(args.len());
                for arg in args {
                    if contains_aggregate(arg) {
                        translated_args.push(self.extract_wrapped_aggregates(
                            arg,
                            agg_counter,
                            aggregates_out,
                        )?);
                    } else {
                        translated_args.push(self.translate_expression(arg)?);
                    }
                }
                Ok(LogicalExpression::FunctionCall {
                    name: name.clone(),
                    args: translated_args,
                    distinct: false,
                })
            }
            ast::Expression::Binary { left, op, right } => {
                let binary_op = self.translate_binary_op(*op)?;
                let left_sub = if contains_aggregate(left) {
                    self.extract_wrapped_aggregates(left, agg_counter, aggregates_out)?
                } else {
                    self.translate_expression(left)?
                };
                let right_sub = if contains_aggregate(right) {
                    self.extract_wrapped_aggregates(right, agg_counter, aggregates_out)?
                } else {
                    self.translate_expression(right)?
                };
                Ok(LogicalExpression::Binary {
                    left: Box::new(left_sub),
                    op: binary_op,
                    right: Box::new(right_sub),
                })
            }
            ast::Expression::Unary { op, operand } => {
                let sub = self.extract_wrapped_aggregates(operand, agg_counter, aggregates_out)?;
                // Unary positive is identity: just return the operand
                if *op == ast::UnaryOp::Pos {
                    return Ok(sub);
                }
                let unary_op = self.translate_unary_op(*op)?;
                Ok(LogicalExpression::Unary {
                    op: unary_op,
                    operand: Box::new(sub),
                })
            }
            ast::Expression::Case {
                input,
                whens,
                else_clause,
            } => {
                let operand = match input {
                    Some(inp) if contains_aggregate(inp) => Some(Box::new(
                        self.extract_wrapped_aggregates(inp, agg_counter, aggregates_out)?,
                    )),
                    Some(inp) => Some(Box::new(self.translate_expression(inp)?)),
                    None => None,
                };
                let mut when_clauses = Vec::with_capacity(whens.len());
                for (cond, then) in whens {
                    let cond_expr = if contains_aggregate(cond) {
                        self.extract_wrapped_aggregates(cond, agg_counter, aggregates_out)?
                    } else {
                        self.translate_expression(cond)?
                    };
                    let then_expr = if contains_aggregate(then) {
                        self.extract_wrapped_aggregates(then, agg_counter, aggregates_out)?
                    } else {
                        self.translate_expression(then)?
                    };
                    when_clauses.push((cond_expr, then_expr));
                }
                let else_expr = match else_clause {
                    Some(el) if contains_aggregate(el) => Some(Box::new(
                        self.extract_wrapped_aggregates(el, agg_counter, aggregates_out)?,
                    )),
                    Some(el) => Some(Box::new(self.translate_expression(el)?)),
                    None => None,
                };
                Ok(LogicalExpression::Case {
                    operand,
                    when_clauses,
                    else_clause: else_expr,
                })
            }
            _ => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Unsupported expression wrapping an aggregate",
            ))),
        }
    }

    /// Tries to extract an aggregate expression from an AST expression.
    fn try_extract_aggregate(
        &self,
        expr: &ast::Expression,
        alias: &Option<String>,
    ) -> Result<Option<AggregateExpr>> {
        match expr {
            ast::Expression::FunctionCall {
                name,
                args,
                distinct,
            } => {
                if let Some(function) = to_aggregate_function(name) {
                    // count(*) is represented as FunctionCall with Variable("*") arg
                    let is_count_star = function == AggregateFunction::Count
                        && args.len() == 1
                        && matches!(&args[0], ast::Expression::Variable(v) if v == "*");
                    let expression = if args.is_empty() || is_count_star {
                        None
                    } else {
                        Some(self.translate_expression(&args[0])?)
                    };
                    // Extract percentile parameter for percentile functions
                    let percentile = if matches!(
                        function,
                        AggregateFunction::PercentileDisc | AggregateFunction::PercentileCont
                    ) && args.len() >= 2
                    {
                        // Second argument is the percentile value
                        if let ast::Expression::Literal(ast::Literal::Float(p)) = &args[1] {
                            Some((*p).clamp(0.0, 1.0))
                        } else if let ast::Expression::Literal(ast::Literal::Integer(p)) = &args[1]
                        {
                            Some((*p as f64).clamp(0.0, 1.0))
                        } else {
                            Some(0.5) // Default to median
                        }
                    } else {
                        None
                    };
                    // The binary set functions (covar_samp(y, x), regr_slope(y, x),
                    // ...) read their independent value from the second argument.
                    let expression2 = if is_binary_set_function(function) && args.len() >= 2 {
                        Some(self.translate_expression(&args[1])?)
                    } else {
                        None
                    };
                    // listagg(x, s) and group_concat(x, s) join with `s`; without
                    // one, listagg joins with a comma and group_concat with a
                    // space, as in GQL and SQL/PGQ.
                    let separator = if function == AggregateFunction::GroupConcat {
                        match args.get(1) {
                            Some(ast::Expression::Literal(ast::Literal::String(separator))) => {
                                Some(separator.clone())
                            }
                            _ if name.eq_ignore_ascii_case("listagg") => Some(",".to_string()),
                            _ => None,
                        }
                    } else {
                        None
                    };

                    // COUNT(expr) uses CountNonNull to skip NULLs;
                    // COUNT(*) uses Count to count all rows.
                    let function = if function == AggregateFunction::Count
                        && !is_count_star
                        && expression.is_some()
                    {
                        AggregateFunction::CountNonNull
                    } else {
                        function
                    };

                    Ok(Some(AggregateExpr {
                        function,
                        expression,
                        expression2,
                        distinct: *distinct,
                        alias: alias.clone(),
                        percentile,
                        separator,
                    }))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    /// Translates the ORDER BY after `input`, the rows of the WITH or RETURN
    /// that `scope` describes (`None` when no projection comes right before).
    /// As in openCypher 9 (ORDER BY), a key reads what the projection
    /// returns, and the variables of its input too unless the projection
    /// aggregates or is DISTINCT; an aggregate in a key needs an aggregating
    /// projection (TCK ReturnOrderBy2 [14], WithOrderBy2 [25]), and repeating
    /// a projected expression reads its column.
    fn translate_order_by(
        &self,
        order_by: &ast::OrderByClause,
        input: Option<LogicalOperator>,
        scope: Option<&SortScope>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "ORDER BY requires input",
            ))
        })?;

        if let Some(scope) = scope
            && !scope.aggregating
            && order_by
                .items
                .iter()
                .any(|item| contains_aggregate(&item.expression))
        {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!(
                    "Invalid use of an aggregating function in ORDER BY: the {} it sorts does \
                     not aggregate",
                    scope.clause
                ),
            )));
        }

        // After an aggregating projection or a DISTINCT WITH only its columns
        // are in scope (TCK WithOrderBy4 [13], [14])
        let projected = match scope {
            Some(scope) if scope.aggregating || scope.distinct_with => scope
                .items
                .as_deref()
                .map(|items| self.projected_columns(items))
                .transpose()?,
            _ => None,
        };

        let keys: Vec<SortKey> = order_by
            .items
            .iter()
            .map(|item| {
                let expression = match (scope, &projected) {
                    (Some(scope), Some(columns)) => {
                        let resolved =
                            self.translate_projected_sort_key(&item.expression, columns, scope)?;
                        check_reads_projection(&resolved, columns, scope)?;
                        resolved
                    }
                    _ => self.translate_sort_key(&item.expression)?,
                };
                Ok(SortKey {
                    expression,
                    order: match item.direction {
                        ast::SortDirection::Asc => SortOrder::Ascending,
                        ast::SortDirection::Desc => SortOrder::Descending,
                    },
                    nulls: None,
                })
            })
            .collect::<Result<_>>()?;

        if let Some(scope) = scope
            && scope.keeps_input
            && let Some(items) = &scope.items
        {
            return Ok(Self::sort_keeping_input(input, keys, items));
        }
        Ok(wrap_sort(input, keys))
    }

    /// Translates an ORDER BY key, reading a column of the RETURN before it
    /// for a variable or property its aliases or grouping keys name.
    fn translate_sort_key(&self, expression: &ast::Expression) -> Result<LogicalExpression> {
        let aliases = self.return_aliases.borrow();
        // Resolve alias references: if ORDER BY uses a variable
        // that matches a RETURN alias, substitute with the actual
        // output column name from the preceding RETURN/Aggregate.
        if let ast::Expression::Variable(name) = expression
            && let Some(col_name) = aliases.get(name)
        {
            return Ok(LogicalExpression::Variable(col_name.clone()));
        }
        // After aggregation, entity variables (o, c, d) no longer
        // exist. Rewrite o.status to Variable("o.status") which
        // matches the aggregate output column name.
        if let ast::Expression::PropertyAccess { base, property } = expression
            && let ast::Expression::Variable(var) = base.as_ref()
        {
            let col_dot = format!("{var}.{property}");
            if aliases.contains_key(&col_dot) {
                return Ok(LogicalExpression::Variable(col_dot));
            }
        }
        drop(aliases);
        self.translate_expression(expression)
    }

    /// The columns of a projection's `items` (see [`ProjectedColumn`]).
    fn projected_columns(&self, items: &[ast::ProjectionItem]) -> Result<Vec<ProjectedColumn>> {
        items
            .iter()
            .map(|item| {
                let text = self.projection_text(&item.expression)?;
                let name = item.alias.clone().unwrap_or_else(|| text.clone());
                Ok(ProjectedColumn { text, name })
            })
            .collect()
    }

    /// The text of a projected expression: the name of its column when it
    /// has no alias (an aggregate's own name, `count(DISTINCT n)`).
    fn projection_text(&self, expression: &ast::Expression) -> Result<String> {
        Ok(match self.try_extract_aggregate(expression, &None)? {
            Some(aggregate) => crate::query::planner::common::aggregate_column_name(&aggregate),
            None => crate::query::planner::common::expression_to_string(
                &self.translate_expression(expression)?,
            ),
        })
    }

    /// Translates an ORDER BY key after an aggregating or DISTINCT
    /// projection: an expression the projection returns, whole or inside
    /// the key (`max(a.age) + 1`), reads its column. An aggregate it does
    /// not return is an error: its groups are gone.
    fn translate_projected_sort_key(
        &self,
        expression: &ast::Expression,
        columns: &[ProjectedColumn],
        scope: &SortScope,
    ) -> Result<LogicalExpression> {
        let text = self.projection_text(expression)?;
        if let Some(column) = columns.iter().find(|column| column.text == text) {
            return Ok(LogicalExpression::Variable(column.name.clone()));
        }
        match expression {
            ast::Expression::Binary { left, op, right } => Ok(LogicalExpression::Binary {
                left: Box::new(self.translate_projected_sort_key(left, columns, scope)?),
                op: self.translate_binary_op(*op)?,
                right: Box::new(self.translate_projected_sort_key(right, columns, scope)?),
            }),
            ast::Expression::Unary { op, operand } => {
                let operand = self.translate_projected_sort_key(operand, columns, scope)?;
                if *op == ast::UnaryOp::Pos {
                    return Ok(operand);
                }
                Ok(LogicalExpression::Unary {
                    op: self.translate_unary_op(*op)?,
                    operand: Box::new(operand),
                })
            }
            ast::Expression::FunctionCall { name, args, .. }
                if !is_aggregate_function(name) && args.iter().any(contains_aggregate) =>
            {
                Ok(LogicalExpression::FunctionCall {
                    name: name.clone(),
                    args: args
                        .iter()
                        .map(|arg| self.translate_projected_sort_key(arg, columns, scope))
                        .collect::<Result<_>>()?,
                    distinct: false,
                })
            }
            other if contains_aggregate(other) => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!(
                    "Invalid use of an aggregating function in ORDER BY: {text} is not an \
                     aggregate the {} returns",
                    scope.clause
                ),
            ))),
            other => self.translate_expression(other),
        }
    }

    /// Sorts the rows of a WITH by `keys` that read variables of its input
    /// it does not project (`WITH a.name AS name ORDER BY a.age`): the
    /// projection keeps those variables for the sort, and a projection of
    /// its own `items` after the sort drops them again. A WITH of another
    /// shape sorts as it is, and planning reports the variables it lacks.
    fn sort_keeping_input(
        plan: LogicalOperator,
        keys: Vec<SortKey>,
        items: &[ast::ProjectionItem],
    ) -> LogicalOperator {
        let projected: Vec<String> = items
            .iter()
            .filter_map(|item| match (&item.alias, &item.expression) {
                (Some(alias), _) => Some(alias.clone()),
                (None, ast::Expression::Variable(name)) => Some(name.clone()),
                (None, _) => None,
            })
            .collect();
        let mut read = HashSet::new();
        for key in &keys {
            collect_expression_variables(&key.expression, &mut read);
        }
        let mut kept: Vec<String> = read
            .into_iter()
            .filter(|name| !projected.contains(name))
            .collect();
        if kept.is_empty() {
            return wrap_sort(plan, keys);
        }
        kept.sort();
        let (plan, kept_them) = keep_columns(plan, &kept);
        if !kept_them {
            return wrap_sort(plan, keys);
        }
        LogicalOperator::Project(ProjectOp {
            projections: projected
                .into_iter()
                .map(|name| Projection {
                    expression: LogicalExpression::Variable(name),
                    alias: None,
                })
                .collect(),
            input: Box::new(wrap_sort(plan, keys)),
            pass_through_input: false,
        })
    }

    fn translate_skip(
        &self,
        expr: &ast::Expression,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "SKIP requires input",
            ))
        })?;
        let count = self.eval_as_count_expr(expr)?;

        Ok(wrap_skip(input, count))
    }

    fn translate_limit(
        &self,
        expr: &ast::Expression,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "LIMIT requires input",
            ))
        })?;
        let count = self.eval_as_count_expr(expr)?;

        Ok(wrap_limit(input, count))
    }

    fn translate_create_clause(
        &self,
        create_clause: &ast::CreateClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let elements = self.create_elements(&create_clause.patterns)?;
        if elements.is_empty() {
            return input.ok_or_else(|| {
                Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    "Empty CREATE pattern",
                ))
            });
        }
        Ok(LogicalOperator::Create(CreateOp {
            elements,
            input: input.map(Box::new),
        }))
    }

    /// The nodes and edges `patterns` create, in order: one [`CreateOp`] for
    /// the clause, however many patterns it has.
    fn create_elements(&self, patterns: &[ast::Pattern]) -> Result<Vec<CreateElement>> {
        let mut elements = Vec::new();
        for pattern in patterns {
            self.push_create_pattern(pattern, &mut elements)?;
        }
        Ok(elements)
    }

    fn push_create_pattern(
        &self,
        pattern: &ast::Pattern,
        elements: &mut Vec<CreateElement>,
    ) -> Result<()> {
        match pattern {
            ast::Pattern::Node(node) => {
                self.push_created_node(node, elements)?;
            }
            ast::Pattern::Path(path) => {
                // The node the next relationship of the chain starts at.
                let mut previous_variable = self.push_created_node(&path.start, elements)?;
                for rel in &path.chain {
                    if rel.direction == ast::Direction::Undirected {
                        return Err(Error::Query(QueryError::new(
                            QueryErrorKind::Semantic,
                            "CREATE needs the direction of each relationship: write \
                             (a)-[:TYPE]->(b) or (a)<-[:TYPE]-(b)",
                        )));
                    }
                    let target_variable = self.push_created_node(&rel.target, elements)?;
                    let edge_type = rel
                        .types
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "RELATED".to_string());
                    // `(a)<-[:T]-(b)` is a relationship from b to a.
                    let (from_variable, to_variable) = if rel.direction == ast::Direction::Incoming
                    {
                        (target_variable.clone(), previous_variable)
                    } else {
                        (previous_variable, target_variable.clone())
                    };
                    elements.push(CreateElement::Edge {
                        variable: rel.variable.clone(),
                        from_variable,
                        to_variable,
                        edge_type,
                        properties: self.create_properties(&rel.properties)?,
                    });
                    previous_variable = target_variable;
                }
            }
            ast::Pattern::NamedPath { pattern, .. } => {
                self.push_create_pattern(pattern, elements)?;
            }
        }
        Ok(())
    }

    /// Adds the node `node` creates, and returns its variable.
    fn push_created_node(
        &self,
        node: &ast::NodePattern,
        elements: &mut Vec<CreateElement>,
    ) -> Result<String> {
        let variable = node
            .variable
            .clone()
            .unwrap_or_else(|| self.next_anon_var());
        elements.push(CreateElement::Node {
            variable: variable.clone(),
            labels: node.labels.clone(),
            properties: self.create_properties(&node.properties)?,
        });
        Ok(variable)
    }

    fn create_properties(
        &self,
        properties: &[(String, ast::Expression)],
    ) -> Result<Vec<(String, LogicalExpression)>> {
        properties
            .iter()
            .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
            .collect()
    }

    fn translate_create_statement(&self, create: &ast::CreateClause) -> Result<LogicalPlan> {
        let elements = self.create_elements(&create.patterns)?;
        if elements.is_empty() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Empty CREATE",
            )));
        }
        let root = LogicalOperator::Create(CreateOp {
            elements,
            input: None,
        });
        Ok(LogicalPlan::new(no_result(root)))
    }

    fn translate_delete(
        &self,
        delete_clause: &ast::DeleteClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // Without a clause before it, nothing binds what DELETE names.
        let input = input.ok_or_else(|| {
            let message = match delete_clause.expressions.first() {
                Some(ast::Expression::Variable(variable)) => format!(
                    "Undefined variable '{variable}': DELETE needs a clause before it that \
                     binds it, such as MATCH"
                ),
                _ => "DELETE requires input".to_string(),
            };
            Error::Query(QueryError::new(QueryErrorKind::Semantic, message))
        })?;

        let mut plan = input;

        // Delete each expression (typically variables)
        for expr in &delete_clause.expressions {
            if let ast::Expression::Variable(var) = expr {
                if self.is_edge_variable(var) {
                    plan = LogicalOperator::DeleteEdge(DeleteEdgeOp {
                        variable: var.clone(),
                        input: Box::new(plan),
                    });
                } else {
                    plan = LogicalOperator::DeleteNode(DeleteNodeOp {
                        variable: var.clone(),
                        detach: delete_clause.detach,
                        input: Box::new(plan),
                    });
                }
            } else {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    "DELETE only supports variable expressions",
                )));
            }
        }

        Ok(plan)
    }

    fn translate_set(
        &self,
        set_clause: &ast::SetClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "SET requires input",
            ))
        })?;

        let mut plan = input;

        // Group items by variable
        for item in &set_clause.items {
            match item {
                ast::SetItem::Property {
                    variable,
                    property,
                    value,
                } => {
                    // SET n.prop = value
                    let value_expr = self.translate_expression(value)?;
                    plan = push_set_property(
                        plan,
                        variable,
                        property.clone(),
                        value_expr,
                        self.is_edge_variable(variable),
                    );
                }
                ast::SetItem::AllProperties {
                    variable,
                    properties,
                } => {
                    // SET n = {...} or SET n = m
                    let value_expr = self.translate_expression(properties)?;
                    plan = LogicalOperator::SetProperty(SetPropertyOp {
                        variable: variable.clone(),
                        properties: vec![("*".to_string(), value_expr)],
                        replace: true,
                        is_edge: self.is_edge_variable(variable),
                        input: Box::new(plan),
                    });
                }
                ast::SetItem::MergeProperties {
                    variable,
                    properties,
                } => {
                    // SET n += {...}
                    let value_expr = self.translate_expression(properties)?;
                    plan = LogicalOperator::SetProperty(SetPropertyOp {
                        variable: variable.clone(),
                        properties: vec![("*".to_string(), value_expr)],
                        replace: false,
                        is_edge: self.is_edge_variable(variable),
                        input: Box::new(plan),
                    });
                }
                ast::SetItem::Labels { variable, labels } => {
                    // SET n:Label1:Label2 adds labels to the node
                    plan = LogicalOperator::AddLabel(AddLabelOp {
                        variable: variable.clone(),
                        labels: labels.clone(),
                        input: Box::new(plan),
                    });
                }
            }
        }

        Ok(plan)
    }

    fn translate_remove(
        &self,
        remove_clause: &ast::RemoveClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "REMOVE requires input",
            ))
        })?;

        let mut plan = input;

        for item in &remove_clause.items {
            match item {
                ast::RemoveItem::Property { variable, property } => {
                    // REMOVE n.prop sets the property to null
                    plan = push_set_property(
                        plan,
                        variable,
                        property.clone(),
                        LogicalExpression::Literal(Value::Null),
                        self.is_edge_variable(variable),
                    );
                }
                ast::RemoveItem::Labels { variable, labels } => {
                    // REMOVE n:Label removes labels from the node
                    plan = LogicalOperator::RemoveLabel(RemoveLabelOp {
                        variable: variable.clone(),
                        labels: labels.clone(),
                        input: Box::new(plan),
                    });
                }
            }
        }

        Ok(plan)
    }

    fn translate_expression(&self, expr: &ast::Expression) -> Result<LogicalExpression> {
        match expr {
            ast::Expression::Literal(lit) => self.translate_literal(lit),
            ast::Expression::Variable(name) => Ok(LogicalExpression::Variable(name.clone())),
            ast::Expression::Parameter(name) => Ok(LogicalExpression::Parameter(name.clone())),
            ast::Expression::PropertyAccess { base, property } => {
                if let ast::Expression::Variable(var) = base.as_ref() {
                    Ok(LogicalExpression::Property {
                        variable: var.clone(),
                        property: property.clone(),
                    })
                } else {
                    // Key access into a map value: `n.meta.route` reads like
                    // `n.meta['route']`.
                    super::common::map_access(self.translate_expression(base)?, property)
                }
            }
            ast::Expression::IndexAccess { base, index } => {
                let base_expr = self.translate_expression(base)?;
                let index_expr = self.translate_expression(index)?;
                Ok(LogicalExpression::IndexAccess {
                    base: Box::new(base_expr),
                    index: Box::new(index_expr),
                })
            }
            ast::Expression::SliceAccess { base, start, end } => {
                let base_expr = self.translate_expression(base)?;
                let start_expr = start
                    .as_ref()
                    .map(|s| self.translate_expression(s))
                    .transpose()?
                    .map(Box::new);
                let end_expr = end
                    .as_ref()
                    .map(|e| self.translate_expression(e))
                    .transpose()?
                    .map(Box::new);
                Ok(LogicalExpression::SliceAccess {
                    base: Box::new(base_expr),
                    start: start_expr,
                    end: end_expr,
                })
            }
            ast::Expression::Binary { left, op, right } => {
                let left_expr = self.translate_expression(left)?;
                let right_expr = self.translate_expression(right)?;
                let binary_op = self.translate_binary_op(*op)?;

                Ok(LogicalExpression::Binary {
                    left: Box::new(left_expr),
                    op: binary_op,
                    right: Box::new(right_expr),
                })
            }
            ast::Expression::Unary { op, operand } => {
                let operand_expr = self.translate_expression(operand)?;
                // Unary positive is identity: just return the operand
                if *op == ast::UnaryOp::Pos {
                    return Ok(operand_expr);
                }
                let unary_op = self.translate_unary_op(*op)?;

                Ok(LogicalExpression::Unary {
                    op: unary_op,
                    operand: Box::new(operand_expr),
                })
            }
            ast::Expression::FunctionCall { name, args, .. } => {
                // `exists((p)-[:T]->())` is the pattern predicate itself
                // (openCypher 9, exists()): true when the pattern has a match
                // for the row. As a call it would test the predicate's value
                // for null, which is never null.
                if name.eq_ignore_ascii_case("exists")
                    && let [pattern @ ast::Expression::Exists(_)] = args.as_slice()
                {
                    return self.translate_expression(pattern);
                }

                // `length(p)` stays a function of the path value: the planner
                // reads the length column of a path the pattern binds, and
                // computes it from the value of any other path (one passed on
                // by a WITH, unwound from a list).
                let translated_args: Vec<LogicalExpression> = args
                    .iter()
                    .map(|a| self.translate_expression(a))
                    .collect::<Result<_>>()?;

                Ok(LogicalExpression::FunctionCall {
                    name: name.clone(),
                    args: translated_args,
                    distinct: false,
                })
            }
            ast::Expression::List(items) => {
                let translated: Vec<LogicalExpression> = items
                    .iter()
                    .map(|i| self.translate_expression(i))
                    .collect::<Result<_>>()?;

                Ok(LogicalExpression::List(translated))
            }
            ast::Expression::Map(pairs) => {
                let translated: Vec<(String, LogicalExpression)> = pairs
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                    .collect::<Result<_>>()?;
                Ok(LogicalExpression::Map(translated))
            }
            ast::Expression::Case {
                input,
                whens,
                else_clause,
            } => {
                let translated_operand = if let Some(op) = input {
                    Some(Box::new(self.translate_expression(op)?))
                } else {
                    None
                };

                let translated_when: Vec<(LogicalExpression, LogicalExpression)> = whens
                    .iter()
                    .map(|(when, then)| {
                        Ok((
                            self.translate_expression(when)?,
                            self.translate_expression(then)?,
                        ))
                    })
                    .collect::<Result<_>>()?;

                let translated_else = if let Some(el) = else_clause {
                    Some(Box::new(self.translate_expression(el)?))
                } else {
                    None
                };

                Ok(LogicalExpression::Case {
                    operand: translated_operand,
                    when_clauses: translated_when,
                    else_clause: translated_else,
                })
            }
            ast::Expression::ListComprehension {
                variable,
                list,
                filter,
                projection,
            } => {
                let list_expr = self.translate_expression(list)?;
                let filter_expr = filter
                    .as_ref()
                    .map(|f| self.translate_expression(f))
                    .transpose()?
                    .map(Box::new);
                // If no projection, use the variable itself as the map expression
                let map_expr = if let Some(proj) = projection {
                    self.translate_expression(proj)?
                } else {
                    LogicalExpression::Variable(variable.clone())
                };

                Ok(LogicalExpression::ListComprehension {
                    variable: variable.clone(),
                    list_expr: Box::new(list_expr),
                    filter_expr,
                    map_expr: Box::new(map_expr),
                })
            }
            ast::Expression::ListPredicate {
                kind,
                variable,
                list,
                predicate,
            } => {
                let ir_kind = match kind {
                    ast::ListPredicateKind::All => ListPredicateKind::All,
                    ast::ListPredicateKind::Any => ListPredicateKind::Any,
                    ast::ListPredicateKind::None => ListPredicateKind::None,
                    ast::ListPredicateKind::Single => ListPredicateKind::Single,
                };
                Ok(LogicalExpression::ListPredicate {
                    kind: ir_kind,
                    variable: variable.clone(),
                    list_expr: Box::new(self.translate_expression(list)?),
                    predicate: Box::new(self.translate_expression(predicate)?),
                })
            }
            ast::Expression::PatternComprehension {
                pattern,
                where_clause,
                projection,
            } => {
                // Build a subplan from the pattern, which binds each
                // relationship once like the patterns of a MATCH
                let pattern_plan =
                    self.translate_comma_patterns(std::slice::from_ref(pattern.as_ref()), None)?;
                // Apply optional WHERE filter
                let subplan = if let Some(where_expr) = where_clause {
                    let pred = self.translate_expression(where_expr)?;
                    wrap_filter(pattern_plan, pred)
                } else {
                    pattern_plan
                };
                let proj = self.translate_expression(projection)?;
                Ok(LogicalExpression::PatternComprehension {
                    subplan: Box::new(subplan),
                    projection: Box::new(proj),
                })
            }
            ast::Expression::MapProjection { base, entries } => {
                let ir_entries = entries
                    .iter()
                    .map(|entry| match entry {
                        ast::MapProjectionEntry::PropertySelector(name) => {
                            Ok(MapProjectionEntry::PropertySelector(name.clone()))
                        }
                        ast::MapProjectionEntry::LiteralEntry(key, expr) => {
                            let translated = self.translate_expression(expr)?;
                            Ok(MapProjectionEntry::LiteralEntry(key.clone(), translated))
                        }
                        ast::MapProjectionEntry::AllProperties => {
                            Ok(MapProjectionEntry::AllProperties)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::MapProjection {
                    base: base.clone(),
                    entries: ir_entries,
                })
            }
            ast::Expression::Reduce {
                accumulator,
                initial,
                variable,
                list,
                expression,
            } => Ok(LogicalExpression::Reduce {
                accumulator: accumulator.clone(),
                initial: Box::new(self.translate_expression(initial)?),
                variable: variable.clone(),
                list: Box::new(self.translate_expression(list)?),
                expression: Box::new(self.translate_expression(expression)?),
            }),
            ast::Expression::Exists(inner_query) => {
                let inner_plan = self.translate_exists_subquery(inner_query)?;
                Ok(LogicalExpression::ExistsSubquery(Box::new(inner_plan)))
            }
            ast::Expression::CountSubquery(inner_query) => {
                let inner_plan = self.translate_exists_subquery(inner_query)?;
                Ok(LogicalExpression::CountSubquery(Box::new(inner_plan)))
            }
        }
    }

    /// Translates the inner query of an EXISTS or COUNT subquery to a
    /// `LogicalOperator`.
    fn translate_exists_subquery(&self, query: &ast::Query) -> Result<LogicalOperator> {
        let mut plan: Option<LogicalOperator> = None;

        for clause in &query.clauses {
            match clause {
                ast::Clause::Match(m) => {
                    plan = Some(self.translate_match(m, plan)?);
                }
                ast::Clause::OptionalMatch(m) => {
                    plan = Some(self.translate_optional_match(m, plan)?);
                }
                ast::Clause::Where(w) => {
                    plan = Some(self.translate_where(w, plan)?);
                }
                _ => {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "EXISTS and COUNT subqueries only support MATCH, OPTIONAL MATCH and WHERE clauses",
                    )));
                }
            }
        }

        plan.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "EXISTS subquery requires at least one MATCH clause",
            ))
        })
    }

    fn translate_literal(&self, lit: &ast::Literal) -> Result<LogicalExpression> {
        let value = match lit {
            ast::Literal::Null => Value::Null,
            ast::Literal::Bool(b) => Value::Bool(*b),
            ast::Literal::Integer(i) => Value::Int64(*i),
            ast::Literal::Float(f) => Value::Float64(*f),
            ast::Literal::String(s) => Value::from(s.as_str()),
        };
        Ok(LogicalExpression::Literal(value))
    }

    fn translate_binary_op(&self, op: ast::BinaryOp) -> Result<BinaryOp> {
        Ok(match op {
            ast::BinaryOp::Eq => BinaryOp::Eq,
            ast::BinaryOp::Ne => BinaryOp::Ne,
            ast::BinaryOp::Lt => BinaryOp::Lt,
            ast::BinaryOp::Le => BinaryOp::Le,
            ast::BinaryOp::Gt => BinaryOp::Gt,
            ast::BinaryOp::Ge => BinaryOp::Ge,
            ast::BinaryOp::And => BinaryOp::And,
            ast::BinaryOp::Or => BinaryOp::Or,
            ast::BinaryOp::Xor => BinaryOp::Xor,
            ast::BinaryOp::Add => BinaryOp::Add,
            ast::BinaryOp::Sub => BinaryOp::Sub,
            ast::BinaryOp::Mul => BinaryOp::Mul,
            ast::BinaryOp::Div => BinaryOp::Div,
            ast::BinaryOp::Mod => BinaryOp::Mod,
            ast::BinaryOp::Pow => BinaryOp::Pow,
            ast::BinaryOp::Concat => BinaryOp::Concat,
            ast::BinaryOp::StartsWith => BinaryOp::StartsWith,
            ast::BinaryOp::EndsWith => BinaryOp::EndsWith,
            ast::BinaryOp::Contains => BinaryOp::Contains,
            ast::BinaryOp::RegexMatch => BinaryOp::Regex,
            ast::BinaryOp::In => BinaryOp::In,
        })
    }

    fn translate_unary_op(&self, op: ast::UnaryOp) -> Result<UnaryOp> {
        Ok(match op {
            ast::UnaryOp::Not => UnaryOp::Not,
            ast::UnaryOp::Neg => UnaryOp::Neg,
            ast::UnaryOp::Pos => {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    "Unary positive not yet supported",
                )));
            }
            ast::UnaryOp::IsNull => UnaryOp::IsNull,
            ast::UnaryOp::IsNotNull => UnaryOp::IsNotNull,
        })
    }

    fn eval_as_count_expr(&self, expr: &ast::Expression) -> Result<CountExpr> {
        match expr {
            ast::Expression::Literal(ast::Literal::Integer(i)) => {
                // Clamp negative values to 0 (LIMIT -1 returns empty, not an error)
                // reason: clamped to >= 0 by .max(0)
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let n = (*i).max(0) as usize;
                Ok(CountExpr::Literal(n))
            }
            ast::Expression::Parameter(name) => Ok(CountExpr::Parameter(name.clone())),
            _ => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Expected integer literal or parameter for SKIP/LIMIT",
            ))),
        }
    }

    fn get_last_variable(plan: &LogicalOperator) -> Result<String> {
        match plan {
            LogicalOperator::NodeScan(scan) => Ok(scan.variable.clone()),
            LogicalOperator::Expand(expand) => Ok(expand.to_variable.clone()),
            LogicalOperator::Filter(filter) => Self::get_last_variable(&filter.input),
            LogicalOperator::Project(project) => Self::get_last_variable(&project.input),
            _ => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Cannot get variable from operator",
            ))),
        }
    }

    // ========================================================================
    // Pattern comprehension rewrite helpers
    // ========================================================================

    /// Extracts the anchor (start) variable from a pattern subplan.
    ///
    /// Walks down the operator tree following the input chain to find
    /// the leaf `NodeScan`, returning its variable name. This is the
    /// variable that needs to be imported from the outer scope.
    fn extract_anchor_variable(op: &LogicalOperator) -> Option<String> {
        match op {
            LogicalOperator::NodeScan(scan) if scan.input.is_none() => Some(scan.variable.clone()),
            LogicalOperator::NodeScan(scan) => Self::extract_anchor_variable(scan.input.as_ref()?),
            LogicalOperator::Expand(expand) => Self::extract_anchor_variable(&expand.input),
            LogicalOperator::Filter(filter) => Self::extract_anchor_variable(&filter.input),
            _ => None,
        }
    }

    /// Starts a pattern subplan from the row it runs for, for correlated
    /// execution: the leaf `NodeScan` of `anchor` reads `ParameterScan(columns)`
    /// (the row's variables the subplan uses), or becomes it when the row
    /// binds `anchor` itself, under a check of the scan's label.
    fn import_row(op: LogicalOperator, anchor: &str, columns: &[String]) -> LogicalOperator {
        match op {
            LogicalOperator::NodeScan(mut scan)
                if scan.variable == anchor && scan.input.is_none() =>
            {
                let row = LogicalOperator::ParameterScan(ParameterScanOp {
                    columns: columns.to_vec(),
                });
                if !columns.iter().any(|column| column == anchor) {
                    scan.input = Some(Box::new(row));
                    return LogicalOperator::NodeScan(scan);
                }
                let label = scan.label.as_slice();
                match has_all_labels(anchor, label) {
                    Some(predicate) => wrap_filter(row, predicate),
                    None => row,
                }
            }
            LogicalOperator::Expand(mut expand) => {
                expand.input = Box::new(Self::import_row(*expand.input, anchor, columns));
                LogicalOperator::Expand(expand)
            }
            LogicalOperator::Filter(mut filter) => {
                filter.input = Box::new(Self::import_row(*filter.input, anchor, columns));
                LogicalOperator::Filter(filter)
            }
            other => other,
        }
    }

    /// The variables of the row that the pattern comprehension `subplan` and
    /// `projection` name, sorted: at either end of the pattern, in its
    /// property maps and WHERE, and in its projection. `outer` holds the
    /// row's variables; when they are not known here, the first node of the
    /// pattern is taken for the row's (`anchor`).
    fn comprehension_imports(
        subplan: &LogicalOperator,
        projection: &LogicalExpression,
        anchor: &str,
        outer: Option<&HashSet<String>>,
    ) -> Vec<String> {
        let Some(outer) = outer else {
            return vec![anchor.to_string()];
        };
        let mut named = HashSet::new();
        pattern_plan_names(subplan, &mut named);
        collect_expression_variables(projection, &mut named);
        let mut imports: Vec<String> = named
            .into_iter()
            .filter(|name| outer.contains(name))
            .collect();
        imports.sort();
        imports
    }

    /// Rewrites the pattern comprehensions in the arguments and group keys of
    /// an aggregation into `Apply`s over its input (see
    /// [`rewrite_pattern_comprehensions`](Self::rewrite_pattern_comprehensions)),
    /// so `sum(size([(b)-->(c) | c]))` aggregates their lists.
    fn lift_aggregate_pattern_comprehensions(
        &self,
        input: LogicalOperator,
        aggregates: &mut [AggregateExpr],
        group_by: &mut [LogicalExpression],
    ) -> Result<LogicalOperator> {
        let expressions = aggregates
            .iter_mut()
            .flat_map(|aggregate| {
                [&mut aggregate.expression, &mut aggregate.expression2]
                    .into_iter()
                    .flatten()
            })
            .chain(group_by.iter_mut());
        let mut lifted = Vec::new();
        for expression in expressions {
            self.take_pattern_comprehensions(expression, &mut lifted);
        }
        if lifted.is_empty() {
            return Ok(input);
        }
        Ok(self.rewrite_pattern_comprehensions(input, lifted)?.0)
    }

    /// Rewrites the pattern comprehensions nested inside `expressions` (not
    /// one that is a whole expression, which the item rewrite handles) into
    /// `Apply`s over `input`.
    fn lift_nested_pattern_comprehensions<'e>(
        &self,
        input: LogicalOperator,
        expressions: impl Iterator<Item = &'e mut LogicalExpression>,
    ) -> Result<LogicalOperator> {
        let mut lifted = Vec::new();
        for expression in expressions {
            if !matches!(expression, LogicalExpression::PatternComprehension { .. }) {
                self.take_pattern_comprehensions(expression, &mut lifted);
            }
        }
        if lifted.is_empty() {
            return Ok(input);
        }
        Ok(self.rewrite_pattern_comprehensions(input, lifted)?.0)
    }

    /// Replaces each pattern comprehension in `expression` with a variable of
    /// its own and adds it to `lifted` as an item that collects into that
    /// variable. Comprehension and predicate bodies are left alone: they can
    /// read their own iteration variable.
    fn take_pattern_comprehensions(
        &self,
        expression: &mut LogicalExpression,
        lifted: &mut Vec<ReturnItem>,
    ) {
        match expression {
            LogicalExpression::PatternComprehension { .. } => {
                let alias = self.next_anon_var();
                let comprehension =
                    std::mem::replace(expression, LogicalExpression::Variable(alias.clone()));
                lifted.push(ReturnItem {
                    expression: comprehension,
                    alias: Some(alias),
                });
            }
            LogicalExpression::Binary { left, right, .. } => {
                self.take_pattern_comprehensions(left, lifted);
                self.take_pattern_comprehensions(right, lifted);
            }
            LogicalExpression::Unary { operand, .. } => {
                self.take_pattern_comprehensions(operand, lifted);
            }
            LogicalExpression::FunctionCall { args, .. } | LogicalExpression::List(args) => {
                for arg in args {
                    self.take_pattern_comprehensions(arg, lifted);
                }
            }
            LogicalExpression::Map(entries) => {
                for (_, value) in entries {
                    self.take_pattern_comprehensions(value, lifted);
                }
            }
            LogicalExpression::IndexAccess { base, index } => {
                self.take_pattern_comprehensions(base, lifted);
                self.take_pattern_comprehensions(index, lifted);
            }
            LogicalExpression::MapAccess { base, .. } => {
                self.take_pattern_comprehensions(base, lifted);
            }
            LogicalExpression::SliceAccess { base, start, end } => {
                self.take_pattern_comprehensions(base, lifted);
                for bound in [start, end].into_iter().flatten() {
                    self.take_pattern_comprehensions(bound, lifted);
                }
            }
            LogicalExpression::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                if let Some(operand) = operand {
                    self.take_pattern_comprehensions(operand, lifted);
                }
                for (condition, result) in when_clauses {
                    self.take_pattern_comprehensions(condition, lifted);
                    self.take_pattern_comprehensions(result, lifted);
                }
                if let Some(else_clause) = else_clause {
                    self.take_pattern_comprehensions(else_clause, lifted);
                }
            }
            LogicalExpression::ListComprehension { list_expr, .. }
            | LogicalExpression::ListPredicate { list_expr, .. } => {
                self.take_pattern_comprehensions(list_expr, lifted);
            }
            LogicalExpression::Reduce { initial, list, .. } => {
                self.take_pattern_comprehensions(initial, lifted);
                self.take_pattern_comprehensions(list, lifted);
            }
            LogicalExpression::MapProjection { entries, .. } => {
                for entry in entries {
                    if let MapProjectionEntry::LiteralEntry(_, value) = entry {
                        self.take_pattern_comprehensions(value, lifted);
                    }
                }
            }
            _ => {}
        }
    }

    /// Rewrites pattern comprehensions in return items into Apply + Aggregate.
    ///
    /// For each `PatternComprehension` found in the items:
    /// 1. Extracts the anchor variable from the subplan
    /// 2. Starts the subplan from the variables of the row it names (see
    ///    [`Self::comprehension_imports`] and [`Self::import_row`]), or from
    ///    its own scan when it names none
    /// 3. Wraps the subplan in `Aggregate(collect(projection) AS alias)`
    /// 4. Wraps the current input in `Apply` that imports those variables
    /// 5. Replaces the expression with `Variable(alias)`
    fn rewrite_pattern_comprehensions(
        &self,
        input: LogicalOperator,
        items: Vec<ReturnItem>,
    ) -> Result<(LogicalOperator, Vec<ReturnItem>)> {
        // The variables of the row each comprehension runs for: those of the
        // input, not the lists collected before it
        let outer = input.bound_variables(self.call_scope.borrow().as_ref());
        let mut current_input = input;
        let mut rewritten_items = Vec::with_capacity(items.len());

        for item in items {
            if let LogicalExpression::PatternComprehension {
                ref subplan,
                ref projection,
            } = item.expression
            {
                // 1. Extract anchor variable
                let anchor = Self::extract_anchor_variable(subplan).ok_or_else(|| {
                    Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "Pattern comprehension must start with a node pattern",
                    ))
                })?;

                // 2. Generate alias for the collected list
                let alias = item.alias.clone().unwrap_or_else(|| self.next_anon_var());

                // 3. Start from the row's variables the comprehension names
                let imports =
                    Self::comprehension_imports(subplan, projection, &anchor, outer.as_ref());
                let rewritten_subplan = if imports.is_empty() {
                    *subplan.clone()
                } else {
                    Self::import_row(*subplan.clone(), &anchor, &imports)
                };

                // 4. Wrap in Aggregate(collect(projection) AS alias)
                let inner_plan = LogicalOperator::Aggregate(AggregateOp {
                    group_by: vec![],
                    aggregates: vec![AggregateExpr {
                        function: AggregateFunction::Collect,
                        expression: Some(*projection.clone()),
                        expression2: None,
                        distinct: false,
                        alias: Some(alias.clone()),
                        percentile: None,
                        separator: None,
                    }],
                    input: Box::new(rewritten_subplan),
                    having: None,
                });

                // 5. Wrap outer input in Apply
                current_input = LogicalOperator::Apply(ApplyOp {
                    input: Box::new(current_input),
                    subplan: Box::new(inner_plan),
                    shared_variables: imports,
                    optional: false,
                    unit: false,
                });

                // 6. Replace expression with Variable reference
                rewritten_items.push(ReturnItem {
                    expression: LogicalExpression::Variable(alias.clone()),
                    alias: Some(alias),
                });
            } else {
                rewritten_items.push(item);
            }
        }

        Ok((current_input, rewritten_items))
    }
}

/// Whether a query that ends with `clause` has no result (openCypher): an
/// update (`CREATE`, `MERGE`, `SET`, `REMOVE`, `DELETE`, `FOREACH`) or a unit
/// subquery `CALL` (one that returns rows cannot end a query). A procedure
/// `CALL` keeps its output.
fn ends_without_result(clause: &ast::Clause) -> bool {
    matches!(
        clause,
        ast::Clause::Create(_)
            | ast::Clause::Merge(_)
            | ast::Clause::Set(_)
            | ast::Clause::Remove(_)
            | ast::Clause::Delete(_)
            | ast::Clause::ForEach(_)
            | ast::Clause::CallSubquery { .. }
    )
}

/// Whether a `CALL` subquery's body ends with a `RETURN` (an `ORDER BY`,
/// `SKIP` or `LIMIT` may follow it): one without is a unit subquery, which
/// runs for its writes and returns no rows of its own.
fn returns_rows(body: &ast::Query) -> bool {
    body.clauses
        .iter()
        .any(|clause| matches!(clause, ast::Clause::Return(_)))
}

/// The relationship patterns of `pattern`, in order.
fn relationship_patterns(pattern: &ast::Pattern) -> Vec<&ast::RelationshipPattern> {
    match pattern {
        ast::Pattern::Node(_) => Vec::new(),
        ast::Pattern::Path(path) => path.chain.iter().collect(),
        ast::Pattern::NamedPath { pattern, .. } => relationship_patterns(pattern),
    }
}

/// The relationship patterns of `pattern`, in order, to change.
fn relationship_patterns_mut(pattern: &mut ast::Pattern) -> Vec<&mut ast::RelationshipPattern> {
    match pattern {
        ast::Pattern::Node(_) => Vec::new(),
        ast::Pattern::Path(path) => path.chain.iter_mut().collect(),
        ast::Pattern::NamedPath { pattern, .. } => relationship_patterns_mut(pattern),
    }
}

/// The labels of a node pattern after the first. A `NodeScan` checks the first
/// label; the others still have to be checked with a filter.
fn extra_labels(node: &ast::NodePattern) -> &[String] {
    node.labels.get(1..).unwrap_or_default()
}

/// The minimum and maximum number of hops (`None` = unbounded) a relationship
/// pattern matches: `[*]` is one or more, and no `*` is exactly one hop.
fn hop_bounds(rel: &ast::RelationshipPattern) -> (u32, Option<u32>) {
    match &rel.length {
        Some(range) => (range.min.unwrap_or(1), range.max),
        None => (1, Some(1)),
    }
}

/// Checks that the ORDER BY key `key`, translated after an aggregating or
/// DISTINCT projection (`scope`), reads only the projection's `columns`.
fn check_reads_projection(
    key: &LogicalExpression,
    columns: &[ProjectedColumn],
    scope: &SortScope,
) -> Result<()> {
    let mut read = HashSet::new();
    collect_expression_variables(key, &mut read);
    let missing = read
        .into_iter()
        .filter(|name| !columns.iter().any(|column| &column.name == name))
        .min();
    match missing {
        None => Ok(()),
        Some(name) => Err(Error::Query(QueryError::new(
            QueryErrorKind::Semantic,
            format!(
                "Undefined variable '{name}': after {} {}, ORDER BY reads only what it returns",
                if scope.aggregating {
                    "an aggregating"
                } else {
                    "a DISTINCT"
                },
                scope.clause
            ),
        ))),
    }
}

/// Adds the variables `kept` to the projection of a WITH (`plan`, maybe
/// under the filter of its WHERE), so that a sort after it reads them, and
/// says whether it did. A plan of another shape, or a WHERE that reads one
/// of them (where they are not in scope), comes back unchanged.
fn keep_columns(plan: LogicalOperator, kept: &[String]) -> (LogicalOperator, bool) {
    match plan {
        LogicalOperator::Project(mut project) if !project.pass_through_input => {
            project
                .projections
                .extend(kept.iter().map(|name| Projection {
                    expression: LogicalExpression::Variable(name.clone()),
                    alias: None,
                }));
            (LogicalOperator::Project(project), true)
        }
        LogicalOperator::Filter(mut filter) => {
            let mut read = HashSet::new();
            collect_expression_variables(&filter.predicate, &mut read);
            if kept.iter().any(|name| read.contains(name)) {
                return (LogicalOperator::Filter(filter), false);
            }
            let (input, kept_them) = keep_columns(*filter.input, kept);
            filter.input = Box::new(input);
            (LogicalOperator::Filter(filter), kept_them)
        }
        other => (other, false),
    }
}

/// The column name of `item`, an item after `*` in a WITH or RETURN
/// (`clause`): its alias or variable, or `None` for an expression without an
/// alias, which is named after its text. A name that `*` passes on already
/// (one of `bound`, the input's variables when they are known) is an error:
/// two columns would have it.
fn star_item_name(
    clause: &str,
    item: &ast::ProjectionItem,
    bound: Option<&HashSet<String>>,
) -> Result<Option<String>> {
    let name = match (&item.alias, &item.expression) {
        (Some(alias), _) => alias.clone(),
        (None, ast::Expression::Variable(name)) => name.clone(),
        (None, _) => return Ok(None),
    };
    if bound.is_some_and(|bound| bound.contains(&name)) {
        return Err(Error::Query(QueryError::new(
            QueryErrorKind::Semantic,
            format!("{clause} *, {name}: {name} is already one of the variables * passes on"),
        )));
    }
    Ok(Some(name))
}

/// Adds the variables the plan of a pattern (a node scan, its expands and
/// filters) names: its nodes, edges and paths, and what its filters read.
fn pattern_plan_names(op: &LogicalOperator, names: &mut HashSet<String>) {
    match op {
        LogicalOperator::NodeScan(scan) => {
            names.insert(scan.variable.clone());
        }
        LogicalOperator::Expand(expand) => {
            names.insert(expand.from_variable.clone());
            names.insert(expand.to_variable.clone());
            names.extend(expand.edge_variable.iter().cloned());
            names.extend(expand.path_alias.iter().cloned());
        }
        LogicalOperator::Filter(filter) => collect_expression_variables(&filter.predicate, names),
        _ => {}
    }
    for child in op.children() {
        pattern_plan_names(child, names);
    }
}

/// Checks if an AST expression contains an aggregate function call.
fn contains_aggregate(expr: &ast::Expression) -> bool {
    match expr {
        ast::Expression::FunctionCall { name, args, .. } => {
            is_aggregate_function(name) || args.iter().any(contains_aggregate)
        }
        ast::Expression::Binary { left, right, .. } => {
            contains_aggregate(left) || contains_aggregate(right)
        }
        ast::Expression::Unary { operand, .. } => contains_aggregate(operand),
        ast::Expression::Case {
            input,
            whens,
            else_clause,
        } => {
            input.as_deref().is_some_and(contains_aggregate)
                || whens
                    .iter()
                    .any(|(w, t)| contains_aggregate(w) || contains_aggregate(t))
                || else_clause.as_deref().is_some_and(contains_aggregate)
        }
        ast::Expression::List(items) => items.iter().any(contains_aggregate),
        ast::Expression::ListComprehension {
            filter, projection, ..
        } => {
            filter.as_deref().is_some_and(contains_aggregate)
                || projection.as_deref().is_some_and(contains_aggregate)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::{FilterOp, LimitOp, SkipOp, SortOp};

    /// The plan of a query that ends with an update, under the `RETURN` of
    /// no items that gives it no result.
    fn without_result(plan: &LogicalPlan) -> &LogicalOperator {
        let LogicalOperator::Return(ret) = &plan.root else {
            panic!(
                "a query ending with an update has no result: {:?}",
                plan.root
            );
        };
        assert!(ret.items.is_empty(), "no columns: {ret:?}");
        &ret.input
    }

    // === Basic MATCH Tests ===

    #[test]
    fn test_translate_simple_match() {
        let plan = translate("MATCH (n:Person) RETURN n").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            if let LogicalOperator::NodeScan(scan) = ret.input.as_ref() {
                assert_eq!(scan.variable, "n");
                assert_eq!(scan.label, Some("Person".into()));
            } else {
                panic!("Expected NodeScan");
            }
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_dotted_map_key_access() {
        let return_expression = |query: &str| {
            let plan = translate(query).unwrap();
            let LogicalOperator::Return(ret) = &plan.root else {
                panic!("Expected Return");
            };
            format!("{:?}", ret.items[0].expression)
        };
        let meta = LogicalExpression::Property {
            variable: "n".to_string(),
            property: "meta".to_string(),
        };
        let key = |base: LogicalExpression, key: &str| LogicalExpression::MapAccess {
            base: Box::new(base),
            key: key.to_string(),
        };
        // `n.meta.route` reads key `route` of the map in `n.meta`, and chains.
        assert_eq!(
            return_expression("MATCH (n) RETURN n.meta.route"),
            format!("{:?}", key(meta.clone(), "route"))
        );
        assert_eq!(
            return_expression("MATCH (n) RETURN n.meta.a.b"),
            format!("{:?}", key(key(meta, "a"), "b"))
        );
        // A node that a function returns is read like a node variable.
        assert_eq!(
            return_expression("MATCH ()-[r]->() RETURN startNode(r).name"),
            format!(
                "{:?}",
                key(
                    LogicalExpression::FunctionCall {
                        name: "startNode".to_string(),
                        args: vec![LogicalExpression::Variable("r".to_string())],
                        distinct: false,
                    },
                    "name"
                )
            )
        );
        // A string is neither a map nor a node.
        let err = translate("MATCH (n) RETURN toUpper(n.name).x").unwrap_err();
        assert!(err.to_string().contains("not a map value"), "{err}");
    }

    #[test]
    fn test_translate_match_with_where() {
        let plan = translate("MATCH (n:Person) WHERE n.age > 30 RETURN n").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::Filter(filter) = ret.input.as_ref() {
                if let LogicalExpression::Binary { op, .. } = &filter.predicate {
                    assert_eq!(*op, BinaryOp::Gt);
                }
            } else {
                panic!("Expected Filter");
            }
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_match_return_distinct() {
        let plan = translate("MATCH (n:Person) RETURN DISTINCT n.name").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            assert!(ret.distinct);
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_match_return_all() {
        let plan = translate("MATCH (n:Person) RETURN *").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            if let LogicalExpression::Variable(v) = &ret.items[0].expression {
                assert_eq!(v, "*");
            }
        } else {
            panic!("Expected Return");
        }
    }

    // === Path Pattern Tests ===

    #[test]
    fn test_translate_outgoing_relationship() {
        let plan = translate("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b").unwrap();

        // Find the Expand operator
        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Return(r) => find_expand(&r.input),
                LogicalOperator::Filter(f) => find_expand(&f.input),
                _ => None,
            }
        }

        let expand = find_expand(&plan.root).expect("Expected Expand");
        assert_eq!(expand.direction, ExpandDirection::Outgoing);
        assert_eq!(expand.edge_types, vec!["KNOWS".to_string()]);
    }

    #[test]
    fn test_translate_incoming_relationship() {
        let plan = translate("MATCH (a:Person)<-[:KNOWS]-(b:Person) RETURN a, b").unwrap();

        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Return(r) => find_expand(&r.input),
                LogicalOperator::Filter(f) => find_expand(&f.input),
                _ => None,
            }
        }

        let expand = find_expand(&plan.root).expect("Expected Expand");
        assert_eq!(expand.direction, ExpandDirection::Incoming);
    }

    #[test]
    fn test_translate_variable_length_path() {
        let plan = translate("MATCH (a:Person)-[:KNOWS*1..3]->(b:Person) RETURN a, b").unwrap();

        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Return(r) => find_expand(&r.input),
                LogicalOperator::Filter(f) => find_expand(&f.input),
                _ => None,
            }
        }

        let expand = find_expand(&plan.root).expect("Expected Expand");
        assert_eq!(expand.min_hops, 1);
        assert_eq!(expand.max_hops, Some(3));
    }

    // === Mutation Tests ===

    #[test]
    fn test_translate_create_node() {
        let plan = translate("CREATE (n:Person {name: 'Alix'})").unwrap();

        let LogicalOperator::Create(create) = without_result(&plan) else {
            panic!("Expected Create, got {:?}", without_result(&plan));
        };
        let [
            CreateElement::Node {
                variable,
                labels,
                properties,
            },
        ] = create.elements.as_slice()
        else {
            panic!("Expected one node, got {:?}", create.elements);
        };
        assert_eq!(variable, "n");
        assert_eq!(labels, &vec!["Person".to_string()]);
        assert_eq!(properties.len(), 1);
        assert_eq!(properties[0].0, "name");
    }

    #[test]
    fn test_translate_create_path() {
        let plan = translate("CREATE (a:Person)-[:KNOWS]->(b:Person)").unwrap();

        // Both nodes, then the edge between them, in one Create.
        let LogicalOperator::Create(create) = without_result(&plan) else {
            panic!("Expected Create, got {:?}", without_result(&plan));
        };
        let [
            CreateElement::Node { variable: a, .. },
            CreateElement::Node { variable: b, .. },
            CreateElement::Edge {
                from_variable,
                to_variable,
                edge_type,
                ..
            },
        ] = create.elements.as_slice()
        else {
            panic!("Expected two nodes and an edge, got {:?}", create.elements);
        };
        assert_eq!((a.as_str(), b.as_str()), ("a", "b"));
        assert_eq!((from_variable, to_variable), (a, b));
        assert_eq!(edge_type, "KNOWS");
    }

    #[test]
    fn test_translate_delete_node() {
        let plan = translate("MATCH (n:Person) DELETE n").unwrap();

        if let LogicalOperator::DeleteNode(delete) = without_result(&plan) {
            assert_eq!(delete.variable, "n");
            if let LogicalOperator::NodeScan(scan) = delete.input.as_ref() {
                assert_eq!(scan.variable, "n");
                assert_eq!(scan.label, Some("Person".into()));
            } else {
                panic!("Expected NodeScan input");
            }
        } else {
            panic!("Expected DeleteNode, got {:?}", without_result(&plan));
        }
    }

    #[test]
    fn test_translate_set_property() {
        let plan = translate("MATCH (n:Person) SET n.name = 'Gus' RETURN n").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::SetProperty(set) = ret.input.as_ref() {
                assert_eq!(set.variable, "n");
                assert_eq!(set.properties.len(), 1);
                assert_eq!(set.properties[0].0, "name");
                assert!(!set.replace);
            } else {
                panic!("Expected SetProperty");
            }
        } else {
            panic!("Expected Return, got {:?}", plan.root);
        }
    }

    #[test]
    fn test_translate_set_multiple_properties() {
        let plan = translate(
            "MATCH (n:Person) SET n.name = 'Alix', n.age = 30, n.next = n.age + 1, n.city = $city \
             RETURN n",
        )
        .unwrap();

        let LogicalOperator::Return(ret) = &plan.root else {
            panic!("Expected Return");
        };
        // Constants set on one node share an operator, in order; a value
        // that reads the node gets its own, so it reads the earlier writes,
        // and the constant after it another.
        let LogicalOperator::SetProperty(city) = ret.input.as_ref() else {
            panic!("Expected SetProperty");
        };
        let LogicalOperator::SetProperty(next) = city.input.as_ref() else {
            panic!("Expected SetProperty below");
        };
        let LogicalOperator::SetProperty(constants) = next.input.as_ref() else {
            panic!("Expected SetProperty below");
        };
        let names = |set: &SetPropertyOp| -> Vec<String> {
            set.properties
                .iter()
                .map(|(name, _)| name.clone())
                .collect()
        };
        assert_eq!(names(constants), ["name", "age"]);
        assert_eq!(names(next), ["next"]);
        assert_eq!(names(city), ["city"]);
        assert!(matches!(
            constants.input.as_ref(),
            LogicalOperator::Filter(_) | LogicalOperator::NodeScan(_)
        ));
    }

    #[test]
    fn test_translate_remove_property() {
        let plan = translate("MATCH (n:Person) REMOVE n.name RETURN n").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::SetProperty(set) = ret.input.as_ref() {
                // REMOVE property is translated to SET property = null
                assert_eq!(set.variable, "n");
                assert_eq!(set.properties.len(), 1);
                assert_eq!(set.properties[0].0, "name");
                // Value should be Null
                if let LogicalExpression::Literal(Value::Null) = &set.properties[0].1 {
                    // OK
                } else {
                    panic!("Expected Null value for REMOVE");
                }
            } else {
                panic!("Expected SetProperty");
            }
        } else {
            panic!("Expected Return, got {:?}", plan.root);
        }
    }

    #[test]
    fn test_translate_remove_label() {
        let plan = translate("MATCH (n:Person:Admin) REMOVE n:Admin RETURN n").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::RemoveLabel(remove) = ret.input.as_ref() {
                assert_eq!(remove.variable, "n");
                assert_eq!(remove.labels, vec!["Admin".to_string()]);
            } else {
                panic!("Expected RemoveLabel");
            }
        } else {
            panic!("Expected Return, got {:?}", plan.root);
        }
    }

    // === WITH, UNWIND, ORDER BY, SKIP, LIMIT Tests ===

    #[test]
    fn test_translate_with_clause() {
        let plan = translate("MATCH (n:Person) WITH n.name AS name RETURN name").unwrap();

        // Find Project operator
        fn find_project(op: &LogicalOperator) -> Option<&ProjectOp> {
            match op {
                LogicalOperator::Project(p) => Some(p),
                LogicalOperator::Return(r) => find_project(&r.input),
                LogicalOperator::Filter(f) => find_project(&f.input),
                _ => None,
            }
        }

        let project = find_project(&plan.root).expect("Expected Project");
        assert_eq!(project.projections.len(), 1);
        assert_eq!(project.projections[0].alias.as_deref(), Some("name"));
    }

    #[test]
    fn test_translate_with_distinct() {
        let plan = translate("MATCH (n:Person) WITH DISTINCT n.city AS city RETURN city").unwrap();

        // Find Distinct operator
        fn find_distinct(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Distinct(_) => true,
                LogicalOperator::Return(r) => find_distinct(&r.input),
                LogicalOperator::Project(p) => find_distinct(&p.input),
                LogicalOperator::Filter(f) => find_distinct(&f.input),
                _ => false,
            }
        }

        assert!(find_distinct(&plan.root));
    }

    #[test]
    fn test_translate_unwind() {
        let plan = translate("UNWIND [1, 2, 3] AS x RETURN x").unwrap();

        // Find Unwind operator
        fn find_unwind(op: &LogicalOperator) -> Option<&UnwindOp> {
            match op {
                LogicalOperator::Unwind(u) => Some(u),
                LogicalOperator::Return(r) => find_unwind(&r.input),
                _ => None,
            }
        }

        let unwind = find_unwind(&plan.root).expect("Expected Unwind");
        assert_eq!(unwind.variable, "x");
    }

    #[test]
    fn test_translate_order_by() {
        let plan = translate("MATCH (n:Person) RETURN n ORDER BY n.name").unwrap();

        fn find_sort(op: &LogicalOperator) -> Option<&SortOp> {
            match op {
                LogicalOperator::Sort(s) => Some(s),
                LogicalOperator::Return(r) => find_sort(&r.input),
                _ => None,
            }
        }

        let sort = find_sort(&plan.root).expect("Expected Sort");
        assert_eq!(sort.keys.len(), 1);
        assert_eq!(sort.keys[0].order, SortOrder::Ascending);
    }

    #[test]
    fn test_translate_order_by_desc() {
        let plan = translate("MATCH (n:Person) RETURN n ORDER BY n.age DESC").unwrap();

        fn find_sort(op: &LogicalOperator) -> Option<&SortOp> {
            match op {
                LogicalOperator::Sort(s) => Some(s),
                LogicalOperator::Return(r) => find_sort(&r.input),
                _ => None,
            }
        }

        let sort = find_sort(&plan.root).expect("Expected Sort");
        assert_eq!(sort.keys[0].order, SortOrder::Descending);
    }

    #[test]
    fn test_translate_limit() {
        let plan = translate("MATCH (n:Person) RETURN n LIMIT 10").unwrap();

        fn find_limit(op: &LogicalOperator) -> Option<&LimitOp> {
            match op {
                LogicalOperator::Limit(l) => Some(l),
                LogicalOperator::Return(r) => find_limit(&r.input),
                _ => None,
            }
        }

        let limit = find_limit(&plan.root).expect("Expected Limit");
        assert_eq!(limit.count, 10);
    }

    #[test]
    fn test_translate_skip() {
        let plan = translate("MATCH (n:Person) RETURN n SKIP 5").unwrap();

        fn find_skip(op: &LogicalOperator) -> Option<&SkipOp> {
            match op {
                LogicalOperator::Skip(s) => Some(s),
                LogicalOperator::Return(r) => find_skip(&r.input),
                LogicalOperator::Limit(l) => find_skip(&l.input),
                _ => None,
            }
        }

        let skip = find_skip(&plan.root).expect("Expected Skip");
        assert_eq!(skip.count, 5);
    }

    // === MERGE Tests ===

    #[test]
    fn test_translate_merge() {
        let plan = translate("MERGE (n:Person {name: 'Alix'})").unwrap();

        if let LogicalOperator::Merge(merge) = without_result(&plan) {
            assert_eq!(merge.variable, "n");
            assert_eq!(merge.labels, vec!["Person".to_string()]);
            assert_eq!(merge.match_properties.len(), 1);
            assert_eq!(merge.match_properties[0].0, "name");
        } else {
            panic!("Expected Merge, got {:?}", without_result(&plan));
        }
    }

    #[test]
    fn test_translate_merge_on_create() {
        let plan =
            translate("MERGE (n:Person {name: 'Alix'}) ON CREATE SET n.created = true").unwrap();

        if let LogicalOperator::Merge(merge) = without_result(&plan) {
            assert_eq!(merge.on_create.len(), 1);
            assert_eq!(merge.on_create[0].0, "created");
        } else {
            panic!("Expected Merge, got {:?}", without_result(&plan));
        }
    }

    // === Expression Tests ===

    #[test]
    fn test_translate_list_expression() {
        // Cypher requires MATCH before RETURN, so use UNWIND to test list
        let plan = translate("UNWIND [1, 2, 3] AS x RETURN x").unwrap();

        fn find_unwind(op: &LogicalOperator) -> Option<&UnwindOp> {
            match op {
                LogicalOperator::Unwind(u) => Some(u),
                LogicalOperator::Return(r) => find_unwind(&r.input),
                _ => None,
            }
        }

        let unwind = find_unwind(&plan.root).expect("Expected Unwind");
        if let LogicalExpression::List(items) = &unwind.expression {
            assert_eq!(items.len(), 3);
        } else {
            panic!("Expected List expression");
        }
    }

    #[test]
    fn test_translate_map_expression() {
        // Test map in CREATE with properties
        let plan = translate("CREATE (n:Person {name: 'Alix', age: 30})").unwrap();

        if let LogicalOperator::Create(create) = without_result(&plan) {
            assert_eq!(create.elements[0].properties().len(), 2);
        } else {
            panic!("Expected Create");
        }
    }

    #[test]
    fn test_translate_function_call() {
        // Use toUpper which is a simple function
        let plan = translate("MATCH (n:Person) RETURN toUpper(n.name)").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalExpression::FunctionCall { name, args, .. } = &ret.items[0].expression {
                assert_eq!(name.to_lowercase(), "toupper");
                assert_eq!(args.len(), 1);
            } else {
                panic!("Expected FunctionCall, got {:?}", ret.items[0].expression);
            }
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_case_expression() {
        let plan =
            translate("MATCH (n:Person) RETURN CASE WHEN n.age > 18 THEN 'adult' ELSE 'minor' END")
                .unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalExpression::Case {
                when_clauses,
                else_clause,
                ..
            } = &ret.items[0].expression
            {
                assert_eq!(when_clauses.len(), 1);
                assert!(else_clause.is_some());
            } else {
                panic!("Expected Case expression");
            }
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_parameter() {
        let plan = translate("MATCH (n:Person) WHERE n.name = $name RETURN n").unwrap();

        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        if let LogicalExpression::Binary { right, .. } = &filter.predicate {
            if let LogicalExpression::Parameter(p) = right.as_ref() {
                assert_eq!(p, "name");
            } else {
                panic!("Expected Parameter");
            }
        }
    }

    // === Error Handling Tests ===

    #[test]
    fn test_translate_binary_op_all() {
        let translator = CypherTranslator::new("");

        // Test all supported binary ops
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Eq).unwrap(),
            BinaryOp::Eq
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Ne).unwrap(),
            BinaryOp::Ne
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Lt).unwrap(),
            BinaryOp::Lt
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Le).unwrap(),
            BinaryOp::Le
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Gt).unwrap(),
            BinaryOp::Gt
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Ge).unwrap(),
            BinaryOp::Ge
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::And).unwrap(),
            BinaryOp::And
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Or).unwrap(),
            BinaryOp::Or
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Xor).unwrap(),
            BinaryOp::Xor
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Add).unwrap(),
            BinaryOp::Add
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Sub).unwrap(),
            BinaryOp::Sub
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Mul).unwrap(),
            BinaryOp::Mul
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Div).unwrap(),
            BinaryOp::Div
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Mod).unwrap(),
            BinaryOp::Mod
        );
    }

    #[test]
    fn test_translate_unary_op_all() {
        let translator = CypherTranslator::new("");

        assert_eq!(
            translator.translate_unary_op(ast::UnaryOp::Not).unwrap(),
            UnaryOp::Not
        );
        assert_eq!(
            translator.translate_unary_op(ast::UnaryOp::Neg).unwrap(),
            UnaryOp::Neg
        );
        assert_eq!(
            translator.translate_unary_op(ast::UnaryOp::IsNull).unwrap(),
            UnaryOp::IsNull
        );
        assert_eq!(
            translator
                .translate_unary_op(ast::UnaryOp::IsNotNull)
                .unwrap(),
            UnaryOp::IsNotNull
        );
    }

    #[test]
    fn test_translate_literal_types() {
        let translator = CypherTranslator::new("");

        // Test all literal types
        let null_lit = translator.translate_literal(&ast::Literal::Null).unwrap();
        assert!(matches!(null_lit, LogicalExpression::Literal(Value::Null)));

        let bool_lit = translator
            .translate_literal(&ast::Literal::Bool(true))
            .unwrap();
        assert!(matches!(
            bool_lit,
            LogicalExpression::Literal(Value::Bool(true))
        ));

        let int_lit = translator
            .translate_literal(&ast::Literal::Integer(42))
            .unwrap();
        assert!(matches!(
            int_lit,
            LogicalExpression::Literal(Value::Int64(42))
        ));

        let float_lit = translator
            .translate_literal(&ast::Literal::Float(std::f64::consts::PI))
            .unwrap();
        if let LogicalExpression::Literal(Value::Float64(f)) = float_lit {
            assert!((f - std::f64::consts::PI).abs() < 0.001);
        } else {
            panic!("Expected Float64");
        }
    }

    #[test]
    fn test_translate_multiple_match_clauses() {
        // Two independent MATCH clauses should produce a valid plan
        let plan = translate(
            "MATCH (a:Person) WHERE a.name = 'Alix' MATCH (b:Person) WHERE b.name = 'Gus' RETURN a.name, b.name",
        )
        .unwrap();

        // The plan should have a Return at the root
        assert!(matches!(&plan.root, LogicalOperator::Return(_)));
    }

    #[test]
    fn test_translate_merge_with_relationship() {
        // MERGE with a relationship pattern should produce MergeRelationship operator
        let plan =
            translate("MATCH (a {id: 'x'}), (b {id: 'y'}) MERGE (a)-[r:KNOWS]->(b) RETURN r")
                .unwrap();

        // The plan root should be a Return
        if let LogicalOperator::Return(ret) = &plan.root {
            // The input to Return should be MergeRelationship
            assert!(
                matches!(ret.input.as_ref(), LogicalOperator::MergeRelationship(_)),
                "Expected MergeRelationship, got: {:?}",
                std::mem::discriminant(ret.input.as_ref())
            );
        } else {
            panic!("Expected Return at root");
        }
    }

    #[test]
    fn test_translate_set_edge_property_is_edge_true() {
        // SET on an edge variable from MATCH should produce is_edge: true
        let plan = translate("MATCH (a)-[r:KNOWS]->(b) SET r.weight = 1.0 RETURN r").unwrap();

        // Walk the plan tree to find SetProperty
        fn find_set_property(op: &LogicalOperator) -> Option<&SetPropertyOp> {
            match op {
                LogicalOperator::SetProperty(set_op) => Some(set_op),
                LogicalOperator::Return(ret) => find_set_property(&ret.input),
                _ => None,
            }
        }

        let set_op = find_set_property(&plan.root).expect("Expected SetProperty in plan");
        assert!(
            set_op.is_edge,
            "SET on edge variable 'r' should have is_edge: true"
        );
        assert_eq!(set_op.variable, "r");
    }

    #[test]
    fn test_translate_set_node_property_is_edge_false() {
        // SET on a node variable should produce is_edge: false
        let plan = translate("MATCH (n:Person) SET n.age = 30 RETURN n").unwrap();

        fn find_set_property(op: &LogicalOperator) -> Option<&SetPropertyOp> {
            match op {
                LogicalOperator::SetProperty(set_op) => Some(set_op),
                LogicalOperator::Return(ret) => find_set_property(&ret.input),
                _ => None,
            }
        }

        let set_op = find_set_property(&plan.root).expect("Expected SetProperty in plan");
        assert!(
            !set_op.is_edge,
            "SET on node variable 'n' should have is_edge: false"
        );
        assert_eq!(set_op.variable, "n");
    }

    #[test]
    fn test_translate_set_edge_after_merge_relationship() {
        // SET on edge variable from MERGE should also have is_edge: true
        let plan = translate(
            "MATCH (a {id: 'x'}), (b {id: 'y'}) MERGE (a)-[r:KNOWS]->(b) SET r.since = '2024' RETURN r",
        )
        .unwrap();

        fn find_set_property(op: &LogicalOperator) -> Option<&SetPropertyOp> {
            match op {
                LogicalOperator::SetProperty(set_op) => Some(set_op),
                LogicalOperator::Return(ret) => find_set_property(&ret.input),
                _ => None,
            }
        }

        let set_op = find_set_property(&plan.root).expect("Expected SetProperty in plan");
        assert!(
            set_op.is_edge,
            "SET on MERGE edge variable 'r' should have is_edge: true"
        );
        assert_eq!(set_op.variable, "r");
    }

    #[test]
    fn test_translate_aggregate_count() {
        let plan = translate("MATCH (n:Person) RETURN count(n)").unwrap();
        // Should produce an Aggregate operator
        fn has_aggregate(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Aggregate(_) => true,
                LogicalOperator::Return(ret) => has_aggregate(&ret.input),
                _ => false,
            }
        }
        assert!(has_aggregate(&plan.root), "Expected Aggregate operator");
    }

    #[test]
    fn test_translate_case_inside_aggregate() {
        // sum(CASE WHEN n.type = 'x' THEN 1 ELSE 0 END) should be detected as aggregate
        let plan = translate(
            "MATCH (n:Person) RETURN sum(CASE WHEN n.type = 'source' THEN 1 ELSE 0 END) AS cnt",
        )
        .unwrap();
        fn has_aggregate(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Aggregate(_) => true,
                LogicalOperator::Return(ret) => has_aggregate(&ret.input),
                _ => false,
            }
        }
        assert!(
            has_aggregate(&plan.root),
            "Expected Aggregate operator for CASE inside aggregate"
        );
    }

    #[test]
    fn test_translate_case_wrapping_aggregate() {
        // CASE WHEN count(*) > 0 should also be detected as containing an aggregate
        let plan = translate(
            "MATCH (n:Person) RETURN CASE WHEN count(*) > 0 THEN 'yes' ELSE 'no' END AS result",
        )
        .unwrap();
        fn has_aggregate(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Aggregate(_) => true,
                LogicalOperator::Return(ret) => has_aggregate(&ret.input),
                _ => false,
            }
        }
        assert!(
            has_aggregate(&plan.root),
            "Expected Aggregate operator for CASE wrapping aggregate"
        );
    }

    #[test]
    fn test_translate_union_all() {
        let plan =
            translate("MATCH (n:Person) RETURN n.name UNION ALL MATCH (m:Animal) RETURN m.name")
                .unwrap();
        assert!(
            matches!(&plan.root, LogicalOperator::Union(_)),
            "Expected Union at root, got {:?}",
            std::mem::discriminant(&plan.root)
        );
    }

    #[test]
    fn test_translate_union_without_all_applies_distinct() {
        let plan = translate("MATCH (n:Person) RETURN n.name UNION MATCH (m:Animal) RETURN m.name")
            .unwrap();
        // UNION (without ALL) should wrap the result in Distinct
        assert!(
            matches!(&plan.root, LogicalOperator::Distinct(_)),
            "Expected Distinct at root for UNION without ALL, got {:?}",
            std::mem::discriminant(&plan.root)
        );
        if let LogicalOperator::Distinct(distinct) = &plan.root {
            assert!(
                matches!(distinct.input.as_ref(), LogicalOperator::Union(_)),
                "Expected Union inside Distinct"
            );
        }
    }

    #[test]
    fn test_translate_call_procedure() {
        let plan = translate("CALL db.labels()").unwrap();
        assert!(
            matches!(&plan.root, LogicalOperator::CallProcedure(_)),
            "Expected CallProcedure, got {:?}",
            std::mem::discriminant(&plan.root)
        );
        if let LogicalOperator::CallProcedure(call) = &plan.root {
            assert_eq!(call.name, vec!["db", "labels"]);
            assert!(call.arguments.is_empty());
            assert!(call.yield_items.is_none());
        }
    }

    #[test]
    fn test_translate_call_with_args_and_yield() {
        let plan = translate("CALL db.index.fulltext('Person', 'name') YIELD status").unwrap();
        if let LogicalOperator::CallProcedure(call) = &plan.root {
            assert_eq!(call.name, vec!["db", "index", "fulltext"]);
            assert_eq!(call.arguments.len(), 2);
            assert!(call.yield_items.is_some());
            let yields = call.yield_items.as_ref().unwrap();
            assert_eq!(yields.len(), 1);
            assert_eq!(yields[0].field_name, "status");
        } else {
            panic!("Expected CallProcedure");
        }
    }

    // === EXISTS Subquery Tests ===

    #[test]
    fn test_translate_exists_subquery() {
        let plan =
            translate("MATCH (n:Person) WHERE EXISTS { MATCH (n)-[:KNOWS]->() } RETURN n").unwrap();

        // Plan should be Return -> Filter -> NodeScan
        // with the Filter predicate being ExistsSubquery
        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::Filter(filter) = ret.input.as_ref() {
                assert!(
                    matches!(&filter.predicate, LogicalExpression::ExistsSubquery(_)),
                    "Expected ExistsSubquery predicate, got {:?}",
                    filter.predicate
                );
            } else {
                panic!("Expected Filter, got {:?}", ret.input);
            }
        } else {
            panic!("Expected Return");
        }
    }

    // === Map Projection Tests ===

    #[test]
    fn test_translate_map_projection() {
        let plan = translate("MATCH (p:Person) RETURN p { .name, .age }").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            if let LogicalExpression::MapProjection { base, entries } = &ret.items[0].expression {
                assert_eq!(base, "p");
                assert_eq!(entries.len(), 2);
                assert!(
                    matches!(&entries[0], MapProjectionEntry::PropertySelector(s) if s == "name")
                );
                assert!(
                    matches!(&entries[1], MapProjectionEntry::PropertySelector(s) if s == "age")
                );
            } else {
                panic!(
                    "Expected MapProjection expression, got {:?}",
                    ret.items[0].expression
                );
            }
        } else {
            panic!("Expected Return");
        }
    }

    // === reduce() Tests ===

    #[test]
    fn test_translate_reduce() {
        let plan = translate("MATCH (n) RETURN reduce(acc = 0, x IN [1,2,3] | acc + x)").unwrap();

        // Walk past possible Aggregate wrapping to find the Reduce expression
        fn find_reduce_expr(op: &LogicalOperator) -> Option<&LogicalExpression> {
            match op {
                LogicalOperator::Return(ret) => {
                    for item in &ret.items {
                        if matches!(&item.expression, LogicalExpression::Reduce { .. }) {
                            return Some(&item.expression);
                        }
                    }
                    find_reduce_expr(&ret.input)
                }
                LogicalOperator::Aggregate(agg) => {
                    // Check group_by expressions
                    for expr in &agg.group_by {
                        if matches!(expr, LogicalExpression::Reduce { .. }) {
                            return Some(expr);
                        }
                    }
                    find_reduce_expr(&agg.input)
                }
                _ => None,
            }
        }

        let reduce_expr =
            find_reduce_expr(&plan.root).expect("Expected Reduce expression in the plan");

        if let LogicalExpression::Reduce {
            accumulator,
            initial,
            variable,
            list,
            expression,
        } = reduce_expr
        {
            assert_eq!(accumulator, "acc");
            assert!(matches!(
                initial.as_ref(),
                LogicalExpression::Literal(Value::Int64(0))
            ));
            assert_eq!(variable, "x");
            // The list should be a List of 3 items
            if let LogicalExpression::List(items) = list.as_ref() {
                assert_eq!(items.len(), 3);
            } else {
                panic!("Expected List for reduce iteration, got {:?}", list);
            }
            // The body should be acc + x (Binary Add)
            if let LogicalExpression::Binary { op, .. } = expression.as_ref() {
                assert_eq!(*op, BinaryOp::Add);
            } else {
                panic!("Expected Binary Add in reduce body, got {:?}", expression);
            }
        } else {
            panic!("Expected Reduce, got {:?}", reduce_expr);
        }
    }

    // === Pattern Comprehension Tests ===

    #[test]
    fn test_translate_pattern_comprehension() {
        let plan = translate("MATCH (p:Person) RETURN [(p)-[:KNOWS]->(f) | f.name]").unwrap();

        // After rewrite: Return -> Apply -> NodeScan
        // The PatternComprehension is rewritten to Apply + Aggregate(Collect) + ParameterScan
        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            // Expression should now be a Variable reference (not PatternComprehension)
            assert!(
                matches!(&ret.items[0].expression, LogicalExpression::Variable(_)),
                "Expected Variable after rewrite, got {:?}",
                ret.items[0].expression
            );
            // The input should be an Apply operator
            if let LogicalOperator::Apply(apply) = ret.input.as_ref() {
                assert_eq!(apply.shared_variables, vec!["p".to_string()]);
                // Inner plan should be Aggregate(Collect)
                if let LogicalOperator::Aggregate(agg) = apply.subplan.as_ref() {
                    assert_eq!(agg.aggregates.len(), 1);
                    assert_eq!(agg.aggregates[0].function, AggregateFunction::Collect);
                    // Aggregate input should contain an Expand over ParameterScan
                    fn has_parameter_scan(op: &LogicalOperator) -> bool {
                        match op {
                            LogicalOperator::ParameterScan(_) => true,
                            LogicalOperator::Expand(e) => has_parameter_scan(&e.input),
                            LogicalOperator::Filter(f) => has_parameter_scan(&f.input),
                            _ => false,
                        }
                    }
                    assert!(
                        has_parameter_scan(&agg.input),
                        "Expected ParameterScan in inner plan, got {:?}",
                        agg.input
                    );
                } else {
                    panic!(
                        "Expected Aggregate in Apply subplan, got {:?}",
                        apply.subplan
                    );
                }
            } else {
                panic!("Expected Apply as Return input, got {:?}", ret.input);
            }
        } else {
            panic!("Expected Return");
        }
    }

    // === COUNT Subquery Tests ===

    #[test]
    fn test_translate_count_subquery() {
        let plan =
            translate("MATCH (p:Person) RETURN COUNT { MATCH (p)-[:KNOWS]->() } AS cnt").unwrap();

        // The COUNT subquery should appear as a CountSubquery expression
        fn find_count_subquery(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Return(ret) => {
                    ret.items
                        .iter()
                        .any(|item| matches!(&item.expression, LogicalExpression::CountSubquery(_)))
                        || find_count_subquery(&ret.input)
                }
                LogicalOperator::Aggregate(agg) => {
                    agg.group_by
                        .iter()
                        .any(|expr| matches!(expr, LogicalExpression::CountSubquery(_)))
                        || find_count_subquery(&agg.input)
                }
                _ => false,
            }
        }

        assert!(
            find_count_subquery(&plan.root),
            "Expected CountSubquery in the plan, got {:?}",
            plan.root
        );
    }

    // === CALL Subquery Tests ===

    #[test]
    fn test_translate_call_subquery() {
        let plan = translate(
            "MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS cnt } RETURN p.name, cnt",
        )
        .unwrap();

        // The plan should have an Apply operator for the CALL subquery
        fn find_apply(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Apply(_) => true,
                LogicalOperator::Return(ret) => find_apply(&ret.input),
                LogicalOperator::Filter(f) => find_apply(&f.input),
                LogicalOperator::Sort(s) => find_apply(&s.input),
                _ => false,
            }
        }

        assert!(
            find_apply(&plan.root),
            "Expected Apply operator for CALL subquery"
        );

        // Verify the final RETURN has two items
        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(
                ret.items.len(),
                2,
                "Expected 2 return items (p.name and cnt)"
            );
        } else {
            panic!("Expected Return at root");
        }
    }

    // === FOREACH Tests ===

    #[test]
    fn test_translate_foreach() {
        let plan = translate("MATCH (n:Person) FOREACH (x IN [1,2,3] | SET n.x = 1)").unwrap();

        // FOREACH translates to a unit Apply over the MATCH, importing every
        // variable of the row: its subplan unwinds the list from the row and
        // runs the SET once per item.
        let LogicalOperator::Apply(apply) = without_result(&plan) else {
            panic!(
                "expected a unit Apply at the root, got {:?}",
                without_result(&plan)
            );
        };
        assert!(apply.unit, "FOREACH passes each row on once");
        assert!(!apply.optional);
        assert_eq!(apply.shared_variables, ["*"]);
        assert!(matches!(&*apply.input, LogicalOperator::NodeScan(_)));
        let LogicalOperator::SetProperty(set) = &*apply.subplan else {
            panic!("expected the SET in the subplan, got {:?}", apply.subplan);
        };
        let LogicalOperator::Unwind(unwind) = &*set.input else {
            panic!("expected the Unwind below the SET, got {:?}", set.input);
        };
        assert_eq!(unwind.variable, "x");
        assert!(matches!(
            &*unwind.input,
            LogicalOperator::ParameterScan(scan) if scan.columns == ["*"]
        ));
    }

    // === Basic Query Translation Tests ===

    #[test]
    fn test_translate_match_with_multiple_labels() {
        // Multi-label node patterns
        let plan = translate("MATCH (n:Person:Employee) RETURN n").unwrap();
        assert!(matches!(&plan.root, LogicalOperator::Return(_)));
    }

    #[test]
    fn test_translate_standalone_return() {
        // RETURN without MATCH (pure expression evaluation)
        let plan = translate("RETURN 2 * 3").unwrap();
        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            // The expression should be a Binary Mul of 2 * 3
            if let LogicalExpression::Binary { op, left, right } = &ret.items[0].expression {
                assert_eq!(*op, BinaryOp::Mul);
                assert!(matches!(
                    left.as_ref(),
                    LogicalExpression::Literal(Value::Int64(2))
                ));
                assert!(matches!(
                    right.as_ref(),
                    LogicalExpression::Literal(Value::Int64(3))
                ));
            } else {
                panic!("Expected Binary expression");
            }
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_undirected_relationship() {
        let plan = translate("MATCH (a:Person)-[:FRIEND]-(b:Person) RETURN a, b").unwrap();

        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Return(r) => find_expand(&r.input),
                LogicalOperator::Filter(f) => find_expand(&f.input),
                _ => None,
            }
        }

        let expand = find_expand(&plan.root).expect("Expected Expand");
        assert_eq!(expand.direction, ExpandDirection::Both);
        assert_eq!(expand.edge_types, vec!["FRIEND".to_string()]);
    }

    #[test]
    fn test_translate_match_with_return_alias() {
        let plan = translate("MATCH (n:Person) RETURN n.name AS personName").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            assert_eq!(ret.items[0].alias.as_deref(), Some("personName"));
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_multiple_return_items() {
        let plan = translate("MATCH (n:Person) RETURN n.name, n.age, n.city").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 3);
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_list_predicate_all() {
        let plan = translate("MATCH (n) WHERE all(x IN [1,2,3] WHERE x > 0) RETURN n").unwrap();

        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        assert!(
            matches!(
                &filter.predicate,
                LogicalExpression::ListPredicate {
                    kind: ListPredicateKind::All,
                    ..
                }
            ),
            "Expected ListPredicate(All), got {:?}",
            filter.predicate
        );
    }

    #[test]
    fn test_translate_list_comprehension() {
        let plan = translate("MATCH (n) RETURN [x IN [1,2,3] WHERE x > 1 | x * 2]").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            assert!(
                matches!(
                    &ret.items[0].expression,
                    LogicalExpression::ListComprehension { .. }
                ),
                "Expected ListComprehension, got {:?}",
                ret.items[0].expression
            );
        } else {
            panic!("Expected Return");
        }
    }
}
