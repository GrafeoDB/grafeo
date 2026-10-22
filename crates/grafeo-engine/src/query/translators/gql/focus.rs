//! The graph a GQL statement is focused on: `USE g <statement>` (ISO/IEC
//! 39075 use graph clause), which reads and writes g in place of the
//! session's current graph without changing it.
//!
//! A statement runs on one graph: the one its parts name, the same on every
//! operand of a `UNION` or `NEXT` chain and on every `USE` inside a query.
//! Parts that name different graphs are not supported yet: each scan and
//! write carrying its own graph comes after the store read work.

use grafeo_adapters::query::gql::ast;
use grafeo_common::utils::error::{Error, QueryError, Result};

/// The graph `statement` is focused on: `None` when no part names one (or
/// they name `CURRENT_GRAPH`), the session's current graph.
///
/// # Errors
///
/// Parts that name different graphs, or a part that names one next to a
/// part that names none: not supported yet.
pub(super) fn statement_graph(statement: &ast::Statement) -> Result<Option<String>> {
    let mut graphs = Graphs::default();
    graphs.statement(statement, None);
    graphs.into_graph()
}

/// The graphs the parts of a statement name, in order, without repeats.
#[derive(Default)]
struct Graphs(Vec<Option<String>>);

impl Graphs {
    fn add(&mut self, graph: Option<&str>) {
        let graph = graph.map(str::to_string);
        if !self.0.contains(&graph) {
            self.0.push(graph);
        }
    }

    /// The graphs of `statement`, an operand whose focus is `outer` until a
    /// `USE` of its own names another.
    fn statement(&mut self, statement: &ast::Statement, outer: Option<&str>) {
        match statement {
            ast::Statement::Use { graph, statement } => {
                self.statement(statement, graph.as_deref());
            }
            ast::Statement::CompositeQuery { left, right, .. } => {
                self.statement(left, outer);
                self.statement(right, outer);
            }
            ast::Statement::Explain(inner) | ast::Statement::Profile(inner) => {
                self.statement(inner, outer);
            }
            ast::Statement::Query(query) => {
                self.add(outer);
                self.query(query);
            }
            ast::Statement::DataModification(_)
            | ast::Statement::Call(_)
            | ast::Statement::Schema(_)
            | ast::Statement::SessionCommand(_) => self.add(outer),
        }
    }

    /// The graphs the `USE` clauses inside `query` and its `CALL`
    /// subqueries name. A subquery without one reads the graph of the query
    /// around it.
    fn query(&mut self, query: &ast::QueryStatement) {
        for clause in &query.ordered_clauses {
            match clause {
                ast::QueryClause::Use(graph) => self.add(graph.as_deref()),
                ast::QueryClause::InlineCall {
                    subquery, combined, ..
                } => {
                    self.query(subquery);
                    for (_, query) in combined {
                        self.query(query);
                    }
                }
                _ => {}
            }
        }
    }

    fn into_graph(self) -> Result<Option<String>> {
        let mut graphs = self.0.into_iter();
        let first = graphs.next().flatten();
        let Some(other) = graphs.next() else {
            return Ok(first);
        };
        let name = |graph: Option<&str>| {
            graph.map_or_else(|| "the session's graph".to_string(), |g| format!("'{g}'"))
        };
        Err(Error::from(QueryError::unsupported(format!(
            "the parts of this statement read different graphs ({} and {}), which is not \
             supported yet: give every part the same USE, or run each part as a statement of \
             its own",
            name(first.as_deref()),
            name(other.as_deref())
        ))))
    }
}
