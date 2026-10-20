//! What an ORDER BY after a WITH or an aggregating RETURN reads.
//!
//! As openCypher's ORDER BY does (and Grafeo's Cypher, see `SortScope` in
//! translators/cypher.rs): after a WITH that neither aggregates nor is
//! DISTINCT, a sort key reads the WITH's columns and the variables of its
//! input; after an aggregating or DISTINCT one, and after an aggregating
//! RETURN, only the clause's columns, where a key that repeats a returned
//! expression (`max(a.age)`, `a.name`) reads its column.

use super::GqlTranslator;
use super::aggregate::contains_aggregate;
use crate::query::functions::resolve_call;
use crate::query::plan::{LogicalExpression, SortKey};
use crate::query::translators::common::{check_sort_reads_projection, is_aggregate_function};
use grafeo_adapters::query::gql::ast;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};

/// What a WITH projects, for the ORDER BY statement right after it.
pub(super) struct SortScope {
    /// Whether the WITH aggregates rows.
    pub(super) aggregating: bool,
    /// Whether the WITH is DISTINCT.
    pub(super) distinct: bool,
    /// The items it projects, after its horizontal aggregates are taken out.
    pub(super) items: Vec<ast::ReturnItem>,
}

/// The column of a projected item and the text an ORDER BY key that repeats
/// the item's expression has (the aggregate's own name for an aggregate).
struct ProjectedColumn {
    text: String,
    name: String,
    /// Whether the item has an alias, which names its column.
    aliased: bool,
}

impl GqlTranslator {
    /// The sort keys of `order_by` after the WITH of `scope`: plain keys after
    /// one that neither aggregates nor is DISTINCT (the caller keeps the input
    /// variables they read, see `common::sort_keeping_input`), keys that read
    /// its columns after one that does.
    ///
    /// # Errors
    ///
    /// Returns a semantic error for a key that reads a variable or an
    /// aggregate the WITH does not return.
    pub(super) fn sort_keys_after_with(
        &self,
        order_by: &ast::OrderByClause,
        scope: &SortScope,
    ) -> Result<Vec<SortKey>> {
        if scope.aggregating || scope.distinct {
            return self.projected_sort_keys(order_by, &scope.items, "WITH", scope.aggregating);
        }
        order_by
            .items
            .iter()
            .map(|item| self.translate_sort_key(item))
            .collect()
    }

    /// The sort keys of `order_by` after a RETURN or WITH (`clause`) of
    /// `items` that aggregates (`aggregating`) or is DISTINCT: an expression
    /// the clause returns, whole or inside the key (`max(a.age) + 1`), reads
    /// its column.
    ///
    /// # Errors
    ///
    /// Returns a semantic error for an aggregate the projection does not
    /// return (its groups are gone) and for a key that reads a variable it
    /// does not return.
    pub(super) fn projected_sort_keys(
        &self,
        order_by: &ast::OrderByClause,
        items: &[ast::ReturnItem],
        clause: &str,
        aggregating: bool,
    ) -> Result<Vec<SortKey>> {
        let columns = items
            .iter()
            .map(|item| {
                let text = self.projection_text(&item.expression)?;
                let name = item.alias.clone().unwrap_or_else(|| text.clone());
                Ok(ProjectedColumn {
                    text,
                    name,
                    aliased: item.alias.is_some(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let names: Vec<&str> = columns.iter().map(|column| column.name.as_str()).collect();
        order_by
            .items
            .iter()
            .map(|item| {
                let expression =
                    self.translate_projected_sort_key(&item.expression, &columns, clause, false)?;
                check_sort_reads_projection(&expression, &names, clause, aggregating)?;
                if aggregating {
                    return Ok(super::sort_key(item, expression));
                }
                let planned =
                    self.translate_projected_sort_key(&item.expression, &columns, clause, true)?;
                Ok(super::sort_key(item, planned))
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

    /// Translates an ORDER BY key after a projection (see
    /// [`projected_sort_keys`](Self::projected_sort_keys)). `as_planned`
    /// gives the key the sort reads after a DISTINCT RETURN, where an
    /// unaliased item's column has no name the binder knows: the item's
    /// expression itself, which has the column's value; without it, each such
    /// expression reads its column, as the check of what the key reads needs.
    fn translate_projected_sort_key(
        &self,
        expression: &ast::Expression,
        columns: &[ProjectedColumn],
        clause: &str,
        as_planned: bool,
    ) -> Result<LogicalExpression> {
        let text = self.projection_text(expression)?;
        if let Some(column) = columns.iter().find(|column| column.text == text) {
            if as_planned && !column.aliased {
                return self.translate_expression(expression);
            }
            return Ok(LogicalExpression::Variable(column.name.clone()));
        }
        match expression {
            ast::Expression::Binary { left, op, right } => Ok(LogicalExpression::Binary {
                left: Box::new(
                    self.translate_projected_sort_key(left, columns, clause, as_planned)?,
                ),
                op: self.translate_binary_op(*op),
                right: Box::new(
                    self.translate_projected_sort_key(right, columns, clause, as_planned)?,
                ),
            }),
            ast::Expression::Unary { op, operand } => {
                let operand =
                    self.translate_projected_sort_key(operand, columns, clause, as_planned)?;
                if *op == ast::UnaryOp::Pos {
                    return Ok(operand);
                }
                Ok(LogicalExpression::Unary {
                    op: self.translate_unary_op(*op),
                    operand: Box::new(operand),
                })
            }
            ast::Expression::FunctionCall {
                name,
                args,
                distinct,
            } if !is_aggregate_function(name) && args.iter().any(contains_aggregate) => {
                resolve_call(
                    name,
                    args.iter()
                        .map(|arg| {
                            self.translate_projected_sort_key(arg, columns, clause, as_planned)
                        })
                        .collect::<Result<_>>()?,
                    *distinct,
                )
            }
            other if contains_aggregate(other) => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!(
                    "Invalid use of an aggregating function in ORDER BY: {text} is not an \
                     aggregate the {clause} returns"
                ),
            ))),
            other => self.translate_expression(other),
        }
    }
}
