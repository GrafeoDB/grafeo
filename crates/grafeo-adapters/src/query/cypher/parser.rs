//! Cypher Parser.
//!
//! Parses Cypher queries into an AST.

#[allow(clippy::wildcard_imports)]
use super::ast::*;
use super::lexer::{Lexer, Token, TokenKind};
use crate::query::keywords::unescape_string;
use grafeo_common::utils::error::{QueryError, QueryErrorKind, Result};

/// Maximum nesting depth for recursive parsing constructs (parenthesized
/// expressions, CASE, EXISTS subqueries, list literals, function calls).
const MAX_NESTING_DEPTH: u32 = 128;

/// Cypher query parser.
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
    previous: Token,
    source: &'a str,
    /// Current nesting depth for recursive parsing constructs.
    nesting_depth: u32,
}

impl<'a> Parser<'a> {
    /// Creates a new parser for the given query.
    pub fn new(query: &'a str) -> Self {
        let mut lexer = Lexer::new(query);
        let current = lexer.next_token();
        let previous = Token {
            kind: TokenKind::Eof,
            text: String::new(),
            span: current.span,
        };
        Self {
            lexer,
            current,
            previous,
            source: query,
            nesting_depth: 0,
        }
    }

    /// Increments the nesting depth and returns an error if the limit is exceeded.
    fn enter_nesting(&mut self) -> Result<()> {
        self.nesting_depth += 1;
        if self.nesting_depth > MAX_NESTING_DEPTH {
            return Err(self.error(&format!(
                "Maximum nesting depth of {MAX_NESTING_DEPTH} exceeded"
            )));
        }
        Ok(())
    }

    /// Decrements the nesting depth.
    fn exit_nesting(&mut self) {
        self.nesting_depth = self.nesting_depth.saturating_sub(1);
    }

    /// Parses the query into a statement.
    ///
    /// # Errors
    ///
    /// Returns an error if the input contains invalid or unexpected Cypher syntax.
    pub fn parse(&mut self) -> Result<Statement> {
        // Handle EXPLAIN/PROFILE prefix: wraps the entire following statement
        if self.current.kind == TokenKind::Identifier
            && self.current.text.eq_ignore_ascii_case("EXPLAIN")
        {
            self.advance(); // consume EXPLAIN
            self.enter_nesting()?;
            let inner = self.parse()?;
            self.exit_nesting();
            return Ok(Statement::Explain(Box::new(inner)));
        }
        if self.current.kind == TokenKind::Identifier
            && self.current.text.eq_ignore_ascii_case("PROFILE")
        {
            self.advance(); // consume PROFILE
            self.enter_nesting()?;
            let inner = self.parse()?;
            self.exit_nesting();
            return Ok(Statement::Profile(Box::new(inner)));
        }

        // Schema DDL: CREATE INDEX, DROP INDEX, CREATE CONSTRAINT, DROP CONSTRAINT
        if self.current.kind == TokenKind::Create
            && self.peek_kind() == TokenKind::Identifier
            && (self.peek_text_eq_ignore_case("INDEX")
                || self.peek_text_eq_ignore_case("CONSTRAINT"))
        {
            let stmt = self.parse_schema_create()?;
            self.expect_end_of_query()?;
            return Ok(stmt);
        }
        if self.current.kind == TokenKind::Identifier
            && self.current.text.eq_ignore_ascii_case("DROP")
            && let Some(stmt) = self.try_parse_schema_drop()?
        {
            self.expect_end_of_query()?;
            return Ok(stmt);
        }
        // ALTER CURRENT GRAPH TYPE
        if self.current.kind == TokenKind::Identifier
            && self.current.text.eq_ignore_ascii_case("ALTER")
            && let Some(stmt) = self.try_parse_alter_current_graph_type()?
        {
            self.expect_end_of_query()?;
            return Ok(stmt);
        }
        // SHOW INDEXES / SHOW CONSTRAINTS / SHOW CURRENT GRAPH TYPE
        if self.current.kind == TokenKind::Identifier
            && self.current.text.eq_ignore_ascii_case("SHOW")
            && let Some(stmt) = self.try_parse_show()?
        {
            self.expect_end_of_query()?;
            return Ok(stmt);
        }

        let stmt = self.parse_statement()?;
        // Check for UNION continuation
        if self.current.kind == TokenKind::Union {
            return self.parse_union_continuation(stmt);
        }
        self.expect_end_of_query()?;
        Ok(stmt)
    }

    /// Requires that only `;` remain after a complete statement.
    fn expect_end_of_query(&mut self) -> Result<()> {
        let saw_semicolon = self.current.kind == TokenKind::Semicolon;
        self.skip_semicolons();
        if self.current.kind == TokenKind::Eof {
            return Ok(());
        }
        if saw_semicolon {
            return Err(self.error(
                "multiple statements separated by ';' are not supported in one call: \
                 run them separately",
            ));
        }
        Err(self.error("Expected end of query"))
    }

    /// Parses UNION / UNION ALL between query blocks.
    fn parse_union_continuation(&mut self, first_stmt: Statement) -> Result<Statement> {
        let Statement::Query(first_query) = first_stmt else {
            return Err(self.error("UNION requires query statements"));
        };
        let mut queries = vec![first_query];
        let mut is_all = false;

        while self.current.kind == TokenKind::Union {
            self.advance(); // consume UNION
            is_all = self.current.kind == TokenKind::All;
            if is_all {
                self.advance(); // consume ALL
            }
            let next_stmt = self.parse_statement()?;
            match next_stmt {
                Statement::Query(q) => queries.push(q),
                _ => return Err(self.error("UNION requires query statements")),
            }
        }

        self.expect_end_of_query()?;

        Ok(Statement::Union {
            queries,
            all: is_all,
        })
    }

    fn parse_statement(&mut self) -> Result<Statement> {
        // Parse reading/writing clauses into a query
        let mut clauses = Vec::new();

        loop {
            match self.current.kind {
                TokenKind::Match => {
                    clauses.push(Clause::Match(self.parse_match_clause()?));
                }
                TokenKind::Optional => {
                    self.advance();
                    self.expect(TokenKind::Match)?;
                    let match_clause = self.parse_match_clause_body()?;
                    clauses.push(Clause::OptionalMatch(match_clause));
                }
                TokenKind::Where => {
                    clauses.push(Clause::Where(self.parse_where_clause()?));
                }
                TokenKind::With => {
                    clauses.push(Clause::With(self.parse_with_clause()?));
                }
                TokenKind::Return => {
                    clauses.push(Clause::Return(self.parse_return_clause()?));
                }
                TokenKind::Unwind => {
                    clauses.push(Clause::Unwind(self.parse_unwind_clause()?));
                }
                TokenKind::Create => {
                    clauses.push(Clause::Create(self.parse_create_clause()?));
                }
                TokenKind::Merge => {
                    clauses.push(Clause::Merge(self.parse_merge_clause()?));
                }
                TokenKind::Delete | TokenKind::Detach => {
                    clauses.push(Clause::Delete(self.parse_delete_clause()?));
                }
                TokenKind::Set => {
                    clauses.push(Clause::Set(self.parse_set_clause()?));
                }
                TokenKind::Remove => {
                    clauses.push(Clause::Remove(self.parse_remove_clause()?));
                }
                TokenKind::Order => {
                    clauses.push(Clause::OrderBy(self.parse_order_by_clause()?));
                }
                TokenKind::Skip => {
                    self.advance();
                    clauses.push(Clause::Skip(self.parse_expression()?));
                }
                TokenKind::Limit => {
                    self.advance();
                    clauses.push(Clause::Limit(self.parse_expression()?));
                }
                TokenKind::Call => {
                    // CALL { subquery } vs CALL procedure(...)
                    if self.peek_kind() == TokenKind::LBrace {
                        self.advance(); // consume CALL
                        self.advance(); // consume {
                        let inner = self.parse_subquery_body()?;
                        self.expect(TokenKind::RBrace)?;
                        clauses.push(Clause::CallSubquery(inner));
                    } else {
                        clauses.push(Clause::Call(self.parse_call_clause()?));
                    }
                }
                _ => {
                    // FOREACH and LOAD are contextual keywords (not reserved)
                    if self.can_be_identifier()
                        && self.get_identifier_text().to_uppercase() == "FOREACH"
                    {
                        clauses.push(Clause::ForEach(self.parse_foreach_clause()?));
                    } else if self.is_contextual("LOAD") {
                        clauses.push(Clause::LoadCsv(self.parse_load_csv_clause()?));
                    } else {
                        break;
                    }
                }
            }
        }

        if clauses.is_empty() {
            return Err(self.error("Expected a Cypher clause"));
        }

        Ok(Statement::Query(Query {
            clauses,
            span: None,
        }))
    }

    /// Parses a CALL clause: `CALL name.space(args) [YIELD field [AS alias], ...]`.
    fn parse_call_clause(&mut self) -> Result<CallClause> {
        let span_start = self.current.span.start;
        self.expect(TokenKind::Call)?;

        // Parse dotted procedure name
        let mut name_parts = vec![self.expect_identifier()?];
        while self.current.kind == TokenKind::Dot {
            self.advance();
            name_parts.push(self.expect_identifier()?);
        }

        // Parse argument list
        self.expect(TokenKind::LParen)?;
        let mut arguments = Vec::new();
        if self.current.kind != TokenKind::RParen {
            arguments.push(self.parse_expression()?);
            while self.current.kind == TokenKind::Comma {
                self.advance();
                arguments.push(self.parse_expression()?);
            }
        }
        self.expect(TokenKind::RParen)?;

        // Parse optional YIELD clause
        let yield_items = if self.current.kind == TokenKind::Yield {
            self.advance();
            let mut items = vec![self.parse_cypher_yield_item()?];
            while self.current.kind == TokenKind::Comma {
                self.advance();
                items.push(self.parse_cypher_yield_item()?);
            }
            Some(items)
        } else {
            None
        };

        Ok(CallClause {
            procedure_name: name_parts,
            arguments,
            yield_items,
            span: Some(grafeo_common::utils::error::SourceSpan::new(
                span_start,
                self.current.span.start,
                1,
                1,
            )),
        })
    }

    /// Parses a FOREACH clause: `FOREACH (var IN expr | clauses)`.
    /// Parses a LOAD CSV clause:
    /// `LOAD CSV [WITH HEADERS] FROM 'path' AS variable [FIELDTERMINATOR 'sep']`
    fn parse_load_csv_clause(&mut self) -> Result<LoadCsvClause> {
        self.advance(); // consume LOAD

        self.expect_contextual("CSV")?;

        // Optional WITH HEADERS
        let with_headers = if self.current.kind == TokenKind::With {
            self.advance(); // consume WITH
            self.expect_contextual("HEADERS")?;
            true
        } else {
            false
        };

        // FROM 'path'
        self.expect_contextual("FROM")?;
        let path = if self.current.kind == TokenKind::String {
            let raw = self.current.text.clone();
            self.advance();
            // Strip quotes and unescape
            unescape_string(&raw[1..raw.len() - 1])
        } else {
            return Err(self.error("Expected string literal for file path"));
        };

        // AS variable
        self.expect(TokenKind::As)?;
        let variable = self.expect_identifier()?;

        // Optional FIELDTERMINATOR 'char'
        let field_terminator = if self.is_contextual("FIELDTERMINATOR") {
            self.advance();
            if self.current.kind == TokenKind::String {
                let raw = self.current.text.clone();
                self.advance();
                let unescaped = unescape_string(&raw[1..raw.len() - 1]);
                let ch = unescaped
                    .chars()
                    .next()
                    .ok_or_else(|| self.error("FIELDTERMINATOR must be a single character"))?;
                Some(ch)
            } else {
                return Err(self.error("Expected string literal for FIELDTERMINATOR"));
            }
        } else {
            None
        };

        Ok(LoadCsvClause {
            with_headers,
            path,
            variable,
            field_terminator,
            span: None,
        })
    }

    fn parse_foreach_clause(&mut self) -> Result<ForEachClause> {
        self.advance(); // consume FOREACH identifier
        self.expect(TokenKind::LParen)?;
        let variable = self.expect_identifier()?;
        self.expect(TokenKind::In)?;
        let list = self.parse_expression()?;
        self.expect(TokenKind::Pipe)?;
        let mut clauses = Vec::new();
        // Parse mutation clauses until closing paren
        while self.current.kind != TokenKind::RParen {
            match self.current.kind {
                TokenKind::Set => clauses.push(Clause::Set(self.parse_set_clause()?)),
                TokenKind::Delete | TokenKind::Detach => {
                    clauses.push(Clause::Delete(self.parse_delete_clause()?));
                }
                TokenKind::Remove => clauses.push(Clause::Remove(self.parse_remove_clause()?)),
                TokenKind::Create => clauses.push(Clause::Create(self.parse_create_clause()?)),
                TokenKind::Merge => clauses.push(Clause::Merge(self.parse_merge_clause()?)),
                _ => {
                    if self.can_be_identifier()
                        && self.get_identifier_text().to_uppercase() == "FOREACH"
                    {
                        clauses.push(Clause::ForEach(self.parse_foreach_clause()?));
                    } else {
                        return Err(self.error("Expected mutation clause in FOREACH"));
                    }
                }
            }
        }
        self.expect(TokenKind::RParen)?;
        Ok(ForEachClause {
            variable,
            list,
            clauses,
        })
    }

    /// Parses the body of an inline subquery (CALL { ... } or COUNT { ... }).
    ///
    /// Expects to be positioned after the opening `{`.
    fn parse_subquery_body(&mut self) -> Result<Query> {
        let mut clauses = Vec::new();
        while self.current.kind != TokenKind::RBrace && self.current.kind != TokenKind::Eof {
            match self.current.kind {
                TokenKind::Match => {
                    clauses.push(Clause::Match(self.parse_match_clause()?));
                }
                TokenKind::Optional => {
                    self.advance();
                    self.expect(TokenKind::Match)?;
                    let match_clause = self.parse_match_clause_body()?;
                    clauses.push(Clause::OptionalMatch(match_clause));
                }
                TokenKind::Where => {
                    clauses.push(Clause::Where(self.parse_where_clause()?));
                }
                TokenKind::With => {
                    clauses.push(Clause::With(self.parse_with_clause()?));
                }
                TokenKind::Return => {
                    clauses.push(Clause::Return(self.parse_return_clause()?));
                }
                TokenKind::Unwind => {
                    clauses.push(Clause::Unwind(self.parse_unwind_clause()?));
                }
                TokenKind::Create => {
                    clauses.push(Clause::Create(self.parse_create_clause()?));
                }
                TokenKind::Set => {
                    clauses.push(Clause::Set(self.parse_set_clause()?));
                }
                _ => break,
            }
        }
        if clauses.is_empty() {
            return Err(self.error("Expected at least one clause in subquery"));
        }
        Ok(Query {
            clauses,
            span: None,
        })
    }

    /// Parses a single YIELD item: `field_name [AS alias]`.
    fn parse_cypher_yield_item(&mut self) -> Result<YieldItem> {
        let field_name = self.expect_identifier()?;
        let alias = if self.current.kind == TokenKind::As {
            self.advance();
            Some(self.expect_identifier()?)
        } else {
            None
        };
        Ok(YieldItem { field_name, alias })
    }

    fn parse_match_clause(&mut self) -> Result<MatchClause> {
        self.expect(TokenKind::Match)?;
        self.parse_match_clause_body()
    }

    fn parse_match_clause_body(&mut self) -> Result<MatchClause> {
        let patterns = self.parse_pattern_list()?;
        Ok(MatchClause {
            patterns,
            span: None,
        })
    }

    /// Parses the inner query of an EXISTS subquery.
    /// Accepts one or more MATCH clauses and an optional WHERE clause.
    fn parse_exists_inner_query(&mut self) -> Result<Query> {
        let mut clauses = Vec::new();

        while self.current.kind == TokenKind::Match || self.current.kind == TokenKind::Optional {
            clauses.push(Clause::Match(self.parse_match_clause()?));
        }

        // Bare pattern form: EXISTS { (a)-[r]->(b) WHERE ... }
        // Treat as implicit MATCH when no MATCH keyword but a pattern starts with (
        if clauses.is_empty() && self.current.kind == TokenKind::LParen {
            clauses.push(Clause::Match(self.parse_match_clause_body()?));
        }

        if clauses.is_empty() {
            return Err(self.error("EXISTS subquery requires at least one MATCH clause"));
        }

        if self.current.kind == TokenKind::Where {
            clauses.push(Clause::Where(self.parse_where_clause()?));
        }

        Ok(Query {
            clauses,
            span: None,
        })
    }

    fn parse_where_clause(&mut self) -> Result<WhereClause> {
        self.expect(TokenKind::Where)?;
        let predicate = self.parse_expression()?;
        Ok(WhereClause {
            predicate,
            span: None,
        })
    }

    fn parse_with_clause(&mut self) -> Result<WithClause> {
        self.expect(TokenKind::With)?;

        let distinct = if self.current.kind == TokenKind::Distinct {
            self.advance();
            true
        } else {
            false
        };

        // Check for WITH *
        let is_wildcard = if self.current.kind == TokenKind::Star {
            self.advance();
            true
        } else {
            false
        };

        let items = if is_wildcard {
            Vec::new()
        } else {
            self.parse_projection_items()?
        };

        let where_clause = if self.current.kind == TokenKind::Where {
            Some(Box::new(self.parse_where_clause()?))
        } else {
            None
        };

        Ok(WithClause {
            distinct,
            items,
            is_wildcard,
            where_clause,
            span: None,
        })
    }

    fn parse_return_clause(&mut self) -> Result<ReturnClause> {
        self.expect(TokenKind::Return)?;

        let distinct = if self.current.kind == TokenKind::Distinct {
            self.advance();
            true
        } else {
            false
        };

        let items = if self.current.kind == TokenKind::Star {
            self.advance();
            ReturnItems::All
        } else {
            ReturnItems::Explicit(self.parse_projection_items()?)
        };

        Ok(ReturnClause {
            distinct,
            items,
            span: None,
        })
    }

    fn parse_unwind_clause(&mut self) -> Result<UnwindClause> {
        self.expect(TokenKind::Unwind)?;
        let expression = self.parse_expression()?;
        self.expect(TokenKind::As)?;
        let variable = self.expect_identifier()?;

        Ok(UnwindClause {
            expression,
            variable,
            span: None,
        })
    }

    fn parse_create_clause(&mut self) -> Result<CreateClause> {
        self.expect(TokenKind::Create)?;
        let patterns = self.parse_pattern_list()?;
        Ok(CreateClause {
            patterns,
            span: None,
        })
    }

    fn parse_merge_clause(&mut self) -> Result<MergeClause> {
        self.expect(TokenKind::Merge)?;
        let pattern = self.parse_pattern()?;

        let mut on_create = None;
        let mut on_match = None;

        while self.current.kind == TokenKind::On {
            self.advance();
            match self.current.kind {
                TokenKind::Create => {
                    self.advance();
                    on_create = Some(self.parse_set_clause()?);
                }
                TokenKind::Match => {
                    self.advance();
                    on_match = Some(self.parse_set_clause()?);
                }
                _ => return Err(self.error("Expected CREATE or MATCH after ON")),
            }
        }

        Ok(MergeClause {
            pattern,
            on_create,
            on_match,
            span: None,
        })
    }

    fn parse_delete_clause(&mut self) -> Result<DeleteClause> {
        let detach = if self.current.kind == TokenKind::Detach {
            self.advance();
            true
        } else {
            false
        };

        self.expect(TokenKind::Delete)?;

        let mut expressions = vec![self.parse_expression()?];
        while self.current.kind == TokenKind::Comma {
            self.advance();
            expressions.push(self.parse_expression()?);
        }

        Ok(DeleteClause {
            detach,
            expressions,
            span: None,
        })
    }

    fn parse_set_clause(&mut self) -> Result<SetClause> {
        self.expect(TokenKind::Set)?;

        let mut items = vec![self.parse_set_item()?];
        while self.current.kind == TokenKind::Comma {
            self.advance();
            items.push(self.parse_set_item()?);
        }

        Ok(SetClause { items, span: None })
    }

    fn parse_set_item(&mut self) -> Result<SetItem> {
        let variable = self.expect_identifier()?;

        if self.current.kind == TokenKind::Dot {
            // n.prop = value
            self.advance();
            let property = self.expect_identifier()?;
            self.expect(TokenKind::Eq)?;
            let value = self.parse_expression()?;
            Ok(SetItem::Property {
                variable,
                property,
                value,
            })
        } else if self.current.kind == TokenKind::PlusEq {
            // n += {props}
            self.advance();
            let properties = self.parse_expression()?;
            Ok(SetItem::MergeProperties {
                variable,
                properties,
            })
        } else if self.current.kind == TokenKind::Eq {
            // n = {props}
            self.advance();
            let properties = self.parse_expression()?;
            Ok(SetItem::AllProperties {
                variable,
                properties,
            })
        } else if self.current.kind == TokenKind::Colon {
            // n:Label1:Label2
            let mut labels = Vec::new();
            while self.current.kind == TokenKind::Colon {
                self.advance();
                labels.push(self.expect_identifier()?);
            }
            Ok(SetItem::Labels { variable, labels })
        } else {
            Err(self.error("Expected property assignment or label"))
        }
    }

    fn parse_remove_clause(&mut self) -> Result<RemoveClause> {
        self.expect(TokenKind::Remove)?;

        let mut items = vec![self.parse_remove_item()?];
        while self.current.kind == TokenKind::Comma {
            self.advance();
            items.push(self.parse_remove_item()?);
        }

        Ok(RemoveClause { items, span: None })
    }

    fn parse_remove_item(&mut self) -> Result<RemoveItem> {
        let variable = self.expect_identifier()?;

        if self.current.kind == TokenKind::Dot {
            // n.prop
            self.advance();
            let property = self.expect_identifier()?;
            Ok(RemoveItem::Property { variable, property })
        } else if self.current.kind == TokenKind::Colon {
            // n:Label1:Label2
            let mut labels = Vec::new();
            while self.current.kind == TokenKind::Colon {
                self.advance();
                labels.push(self.expect_identifier()?);
            }
            Ok(RemoveItem::Labels { variable, labels })
        } else {
            Err(self.error("Expected property or label to remove"))
        }
    }

    fn parse_order_by_clause(&mut self) -> Result<OrderByClause> {
        self.expect(TokenKind::Order)?;
        self.expect(TokenKind::By)?;

        let mut items = vec![self.parse_sort_item()?];
        while self.current.kind == TokenKind::Comma {
            self.advance();
            items.push(self.parse_sort_item()?);
        }

        Ok(OrderByClause { items, span: None })
    }

    fn parse_sort_item(&mut self) -> Result<SortItem> {
        let expression = self.parse_expression()?;
        let direction = match self.current.kind {
            TokenKind::Asc | TokenKind::Ascending => {
                self.advance();
                SortDirection::Asc
            }
            TokenKind::Desc | TokenKind::Descending => {
                self.advance();
                SortDirection::Desc
            }
            _ => SortDirection::default(),
        };
        Ok(SortItem {
            expression,
            direction,
        })
    }

    fn parse_projection_items(&mut self) -> Result<Vec<ProjectionItem>> {
        let mut items = vec![self.parse_projection_item()?];
        while self.current.kind == TokenKind::Comma {
            self.advance();
            items.push(self.parse_projection_item()?);
        }
        Ok(items)
    }

    fn parse_projection_item(&mut self) -> Result<ProjectionItem> {
        let expression = self.parse_expression()?;
        let alias = if self.current.kind == TokenKind::As {
            self.advance();
            Some(self.expect_identifier()?)
        } else {
            None
        };
        Ok(ProjectionItem {
            expression,
            alias,
            span: None,
        })
    }

    fn parse_pattern_list(&mut self) -> Result<Vec<Pattern>> {
        let mut patterns = vec![self.parse_pattern()?];
        while self.current.kind == TokenKind::Comma {
            self.advance();
            patterns.push(self.parse_pattern()?);
        }
        Ok(patterns)
    }

    fn parse_pattern(&mut self) -> Result<Pattern> {
        // Check for named path: p = (...)
        // Allow contextual keywords to be used as path variable names
        if self.can_be_identifier() && self.peek_kind() == TokenKind::Eq {
            let name = self.expect_identifier()?;
            self.expect(TokenKind::Eq)?;

            // Check for path function: shortestPath(...) or allShortestPaths(...)
            let (path_function, inner_pattern) = self.parse_path_function_or_pattern()?;

            return Ok(Pattern::NamedPath {
                name,
                path_function,
                pattern: Box::new(inner_pattern),
            });
        }

        let start = self.parse_node_pattern()?;

        // Check for path continuation
        if matches!(
            self.current.kind,
            TokenKind::Arrow | TokenKind::LeftArrow | TokenKind::Minus | TokenKind::DoubleDash
        ) {
            let mut chain = Vec::new();
            while matches!(
                self.current.kind,
                TokenKind::Arrow | TokenKind::LeftArrow | TokenKind::Minus | TokenKind::DoubleDash
            ) {
                chain.push(self.parse_relationship_pattern()?);
            }
            Ok(Pattern::Path(PathPattern {
                start,
                chain,
                span: None,
            }))
        } else {
            Ok(Pattern::Node(start))
        }
    }

    /// Parse an optional path function followed by a pattern.
    /// Handles: `shortestPath(pattern)`, `allShortestPaths(pattern)`, or just `pattern`
    fn parse_path_function_or_pattern(&mut self) -> Result<(Option<PathFunction>, Pattern)> {
        // Check for path function: shortestPath or allShortestPaths
        if self.can_be_identifier() {
            let func_name = self.get_identifier_text().to_lowercase();
            if func_name == "shortestpath" {
                self.advance();
                self.expect(TokenKind::LParen)?;
                let pattern = self.parse_inner_pattern()?;
                self.expect(TokenKind::RParen)?;
                return Ok((Some(PathFunction::ShortestPath), pattern));
            } else if func_name == "allshortestpaths" {
                self.advance();
                self.expect(TokenKind::LParen)?;
                let pattern = self.parse_inner_pattern()?;
                self.expect(TokenKind::RParen)?;
                return Ok((Some(PathFunction::AllShortestPaths), pattern));
            }
        }

        // No path function, just parse the pattern
        let pattern = self.parse_inner_pattern()?;
        Ok((None, pattern))
    }

    /// Parse a pattern without checking for named paths (to avoid recursion).
    fn parse_inner_pattern(&mut self) -> Result<Pattern> {
        let start = self.parse_node_pattern()?;

        // Check for path continuation
        if matches!(
            self.current.kind,
            TokenKind::Arrow | TokenKind::LeftArrow | TokenKind::Minus | TokenKind::DoubleDash
        ) {
            let mut chain = Vec::new();
            while matches!(
                self.current.kind,
                TokenKind::Arrow | TokenKind::LeftArrow | TokenKind::Minus | TokenKind::DoubleDash
            ) {
                chain.push(self.parse_relationship_pattern()?);
            }
            Ok(Pattern::Path(PathPattern {
                start,
                chain,
                span: None,
            }))
        } else {
            Ok(Pattern::Node(start))
        }
    }

    fn parse_node_pattern(&mut self) -> Result<NodePattern> {
        self.expect(TokenKind::LParen)?;

        // Variable can be an identifier or a contextual keyword like 'end'
        let variable = if self.can_be_identifier() && self.current.kind != TokenKind::Colon {
            let name = self.get_identifier_text();
            self.advance();
            Some(name)
        } else {
            None
        };

        let mut labels = Vec::new();
        while self.current.kind == TokenKind::Colon {
            self.advance();
            labels.push(self.expect_identifier()?);
        }

        let properties = if self.current.kind == TokenKind::LBrace {
            self.parse_property_map()?
        } else {
            Vec::new()
        };

        self.expect(TokenKind::RParen)?;

        Ok(NodePattern {
            variable,
            labels,
            properties,
            span: None,
        })
    }

    fn parse_relationship_pattern(&mut self) -> Result<RelationshipPattern> {
        // Parse direction and relationship details
        let (direction, has_bracket) = match self.current.kind {
            TokenKind::Arrow => {
                // ->
                self.advance();
                (Direction::Outgoing, false)
            }
            TokenKind::LeftArrow => {
                // <- possibly followed by - for <-- (anonymous incoming)
                self.advance();
                if self.current.kind == TokenKind::LBracket {
                    // <-[...]- (incoming with bracket details)
                    (Direction::Incoming, true)
                } else if self.current.kind == TokenKind::Minus {
                    // <-- (anonymous incoming shorthand)
                    self.advance();
                    (Direction::Incoming, false)
                } else {
                    (Direction::Incoming, false)
                }
            }
            TokenKind::Minus => {
                // - followed by [ or > or ->
                self.advance();

                if self.current.kind == TokenKind::LBracket {
                    // -[...]- or -[...]->
                    (Direction::Undirected, true) // Direction will be updated based on closing
                } else if self.current.kind == TokenKind::Arrow {
                    // --> (Minus then Arrow): anonymous outgoing shorthand
                    self.advance();
                    (Direction::Outgoing, false)
                } else if self.current.kind == TokenKind::Gt {
                    // -> (Minus then Gt, fallback)
                    self.advance();
                    (Direction::Outgoing, false)
                } else {
                    return Err(self.error("Expected relationship pattern"));
                }
            }
            TokenKind::DoubleDash => {
                // -- or -->
                self.advance();
                if self.current.kind == TokenKind::Gt {
                    // --> anonymous outgoing
                    self.advance();
                    (Direction::Outgoing, false)
                } else {
                    // -- anonymous undirected
                    (Direction::Undirected, false)
                }
            }
            _ => return Err(self.error("Expected relationship pattern")),
        };

        // Parse relationship details [r:TYPE*1..3 {props}]
        let (variable, types, length, properties, rel_where, final_direction) =
            if has_bracket || self.current.kind == TokenKind::LBracket {
                if self.current.kind == TokenKind::LBracket {
                    self.advance();
                }

                // Parse optional variable name - could be followed by : for type
                // Allow contextual keywords like 'end' to be used as variable names
                let var = if self.can_be_identifier() {
                    // Check if this is a variable (followed by :, ], {, *, or WHERE)
                    let is_variable = self.peek_kind() == TokenKind::Colon
                        || self.peek_kind() == TokenKind::RBracket
                        || self.peek_kind() == TokenKind::LBrace
                        || self.peek_kind() == TokenKind::Star
                        || self.peek_kind() == TokenKind::Where;
                    if is_variable {
                        let name = self.get_identifier_text();
                        self.advance();
                        Some(name)
                    } else {
                        None
                    }
                } else {
                    None
                };

                let mut rel_types = Vec::new();
                while self.current.kind == TokenKind::Colon {
                    self.advance();
                    rel_types.push(self.expect_identifier()?);
                    // Handle type alternatives with |
                    while self.current.kind == TokenKind::Pipe {
                        self.advance();
                        rel_types.push(self.expect_identifier()?);
                    }
                }

                // Parse variable length *min..max
                let len = if self.current.kind == TokenKind::Star {
                    self.advance();
                    Some(self.parse_length_range()?)
                } else {
                    None
                };

                let props = if self.current.kind == TokenKind::LBrace {
                    self.parse_property_map()?
                } else {
                    Vec::new()
                };

                // Parse inline WHERE clause: -[r WHERE expr]->
                let where_expr = if self.current.kind == TokenKind::Where {
                    self.advance();
                    Some(self.parse_expression()?)
                } else {
                    None
                };

                self.expect(TokenKind::RBracket)?;

                // Determine direction from closing symbol
                let dir = if self.current.kind == TokenKind::Arrow {
                    self.advance();
                    Direction::Outgoing
                } else if self.current.kind == TokenKind::Minus {
                    self.advance();
                    if direction == Direction::Incoming {
                        Direction::Incoming
                    } else {
                        Direction::Undirected
                    }
                } else {
                    direction
                };

                (var, rel_types, len, props, where_expr, dir)
            } else {
                (None, Vec::new(), None, Vec::new(), None, direction)
            };

        let target = self.parse_node_pattern()?;

        Ok(RelationshipPattern {
            variable,
            types,
            direction: final_direction,
            length,
            properties,
            where_clause: rel_where,
            target,
            span: None,
        })
    }

    fn parse_length_range(&mut self) -> Result<LengthRange> {
        let min = if self.current.kind == TokenKind::Integer {
            let val = self.current.text.parse().unwrap_or(1);
            self.advance();
            Some(val)
        } else {
            None
        };

        let max = if self.current.kind == TokenKind::DotDot {
            self.advance();
            if self.current.kind == TokenKind::Integer {
                let val = self.current.text.parse().unwrap_or(u32::MAX);
                self.advance();
                Some(val)
            } else {
                None // Unbounded
            }
        } else {
            min // If no .., max = min (exact length)
        };

        Ok(LengthRange { min, max })
    }

    fn parse_property_map(&mut self) -> Result<Vec<(String, Expression)>> {
        self.expect(TokenKind::LBrace)?;

        let mut props = Vec::new();

        if self.current.kind != TokenKind::RBrace {
            props.push(self.parse_property_pair()?);
            while self.current.kind == TokenKind::Comma {
                self.advance();
                props.push(self.parse_property_pair()?);
            }
        }

        self.expect(TokenKind::RBrace)?;
        Ok(props)
    }

    fn parse_property_pair(&mut self) -> Result<(String, Expression)> {
        let key = self.expect_identifier()?;
        self.expect(TokenKind::Colon)?;
        let value = self.parse_expression()?;
        Ok((key, value))
    }

    // Expression parsing with precedence climbing
    fn parse_expression(&mut self) -> Result<Expression> {
        self.parse_or_expression()
    }

    fn parse_or_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_xor_expression()?;
        while self.current.kind == TokenKind::Or {
            self.advance();
            let right = self.parse_xor_expression()?;
            left = Expression::Binary {
                left: Box::new(left),
                op: BinaryOp::Or,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_xor_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_and_expression()?;
        while self.current.kind == TokenKind::Xor {
            self.advance();
            let right = self.parse_and_expression()?;
            left = Expression::Binary {
                left: Box::new(left),
                op: BinaryOp::Xor,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_and_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_not_expression()?;
        while self.current.kind == TokenKind::And {
            self.advance();
            let right = self.parse_not_expression()?;
            left = Expression::Binary {
                left: Box::new(left),
                op: BinaryOp::And,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_not_expression(&mut self) -> Result<Expression> {
        if self.current.kind == TokenKind::Not {
            self.advance();
            let operand = self.parse_not_expression()?;
            Ok(Expression::Unary {
                op: UnaryOp::Not,
                operand: Box::new(operand),
            })
        } else {
            self.parse_comparison_expression()
        }
    }

    fn parse_comparison_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_additive_expression()?;

        loop {
            let op = match self.current.kind {
                TokenKind::Eq => BinaryOp::Eq,
                TokenKind::Ne => BinaryOp::Ne,
                TokenKind::Lt => BinaryOp::Lt,
                TokenKind::Le => BinaryOp::Le,
                TokenKind::Gt => BinaryOp::Gt,
                TokenKind::Ge => BinaryOp::Ge,
                TokenKind::In => BinaryOp::In,
                TokenKind::Starts => {
                    self.advance();
                    self.expect(TokenKind::With)?;
                    let right = self.parse_additive_expression()?;
                    left = Expression::Binary {
                        left: Box::new(left),
                        op: BinaryOp::StartsWith,
                        right: Box::new(right),
                    };
                    continue;
                }
                TokenKind::Ends => {
                    self.advance();
                    self.expect(TokenKind::With)?;
                    let right = self.parse_additive_expression()?;
                    left = Expression::Binary {
                        left: Box::new(left),
                        op: BinaryOp::EndsWith,
                        right: Box::new(right),
                    };
                    continue;
                }
                TokenKind::Contains => {
                    self.advance();
                    let right = self.parse_additive_expression()?;
                    left = Expression::Binary {
                        left: Box::new(left),
                        op: BinaryOp::Contains,
                        right: Box::new(right),
                    };
                    continue;
                }
                TokenKind::RegexMatch => BinaryOp::RegexMatch,
                TokenKind::Is => {
                    self.advance();
                    let not = self.current.kind == TokenKind::Not;
                    if not {
                        self.advance();
                    }
                    self.expect(TokenKind::Null)?;
                    left = Expression::Unary {
                        op: if not {
                            UnaryOp::IsNotNull
                        } else {
                            UnaryOp::IsNull
                        },
                        operand: Box::new(left),
                    };
                    continue;
                }
                _ => break,
            };

            self.advance();
            let right = self.parse_additive_expression()?;
            left = Expression::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_additive_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_multiplicative_expression()?;

        loop {
            let op = match self.current.kind {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                _ => break,
            };

            self.advance();
            let right = self.parse_multiplicative_expression()?;
            left = Expression::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_multiplicative_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_power_expression()?;

        loop {
            let op = match self.current.kind {
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::Slash => BinaryOp::Div,
                TokenKind::Percent => BinaryOp::Mod,
                _ => break,
            };

            self.advance();
            let right = self.parse_power_expression()?;
            left = Expression::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_power_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_unary_expression()?;

        if self.current.kind == TokenKind::Caret {
            self.advance();
            let right = self.parse_power_expression()?; // Right associative
            left = Expression::Binary {
                left: Box::new(left),
                op: BinaryOp::Pow,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_unary_expression(&mut self) -> Result<Expression> {
        match self.current.kind {
            TokenKind::Minus => {
                self.advance();
                // Fold -<integer> at parse time to handle i64::MIN correctly
                if self.current.kind == TokenKind::Integer {
                    let text = &self.current.text;
                    if let Ok(val) = format!("-{text}").parse::<i64>() {
                        self.advance();
                        return Ok(Expression::Literal(Literal::Integer(val)));
                    }
                }
                if self.current.kind == TokenKind::Float {
                    let text = &self.current.text;
                    if let Ok(val) = format!("-{text}").parse::<f64>() {
                        self.advance();
                        return Ok(Expression::Literal(Literal::Float(val)));
                    }
                }
                let operand = self.parse_unary_expression()?;
                Ok(Expression::Unary {
                    op: UnaryOp::Neg,
                    operand: Box::new(operand),
                })
            }
            TokenKind::Plus => {
                self.advance();
                let operand = self.parse_unary_expression()?;
                Ok(Expression::Unary {
                    op: UnaryOp::Pos,
                    operand: Box::new(operand),
                })
            }
            _ => self.parse_postfix_expression(),
        }
    }

    fn parse_postfix_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_primary_expression()?;

        loop {
            match self.current.kind {
                TokenKind::Dot => {
                    self.advance();
                    let property = self.expect_identifier()?;
                    expr = Expression::PropertyAccess {
                        base: Box::new(expr),
                        property,
                    };
                }
                TokenKind::LBracket => {
                    self.advance();
                    // Detect slice: [start..end], [start..], [..end], [..]
                    if self.current.kind == TokenKind::DotDot {
                        // [..end] or [..]
                        self.advance();
                        let end = if self.current.kind != TokenKind::RBracket {
                            Some(Box::new(self.parse_expression()?))
                        } else {
                            None
                        };
                        self.expect(TokenKind::RBracket)?;
                        expr = Expression::SliceAccess {
                            base: Box::new(expr),
                            start: None,
                            end,
                        };
                    } else {
                        let index = self.parse_expression()?;
                        if self.current.kind == TokenKind::DotDot {
                            // [start..end] or [start..]
                            self.advance();
                            let end = if self.current.kind != TokenKind::RBracket {
                                Some(Box::new(self.parse_expression()?))
                            } else {
                                None
                            };
                            self.expect(TokenKind::RBracket)?;
                            expr = Expression::SliceAccess {
                                base: Box::new(expr),
                                start: Some(Box::new(index)),
                                end,
                            };
                        } else {
                            // Regular index access
                            self.expect(TokenKind::RBracket)?;
                            expr = Expression::IndexAccess {
                                base: Box::new(expr),
                                index: Box::new(index),
                            };
                        }
                    }
                }
                TokenKind::Colon => {
                    // n:Label label-check syntax, emits hasLabel() function calls.
                    // Multiple labels (n:Person:Actor) are ANDead together.
                    let base = expr;
                    let mut combined: Option<Expression> = None;
                    while self.current.kind == TokenKind::Colon {
                        self.advance();
                        let label = self.expect_identifier()?;
                        let check = Expression::FunctionCall {
                            name: "hasLabel".to_string(),
                            distinct: false,
                            args: vec![base.clone(), Expression::Literal(Literal::String(label))],
                        };
                        combined = Some(match combined {
                            None => check,
                            Some(prev) => Expression::Binary {
                                left: Box::new(prev),
                                op: BinaryOp::And,
                                right: Box::new(check),
                            },
                        });
                    }
                    expr = combined
                        .ok_or_else(|| self.error("expected at least one label after ':'"))?;
                    break;
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_primary_expression(&mut self) -> Result<Expression> {
        match self.current.kind {
            TokenKind::Null => {
                self.advance();
                Ok(Expression::Literal(Literal::Null))
            }
            TokenKind::True => {
                self.advance();
                Ok(Expression::Literal(Literal::Bool(true)))
            }
            TokenKind::False => {
                self.advance();
                Ok(Expression::Literal(Literal::Bool(false)))
            }
            TokenKind::Integer => {
                let text = &self.current.text;
                let value = if text.starts_with("0x") || text.starts_with("0X") {
                    i64::from_str_radix(&text[2..], 16)
                } else if text.starts_with("0o") || text.starts_with("0O") {
                    i64::from_str_radix(&text[2..], 8)
                } else {
                    text.parse()
                }
                .map_err(|e: std::num::ParseIntError| {
                    if *e.kind() == std::num::IntErrorKind::PosOverflow
                        || *e.kind() == std::num::IntErrorKind::NegOverflow
                    {
                        self.error(&format!(
                            "Integer literal '{}' overflows the valid range ({} to {})",
                            text,
                            i64::MIN,
                            i64::MAX
                        ))
                    } else {
                        self.error(&format!("Invalid integer literal: '{}'", text))
                    }
                })?;
                self.advance();
                Ok(Expression::Literal(Literal::Integer(value)))
            }
            TokenKind::Float => {
                let text = &self.current.text;
                let value = text.parse().map_err(|_: std::num::ParseFloatError| {
                    self.error(&format!("Invalid float literal: '{}'", text))
                })?;
                self.advance();
                Ok(Expression::Literal(Literal::Float(value)))
            }
            TokenKind::String => {
                let text = &self.current.text;
                let inner = &text[1..text.len() - 1];
                let value = unescape_string(inner);
                self.advance();
                Ok(Expression::Literal(Literal::String(value)))
            }
            TokenKind::Dollar => {
                self.advance();
                let name = self.expect_identifier()?;
                Ok(Expression::Parameter(name))
            }
            _ if self.can_be_identifier() => {
                let name = self.get_identifier_text();
                let lower = name.to_lowercase();

                // IEEE 754 special float literals (only when not a function call)
                if lower == "nan" && self.peek_kind() != TokenKind::LParen {
                    self.advance();
                    return Ok(Expression::Literal(Literal::Float(f64::NAN)));
                }
                if (lower == "inf" || lower == "infinity") && self.peek_kind() != TokenKind::LParen
                {
                    self.advance();
                    return Ok(Expression::Literal(Literal::Float(f64::INFINITY)));
                }

                // Check for list predicate functions: all, any, none, single
                if matches!(lower.as_str(), "all" | "any" | "none" | "single")
                    && self.peek_kind() == TokenKind::LParen
                {
                    // Tentatively parse as list predicate
                    // Save state so we can fall back to function call if this is not
                    // the `var IN list WHERE pred` form.
                    let saved_lexer = self.lexer.clone();
                    let saved_current = self.current.clone();
                    let saved_previous = self.previous.clone();

                    self.advance(); // consume identifier
                    self.advance(); // consume '('

                    // Expect an identifier (the iteration variable)
                    if self.can_be_identifier() {
                        let variable = self.get_identifier_text();
                        self.advance();

                        if self.current.kind == TokenKind::In {
                            // This is the list predicate form
                            self.advance(); // consume IN
                            let list = self.parse_expression()?;
                            self.expect(TokenKind::Where)?;
                            let predicate = self.parse_expression()?;
                            self.expect(TokenKind::RParen)?;

                            let kind = match lower.as_str() {
                                "all" => ListPredicateKind::All,
                                "any" => ListPredicateKind::Any,
                                "none" => ListPredicateKind::None,
                                "single" => ListPredicateKind::Single,
                                _ => unreachable!(),
                            };

                            return Ok(Expression::ListPredicate {
                                kind,
                                variable,
                                list: Box::new(list),
                                predicate: Box::new(predicate),
                            });
                        }
                    }

                    // Fall back: restore state and parse as regular identifier/function
                    self.lexer = saved_lexer;
                    self.current = saved_current;
                    self.previous = saved_previous;
                }

                self.advance();

                // EXISTS { MATCH ... WHERE ... } subquery form
                if lower == "exists" && self.current.kind == TokenKind::LBrace {
                    self.enter_nesting()?;
                    self.advance(); // consume {
                    let inner_query = self.parse_exists_inner_query()?;
                    self.expect(TokenKind::RBrace)?;
                    self.exit_nesting();
                    return Ok(Expression::Exists(Box::new(inner_query)));
                }

                // COUNT { MATCH ... WHERE ... } subquery form
                if lower == "count" && self.current.kind == TokenKind::LBrace {
                    self.enter_nesting()?;
                    self.advance(); // consume {
                    let inner_query = self.parse_exists_inner_query()?;
                    self.expect(TokenKind::RBrace)?;
                    self.exit_nesting();
                    return Ok(Expression::CountSubquery(Box::new(inner_query)));
                }

                // reduce(acc = init, x IN list | expr)
                if lower == "reduce" && self.current.kind == TokenKind::LParen {
                    let saved = (
                        self.lexer.clone(),
                        self.current.clone(),
                        self.previous.clone(),
                    );
                    self.advance(); // consume (
                    if self.can_be_identifier() {
                        let accumulator = self.get_identifier_text();
                        self.advance();
                        if self.current.kind == TokenKind::Eq {
                            self.advance(); // consume =
                            let initial = self.parse_expression()?;
                            if self.current.kind == TokenKind::Comma {
                                self.advance(); // consume ,
                                let variable = self.get_identifier_text();
                                self.advance();
                                self.expect(TokenKind::In)?;
                                let list = self.parse_expression()?;
                                self.expect(TokenKind::Pipe)?;
                                let expression = self.parse_expression()?;
                                self.expect(TokenKind::RParen)?;
                                return Ok(Expression::Reduce {
                                    accumulator,
                                    initial: Box::new(initial),
                                    variable,
                                    list: Box::new(list),
                                    expression: Box::new(expression),
                                });
                            }
                        }
                    }
                    // Not valid reduce syntax, restore and fall through
                    (self.lexer, self.current, self.previous) = saved;
                }

                // Map projection: variable { .prop, key: expr, .* }
                if self.current.kind == TokenKind::LBrace {
                    let saved = (
                        self.lexer.clone(),
                        self.current.clone(),
                        self.previous.clone(),
                    );
                    self.advance(); // consume {
                    if self.current.kind == TokenKind::Dot || self.current.kind == TokenKind::RBrace
                    {
                        let mut entries = Vec::new();
                        while self.current.kind != TokenKind::RBrace {
                            if self.current.kind == TokenKind::Dot {
                                self.advance();
                                if self.current.kind == TokenKind::Star {
                                    self.advance();
                                    entries.push(MapProjectionEntry::AllProperties);
                                } else if self.can_be_identifier() {
                                    let prop = self.get_identifier_text();
                                    self.advance();
                                    entries.push(MapProjectionEntry::PropertySelector(prop));
                                } else {
                                    break;
                                }
                            } else if self.can_be_identifier() {
                                let key = self.get_identifier_text();
                                self.advance();
                                self.expect(TokenKind::Colon)?;
                                let value = self.parse_expression()?;
                                entries.push(MapProjectionEntry::LiteralEntry(key, value));
                            } else {
                                break;
                            }
                            if self.current.kind == TokenKind::Comma {
                                self.advance();
                            }
                        }
                        if self.current.kind == TokenKind::RBrace {
                            self.advance();
                            return Ok(Expression::MapProjection {
                                base: name,
                                entries,
                            });
                        }
                    }
                    // Not valid map projection, restore
                    (self.lexer, self.current, self.previous) = saved;
                }

                // Check if function call
                if self.current.kind == TokenKind::LParen {
                    self.advance();
                    let distinct = if self.current.kind == TokenKind::Distinct {
                        self.advance();
                        true
                    } else {
                        false
                    };

                    let mut args = Vec::new();
                    // Handle count(*) special case
                    if self.current.kind == TokenKind::Star {
                        self.advance();
                        args.push(Expression::Variable("*".to_string()));
                    } else if self.current.kind != TokenKind::RParen {
                        args.push(self.parse_expression()?);
                        while self.current.kind == TokenKind::Comma {
                            self.advance();
                            args.push(self.parse_expression()?);
                        }
                    }
                    self.expect(TokenKind::RParen)?;

                    Ok(Expression::FunctionCall {
                        name,
                        distinct,
                        args,
                    })
                } else {
                    Ok(Expression::Variable(name))
                }
            }
            TokenKind::LParen => {
                self.enter_nesting()?;
                self.advance();
                let expr = self.parse_expression()?;
                self.expect(TokenKind::RParen)?;
                self.exit_nesting();
                Ok(expr)
            }
            TokenKind::LBracket => {
                self.enter_nesting()?;
                self.advance();

                // Detect pattern comprehension: [(pattern) WHERE pred | expr]
                // A pattern starts with `(`, while list elements starting with `(`
                // are parenthesized expressions. We use backtracking to distinguish.
                if self.current.kind == TokenKind::LParen {
                    let saved = (
                        self.lexer.clone(),
                        self.current.clone(),
                        self.previous.clone(),
                    );
                    if let Ok(pattern) = self.parse_pattern() {
                        let where_clause = if self.current.kind == TokenKind::Where {
                            self.advance();
                            Some(Box::new(self.parse_expression()?))
                        } else {
                            None
                        };
                        if self.current.kind == TokenKind::Pipe {
                            self.advance();
                            let projection = self.parse_expression()?;
                            self.expect(TokenKind::RBracket)?;
                            self.exit_nesting();
                            return Ok(Expression::PatternComprehension {
                                pattern: Box::new(pattern),
                                where_clause,
                                projection: Box::new(projection),
                            });
                        }
                    }
                    // Not a pattern comprehension, restore and continue
                    (self.lexer, self.current, self.previous) = saved;
                }

                // Detect list comprehension: [var IN list WHERE pred | expr]
                if self.can_be_identifier() && self.peek_kind() == TokenKind::In {
                    let variable = self.get_identifier_text();
                    self.advance(); // consume variable
                    self.advance(); // consume IN
                    let list = self.parse_expression()?;

                    let filter = if self.current.kind == TokenKind::Where {
                        self.advance();
                        Some(Box::new(self.parse_expression()?))
                    } else {
                        None
                    };

                    let projection = if self.current.kind == TokenKind::Pipe {
                        self.advance();
                        Some(Box::new(self.parse_expression()?))
                    } else {
                        None
                    };

                    self.expect(TokenKind::RBracket)?;
                    self.exit_nesting();
                    return Ok(Expression::ListComprehension {
                        variable,
                        list: Box::new(list),
                        filter,
                        projection,
                    });
                }

                // List literal
                let mut items = Vec::new();
                if self.current.kind != TokenKind::RBracket {
                    items.push(self.parse_expression()?);
                    while self.current.kind == TokenKind::Comma {
                        self.advance();
                        items.push(self.parse_expression()?);
                    }
                }
                self.expect(TokenKind::RBracket)?;
                self.exit_nesting();
                Ok(Expression::List(items))
            }
            TokenKind::LBrace => {
                // Map literal
                self.advance();
                let mut pairs = Vec::new();
                if self.current.kind != TokenKind::RBrace {
                    let key = self.expect_identifier()?;
                    self.expect(TokenKind::Colon)?;
                    let value = self.parse_expression()?;
                    pairs.push((key, value));

                    while self.current.kind == TokenKind::Comma {
                        self.advance();
                        let key = self.expect_identifier()?;
                        self.expect(TokenKind::Colon)?;
                        let value = self.parse_expression()?;
                        pairs.push((key, value));
                    }
                }
                self.expect(TokenKind::RBrace)?;
                Ok(Expression::Map(pairs))
            }
            TokenKind::Case => {
                self.advance();
                self.parse_case_expression()
            }
            // COUNT aggregate or COUNT { subquery }
            TokenKind::Count => {
                self.advance();
                if self.current.kind == TokenKind::LBrace {
                    // COUNT { MATCH ... WHERE ... } subquery
                    self.advance(); // consume {
                    let inner_query = self.parse_subquery_body()?;
                    self.expect(TokenKind::RBrace)?;
                    Ok(Expression::CountSubquery(Box::new(inner_query)))
                } else {
                    self.parse_aggregate_function("count")
                }
            }
            _ => Err(self.error("Expected expression")),
        }
    }

    fn parse_aggregate_function(&mut self, name: &str) -> Result<Expression> {
        self.expect(TokenKind::LParen)?;

        let distinct = if self.current.kind == TokenKind::Distinct {
            self.advance();
            true
        } else {
            false
        };

        let mut args = Vec::new();
        // Handle COUNT(*) special case
        if self.current.kind == TokenKind::Star {
            self.advance();
            // For COUNT(*), we use a special marker
            args.push(Expression::Variable("*".to_string()));
        } else if self.current.kind != TokenKind::RParen {
            args.push(self.parse_expression()?);
            while self.current.kind == TokenKind::Comma {
                self.advance();
                args.push(self.parse_expression()?);
            }
        }

        self.expect(TokenKind::RParen)?;

        Ok(Expression::FunctionCall {
            name: name.to_string(),
            distinct,
            args,
        })
    }

    fn parse_case_expression(&mut self) -> Result<Expression> {
        self.enter_nesting()?;

        let input = if self.current.kind != TokenKind::When {
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };

        let mut whens = Vec::new();
        while self.current.kind == TokenKind::When {
            self.advance();
            let when_expr = self.parse_expression()?;
            self.expect(TokenKind::Then)?;
            let then_expr = self.parse_expression()?;
            whens.push((when_expr, then_expr));
        }

        let else_clause = if self.current.kind == TokenKind::Else {
            self.advance();
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };

        self.expect(TokenKind::End)?;
        self.exit_nesting();

        Ok(Expression::Case {
            input,
            whens,
            else_clause,
        })
    }

    // Helper methods
    fn advance(&mut self) {
        self.previous = std::mem::replace(&mut self.current, self.lexer.next_token());
    }

    fn expect(&mut self, kind: TokenKind) -> Result<()> {
        if self.current.kind == kind {
            self.advance();
            Ok(())
        } else {
            Err(self.error(&format!("Expected {:?}", kind)))
        }
    }

    fn expect_identifier(&mut self) -> Result<String> {
        if self.can_be_identifier() {
            let text = self.get_identifier_text();
            self.advance();
            Ok(text)
        } else {
            Err(self.error("Expected identifier"))
        }
    }

    /// Check if the current token can be used as an identifier.
    /// This includes true identifiers and contextual keywords that can be used as names.
    fn can_be_identifier(&self) -> bool {
        matches!(
            self.current.kind,
            TokenKind::Identifier
                | TokenKind::QuotedIdentifier
                // Contextual keywords that can be used as identifiers
                | TokenKind::End
                | TokenKind::Count
                | TokenKind::Starts
                | TokenKind::Ends
                | TokenKind::Contains
                | TokenKind::All
                | TokenKind::Asc
                | TokenKind::Desc
                | TokenKind::Ascending
                | TokenKind::Descending
                | TokenKind::On
                | TokenKind::Call
                | TokenKind::Yield
                | TokenKind::Exists
                | TokenKind::Order
                | TokenKind::By
                | TokenKind::Skip
                | TokenKind::Limit
        )
    }

    /// Get the text of the current token as an identifier.
    fn get_identifier_text(&self) -> String {
        let mut text = self.current.text.clone();
        // Remove backticks from quoted identifier
        if self.current.kind == TokenKind::QuotedIdentifier {
            text = text[1..text.len() - 1].to_string();
        }
        text
    }

    fn peek_kind(&mut self) -> TokenKind {
        // Lookahead - we need to save and restore state
        let saved_pos = self.lexer.clone();
        let token = self.lexer.next_token();
        let kind = token.kind;
        self.lexer = saved_pos;
        kind
    }

    fn peek_text_eq_ignore_case(&mut self, expected: &str) -> bool {
        let saved_pos = self.lexer.clone();
        let token = self.lexer.next_token();
        let matches = token.text.eq_ignore_ascii_case(expected);
        self.lexer = saved_pos;
        matches
    }

    fn skip_semicolons(&mut self) {
        while self.current.kind == TokenKind::Semicolon {
            self.advance();
        }
    }

    /// Expects the current token to be an identifier matching `expected` (case-insensitive).
    fn expect_contextual(&mut self, expected: &str) -> Result<()> {
        if self.can_be_identifier() && self.get_identifier_text().eq_ignore_ascii_case(expected) {
            self.advance();
            Ok(())
        } else {
            Err(self.error(&format!("Expected '{expected}'")))
        }
    }

    /// Checks if the current token is a contextual keyword (case-insensitive).
    fn is_contextual(&self, keyword: &str) -> bool {
        self.current.kind == TokenKind::Identifier
            && self.current.text.eq_ignore_ascii_case(keyword)
    }

    /// Parses IF NOT EXISTS, returns true if present.
    fn parse_if_not_exists(&mut self) -> bool {
        if self.current.kind == TokenKind::Identifier
            && self.current.text.eq_ignore_ascii_case("IF")
        {
            let saved = self.lexer.clone();
            let saved_cur = self.current.clone();
            self.advance();
            if self.current.kind == TokenKind::Not {
                self.advance();
                if self.current.kind == TokenKind::Exists {
                    self.advance();
                    return true;
                }
            }
            // Rewind
            self.lexer = saved;
            self.current = saved_cur;
        }
        false
    }

    /// Parses IF EXISTS, returns true if present.
    fn parse_if_exists(&mut self) -> bool {
        if self.current.kind == TokenKind::Identifier
            && self.current.text.eq_ignore_ascii_case("IF")
        {
            let saved = self.lexer.clone();
            let saved_cur = self.current.clone();
            self.advance();
            if self.current.kind == TokenKind::Exists {
                self.advance();
                return true;
            }
            // Rewind
            self.lexer = saved;
            self.current = saved_cur;
        }
        false
    }

    // ========================================================================
    // Schema DDL parsing
    // ========================================================================

    /// Parses CREATE INDEX or CREATE CONSTRAINT.
    fn parse_schema_create(&mut self) -> Result<Statement> {
        self.expect(TokenKind::Create)?; // consume CREATE

        if self.is_contextual("INDEX") {
            self.advance();
            self.parse_create_index()
        } else if self.is_contextual("CONSTRAINT") {
            self.advance();
            self.parse_create_constraint()
        } else {
            Err(self.error("Expected INDEX or CONSTRAINT after CREATE"))
        }
    }

    /// CREATE INDEX name [IF NOT EXISTS] FOR (n:Label) ON (n.property)
    fn parse_create_index(&mut self) -> Result<Statement> {
        let name = self.expect_identifier()?;
        let if_not_exists = self.parse_if_not_exists();

        self.expect_contextual("FOR")?;
        self.expect(TokenKind::LParen)?;
        let _var = self.expect_identifier()?;
        self.expect(TokenKind::Colon)?;
        let label = self.expect_identifier()?;
        self.expect(TokenKind::RParen)?;

        self.expect_contextual("ON")?;
        self.expect(TokenKind::LParen)?;
        let mut properties = Vec::new();
        loop {
            let _prop_var = self.expect_identifier()?;
            self.expect(TokenKind::Dot)?;
            properties.push(self.expect_identifier()?);
            if self.current.kind != TokenKind::Comma {
                break;
            }
            self.advance();
        }
        self.expect(TokenKind::RParen)?;

        use crate::query::schema as gql;
        Ok(Statement::Schema(gql::SchemaStatement::CreateIndex(
            gql::CreateIndexStatement {
                name,
                index_kind: gql::IndexKind::Property,
                label,
                properties,
                options: gql::IndexOptions::default(),
                if_not_exists,
                span: None,
            },
        )))
    }

    /// CREATE CONSTRAINT name [IF NOT EXISTS] FOR (n:Label) REQUIRE n.property IS UNIQUE
    fn parse_create_constraint(&mut self) -> Result<Statement> {
        let name = self.expect_identifier()?;
        let if_not_exists = self.parse_if_not_exists();

        self.expect_contextual("FOR")?;
        self.expect(TokenKind::LParen)?;
        let var = self.expect_identifier()?;
        self.expect(TokenKind::Colon)?;
        let label = self.expect_identifier()?;
        self.expect(TokenKind::RParen)?;

        self.expect_contextual("REQUIRE")?;

        // Parse property references: n.prop1, n.prop2, ...
        // or (n.prop1, n.prop2, ...)
        let has_parens = self.current.kind == TokenKind::LParen;
        if has_parens {
            self.advance();
        }
        let mut properties = Vec::new();
        loop {
            // Expect var.prop or just prop
            if self.can_be_identifier()
                && self.get_identifier_text().eq_ignore_ascii_case(&var)
                && self.peek_kind() == TokenKind::Dot
            {
                self.advance(); // skip var
                self.advance(); // skip dot
            }
            properties.push(self.expect_identifier()?);
            if self.current.kind != TokenKind::Comma {
                break;
            }
            self.advance();
        }
        if has_parens {
            self.expect(TokenKind::RParen)?;
        }

        // Parse constraint kind: IS UNIQUE, IS NOT NULL, IS NODE KEY
        use crate::query::schema as gql;
        let constraint_kind = if self.current.kind == TokenKind::Is {
            self.advance();
            if self.is_contextual("UNIQUE") {
                self.advance();
                gql::ConstraintKind::Unique
            } else if self.current.kind == TokenKind::Not {
                self.advance();
                self.expect(TokenKind::Null)?;
                gql::ConstraintKind::NotNull
            } else if self.is_contextual("NODE") {
                self.advance();
                self.expect_contextual("KEY")?;
                gql::ConstraintKind::NodeKey
            } else {
                return Err(self.error("Expected UNIQUE, NOT NULL, or NODE KEY after IS"));
            }
        } else {
            return Err(self.error("Expected IS after property reference"));
        };

        Ok(Statement::Schema(gql::SchemaStatement::CreateConstraint(
            gql::CreateConstraintStatement {
                name: Some(name),
                constraint_kind,
                label,
                properties,
                if_not_exists,
                span: None,
            },
        )))
    }

    /// DROP INDEX name [IF EXISTS] / DROP CONSTRAINT name [IF EXISTS]
    fn try_parse_schema_drop(&mut self) -> Result<Option<Statement>> {
        // Current token is "DROP" identifier
        let saved_lexer = self.lexer.clone();
        let saved_cur = self.current.clone();
        self.advance(); // consume DROP

        use crate::query::schema as gql;

        if self.is_contextual("INDEX") {
            self.advance();
            let name = self.expect_identifier()?;
            let if_exists = self.parse_if_exists();
            Ok(Some(Statement::Schema(gql::SchemaStatement::DropIndex {
                name,
                if_exists,
            })))
        } else if self.is_contextual("CONSTRAINT") {
            self.advance();
            let name = self.expect_identifier()?;
            let if_exists = self.parse_if_exists();
            Ok(Some(Statement::Schema(
                gql::SchemaStatement::DropConstraint { name, if_exists },
            )))
        } else {
            // Not a schema DROP, rewind
            self.lexer = saved_lexer;
            self.current = saved_cur;
            Ok(None)
        }
    }

    /// SHOW INDEXES / SHOW CONSTRAINTS / SHOW CURRENT GRAPH TYPE
    fn try_parse_show(&mut self) -> Result<Option<Statement>> {
        let saved_lexer = self.lexer.clone();
        let saved_cur = self.current.clone();
        self.advance(); // consume SHOW

        if self.is_contextual("INDEXES") || self.is_contextual("INDEX") {
            self.advance();
            Ok(Some(Statement::ShowIndexes))
        } else if self.is_contextual("CONSTRAINTS") || self.is_contextual("CONSTRAINT") {
            self.advance();
            Ok(Some(Statement::ShowConstraints))
        } else if self.is_contextual("CURRENT") {
            // SHOW CURRENT GRAPH TYPE
            self.advance(); // consume CURRENT
            self.expect_contextual("GRAPH")?;
            self.expect_contextual("TYPE")?;
            Ok(Some(Statement::ShowCurrentGraphType))
        } else {
            // Not a SHOW command, rewind
            self.lexer = saved_lexer;
            self.current = saved_cur;
            Ok(None)
        }
    }

    /// ALTER CURRENT GRAPH TYPE SET/ADD/DROP { ... }
    ///
    /// Neo4j uses `ALTER CURRENT GRAPH TYPE` instead of GQL's `ALTER GRAPH TYPE <name>`.
    /// Maps to `AlterGraphTypeStatement` with the sentinel name `__current__`.
    ///
    /// - `SET { ... }` replaces the entire graph type (or_replace CREATE).
    /// - `ADD { ... }` adds node/edge types via AlterGraphType alterations.
    /// - `DROP { ... }` removes node/edge types via AlterGraphType alterations.
    fn try_parse_alter_current_graph_type(&mut self) -> Result<Option<Statement>> {
        let saved_lexer = self.lexer.clone();
        let saved_cur = self.current.clone();
        self.advance(); // consume ALTER

        // Check for CURRENT GRAPH TYPE sequence
        if !self.is_contextual("CURRENT") {
            self.lexer = saved_lexer;
            self.current = saved_cur;
            return Ok(None);
        }
        self.advance(); // consume CURRENT

        if !self.is_contextual("GRAPH") {
            self.lexer = saved_lexer;
            self.current = saved_cur;
            return Ok(None);
        }
        self.advance(); // consume GRAPH

        self.expect_contextual("TYPE")?;

        use crate::query::schema as gql;

        // Determine the operation: SET, ADD, or DROP
        if self.current.kind == TokenKind::Set {
            self.advance(); // consume SET
            let (node_types, edge_types) = self.parse_inline_graph_type_body()?;
            // SET replaces the entire graph type: map to CreateGraphType with or_replace=true
            Ok(Some(Statement::Schema(
                gql::SchemaStatement::CreateGraphType(gql::CreateGraphTypeStatement {
                    name: "__current__".to_string(),
                    node_types: node_types.iter().map(|t| t.name().to_string()).collect(),
                    edge_types: edge_types.iter().map(|t| t.name().to_string()).collect(),
                    inline_types: node_types.into_iter().chain(edge_types).collect(),
                    like_graph: None,
                    open: false,
                    if_not_exists: false,
                    or_replace: true,
                    span: None,
                }),
            )))
        } else if self.is_contextual("ADD") {
            self.advance(); // consume ADD
            let (node_types, edge_types) = self.parse_inline_graph_type_body()?;
            let mut alterations = Vec::new();
            for nt in &node_types {
                alterations.push(gql::GraphTypeAlteration::AddNodeType(nt.name().to_string()));
            }
            for et in &edge_types {
                alterations.push(gql::GraphTypeAlteration::AddEdgeType(et.name().to_string()));
            }
            Ok(Some(Statement::Schema(
                gql::SchemaStatement::AlterGraphType(gql::AlterGraphTypeStatement {
                    name: "__current__".to_string(),
                    alterations,
                    span: None,
                }),
            )))
        } else if self.current.kind == TokenKind::Identifier
            && self.current.text.eq_ignore_ascii_case("DROP")
        {
            self.advance(); // consume DROP
            let (node_types, edge_types) = self.parse_inline_graph_type_body()?;
            let mut alterations = Vec::new();
            for nt in &node_types {
                alterations.push(gql::GraphTypeAlteration::DropNodeType(
                    nt.name().to_string(),
                ));
            }
            for et in &edge_types {
                alterations.push(gql::GraphTypeAlteration::DropEdgeType(
                    et.name().to_string(),
                ));
            }
            Ok(Some(Statement::Schema(
                gql::SchemaStatement::AlterGraphType(gql::AlterGraphTypeStatement {
                    name: "__current__".to_string(),
                    alterations,
                    span: None,
                }),
            )))
        } else {
            Err(self.error("Expected SET, ADD, or DROP after ALTER CURRENT GRAPH TYPE"))
        }
    }

    /// Parses the Neo4j inline graph type body: `{ type_elements, ... }`.
    ///
    /// Each element is one of:
    /// - `(:Label { prop TYPE [NOT NULL], ... })` for a node type
    /// - `(:Src)-[:Type { prop TYPE, ... }]->(:Dst)` for an edge type with endpoints
    /// - `(:Label)` for a node type without properties
    ///
    /// Property types support both `prop TYPE` (GQL) and `prop :: TYPE` (Neo4j) syntax.
    fn parse_inline_graph_type_body(
        &mut self,
    ) -> Result<(
        Vec<crate::query::schema::InlineElementType>,
        Vec<crate::query::schema::InlineElementType>,
    )> {
        use crate::query::schema::InlineElementType;

        self.expect(TokenKind::LBrace)?;

        let mut node_types = Vec::new();
        let mut edge_types = Vec::new();

        while self.current.kind != TokenKind::RBrace && self.current.kind != TokenKind::Eof {
            // Each element starts with a node pattern: (...)
            let element = self.parse_inline_type_element()?;
            match element {
                InlineElementType::Node { .. } => node_types.push(element),
                InlineElementType::Edge { .. } => edge_types.push(element),
            }

            // Optional comma between elements
            if self.current.kind == TokenKind::Comma {
                self.advance();
            }
        }

        self.expect(TokenKind::RBrace)?;
        Ok((node_types, edge_types))
    }

    /// Parses a single inline type element.
    ///
    /// Starting from `(`, determines whether it is a node type or an edge type
    /// by looking for a following `-[` pattern.
    fn parse_inline_type_element(&mut self) -> Result<crate::query::schema::InlineElementType> {
        use crate::query::schema::InlineElementType;

        self.expect(TokenKind::LParen)?;

        // Parse the node label: :Label
        self.expect(TokenKind::Colon)?;
        let label = self.expect_identifier()?;

        // Parse optional properties: { prop TYPE [NOT NULL], ... }
        let properties = if self.current.kind == TokenKind::LBrace {
            self.parse_inline_property_defs()?
        } else {
            Vec::new()
        };

        self.expect(TokenKind::RParen)?;

        // Check if this is an edge pattern: (...)-[:TYPE]->(...) or (...)<-[:TYPE]-(...)
        if self.current.kind == TokenKind::Minus
            || self.current.kind == TokenKind::Arrow
            || self.current.kind == TokenKind::LeftArrow
            || self.current.kind == TokenKind::DoubleDash
        {
            // This node is the source of an edge type
            let source_label = label;
            let source_props = properties;

            // Parse the edge direction and type
            self.parse_inline_edge_type(source_label, source_props)
        } else {
            // Stand-alone node type
            Ok(InlineElementType::Node {
                name: label,
                properties,
                key_labels: Vec::new(),
                is_reference: false,
            })
        }
    }

    /// Parses an inline edge type pattern after the source node has been parsed.
    ///
    /// Handles: `(...)-[:TYPE { props }]->(:Dst)` and `(...)<-[:TYPE { props }]-(:Dst)`
    fn parse_inline_edge_type(
        &mut self,
        source_label: String,
        _source_props: Vec<crate::query::schema::PropertyDefinition>,
    ) -> Result<crate::query::schema::InlineElementType> {
        use crate::query::schema::InlineElementType;

        // Determine direction and consume leading arrow part
        let (is_outgoing, _is_undirected) = match self.current.kind {
            TokenKind::Minus => {
                // - followed by [ => outgoing: -[:TYPE]-> or undirected: -[:TYPE]-
                self.advance(); // consume -
                (true, false)
            }
            TokenKind::LeftArrow => {
                // <- is incoming: <-[:TYPE]-
                self.advance(); // consume <-
                (false, false)
            }
            TokenKind::DoubleDash => {
                // -- undirected (edge without brackets)
                self.advance();
                (true, true)
            }
            _ => return Err(self.error("Expected edge pattern (-, <-, or --)")),
        };

        // Parse edge type: [:TYPE { props }]
        self.expect(TokenKind::LBracket)?;
        self.expect(TokenKind::Colon)?;
        let edge_label = self.expect_identifier()?;

        let edge_props = if self.current.kind == TokenKind::LBrace {
            self.parse_inline_property_defs()?
        } else {
            Vec::new()
        };

        self.expect(TokenKind::RBracket)?;

        // Consume trailing arrow
        if self.current.kind == TokenKind::Arrow {
            self.advance(); // consume ->
        } else if self.current.kind == TokenKind::Minus {
            self.advance(); // consume - (undirected or incoming closing)
        }

        // Parse target node: (:Label)
        self.expect(TokenKind::LParen)?;
        self.expect(TokenKind::Colon)?;
        let target_label = self.expect_identifier()?;
        self.expect(TokenKind::RParen)?;

        let (src_types, tgt_types) = if is_outgoing {
            (vec![source_label], vec![target_label])
        } else {
            (vec![target_label], vec![source_label])
        };

        Ok(InlineElementType::Edge {
            name: edge_label,
            properties: edge_props,
            key_labels: Vec::new(),
            source_node_types: src_types,
            target_node_types: tgt_types,
            is_reference: false,
        })
    }

    /// Parses inline property definitions: `{ prop TYPE [NOT NULL], ... }`
    ///
    /// Supports both GQL-style (`prop TYPE`) and Neo4j-style (`prop :: TYPE`).
    fn parse_inline_property_defs(
        &mut self,
    ) -> Result<Vec<crate::query::schema::PropertyDefinition>> {
        use crate::query::schema::PropertyDefinition;

        self.expect(TokenKind::LBrace)?;

        let mut props = Vec::new();

        while self.current.kind != TokenKind::RBrace && self.current.kind != TokenKind::Eof {
            let prop_name = self.expect_identifier()?;

            // Support both `prop TYPE` and `prop :: TYPE` syntax
            if self.current.kind == TokenKind::Colon && self.peek_kind() == TokenKind::Colon {
                self.advance(); // consume first :
                self.advance(); // consume second :
            }

            let data_type = self.expect_identifier()?;

            // Optional NOT NULL
            let nullable = if self.current.kind == TokenKind::Not {
                self.advance();
                self.expect(TokenKind::Null)?;
                false
            } else {
                true
            };

            props.push(PropertyDefinition {
                name: prop_name,
                data_type,
                nullable,
                default_value: None,
            });

            // Optional comma
            if self.current.kind == TokenKind::Comma {
                self.advance();
            }
        }

        self.expect(TokenKind::RBrace)?;
        Ok(props)
    }

    fn error(&self, message: &str) -> grafeo_common::utils::error::Error {
        QueryError::new(QueryErrorKind::Syntax, format!("[Cypher] {message}"))
            .with_span(self.current.span)
            .with_source(self.source.to_string())
            .into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to parse and expect success
    fn parse_ok(query: &str) -> Statement {
        let mut parser = Parser::new(query);
        parser
            .parse()
            .unwrap_or_else(|_| panic!("Failed to parse: {query}"))
    }

    // Helper to parse and expect failure
    fn parse_err(query: &str) {
        let mut parser = Parser::new(query);
        assert!(
            parser.parse().is_err(),
            "Expected parse error for: {}",
            query
        );
    }

    // ==================== MATCH Clause Tests ====================

    #[test]
    fn test_parse_simple_match() {
        let stmt = parse_ok("MATCH (n) RETURN n");
        assert!(matches!(stmt, Statement::Query(_)));
    }

    #[test]
    fn test_parse_match_with_label() {
        let stmt = parse_ok("MATCH (n:Person) RETURN n");
        if let Statement::Query(Query { clauses, .. }) = stmt {
            assert!(matches!(&clauses[0], Clause::Match(_)));
        } else {
            panic!("Expected Query");
        }
    }

    #[test]
    fn test_parse_match_with_multiple_labels() {
        let stmt = parse_ok("MATCH (n:Person:Employee) RETURN n");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Match(MatchClause { patterns, .. }) = &clauses[0]
        {
            if let Pattern::Node(node) = &patterns[0] {
                assert_eq!(node.labels.len(), 2);
                assert_eq!(node.labels[0], "Person");
                assert_eq!(node.labels[1], "Employee");
            } else {
                panic!("Expected Node pattern");
            }
        }
    }

    #[test]
    fn test_parse_match_with_properties() {
        let stmt = parse_ok("MATCH (n:Person {name: 'Alix', age: 30}) RETURN n");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Match(MatchClause { patterns, .. }) = &clauses[0]
        {
            if let Pattern::Node(node) = &patterns[0] {
                assert_eq!(node.properties.len(), 2);
                assert_eq!(node.properties[0].0, "name");
                assert_eq!(node.properties[1].0, "age");
            } else {
                panic!("Expected Node pattern");
            }
        }
    }

    #[test]
    fn test_parse_match_with_variable() {
        let stmt = parse_ok("MATCH (person:Person) RETURN person");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Match(MatchClause { patterns, .. }) = &clauses[0]
        {
            if let Pattern::Node(node) = &patterns[0] {
                assert_eq!(node.variable, Some("person".to_string()));
            } else {
                panic!("Expected Node pattern");
            }
        }
    }

    // ==================== Path Pattern Tests ====================

    #[test]
    fn test_parse_outgoing_edge() {
        let stmt = parse_ok("MATCH (a)-[:KNOWS]->(b) RETURN a, b");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Match(MatchClause { patterns, .. }) = &clauses[0]
        {
            if let Pattern::Path(path) = &patterns[0] {
                assert_eq!(path.chain.len(), 1);
                assert_eq!(path.chain[0].direction, Direction::Outgoing);
                assert_eq!(path.chain[0].types, vec!["KNOWS"]);
            } else {
                panic!("Expected Path pattern");
            }
        }
    }

    #[test]
    fn test_parse_incoming_edge() {
        let stmt = parse_ok("MATCH (a)<-[:KNOWS]-(b) RETURN a, b");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Match(MatchClause { patterns, .. }) = &clauses[0]
        {
            if let Pattern::Path(path) = &patterns[0] {
                assert_eq!(path.chain[0].direction, Direction::Incoming);
            } else {
                panic!("Expected Path pattern");
            }
        }
    }

    #[test]
    fn test_parse_undirected_edge() {
        let stmt = parse_ok("MATCH (a)-[:KNOWS]-(b) RETURN a, b");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Match(MatchClause { patterns, .. }) = &clauses[0]
        {
            if let Pattern::Path(path) = &patterns[0] {
                assert_eq!(path.chain[0].direction, Direction::Undirected);
            } else {
                panic!("Expected Path pattern");
            }
        }
    }

    #[test]
    fn test_parse_variable_length_edge() {
        let stmt = parse_ok("MATCH (a)-[:KNOWS*1..3]->(b) RETURN a, b");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Match(MatchClause { patterns, .. }) = &clauses[0]
        {
            if let Pattern::Path(path) = &patterns[0] {
                let length = path.chain[0].length.as_ref().unwrap();
                assert_eq!(length.min, Some(1));
                assert_eq!(length.max, Some(3));
            } else {
                panic!("Expected Path pattern");
            }
        }
    }

    #[test]
    fn test_parse_variable_length_unbounded() {
        let stmt = parse_ok("MATCH (a)-[:KNOWS*]->(b) RETURN a, b");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Match(MatchClause { patterns, .. }) = &clauses[0]
        {
            if let Pattern::Path(path) = &patterns[0] {
                let length = path.chain[0].length.as_ref().unwrap();
                assert_eq!(length.min, None);
                assert_eq!(length.max, None);
            } else {
                panic!("Expected Path pattern");
            }
        }
    }

    #[test]
    fn test_parse_relationship_where() {
        let stmt = parse_ok("MATCH (a)-[r:KNOWS WHERE r.since > 2020]->(b) RETURN a, b");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Match(MatchClause { patterns, .. }) = &clauses[0]
        {
            if let Pattern::Path(path) = &patterns[0] {
                assert!(
                    path.chain[0].where_clause.is_some(),
                    "Expected inline WHERE on relationship"
                );
                assert!(
                    path.chain[0].variable.as_deref() == Some("r"),
                    "Expected variable 'r'"
                );
            } else {
                panic!("Expected Path pattern");
            }
        }
    }

    #[test]
    fn test_parse_relationship_where_no_type() {
        let stmt = parse_ok("MATCH (a)-[r WHERE r.weight > 0.5]->(b) RETURN a, b");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Match(MatchClause { patterns, .. }) = &clauses[0]
        {
            if let Pattern::Path(path) = &patterns[0] {
                assert!(path.chain[0].where_clause.is_some());
                assert!(path.chain[0].types.is_empty());
            } else {
                panic!("Expected Path pattern");
            }
        }
    }

    #[test]
    fn test_parse_multiple_edge_types() {
        let stmt = parse_ok("MATCH (a)-[:KNOWS|LIKES|FOLLOWS]->(b) RETURN a, b");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Match(MatchClause { patterns, .. }) = &clauses[0]
        {
            if let Pattern::Path(path) = &patterns[0] {
                assert_eq!(path.chain[0].types.len(), 3);
            } else {
                panic!("Expected Path pattern");
            }
        }
    }

    #[test]
    fn test_parse_chain_pattern() {
        let stmt = parse_ok("MATCH (a)-[:KNOWS]->(b)-[:WORKS_AT]->(c) RETURN a, c");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Match(MatchClause { patterns, .. }) = &clauses[0]
        {
            if let Pattern::Path(path) = &patterns[0] {
                assert_eq!(path.chain.len(), 2);
            } else {
                panic!("Expected Path pattern");
            }
        }
    }

    // ==================== WHERE Clause Tests ====================

    #[test]
    fn test_parse_where_simple() {
        let stmt = parse_ok("MATCH (n) WHERE n.age > 30 RETURN n");
        if let Statement::Query(Query { clauses, .. }) = stmt {
            assert!(matches!(&clauses[1], Clause::Where(_)));
        }
    }

    #[test]
    fn test_parse_where_and() {
        let stmt = parse_ok("MATCH (n) WHERE n.age > 30 AND n.name = 'Alix' RETURN n");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Where(WhereClause { predicate, .. }) = &clauses[1]
        {
            assert!(matches!(
                predicate,
                Expression::Binary {
                    op: BinaryOp::And,
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_parse_where_or() {
        let stmt = parse_ok("MATCH (n) WHERE n.age < 20 OR n.age > 60 RETURN n");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Where(WhereClause { predicate, .. }) = &clauses[1]
        {
            assert!(matches!(
                predicate,
                Expression::Binary {
                    op: BinaryOp::Or,
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_parse_where_not() {
        let stmt = parse_ok("MATCH (n) WHERE NOT n.active RETURN n");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Where(WhereClause { predicate, .. }) = &clauses[1]
        {
            assert!(matches!(
                predicate,
                Expression::Unary {
                    op: UnaryOp::Not,
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_parse_where_is_null() {
        let stmt = parse_ok("MATCH (n) WHERE n.email IS NULL RETURN n");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Where(WhereClause { predicate, .. }) = &clauses[1]
        {
            assert!(matches!(
                predicate,
                Expression::Unary {
                    op: UnaryOp::IsNull,
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_parse_where_is_not_null() {
        let stmt = parse_ok("MATCH (n) WHERE n.email IS NOT NULL RETURN n");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Where(WhereClause { predicate, .. }) = &clauses[1]
        {
            assert!(matches!(
                predicate,
                Expression::Unary {
                    op: UnaryOp::IsNotNull,
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_parse_where_in() {
        let stmt = parse_ok("MATCH (n) WHERE n.status IN ['active', 'pending'] RETURN n");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Where(WhereClause { predicate, .. }) = &clauses[1]
        {
            assert!(matches!(
                predicate,
                Expression::Binary {
                    op: BinaryOp::In,
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_parse_where_starts_with() {
        let stmt = parse_ok("MATCH (n) WHERE n.name STARTS WITH 'A' RETURN n");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Where(WhereClause { predicate, .. }) = &clauses[1]
        {
            assert!(matches!(
                predicate,
                Expression::Binary {
                    op: BinaryOp::StartsWith,
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_parse_where_ends_with() {
        let stmt = parse_ok("MATCH (n) WHERE n.email ENDS WITH '.com' RETURN n");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Where(WhereClause { predicate, .. }) = &clauses[1]
        {
            assert!(matches!(
                predicate,
                Expression::Binary {
                    op: BinaryOp::EndsWith,
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_parse_where_contains() {
        let stmt = parse_ok("MATCH (n) WHERE n.bio CONTAINS 'engineer' RETURN n");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Where(WhereClause { predicate, .. }) = &clauses[1]
        {
            assert!(matches!(
                predicate,
                Expression::Binary {
                    op: BinaryOp::Contains,
                    ..
                }
            ));
        }
    }

    // ==================== RETURN Clause Tests ====================

    #[test]
    fn test_parse_return_all() {
        let stmt = parse_ok("MATCH (n) RETURN *");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Return(ReturnClause { items, .. }) = &clauses[1]
        {
            assert!(matches!(items, ReturnItems::All));
        }
    }

    #[test]
    fn test_parse_return_distinct() {
        let stmt = parse_ok("MATCH (n) RETURN DISTINCT n.name");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Return(ReturnClause { distinct, .. }) = &clauses[1]
        {
            assert!(*distinct);
        }
    }

    #[test]
    fn test_parse_return_with_alias() {
        let stmt = parse_ok("MATCH (n) RETURN n.name AS name");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Return(ReturnClause {
                items: ReturnItems::Explicit(items),
                ..
            }) = &clauses[1]
        {
            assert_eq!(items[0].alias, Some("name".to_string()));
        }
    }

    // ==================== CREATE Clause Tests ====================

    #[test]
    fn test_parse_create_node() {
        let stmt = parse_ok("CREATE (n:Person {name: 'Alix'})");
        if let Statement::Query(Query { clauses, .. }) = stmt {
            assert!(matches!(&clauses[0], Clause::Create(_)));
        }
    }

    #[test]
    fn test_parse_create_relationship() {
        let stmt = parse_ok("MATCH (a:Person), (b:Person) CREATE (a)-[:KNOWS]->(b)");
        if let Statement::Query(Query { clauses, .. }) = stmt {
            assert!(matches!(&clauses[1], Clause::Create(_)));
        }
    }

    // ==================== MERGE Clause Tests ====================

    #[test]
    fn test_parse_merge() {
        let stmt = parse_ok("MERGE (n:Person {name: 'Alix'})");
        if let Statement::Query(Query { clauses, .. }) = stmt {
            assert!(matches!(&clauses[0], Clause::Merge(_)));
        }
    }

    #[test]
    fn test_parse_merge_on_create() {
        let stmt =
            parse_ok("MERGE (n:Person {name: 'Alix'}) ON CREATE SET n.created = timestamp()");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Merge(MergeClause { on_create, .. }) = &clauses[0]
        {
            assert!(on_create.is_some());
        }
    }

    #[test]
    fn test_parse_merge_on_match() {
        let stmt = parse_ok("MERGE (n:Person {name: 'Alix'}) ON MATCH SET n.seen = true");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Merge(MergeClause { on_match, .. }) = &clauses[0]
        {
            assert!(on_match.is_some());
        }
    }

    // ==================== DELETE Clause Tests ====================

    #[test]
    fn test_parse_delete() {
        let stmt = parse_ok("MATCH (n) DELETE n");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Delete(DeleteClause { detach, .. }) = &clauses[1]
        {
            assert!(!*detach);
        }
    }

    #[test]
    fn test_parse_detach_delete() {
        let stmt = parse_ok("MATCH (n) DETACH DELETE n");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Delete(DeleteClause { detach, .. }) = &clauses[1]
        {
            assert!(*detach);
        }
    }

    // ==================== SET Clause Tests ====================

    #[test]
    fn test_parse_set_property() {
        let stmt = parse_ok("MATCH (n) SET n.name = 'Gus'");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Set(SetClause { items, .. }) = &clauses[1]
        {
            assert!(matches!(&items[0], SetItem::Property { .. }));
        }
    }

    #[test]
    fn test_parse_set_labels() {
        let stmt = parse_ok("MATCH (n) SET n:Admin:Manager");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Set(SetClause { items, .. }) = &clauses[1]
            && let SetItem::Labels { labels, .. } = &items[0]
        {
            assert_eq!(labels.len(), 2);
        }
    }

    #[test]
    fn test_parse_set_all_properties() {
        let stmt = parse_ok("MATCH (n) SET n = {name: 'Alix', age: 30}");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Set(SetClause { items, .. }) = &clauses[1]
        {
            assert!(matches!(&items[0], SetItem::AllProperties { .. }));
        }
    }

    #[test]
    fn test_parse_set_merge_properties() {
        let stmt = parse_ok("MATCH (n) SET n += {updated: true}");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Set(SetClause { items, .. }) = &clauses[1]
        {
            assert!(matches!(&items[0], SetItem::MergeProperties { .. }));
        }
    }

    // ==================== REMOVE Clause Tests ====================

    #[test]
    fn test_parse_remove_property() {
        let stmt = parse_ok("MATCH (n) REMOVE n.temp");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Remove(RemoveClause { items, .. }) = &clauses[1]
        {
            assert!(matches!(&items[0], RemoveItem::Property { .. }));
        }
    }

    #[test]
    fn test_parse_remove_labels() {
        let stmt = parse_ok("MATCH (n) REMOVE n:Temp:Staging");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Remove(RemoveClause { items, .. }) = &clauses[1]
            && let RemoveItem::Labels { labels, .. } = &items[0]
        {
            assert_eq!(labels.len(), 2);
        }
    }

    // ==================== WITH Clause Tests ====================

    #[test]
    fn test_parse_with() {
        let stmt = parse_ok("MATCH (n) WITH n.name AS name RETURN name");
        if let Statement::Query(Query { clauses, .. }) = stmt {
            assert!(matches!(&clauses[1], Clause::With(_)));
        }
    }

    #[test]
    fn test_parse_with_distinct() {
        let stmt = parse_ok("MATCH (n) WITH DISTINCT n.city AS city RETURN city");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::With(WithClause { distinct, .. }) = &clauses[1]
        {
            assert!(*distinct);
        }
    }

    #[test]
    fn test_parse_with_where() {
        let stmt = parse_ok("MATCH (n) WITH n WHERE n.age > 30 RETURN n");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::With(WithClause { where_clause, .. }) = &clauses[1]
        {
            assert!(where_clause.is_some());
        }
    }

    // ==================== OPTIONAL MATCH Tests ====================

    #[test]
    fn test_parse_optional_match() {
        let stmt = parse_ok("MATCH (a) OPTIONAL MATCH (a)-[:KNOWS]->(b) RETURN a, b");
        if let Statement::Query(Query { clauses, .. }) = stmt {
            assert!(matches!(&clauses[0], Clause::Match(_)));
            assert!(matches!(&clauses[1], Clause::OptionalMatch(_)));
        }
    }

    // ==================== UNWIND Tests ====================

    #[test]
    fn test_parse_unwind() {
        let stmt = parse_ok("UNWIND [1, 2, 3] AS x RETURN x");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Unwind(UnwindClause { variable, .. }) = &clauses[0]
        {
            assert_eq!(variable, "x");
        }
    }

    // ==================== ORDER BY Tests ====================

    #[test]
    fn test_parse_order_by_asc() {
        let stmt = parse_ok("MATCH (n) RETURN n ORDER BY n.name ASC");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::OrderBy(OrderByClause { items, .. }) = &clauses[2]
        {
            assert_eq!(items[0].direction, SortDirection::Asc);
        }
    }

    #[test]
    fn test_parse_order_by_desc() {
        let stmt = parse_ok("MATCH (n) RETURN n ORDER BY n.age DESC");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::OrderBy(OrderByClause { items, .. }) = &clauses[2]
        {
            assert_eq!(items[0].direction, SortDirection::Desc);
        }
    }

    #[test]
    fn test_parse_order_by_multiple() {
        let stmt = parse_ok("MATCH (n) RETURN n ORDER BY n.name, n.age DESC");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::OrderBy(OrderByClause { items, .. }) = &clauses[2]
        {
            assert_eq!(items.len(), 2);
        }
    }

    // ==================== SKIP and LIMIT Tests ====================

    #[test]
    fn test_parse_skip() {
        let stmt = parse_ok("MATCH (n) RETURN n SKIP 10");
        if let Statement::Query(Query { clauses, .. }) = stmt {
            assert!(matches!(&clauses[2], Clause::Skip(_)));
        }
    }

    #[test]
    fn test_parse_limit() {
        let stmt = parse_ok("MATCH (n) RETURN n LIMIT 5");
        if let Statement::Query(Query { clauses, .. }) = stmt {
            assert!(matches!(&clauses[2], Clause::Limit(_)));
        }
    }

    #[test]
    fn test_parse_skip_and_limit() {
        let stmt = parse_ok("MATCH (n) RETURN n SKIP 10 LIMIT 5");
        if let Statement::Query(Query { clauses, .. }) = stmt {
            assert!(matches!(&clauses[2], Clause::Skip(_)));
            assert!(matches!(&clauses[3], Clause::Limit(_)));
        }
    }

    // ==================== Expression Tests ====================

    #[test]
    fn test_parse_literal_integer() {
        let stmt = parse_ok("RETURN 42");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Return(ReturnClause {
                items: ReturnItems::Explicit(items),
                ..
            }) = &clauses[0]
        {
            assert!(matches!(
                &items[0].expression,
                Expression::Literal(Literal::Integer(42))
            ));
        }
    }

    #[test]
    fn test_parse_literal_float() {
        let stmt = parse_ok("RETURN 2.78");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Return(ReturnClause {
                items: ReturnItems::Explicit(items),
                ..
            }) = &clauses[0]
            && let Expression::Literal(Literal::Float(val)) = &items[0].expression
        {
            assert!((val - 2.78).abs() < 0.001);
        }
    }

    #[test]
    fn test_parse_literal_string() {
        let stmt = parse_ok("RETURN 'hello'");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Return(ReturnClause {
                items: ReturnItems::Explicit(items),
                ..
            }) = &clauses[0]
            && let Expression::Literal(Literal::String(s)) = &items[0].expression
        {
            assert_eq!(s, "hello");
        }
    }

    #[test]
    fn test_parse_literal_bool() {
        let stmt = parse_ok("RETURN true, false");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Return(ReturnClause {
                items: ReturnItems::Explicit(items),
                ..
            }) = &clauses[0]
        {
            assert!(matches!(
                &items[0].expression,
                Expression::Literal(Literal::Bool(true))
            ));
            assert!(matches!(
                &items[1].expression,
                Expression::Literal(Literal::Bool(false))
            ));
        }
    }

    #[test]
    fn test_parse_literal_null() {
        let stmt = parse_ok("RETURN null");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Return(ReturnClause {
                items: ReturnItems::Explicit(items),
                ..
            }) = &clauses[0]
        {
            assert!(matches!(
                &items[0].expression,
                Expression::Literal(Literal::Null)
            ));
        }
    }

    #[test]
    fn test_parse_list_literal() {
        let stmt = parse_ok("RETURN [1, 2, 3]");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Return(ReturnClause {
                items: ReturnItems::Explicit(items),
                ..
            }) = &clauses[0]
            && let Expression::List(list) = &items[0].expression
        {
            assert_eq!(list.len(), 3);
        }
    }

    #[test]
    fn test_parse_map_literal() {
        let stmt = parse_ok("RETURN {name: 'Alix', age: 30}");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Return(ReturnClause {
                items: ReturnItems::Explicit(items),
                ..
            }) = &clauses[0]
            && let Expression::Map(map) = &items[0].expression
        {
            assert_eq!(map.len(), 2);
        }
    }

    #[test]
    fn test_parse_parameter() {
        let stmt = parse_ok("MATCH (n) WHERE n.id = $id RETURN n");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Where(WhereClause { predicate, .. }) = &clauses[1]
            && let Expression::Binary { right, .. } = predicate
            && let Expression::Parameter(name) = right.as_ref()
        {
            assert_eq!(name, "id");
        }
    }

    #[test]
    fn test_parse_function_call() {
        let stmt = parse_ok("RETURN count(n)");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Return(ReturnClause {
                items: ReturnItems::Explicit(items),
                ..
            }) = &clauses[0]
            && let Expression::FunctionCall { name, .. } = &items[0].expression
        {
            assert_eq!(name, "count");
        }
    }

    #[test]
    fn test_parse_function_call_distinct() {
        let stmt = parse_ok("RETURN count(DISTINCT n)");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Return(ReturnClause {
                items: ReturnItems::Explicit(items),
                ..
            }) = &clauses[0]
            && let Expression::FunctionCall { distinct, .. } = &items[0].expression
        {
            assert!(*distinct);
        }
    }

    #[test]
    fn test_parse_arithmetic() {
        let stmt = parse_ok("RETURN 1 + 2 * 3");
        assert!(matches!(stmt, Statement::Query(_)));
    }

    #[test]
    fn test_parse_property_access() {
        let stmt = parse_ok("MATCH (n) RETURN n.name");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Return(ReturnClause {
                items: ReturnItems::Explicit(items),
                ..
            }) = &clauses[1]
        {
            assert!(matches!(
                &items[0].expression,
                Expression::PropertyAccess { .. }
            ));
        }
    }

    #[test]
    fn test_parse_index_access() {
        let stmt = parse_ok("RETURN list[0]");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Return(ReturnClause {
                items: ReturnItems::Explicit(items),
                ..
            }) = &clauses[0]
        {
            assert!(matches!(
                &items[0].expression,
                Expression::IndexAccess { .. }
            ));
        }
    }

    // ==================== CASE Expression Tests ====================

    #[test]
    fn test_parse_case_simple() {
        let stmt = parse_ok("RETURN CASE n.status WHEN 'active' THEN 1 ELSE 0 END");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Return(ReturnClause {
                items: ReturnItems::Explicit(items),
                ..
            }) = &clauses[0]
            && let Expression::Case {
                input,
                whens,
                else_clause,
            } = &items[0].expression
        {
            assert!(input.is_some());
            assert_eq!(whens.len(), 1);
            assert!(else_clause.is_some());
        }
    }

    #[test]
    fn test_parse_case_searched() {
        let stmt = parse_ok(
            "RETURN CASE WHEN n.age < 18 THEN 'minor' WHEN n.age < 65 THEN 'adult' ELSE 'senior' END",
        );
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Return(ReturnClause {
                items: ReturnItems::Explicit(items),
                ..
            }) = &clauses[0]
            && let Expression::Case { input, whens, .. } = &items[0].expression
        {
            assert!(input.is_none());
            assert_eq!(whens.len(), 2);
        }
    }

    // ==================== Named Path Tests ====================

    #[test]
    fn test_parse_named_path() {
        let stmt = parse_ok("MATCH p = (a)-[:KNOWS]->(b) RETURN p");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Match(MatchClause { patterns, .. }) = &clauses[0]
            && let Pattern::NamedPath { name, .. } = &patterns[0]
        {
            assert_eq!(name, "p");
        }
    }

    #[test]
    fn test_parse_shortest_path() {
        let stmt = parse_ok("MATCH p = shortestPath((a)-[:KNOWS*]->(b)) RETURN p");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Match(MatchClause { patterns, .. }) = &clauses[0]
            && let Pattern::NamedPath { path_function, .. } = &patterns[0]
        {
            assert_eq!(*path_function, Some(PathFunction::ShortestPath));
        }
    }

    // ==================== Error Cases ====================

    #[test]
    fn test_parse_error_empty() {
        parse_err("");
    }

    #[test]
    fn test_parse_error_invalid_syntax() {
        parse_err("MATCH");
    }

    #[test]
    fn test_parse_error_unclosed_paren() {
        parse_err("MATCH (n RETURN n");
    }

    #[test]
    fn test_parse_error_missing_return() {
        // This is a valid Cypher query without RETURN (for side effects)
        // so we test something else
        parse_err("RETURN RETURN");
    }

    // ==================== Cypher Compatibility Tests ====================

    #[test]
    fn test_parse_exists_as_alias() {
        // Issue: `exists` should be usable as an alias name
        parse_ok("MATCH (n) RETURN count(n) as exists");
    }

    #[test]
    fn test_parse_multiple_match_clauses() {
        // Issue: sequential MATCH clauses should parse
        let stmt = parse_ok("MATCH (a) WHERE a.id = 'x' MATCH (b) WHERE b.id = 'y' RETURN a, b");
        if let Statement::Query(Query { clauses, .. }) = stmt {
            assert!(matches!(&clauses[0], Clause::Match(_)));
            assert!(matches!(&clauses[1], Clause::Where(_)));
            assert!(matches!(&clauses[2], Clause::Match(_)));
            assert!(matches!(&clauses[3], Clause::Where(_)));
            assert!(matches!(&clauses[4], Clause::Return(_)));
        }
    }

    #[test]
    fn test_parse_merge_with_relationship() {
        // Issue: MERGE with path patterns should parse
        let stmt = parse_ok("MATCH (a {id: 'x'}), (b {id: 'y'}) MERGE (a)-[r:KNOWS]->(b) RETURN r");
        if let Statement::Query(Query { clauses, .. }) = stmt {
            assert!(matches!(&clauses[0], Clause::Match(_)));
            assert!(matches!(&clauses[1], Clause::Merge(_)));
            assert!(matches!(&clauses[2], Clause::Return(_)));
        }
    }

    // ==================== EXISTS Subquery Tests ====================

    #[test]
    fn test_parse_exists_subquery() {
        let stmt = parse_ok("MATCH (n) WHERE EXISTS { MATCH (n)-[:KNOWS]->() } RETURN n");
        if let Statement::Query(Query { clauses, .. }) = stmt {
            assert!(matches!(&clauses[0], Clause::Match(_)));
            // WHERE clause should contain an Exists expression
            if let Clause::Where(w) = &clauses[1] {
                assert!(matches!(&w.predicate, Expression::Exists(_)));
            } else {
                panic!("expected WHERE clause");
            }
        }
    }

    #[test]
    fn test_parse_not_exists_subquery() {
        let stmt = parse_ok("MATCH (n) WHERE NOT EXISTS { MATCH (n)-[:KNOWS]->() } RETURN n");
        if let Statement::Query(Query { clauses, .. }) = stmt {
            if let Clause::Where(w) = &clauses[1] {
                assert!(matches!(
                    &w.predicate,
                    Expression::Unary {
                        op: UnaryOp::Not,
                        ..
                    }
                ));
            } else {
                panic!("expected WHERE clause");
            }
        }
    }

    #[test]
    fn test_parse_exists_with_inner_where() {
        parse_ok("MATCH (n) WHERE EXISTS { MATCH (n)-[:KNOWS]->(m) WHERE m.age > 30 } RETURN n");
    }

    #[test]
    fn test_parse_exists_function_still_works() {
        // exists(n.prop) should still parse as a function call
        let stmt = parse_ok("MATCH (n) WHERE exists(n.name) RETURN n");
        if let Statement::Query(Query { clauses, .. }) = stmt {
            if let Clause::Where(w) = &clauses[1] {
                assert!(matches!(&w.predicate, Expression::FunctionCall { .. }));
            } else {
                panic!("expected WHERE clause");
            }
        }
    }

    #[test]
    fn test_parse_exists_bare_pattern() {
        // Bare pattern without MATCH keyword
        let stmt = parse_ok("MATCH (n) WHERE EXISTS { (n)-[:KNOWS]->() } RETURN n");
        if let Statement::Query(Query { clauses, .. }) = stmt {
            if let Clause::Where(w) = &clauses[1] {
                assert!(matches!(&w.predicate, Expression::Exists(_)));
            } else {
                panic!("expected WHERE clause");
            }
        }
    }

    #[test]
    fn test_parse_exists_bare_pattern_with_where() {
        // Bare pattern with WHERE inside EXISTS
        parse_ok(
            "MATCH (a), (b) WHERE NOT EXISTS { (a)-[r]->(b) WHERE type(r) = 'KNOWS' } RETURN a",
        );
    }

    #[test]
    fn test_parse_count_bare_pattern() {
        // COUNT { bare pattern } should also work
        parse_ok("MATCH (n) RETURN COUNT { (n)-[:KNOWS]->() } AS friend_count");
    }

    #[test]
    fn test_parse_hex_integer_literal() {
        let stmt = parse_ok("RETURN 0xFF");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Return(ret) = &clauses[0]
            && let ReturnItems::Explicit(items) = &ret.items
            && let Expression::Literal(Literal::Integer(val)) = &items[0].expression
        {
            assert_eq!(*val, 255, "0xFF should parse to 255");
        } else {
            panic!("Expected integer literal in RETURN");
        }
    }

    #[test]
    fn test_parse_octal_integer_literal() {
        let stmt = parse_ok("RETURN 0o77");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Return(ret) = &clauses[0]
            && let ReturnItems::Explicit(items) = &ret.items
            && let Expression::Literal(Literal::Integer(val)) = &items[0].expression
        {
            assert_eq!(*val, 63, "0o77 should parse to 63");
        } else {
            panic!("Expected integer literal in RETURN");
        }
    }

    #[test]
    fn test_parse_scientific_float_literal() {
        let stmt = parse_ok("RETURN 1.5e10");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Return(ret) = &clauses[0]
            && let ReturnItems::Explicit(items) = &ret.items
            && let Expression::Literal(Literal::Float(val)) = &items[0].expression
        {
            assert!((*val - 1.5e10).abs() < 1.0, "1.5e10 should parse correctly");
        } else {
            panic!("Expected float literal in RETURN");
        }
    }

    #[test]
    fn test_trailing_semicolons() {
        // Single trailing semicolon
        parse_ok("RETURN 1;");
        // Multiple trailing semicolons
        parse_ok("RETURN 1;;;");
        // Semicolon after full query
        parse_ok("MATCH (n) RETURN n;");
    }

    fn parse_error_message(query: &str) -> String {
        Parser::new(query).parse().expect_err(query).to_string()
    }

    #[test]
    fn test_union_and_schema_statements_check_end_of_input() {
        // A trailing ';' after a UNION used to fail; trailing input after a
        // schema statement used to be ignored (#380).
        parse_ok("MATCH (n) RETURN n.a AS a UNION MATCH (m) RETURN m.a AS a;");
        parse_ok("CREATE INDEX idx_name FOR (n:Person) ON (n.name);");
        let err = parse_error_message("CREATE INDEX idx_name FOR (n:Person) ON (n.name) garbage");
        assert!(err.contains("Expected end of query"), "{err}");
        let err = parse_error_message("SHOW INDEXES garbage");
        assert!(err.contains("Expected end of query"), "{err}");
    }

    #[test]
    fn test_statements_separated_by_semicolon_are_rejected() {
        let err = parse_error_message("RETURN 1; RETURN 2");
        assert!(
            err.contains("multiple statements separated by ';' are not supported in one call"),
            "{err}"
        );
    }

    // === Schema DDL Tests ===

    #[test]
    fn test_parse_create_index() {
        let stmt = parse_ok("CREATE INDEX idx_name FOR (n:Person) ON (n.name)");
        assert!(matches!(stmt, Statement::Schema(_)));
    }

    #[test]
    fn test_parse_create_index_if_not_exists() {
        let stmt = parse_ok("CREATE INDEX idx_name IF NOT EXISTS FOR (n:Person) ON (n.name)");
        assert!(matches!(stmt, Statement::Schema(_)));
    }

    #[test]
    fn test_parse_drop_index() {
        let stmt = parse_ok("DROP INDEX idx_name");
        assert!(matches!(stmt, Statement::Schema(_)));
    }

    #[test]
    fn test_parse_drop_index_if_exists() {
        let stmt = parse_ok("DROP INDEX idx_name IF EXISTS");
        assert!(matches!(stmt, Statement::Schema(_)));
    }

    #[test]
    fn test_parse_create_constraint_unique() {
        let stmt = parse_ok("CREATE CONSTRAINT uniq FOR (n:Person) REQUIRE n.email IS UNIQUE");
        assert!(matches!(stmt, Statement::Schema(_)));
    }

    #[test]
    fn test_parse_create_constraint_not_null() {
        let stmt = parse_ok("CREATE CONSTRAINT nn FOR (n:Person) REQUIRE n.name IS NOT NULL");
        assert!(matches!(stmt, Statement::Schema(_)));
    }

    #[test]
    fn test_parse_drop_constraint() {
        let stmt = parse_ok("DROP CONSTRAINT uniq");
        assert!(matches!(stmt, Statement::Schema(_)));
    }

    #[test]
    fn test_parse_show_indexes() {
        let stmt = parse_ok("SHOW INDEXES");
        assert!(matches!(stmt, Statement::ShowIndexes));
    }

    #[test]
    fn test_parse_show_constraints() {
        let stmt = parse_ok("SHOW CONSTRAINTS");
        assert!(matches!(stmt, Statement::ShowConstraints));
    }

    #[test]
    fn test_parse_load_csv_with_headers() {
        let stmt = parse_ok("LOAD CSV WITH HEADERS FROM 'data.csv' AS row RETURN row.name");
        if let Statement::Query(q) = stmt {
            assert!(
                matches!(&q.clauses[0], Clause::LoadCsv(lc) if lc.with_headers && lc.path == "data.csv" && lc.variable == "row")
            );
        } else {
            panic!("Expected Query");
        }
    }

    #[test]
    fn test_parse_load_csv_without_headers() {
        let stmt = parse_ok("LOAD CSV FROM 'data.csv' AS line RETURN line[0]");
        if let Statement::Query(q) = stmt {
            assert!(
                matches!(&q.clauses[0], Clause::LoadCsv(lc) if !lc.with_headers && lc.variable == "line")
            );
        } else {
            panic!("Expected Query");
        }
    }

    #[test]
    fn test_parse_load_csv_fieldterminator() {
        let stmt = parse_ok(
            "LOAD CSV WITH HEADERS FROM 'data.tsv' AS row FIELDTERMINATOR '\\t' RETURN row.name",
        );
        if let Statement::Query(q) = stmt {
            assert!(
                matches!(&q.clauses[0], Clause::LoadCsv(lc) if lc.field_terminator == Some('\t'))
            );
        } else {
            panic!("Expected Query");
        }
    }

    #[test]
    fn test_parse_load_csv_with_create() {
        let stmt = parse_ok(
            "LOAD CSV WITH HEADERS FROM 'people.csv' AS row CREATE (p:Person {name: row.name})",
        );
        if let Statement::Query(q) = stmt {
            assert!(matches!(&q.clauses[0], Clause::LoadCsv(_)));
            assert!(matches!(&q.clauses[1], Clause::Create(_)));
        } else {
            panic!("Expected Query");
        }
    }

    #[test]
    fn test_parse_load_csv_file_uri() {
        let stmt =
            parse_ok("LOAD CSV WITH HEADERS FROM 'file:///data/people.csv' AS row RETURN row.name");
        if let Statement::Query(q) = stmt {
            assert!(
                matches!(&q.clauses[0], Clause::LoadCsv(lc) if lc.path == "file:///data/people.csv")
            );
        } else {
            panic!("Expected Query");
        }
    }

    // ================================================================
    // SHOW CURRENT GRAPH TYPE
    // ================================================================

    #[test]
    fn test_parse_show_current_graph_type() {
        let stmt = parse_ok("SHOW CURRENT GRAPH TYPE");
        assert!(matches!(stmt, Statement::ShowCurrentGraphType));
    }

    #[test]
    fn test_parse_show_current_graph_type_lowercase() {
        let stmt = parse_ok("show current graph type");
        assert!(matches!(stmt, Statement::ShowCurrentGraphType));
    }

    #[test]
    fn test_parse_show_current_graph_type_mixed_case() {
        let stmt = parse_ok("Show Current Graph Type");
        assert!(matches!(stmt, Statement::ShowCurrentGraphType));
    }

    // ================================================================
    // ALTER CURRENT GRAPH TYPE SET
    // ================================================================

    #[test]
    fn test_parse_alter_current_graph_type_set_node() {
        use crate::query::schema::{InlineElementType, SchemaStatement};

        let stmt = parse_ok("ALTER CURRENT GRAPH TYPE SET { (:Person { name STRING NOT NULL }) }");
        if let Statement::Schema(SchemaStatement::CreateGraphType(create)) = stmt {
            assert_eq!(create.name, "__current__");
            assert!(create.or_replace);
            assert_eq!(create.inline_types.len(), 1);
            assert!(matches!(
                &create.inline_types[0],
                InlineElementType::Node { name, properties, .. }
                    if name == "Person" && properties.len() == 1
                       && properties[0].name == "name"
                       && properties[0].data_type == "STRING"
                       && !properties[0].nullable
            ));
        } else {
            panic!("Expected Schema(CreateGraphType), got {stmt:?}");
        }
    }

    #[test]
    fn test_parse_alter_current_graph_type_set_node_neo4j_type_syntax() {
        use crate::query::schema::{InlineElementType, SchemaStatement};

        // Neo4j uses :: for type annotations
        let stmt =
            parse_ok("ALTER CURRENT GRAPH TYPE SET { (:Person { name :: STRING NOT NULL }) }");
        if let Statement::Schema(SchemaStatement::CreateGraphType(create)) = stmt {
            assert_eq!(create.inline_types.len(), 1);
            assert!(matches!(
                &create.inline_types[0],
                InlineElementType::Node { name, properties, .. }
                    if name == "Person"
                       && properties[0].data_type == "STRING"
                       && !properties[0].nullable
            ));
        } else {
            panic!("Expected Schema(CreateGraphType), got {stmt:?}");
        }
    }

    #[test]
    fn test_parse_alter_current_graph_type_set_edge() {
        use crate::query::schema::{InlineElementType, SchemaStatement};

        let stmt = parse_ok("ALTER CURRENT GRAPH TYPE SET { (:Person)-[:KNOWS]->(:Person) }");
        if let Statement::Schema(SchemaStatement::CreateGraphType(create)) = stmt {
            assert_eq!(create.name, "__current__");
            assert!(create.or_replace);
            // Should have one edge type (no separate node types extracted from patterns)
            assert_eq!(create.inline_types.len(), 1);
            assert!(matches!(
                &create.inline_types[0],
                InlineElementType::Edge {
                    name,
                    source_node_types,
                    target_node_types,
                    ..
                } if name == "KNOWS"
                     && source_node_types == &["Person"]
                     && target_node_types == &["Person"]
            ));
        } else {
            panic!("Expected Schema(CreateGraphType), got {stmt:?}");
        }
    }

    #[test]
    fn test_parse_alter_current_graph_type_set_mixed() {
        use crate::query::schema::{InlineElementType, SchemaStatement};

        let stmt = parse_ok(
            "ALTER CURRENT GRAPH TYPE SET { (:Person { name STRING }), (:Person)-[:KNOWS]->(:Person) }",
        );
        if let Statement::Schema(SchemaStatement::CreateGraphType(create)) = stmt {
            assert_eq!(create.name, "__current__");
            assert_eq!(create.inline_types.len(), 2);
            // First is a node type
            assert!(
                matches!(&create.inline_types[0], InlineElementType::Node { name, .. } if name == "Person")
            );
            // Second is an edge type
            assert!(
                matches!(&create.inline_types[1], InlineElementType::Edge { name, .. } if name == "KNOWS")
            );
        } else {
            panic!("Expected Schema(CreateGraphType), got {stmt:?}");
        }
    }

    // ================================================================
    // ALTER CURRENT GRAPH TYPE ADD
    // ================================================================

    #[test]
    fn test_parse_alter_current_graph_type_add() {
        use crate::query::schema::{GraphTypeAlteration, SchemaStatement};

        let stmt = parse_ok("ALTER CURRENT GRAPH TYPE ADD { (:Movie { title STRING }) }");
        if let Statement::Schema(SchemaStatement::AlterGraphType(alter)) = stmt {
            assert_eq!(alter.name, "__current__");
            assert_eq!(alter.alterations.len(), 1);
            assert!(matches!(
                &alter.alterations[0],
                GraphTypeAlteration::AddNodeType(name) if name == "Movie"
            ));
        } else {
            panic!("Expected Schema(AlterGraphType), got {stmt:?}");
        }
    }

    #[test]
    fn test_parse_alter_current_graph_type_add_edge() {
        use crate::query::schema::{GraphTypeAlteration, SchemaStatement};

        let stmt = parse_ok("ALTER CURRENT GRAPH TYPE ADD { (:Person)-[:ACTED_IN]->(:Movie) }");
        if let Statement::Schema(SchemaStatement::AlterGraphType(alter)) = stmt {
            assert_eq!(alter.name, "__current__");
            assert_eq!(alter.alterations.len(), 1);
            assert!(matches!(
                &alter.alterations[0],
                GraphTypeAlteration::AddEdgeType(name) if name == "ACTED_IN"
            ));
        } else {
            panic!("Expected Schema(AlterGraphType), got {stmt:?}");
        }
    }

    // ================================================================
    // ALTER CURRENT GRAPH TYPE DROP
    // ================================================================

    #[test]
    fn test_parse_alter_current_graph_type_drop() {
        use crate::query::schema::{GraphTypeAlteration, SchemaStatement};

        let stmt = parse_ok("ALTER CURRENT GRAPH TYPE DROP { (:Movie) }");
        if let Statement::Schema(SchemaStatement::AlterGraphType(alter)) = stmt {
            assert_eq!(alter.name, "__current__");
            assert_eq!(alter.alterations.len(), 1);
            assert!(matches!(
                &alter.alterations[0],
                GraphTypeAlteration::DropNodeType(name) if name == "Movie"
            ));
        } else {
            panic!("Expected Schema(AlterGraphType), got {stmt:?}");
        }
    }

    #[test]
    fn test_parse_alter_current_graph_type_drop_edge() {
        use crate::query::schema::{GraphTypeAlteration, SchemaStatement};

        let stmt = parse_ok("ALTER CURRENT GRAPH TYPE DROP { (:Person)-[:ACTED_IN]->(:Movie) }");
        if let Statement::Schema(SchemaStatement::AlterGraphType(alter)) = stmt {
            assert_eq!(alter.name, "__current__");
            assert_eq!(alter.alterations.len(), 1);
            assert!(matches!(
                &alter.alterations[0],
                GraphTypeAlteration::DropEdgeType(name) if name == "ACTED_IN"
            ));
        } else {
            panic!("Expected Schema(AlterGraphType), got {stmt:?}");
        }
    }

    // ================================================================
    // Edge cases and error handling
    // ================================================================

    #[test]
    fn test_parse_alter_current_graph_type_set_empty_body() {
        let stmt = parse_ok("ALTER CURRENT GRAPH TYPE SET { }");
        if let Statement::Schema(crate::query::schema::SchemaStatement::CreateGraphType(create)) =
            stmt
        {
            assert_eq!(create.name, "__current__");
            assert!(create.or_replace);
            assert!(create.inline_types.is_empty());
        } else {
            panic!("Expected Schema(CreateGraphType) with empty body");
        }
    }

    #[test]
    fn test_parse_alter_current_graph_type_multiple_properties() {
        use crate::query::schema::{InlineElementType, SchemaStatement};

        let stmt = parse_ok(
            "ALTER CURRENT GRAPH TYPE SET { (:Person { name STRING NOT NULL, age INTEGER }) }",
        );
        if let Statement::Schema(SchemaStatement::CreateGraphType(create)) = stmt {
            assert_eq!(create.inline_types.len(), 1);
            if let InlineElementType::Node { properties, .. } = &create.inline_types[0] {
                assert_eq!(properties.len(), 2);
                assert_eq!(properties[0].name, "name");
                assert!(!properties[0].nullable);
                assert_eq!(properties[1].name, "age");
                assert!(properties[1].nullable);
            } else {
                panic!("Expected node type");
            }
        } else {
            panic!("Expected Schema(CreateGraphType)");
        }
    }

    #[test]
    fn test_parse_alter_current_graph_type_edge_with_properties() {
        use crate::query::schema::{InlineElementType, SchemaStatement};

        let stmt = parse_ok(
            "ALTER CURRENT GRAPH TYPE SET { (:Person)-[:KNOWS { since INTEGER }]->(:Person) }",
        );
        if let Statement::Schema(SchemaStatement::CreateGraphType(create)) = stmt {
            assert_eq!(create.inline_types.len(), 1);
            if let InlineElementType::Edge {
                name, properties, ..
            } = &create.inline_types[0]
            {
                assert_eq!(name, "KNOWS");
                assert_eq!(properties.len(), 1);
                assert_eq!(properties[0].name, "since");
                assert_eq!(properties[0].data_type, "INTEGER");
            } else {
                panic!("Expected edge type");
            }
        } else {
            panic!("Expected Schema(CreateGraphType)");
        }
    }

    #[test]
    fn test_parse_alter_current_graph_type_missing_operation() {
        let mut parser = Parser::new("ALTER CURRENT GRAPH TYPE INVALID {}");
        assert!(parser.parse().is_err());
    }

    #[test]
    fn test_parse_alter_current_graph_type_case_insensitive() {
        let stmt = parse_ok("alter current graph type set { (:Label) }");
        assert!(matches!(stmt, Statement::Schema(_)));
    }

    #[test]
    fn test_parse_alter_current_graph_type_with_semicolon() {
        let stmt = parse_ok("SHOW CURRENT GRAPH TYPE;");
        assert!(matches!(stmt, Statement::ShowCurrentGraphType));
    }

    // ==================== Recursion depth limit tests ====================

    #[test]
    fn test_deeply_nested_parentheses_errors_not_stack_overflow() {
        let deep = "(".repeat(300) + "1" + &")".repeat(300);
        let query = format!("MATCH (n) RETURN {deep}");
        let mut parser = Parser::new(&query);
        let result = parser.parse();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("nesting depth"),
            "Expected nesting depth error, got: {err}"
        );
    }

    #[test]
    fn test_moderate_nesting_succeeds() {
        let depth = 50;
        let deep = "(".repeat(depth) + "1" + &")".repeat(depth);
        let query = format!("RETURN {deep}");
        let mut parser = Parser::new(&query);
        let result = parser.parse();
        assert!(result.is_ok(), "50 levels of nesting should succeed");
    }

    #[test]
    fn test_explain_recursion_depth_limit() {
        let query = "EXPLAIN ".repeat(200) + "RETURN 1";
        let mut parser = Parser::new(&query);
        let result = parser.parse();
        assert!(result.is_err(), "Deeply nested EXPLAIN should error");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("nesting depth"),
            "Expected nesting depth error, got: {err}"
        );
    }

    #[test]
    fn test_profile_recursion_depth_limit() {
        let query = "PROFILE ".repeat(200) + "RETURN 1";
        let mut parser = Parser::new(&query);
        let result = parser.parse();
        assert!(result.is_err(), "Deeply nested PROFILE should error");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("nesting depth"),
            "Expected nesting depth error, got: {err}"
        );
    }

    #[test]
    fn test_single_explain_succeeds() {
        let query = "EXPLAIN RETURN 1";
        let mut parser = Parser::new(query);
        let result = parser.parse();
        assert!(result.is_ok(), "Single EXPLAIN should succeed");
    }

    #[test]
    fn test_nested_case_expression_depth_limit() {
        let prefix = "CASE WHEN true THEN ".repeat(300);
        let suffix = " END".repeat(300);
        let query = format!("RETURN {prefix}1{suffix}");
        let mut parser = Parser::new(&query);
        let result = parser.parse();
        assert!(
            result.is_err(),
            "Deeply nested CASE should error, not stack overflow"
        );
    }

    #[test]
    fn test_nested_list_comprehension_depth_limit() {
        let deep = "[x IN ".repeat(300) + "[1]" + &" | x]".repeat(300);
        let query = format!("RETURN {deep}");
        let mut parser = Parser::new(&query);
        // Should error or parse, but never stack overflow
        let _ = parser.parse();
    }

    #[test]
    fn test_cypher_error_includes_language_prefix() {
        let mut parser = Parser::new("INVALID SYNTAX");
        let result = parser.parse();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("[Cypher]"),
            "Cypher errors should be prefixed with [Cypher], got: {err}"
        );
    }
}
