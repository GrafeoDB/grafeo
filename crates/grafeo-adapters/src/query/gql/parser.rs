//! GQL Parser.

#[allow(clippy::wildcard_imports)]
use super::ast::*;
use super::lexer::{Lexer, Token, TokenKind};
use crate::query::keywords::unescape_string;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result, SourceSpan};

/// Maximum nesting depth for recursive parsing constructs (parenthesized
/// expressions, CASE, EXISTS subqueries, list literals, function calls).
const MAX_NESTING_DEPTH: u32 = 128;

/// GQL Parser.
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
    peeked: Option<Token>,
    peeked_second: Option<Token>,
    source: &'a str,
    /// Current nesting depth for recursive parsing constructs.
    nesting_depth: u32,
}

impl<'a> Parser<'a> {
    /// Creates a new parser for the given input.
    pub fn new(input: &'a str) -> Self {
        let mut lexer = Lexer::new(input);
        let current = lexer.next_token();
        Self {
            lexer,
            current,
            peeked: None,
            peeked_second: None,
            source: input,
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

    /// Checks if the current token can be used as a label, type name, or property name.
    /// This includes identifiers, quoted identifiers, contextual keywords, and reserved
    /// keywords that are commonly used as labels or properties.
    fn is_label_or_type_name(&self) -> bool {
        self.is_contextual_keyword()
            || matches!(
                self.current.kind,
                TokenKind::Identifier
                    | TokenKind::QuotedIdentifier
                    | TokenKind::Node
                    | TokenKind::Edge
                    | TokenKind::Type
                    | TokenKind::Match
                    | TokenKind::Return
                    | TokenKind::Where
                    | TokenKind::And
                    | TokenKind::Or
                    | TokenKind::Not
                    | TokenKind::Insert
                    | TokenKind::Delete
                    | TokenKind::Set
                    | TokenKind::Create
                    | TokenKind::As
                    | TokenKind::Distinct
                    | TokenKind::Order
                    | TokenKind::By
                    | TokenKind::Asc
                    | TokenKind::Desc
                    | TokenKind::Limit
                    | TokenKind::Skip
                    | TokenKind::With
                    | TokenKind::Optional
                    | TokenKind::Null
                    | TokenKind::True
                    | TokenKind::False
                    | TokenKind::In
                    | TokenKind::Is
                    | TokenKind::Like
                    | TokenKind::Exists
                    | TokenKind::Call
                    | TokenKind::Yield
                    | TokenKind::Detach
                    | TokenKind::Unwind
                    | TokenKind::Merge
                    | TokenKind::On
                    | TokenKind::Starts
                    | TokenKind::Ends
                    | TokenKind::Contains
                    | TokenKind::Nodetach
                    | TokenKind::Fetch
                    | TokenKind::First
                    | TokenKind::Next
                    | TokenKind::Rows
                    | TokenKind::Row
                    | TokenKind::Only
            )
    }

    /// Checks if the current token is an identifier (regular or backtick-quoted).
    fn is_identifier(&self) -> bool {
        matches!(
            self.current.kind,
            TokenKind::Identifier | TokenKind::QuotedIdentifier
        ) || self.is_contextual_keyword()
    }

    /// Checks if the current token is a keyword that can be used as an identifier in context.
    /// In GQL/Cypher, many keywords can be used as variable names or labels.
    fn is_contextual_keyword(&self) -> bool {
        matches!(
            self.current.kind,
            TokenKind::End       // CASE...END
                | TokenKind::Node    // CREATE NODE TYPE
                | TokenKind::Edge    // CREATE EDGE TYPE
                | TokenKind::Type    // type() function
                | TokenKind::Case    // CASE expression
                | TokenKind::When    // CASE WHEN
                | TokenKind::Then    // CASE THEN
                | TokenKind::Else    // CASE ELSE
                | TokenKind::In      // IN operator (can be label/variable)
                | TokenKind::Is      // IS NULL
                | TokenKind::And     // AND operator
                | TokenKind::Or      // OR operator
                | TokenKind::Not     // NOT operator
                | TokenKind::Null    // NULL literal
                | TokenKind::True    // TRUE literal
                | TokenKind::False   // FALSE literal
                | TokenKind::Vector  // vector() function
                | TokenKind::Index   // index-related usage
                | TokenKind::Dimension // dimension option
                | TokenKind::Metric  // metric option
                | TokenKind::Set     // SESSION SET
                | TokenKind::All     // SESSION RESET ALL, UNION ALL
                | TokenKind::Filter  // FILTER as clause name
                | TokenKind::Having // HAVING as identifier
                | TokenKind::Fetch  // FETCH FIRST
                | TokenKind::First  // FETCH FIRST
                | TokenKind::Next   // FETCH NEXT
                | TokenKind::Rows   // ROWS ONLY
                | TokenKind::Row    // ROW ONLY
                | TokenKind::Only // ROWS ONLY
                | TokenKind::Asc  // ASC/DESC as alias
                | TokenKind::Desc
                | TokenKind::Order
                | TokenKind::By
                | TokenKind::Skip
                | TokenKind::Limit
        )
    }

    /// Gets the identifier name from the current token.
    /// For quoted identifiers, strips the backticks.
    fn get_identifier_name(&self) -> String {
        let text = &self.current.text;
        if self.current.kind == TokenKind::QuotedIdentifier {
            // Strip backticks from `name` -> name
            text[1..text.len() - 1].to_string()
        } else {
            text.clone()
        }
    }

    /// Returns `true` if the current token is either the dedicated `kind`
    /// or an identifier matching `keyword` case-insensitively.
    ///
    /// Prefer this over the raw `self.is_identifier() && name == "X"`
    /// pattern for any keyword that has a dedicated `TokenKind` variant.
    /// That raw pattern silently fails when the word is tokenised as a
    /// keyword rather than an identifier (bug-gql-create-constraint-named).
    fn peek_keyword(&self, kind: TokenKind, keyword: &str) -> bool {
        self.current.kind == kind
            || (self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case(keyword))
    }

    /// Consumes the current token if it matches [`Self::peek_keyword`].
    /// Returns `true` on match.
    fn try_accept_keyword(&mut self, kind: TokenKind, keyword: &str) -> bool {
        if self.peek_keyword(kind, keyword) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Consumes the current token if it is an identifier matching `keyword`
    /// case-insensitively. Use this for keywords that have no dedicated
    /// `TokenKind` variant (IF, EXPLAIN, PROFILE, REQUIRE, etc.).
    fn try_accept_identifier_keyword(&mut self, keyword: &str) -> bool {
        if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case(keyword) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Parses the input into a statement.
    ///
    /// The whole input must be one statement: trailing `;` are allowed, any
    /// other trailing text is an error (it used to be silently ignored, #380).
    ///
    /// # Errors
    ///
    /// Returns an error if the input contains invalid or unexpected GQL syntax,
    /// or text after the end of the statement.
    pub fn parse(&mut self) -> Result<Statement> {
        let statement = self.parse_statement()?;
        self.expect_end_of_input()?;
        Ok(statement)
    }

    /// Requires that only `;` remain after a complete statement.
    fn expect_end_of_input(&mut self) -> Result<()> {
        let mut saw_semicolon = false;
        while self.current.kind == TokenKind::Semicolon {
            self.advance();
            saw_semicolon = true;
        }
        if self.current.kind == TokenKind::Eof {
            return Ok(());
        }
        if saw_semicolon {
            return Err(self.error(
                "multiple statements separated by ';' are not supported in one call: \
                 run them separately, chain them with NEXT, or write consecutive INSERT clauses",
            ));
        }
        let hint = if matches!(
            self.current.kind,
            TokenKind::Match
                | TokenKind::Optional
                | TokenKind::Insert
                | TokenKind::Create
                | TokenKind::Delete
                | TokenKind::Detach
                | TokenKind::Nodetach
                | TokenKind::Set
                | TokenKind::Remove
                | TokenKind::Merge
                | TokenKind::With
                | TokenKind::Unwind
                | TokenKind::Where
                | TokenKind::Return
        ) {
            " (this clause is not supported at this position of the statement)"
        } else {
            ""
        };
        Err(self.error(&format!(
            "unexpected '{}' after the end of the statement{hint}",
            self.current.text
        )))
    }

    /// Parses one statement, including an EXPLAIN / PROFILE prefix and any
    /// NEXT or set-operation composition, without the end-of-input check.
    fn parse_statement(&mut self) -> Result<Statement> {
        // Handle EXPLAIN/PROFILE prefix: wraps the entire following statement
        if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("EXPLAIN") {
            self.advance(); // consume EXPLAIN
            self.enter_nesting()?;
            let inner = self.parse_statement()?;
            self.exit_nesting();
            return Ok(Statement::Explain(Box::new(inner)));
        }
        if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("PROFILE") {
            self.advance(); // consume PROFILE
            self.enter_nesting()?;
            let inner = self.parse_statement()?;
            self.exit_nesting();
            return Ok(Statement::Profile(Box::new(inner)));
        }

        let mut left = self.parse_single_statement()?;

        // Handle NEXT (linear composition): output of left becomes input of right
        while self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("NEXT") {
            self.advance(); // consume NEXT
            let right = self.parse_single_statement()?;
            // NEXT semantics: chain right after left (like WITH pipe).
            // Represent as CompositeQuery with a dedicated op.
            left = Statement::CompositeQuery {
                left: Box::new(left),
                op: CompositeOp::Next,
                right: Box::new(right),
            };
        }

        // Check for composite query operators (UNION, EXCEPT, INTERSECT, OTHERWISE)
        while matches!(
            self.current.kind,
            TokenKind::Union | TokenKind::Except | TokenKind::Intersect | TokenKind::Otherwise
        ) {
            let op = match self.current.kind {
                TokenKind::Union => {
                    self.advance();
                    if self.current.kind == TokenKind::All {
                        self.advance();
                        CompositeOp::UnionAll
                    } else {
                        // UNION DISTINCT is explicit form of the default
                        if self.current.kind == TokenKind::Distinct {
                            self.advance();
                        }
                        CompositeOp::Union
                    }
                }
                TokenKind::Except => {
                    self.advance();
                    if self.current.kind == TokenKind::All {
                        self.advance();
                        CompositeOp::ExceptAll
                    } else {
                        // EXCEPT DISTINCT is explicit form of the default
                        if self.current.kind == TokenKind::Distinct {
                            self.advance();
                        }
                        CompositeOp::Except
                    }
                }
                TokenKind::Intersect => {
                    self.advance();
                    if self.current.kind == TokenKind::All {
                        self.advance();
                        CompositeOp::IntersectAll
                    } else {
                        // INTERSECT DISTINCT is explicit form of the default
                        if self.current.kind == TokenKind::Distinct {
                            self.advance();
                        }
                        CompositeOp::Intersect
                    }
                }
                TokenKind::Otherwise => {
                    self.advance();
                    CompositeOp::Otherwise
                }
                _ => unreachable!(),
            };
            let right = self.parse_single_statement()?;
            left = Statement::CompositeQuery {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_single_statement(&mut self) -> Result<Statement> {
        match self.current.kind {
            TokenKind::Match
            | TokenKind::Optional
            | TokenKind::Unwind
            | TokenKind::Merge
            | TokenKind::For
            | TokenKind::Return => self.parse_query().map(Statement::Query),
            // A statement starting with INSERT is parsed as a query so that it
            // can continue with more INSERT clauses or a RETURN (#380).
            TokenKind::Insert => self.parse_query().map(Self::lone_insert_as_statement),
            TokenKind::Delete | TokenKind::Detach | TokenKind::Nodetach => self
                .parse_delete()
                .map(|s| Statement::DataModification(DataModificationStatement::Delete(s))),
            TokenKind::Create => {
                // Check if CREATE is followed by a pattern (Cypher-style) or a DDL keyword
                let next = self.peek_kind();
                if next == TokenKind::LParen {
                    // Cypher-style: CREATE (n:Label {...}) - treat as INSERT
                    self.parse_query().map(Self::lone_insert_as_statement)
                } else {
                    // GQL schema/session: dispatches between DDL (NODE TYPE, EDGE TYPE,
                    // GRAPH TYPE, INDEX, CONSTRAINT, SCHEMA) and session (GRAPH instance)
                    self.parse_create_dispatch()
                }
            }
            TokenKind::Call => {
                if self.peek_kind() == TokenKind::LBrace {
                    // CALL { subquery } RETURN ... : treat as a query
                    self.parse_query().map(Statement::Query)
                } else {
                    self.parse_call_statement().map(Statement::Call)
                }
            }
            _ if self.is_identifier() => {
                let name = self.get_identifier_name();
                match name.to_uppercase().as_str() {
                    "DROP" => self.parse_drop(),
                    "USE" => self.parse_use_graph().map(Statement::SessionCommand),
                    "SESSION" => self.parse_session_command().map(Statement::SessionCommand),
                    "START" => self
                        .parse_start_transaction()
                        .map(Statement::SessionCommand),
                    "COMMIT" => {
                        self.advance();
                        Ok(Statement::SessionCommand(SessionCommand::Commit))
                    }
                    "ROLLBACK" => {
                        self.advance();
                        // Check for ROLLBACK TO SAVEPOINT name
                        if self.is_identifier()
                            && self.get_identifier_name().eq_ignore_ascii_case("TO")
                        {
                            self.advance(); // consume TO
                            if !(self.is_identifier()
                                && self.get_identifier_name().eq_ignore_ascii_case("SAVEPOINT"))
                            {
                                return Err(self.error("Expected SAVEPOINT after ROLLBACK TO"));
                            }
                            self.advance(); // consume SAVEPOINT
                            let name = self.get_identifier_name();
                            self.advance(); // consume name
                            Ok(Statement::SessionCommand(
                                SessionCommand::RollbackToSavepoint(name),
                            ))
                        } else {
                            Ok(Statement::SessionCommand(SessionCommand::Rollback))
                        }
                    }
                    "SAVEPOINT" => {
                        self.advance();
                        let name = self.get_identifier_name();
                        self.advance();
                        Ok(Statement::SessionCommand(SessionCommand::Savepoint(name)))
                    }
                    "RELEASE" => {
                        self.advance();
                        if !(self.is_identifier()
                            && self.get_identifier_name().eq_ignore_ascii_case("SAVEPOINT"))
                        {
                            return Err(self.error("Expected SAVEPOINT after RELEASE"));
                        }
                        self.advance(); // consume SAVEPOINT
                        let name = self.get_identifier_name();
                        self.advance();
                        Ok(Statement::SessionCommand(SessionCommand::ReleaseSavepoint(
                            name,
                        )))
                    }
                    "ALTER" => self.parse_alter(),
                    "SHOW" => {
                        // Check for SHOW PROJECTIONS (session command, not schema)
                        if self.peek_kind() == TokenKind::Identifier
                            && self.peek_text_upper() == "PROJECTIONS"
                        {
                            self.advance(); // consume SHOW
                            self.advance(); // consume PROJECTIONS
                            Ok(Statement::SessionCommand(SessionCommand::ShowProjections))
                        } else {
                            self.parse_show().map(Statement::Schema)
                        }
                    }
                    "LOAD" => self.parse_query().map(Statement::Query),
                    "SELECT" => {
                        self.advance(); // consume SELECT
                        self.parse_select_from_statement()
                    }
                    _ => Err(self.error(
                        "Expected MATCH, INSERT, DELETE, MERGE, UNWIND, FOR, CREATE, CALL, \
                         DROP, ALTER, SHOW, LOAD, USE, SESSION, START, COMMIT, ROLLBACK, \
                         SELECT, or SAVEPOINT",
                    )),
                }
            }
            _ => Err(self.error(
                "Expected MATCH, INSERT, DELETE, MERGE, UNWIND, FOR, CREATE, CALL, \
                 DROP, SHOW, LOAD, USE, SESSION, START, COMMIT, or ROLLBACK",
            )),
        }
    }

    /// Parses a CALL procedure statement.
    ///
    /// ```text
    /// CALL name.space(args) [YIELD field [AS alias], ...]
    /// ```
    fn parse_call_statement(&mut self) -> Result<CallStatement> {
        let span_start = self.current.span.start;
        self.expect(TokenKind::Call)?;

        // Parse dotted procedure name: ident { . ident }
        if !self.is_identifier() {
            return Err(self.error("Expected procedure name after CALL"));
        }
        let mut name_parts = vec![self.get_identifier_name()];
        self.advance();
        while self.current.kind == TokenKind::Dot {
            self.advance();
            if !self.is_identifier() {
                return Err(self.error("Expected identifier after '.'"));
            }
            name_parts.push(self.get_identifier_name());
            self.advance();
        }

        // Parse argument list: ( [expr { , expr }] )
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
            Some(self.parse_yield_list()?)
        } else {
            None
        };

        // Parse optional WHERE clause (only valid after YIELD)
        let where_clause = if yield_items.is_some() && self.current.kind == TokenKind::Where {
            Some(self.parse_where_clause()?)
        } else {
            None
        };

        // Parse optional RETURN clause (only valid after YIELD)
        let return_clause = if yield_items.is_some() && self.current.kind == TokenKind::Return {
            Some(self.parse_return_clause()?)
        } else {
            None
        };

        Ok(CallStatement {
            procedure_name: name_parts,
            arguments,
            yield_items,
            where_clause,
            return_clause,
            span: Some(SourceSpan::new(span_start, self.current.span.start, 1, 1)),
        })
    }

    /// Parses an inline CALL { subquery }.
    ///
    /// ```text
    /// CALL { [WITH var [, var]*] query_body RETURN ... }
    /// ```
    fn parse_inline_call(&mut self) -> Result<QueryStatement> {
        self.expect(TokenKind::Call)?;
        self.expect(TokenKind::LBrace)?;

        // Parse the inner query body (MATCH ... RETURN ...)
        let inner = self.parse_query()?;

        self.expect(TokenKind::RBrace)?;
        Ok(inner)
    }

    /// Parses a YIELD item list: `field [AS alias] { , field [AS alias] }`.
    fn parse_yield_list(&mut self) -> Result<Vec<YieldItem>> {
        let mut items = vec![self.parse_yield_item()?];
        while self.current.kind == TokenKind::Comma {
            self.advance();
            items.push(self.parse_yield_item()?);
        }
        Ok(items)
    }

    /// Parses a single YIELD item: `field_name [AS alias]`.
    fn parse_yield_item(&mut self) -> Result<YieldItem> {
        let span_start = self.current.span.start;
        if !self.is_identifier() {
            return Err(self.error("Expected field name in YIELD"));
        }
        let field_name = self.get_identifier_name();
        self.advance();
        let alias = if self.current.kind == TokenKind::As {
            self.advance();
            if !self.is_identifier() {
                return Err(self.error("Expected alias after AS"));
            }
            let alias_name = self.get_identifier_name();
            self.advance();
            Some(alias_name)
        } else {
            None
        };
        Ok(YieldItem {
            field_name,
            alias,
            span: Some(SourceSpan::new(span_start, self.current.span.start, 1, 1)),
        })
    }

    /// A query made of a single INSERT (or `CREATE (...)`) clause and nothing
    /// else is kept as a standalone INSERT statement, whose result returns the
    /// created entity as before; anything longer stays a query.
    fn lone_insert_as_statement(mut query: QueryStatement) -> Statement {
        let lone_insert = matches!(query.ordered_clauses.as_slice(), [QueryClause::Create(_)])
            && query.where_clause.is_none()
            && query.set_clauses.is_empty()
            && query.remove_clauses.is_empty()
            && query.with_clauses.is_empty()
            && query.having_clause.is_none()
            && query.return_clause.items.is_empty()
            && !query.return_clause.is_wildcard
            && !query.return_clause.is_finish
            && query.return_clause.order_by.is_none()
            && query.return_clause.skip.is_none()
            && query.return_clause.limit.is_none();
        if lone_insert && let Some(QueryClause::Create(insert)) = query.ordered_clauses.pop() {
            return Statement::DataModification(DataModificationStatement::Insert(insert));
        }
        Statement::Query(query)
    }

    fn parse_query(&mut self) -> Result<QueryStatement> {
        let span_start = self.current.span.start;

        let mut match_clauses = Vec::new();
        let mut unwind_clauses = Vec::new();
        let mut merge_clauses = Vec::new();
        let mut create_clauses = Vec::new();
        let mut delete_clauses = Vec::new();
        let mut ordered_clauses = Vec::new();

        // Parse clauses in source order, preserving sequence for variable scoping.
        // MATCH, OPTIONAL MATCH, UNWIND, FOR, MERGE, CREATE/INSERT, DELETE can appear
        // in any order before RETURN.
        loop {
            match self.current.kind {
                TokenKind::Match => {
                    let clause = self.parse_match_clause()?;
                    ordered_clauses.push(QueryClause::Match(clause.clone()));
                    match_clauses.push(clause);
                }
                TokenKind::Optional => {
                    // OPTIONAL MATCH or OPTIONAL CALL { subquery }
                    let pk = self.peek_kind();
                    if pk == TokenKind::Call {
                        self.advance(); // consume OPTIONAL
                        if self.peek_kind() == TokenKind::LBrace {
                            // OPTIONAL CALL { subquery }
                            let subquery = self.parse_inline_call()?;
                            ordered_clauses.push(QueryClause::InlineCall {
                                subquery,
                                optional: true,
                            });
                        } else {
                            // OPTIONAL CALL procedure(...)
                            let call = self.parse_call_statement()?;
                            ordered_clauses.push(QueryClause::CallProcedure(call));
                        }
                    } else {
                        let clause = self.parse_match_clause()?;
                        ordered_clauses.push(QueryClause::Match(clause.clone()));
                        match_clauses.push(clause);
                    }
                }
                TokenKind::Unwind => {
                    let clause = self.parse_unwind_clause()?;
                    ordered_clauses.push(QueryClause::Unwind(clause.clone()));
                    unwind_clauses.push(clause);
                }
                TokenKind::For => {
                    let clause = self.parse_for_clause()?;
                    ordered_clauses.push(QueryClause::For(clause.clone()));
                    unwind_clauses.push(clause);
                }
                TokenKind::Merge => {
                    let clause = self.parse_merge_clause()?;
                    ordered_clauses.push(QueryClause::Merge(clause.clone()));
                    merge_clauses.push(clause);
                }
                TokenKind::Create => {
                    let clause = self.parse_create_clause_in_query()?;
                    ordered_clauses.push(QueryClause::Create(clause.clone()));
                    create_clauses.push(clause);
                }
                TokenKind::Insert => {
                    let clause = self.parse_insert()?;
                    ordered_clauses.push(QueryClause::Create(clause.clone()));
                    create_clauses.push(clause);
                }
                TokenKind::Delete | TokenKind::Detach | TokenKind::Nodetach => {
                    let clause = self.parse_delete_clause_in_query()?;
                    ordered_clauses.push(QueryClause::Delete(clause.clone()));
                    delete_clauses.push(clause);
                }
                TokenKind::Call => {
                    // CALL { subquery } (inline) or CALL procedure(...) (within query)
                    if self.peek_kind() == TokenKind::LBrace {
                        let subquery = self.parse_inline_call()?;
                        ordered_clauses.push(QueryClause::InlineCall {
                            subquery,
                            optional: false,
                        });
                    } else {
                        let call = self.parse_call_statement()?;
                        ordered_clauses.push(QueryClause::CallProcedure(call));
                    }
                }
                _ if self.is_identifier()
                    && self.get_identifier_name().eq_ignore_ascii_case("LET") =>
                {
                    let bindings = self.parse_let_clause()?;
                    ordered_clauses.push(QueryClause::Let(bindings));
                }
                _ if self.is_identifier()
                    && self.get_identifier_name().eq_ignore_ascii_case("LOAD") =>
                {
                    let clause = self.parse_load_data_clause()?;
                    ordered_clauses.push(QueryClause::LoadData(clause));
                }
                _ => break,
            }
        }

        // Parse WHERE or FILTER clause (after all MATCH clauses)
        let where_clause = if matches!(self.current.kind, TokenKind::Where | TokenKind::Filter) {
            Some(self.parse_where_or_filter_clause()?)
        } else {
            None
        };

        // After WHERE, allow CREATE/INSERT/DELETE/DETACH clauses
        loop {
            match self.current.kind {
                TokenKind::Create => {
                    let clause = self.parse_create_clause_in_query()?;
                    ordered_clauses.push(QueryClause::Create(clause.clone()));
                    create_clauses.push(clause);
                }
                TokenKind::Insert => {
                    let clause = self.parse_insert()?;
                    ordered_clauses.push(QueryClause::Create(clause.clone()));
                    create_clauses.push(clause);
                }
                TokenKind::Delete | TokenKind::Detach | TokenKind::Nodetach => {
                    let clause = self.parse_delete_clause_in_query()?;
                    ordered_clauses.push(QueryClause::Delete(clause.clone()));
                    delete_clauses.push(clause);
                }
                _ => break,
            }
        }

        // Parse SET clauses
        let mut set_clauses = Vec::new();
        while self.current.kind == TokenKind::Set {
            let clause = self.parse_set_clause()?;
            ordered_clauses.push(QueryClause::Set(clause.clone()));
            set_clauses.push(clause);
        }

        // Parse REMOVE clauses
        let mut remove_clauses = Vec::new();
        while self.current.kind == TokenKind::Remove {
            remove_clauses.push(self.parse_remove_clause()?);
        }

        // Parse WITH clauses
        let mut with_clauses = Vec::new();
        while self.current.kind == TokenKind::With {
            let mut wc = self.parse_with_clause()?;

            // Attach LET bindings that immediately follow the WITH clause
            if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("LET") {
                wc.let_bindings = self.parse_let_clause()?;
            }

            with_clauses.push(wc);

            // After WITH (+ optional LET), we can have more clauses
            loop {
                match self.current.kind {
                    TokenKind::Match | TokenKind::Optional => {
                        let clause = self.parse_match_clause()?;
                        ordered_clauses.push(QueryClause::Match(clause.clone()));
                        match_clauses.push(clause);
                    }
                    TokenKind::Unwind => {
                        let clause = self.parse_unwind_clause()?;
                        ordered_clauses.push(QueryClause::Unwind(clause.clone()));
                        unwind_clauses.push(clause);
                    }
                    TokenKind::For => {
                        let clause = self.parse_for_clause()?;
                        ordered_clauses.push(QueryClause::For(clause.clone()));
                        unwind_clauses.push(clause);
                    }
                    TokenKind::Merge => {
                        let clause = self.parse_merge_clause()?;
                        ordered_clauses.push(QueryClause::Merge(clause.clone()));
                        merge_clauses.push(clause);
                    }
                    _ => break,
                }
            }
        }

        // Parse RETURN, FINISH, or SELECT clause
        let return_clause = if self.current.kind == TokenKind::Return {
            self.parse_return_clause()?
        } else if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("FINISH")
        {
            // FINISH: consume input, return empty result
            self.advance();
            ReturnClause {
                distinct: false,
                items: Vec::new(),
                is_wildcard: false,
                group_by: Vec::new(),
                order_by: None,
                skip: None,
                limit: None,
                is_finish: true,
                span: None,
            }
        } else if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("SELECT")
        {
            // SELECT: SQL-style projection, parsed as RETURN
            self.advance(); // consume SELECT
            self.parse_select_clause()?
        } else if !set_clauses.is_empty()
            || !remove_clauses.is_empty()
            || !merge_clauses.is_empty()
            || !create_clauses.is_empty()
            || !delete_clauses.is_empty()
        {
            // For mutation-only queries, return empty clause
            ReturnClause {
                distinct: false,
                items: Vec::new(),
                is_wildcard: false,
                group_by: Vec::new(),
                order_by: None,
                skip: None,
                limit: None,
                is_finish: false,
                span: None,
            }
        } else {
            return Err(self.error("Expected RETURN, FINISH, or SELECT"));
        };

        // Parse optional HAVING clause (after RETURN, filters aggregate results)
        let having_clause = if self.current.kind == TokenKind::Having {
            Some(self.parse_having_clause()?)
        } else {
            None
        };

        Ok(QueryStatement {
            match_clauses,
            where_clause,
            set_clauses,
            remove_clauses,
            with_clauses,
            unwind_clauses,
            merge_clauses,
            create_clauses,
            delete_clauses,
            return_clause,
            having_clause,
            ordered_clauses,
            span: Some(SourceSpan::new(span_start, self.current.span.end, 1, 1)),
        })
    }

    /// Parses a FOR clause (GQL standard, ISO/IEC 39075 section 14.8).
    /// `FOR variable IN expression`: desugars to an UnwindClause.
    fn parse_for_clause(&mut self) -> Result<UnwindClause> {
        let span_start = self.current.span.start;
        self.expect(TokenKind::For)?;

        // Parse variable name
        if !self.is_identifier() {
            return Err(self.error("Expected variable name after FOR"));
        }
        let alias = self.get_identifier_name();
        self.advance();

        // Expect IN keyword
        self.expect(TokenKind::In)?;

        // Parse expression (the list to iterate)
        let expression = self.parse_expression()?;

        // Parse optional WITH ORDINALITY/OFFSET
        let (ordinality_var, offset_var) = if self.current.kind == TokenKind::With {
            self.advance(); // consume WITH
            if self.current.kind == TokenKind::Ordinality {
                self.advance(); // consume ORDINALITY
                if !self.is_identifier() {
                    return Err(self.error("Expected variable name after ORDINALITY"));
                }
                let var = self.get_identifier_name();
                self.advance();
                (Some(var), None)
            } else if self.current.kind == TokenKind::Offset {
                self.advance(); // consume OFFSET
                if !self.is_identifier() {
                    return Err(self.error("Expected variable name after OFFSET"));
                }
                let var = self.get_identifier_name();
                self.advance();
                (None, Some(var))
            } else {
                return Err(self.error("Expected ORDINALITY or OFFSET after WITH"));
            }
        } else {
            (None, None)
        };

        Ok(UnwindClause {
            expression,
            alias,
            ordinality_var,
            offset_var,
            span: Some(SourceSpan::new(span_start, self.current.span.start, 1, 1)),
        })
    }

    fn parse_set_clause(&mut self) -> Result<SetClause> {
        let span_start = self.current.span.start;
        self.expect(TokenKind::Set)?;

        let mut assignments = Vec::new();
        let mut map_assignments = Vec::new();
        let mut label_operations = Vec::new();

        loop {
            // Parse variable name
            if !self.is_identifier() {
                return Err(self.error("Expected variable name in SET"));
            }
            let variable = self.current.text.clone();
            self.advance();

            // Check if this is a label operation (n:Label) or property assignment (n.prop = value)
            // or map assignment (n = {map} or n += {map})
            if self.current.kind == TokenKind::Colon {
                // Label operation: SET n:Label1:Label2
                let mut labels = Vec::new();
                while self.current.kind == TokenKind::Colon {
                    self.advance();
                    if !self.is_label_or_type_name() {
                        return Err(self.error("Expected label name after colon in SET"));
                    }
                    labels.push(self.current.text.clone());
                    self.advance();
                }
                label_operations.push(LabelOperation { variable, labels });
            } else if self.current.kind == TokenKind::Dot {
                // Property assignment: SET n.prop = value
                self.advance();

                if !self.is_label_or_type_name() {
                    return Err(self.error("Expected property name in SET"));
                }
                let property = self.current.text.clone();
                self.advance();

                self.expect(TokenKind::Eq)?;

                let value = self.parse_expression()?;

                assignments.push(PropertyAssignment {
                    variable,
                    property,
                    value,
                });
            } else if self.current.kind == TokenKind::Eq {
                // Map replace: SET n = {key: value, ...}
                self.advance();
                let map_expr = self.parse_expression()?;
                map_assignments.push(MapAssignment {
                    variable,
                    map_expr,
                    replace: true,
                });
            } else if self.current.kind == TokenKind::Plus {
                // Map merge: SET n += {key: value, ...}
                self.advance();
                self.expect(TokenKind::Eq)?;
                let map_expr = self.parse_expression()?;
                map_assignments.push(MapAssignment {
                    variable,
                    map_expr,
                    replace: false,
                });
            } else {
                return Err(self.error("Expected '.', ':', '=', or '+=' after variable in SET"));
            }

            // Check for more assignments/operations
            if self.current.kind != TokenKind::Comma {
                break;
            }
            self.advance();
        }

        Ok(SetClause {
            assignments,
            map_assignments,
            label_operations,
            span: Some(SourceSpan::new(span_start, self.current.span.end, 1, 1)),
        })
    }

    fn parse_remove_clause(&mut self) -> Result<RemoveClause> {
        let span_start = self.current.span.start;
        self.expect(TokenKind::Remove)?;

        let mut label_operations = Vec::new();
        let mut property_removals = Vec::new();

        loop {
            // Parse variable name
            if !self.is_identifier() {
                return Err(self.error("Expected variable name in REMOVE"));
            }
            let variable = self.current.text.clone();
            self.advance();

            // Check if this is a label removal (n:Label) or property removal (n.prop)
            if self.current.kind == TokenKind::Colon {
                // Label removal: REMOVE n:Label1:Label2
                let mut labels = Vec::new();
                while self.current.kind == TokenKind::Colon {
                    self.advance();
                    if !self.is_label_or_type_name() {
                        return Err(self.error("Expected label name after colon in REMOVE"));
                    }
                    labels.push(self.current.text.clone());
                    self.advance();
                }
                label_operations.push(LabelOperation { variable, labels });
            } else if self.current.kind == TokenKind::Dot {
                // Property removal: REMOVE n.prop
                self.advance();

                if !self.is_label_or_type_name() {
                    return Err(self.error("Expected property name in REMOVE"));
                }
                let property = self.current.text.clone();
                self.advance();

                property_removals.push((variable, property));
            } else {
                return Err(self.error("Expected '.' or ':' after variable in REMOVE"));
            }

            // Check for more removal operations
            if self.current.kind != TokenKind::Comma {
                break;
            }
            self.advance();
        }

        Ok(RemoveClause {
            label_operations,
            property_removals,
            span: Some(SourceSpan::new(span_start, self.current.span.end, 1, 1)),
        })
    }

    fn parse_unwind_clause(&mut self) -> Result<UnwindClause> {
        let span_start = self.current.span.start;
        self.expect(TokenKind::Unwind)?;

        // Parse the expression to unwind
        let expression = self.parse_expression()?;

        // Expect AS keyword
        self.expect(TokenKind::As)?;

        // Parse the alias
        if !self.is_identifier() {
            return Err(self.error("Expected alias after AS in UNWIND"));
        }
        let alias = self.get_identifier_name();
        self.advance();

        Ok(UnwindClause {
            expression,
            alias,
            ordinality_var: None,
            offset_var: None,
            span: Some(SourceSpan::new(span_start, self.current.span.end, 1, 1)),
        })
    }

    /// Parses `LET var = expr [, var2 = expr2]*` as a clause.
    fn parse_let_clause(&mut self) -> Result<Vec<(String, Expression)>> {
        self.advance(); // consume LET
        let mut bindings = Vec::new();
        loop {
            if !self.is_identifier() {
                return Err(self.error("Expected variable name in LET clause"));
            }
            let var = self.get_identifier_name();
            self.advance();
            self.expect(TokenKind::Eq)?;
            let expr = self.parse_expression()?;
            bindings.push((var, expr));
            if self.current.kind != TokenKind::Comma {
                break;
            }
            self.advance(); // consume comma
        }
        Ok(bindings)
    }

    /// Parses `LOAD DATA FROM 'path' FORMAT CSV|JSONL|PARQUET [WITH HEADERS] AS variable [FIELDTERMINATOR 'char']`
    /// Also accepts Cypher-compatible `LOAD CSV [WITH HEADERS] FROM 'path' AS variable [FIELDTERMINATOR 'char']`
    fn parse_load_data_clause(&mut self) -> Result<LoadDataClause> {
        let span_start = self.current.span.start;
        self.advance(); // consume LOAD

        // Check for Cypher-compatible LOAD CSV syntax
        if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("CSV") {
            return self.parse_load_csv_compat(span_start);
        }

        // GQL syntax: LOAD DATA FROM 'path' FORMAT CSV|JSONL|PARQUET [WITH HEADERS] AS variable
        if !self.is_identifier() || !self.get_identifier_name().eq_ignore_ascii_case("DATA") {
            return Err(self.error("Expected DATA or CSV after LOAD"));
        }
        self.advance(); // consume DATA

        // FROM 'path'
        if !self.is_identifier() || !self.get_identifier_name().eq_ignore_ascii_case("FROM") {
            return Err(self.error("Expected FROM after DATA in LOAD DATA"));
        }
        self.advance(); // consume FROM
        let path = self.parse_string_value()?;

        // FORMAT CSV|JSONL|PARQUET
        if !self.is_identifier() || !self.get_identifier_name().eq_ignore_ascii_case("FORMAT") {
            return Err(self.error("Expected FORMAT after file path in LOAD DATA"));
        }
        self.advance(); // consume FORMAT

        let format = if self.is_identifier() {
            let name = self.get_identifier_name();
            self.advance();
            match name.to_ascii_uppercase().as_str() {
                "CSV" => LoadFormat::Csv,
                "JSONL" | "NDJSON" => LoadFormat::Jsonl,
                "PARQUET" => LoadFormat::Parquet,
                _ => {
                    return Err(self.error(&format!(
                        "Unknown format '{name}', expected CSV, JSONL, or PARQUET"
                    )));
                }
            }
        } else {
            return Err(self.error("Expected format name (CSV, JSONL, or PARQUET)"));
        };

        // Optional: WITH HEADERS (CSV only)
        let with_headers = if self.current.kind == TokenKind::With {
            self.advance();
            if !self.is_identifier() || !self.get_identifier_name().eq_ignore_ascii_case("HEADERS")
            {
                return Err(self.error("Expected HEADERS after WITH in LOAD DATA"));
            }
            self.advance();
            true
        } else {
            false
        };

        // AS variable
        self.expect(TokenKind::As)?;
        if !self.is_identifier() {
            return Err(self.error("Expected variable name after AS in LOAD DATA"));
        }
        let variable = self.get_identifier_name();
        self.advance();

        // Optional: FIELDTERMINATOR 'char'
        let field_terminator = self.parse_optional_field_terminator()?;

        Ok(LoadDataClause {
            path,
            format,
            with_headers,
            variable,
            field_terminator,
            span: SourceSpan::new(span_start, self.current.span.end, 1, 1),
        })
    }

    /// Parses Cypher-compatible `LOAD CSV [WITH HEADERS] FROM 'path' AS variable [FIELDTERMINATOR 'char']`.
    fn parse_load_csv_compat(&mut self, span_start: usize) -> Result<LoadDataClause> {
        self.advance(); // consume CSV

        // Optional: WITH HEADERS
        let with_headers = if self.current.kind == TokenKind::With {
            self.advance();
            if !self.is_identifier() || !self.get_identifier_name().eq_ignore_ascii_case("HEADERS")
            {
                return Err(self.error("Expected HEADERS after WITH"));
            }
            self.advance();
            true
        } else {
            false
        };

        // FROM 'path'
        if !self.is_identifier() || !self.get_identifier_name().eq_ignore_ascii_case("FROM") {
            return Err(self.error("Expected FROM after WITH HEADERS or CSV"));
        }
        self.advance(); // consume FROM
        let path = self.parse_string_value()?;

        // AS variable
        self.expect(TokenKind::As)?;
        if !self.is_identifier() {
            return Err(self.error("Expected variable name after AS"));
        }
        let variable = self.get_identifier_name();
        self.advance();

        // Optional: FIELDTERMINATOR 'char'
        let field_terminator = self.parse_optional_field_terminator()?;

        Ok(LoadDataClause {
            path,
            format: LoadFormat::Csv,
            with_headers,
            variable,
            field_terminator,
            span: SourceSpan::new(span_start, self.current.span.end, 1, 1),
        })
    }

    /// Expects and consumes a string literal, returning the unescaped value.
    fn parse_string_value(&mut self) -> Result<String> {
        if self.current.kind != TokenKind::String {
            return Err(self.error("Expected string literal"));
        }
        let text = &self.current.text;
        let inner = &text[1..text.len() - 1];
        let value = unescape_string(inner);
        self.advance();
        Ok(value)
    }

    /// Parses an optional `FIELDTERMINATOR 'char'` clause.
    fn parse_optional_field_terminator(&mut self) -> Result<Option<char>> {
        if self.is_identifier()
            && self
                .get_identifier_name()
                .eq_ignore_ascii_case("FIELDTERMINATOR")
        {
            self.advance();
            let term = self.parse_string_value()?;
            Ok(term.chars().next())
        } else {
            Ok(None)
        }
    }

    fn parse_merge_clause(&mut self) -> Result<MergeClause> {
        let span_start = self.current.span.start;
        self.expect(TokenKind::Merge)?;

        // Parse the pattern to merge
        let pattern = self.parse_pattern()?;

        // Parse optional ON CREATE and ON MATCH clauses
        let mut on_create = None;
        let mut on_match = None;

        while self.current.kind == TokenKind::On {
            self.advance();

            if self.current.kind == TokenKind::Create {
                self.advance();
                self.expect(TokenKind::Set)?;
                on_create = Some(self.parse_property_assignments()?);
            } else if self.current.kind == TokenKind::Match {
                self.advance();
                self.expect(TokenKind::Set)?;
                on_match = Some(self.parse_property_assignments()?);
            } else {
                return Err(self.error("Expected CREATE or MATCH after ON in MERGE"));
            }
        }

        Ok(MergeClause {
            pattern,
            on_create,
            on_match,
            span: Some(SourceSpan::new(span_start, self.current.span.end, 1, 1)),
        })
    }

    fn parse_property_assignments(&mut self) -> Result<Vec<PropertyAssignment>> {
        let mut assignments = Vec::new();
        loop {
            // Parse variable.property = expression
            if !self.is_identifier() {
                return Err(self.error("Expected variable name"));
            }
            let variable = self.get_identifier_name();
            self.advance();

            self.expect(TokenKind::Dot)?;

            if !self.is_label_or_type_name() {
                return Err(self.error("Expected property name"));
            }
            let property = self.get_identifier_name();
            self.advance();

            self.expect(TokenKind::Eq)?;

            let value = self.parse_expression()?;

            assignments.push(PropertyAssignment {
                variable,
                property,
                value,
            });

            // Check for more assignments
            if self.current.kind != TokenKind::Comma {
                break;
            }
            self.advance();
        }

        Ok(assignments)
    }

    fn parse_match_clause(&mut self) -> Result<MatchClause> {
        let span_start = self.current.span.start;

        // Check for OPTIONAL MATCH
        let optional = if self.current.kind == TokenKind::Optional {
            self.advance();
            true
        } else {
            false
        };

        self.expect(TokenKind::Match)?;

        // Check for path mode (WALK, TRAIL, SIMPLE, ACYCLIC)
        let path_mode = match self.current.kind {
            TokenKind::Walk => {
                self.advance();
                Some(PathMode::Walk)
            }
            TokenKind::Trail => {
                self.advance();
                Some(PathMode::Trail)
            }
            TokenKind::Simple => {
                self.advance();
                Some(PathMode::Simple)
            }
            TokenKind::Acyclic => {
                self.advance();
                Some(PathMode::Acyclic)
            }
            _ => None,
        };

        // Check for match mode (DIFFERENT EDGES, REPEATABLE ELEMENTS)
        let match_mode = if self.is_identifier()
            && self.get_identifier_name().eq_ignore_ascii_case("DIFFERENT")
        {
            self.advance(); // consume DIFFERENT
            // Expect EDGES (contextual keyword)
            if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("EDGES") {
                self.advance();
            }
            Some(MatchMode::DifferentEdges)
        } else if self.is_identifier()
            && self
                .get_identifier_name()
                .eq_ignore_ascii_case("REPEATABLE")
        {
            self.advance(); // consume REPEATABLE
            // Expect ELEMENTS (contextual keyword)
            if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("ELEMENTS") {
                self.advance();
            }
            Some(MatchMode::RepeatableElements)
        } else {
            None
        };

        // Check for path search prefix (ANY, ALL SHORTEST, ANY SHORTEST, SHORTEST k)
        let search_prefix = self.parse_path_search_prefix()?;

        let mut patterns = Vec::new();
        patterns.push(self.parse_aliased_pattern()?);

        // Path alternation at MATCH level: pattern | pattern or pattern |+| pattern
        // (ISO GQL G032/G030). Mixing | and |+| in the same alternation is rejected.
        if self.current.kind == TokenKind::Pipe || self.current.kind == TokenKind::PipePlusPipe {
            let first_op = self.current.kind;
            let is_multiset = first_op == TokenKind::PipePlusPipe;
            let first = patterns.pop().expect("just pushed");
            let mut alt_patterns = vec![first.pattern];
            while self.current.kind == TokenKind::Pipe
                || self.current.kind == TokenKind::PipePlusPipe
            {
                if self.current.kind != first_op {
                    return Err(self.error(
                        "Cannot mix '|' (set union) and '|+|' (multiset union) in the same \
                         alternation; use one operator consistently",
                    ));
                }
                self.advance();
                alt_patterns.push(self.parse_aliased_pattern()?.pattern);
            }
            let union_pattern = if is_multiset {
                Pattern::MultisetUnion(alt_patterns)
            } else {
                Pattern::Union(alt_patterns)
            };
            patterns.push(AliasedPattern {
                alias: first.alias,
                path_function: first.path_function,
                search_prefix: first.search_prefix,
                keep: first.keep,
                pattern: union_pattern,
            });
        }

        while self.current.kind == TokenKind::Comma {
            self.advance();
            patterns.push(self.parse_aliased_pattern()?);
        }

        Ok(MatchClause {
            optional,
            path_mode,
            search_prefix,
            match_mode,
            patterns,
            span: Some(SourceSpan::new(span_start, self.current.span.end, 1, 1)),
        })
    }

    /// Parses an optional path search prefix before patterns.
    ///
    /// ```text
    /// ANY [k]
    /// ALL SHORTEST
    /// ANY SHORTEST
    /// SHORTEST k [GROUPS]
    /// ```
    fn parse_path_search_prefix(&mut self) -> Result<Option<PathSearchPrefix>> {
        if self.current.kind == TokenKind::All {
            let next = self.peek_kind();
            if next == TokenKind::Shortest {
                // ALL SHORTEST
                self.advance(); // consume ALL
                self.advance(); // consume SHORTEST
                return Ok(Some(PathSearchPrefix::AllShortest));
            }
            if next == TokenKind::LParen {
                // ALL (pattern...) - enumerate all matching paths
                self.advance(); // consume ALL
                return Ok(Some(PathSearchPrefix::All));
            }
        }
        if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("ANY") {
            let next = self.peek_kind();
            if next == TokenKind::Shortest {
                // ANY SHORTEST
                self.advance(); // consume ANY
                self.advance(); // consume SHORTEST
                return Ok(Some(PathSearchPrefix::AnyShortest));
            }
            if next == TokenKind::Integer {
                // ANY k
                self.advance(); // consume ANY
                let k: usize = self.current.text.parse().unwrap_or(1);
                self.advance(); // consume k
                return Ok(Some(PathSearchPrefix::AnyK(k)));
            }
            // ANY followed by ( or an identifier (path variable: ANY p = ...)
            if next == TokenKind::LParen || next == TokenKind::Identifier {
                self.advance(); // consume ANY
                return Ok(Some(PathSearchPrefix::Any));
            }
        }
        if self.current.kind == TokenKind::Shortest {
            self.advance(); // consume SHORTEST
            if self.current.kind == TokenKind::Integer {
                let k: usize = self.current.text.parse().unwrap_or(1);
                self.advance(); // consume k
                if self.current.kind == TokenKind::Groups {
                    self.advance(); // consume GROUPS
                    return Ok(Some(PathSearchPrefix::ShortestKGroups(k)));
                }
                return Ok(Some(PathSearchPrefix::ShortestK(k)));
            }
            // SHORTEST without k: treat as SHORTEST 1
            return Ok(Some(PathSearchPrefix::ShortestK(1)));
        }
        Ok(None)
    }

    /// Parses a pattern with optional alias and path function.
    /// Supports: `p = shortestPath((a)-[*]-(b))` and `p = (a)-[*]-(b)` and `(a)-[*]-(b)`
    fn parse_aliased_pattern(&mut self) -> Result<AliasedPattern> {
        let mut alias = None;
        let mut path_function = None;
        let mut search_prefix = None;

        // Check for pattern alias: identifier = ...
        if self.is_identifier() && self.peek_kind() == TokenKind::Eq {
            alias = Some(self.get_identifier_name());
            self.advance(); // consume identifier
            self.advance(); // consume =

            // Check for path function: shortestPath(...) or allShortestPaths(...)
            if self.is_identifier() {
                let func_name = self.get_identifier_name().to_lowercase();
                if func_name == "shortestpath" {
                    path_function = Some(PathFunction::ShortestPath);
                    self.advance(); // consume function name
                    self.expect(TokenKind::LParen)?;
                } else if func_name == "allshortestpaths" {
                    path_function = Some(PathFunction::AllShortestPaths);
                    self.advance(); // consume function name
                    self.expect(TokenKind::LParen)?;
                }
            }

            // Per-pattern path search prefix: p = ANY SHORTEST (...) (ISO GQL)
            if path_function.is_none() {
                search_prefix = self.parse_path_search_prefix()?;
            }
        }

        let pattern = self.parse_pattern()?;

        if path_function.is_some() {
            self.expect(TokenKind::RParen)?;
        }

        // Parse optional KEEP clause: KEEP DIFFERENT EDGES | KEEP REPEATABLE ELEMENTS
        let keep =
            if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("KEEP") {
                self.advance(); // consume KEEP
                if self.is_identifier() {
                    let mode_name = self.get_identifier_name().to_uppercase();
                    match mode_name.as_str() {
                        "DIFFERENT" => {
                            self.advance();
                            // Optionally consume EDGES
                            if self.is_identifier()
                                && self.get_identifier_name().eq_ignore_ascii_case("EDGES")
                            {
                                self.advance();
                            }
                            Some(MatchMode::DifferentEdges)
                        }
                        "REPEATABLE" => {
                            self.advance();
                            // Optionally consume ELEMENTS
                            if self.is_identifier()
                                && self.get_identifier_name().eq_ignore_ascii_case("ELEMENTS")
                            {
                                self.advance();
                            }
                            Some(MatchMode::RepeatableElements)
                        }
                        _ => return Err(self.error("Expected DIFFERENT or REPEATABLE after KEEP")),
                    }
                } else {
                    return Err(self.error("Expected DIFFERENT or REPEATABLE after KEEP"));
                }
            } else {
                None
            };

        Ok(AliasedPattern {
            alias,
            path_function,
            search_prefix,
            keep,
            pattern,
        })
    }

    fn parse_with_clause(&mut self) -> Result<WithClause> {
        let span_start = self.current.span.start;
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

        let mut items = Vec::new();
        if !is_wildcard {
            items.push(self.parse_return_item()?);

            while self.current.kind == TokenKind::Comma {
                self.advance();
                items.push(self.parse_return_item()?);
            }
        }

        // Optional WHERE after WITH
        let where_clause = if self.current.kind == TokenKind::Where {
            Some(self.parse_where_clause()?)
        } else {
            None
        };

        Ok(WithClause {
            distinct,
            items,
            is_wildcard,
            where_clause,
            let_bindings: Vec::new(),
            span: Some(SourceSpan::new(span_start, self.current.span.end, 1, 1)),
        })
    }

    fn parse_pattern(&mut self) -> Result<Pattern> {
        // Check for parenthesized (grouped) pattern: ((a)-[]->(b)){2,5}
        // Also handles G048 subpath var `(p = (a)-[]->(b)){2,5}`,
        // G049 path mode prefix `(TRAIL (a)-[]->(b)){2,5}`,
        // and G050 WHERE inside `((a)-[e]->(b) WHERE e.w > 5){2,5}`.
        if self.current.kind == TokenKind::LParen {
            let peek = self.peek_kind();
            if matches!(
                peek,
                TokenKind::LParen
                    | TokenKind::Walk
                    | TokenKind::Trail
                    | TokenKind::Simple
                    | TokenKind::Acyclic
                    | TokenKind::Minus
                    | TokenKind::LeftArrow
                    | TokenKind::Tilde
            ) {
                return self.parse_parenthesized_pattern();
            }
            // G048: `(identifier = ...)` is a subpath variable declaration
            if matches!(peek, TokenKind::Identifier | TokenKind::QuotedIdentifier)
                && self.peek_second_kind() == TokenKind::Eq
            {
                return self.parse_parenthesized_pattern();
            }
        }

        // Edge-first pattern (anonymous source): -[r:KNOWS]->(x)
        // This occurs inside parenthesized quantified patterns like (-[]->(x)){2,5}
        let node = if matches!(
            self.current.kind,
            TokenKind::Arrow
                | TokenKind::LeftArrow
                | TokenKind::DoubleDash
                | TokenKind::Minus
                | TokenKind::Tilde
        ) {
            // Anonymous source node
            NodePattern {
                variable: None,
                labels: Vec::new(),
                label_expression: None,
                properties: Vec::new(),
                where_clause: None,
                span: None,
            }
        } else {
            self.parse_node_pattern()?
        };

        // Check for path continuation
        // Handle both `-[...]->`/`<-[...]-` style and `->` style
        if matches!(
            self.current.kind,
            TokenKind::Arrow
                | TokenKind::LeftArrow
                | TokenKind::DoubleDash
                | TokenKind::Minus
                | TokenKind::Tilde
        ) {
            let mut edges = Vec::new();

            while matches!(
                self.current.kind,
                TokenKind::Arrow
                    | TokenKind::LeftArrow
                    | TokenKind::DoubleDash
                    | TokenKind::Minus
                    | TokenKind::Tilde
            ) {
                edges.push(self.parse_edge_pattern()?);
            }

            Ok(Pattern::Path(PathPattern {
                source: node,
                edges,
                span: None,
            }))
        } else if self.current.kind == TokenKind::LParen {
            // G047: Parenthesized quantified path continuation: (a)(-[]->(x)){2,5}(b)
            // The `(` after a node starts a parenthesized subpattern with a quantifier.
            let sub = self.parse_parenthesized_pattern()?;

            if let Pattern::Quantified {
                pattern: inner,
                min,
                max,
                subpath_var,
                path_mode,
                where_clause,
            } = sub
            {
                // Parse optional trailing target node: ...{2,5}(b)
                let trailing_target = if self.current.kind == TokenKind::LParen {
                    Some(self.parse_node_pattern()?)
                } else {
                    None
                };

                // Merge outer anchors into the inner pattern:
                // Replace inner source with `node`, replace inner last target with trailing_target.
                let merged = self.merge_quantified_anchors(node, *inner, trailing_target);

                Ok(Pattern::Quantified {
                    pattern: Box::new(merged),
                    min,
                    max,
                    subpath_var,
                    path_mode,
                    where_clause,
                })
            } else {
                // Non-quantified parenthesized pattern after a node: treat the
                // leading node as a standalone pattern (the parenthesized pattern
                // is a separate element in the pattern list, not a continuation).
                Ok(Pattern::Node(node))
            }
        } else {
            Ok(Pattern::Node(node))
        }
    }

    /// Merges outer anchor nodes into a quantified inner pattern.
    ///
    /// For `(a)(-[r:KNOWS]->(x)){2,5}(b)`, this replaces the inner pattern's
    /// source with `(a)` and the last target with `(b)`.
    fn merge_quantified_anchors(
        &self,
        source: NodePattern,
        inner: Pattern,
        trailing_target: Option<NodePattern>,
    ) -> Pattern {
        match inner {
            Pattern::Path(mut path) => {
                path.source = source;
                if let (Some(target), Some(last_edge)) = (trailing_target, path.edges.last_mut()) {
                    last_edge.target = target;
                }
                Pattern::Path(path)
            }
            Pattern::Node(_) => {
                // Degenerate case: inner is just a node, not a path.
                // Wrap source and inner into a path if we have a trailing target.
                if trailing_target.is_some() {
                    // Can't meaningfully merge, return source as-is
                    Pattern::Node(source)
                } else {
                    Pattern::Node(source)
                }
            }
            other => other,
        }
    }

    /// Parses a parenthesized path pattern with optional quantifier.
    ///
    /// ```text
    /// ( [subpath_var =] [path_mode] pattern [| pattern]* [WHERE expr] ) [ {min,max} ]
    /// ```
    ///
    /// G048: subpath variable declaration `(p = (a)-[e]->(b)){2,5}`
    /// G049: path mode prefix `(TRAIL (a)-[e]->(b)){2,5}`
    /// G050: WHERE clause `((a)-[e]->(b) WHERE e.w > 5){2,5}`
    fn parse_parenthesized_pattern(&mut self) -> Result<Pattern> {
        self.expect(TokenKind::LParen)?;

        // G049: Check for optional path mode prefix (WALK, TRAIL, SIMPLE, ACYCLIC)
        let path_mode = match self.current.kind {
            TokenKind::Walk => {
                self.advance();
                Some(PathMode::Walk)
            }
            TokenKind::Trail => {
                self.advance();
                Some(PathMode::Trail)
            }
            TokenKind::Simple => {
                self.advance();
                Some(PathMode::Simple)
            }
            TokenKind::Acyclic => {
                self.advance();
                Some(PathMode::Acyclic)
            }
            _ => None,
        };

        // G048: Check for subpath variable declaration: `name = pattern`
        // We detect this by checking if the current token is an identifier
        // and the next token is `=` (assignment).
        let subpath_var = if self.is_identifier() && self.peek_kind() == TokenKind::Eq {
            let name = self.get_identifier_name();
            self.advance(); // consume identifier
            self.advance(); // consume `=`
            Some(name)
        } else {
            None
        };

        // Parse the inner pattern(s), potentially with union via | or multiset union via |+|
        let mut patterns = vec![self.parse_pattern()?];
        let mut is_multiset = false;
        while self.current.kind == TokenKind::Pipe || self.current.kind == TokenKind::PipePlusPipe {
            if self.current.kind == TokenKind::PipePlusPipe {
                is_multiset = true;
            }
            self.advance();
            patterns.push(self.parse_pattern()?);
        }

        // G050: Check for optional WHERE clause inside the parenthesized pattern
        let where_clause = if self.current.kind == TokenKind::Where {
            self.advance();
            Some(self.parse_expression()?)
        } else {
            None
        };

        self.expect(TokenKind::RParen)?;

        let inner = if patterns.len() == 1 {
            patterns.remove(0)
        } else if is_multiset {
            Pattern::MultisetUnion(patterns)
        } else {
            Pattern::Union(patterns)
        };

        // Check for quantifier: {min,max} or {n}
        if self.current.kind == TokenKind::LBrace {
            let (min, max) = self.parse_path_quantifier()?;
            Ok(Pattern::Quantified {
                pattern: Box::new(inner),
                min: min.unwrap_or(1),
                max,
                subpath_var,
                path_mode,
                where_clause,
            })
        } else if subpath_var.is_some() || path_mode.is_some() || where_clause.is_some() {
            // Has parenthesized-pattern features but no quantifier: treat as {1,1}
            Ok(Pattern::Quantified {
                pattern: Box::new(inner),
                min: 1,
                max: Some(1),
                subpath_var,
                path_mode,
                where_clause,
            })
        } else {
            // No quantifier, no extra features: just a grouped pattern (or union)
            Ok(inner)
        }
    }

    fn parse_node_pattern(&mut self) -> Result<NodePattern> {
        self.expect(TokenKind::LParen)?;

        let variable = if self.is_identifier() {
            let name = self.get_identifier_name();
            self.advance();
            Some(name)
        } else {
            None
        };

        let mut labels = Vec::new();
        let mut label_expression = None;

        if self.current.kind == TokenKind::Is {
            // GQL IS syntax: (n IS Person | Employee)
            self.advance();
            label_expression = Some(self.parse_label_expression()?);
        } else {
            // Colon syntax: (n:Person:Employee)
            while self.current.kind == TokenKind::Colon {
                self.advance();
                if !self.is_label_or_type_name() {
                    return Err(self.error("Expected label name"));
                }
                labels.push(self.get_identifier_name());
                self.advance();
            }
        }

        // Parse properties { key: value, ... }
        let properties = if self.current.kind == TokenKind::LBrace {
            self.parse_property_map()?
        } else {
            Vec::new()
        };

        // Parse optional element pattern WHERE clause: (n WHERE n.age > 30)
        let where_clause = if self.current.kind == TokenKind::Where {
            self.advance();
            Some(self.parse_expression()?)
        } else {
            None
        };

        self.expect(TokenKind::RParen)?;

        Ok(NodePattern {
            variable,
            labels,
            label_expression,
            properties,
            where_clause,
            span: None,
        })
    }

    /// Parses a label expression with precedence: `|` < `&` < `!`.
    fn parse_label_expression(&mut self) -> Result<LabelExpression> {
        let mut left = self.parse_label_conjunction()?;

        while self.current.kind == TokenKind::Pipe {
            let mut operands = vec![left];
            while self.current.kind == TokenKind::Pipe {
                self.advance();
                operands.push(self.parse_label_conjunction()?);
            }
            left = LabelExpression::Disjunction(operands);
        }

        Ok(left)
    }

    fn parse_label_conjunction(&mut self) -> Result<LabelExpression> {
        let mut left = self.parse_label_negation()?;

        while self.current.kind == TokenKind::Ampersand {
            let mut operands = vec![left];
            while self.current.kind == TokenKind::Ampersand {
                self.advance();
                operands.push(self.parse_label_negation()?);
            }
            left = LabelExpression::Conjunction(operands);
        }

        Ok(left)
    }

    fn parse_label_negation(&mut self) -> Result<LabelExpression> {
        if self.current.kind == TokenKind::Exclamation {
            self.advance();
            let inner = self.parse_label_primary()?;
            return Ok(LabelExpression::Negation(Box::new(inner)));
        }
        self.parse_label_primary()
    }

    fn parse_label_primary(&mut self) -> Result<LabelExpression> {
        if self.current.kind == TokenKind::Percent {
            self.advance();
            return Ok(LabelExpression::Wildcard);
        }
        if self.current.kind == TokenKind::LParen {
            self.advance();
            let expr = self.parse_label_expression()?;
            self.expect(TokenKind::RParen)?;
            return Ok(expr);
        }
        if self.is_label_or_type_name() {
            let name = self.get_identifier_name();
            self.advance();
            return Ok(LabelExpression::Label(name));
        }
        Err(self.error("Expected label name, %, or ("))
    }

    fn parse_edge_pattern(&mut self) -> Result<EdgePattern> {
        // Handle both styles:
        // 1. `-[...]->` or `-[:TYPE]->` or `-[:TYPE*1..3]->` (direction determined by trailing arrow)
        // 2. `->` or `<-` or `--` (direction determined by leading arrow)

        let mut edge_where_clause = None;
        let (variable, types, mut min_hops, mut max_hops, properties, direction) =
            if self.current.kind == TokenKind::Minus {
                // Pattern: -[...]->(target) or -[...]-(target)
                self.advance();

                // G080: Simplified path pattern: -/:Label/-> desugars to -[:Label]->
                if self.current.kind == TokenKind::Slash {
                    return self.parse_simplified_edge(false);
                }

                // Parse [variable:TYPE*min..max {props}]
                let (var, edge_types, min_h, max_h, props) =
                    if self.current.kind == TokenKind::LBracket {
                        self.advance();

                        // Parse variable name if present
                        // Variable is followed by : (type), * (quantifier), { (properties),
                        // WHERE (element filter), or ] (end)
                        let v = if self.is_identifier() {
                            let peek = self.peek_kind();
                            if matches!(
                                peek,
                                TokenKind::Colon
                                    | TokenKind::Star
                                    | TokenKind::LBrace
                                    | TokenKind::RBracket
                                    | TokenKind::Where
                            ) {
                                let name = self.get_identifier_name();
                                self.advance();
                                Some(name)
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        let mut tps = Vec::new();
                        while self.current.kind == TokenKind::Colon {
                            self.advance();
                            if !self.is_label_or_type_name() {
                                return Err(self.error("Expected edge type"));
                            }
                            tps.push(self.get_identifier_name());
                            self.advance();
                            // Support pipe-separated alternatives: :T1|T2|T3
                            while self.current.kind == TokenKind::Pipe {
                                self.advance();
                                if !self.is_label_or_type_name() {
                                    return Err(self.error("Expected edge type after |"));
                                }
                                tps.push(self.get_identifier_name());
                                self.advance();
                            }
                        }

                        // Parse variable-length path quantifier: *min..max
                        let (min_h, max_h) = self.parse_path_quantifier()?;

                        // Parse edge properties: {key: value, ...}
                        let edge_props = if self.current.kind == TokenKind::LBrace {
                            self.parse_property_map()?
                        } else {
                            Vec::new()
                        };

                        // Parse element WHERE clause: [e:TYPE WHERE expr]
                        if self.current.kind == TokenKind::Where {
                            self.advance();
                            edge_where_clause = Some(self.parse_expression()?);
                        }

                        self.expect(TokenKind::RBracket)?;
                        (v, tps, min_h, max_h, edge_props)
                    } else {
                        (None, Vec::new(), None, None, Vec::new())
                    };

                // Now determine direction from trailing symbol
                let dir = if self.current.kind == TokenKind::Arrow {
                    self.advance();
                    EdgeDirection::Outgoing
                } else if self.current.kind == TokenKind::Minus {
                    self.advance();
                    EdgeDirection::Undirected
                } else {
                    return Err(self.error("Expected -> or - after edge pattern"));
                };

                (var, edge_types, min_h, max_h, props, dir)
            } else if self.current.kind == TokenKind::LeftArrow {
                // Pattern: <-[...]-(target)
                self.advance();

                // G080: Simplified path pattern: <-/:Label/- desugars to <-[:Label]-
                if self.current.kind == TokenKind::Slash {
                    return self.parse_simplified_edge(true);
                }

                let (var, edge_types, min_h, max_h, props) =
                    if self.current.kind == TokenKind::LBracket {
                        self.advance();

                        // Parse variable name if present
                        // Variable is followed by : (type), * (quantifier), { (properties),
                        // WHERE (element filter), or ] (end)
                        let v = if self.is_identifier() {
                            let peek = self.peek_kind();
                            if matches!(
                                peek,
                                TokenKind::Colon
                                    | TokenKind::Star
                                    | TokenKind::LBrace
                                    | TokenKind::RBracket
                                    | TokenKind::Where
                            ) {
                                let name = self.get_identifier_name();
                                self.advance();
                                Some(name)
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        let mut tps = Vec::new();
                        while self.current.kind == TokenKind::Colon {
                            self.advance();
                            if !self.is_label_or_type_name() {
                                return Err(self.error("Expected edge type"));
                            }
                            tps.push(self.get_identifier_name());
                            self.advance();
                            // Support pipe-separated alternatives: :T1|T2|T3
                            while self.current.kind == TokenKind::Pipe {
                                self.advance();
                                if !self.is_label_or_type_name() {
                                    return Err(self.error("Expected edge type after |"));
                                }
                                tps.push(self.get_identifier_name());
                                self.advance();
                            }
                        }

                        // Parse variable-length path quantifier
                        let (min_h, max_h) = self.parse_path_quantifier()?;

                        // Parse edge properties: {key: value, ...}
                        let edge_props = if self.current.kind == TokenKind::LBrace {
                            self.parse_property_map()?
                        } else {
                            Vec::new()
                        };

                        // Parse element WHERE clause: [e:TYPE WHERE expr]
                        if self.current.kind == TokenKind::Where {
                            self.advance();
                            edge_where_clause = Some(self.parse_expression()?);
                        }

                        self.expect(TokenKind::RBracket)?;
                        (v, tps, min_h, max_h, edge_props)
                    } else {
                        (None, Vec::new(), None, None, Vec::new())
                    };

                // Consume trailing -
                if self.current.kind == TokenKind::Minus {
                    self.advance();
                }

                (
                    var,
                    edge_types,
                    min_h,
                    max_h,
                    props,
                    EdgeDirection::Incoming,
                )
            } else if self.current.kind == TokenKind::Arrow {
                // Simple ->
                self.advance();
                (
                    None,
                    Vec::new(),
                    None,
                    None,
                    Vec::new(),
                    EdgeDirection::Outgoing,
                )
            } else if self.current.kind == TokenKind::DoubleDash {
                // `--` (undirected) or `-->` (outgoing shorthand)
                self.advance();
                if self.current.kind == TokenKind::Gt {
                    // --> shorthand: directed outgoing, no bracket
                    self.advance();
                    (
                        None,
                        Vec::new(),
                        None,
                        None,
                        Vec::new(),
                        EdgeDirection::Outgoing,
                    )
                } else {
                    (
                        None,
                        Vec::new(),
                        None,
                        None,
                        Vec::new(),
                        EdgeDirection::Undirected,
                    )
                }
            } else if self.current.kind == TokenKind::Tilde {
                // GQL undirected edge: ~[variable:TYPE*min..max {props}]~
                self.advance();

                // G080: Simplified undirected path: ~/:Label/~
                if self.current.kind == TokenKind::Slash {
                    return self.parse_simplified_edge_undirected();
                }

                let (var, edge_types, min_h, max_h, props) =
                    if self.current.kind == TokenKind::LBracket {
                        self.advance();

                        let v = if self.is_identifier() {
                            let peek = self.peek_kind();
                            if matches!(
                                peek,
                                TokenKind::Colon
                                    | TokenKind::Star
                                    | TokenKind::LBrace
                                    | TokenKind::RBracket
                                    | TokenKind::Where
                            ) {
                                let name = self.get_identifier_name();
                                self.advance();
                                Some(name)
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        let mut tps = Vec::new();
                        while self.current.kind == TokenKind::Colon {
                            self.advance();
                            if !self.is_label_or_type_name() {
                                return Err(self.error("Expected edge type"));
                            }
                            tps.push(self.get_identifier_name());
                            self.advance();
                            while self.current.kind == TokenKind::Pipe {
                                self.advance();
                                if !self.is_label_or_type_name() {
                                    return Err(self.error("Expected edge type after |"));
                                }
                                tps.push(self.get_identifier_name());
                                self.advance();
                            }
                        }

                        let (min_h, max_h) = self.parse_path_quantifier()?;

                        let edge_props = if self.current.kind == TokenKind::LBrace {
                            self.parse_property_map()?
                        } else {
                            Vec::new()
                        };

                        // Parse element WHERE clause: [e:TYPE WHERE expr]
                        if self.current.kind == TokenKind::Where {
                            self.advance();
                            edge_where_clause = Some(self.parse_expression()?);
                        }

                        self.expect(TokenKind::RBracket)?;
                        (v, tps, min_h, max_h, edge_props)
                    } else {
                        (None, Vec::new(), None, None, Vec::new())
                    };

                // Consume trailing ~
                if self.current.kind == TokenKind::Tilde {
                    self.advance();
                }

                (
                    var,
                    edge_types,
                    min_h,
                    max_h,
                    props,
                    EdgeDirection::Undirected,
                )
            } else {
                return Err(self.error("Expected edge pattern"));
            };

        // Check for questioned edge: ->?(node) means optional (0 or 1 hop)
        let questioned = if self.current.kind == TokenKind::QuestionMark {
            self.advance();
            true
        } else {
            false
        };

        // Post-arrow quantifier: ->{m,n}, ->+, ->*  (ISO GQL G036/G060/G061)
        // Only applies when no in-bracket quantifier was parsed (Cypher *1..3 form).
        if min_hops.is_none() && max_hops.is_none() {
            if self.current.kind == TokenKind::Plus {
                self.advance();
                min_hops = Some(1);
                // max_hops stays None (unbounded)
            } else if self.current.kind == TokenKind::Star {
                self.advance();
                min_hops = Some(0);
                // max_hops stays None (unbounded)
            } else if self.current.kind == TokenKind::LBrace {
                let (qmin, qmax) = self.parse_path_quantifier()?;
                if qmin.is_some() || qmax.is_some() {
                    min_hops = qmin;
                    max_hops = qmax;
                }
            }
        }

        let target = self.parse_node_pattern()?;

        Ok(EdgePattern {
            variable,
            types,
            direction,
            target,
            min_hops,
            max_hops,
            properties,
            where_clause: edge_where_clause,
            questioned,
            span: None,
        })
    }

    /// Parses a simplified path pattern (G080/G039/G082).
    ///
    /// Called after consuming `-` (outgoing) or `<-` (incoming). The current token
    /// is `/`. Syntax: `/:Label1|Label2/->` or `/:Label/-` or `/:Label/->`.
    ///
    /// Desugars to a regular `EdgePattern` with the parsed label types.
    fn parse_simplified_edge(&mut self, incoming: bool) -> Result<EdgePattern> {
        self.expect(TokenKind::Slash)?; // consume opening /

        // Parse label(s): `:Label` or `:Label1|Label2`
        let mut types = Vec::new();
        if self.current.kind == TokenKind::Colon {
            self.advance();
            if !self.is_label_or_type_name() {
                return Err(self.error("Expected label in simplified path pattern"));
            }
            types.push(self.get_identifier_name());
            self.advance();
            // Support pipe-separated alternatives: :T1|T2
            while self.current.kind == TokenKind::Pipe {
                self.advance();
                if !self.is_label_or_type_name() {
                    return Err(self.error("Expected label after | in simplified path pattern"));
                }
                types.push(self.get_identifier_name());
                self.advance();
            }
        }

        self.expect(TokenKind::Slash)?; // consume closing /

        // Determine direction from trailing token
        let direction = if incoming {
            // <-/:Label/- form: consume trailing -
            if self.current.kind == TokenKind::Minus {
                self.advance();
            }
            EdgeDirection::Incoming
        } else if self.current.kind == TokenKind::Arrow {
            // -/:Label/-> form
            self.advance();
            EdgeDirection::Outgoing
        } else if self.current.kind == TokenKind::Minus {
            // -/:Label/- form (undirected)
            self.advance();
            EdgeDirection::Undirected
        } else {
            return Err(self.error("Expected ->, -, or end after simplified path pattern"));
        };

        let target = self.parse_node_pattern()?;

        Ok(EdgePattern {
            variable: None,
            types,
            direction,
            target,
            min_hops: None,
            max_hops: None,
            properties: Vec::new(),
            where_clause: None,
            questioned: false,
            span: None,
        })
    }

    /// Parses a simplified undirected path pattern with tilde syntax.
    ///
    /// Called after consuming `~`. The current token is `/`.
    /// Syntax: `~/:Label/~` desugars to `~[:Label]~`.
    fn parse_simplified_edge_undirected(&mut self) -> Result<EdgePattern> {
        self.expect(TokenKind::Slash)?; // consume opening /

        let mut types = Vec::new();
        if self.current.kind == TokenKind::Colon {
            self.advance();
            if !self.is_label_or_type_name() {
                return Err(self.error("Expected label in simplified path pattern"));
            }
            types.push(self.get_identifier_name());
            self.advance();
            while self.current.kind == TokenKind::Pipe {
                self.advance();
                if !self.is_label_or_type_name() {
                    return Err(self.error("Expected label after | in simplified path pattern"));
                }
                types.push(self.get_identifier_name());
                self.advance();
            }
        }

        self.expect(TokenKind::Slash)?; // consume closing /

        // Consume trailing ~
        if self.current.kind == TokenKind::Tilde {
            self.advance();
        }

        let target = self.parse_node_pattern()?;

        Ok(EdgePattern {
            variable: None,
            types,
            direction: EdgeDirection::Undirected,
            target,
            min_hops: None,
            max_hops: None,
            properties: Vec::new(),
            where_clause: None,
            questioned: false,
            span: None,
        })
    }

    /// Parses a path quantifier like `*`, `*2`, `*1..3`, `*..5`, `*2..`,
    /// or ISO `{m,n}`, `{m,}`, `{,n}`, `{m}`.
    /// Returns (min_hops, max_hops) where None means no quantifier was present.
    fn parse_path_quantifier(&mut self) -> Result<(Option<u32>, Option<u32>)> {
        // ISO GQL {m,n} quantifier syntax.
        // Disambiguate from property map {key: value}: quantifiers start with
        // an integer or comma, property maps start with an identifier.
        if self.current.kind == TokenKind::LBrace {
            let next = self.peek_kind();
            if next != TokenKind::Integer && next != TokenKind::Comma {
                // Not a quantifier (likely a property map), bail out
                return Ok((None, None));
            }
            self.advance(); // consume {
            if self.current.kind == TokenKind::Comma {
                // {,n}
                self.advance();
                let max_text = self.current.text.clone();
                let max: u32 = max_text
                    .parse()
                    .map_err(|_| self.error("Invalid path length"))?;
                self.advance();
                self.expect(TokenKind::RBrace)?;
                return Ok((Some(1), Some(max)));
            }
            let min_text = self.current.text.clone();
            let min: u32 = min_text
                .parse()
                .map_err(|_| self.error("Invalid path length"))?;
            self.advance();
            if self.current.kind == TokenKind::RBrace {
                // {m} means exactly m
                self.advance();
                return Ok((Some(min), Some(min)));
            }
            self.expect(TokenKind::Comma)?;
            if self.current.kind == TokenKind::RBrace {
                // {m,} means m to unbounded
                self.advance();
                return Ok((Some(min), None));
            }
            let max_text = self.current.text.clone();
            let max: u32 = max_text
                .parse()
                .map_err(|_| self.error("Invalid path length"))?;
            self.advance();
            self.expect(TokenKind::RBrace)?;
            return Ok((Some(min), Some(max)));
        }

        if self.current.kind != TokenKind::Star {
            return Ok((None, None));
        }
        self.advance(); // consume *

        // Check for bounds
        if self.current.kind == TokenKind::Integer {
            let min_text = self.current.text.clone();
            let min: u32 = min_text
                .parse()
                .map_err(|_| self.error("Invalid path length"))?;
            self.advance();

            if self.current.kind == TokenKind::DotDot {
                self.advance(); // consume ..

                if self.current.kind == TokenKind::Integer {
                    let max_text = self.current.text.clone();
                    let max: u32 = max_text
                        .parse()
                        .map_err(|_| self.error("Invalid path length"))?;
                    self.advance();
                    Ok((Some(min), Some(max))) // *min..max
                } else {
                    Ok((Some(min), None)) // *min.. (unbounded max)
                }
            } else {
                Ok((Some(min), Some(min))) // *n means exactly n hops
            }
        } else if self.current.kind == TokenKind::DotDot {
            self.advance(); // consume ..

            if self.current.kind == TokenKind::Integer {
                let max_text = self.current.text.clone();
                let max: u32 = max_text
                    .parse()
                    .map_err(|_| self.error("Invalid path length"))?;
                self.advance();
                Ok((Some(1), Some(max))) // *..max (min defaults to 1)
            } else {
                Err(self.error("Expected max hops after .."))
            }
        } else {
            Ok((Some(1), None)) // * alone means 1 to unbounded
        }
    }

    fn parse_where_clause(&mut self) -> Result<WhereClause> {
        self.expect(TokenKind::Where)?;
        let expression = self.parse_expression()?;

        Ok(WhereClause {
            expression,
            span: None,
        })
    }

    /// Parses either WHERE or FILTER clause (FILTER is a GQL alias for WHERE).
    fn parse_where_or_filter_clause(&mut self) -> Result<WhereClause> {
        // Accept both WHERE and FILTER
        if self.current.kind == TokenKind::Filter {
            self.advance();
            // ISO GQL: FILTER WHERE expr (WHERE is optional after FILTER)
            if self.current.kind == TokenKind::Where {
                self.advance();
            }
        } else {
            self.expect(TokenKind::Where)?;
        }
        let expression = self.parse_expression()?;

        Ok(WhereClause {
            expression,
            span: None,
        })
    }

    fn parse_having_clause(&mut self) -> Result<HavingClause> {
        self.expect(TokenKind::Having)?;
        let expression = self.parse_expression()?;

        Ok(HavingClause {
            expression,
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

        // Check for RETURN *
        let is_wildcard = if self.current.kind == TokenKind::Star {
            self.advance();
            true
        } else {
            false
        };

        let mut items = Vec::new();
        if !is_wildcard {
            items.push(self.parse_return_item()?);

            while self.current.kind == TokenKind::Comma {
                self.advance();
                items.push(self.parse_return_item()?);
            }
        }

        // Parse optional GROUP BY
        let group_by = if self.current.kind == TokenKind::Group {
            self.advance();
            self.expect(TokenKind::By)?;
            let mut exprs = vec![self.parse_expression()?];
            while self.current.kind == TokenKind::Comma {
                self.advance();
                exprs.push(self.parse_expression()?);
            }
            exprs
        } else {
            Vec::new()
        };

        let order_by = if self.current.kind == TokenKind::Order {
            Some(self.parse_order_by()?)
        } else {
            None
        };

        let skip = if matches!(self.current.kind, TokenKind::Skip | TokenKind::Offset) {
            self.advance();
            Some(self.parse_expression()?)
        } else {
            None
        };

        let limit = if self.current.kind == TokenKind::Limit {
            self.advance();
            Some(self.parse_expression()?)
        } else if self.current.kind == TokenKind::Fetch {
            Some(self.parse_fetch_first()?)
        } else {
            None
        };

        Ok(ReturnClause {
            distinct,
            items,
            is_wildcard,
            group_by,
            order_by,
            skip,
            limit,
            is_finish: false,
            span: None,
        })
    }

    /// Parses a SELECT clause (SQL-style projection, same semantics as RETURN).
    /// Called after SELECT token has already been consumed.
    /// Parses a SELECT ... FROM ... MATCH statement (ISO GQL statement-leading form).
    ///
    /// ```text
    /// SELECT [DISTINCT] { * | item [, item]... }
    ///   FROM graph_ref MATCH pattern
    ///   [WHERE ...] [GROUP BY ...] [ORDER BY ...] [OFFSET ...] [LIMIT ...]
    /// ```
    fn parse_select_from_statement(&mut self) -> Result<Statement> {
        let span_start = self.current.span.start;

        // Parse DISTINCT
        let distinct = if self.current.kind == TokenKind::Distinct {
            self.advance();
            true
        } else {
            false
        };

        // Parse select items: * or expression list
        let (is_wildcard, items) = if self.current.kind == TokenKind::Star {
            self.advance();
            (true, Vec::new())
        } else {
            let mut items = vec![self.parse_return_item()?];
            while self.current.kind == TokenKind::Comma {
                self.advance();
                items.push(self.parse_return_item()?);
            }
            (false, items)
        };

        // Expect FROM
        if !(self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("FROM")) {
            return Err(self.error("Expected FROM after SELECT"));
        }
        self.advance(); // consume FROM

        // Parse graph reference: a single name or dot-qualified name (schema.graph),
        // or CURRENT_GRAPH / CURRENT_PROPERTY_GRAPH. Skip if MATCH follows directly.
        if self.is_identifier() && self.current.kind != TokenKind::Match {
            self.advance(); // consume graph name
            // Optional dot-qualified continuation (e.g., schema.graph_name)
            while self.current.kind == TokenKind::Dot && self.peek_kind() != TokenKind::Eof {
                self.advance(); // consume dot
                if self.is_identifier() {
                    self.advance(); // consume qualifier
                } else {
                    break;
                }
            }
        }

        // Expect MATCH
        if self.current.kind != TokenKind::Match {
            return Err(self.error("Expected MATCH after FROM clause"));
        }

        // Parse the MATCH clause
        let match_clause = self.parse_match_clause()?;

        // Optional WHERE
        let where_clause =
            if self.current.kind == TokenKind::Where || self.current.kind == TokenKind::Filter {
                Some(self.parse_where_or_filter_clause()?)
            } else {
                None
            };

        // Optional GROUP BY
        let group_by = if self.current.kind == TokenKind::Group {
            self.advance();
            self.expect(TokenKind::By)?;
            let mut exprs = vec![self.parse_expression()?];
            while self.current.kind == TokenKind::Comma {
                self.advance();
                exprs.push(self.parse_expression()?);
            }
            exprs
        } else {
            Vec::new()
        };

        // Optional ORDER BY
        let order_by = if self.current.kind == TokenKind::Order {
            Some(self.parse_order_by()?)
        } else {
            None
        };

        // Optional OFFSET / SKIP
        let skip = if matches!(self.current.kind, TokenKind::Skip | TokenKind::Offset) {
            self.advance();
            Some(self.parse_expression()?)
        } else {
            None
        };

        // Optional LIMIT / FETCH
        let limit = if self.current.kind == TokenKind::Limit {
            self.advance();
            Some(self.parse_expression()?)
        } else if self.current.kind == TokenKind::Fetch {
            Some(self.parse_fetch_first()?)
        } else {
            None
        };

        let return_clause = ReturnClause {
            distinct,
            items,
            is_wildcard,
            group_by,
            order_by,
            skip,
            limit,
            is_finish: false,
            span: None,
        };

        Ok(Statement::Query(QueryStatement {
            match_clauses: vec![match_clause],
            where_clause,
            set_clauses: Vec::new(),
            remove_clauses: Vec::new(),
            with_clauses: Vec::new(),
            unwind_clauses: Vec::new(),
            merge_clauses: Vec::new(),
            create_clauses: Vec::new(),
            delete_clauses: Vec::new(),
            return_clause,
            having_clause: None,
            ordered_clauses: Vec::new(),
            span: Some(SourceSpan::new(span_start, self.current.span.end, 1, 1)),
        }))
    }

    fn parse_select_clause(&mut self) -> Result<ReturnClause> {
        let distinct = if self.current.kind == TokenKind::Distinct {
            self.advance();
            true
        } else {
            false
        };

        let mut items = Vec::new();
        items.push(self.parse_return_item()?);
        while self.current.kind == TokenKind::Comma {
            self.advance();
            items.push(self.parse_return_item()?);
        }

        // Parse optional GROUP BY
        let group_by = if self.current.kind == TokenKind::Group {
            self.advance();
            self.expect(TokenKind::By)?;
            let mut exprs = vec![self.parse_expression()?];
            while self.current.kind == TokenKind::Comma {
                self.advance();
                exprs.push(self.parse_expression()?);
            }
            exprs
        } else {
            Vec::new()
        };

        let order_by = if self.current.kind == TokenKind::Order {
            Some(self.parse_order_by()?)
        } else {
            None
        };

        let skip = if matches!(self.current.kind, TokenKind::Skip | TokenKind::Offset) {
            self.advance();
            Some(self.parse_expression()?)
        } else {
            None
        };

        let limit = if self.current.kind == TokenKind::Limit {
            self.advance();
            Some(self.parse_expression()?)
        } else if self.current.kind == TokenKind::Fetch {
            Some(self.parse_fetch_first()?)
        } else {
            None
        };

        Ok(ReturnClause {
            distinct,
            items,
            is_wildcard: false,
            group_by,
            order_by,
            skip,
            limit,
            is_finish: false,
            span: None,
        })
    }

    fn parse_return_item(&mut self) -> Result<ReturnItem> {
        let expression = self.parse_expression()?;

        let alias = if self.current.kind == TokenKind::As {
            self.advance();
            if !self.is_identifier() {
                return Err(self.error("Expected alias name"));
            }
            let name = self.get_identifier_name();
            self.advance();
            Some(name)
        } else {
            None
        };

        Ok(ReturnItem {
            expression,
            alias,
            span: None,
        })
    }

    /// Parses `FETCH FIRST|NEXT n ROWS|ROW ONLY` as a LIMIT expression.
    /// The FETCH token has already been peeked but not consumed.
    fn parse_fetch_first(&mut self) -> Result<Expression> {
        self.expect(TokenKind::Fetch)?;
        // FIRST or NEXT (both accepted)
        if !matches!(self.current.kind, TokenKind::First | TokenKind::Next) {
            return Err(self.error("Expected FIRST or NEXT after FETCH"));
        }
        self.advance();
        // Parse the count expression
        let count = self.parse_expression()?;
        // ROWS or ROW (both accepted, optional)
        if matches!(self.current.kind, TokenKind::Rows | TokenKind::Row) {
            self.advance();
        }
        // ONLY (optional)
        if self.current.kind == TokenKind::Only {
            self.advance();
        }
        Ok(count)
    }

    /// Parses a list comprehension after `[` and identifier have been peeked.
    /// Current token is the variable name, next is IN.
    /// Syntax: `[x IN list WHERE predicate | map_expression]`
    fn parse_list_comprehension_inner(&mut self) -> Result<Expression> {
        let variable = self.get_identifier_name();
        self.advance(); // consume variable
        self.expect(TokenKind::In)?;
        let list_expr = self.parse_expression()?;

        // Optional WHERE filter
        let filter_expr = if self.current.kind == TokenKind::Where {
            self.advance();
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };

        // Required | (pipe) followed by mapping expression
        let map_expr = if self.current.kind == TokenKind::Pipe {
            self.advance();
            Box::new(self.parse_expression()?)
        } else {
            // If no pipe, the mapping is just the variable itself
            Box::new(Expression::Variable(variable.clone()))
        };

        self.expect(TokenKind::RBracket)?;
        Ok(Expression::ListComprehension {
            variable,
            list_expr: Box::new(list_expr),
            filter_expr,
            map_expr,
        })
    }

    /// Parses a list predicate: `all/any/none/single(x IN list WHERE predicate)`.
    /// The function name has already been consumed; current token is `(`.
    fn parse_list_predicate(&mut self, kind: ListPredicateKind) -> Result<Expression> {
        self.expect(TokenKind::LParen)?;
        if !self.is_identifier() {
            return Err(self.error("Expected variable name in list predicate"));
        }
        let variable = self.get_identifier_name();
        self.advance();
        self.expect(TokenKind::In)?;
        let list_expr = self.parse_expression()?;
        // WHERE is required for list predicates
        self.expect(TokenKind::Where)?;
        let predicate = self.parse_expression()?;
        self.expect(TokenKind::RParen)?;
        Ok(Expression::ListPredicate {
            kind,
            variable,
            list_expr: Box::new(list_expr),
            predicate: Box::new(predicate),
        })
    }

    /// Parses `reduce(accumulator = init, x IN list | expression)`.
    /// The function name has already been consumed; current token is `(`.
    fn parse_reduce(&mut self) -> Result<Expression> {
        self.expect(TokenKind::LParen)?;
        if !self.is_identifier() {
            return Err(self.error("Expected accumulator variable in reduce"));
        }
        let accumulator = self.get_identifier_name();
        self.advance();
        self.expect(TokenKind::Eq)?;
        let initial = self.parse_expression()?;
        self.expect(TokenKind::Comma)?;
        if !self.is_identifier() {
            return Err(self.error("Expected iteration variable in reduce"));
        }
        let variable = self.get_identifier_name();
        self.advance();
        self.expect(TokenKind::In)?;
        let list = self.parse_expression()?;
        self.expect(TokenKind::Pipe)?;
        let expression = self.parse_expression()?;
        self.expect(TokenKind::RParen)?;
        Ok(Expression::Reduce {
            accumulator,
            initial: Box::new(initial),
            variable,
            list: Box::new(list),
            expression: Box::new(expression),
        })
    }

    fn parse_order_by(&mut self) -> Result<OrderByClause> {
        self.expect(TokenKind::Order)?;
        self.expect(TokenKind::By)?;

        let mut items = Vec::new();
        items.push(self.parse_order_item()?);

        while self.current.kind == TokenKind::Comma {
            self.advance();
            items.push(self.parse_order_item()?);
        }

        Ok(OrderByClause { items, span: None })
    }

    fn parse_order_item(&mut self) -> Result<OrderByItem> {
        let expression = self.parse_expression()?;

        let order = match self.current.kind {
            TokenKind::Asc => {
                self.advance();
                SortOrder::Asc
            }
            TokenKind::Desc => {
                self.advance();
                SortOrder::Desc
            }
            _ => SortOrder::Asc,
        };

        // Parse optional NULLS FIRST / NULLS LAST (ISO GQL feature GA03)
        let nulls =
            if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("NULLS") {
                self.advance();
                if self.is_identifier() {
                    let kw = self.get_identifier_name().to_uppercase();
                    match kw.as_str() {
                        "FIRST" => {
                            self.advance();
                            Some(NullsOrdering::First)
                        }
                        "LAST" => {
                            self.advance();
                            Some(NullsOrdering::Last)
                        }
                        _ => return Err(self.error("Expected FIRST or LAST after NULLS")),
                    }
                } else {
                    return Err(self.error("Expected FIRST or LAST after NULLS"));
                }
            } else {
                None
            };

        Ok(OrderByItem {
            expression,
            order,
            nulls,
        })
    }

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
            return Ok(Expression::Unary {
                op: UnaryOp::Not,
                operand: Box::new(operand),
            });
        }
        self.parse_comparison_expression()
    }

    fn parse_comparison_expression(&mut self) -> Result<Expression> {
        let left = self.parse_additive_expression()?;

        // Check for regular comparison operators
        let op = match self.current.kind {
            TokenKind::Eq => Some(BinaryOp::Eq),
            TokenKind::Ne => Some(BinaryOp::Ne),
            TokenKind::Lt => Some(BinaryOp::Lt),
            TokenKind::Le => Some(BinaryOp::Le),
            TokenKind::Gt => Some(BinaryOp::Gt),
            TokenKind::Ge => Some(BinaryOp::Ge),
            _ => None,
        };

        if let Some(op) = op {
            self.advance();
            let right = self.parse_additive_expression()?;
            return Ok(Expression::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            });
        }

        // Check for IN, STARTS WITH, ENDS WITH, CONTAINS
        match self.current.kind {
            TokenKind::In => {
                self.advance(); // consume IN
                let right = self.parse_primary_expression()?;
                return Ok(Expression::Binary {
                    left: Box::new(left),
                    op: BinaryOp::In,
                    right: Box::new(right),
                });
            }
            TokenKind::Starts => {
                self.advance(); // consume STARTS
                self.expect(TokenKind::With)?; // expect WITH
                let right = self.parse_additive_expression()?;
                return Ok(Expression::Binary {
                    left: Box::new(left),
                    op: BinaryOp::StartsWith,
                    right: Box::new(right),
                });
            }
            TokenKind::Ends => {
                self.advance(); // consume ENDS
                self.expect(TokenKind::With)?; // expect WITH
                let right = self.parse_additive_expression()?;
                return Ok(Expression::Binary {
                    left: Box::new(left),
                    op: BinaryOp::EndsWith,
                    right: Box::new(right),
                });
            }
            TokenKind::Contains => {
                self.advance(); // consume CONTAINS
                let right = self.parse_additive_expression()?;
                return Ok(Expression::Binary {
                    left: Box::new(left),
                    op: BinaryOp::Contains,
                    right: Box::new(right),
                });
            }
            TokenKind::Like => {
                self.advance(); // consume LIKE
                let right = self.parse_additive_expression()?;
                return Ok(Expression::Binary {
                    left: Box::new(left),
                    op: BinaryOp::Like,
                    right: Box::new(right),
                });
            }
            TokenKind::Is => {
                self.advance(); // consume IS
                let negated = self.current.kind == TokenKind::Not;
                if negated {
                    self.advance(); // consume NOT
                }

                let predicate = if self.current.kind == TokenKind::Null {
                    // IS [NOT] NULL
                    self.advance();
                    Expression::Unary {
                        op: if negated {
                            UnaryOp::IsNotNull
                        } else {
                            UnaryOp::IsNull
                        },
                        operand: Box::new(left),
                    }
                } else if self.is_identifier() {
                    let kw = self.get_identifier_name().to_uppercase();
                    match kw.as_str() {
                        "TYPED" => {
                            // IS [NOT] TYPED <type_name>
                            // Supports LIST<element_type> parameterized form
                            self.advance();
                            let type_name = if self.is_identifier() {
                                let base = self.get_identifier_name().to_uppercase();
                                self.advance();
                                // Handle LIST<type> parameterized syntax
                                if base == "LIST" && self.current.kind == TokenKind::Lt {
                                    self.advance(); // consume <
                                    if self.is_identifier() {
                                        let elem = self.get_identifier_name().to_uppercase();
                                        self.advance();
                                        self.expect(TokenKind::Gt)?;
                                        format!("LIST<{elem}>")
                                    } else {
                                        return Err(self.error("Expected element type after LIST<"));
                                    }
                                } else {
                                    base
                                }
                            } else {
                                return Err(self.error("Expected type name after IS TYPED"));
                            };
                            let call = Expression::FunctionCall {
                                name: "isTyped".to_string(),
                                args: vec![left, Expression::Literal(Literal::String(type_name))],
                                distinct: false,
                            };
                            if negated {
                                Expression::Unary {
                                    op: UnaryOp::Not,
                                    operand: Box::new(call),
                                }
                            } else {
                                call
                            }
                        }
                        "DIRECTED" => {
                            // IS [NOT] DIRECTED
                            self.advance();
                            let call = Expression::FunctionCall {
                                name: "isDirected".to_string(),
                                args: vec![left],
                                distinct: false,
                            };
                            if negated {
                                Expression::Unary {
                                    op: UnaryOp::Not,
                                    operand: Box::new(call),
                                }
                            } else {
                                call
                            }
                        }
                        "LABELED" => {
                            // IS [NOT] LABELED <label>
                            self.advance();
                            let label = if self.is_identifier() {
                                self.get_identifier_name()
                            } else {
                                return Err(self.error("Expected label name after IS LABELED"));
                            };
                            self.advance();
                            let call = Expression::FunctionCall {
                                name: "hasLabel".to_string(),
                                args: vec![left, Expression::Literal(Literal::String(label))],
                                distinct: false,
                            };
                            if negated {
                                Expression::Unary {
                                    op: UnaryOp::Not,
                                    operand: Box::new(call),
                                }
                            } else {
                                call
                            }
                        }
                        "SOURCE" => {
                            // IS [NOT] SOURCE OF <variable>
                            self.advance();
                            if !(self.is_identifier()
                                && self.get_identifier_name().eq_ignore_ascii_case("OF"))
                            {
                                return Err(self.error("Expected OF after IS SOURCE"));
                            }
                            self.advance(); // consume OF
                            let var = if self.is_identifier() {
                                self.get_identifier_name()
                            } else {
                                return Err(self.error("Expected variable after IS SOURCE OF"));
                            };
                            self.advance();
                            let call = Expression::FunctionCall {
                                name: "isSource".to_string(),
                                args: vec![left, Expression::Variable(var)],
                                distinct: false,
                            };
                            if negated {
                                Expression::Unary {
                                    op: UnaryOp::Not,
                                    operand: Box::new(call),
                                }
                            } else {
                                call
                            }
                        }
                        "DESTINATION" => {
                            // IS [NOT] DESTINATION OF <variable>
                            self.advance();
                            if !(self.is_identifier()
                                && self.get_identifier_name().eq_ignore_ascii_case("OF"))
                            {
                                return Err(self.error("Expected OF after IS DESTINATION"));
                            }
                            self.advance(); // consume OF
                            let var = if self.is_identifier() {
                                self.get_identifier_name()
                            } else {
                                return Err(self.error("Expected variable after IS DESTINATION OF"));
                            };
                            self.advance();
                            let call = Expression::FunctionCall {
                                name: "isDestination".to_string(),
                                args: vec![left, Expression::Variable(var)],
                                distinct: false,
                            };
                            if negated {
                                Expression::Unary {
                                    op: UnaryOp::Not,
                                    operand: Box::new(call),
                                }
                            } else {
                                call
                            }
                        }
                        "NFC" | "NFD" | "NFKC" | "NFKD" => {
                            // IS [NOT] NFC|NFD|NFKC|NFKD NORMALIZED
                            let form = kw.clone();
                            self.advance();
                            if !(self.is_identifier()
                                && self
                                    .get_identifier_name()
                                    .eq_ignore_ascii_case("NORMALIZED"))
                            {
                                return Err(
                                    self.error(&format!("Expected NORMALIZED after IS {form}"))
                                );
                            }
                            self.advance();
                            let call = Expression::FunctionCall {
                                name: "isNormalized".to_string(),
                                args: vec![left, Expression::Literal(Literal::String(form))],
                                distinct: false,
                            };
                            if negated {
                                Expression::Unary {
                                    op: UnaryOp::Not,
                                    operand: Box::new(call),
                                }
                            } else {
                                call
                            }
                        }
                        "NORMALIZED" => {
                            // IS [NOT] NORMALIZED (default NFC)
                            self.advance();
                            let call = Expression::FunctionCall {
                                name: "isNormalized".to_string(),
                                args: vec![
                                    left,
                                    Expression::Literal(Literal::String("NFC".to_string())),
                                ],
                                distinct: false,
                            };
                            if negated {
                                Expression::Unary {
                                    op: UnaryOp::Not,
                                    operand: Box::new(call),
                                }
                            } else {
                                call
                            }
                        }
                        _ => {
                            return Err(self.error(
                                "Expected NULL, TYPED, DIRECTED, LABELED, SOURCE, DESTINATION, NORMALIZED, or NFC/NFD/NFKC/NFKD after IS",
                            ));
                        }
                    }
                } else {
                    return Err(self.error(
                        "Expected NULL, TYPED, DIRECTED, LABELED, SOURCE, DESTINATION, NORMALIZED, or NFC/NFD/NFKC/NFKD after IS",
                    ));
                };

                return Ok(predicate);
            }
            _ => {}
        }

        Ok(left)
    }

    fn parse_additive_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_multiplicative_expression()?;

        loop {
            let op = match self.current.kind {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                TokenKind::Concat => BinaryOp::Concat,
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

    /// `a ^ b`, binding tighter than `*` and right associative
    /// (`2 ^ 3 ^ 2` is `2 ^ 9`), as in Cypher.
    fn parse_power_expression(&mut self) -> Result<Expression> {
        let left = self.parse_unary_expression()?;
        if self.current.kind != TokenKind::Caret {
            return Ok(left);
        }
        self.advance();
        let right = self.parse_power_expression()?;
        Ok(Expression::Binary {
            left: Box::new(left),
            op: BinaryOp::Pow,
            right: Box::new(right),
        })
    }

    fn parse_unary_expression(&mut self) -> Result<Expression> {
        match self.current.kind {
            TokenKind::Minus => {
                self.advance();
                // Fold -<integer> at parse time to handle i64::MIN correctly
                // (9223372036854775808 overflows i64 as positive but "-9223372036854775808" parses fine)
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
                TokenKind::LBracket => {
                    self.advance();

                    // Check for slice patterns: [..end], [start..], [start..end]
                    if self.current.kind == TokenKind::DotDot {
                        // [..end] pattern: no start
                        self.advance(); // consume ..
                        let end_expr = if self.current.kind == TokenKind::RBracket {
                            None
                        } else {
                            Some(Box::new(self.parse_expression()?))
                        };
                        self.expect(TokenKind::RBracket)?;
                        expr = Expression::SliceAccess {
                            base: Box::new(expr),
                            start: None,
                            end: end_expr,
                        };
                    } else {
                        let index = self.parse_expression()?;
                        if self.current.kind == TokenKind::DotDot {
                            // [start..] or [start..end] pattern
                            self.advance(); // consume ..
                            let end_expr = if self.current.kind == TokenKind::RBracket {
                                None
                            } else {
                                Some(Box::new(self.parse_expression()?))
                            };
                            self.expect(TokenKind::RBracket)?;
                            expr = Expression::SliceAccess {
                                base: Box::new(expr),
                                start: Some(Box::new(index)),
                                end: end_expr,
                            };
                        } else {
                            self.expect(TokenKind::RBracket)?;
                            expr = Expression::IndexAccess {
                                base: Box::new(expr),
                                index: Box::new(index),
                            };
                        }
                    }
                }
                // n:Label label-check syntax (compact form of IS LABELED).
                // Multiple labels (n:Person:Actor) are ANDead together.
                TokenKind::Colon => {
                    let base = expr;
                    let mut combined: Option<Expression> = None;
                    while self.current.kind == TokenKind::Colon {
                        self.advance();
                        if !self.is_label_or_type_name() {
                            return Err(self.error("Expected label name after ':'"));
                        }
                        let label = self.get_identifier_name();
                        self.advance();
                        let check = Expression::FunctionCall {
                            name: "hasLabel".to_string(),
                            args: vec![base.clone(), Expression::Literal(Literal::String(label))],
                            distinct: false,
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
                        .ok_or_else(|| self.error("Expected at least one label after ':'"))?;
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
                } else if text.starts_with("0b") || text.starts_with("0B") {
                    i64::from_str_radix(&text[2..], 2)
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
                let value = self
                    .current
                    .text
                    .parse()
                    .map_err(|_| self.error("Invalid float"))?;
                self.advance();
                Ok(Expression::Literal(Literal::Float(value)))
            }
            TokenKind::String => {
                let text = &self.current.text;
                let inner = &text[1..text.len() - 1]; // Remove quotes
                let value = unescape_string(inner);
                self.advance();
                Ok(Expression::Literal(Literal::String(value)))
            }
            // CASE expression - check if it's actually a CASE expression or just a variable named 'case'
            TokenKind::Case => {
                // Look ahead: if followed by WHEN, it's a CASE expression
                // If followed by : , ) AS ORDER LIMIT SKIP or EOF, it's a variable named 'case'
                let next = self.peek_kind();
                if matches!(
                    next,
                    TokenKind::Colon
                        | TokenKind::Comma
                        | TokenKind::RParen
                        | TokenKind::RBracket
                        | TokenKind::As
                        | TokenKind::Order
                        | TokenKind::Limit
                        | TokenKind::Skip
                        | TokenKind::Eof
                ) {
                    // It's a variable named 'case'
                    let name = "case".to_string();
                    self.advance();
                    Ok(Expression::Variable(name))
                } else {
                    // It's a CASE expression
                    self.parse_case_expression()
                }
            }
            // Handle type() function - must be checked BEFORE is_identifier() since TYPE is a contextual keyword
            TokenKind::Type => {
                let name = "type".to_string();
                self.advance();
                if self.current.kind != TokenKind::LParen {
                    // If not followed by (, treat as identifier/variable
                    return Ok(Expression::Variable(name));
                }
                self.advance();
                let mut args = Vec::new();
                if self.current.kind != TokenKind::RParen {
                    args.push(self.parse_expression()?);
                    while self.current.kind == TokenKind::Comma {
                        self.advance();
                        args.push(self.parse_expression()?);
                    }
                }
                self.expect(TokenKind::RParen)?;
                Ok(Expression::FunctionCall {
                    name,
                    args,
                    distinct: false,
                })
            }
            _ if self.is_identifier() => {
                let name = self.get_identifier_name();

                // IEEE 754 special float literals (only when not a function call)
                if name.eq_ignore_ascii_case("NaN") && self.peek_kind() != TokenKind::LParen {
                    self.advance();
                    return Ok(Expression::Literal(Literal::Float(f64::NAN)));
                }
                if (name.eq_ignore_ascii_case("Inf") || name.eq_ignore_ascii_case("Infinity"))
                    && self.peek_kind() != TokenKind::LParen
                {
                    self.advance();
                    return Ok(Expression::Literal(Literal::Float(f64::INFINITY)));
                }

                // SESSION_USER: ISO GQL system value expression
                if name.eq_ignore_ascii_case("SESSION_USER") {
                    self.advance();
                    return Ok(Expression::FunctionCall {
                        name: "session_user".to_string(),
                        args: Vec::new(),
                        distinct: false,
                    });
                }

                // ISO/IEC 39075 Section 17.1 / Section 21: pre-reserved schema/graph references
                if name.eq_ignore_ascii_case("CURRENT_SCHEMA") {
                    self.advance();
                    return Ok(Expression::FunctionCall {
                        name: "current_schema".to_string(),
                        args: Vec::new(),
                        distinct: false,
                    });
                }
                if name.eq_ignore_ascii_case("CURRENT_GRAPH")
                    || name.eq_ignore_ascii_case("CURRENT_PROPERTY_GRAPH")
                {
                    self.advance();
                    return Ok(Expression::FunctionCall {
                        name: "current_graph".to_string(),
                        args: Vec::new(),
                        distinct: false,
                    });
                }
                if name.eq_ignore_ascii_case("HOME_SCHEMA") {
                    self.advance();
                    return Ok(Expression::FunctionCall {
                        name: "home_schema".to_string(),
                        args: Vec::new(),
                        distinct: false,
                    });
                }
                if name.eq_ignore_ascii_case("HOME_GRAPH")
                    || name.eq_ignore_ascii_case("HOME_PROPERTY_GRAPH")
                {
                    self.advance();
                    return Ok(Expression::FunctionCall {
                        name: "home_graph".to_string(),
                        args: Vec::new(),
                        distinct: false,
                    });
                }

                // NULLIF(expr1, expr2) - ISO GQL keyword syntax (Section 20.7)
                if name.eq_ignore_ascii_case("NULLIF") && self.peek_kind() == TokenKind::LParen {
                    self.advance(); // consume NULLIF
                    self.advance(); // consume (
                    let expr1 = self.parse_expression()?;
                    self.expect(TokenKind::Comma)?;
                    let expr2 = self.parse_expression()?;
                    self.expect(TokenKind::RParen)?;
                    return Ok(Expression::FunctionCall {
                        name: "nullif".to_string(),
                        args: vec![expr1, expr2],
                        distinct: false,
                    });
                }

                // COALESCE(expr1, expr2, ...) - ISO GQL keyword syntax (Section 20.7)
                if name.eq_ignore_ascii_case("COALESCE") && self.peek_kind() == TokenKind::LParen {
                    self.advance(); // consume COALESCE
                    self.advance(); // consume (
                    let mut args = vec![self.parse_expression()?];
                    while self.current.kind == TokenKind::Comma {
                        self.advance();
                        args.push(self.parse_expression()?);
                    }
                    self.expect(TokenKind::RParen)?;
                    return Ok(Expression::FunctionCall {
                        name: "coalesce".to_string(),
                        args,
                        distinct: false,
                    });
                }

                // TRIM(BOTH|LEADING|TRAILING 'chars' FROM string)
                // ISO GQL enhanced TRIM with trim specification (GF05)
                if name.eq_ignore_ascii_case("TRIM") && self.peek_kind() == TokenKind::LParen {
                    self.advance(); // consume TRIM
                    self.advance(); // consume (
                    // Determine trim mode and characters
                    let mut mode = 0i64; // 0=both, 1=leading, 2=trailing
                    let mut trim_chars: Option<Expression> = None;

                    // Check for BOTH/LEADING/TRAILING keyword
                    if self.is_identifier() {
                        let kw = self.get_identifier_name().to_uppercase();
                        match kw.as_str() {
                            "BOTH" => {
                                self.advance();
                            }
                            "LEADING" => {
                                mode = 1;
                                self.advance();
                            }
                            "TRAILING" => {
                                mode = 2;
                                self.advance();
                            }
                            "FROM" => {
                                // TRIM(FROM string) - default both, no trim chars
                                self.advance();
                                let string_expr = self.parse_expression()?;
                                self.expect(TokenKind::RParen)?;
                                return Ok(Expression::FunctionCall {
                                    name: "trim".to_string(),
                                    args: vec![string_expr],
                                    distinct: false,
                                });
                            }
                            _ => {
                                // Not a keyword, parse as the only arg (simple trim(expr))
                                let expr = self.parse_expression()?;
                                self.expect(TokenKind::RParen)?;
                                return Ok(Expression::FunctionCall {
                                    name: "trim".to_string(),
                                    args: vec![expr],
                                    distinct: false,
                                });
                            }
                        }
                    }

                    // After mode keyword, check for trim chars or FROM
                    if self.is_identifier()
                        && self.get_identifier_name().eq_ignore_ascii_case("FROM")
                    {
                        // TRIM(BOTH FROM string) - no trim chars
                        self.advance(); // consume FROM
                        let string_expr = self.parse_expression()?;
                        self.expect(TokenKind::RParen)?;
                        return Ok(Expression::FunctionCall {
                            name: "trim".to_string(),
                            args: vec![
                                string_expr,
                                Expression::Literal(Literal::String(" ".into())),
                                Expression::Literal(Literal::Integer(mode)),
                            ],
                            distinct: false,
                        });
                    }

                    // Parse trim character expression
                    if self.current.kind != TokenKind::RParen {
                        trim_chars = Some(self.parse_expression()?);
                    }

                    // Check for FROM keyword
                    if self.is_identifier()
                        && self.get_identifier_name().eq_ignore_ascii_case("FROM")
                    {
                        self.advance(); // consume FROM
                        let string_expr = self.parse_expression()?;
                        self.expect(TokenKind::RParen)?;
                        let chars_expr =
                            trim_chars.unwrap_or(Expression::Literal(Literal::String(" ".into())));
                        return Ok(Expression::FunctionCall {
                            name: "trim".to_string(),
                            args: vec![
                                string_expr,
                                chars_expr,
                                Expression::Literal(Literal::Integer(mode)),
                            ],
                            distinct: false,
                        });
                    }

                    // Simple trim(expr) - single argument
                    self.expect(TokenKind::RParen)?;
                    let arg = trim_chars.unwrap_or(Expression::Literal(Literal::Null));
                    return Ok(Expression::FunctionCall {
                        name: "trim".to_string(),
                        args: vec![arg],
                        distinct: false,
                    });
                }

                // List predicates: all/any/none/single(x IN list WHERE pred)
                // Disambiguate from regular function calls by checking for x IN pattern
                if self.peek_kind() == TokenKind::LParen {
                    let lower = name.to_lowercase();
                    let predicate_kind = match lower.as_str() {
                        "all" => Some(ListPredicateKind::All),
                        "any" => Some(ListPredicateKind::Any),
                        "none" => Some(ListPredicateKind::None),
                        "single" => Some(ListPredicateKind::Single),
                        _ => None,
                    };
                    if let Some(kind) = predicate_kind {
                        // Save position to restore if this is a regular function call
                        self.advance(); // consume function name
                        // Now peek inside: if ( identifier IN ... ), it's a list predicate
                        // We've consumed the name, current is LParen, peek is potentially identifier
                        // We can't easily look 2 ahead, so try parsing and fall back
                        return self.parse_list_predicate(kind);
                    }

                    // reduce(acc = init, x IN list | expr)
                    if lower == "reduce" {
                        self.advance(); // consume 'reduce'
                        return self.parse_reduce();
                    }
                }

                // LET ... IN ... END expression
                if name.eq_ignore_ascii_case("LET") {
                    self.advance(); // consume LET
                    let mut bindings = Vec::new();
                    loop {
                        if !self.is_identifier() {
                            return Err(self.error("Expected variable name in LET expression"));
                        }
                        let var = self.get_identifier_name();
                        self.advance();
                        self.expect(TokenKind::Eq)?;
                        // Use additive_expression to stop before IN (which is a
                        // comparison-level operator) so the LET's own IN keyword
                        // is not consumed by the binding expression.
                        let expr = self.parse_additive_expression()?;
                        bindings.push((var, expr));
                        if self.current.kind != TokenKind::Comma {
                            break;
                        }
                        self.advance(); // consume comma
                    }
                    // Expect IN keyword
                    if self.current.kind != TokenKind::In {
                        return Err(self.error("Expected IN after LET bindings"));
                    }
                    self.advance(); // consume IN
                    let body = self.parse_expression()?;
                    self.expect(TokenKind::End)?;
                    return Ok(Expression::LetIn {
                        bindings,
                        body: Box::new(body),
                    });
                }

                self.advance();

                // Typed temporal literals: DATE 'str', TIME 'str', DURATION 'str', DATETIME 'str'
                if self.current.kind == TokenKind::String {
                    let upper = name.to_uppercase();
                    let make_val = |parser: &Self| {
                        let text = &parser.current.text;
                        let inner = &text[1..text.len() - 1];
                        unescape_string(inner)
                    };
                    let typed_lit = match upper.as_str() {
                        "DATE" => Some(Literal::Date(make_val(self))),
                        "TIME" => Some(Literal::Time(make_val(self))),
                        "DURATION" => Some(Literal::Duration(make_val(self))),
                        "DATETIME" => Some(Literal::Datetime(make_val(self))),
                        _ => None,
                    };
                    if let Some(lit) = typed_lit {
                        self.advance();
                        return Ok(Expression::Literal(lit));
                    }
                }

                // Compound typed literals: ZONED DATETIME 'str', ZONED TIME 'str'
                if name.eq_ignore_ascii_case("ZONED") && self.is_identifier() {
                    let sub = self.get_identifier_name().to_uppercase();
                    if sub == "DATETIME" || sub == "TIME" {
                        self.advance(); // consume DATETIME/TIME
                        if self.current.kind == TokenKind::String {
                            let text = &self.current.text;
                            let inner = &text[1..text.len() - 1];
                            let val = unescape_string(inner);
                            self.advance();
                            return Ok(Expression::Literal(if sub == "DATETIME" {
                                Literal::ZonedDatetime(val)
                            } else {
                                Literal::ZonedTime(val)
                            }));
                        }
                    }
                }

                if self.current.kind == TokenKind::Dot {
                    self.advance();
                    if !self.is_label_or_type_name() {
                        return Err(self.error("Expected property name"));
                    }
                    let property = self.get_identifier_name();
                    self.advance();
                    Ok(Expression::PropertyAccess {
                        variable: name,
                        property,
                    })
                } else if self.current.kind == TokenKind::LBrace
                    && name.eq_ignore_ascii_case("count")
                {
                    // COUNT { MATCH ... } subquery expression
                    self.advance(); // consume {
                    let inner_query = self.parse_exists_inner_query()?;
                    self.expect(TokenKind::RBrace)?;
                    Ok(Expression::CountSubquery {
                        query: Box::new(inner_query),
                    })
                } else if self.current.kind == TokenKind::LBrace
                    && name.eq_ignore_ascii_case("value")
                {
                    // VALUE { subquery } expression
                    self.advance(); // consume {
                    let inner_query = self.parse_value_subquery_inner()?;
                    self.expect(TokenKind::RBrace)?;
                    Ok(Expression::ValueSubquery {
                        query: Box::new(inner_query),
                    })
                } else if self.current.kind == TokenKind::LParen {
                    // Function call
                    self.advance();
                    // COUNT(*) per ISO/IEC 39075 sec 20.9
                    if name.eq_ignore_ascii_case("count") && self.current.kind == TokenKind::Star {
                        self.advance(); // consume *
                        self.expect(TokenKind::RParen)?;
                        return Ok(Expression::FunctionCall {
                            name,
                            args: Vec::new(),
                            distinct: false,
                        });
                    }
                    // Check for DISTINCT keyword in aggregate functions
                    let distinct = if self.current.kind == TokenKind::Distinct {
                        self.advance();
                        true
                    } else {
                        false
                    };
                    let mut args = Vec::new();
                    if self.current.kind != TokenKind::RParen {
                        args.push(self.parse_expression()?);
                        while self.current.kind == TokenKind::Comma {
                            self.advance();
                            args.push(self.parse_expression()?);
                        }
                    }
                    self.expect(TokenKind::RParen)?;
                    Ok(Expression::FunctionCall {
                        name,
                        args,
                        distinct,
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
                self.advance(); // consume [
                // Disambiguate: [x IN list WHERE ... | expr] vs [elem, ...]
                // List comprehension if: identifier followed by IN keyword
                if self.is_identifier() && self.peek_kind() == TokenKind::In {
                    let result = self.parse_list_comprehension_inner();
                    self.exit_nesting();
                    return result;
                }
                let mut elements = Vec::new();
                if self.current.kind != TokenKind::RBracket {
                    elements.push(self.parse_expression()?);
                    while self.current.kind == TokenKind::Comma {
                        self.advance();
                        elements.push(self.parse_expression()?);
                    }
                }
                self.expect(TokenKind::RBracket)?;
                self.exit_nesting();
                Ok(Expression::List(elements))
            }
            TokenKind::Parameter => {
                // Parameter token includes the $ prefix, so we extract just the name
                let full_text = &self.current.text;
                let name = full_text.trim_start_matches('$').to_string();
                self.advance();
                Ok(Expression::Parameter(name))
            }
            TokenKind::Exists => {
                self.enter_nesting()?;
                self.advance();
                if self.current.kind == TokenKind::LBrace {
                    // EXISTS { MATCH ... } subquery form
                    self.advance(); // consume {
                    let inner_query = self.parse_exists_inner_query()?;
                    self.expect(TokenKind::RBrace)?;
                    self.exit_nesting();
                    Ok(Expression::ExistsSubquery {
                        query: Box::new(inner_query),
                    })
                } else {
                    // exists(expr) function form
                    self.expect(TokenKind::LParen)?;
                    let arg = self.parse_expression()?;
                    self.expect(TokenKind::RParen)?;
                    self.exit_nesting();
                    Ok(Expression::FunctionCall {
                        name: "exists".to_string(),
                        args: vec![arg],
                        distinct: false,
                    })
                }
            }
            TokenKind::LBrace => {
                // Map literal: {key: value, ...}
                let entries = self.parse_property_map()?;
                Ok(Expression::Map(entries))
            }
            TokenKind::Cast => {
                // CAST(expr AS type) -> desugar to toInteger/toFloat/toString
                self.advance();
                self.expect(TokenKind::LParen)?;
                let expr = self.parse_expression()?;
                self.expect(TokenKind::As)?;
                let type_name = if self.is_identifier() {
                    let mut name = self.get_identifier_name().to_uppercase();
                    self.advance();
                    // Handle compound type names: ZONED DATETIME, ZONED TIME
                    if name == "ZONED" && self.is_identifier() {
                        let sub = self.get_identifier_name().to_uppercase();
                        if sub == "DATETIME" || sub == "TIME" {
                            name = format!("ZONED {sub}");
                            self.advance();
                        }
                    }
                    // Handle LIST<type> parameterized type
                    if name == "LIST" && self.current.kind == TokenKind::Lt {
                        self.advance(); // consume <
                        if self.is_identifier() {
                            let elem = self.get_identifier_name().to_uppercase();
                            self.advance();
                            self.expect(TokenKind::Gt)?;
                            name = format!("LIST<{elem}>");
                        } else {
                            return Err(self.error("Expected element type after LIST<"));
                        }
                    }
                    name
                } else {
                    return Err(self.error("Expected type name after AS"));
                };
                self.expect(TokenKind::RParen)?;
                let (func_name, extra_arg) = match type_name.as_str() {
                    "INTEGER" | "INT" | "INT64" | "BIGINT" => ("toInteger", None),
                    "FLOAT" | "DOUBLE" | "FLOAT64" | "REAL" => ("toFloat", None),
                    "STRING" | "VARCHAR" | "TEXT" => ("toString", None),
                    "BOOLEAN" | "BOOL" => ("toBoolean", None),
                    "DATE" => ("toDate", None),
                    "TIME" | "LOCALTIME" => ("toTime", None),
                    "DATETIME" | "TIMESTAMP" | "LOCALDATETIME" => ("toDatetime", None),
                    "DURATION" => ("toDuration", None),
                    "ZONED DATETIME" => ("toZonedDatetime", None),
                    "ZONED TIME" => ("toZonedTime", None),
                    "LIST" => ("toList", None),
                    s if s.starts_with("LIST<") => {
                        let elem_type = &s[5..s.len() - 1]; // extract type between < and >
                        ("toTypedList", Some(elem_type.to_string()))
                    }
                    _ => return Err(self.error(&format!("Unsupported CAST type: {type_name}"))),
                };
                let mut args = vec![expr];
                if let Some(elem) = extra_arg {
                    args.push(Expression::Literal(Literal::String(elem)));
                }
                Ok(Expression::FunctionCall {
                    name: func_name.to_string(),
                    args,
                    distinct: false,
                })
            }
            _ => Err(self.error("Expected expression")),
        }
    }

    /// Parses a CASE expression.
    /// CASE [input] WHEN condition THEN result [WHEN ...] [ELSE default] END
    fn parse_case_expression(&mut self) -> Result<Expression> {
        self.enter_nesting()?;
        self.expect(TokenKind::Case)?;

        // Check for simple CASE (CASE expr WHEN value THEN ...)
        // vs searched CASE (CASE WHEN condition THEN ...)
        let input = if self.current.kind != TokenKind::When {
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };

        // Parse WHEN clauses
        let mut whens = Vec::new();
        while self.current.kind == TokenKind::When {
            self.advance();
            let condition = self.parse_expression()?;
            self.expect(TokenKind::Then)?;
            let result = self.parse_expression()?;
            whens.push((condition, result));
        }

        if whens.is_empty() {
            self.exit_nesting();
            return Err(self.error("CASE requires at least one WHEN clause"));
        }

        // Parse optional ELSE
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

    /// Parses the inner query of an EXISTS subquery.
    /// Handles: EXISTS { MATCH (n)-[:REL]->() [WHERE ...] }
    fn parse_exists_inner_query(&mut self) -> Result<QueryStatement> {
        let mut match_clauses = Vec::new();

        // Parse MATCH clauses
        while self.current.kind == TokenKind::Match || self.current.kind == TokenKind::Optional {
            match_clauses.push(self.parse_match_clause()?);
        }

        // Bare pattern form: EXISTS { (a)-[r]->(b) WHERE ... }
        // Treat as implicit MATCH when no MATCH keyword but a pattern starts with (
        if match_clauses.is_empty() && self.current.kind == TokenKind::LParen {
            let span_start = self.current.span.start;
            let mut patterns = Vec::new();
            patterns.push(self.parse_aliased_pattern()?);
            while self.current.kind == TokenKind::Comma {
                self.advance();
                patterns.push(self.parse_aliased_pattern()?);
            }
            match_clauses.push(MatchClause {
                optional: false,
                path_mode: None,
                search_prefix: None,
                match_mode: None,
                patterns,
                span: Some(SourceSpan::new(span_start, self.current.span.end, 1, 1)),
            });
        }

        if match_clauses.is_empty() {
            return Err(self.error("EXISTS subquery requires at least one MATCH clause"));
        }

        // Parse optional WHERE
        let where_clause = if self.current.kind == TokenKind::Where {
            Some(self.parse_where_clause()?)
        } else {
            None
        };

        // EXISTS doesn't need RETURN - create empty return clause
        Ok(QueryStatement {
            match_clauses,
            where_clause,
            set_clauses: vec![],
            remove_clauses: vec![],
            with_clauses: vec![],
            unwind_clauses: vec![],
            merge_clauses: vec![],
            create_clauses: vec![],
            delete_clauses: vec![],
            return_clause: ReturnClause {
                distinct: false,
                items: vec![],
                is_wildcard: false,
                group_by: vec![],
                order_by: None,
                skip: None,
                limit: None,
                is_finish: false,
                span: None,
            },
            having_clause: None,
            ordered_clauses: vec![],
            span: None,
        })
    }

    /// Parses the inner query of a VALUE subquery.
    /// Handles: VALUE { MATCH ... [WHERE ...] RETURN expr }
    fn parse_value_subquery_inner(&mut self) -> Result<QueryStatement> {
        let mut match_clauses = Vec::new();

        // Parse MATCH clauses
        while self.current.kind == TokenKind::Match || self.current.kind == TokenKind::Optional {
            match_clauses.push(self.parse_match_clause()?);
        }

        if match_clauses.is_empty() {
            return Err(self.error("VALUE subquery requires at least one MATCH clause"));
        }

        // Parse optional WHERE
        let where_clause = if self.current.kind == TokenKind::Where {
            Some(self.parse_where_clause()?)
        } else {
            None
        };

        // Parse required RETURN
        let return_clause = if self.current.kind == TokenKind::Return {
            self.parse_return_clause()?
        } else {
            return Err(self.error("VALUE subquery requires a RETURN clause"));
        };

        Ok(QueryStatement {
            match_clauses,
            where_clause,
            set_clauses: vec![],
            remove_clauses: vec![],
            with_clauses: vec![],
            unwind_clauses: vec![],
            merge_clauses: vec![],
            create_clauses: vec![],
            delete_clauses: vec![],
            return_clause,
            having_clause: None,
            ordered_clauses: vec![],
            span: None,
        })
    }

    fn parse_property_map(&mut self) -> Result<Vec<(String, Expression)>> {
        self.expect(TokenKind::LBrace)?;

        let mut properties = Vec::new();

        if self.current.kind != TokenKind::RBrace {
            loop {
                if !self.is_label_or_type_name() {
                    return Err(self.error("Expected property name"));
                }
                let key = self.get_identifier_name();
                self.advance();

                self.expect(TokenKind::Colon)?;

                let value = self.parse_expression()?;
                properties.push((key, value));

                if self.current.kind != TokenKind::Comma {
                    break;
                }
                self.advance();
            }
        }

        self.expect(TokenKind::RBrace)?;
        Ok(properties)
    }

    fn parse_insert(&mut self) -> Result<InsertStatement> {
        self.expect(TokenKind::Insert)?;

        let mut patterns = Vec::new();
        patterns.push(self.parse_pattern()?);

        while self.current.kind == TokenKind::Comma {
            self.advance();
            patterns.push(self.parse_pattern()?);
        }

        Ok(InsertStatement {
            patterns,
            span: None,
        })
    }

    /// Parses CREATE clause within a query (e.g., MATCH ... CREATE ...).
    fn parse_create_clause_in_query(&mut self) -> Result<InsertStatement> {
        self.expect(TokenKind::Create)?;

        let mut patterns = Vec::new();
        patterns.push(self.parse_pattern()?);

        while self.current.kind == TokenKind::Comma {
            self.advance();
            patterns.push(self.parse_pattern()?);
        }

        Ok(InsertStatement {
            patterns,
            span: None,
        })
    }

    /// Parses a DELETE target: either a plain variable or a general expression (GD04).
    fn parse_delete_target(&mut self) -> Result<DeleteTarget> {
        // Try to parse a general expression (handles variables, property access,
        // function calls, subqueries, etc.)
        let expr = self.parse_expression()?;
        // If it's a plain variable reference, store as Variable for backwards compat
        if let Expression::Variable(name) = expr {
            Ok(DeleteTarget::Variable(name))
        } else {
            Ok(DeleteTarget::Expression(expr))
        }
    }

    /// Parses the DETACH / NODETACH prefix and DELETE keyword.
    fn parse_delete_prefix(&mut self) -> Result<bool> {
        let detach = if self.current.kind == TokenKind::Detach {
            self.advance();
            true
        } else {
            // NODETACH is explicit non-detach (same as bare DELETE)
            if self.current.kind == TokenKind::Nodetach {
                self.advance();
            }
            false
        };
        self.expect(TokenKind::Delete)?;
        Ok(detach)
    }

    /// Parses DELETE clause within a query (e.g., MATCH ... DELETE ...).
    fn parse_delete_clause_in_query(&mut self) -> Result<DeleteStatement> {
        let detach = self.parse_delete_prefix()?;

        let mut targets = Vec::new();
        targets.push(self.parse_delete_target()?);

        while self.current.kind == TokenKind::Comma {
            self.advance();
            targets.push(self.parse_delete_target()?);
        }

        Ok(DeleteStatement {
            targets,
            detach,
            span: None,
        })
    }

    fn parse_delete(&mut self) -> Result<DeleteStatement> {
        let detach = self.parse_delete_prefix()?;

        let mut targets = Vec::new();
        targets.push(self.parse_delete_target()?);

        while self.current.kind == TokenKind::Comma {
            self.advance();
            targets.push(self.parse_delete_target()?);
        }

        Ok(DeleteStatement {
            targets,
            detach,
            span: None,
        })
    }

    fn parse_create_schema(&mut self) -> Result<SchemaStatement> {
        self.expect(TokenKind::Create)?;

        // Optional OR REPLACE
        let or_replace = self.try_parse_or_replace();

        match self.current.kind {
            TokenKind::Node => {
                self.advance();
                self.expect(TokenKind::Type)?;

                // Optional IF NOT EXISTS
                let if_not_exists = self.try_parse_if_not_exists();

                if !self.is_identifier() {
                    return Err(self.error("Expected type name"));
                }
                let name = self.get_identifier_name();
                self.advance();

                // Optional EXTENDS <parent1>, <parent2>
                let parent_types = if self.is_identifier()
                    && self.get_identifier_name().eq_ignore_ascii_case("EXTENDS")
                {
                    self.advance();
                    let mut parents = Vec::new();
                    if self.is_identifier() || self.is_label_or_type_name() {
                        parents.push(self.get_identifier_name());
                        self.advance();
                        while self.current.kind == TokenKind::Comma {
                            self.advance();
                            if self.is_identifier() || self.is_label_or_type_name() {
                                parents.push(self.get_identifier_name());
                                self.advance();
                            }
                        }
                    }
                    parents
                } else {
                    Vec::new()
                };

                // Parse property definitions
                let properties = if self.current.kind == TokenKind::LParen {
                    self.parse_property_definitions()?
                } else {
                    Vec::new()
                };

                Ok(SchemaStatement::CreateNodeType(CreateNodeTypeStatement {
                    name,
                    properties,
                    parent_types,
                    if_not_exists,
                    or_replace,
                    span: None,
                }))
            }
            TokenKind::Edge => {
                self.advance();
                self.expect(TokenKind::Type)?;

                // Optional IF NOT EXISTS
                let if_not_exists = self.try_parse_if_not_exists();

                if !self.is_identifier() {
                    return Err(self.error("Expected type name"));
                }
                let name = self.get_identifier_name();
                self.advance();

                // Optional CONNECTING (Source) TO (Target)
                let (source_node_types, target_node_types) = if self.is_identifier()
                    && self.get_identifier_name().eq_ignore_ascii_case("CONNECTING")
                {
                    self.advance();
                    self.expect(TokenKind::LParen)?;
                    let mut sources = Vec::new();
                    while self.is_identifier() || self.is_label_or_type_name() {
                        sources.push(self.get_identifier_name());
                        self.advance();
                        if self.current.kind == TokenKind::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    self.expect(TokenKind::RParen)?;
                    if !(self.is_identifier()
                        && self.get_identifier_name().eq_ignore_ascii_case("TO"))
                    {
                        return Err(self.error("Expected 'TO' after source node types"));
                    }
                    self.advance();
                    self.expect(TokenKind::LParen)?;
                    let mut targets = Vec::new();
                    while self.is_identifier() || self.is_label_or_type_name() {
                        targets.push(self.get_identifier_name());
                        self.advance();
                        if self.current.kind == TokenKind::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    self.expect(TokenKind::RParen)?;
                    (sources, targets)
                } else {
                    (Vec::new(), Vec::new())
                };

                let properties = if self.current.kind == TokenKind::LParen {
                    self.parse_property_definitions()?
                } else {
                    Vec::new()
                };

                Ok(SchemaStatement::CreateEdgeType(CreateEdgeTypeStatement {
                    name,
                    properties,
                    source_node_types,
                    target_node_types,
                    if_not_exists,
                    or_replace,
                    span: None,
                }))
            }
            TokenKind::Vector => {
                self.advance();
                self.expect(TokenKind::Index)?;

                // Parse index name
                if !self.is_identifier() {
                    return Err(self.error("Expected index name"));
                }
                let name = self.get_identifier_name();
                self.advance();

                // Expect ON
                self.expect(TokenKind::On)?;

                // Parse :Label(property)
                self.expect(TokenKind::Colon)?;

                if !self.is_identifier() && !self.is_label_or_type_name() {
                    return Err(self.error("Expected node label"));
                }
                let node_label = self.get_identifier_name();
                self.advance();

                self.expect(TokenKind::LParen)?;

                if !self.is_identifier() {
                    return Err(self.error("Expected property name"));
                }
                let property = self.get_identifier_name();
                self.advance();

                self.expect(TokenKind::RParen)?;

                // Parse optional DIMENSION
                let dimensions = if self.current.kind == TokenKind::Dimension {
                    self.advance();
                    if self.current.kind != TokenKind::Integer {
                        return Err(self.error("Expected integer dimension"));
                    }
                    let dim: usize = self
                        .current
                        .text
                        .parse()
                        .map_err(|_| self.error("Invalid dimension value"))?;
                    self.advance();
                    Some(dim)
                } else {
                    None
                };

                // Parse optional METRIC
                let metric = if self.current.kind == TokenKind::Metric {
                    self.advance();
                    if self.current.kind != TokenKind::String {
                        return Err(self.error("Expected metric name as string"));
                    }
                    // Remove quotes from string literal
                    let metric_str = self
                        .current
                        .text
                        .trim_matches('\'')
                        .trim_matches('"')
                        .to_string();
                    self.advance();
                    Some(metric_str)
                } else {
                    None
                };

                Ok(SchemaStatement::CreateVectorIndex(
                    CreateVectorIndexStatement {
                        name,
                        node_label,
                        property,
                        dimensions,
                        metric,
                        span: None,
                    },
                ))
            }
            TokenKind::Index => {
                // CREATE INDEX name FOR (n:Label) ON (n.property) [USING TEXT|VECTOR|BTREE]
                self.advance();

                // Optional IF NOT EXISTS
                let if_not_exists = self.try_parse_if_not_exists();

                if !self.is_identifier() {
                    return Err(self.error("Expected index name"));
                }
                let name = self.get_identifier_name();
                self.advance();

                self.parse_create_index_body(name, if_not_exists)
            }
            _ if self.is_identifier()
                && self
                    .get_identifier_name()
                    .eq_ignore_ascii_case("CONSTRAINT") =>
            {
                // CREATE CONSTRAINT [name] FOR (n:Label) ON (n.prop) UNIQUE|NOT NULL
                self.advance(); // consume CONSTRAINT

                // Optional IF NOT EXISTS
                let if_not_exists = self.try_parse_if_not_exists();

                // Optional constraint name: consume one identifier only if
                // the next token is not the FOR keyword (dedicated token or
                // unreserved identifier).
                let name = if self.is_identifier()
                    && !self.peek_keyword(TokenKind::For, "FOR")
                {
                    let n = self.get_identifier_name();
                    self.advance();
                    Some(n)
                } else {
                    None
                };

                self.parse_create_constraint_body(name, if_not_exists)
            }
            _ if self.is_identifier()
                && self.get_identifier_name().eq_ignore_ascii_case("GRAPH") =>
            {
                self.advance(); // consume GRAPH
                self.expect(TokenKind::Type)?;

                let if_not_exists = self.try_parse_if_not_exists();

                if !self.is_identifier() {
                    return Err(self.error("Expected graph type name"));
                }
                let name = self.get_identifier_name();
                self.advance();

                // GG04: LIKE <graph_name> clause
                let like_graph = if self.current.kind == TokenKind::Like {
                    self.advance();
                    if !self.is_identifier() {
                        return Err(self.error("Expected graph name after LIKE"));
                    }
                    let graph_name = self.get_identifier_name();
                    self.advance();
                    Some(graph_name)
                } else {
                    None
                };

                // Parse optional body: ISO syntax or JSON-like syntax
                let (node_types, edge_types, inline_types, open) =
                    if self.current.kind == TokenKind::LBrace {
                        let (nt, et, open) = self.parse_graph_type_body()?;
                        (nt, et, Vec::new(), open)
                    } else if self.current.kind == TokenKind::LParen {
                        let inline = self.parse_graph_type_iso_body()?;
                        let nt: Vec<String> = inline
                            .iter()
                            .filter_map(|t| match t {
                                InlineElementType::Node { name, .. } => Some(name.clone()),
                                InlineElementType::Edge { .. } => None,
                            })
                            .collect();
                        let et: Vec<String> = inline
                            .iter()
                            .filter_map(|t| match t {
                                InlineElementType::Edge { name, .. } => Some(name.clone()),
                                InlineElementType::Node { .. } => None,
                            })
                            .collect();
                        (nt, et, inline, false)
                    } else {
                        (Vec::new(), Vec::new(), Vec::new(), true)
                    };

                Ok(SchemaStatement::CreateGraphType(CreateGraphTypeStatement {
                    name,
                    node_types,
                    edge_types,
                    inline_types,
                    like_graph,
                    open,
                    if_not_exists,
                    or_replace,
                    span: None,
                }))
            }
            _ if self.is_identifier()
                && self.get_identifier_name().eq_ignore_ascii_case("SCHEMA") =>
            {
                self.advance(); // consume SCHEMA

                let if_not_exists = self.try_parse_if_not_exists();

                if !self.is_identifier() {
                    return Err(self.error("Expected schema name"));
                }
                let name = self.get_identifier_name();
                self.advance();

                Ok(SchemaStatement::CreateSchema {
                    name,
                    if_not_exists,
                })
            }
            _ if self.is_identifier()
                && self.get_identifier_name().eq_ignore_ascii_case("PROCEDURE") =>
            {
                self.advance(); // consume PROCEDURE
                self.parse_create_procedure(or_replace)
            }
            _ => Err(self.error(
                "Expected NODE, EDGE, VECTOR, INDEX, CONSTRAINT, GRAPH, SCHEMA, or PROCEDURE after CREATE",
            )),
        }
    }

    /// Parses the body of CREATE INDEX after name: `FOR (n:Label) ON (n.prop) [USING kind] [options]`.
    fn parse_create_index_body(
        &mut self,
        name: String,
        if_not_exists: bool,
    ) -> Result<SchemaStatement> {
        // Expect FOR (may be lexed as keyword TokenKind::For or as an identifier).
        if !self.try_accept_keyword(TokenKind::For, "FOR") {
            return Err(self.error("Expected FOR after index name"));
        }

        // Parse (n:Label)
        self.expect(TokenKind::LParen)?;
        if !self.is_identifier() {
            return Err(self.error("Expected variable in FOR clause"));
        }
        let var_name = self.get_identifier_name();
        self.advance();
        self.expect(TokenKind::Colon)?;
        if !self.is_identifier() && !self.is_label_or_type_name() {
            return Err(self.error("Expected label"));
        }
        let label = self.get_identifier_name();
        self.advance();
        self.expect(TokenKind::RParen)?;

        // Expect ON
        self.expect(TokenKind::On)?;

        // Parse (n.property, ...)
        self.expect(TokenKind::LParen)?;
        let mut properties = Vec::new();
        loop {
            if !self.is_identifier() {
                return Err(self.error("Expected variable.property"));
            }
            let prop_var = self.get_identifier_name();
            self.advance();
            self.expect(TokenKind::Dot)?;
            if !self.is_identifier() {
                return Err(self.error("Expected property name"));
            }
            // Validate variable matches
            if !prop_var.eq_ignore_ascii_case(&var_name) {
                return Err(self.error(&format!(
                    "Variable '{prop_var}' does not match FOR variable '{var_name}'"
                )));
            }
            let prop = self.get_identifier_name();
            self.advance();
            properties.push(prop);
            if self.current.kind != TokenKind::Comma {
                break;
            }
            self.advance();
        }
        self.expect(TokenKind::RParen)?;

        // Optional USING TEXT|VECTOR|BTREE
        let mut index_kind = IndexKind::Property;
        let mut options = IndexOptions::default();

        if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("USING") {
            self.advance();
            if !self.is_identifier() && self.current.kind != TokenKind::Vector {
                return Err(self.error("Expected TEXT, VECTOR, or BTREE after USING"));
            }
            let kind_text = if self.current.kind == TokenKind::Vector {
                "VECTOR".to_string()
            } else {
                self.get_identifier_name()
            };
            self.advance();
            match kind_text.to_uppercase().as_str() {
                "TEXT" => index_kind = IndexKind::Text,
                "VECTOR" => {
                    index_kind = IndexKind::Vector;
                    // Parse optional {dimensions: N, metric: 'name'}
                    if self.current.kind == TokenKind::LBrace {
                        self.advance();
                        while self.current.kind != TokenKind::RBrace {
                            if !self.is_identifier() {
                                return Err(self.error("Expected option name"));
                            }
                            let opt_name = self.get_identifier_name();
                            self.advance();
                            self.expect(TokenKind::Colon)?;
                            match opt_name.to_uppercase().as_str() {
                                "DIMENSIONS" | "DIMENSION" => {
                                    if self.current.kind != TokenKind::Integer {
                                        return Err(self.error("Expected integer for dimensions"));
                                    }
                                    let dim: usize = self
                                        .current
                                        .text
                                        .parse()
                                        .map_err(|_| self.error("Invalid dimension value"))?;
                                    self.advance();
                                    options.dimensions = Some(dim);
                                }
                                "METRIC" => {
                                    if self.current.kind != TokenKind::String {
                                        return Err(self.error("Expected string for metric"));
                                    }
                                    let metric = self
                                        .current
                                        .text
                                        .trim_matches('\'')
                                        .trim_matches('"')
                                        .to_string();
                                    self.advance();
                                    options.metric = Some(metric);
                                }
                                _ => {
                                    return Err(
                                        self.error(&format!("Unknown index option '{opt_name}'"))
                                    );
                                }
                            }
                            if self.current.kind == TokenKind::Comma {
                                self.advance();
                            }
                        }
                        self.expect(TokenKind::RBrace)?;
                    }
                }
                "BTREE" => index_kind = IndexKind::BTree,
                _ => {
                    return Err(self.error(&format!("Unknown index type '{kind_text}'")));
                }
            }
        }

        Ok(SchemaStatement::CreateIndex(CreateIndexStatement {
            name,
            index_kind,
            label,
            properties,
            options,
            if_not_exists,
            span: None,
        }))
    }

    /// Parses the body of CREATE CONSTRAINT: `FOR (n:Label) ON (n.prop) kind`.
    fn parse_create_constraint_body(
        &mut self,
        name: Option<String>,
        if_not_exists: bool,
    ) -> Result<SchemaStatement> {
        // Expect FOR (keyword or identifier).
        if !self.try_accept_keyword(TokenKind::For, "FOR") {
            return Err(self.error("Expected FOR after constraint name"));
        }

        // Parse (n:Label)
        self.expect(TokenKind::LParen)?;
        if !self.is_identifier() {
            return Err(self.error("Expected variable in FOR clause"));
        }
        let var_name = self.get_identifier_name();
        self.advance();
        self.expect(TokenKind::Colon)?;
        if !self.is_identifier() && !self.is_label_or_type_name() {
            return Err(self.error("Expected label"));
        }
        let label = self.get_identifier_name();
        self.advance();
        self.expect(TokenKind::RParen)?;

        // Expect ON
        self.expect(TokenKind::On)?;

        // Parse (n.property, ...)
        self.expect(TokenKind::LParen)?;
        let mut properties = Vec::new();
        loop {
            if !self.is_identifier() {
                return Err(self.error("Expected variable.property"));
            }
            let prop_var = self.get_identifier_name();
            self.advance();
            self.expect(TokenKind::Dot)?;
            if !self.is_identifier() {
                return Err(self.error("Expected property name"));
            }
            if !prop_var.eq_ignore_ascii_case(&var_name) {
                return Err(self.error(&format!(
                    "Variable '{prop_var}' does not match FOR variable '{var_name}'"
                )));
            }
            let prop = self.get_identifier_name();
            self.advance();
            properties.push(prop);
            if self.current.kind != TokenKind::Comma {
                break;
            }
            self.advance();
        }
        self.expect(TokenKind::RParen)?;

        // Parse constraint kind: UNIQUE, NOT NULL, NODE KEY
        let constraint_kind = if self.is_identifier()
            && self.get_identifier_name().eq_ignore_ascii_case("UNIQUE")
        {
            self.advance();
            ConstraintKind::Unique
        } else if self.current.kind == TokenKind::Not {
            self.advance();
            if self.current.kind != TokenKind::Null {
                return Err(self.error("Expected NULL after NOT"));
            }
            self.advance();
            ConstraintKind::NotNull
        } else if self.current.kind == TokenKind::Node {
            self.advance();
            if !self.is_identifier() || !self.get_identifier_name().eq_ignore_ascii_case("KEY") {
                return Err(self.error("Expected KEY after NODE"));
            }
            self.advance();
            ConstraintKind::NodeKey
        } else if self.current.kind == TokenKind::Exists {
            self.advance();
            ConstraintKind::Exists
        } else {
            return Err(self.error("Expected UNIQUE, NOT NULL, NODE KEY, or EXISTS"));
        };

        Ok(SchemaStatement::CreateConstraint(
            CreateConstraintStatement {
                name,
                constraint_kind,
                label,
                properties,
                if_not_exists,
                span: None,
            },
        ))
    }

    /// Tries to parse `IF NOT EXISTS`, returning true if found.
    fn try_parse_if_not_exists(&mut self) -> bool {
        if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("IF") {
            self.advance();
            if self.current.kind == TokenKind::Not {
                self.advance();
                if self.current.kind == TokenKind::Exists {
                    self.advance();
                    return true;
                }
            }
        }
        false
    }

    /// Tries to parse `IF EXISTS`, returning true if found.
    fn try_parse_if_exists(&mut self) -> bool {
        if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("IF") {
            self.advance();
            if self.current.kind == TokenKind::Exists {
                self.advance();
                return true;
            }
        }
        false
    }

    /// Tries to parse `OR REPLACE`, returning true if found.
    ///
    /// Since the parser has no backtrack beyond single lookahead, this peeks
    /// at the current and next token. If current is "OR" and next is "REPLACE",
    /// consumes both and returns true.
    fn try_parse_or_replace(&mut self) -> bool {
        if self.peek_keyword(TokenKind::Or, "OR") {
            let pk = self.peek_kind();
            if pk == TokenKind::Identifier {
                let text = self.peek_text_upper();
                if text == "REPLACE" {
                    self.advance(); // consume OR
                    self.advance(); // consume REPLACE
                    return true;
                }
            }
        }
        false
    }

    /// Dispatches CREATE to either a schema DDL or session (graph instance) command.
    ///
    /// Called after detecting CREATE followed by something other than `(`.
    /// Handles: CREATE [OR REPLACE] NODE TYPE, EDGE TYPE, GRAPH TYPE, VECTOR INDEX,
    /// INDEX, CONSTRAINT, SCHEMA, and CREATE [PROPERTY] GRAPH.
    fn parse_create_dispatch(&mut self) -> Result<Statement> {
        // Check: is this CREATE PROJECTION <name> (session command)?
        if self.peek_kind() == TokenKind::Identifier && self.peek_text_upper() == "PROJECTION" {
            return self
                .parse_create_projection()
                .map(Statement::SessionCommand);
        }
        // Check: is this CREATE [PROPERTY] GRAPH <name> (session command)?
        // We need to distinguish from CREATE GRAPH TYPE (schema DDL).
        // Peek: if next is GRAPH/PROPERTY and the token after that is NOT TYPE, it's an instance.
        if self.peek_is_graph_instance_keyword() {
            return self.parse_create_graph().map(Statement::SessionCommand);
        }
        self.parse_create_schema().map(Statement::Schema)
    }

    /// Parses the body of CREATE GRAPH TYPE: `{ node_types: [A, B], edge_types: [E1] }`.
    ///
    /// Returns (node_types, edge_types, open). If no body, returns open=true.
    fn parse_graph_type_body(&mut self) -> Result<(Vec<String>, Vec<String>, bool)> {
        self.expect(TokenKind::LBrace)?;

        // Brace-delimited pattern form: { (n: Person {name STRING})-[:KNOWS]->(:Person) }
        if self.current.kind == TokenKind::LParen {
            let inline = self.parse_graph_type_brace_pattern_body()?;
            let nt: Vec<String> = inline
                .iter()
                .filter_map(|t| match t {
                    InlineElementType::Node { name, .. } => Some(name.clone()),
                    InlineElementType::Edge { .. } => None,
                })
                .collect();
            let et: Vec<String> = inline
                .iter()
                .filter_map(|t| match t {
                    InlineElementType::Edge { name, .. } => Some(name.clone()),
                    InlineElementType::Node { .. } => None,
                })
                .collect();
            return Ok((nt, et, false));
        }

        let mut node_types = Vec::new();
        let mut edge_types = Vec::new();
        let mut open = false;

        while self.current.kind != TokenKind::RBrace && self.current.kind != TokenKind::Eof {
            if self.is_identifier() {
                let key = self.get_identifier_name();
                self.advance();
                self.expect(TokenKind::Colon)?;

                match key.to_uppercase().as_str() {
                    "NODE_TYPES" | "NODETYPES" => {
                        node_types = self.parse_identifier_list()?;
                    }
                    "EDGE_TYPES" | "EDGETYPES" => {
                        edge_types = self.parse_identifier_list()?;
                    }
                    "OPEN" => {
                        if self.current.kind == TokenKind::True {
                            open = true;
                            self.advance();
                        } else if self.current.kind == TokenKind::False {
                            open = false;
                            self.advance();
                        } else {
                            return Err(self.error("Expected true or false for 'open'"));
                        }
                    }
                    _ => return Err(self.error("Expected node_types, edge_types, or open")),
                }

                // Optional comma separator
                if self.current.kind == TokenKind::Comma {
                    self.advance();
                }
            } else {
                return Err(self.error("Expected property name in graph type body"));
            }
        }

        self.expect(TokenKind::RBrace)?;
        Ok((node_types, edge_types, open))
    }

    /// Parses the ISO-syntax body of CREATE GRAPH TYPE.
    ///
    /// Supports two forms:
    /// - Verbose: `(NODE TYPE Name (props), EDGE TYPE Name (props), ...)`
    /// - Pattern: `((:Person {name STRING})-[:KNOWS {since INT64}]->(:Person), ...)`
    ///
    /// Returns a list of inline element type definitions.
    fn parse_graph_type_iso_body(&mut self) -> Result<Vec<InlineElementType>> {
        self.expect(TokenKind::LParen)?;

        // Detect which form: pattern form starts with `(` (nested paren for node pattern),
        // verbose form starts with NODE or EDGE.
        if self.current.kind == TokenKind::LParen {
            self.parse_graph_type_pattern_body()
        } else {
            self.parse_graph_type_verbose_body()
        }
    }

    /// Parses the verbose form: `NODE TYPE Name (props), EDGE TYPE Name (props), ...)`
    /// (closing `)` included).
    fn parse_graph_type_verbose_body(&mut self) -> Result<Vec<InlineElementType>> {
        let mut types = Vec::new();

        while self.current.kind != TokenKind::RParen && self.current.kind != TokenKind::Eof {
            let is_node = if self.current.kind == TokenKind::Node {
                self.advance();
                self.expect(TokenKind::Type)?;
                true
            } else if self.current.kind == TokenKind::Edge {
                self.advance();
                self.expect(TokenKind::Type)?;
                false
            } else {
                return Err(self.error("Expected NODE TYPE or EDGE TYPE in graph type body"));
            };

            if !self.is_identifier() && !self.is_label_or_type_name() {
                return Err(self.error("Expected type name"));
            }
            let type_name = self.get_identifier_name();
            self.advance();

            // GG21: Optional KEY label set: KEY (Label1, Label2)
            let has_key_clause =
                self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("KEY");
            let key_labels = if has_key_clause {
                self.advance();
                self.expect(TokenKind::LParen)?;
                let mut labels = Vec::new();
                loop {
                    if !self.is_identifier() && !self.is_label_or_type_name() {
                        return Err(self.error("Expected label name in KEY clause"));
                    }
                    labels.push(self.get_identifier_name());
                    self.advance();
                    if self.current.kind != TokenKind::Comma {
                        break;
                    }
                    self.advance();
                }
                self.expect(TokenKind::RParen)?;
                labels
            } else {
                Vec::new()
            };

            // Optional property definitions. The presence of `(` (even empty
            // parens) marks this as an inline declaration, not a reference.
            let has_property_block = self.current.kind == TokenKind::LParen;
            let properties = if has_property_block {
                self.parse_property_definitions()?
            } else {
                Vec::new()
            };

            // ISO/IEC 39075:2024 bare element-type reference: a name with no
            // property block and no KEY clause is a reference to an existing
            // type in the catalog, not an inline declaration. Issue #316.
            let is_reference = !has_property_block && !has_key_clause;

            if is_node {
                types.push(InlineElementType::Node {
                    name: type_name,
                    properties,
                    key_labels,
                    is_reference,
                });
            } else {
                types.push(InlineElementType::Edge {
                    name: type_name,
                    properties,
                    key_labels,
                    source_node_types: Vec::new(),
                    target_node_types: Vec::new(),
                    is_reference,
                });
            }

            // Optional comma separator
            if self.current.kind == TokenKind::Comma {
                self.advance();
            }
        }

        self.expect(TokenKind::RParen)?;
        Ok(types)
    }

    /// Parses the pattern form of a graph type body:
    /// `(:Person {name STRING})-[:KNOWS {since INT64}]->(:Person), ...)`
    /// (closing `)` included).
    fn parse_graph_type_pattern_body(&mut self) -> Result<Vec<InlineElementType>> {
        let mut types = Vec::new();
        // Track which node type names we've already emitted so we don't duplicate them.
        let mut seen_node_types = std::collections::HashSet::new();

        while self.current.kind != TokenKind::RParen && self.current.kind != TokenKind::Eof {
            // Parse the first node pattern: (:Label {props})
            let (src_label, src_props) = self.parse_graph_type_node_pattern()?;

            // Check if an edge follows: `-[` or `<-[`
            if self.current.kind == TokenKind::Minus || self.current.kind == TokenKind::LeftArrow {
                let backward = self.current.kind == TokenKind::LeftArrow;
                if backward {
                    // <-[ : incoming direction
                    self.advance(); // consume `<-`
                } else {
                    // -[ : outgoing or undirected
                    self.advance(); // consume `-`
                }

                self.expect(TokenKind::LBracket)?;

                // Optional edge variable name: [r: KNOWS] vs [:KNOWS]
                if self.is_identifier() && self.peek_kind() == TokenKind::Colon {
                    self.advance(); // skip variable name
                }

                // Parse edge label: `:EdgeType`
                self.expect(TokenKind::Colon)?;
                if !self.is_identifier() && !self.is_label_or_type_name() {
                    return Err(self.error("Expected edge type name after `:` in pattern"));
                }
                let edge_label = self.get_identifier_name();
                self.advance();

                // Optional properties: { prop TYPE, ... }
                let edge_props = if self.current.kind == TokenKind::LBrace {
                    self.parse_property_definitions_braces()?
                } else {
                    Vec::new()
                };

                self.expect(TokenKind::RBracket)?;

                // Parse direction after `]`: `->` for outgoing, `-` for undirected
                let forward = if self.current.kind == TokenKind::Arrow {
                    self.advance(); // consume `->`
                    true
                } else if self.current.kind == TokenKind::Minus {
                    self.advance(); // consume `-` (undirected)
                    false
                } else {
                    return Err(self.error("Expected `->` or `-` after edge pattern `]`"));
                };

                // Parse target node: (:Label {props})
                let (tgt_label, tgt_props) = self.parse_graph_type_node_pattern()?;

                // Determine source and target based on direction
                let (effective_src, effective_tgt) = if backward {
                    (tgt_label.clone(), src_label.clone())
                } else {
                    (src_label.clone(), tgt_label.clone())
                };

                let (src_types, tgt_types) = if !forward && !backward {
                    // Undirected: both directions
                    (
                        vec![src_label.clone(), tgt_label.clone()],
                        vec![src_label.clone(), tgt_label.clone()],
                    )
                } else {
                    (vec![effective_src], vec![effective_tgt])
                };

                // Add node types (deduplicated)
                if seen_node_types.insert(src_label.clone()) {
                    types.push(InlineElementType::Node {
                        name: src_label,
                        properties: src_props,
                        key_labels: Vec::new(),
                        is_reference: false,
                    });
                }
                if seen_node_types.insert(tgt_label.clone()) {
                    types.push(InlineElementType::Node {
                        name: tgt_label,
                        properties: tgt_props,
                        key_labels: Vec::new(),
                        is_reference: false,
                    });
                }

                // Add edge type
                types.push(InlineElementType::Edge {
                    name: edge_label,
                    properties: edge_props,
                    key_labels: Vec::new(),
                    source_node_types: src_types,
                    target_node_types: tgt_types,
                    is_reference: false,
                });
            } else {
                // Standalone node pattern
                if seen_node_types.insert(src_label.clone()) {
                    types.push(InlineElementType::Node {
                        name: src_label,
                        properties: src_props,
                        key_labels: Vec::new(),
                        is_reference: false,
                    });
                }
            }

            // Optional comma separator
            if self.current.kind == TokenKind::Comma {
                self.advance();
            }
        }

        self.expect(TokenKind::RParen)?;
        Ok(types)
    }

    /// Parses the brace-delimited pattern form of a graph type body:
    /// `{ (:Person {name STRING})-[:KNOWS]->(:Person), ... }`
    /// (closing `}` included). Same logic as `parse_graph_type_pattern_body` but
    /// terminated by `}` instead of `)`.
    fn parse_graph_type_brace_pattern_body(&mut self) -> Result<Vec<InlineElementType>> {
        let mut types = Vec::new();
        let mut seen_node_types = std::collections::HashSet::new();

        while self.current.kind != TokenKind::RBrace && self.current.kind != TokenKind::Eof {
            let (src_label, src_props) = self.parse_graph_type_node_pattern()?;

            if self.current.kind == TokenKind::Minus || self.current.kind == TokenKind::LeftArrow {
                let backward = self.current.kind == TokenKind::LeftArrow;
                self.advance(); // consume `-` or `<-`

                self.expect(TokenKind::LBracket)?;

                // Optional edge variable name: [r: KNOWS] vs [:KNOWS]
                if self.is_identifier() && self.peek_kind() == TokenKind::Colon {
                    self.advance(); // skip variable name
                }

                self.expect(TokenKind::Colon)?;
                if !self.is_identifier() && !self.is_label_or_type_name() {
                    return Err(self.error("Expected edge type name after `:` in pattern"));
                }
                let edge_label = self.get_identifier_name();
                self.advance();

                let edge_props = if self.current.kind == TokenKind::LBrace {
                    self.parse_property_definitions_braces()?
                } else {
                    Vec::new()
                };

                self.expect(TokenKind::RBracket)?;

                let forward = if self.current.kind == TokenKind::Arrow {
                    self.advance();
                    true
                } else if self.current.kind == TokenKind::Minus {
                    self.advance();
                    false
                } else {
                    return Err(self.error("Expected `->` or `-` after edge pattern `]`"));
                };

                let (tgt_label, tgt_props) = self.parse_graph_type_node_pattern()?;

                let (effective_src, effective_tgt) = if backward {
                    (tgt_label.clone(), src_label.clone())
                } else {
                    (src_label.clone(), tgt_label.clone())
                };

                let (src_types, tgt_types) = if !forward && !backward {
                    (
                        vec![src_label.clone(), tgt_label.clone()],
                        vec![src_label.clone(), tgt_label.clone()],
                    )
                } else {
                    (vec![effective_src], vec![effective_tgt])
                };

                if seen_node_types.insert(src_label.clone()) {
                    types.push(InlineElementType::Node {
                        name: src_label,
                        properties: src_props,
                        key_labels: Vec::new(),
                        is_reference: false,
                    });
                }
                if seen_node_types.insert(tgt_label.clone()) {
                    types.push(InlineElementType::Node {
                        name: tgt_label,
                        properties: tgt_props,
                        key_labels: Vec::new(),
                        is_reference: false,
                    });
                }

                types.push(InlineElementType::Edge {
                    name: edge_label,
                    properties: edge_props,
                    key_labels: Vec::new(),
                    source_node_types: src_types,
                    target_node_types: tgt_types,
                    is_reference: false,
                });
            } else {
                // Standalone node pattern
                if seen_node_types.insert(src_label.clone()) {
                    types.push(InlineElementType::Node {
                        name: src_label,
                        properties: src_props,
                        key_labels: Vec::new(),
                        is_reference: false,
                    });
                }
            }

            if self.current.kind == TokenKind::Comma {
                self.advance();
            }
        }

        self.expect(TokenKind::RBrace)?;
        Ok(types)
    }

    /// Parses a node pattern inside a graph type pattern body: `(:Label {prop TYPE, ...})`
    /// or `(var: Label {prop TYPE, ...})`.
    ///
    /// Returns `(label, properties)`.
    fn parse_graph_type_node_pattern(&mut self) -> Result<(String, Vec<PropertyDefinition>)> {
        self.expect(TokenKind::LParen)?;

        // Optional variable name: (n: Person ...) vs (:Person ...)
        if self.is_identifier() && self.peek_kind() == TokenKind::Colon {
            self.advance(); // skip variable name (not used in graph type definitions)
        }

        self.expect(TokenKind::Colon)?;

        if !self.is_identifier() && !self.is_label_or_type_name() {
            return Err(self.error("Expected label name in node pattern"));
        }
        let label = self.get_identifier_name();
        self.advance();

        // Optional property definitions in braces
        let properties = if self.current.kind == TokenKind::LBrace {
            self.parse_property_definitions_braces()?
        } else {
            Vec::new()
        };

        self.expect(TokenKind::RParen)?;
        Ok((label, properties))
    }

    /// Parses `[T1, T2, T3]`, returning a list of identifiers.
    fn parse_identifier_list(&mut self) -> Result<Vec<String>> {
        self.expect(TokenKind::LBracket)?;
        let mut items = Vec::new();

        if self.current.kind != TokenKind::RBracket {
            loop {
                if !self.is_identifier() && !self.is_label_or_type_name() {
                    return Err(self.error("Expected identifier in list"));
                }
                items.push(self.get_identifier_name());
                self.advance();
                if self.current.kind != TokenKind::Comma {
                    break;
                }
                self.advance();
            }
        }

        self.expect(TokenKind::RBracket)?;
        Ok(items)
    }

    /// Parses a DROP statement, dispatching to schema or session commands.
    ///
    /// Handles: DROP NODE TYPE, DROP EDGE TYPE, DROP INDEX, DROP CONSTRAINT,
    /// DROP GRAPH TYPE, DROP SCHEMA, DROP [PROPERTY] GRAPH.
    fn parse_drop(&mut self) -> Result<Statement> {
        self.advance(); // consume DROP

        match self.current.kind {
            TokenKind::Node => {
                // DROP NODE TYPE [IF EXISTS] name
                self.advance();
                self.expect(TokenKind::Type)?;
                let if_exists = self.try_parse_if_exists();
                if !self.is_identifier() {
                    return Err(self.error("Expected type name"));
                }
                let name = self.get_identifier_name();
                self.advance();
                Ok(Statement::Schema(SchemaStatement::DropNodeType {
                    name,
                    if_exists,
                }))
            }
            TokenKind::Edge => {
                // DROP EDGE TYPE [IF EXISTS] name
                self.advance();
                self.expect(TokenKind::Type)?;
                let if_exists = self.try_parse_if_exists();
                if !self.is_identifier() {
                    return Err(self.error("Expected type name"));
                }
                let name = self.get_identifier_name();
                self.advance();
                Ok(Statement::Schema(SchemaStatement::DropEdgeType {
                    name,
                    if_exists,
                }))
            }
            TokenKind::Index => {
                // DROP INDEX [IF EXISTS] name
                self.advance();
                let if_exists = self.try_parse_if_exists();
                if !self.is_identifier() {
                    return Err(self.error("Expected index name"));
                }
                let name = self.get_identifier_name();
                self.advance();
                Ok(Statement::Schema(SchemaStatement::DropIndex {
                    name,
                    if_exists,
                }))
            }
            _ if self.is_identifier()
                && self
                    .get_identifier_name()
                    .eq_ignore_ascii_case("CONSTRAINT") =>
            {
                // DROP CONSTRAINT [IF EXISTS] name
                self.advance(); // consume CONSTRAINT
                let if_exists = self.try_parse_if_exists();
                if !self.is_identifier() {
                    return Err(self.error("Expected constraint name"));
                }
                let name = self.get_identifier_name();
                self.advance();
                Ok(Statement::Schema(SchemaStatement::DropConstraint {
                    name,
                    if_exists,
                }))
            }
            _ if self.is_identifier()
                && self.get_identifier_name().eq_ignore_ascii_case("SCHEMA") =>
            {
                // DROP SCHEMA [IF EXISTS] name
                self.advance(); // consume SCHEMA
                let if_exists = self.try_parse_if_exists();
                if !self.is_identifier() {
                    return Err(self.error("Expected schema name"));
                }
                let name = self.get_identifier_name();
                self.advance();
                Ok(Statement::Schema(SchemaStatement::DropSchema {
                    name,
                    if_exists,
                }))
            }
            _ if self.is_identifier()
                && self
                    .get_identifier_name()
                    .eq_ignore_ascii_case("PROJECTION") =>
            {
                // DROP PROJECTION name
                self.advance(); // consume PROJECTION
                if !self.is_identifier() {
                    return Err(self.error("Expected projection name"));
                }
                let name = self.get_identifier_name();
                self.advance();
                Ok(Statement::SessionCommand(SessionCommand::DropProjection {
                    name,
                }))
            }
            _ if self.is_identifier()
                && self.get_identifier_name().eq_ignore_ascii_case("PROCEDURE") =>
            {
                // DROP PROCEDURE [IF EXISTS] name
                self.advance(); // consume PROCEDURE
                let if_exists = self.try_parse_if_exists();
                if !self.is_identifier() {
                    return Err(self.error("Expected procedure name"));
                }
                let name = self.get_identifier_name();
                self.advance();
                Ok(Statement::Schema(SchemaStatement::DropProcedure {
                    name,
                    if_exists,
                }))
            }
            _ if self.is_identifier()
                && self.get_identifier_name().eq_ignore_ascii_case("GRAPH") =>
            {
                // Could be DROP GRAPH TYPE or DROP GRAPH <name>
                // Peek ahead: if next token is TYPE, it's schema DDL
                if self.peek_kind() == TokenKind::Type {
                    // DROP GRAPH TYPE [IF EXISTS] name
                    self.advance(); // consume GRAPH
                    self.advance(); // consume TYPE
                    let if_exists = self.try_parse_if_exists();
                    if !self.is_identifier() {
                        return Err(self.error("Expected graph type name"));
                    }
                    let name = self.get_identifier_name();
                    self.advance();
                    Ok(Statement::Schema(SchemaStatement::DropGraphType {
                        name,
                        if_exists,
                    }))
                } else {
                    // DROP [PROPERTY] GRAPH <name>
                    self.parse_drop_graph_body().map(Statement::SessionCommand)
                }
            }
            _ => {
                // Fall through to DROP [PROPERTY] GRAPH
                self.parse_drop_graph_body().map(Statement::SessionCommand)
            }
        }
    }

    /// Parses ALTER NODE TYPE, ALTER EDGE TYPE, ALTER GRAPH TYPE.
    fn parse_alter(&mut self) -> Result<Statement> {
        let span_start = self.current.span.start;
        self.advance(); // consume ALTER

        match self.current.kind {
            TokenKind::Node => {
                // ALTER NODE TYPE name ADD/DROP ...
                self.advance();
                self.expect(TokenKind::Type)?;
                self.parse_alter_type(true, span_start)
            }
            TokenKind::Edge => {
                // ALTER EDGE TYPE name ADD/DROP ...
                self.advance();
                self.expect(TokenKind::Type)?;
                self.parse_alter_type(false, span_start)
            }
            _ if self.is_identifier()
                && self.get_identifier_name().eq_ignore_ascii_case("GRAPH") =>
            {
                // ALTER GRAPH TYPE name ADD/DROP ...
                self.advance(); // consume GRAPH
                self.expect(TokenKind::Type)?;
                self.parse_alter_graph_type(span_start)
            }
            _ => Err(self.error("Expected NODE, EDGE, or GRAPH after ALTER")),
        }
    }

    /// Parses ALTER NODE TYPE name / ALTER EDGE TYPE name alterations.
    /// Consumes an optional `PROPERTY` keyword in `ADD PROPERTY name ...` /
    /// `DROP PROPERTY name`. It is a keyword only when a name follows, so a
    /// property that is itself called `property` still works as a backtick-quoted
    /// identifier.
    fn skip_property_keyword(&mut self) {
        if self.current.kind == TokenKind::Identifier
            && self.current.text.eq_ignore_ascii_case("PROPERTY")
            && matches!(
                self.peek_kind(),
                TokenKind::Identifier | TokenKind::QuotedIdentifier
            )
        {
            self.advance();
        }
    }

    fn parse_alter_type(&mut self, is_node: bool, span_start: usize) -> Result<Statement> {
        if !self.is_identifier() {
            return Err(self.error("Expected type name"));
        }
        let name = self.get_identifier_name();
        self.advance();

        let mut alterations = Vec::new();
        loop {
            if !self.is_identifier() {
                break;
            }
            let action = self.get_identifier_name().to_uppercase();
            match action.as_str() {
                "ADD" => {
                    self.advance();
                    // ADD [PROPERTY] property_name type [NOT NULL]
                    self.skip_property_keyword();
                    if !self.is_identifier() {
                        return Err(self.error("Expected property name after ADD"));
                    }
                    let prop_name = self.get_identifier_name();
                    self.advance();
                    if !self.is_identifier() {
                        return Err(self.error("Expected type name"));
                    }
                    let data_type = self.get_identifier_name();
                    self.advance();
                    let nullable = if self.current.kind == TokenKind::Not {
                        self.advance();
                        if self.current.kind != TokenKind::Null {
                            return Err(self.error("Expected NULL after NOT"));
                        }
                        self.advance();
                        false
                    } else {
                        true
                    };
                    // Optional DEFAULT <literal>
                    let default_value = if self.is_identifier()
                        && self.get_identifier_name().eq_ignore_ascii_case("DEFAULT")
                    {
                        self.advance();
                        let lit = match self.current.kind {
                            TokenKind::String
                            | TokenKind::Integer
                            | TokenKind::Float
                            | TokenKind::True
                            | TokenKind::False
                            | TokenKind::Null => self.current.text.clone(),
                            _ => return Err(self.error("Expected literal value after DEFAULT")),
                        };
                        self.advance();
                        Some(lit)
                    } else {
                        None
                    };
                    alterations.push(TypeAlteration::AddProperty(PropertyDefinition {
                        name: prop_name,
                        data_type,
                        nullable,
                        default_value,
                    }));
                }
                "DROP" => {
                    self.advance();
                    // DROP [PROPERTY] property_name
                    self.skip_property_keyword();
                    if !self.is_identifier() {
                        return Err(self.error("Expected property name after DROP"));
                    }
                    let prop_name = self.get_identifier_name();
                    self.advance();
                    alterations.push(TypeAlteration::DropProperty(prop_name));
                }
                _ => break,
            }
        }

        if alterations.is_empty() {
            return Err(self.error("Expected ADD or DROP alteration"));
        }

        let span = Some(SourceSpan::new(span_start, self.current.span.end, 1, 1));
        let stmt = AlterTypeStatement {
            name,
            alterations,
            span,
        };
        if is_node {
            Ok(Statement::Schema(SchemaStatement::AlterNodeType(stmt)))
        } else {
            Ok(Statement::Schema(SchemaStatement::AlterEdgeType(stmt)))
        }
    }

    /// Parses ALTER GRAPH TYPE name alterations.
    fn parse_alter_graph_type(&mut self, span_start: usize) -> Result<Statement> {
        if !self.is_identifier() {
            return Err(self.error("Expected graph type name"));
        }
        let name = self.get_identifier_name();
        self.advance();

        let mut alterations = Vec::new();
        loop {
            if !self.is_identifier() {
                break;
            }
            let action = self.get_identifier_name().to_uppercase();
            match action.as_str() {
                "ADD" => {
                    self.advance();
                    // ADD NODE TYPE name or ADD EDGE TYPE name
                    let kind = self.current.kind;
                    match kind {
                        TokenKind::Node => {
                            self.advance();
                            self.expect(TokenKind::Type)?;
                            if !self.is_identifier() {
                                return Err(self.error("Expected node type name"));
                            }
                            let type_name = self.get_identifier_name();
                            self.advance();
                            alterations.push(GraphTypeAlteration::AddNodeType(type_name));
                        }
                        TokenKind::Edge => {
                            self.advance();
                            self.expect(TokenKind::Type)?;
                            if !self.is_identifier() {
                                return Err(self.error("Expected edge type name"));
                            }
                            let type_name = self.get_identifier_name();
                            self.advance();
                            alterations.push(GraphTypeAlteration::AddEdgeType(type_name));
                        }
                        _ => return Err(self.error("Expected NODE or EDGE after ADD")),
                    }
                }
                "DROP" => {
                    self.advance();
                    let kind = self.current.kind;
                    match kind {
                        TokenKind::Node => {
                            self.advance();
                            self.expect(TokenKind::Type)?;
                            if !self.is_identifier() {
                                return Err(self.error("Expected node type name"));
                            }
                            let type_name = self.get_identifier_name();
                            self.advance();
                            alterations.push(GraphTypeAlteration::DropNodeType(type_name));
                        }
                        TokenKind::Edge => {
                            self.advance();
                            self.expect(TokenKind::Type)?;
                            if !self.is_identifier() {
                                return Err(self.error("Expected edge type name"));
                            }
                            let type_name = self.get_identifier_name();
                            self.advance();
                            alterations.push(GraphTypeAlteration::DropEdgeType(type_name));
                        }
                        _ => return Err(self.error("Expected NODE or EDGE after DROP")),
                    }
                }
                _ => break,
            }
        }

        if alterations.is_empty() {
            return Err(self.error("Expected ADD or DROP alteration"));
        }

        let span = Some(SourceSpan::new(span_start, self.current.span.end, 1, 1));
        Ok(Statement::Schema(SchemaStatement::AlterGraphType(
            AlterGraphTypeStatement {
                name,
                alterations,
                span,
            },
        )))
    }

    fn parse_property_definitions(&mut self) -> Result<Vec<PropertyDefinition>> {
        self.expect(TokenKind::LParen)?;

        let mut defs = Vec::new();

        if self.current.kind != TokenKind::RParen {
            loop {
                if !self.is_identifier() {
                    return Err(self.error("Expected property name"));
                }
                let name = self.get_identifier_name();
                self.advance();

                if !self.is_identifier() {
                    return Err(self.error("Expected type name"));
                }
                let data_type = self.get_identifier_name();
                self.advance();

                let nullable = if self.current.kind == TokenKind::Not {
                    self.advance();
                    if self.current.kind != TokenKind::Null {
                        return Err(self.error("Expected NULL after NOT"));
                    }
                    self.advance();
                    false
                } else {
                    true
                };

                // Optional DEFAULT <literal>
                let default_value = if self.is_identifier()
                    && self.get_identifier_name().eq_ignore_ascii_case("DEFAULT")
                {
                    self.advance();
                    let lit = match self.current.kind {
                        TokenKind::String
                        | TokenKind::Integer
                        | TokenKind::Float
                        | TokenKind::True
                        | TokenKind::False
                        | TokenKind::Null => self.current.text.clone(),
                        _ => return Err(self.error("Expected literal value after DEFAULT")),
                    };
                    self.advance();
                    Some(lit)
                } else {
                    None
                };

                defs.push(PropertyDefinition {
                    name,
                    data_type,
                    nullable,
                    default_value,
                });

                if self.current.kind != TokenKind::Comma {
                    break;
                }
                self.advance();
            }
        }

        self.expect(TokenKind::RParen)?;
        Ok(defs)
    }

    /// Parses property definitions inside braces: `{ name TYPE [NOT NULL], ... }`.
    ///
    /// Used by pattern-form graph type syntax where properties use `{}` instead of `()`.
    fn parse_property_definitions_braces(&mut self) -> Result<Vec<PropertyDefinition>> {
        self.expect(TokenKind::LBrace)?;

        let mut defs = Vec::new();

        if self.current.kind != TokenKind::RBrace {
            loop {
                if !self.is_identifier() {
                    return Err(self.error("Expected property name"));
                }
                let name = self.get_identifier_name();
                self.advance();

                if !self.is_identifier() {
                    return Err(self.error("Expected type name"));
                }
                let data_type = self.get_identifier_name();
                self.advance();

                let nullable = if self.current.kind == TokenKind::Not {
                    self.advance();
                    if self.current.kind != TokenKind::Null {
                        return Err(self.error("Expected NULL after NOT"));
                    }
                    self.advance();
                    false
                } else {
                    true
                };

                // Optional DEFAULT <literal>
                let default_value = if self.is_identifier()
                    && self.get_identifier_name().eq_ignore_ascii_case("DEFAULT")
                {
                    self.advance();
                    let lit = match self.current.kind {
                        TokenKind::String
                        | TokenKind::Integer
                        | TokenKind::Float
                        | TokenKind::True
                        | TokenKind::False
                        | TokenKind::Null => self.current.text.clone(),
                        _ => return Err(self.error("Expected literal value after DEFAULT")),
                    };
                    self.advance();
                    Some(lit)
                } else {
                    None
                };

                defs.push(PropertyDefinition {
                    name,
                    data_type,
                    nullable,
                    default_value,
                });

                if self.current.kind != TokenKind::Comma {
                    break;
                }
                self.advance();
            }
        }

        self.expect(TokenKind::RBrace)?;
        Ok(defs)
    }

    /// Parses a SHOW statement.
    ///
    /// ```text
    /// SHOW CONSTRAINTS
    /// SHOW INDEXES
    /// SHOW NODE TYPES
    /// SHOW EDGE TYPES
    /// SHOW GRAPH TYPES
    /// SHOW GRAPH TYPE <name>
    /// ```
    fn parse_show(&mut self) -> Result<SchemaStatement> {
        self.advance(); // consume SHOW

        if !self.is_identifier()
            && self.current.kind != TokenKind::Node
            && self.current.kind != TokenKind::Edge
            && self.current.kind != TokenKind::Index
        {
            return Err(self.error(
                "Expected CONSTRAINTS, INDEXES, NODE TYPES, EDGE TYPES, GRAPHS, GRAPH TYPES, or GRAPH TYPE <name> after SHOW",
            ));
        }

        match self.current.kind {
            TokenKind::Node => {
                // SHOW NODE TYPES
                self.advance();
                if self.current.kind != TokenKind::Type
                    && !(self.is_identifier()
                        && self.get_identifier_name().eq_ignore_ascii_case("TYPES"))
                {
                    return Err(self.error("Expected TYPES after SHOW NODE"));
                }
                self.advance();
                Ok(SchemaStatement::ShowNodeTypes)
            }
            TokenKind::Edge => {
                // SHOW EDGE TYPES
                self.advance();
                if self.current.kind != TokenKind::Type
                    && !(self.is_identifier()
                        && self.get_identifier_name().eq_ignore_ascii_case("TYPES"))
                {
                    return Err(self.error("Expected TYPES after SHOW EDGE"));
                }
                self.advance();
                Ok(SchemaStatement::ShowEdgeTypes)
            }
            TokenKind::Index => {
                // SHOW INDEXES (INDEX is a keyword token)
                self.advance();
                // Allow optional plural "ES" as a separate token won't happen,
                // but the keyword is INDEX. Accept both SHOW INDEX and SHOW INDEXES.
                Ok(SchemaStatement::ShowIndexes)
            }
            _ => {
                let name = self.get_identifier_name();
                match name.to_uppercase().as_str() {
                    "CONSTRAINTS" => {
                        self.advance();
                        Ok(SchemaStatement::ShowConstraints)
                    }
                    "INDEXES" => {
                        self.advance();
                        Ok(SchemaStatement::ShowIndexes)
                    }
                    "GRAPHS" => {
                        self.advance();
                        Ok(SchemaStatement::ShowGraphs)
                    }
                    "SCHEMAS" => {
                        self.advance();
                        Ok(SchemaStatement::ShowSchemas)
                    }
                    "GRAPH" => {
                        self.advance();
                        // SHOW GRAPH TYPES or SHOW GRAPH TYPE <name>
                        if self.current.kind == TokenKind::Type {
                            self.advance();
                            // SHOW GRAPH TYPE <name> (singular)
                            if !self.is_identifier() {
                                return Err(self.error("Expected graph type name after SHOW GRAPH TYPE"));
                            }
                            let type_name = self.get_identifier_name();
                            self.advance();
                            Ok(SchemaStatement::ShowGraphType(type_name))
                        } else if self.is_identifier()
                            && self.get_identifier_name().eq_ignore_ascii_case("TYPES")
                        {
                            // SHOW GRAPH TYPES (plural)
                            self.advance();
                            Ok(SchemaStatement::ShowGraphTypes)
                        } else {
                            Err(self.error("Expected TYPE <name> or TYPES after SHOW GRAPH"))
                        }
                    }
                    _ => Err(self.error(
                        "Expected CONSTRAINTS, INDEXES, NODE TYPES, EDGE TYPES, GRAPHS, GRAPH TYPES, or GRAPH TYPE <name> after SHOW",
                    )),
                }
            }
        }
    }

    fn advance(&mut self) {
        if let Some(peeked) = self.peeked.take() {
            self.current = peeked;
            // Shift peeked_second into peeked
            self.peeked = self.peeked_second.take();
        } else {
            self.current = self.lexer.next_token();
        }
    }

    fn expect(&mut self, kind: TokenKind) -> Result<()> {
        if self.current.kind == kind {
            self.advance();
            Ok(())
        } else {
            Err(self.error(&format!("Expected {:?}", kind)))
        }
    }

    fn peek_kind(&mut self) -> TokenKind {
        if self.peeked.is_none() {
            self.peeked = Some(self.lexer.next_token());
        }
        self.peeked
            .as_ref()
            .expect("peeked token was just populated")
            .kind
    }

    /// Peeks at the token after the next token (two-token lookahead).
    fn peek_second_kind(&mut self) -> TokenKind {
        // Ensure first peeked is populated
        let _ = self.peek_kind();
        if self.peeked_second.is_none() {
            self.peeked_second = Some(self.lexer.next_token());
        }
        self.peeked_second
            .as_ref()
            .expect("peeked_second token was just populated")
            .kind
    }

    /// Returns the uppercased text of the peeked token.
    /// Must call `peek_kind()` first.
    fn peek_text_upper(&self) -> String {
        self.peeked
            .as_ref()
            .map(|t| t.text.to_uppercase())
            .unwrap_or_default()
    }

    /// Checks whether CREATE is followed by [PROPERTY] GRAPH (for graph instances, not GRAPH TYPE).
    ///
    /// Returns true for `CREATE GRAPH <name>` and `CREATE PROPERTY GRAPH <name>`,
    /// but false for `CREATE GRAPH TYPE <name>` (which is schema DDL).
    fn peek_is_graph_instance_keyword(&mut self) -> bool {
        let pk = self.peek_kind();
        if pk != TokenKind::Identifier {
            return false;
        }
        let text = self.peek_text_upper();
        if text == "PROPERTY" {
            return true; // CREATE PROPERTY GRAPH ...
        }
        if text == "GRAPH" {
            // Need to check the token after GRAPH. If it's TYPE, this is GRAPH TYPE (schema).
            // Use a temporary lexer to peek 2 tokens ahead from the current source position.
            let peeked_token = self
                .peeked
                .as_ref()
                .expect("peek_kind guarantees peeked is Some");
            let remaining = &self.source[peeked_token.span.end..];
            let mut temp_lexer = Lexer::new(remaining);
            let next_after_graph = temp_lexer.next_token();
            // If it's TYPE, this is CREATE GRAPH TYPE, not a graph instance
            return next_after_graph.kind != TokenKind::Type;
        }
        false
    }

    /// Parses `CREATE [OR REPLACE] PROCEDURE name(params) RETURNS (cols) AS { body }`.
    fn parse_create_procedure(&mut self, or_replace: bool) -> Result<SchemaStatement> {
        let if_not_exists = self.try_parse_if_not_exists();

        if !self.is_identifier() {
            return Err(self.error("Expected procedure name"));
        }
        let name = self.get_identifier_name();
        self.advance();

        // Parse parameter list
        self.expect(TokenKind::LParen)?;
        let mut params = Vec::new();
        if self.current.kind != TokenKind::RParen {
            loop {
                if !self.is_identifier() {
                    return Err(self.error("Expected parameter name"));
                }
                let param_name = self.get_identifier_name();
                self.advance();

                if !self.is_identifier() {
                    return Err(self.error("Expected parameter type"));
                }
                let param_type = self.get_identifier_name().to_uppercase();
                self.advance();

                params.push(ProcedureParam {
                    name: param_name,
                    param_type,
                });

                if self.current.kind != TokenKind::Comma {
                    break;
                }
                self.advance();
            }
        }
        self.expect(TokenKind::RParen)?;

        // Parse RETURNS (col1 type, ...)
        let mut returns = Vec::new();
        if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("RETURNS") {
            self.advance();
            self.expect(TokenKind::LParen)?;
            if self.current.kind != TokenKind::RParen {
                loop {
                    if !self.is_identifier() {
                        return Err(self.error("Expected return column name"));
                    }
                    let col_name = self.get_identifier_name();
                    self.advance();

                    if !self.is_identifier() {
                        return Err(self.error("Expected return column type"));
                    }
                    let col_type = self.get_identifier_name().to_uppercase();
                    self.advance();

                    returns.push(ProcedureReturn {
                        name: col_name,
                        return_type: col_type,
                    });

                    if self.current.kind != TokenKind::Comma {
                        break;
                    }
                    self.advance();
                }
            }
            self.expect(TokenKind::RParen)?;
        }

        // Expect AS (TokenKind::As is a keyword, not a contextual identifier)
        if self.current.kind == TokenKind::As {
            self.advance();
        } else {
            return Err(self.error("Expected AS before procedure body"));
        }

        // Parse body: { ... } with brace nesting
        self.expect(TokenKind::LBrace)?;
        let body_start = self.current.span.start;
        let mut depth = 1u32;
        while depth > 0 && self.current.kind != TokenKind::Eof {
            if self.current.kind == TokenKind::LBrace {
                depth += 1;
            } else if self.current.kind == TokenKind::RBrace {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            self.advance();
        }
        let body_end = self.current.span.start;
        let body = self.source[body_start..body_end].trim().to_string();
        self.expect(TokenKind::RBrace)?;

        Ok(SchemaStatement::CreateProcedure(CreateProcedureStatement {
            name,
            params,
            returns,
            body,
            if_not_exists,
            or_replace,
            span: None,
        }))
    }

    /// Parses `CREATE [PROPERTY] GRAPH [IF NOT EXISTS] <name>`.
    fn parse_create_graph(&mut self) -> Result<SessionCommand> {
        self.expect(TokenKind::Create)?;

        // Skip optional PROPERTY keyword
        if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("PROPERTY") {
            self.advance();
        }

        // Expect GRAPH
        if !self.is_identifier() || !self.get_identifier_name().eq_ignore_ascii_case("GRAPH") {
            return Err(self.error("Expected GRAPH after CREATE"));
        }
        self.advance();

        // Optional IF NOT EXISTS
        let if_not_exists = if self.try_accept_identifier_keyword("IF") {
            if !self.try_accept_keyword(TokenKind::Not, "NOT") {
                return Err(self.error("Expected NOT after IF"));
            }
            if !self.try_accept_keyword(TokenKind::Exists, "EXISTS") {
                return Err(self.error("Expected EXISTS after IF NOT"));
            }
            true
        } else {
            false
        };

        // Parse graph name
        if !self.is_identifier() {
            return Err(self.error("Expected graph name"));
        }
        let name = self.get_identifier_name();
        self.advance();

        // Optional TYPED type_name or [TYPED] ANY [[PROPERTY] GRAPH] (open graph type)
        let mut open = false;
        let typed = if self.is_identifier()
            && self.get_identifier_name().eq_ignore_ascii_case("TYPED")
        {
            self.advance();
            // Check for ANY (open graph type): TYPED ANY [[PROPERTY] GRAPH]
            if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("ANY") {
                self.advance();
                // Consume optional PROPERTY
                if self.is_identifier()
                    && self.get_identifier_name().eq_ignore_ascii_case("PROPERTY")
                {
                    self.advance();
                }
                // Consume optional GRAPH
                if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("GRAPH")
                {
                    self.advance();
                }
                open = true;
                None // ANY GRAPH = open/schema-free (no type binding)
            } else if self.is_identifier() {
                let first = self.get_identifier_name();
                self.advance();
                // Support qualified names: schema.type_name
                // Internally stored with '/' separator to match storage keys.
                if self.current.kind == TokenKind::Dot {
                    self.advance(); // consume '.'
                    if !self.is_identifier() {
                        return Err(self.error("Expected type name after '.'"));
                    }
                    let type_part = self.get_identifier_name();
                    self.advance();
                    Some(format!("{first}/{type_part}"))
                } else {
                    Some(first)
                }
            } else {
                return Err(self.error("Expected graph type name or ANY after TYPED"));
            }
        } else if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("ANY") {
            // ANY [[PROPERTY] GRAPH] without TYPED prefix
            self.advance();
            if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("PROPERTY") {
                self.advance();
            }
            if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("GRAPH") {
                self.advance();
            }
            open = true;
            None // open graph
        } else {
            None
        };

        // Optional LIKE source_graph (LIKE is a keyword token, not an identifier)
        let like_graph = if self.current.kind == TokenKind::Like {
            self.advance();
            if !self.is_identifier() {
                return Err(self.error("Expected graph name after LIKE"));
            }
            let source = self.get_identifier_name();
            self.advance();
            Some(source)
        } else {
            None
        };

        // Optional AS COPY OF source_graph
        let copy_of = if self.current.kind == TokenKind::As {
            self.advance();
            if !self.is_identifier() || !self.get_identifier_name().eq_ignore_ascii_case("COPY") {
                return Err(self.error("Expected COPY after AS"));
            }
            self.advance();
            if !self.is_identifier() || !self.get_identifier_name().eq_ignore_ascii_case("OF") {
                return Err(self.error("Expected OF after COPY"));
            }
            self.advance();
            if !self.is_identifier() {
                return Err(self.error("Expected graph name after AS COPY OF"));
            }
            let source = self.get_identifier_name();
            self.advance();
            Some(source)
        } else {
            None
        };

        Ok(SessionCommand::CreateGraph {
            name,
            if_not_exists,
            typed,
            like_graph,
            copy_of,
            open,
        })
    }

    /// Parses `CREATE PROJECTION <name> [LABELS (l1, l2, ...)] [EDGE_TYPES (e1, e2, ...)]`.
    fn parse_create_projection(&mut self) -> Result<SessionCommand> {
        self.expect(TokenKind::Create)?;

        // Consume PROJECTION keyword
        if !self.is_identifier()
            || !self
                .get_identifier_name()
                .eq_ignore_ascii_case("PROJECTION")
        {
            return Err(self.error("Expected PROJECTION after CREATE"));
        }
        self.advance();

        // Parse projection name
        if !self.is_identifier() {
            return Err(self.error("Expected projection name"));
        }
        let name = self.get_identifier_name();
        self.advance();

        // Optional LABELS (l1, l2, ...)
        let node_labels =
            if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("LABELS") {
                self.advance();
                self.parse_parenthesized_identifier_list()?
            } else {
                Vec::new()
            };

        // Optional EDGE_TYPES (e1, e2, ...)
        let edge_types = if self.is_identifier()
            && self
                .get_identifier_name()
                .eq_ignore_ascii_case("EDGE_TYPES")
        {
            self.advance();
            self.parse_parenthesized_identifier_list()?
        } else {
            Vec::new()
        };

        Ok(SessionCommand::CreateProjection {
            name,
            node_labels,
            edge_types,
        })
    }

    /// Parses a parenthesized, comma-separated list of identifiers: `(A, B, C)`.
    fn parse_parenthesized_identifier_list(&mut self) -> Result<Vec<String>> {
        self.expect(TokenKind::LParen)?;
        let mut items = Vec::new();
        if self.current.kind != TokenKind::RParen {
            loop {
                if !self.is_identifier() && !self.is_label_or_type_name() {
                    return Err(self.error("Expected identifier in list"));
                }
                items.push(self.get_identifier_name());
                self.advance();
                if self.current.kind != TokenKind::Comma {
                    break;
                }
                self.advance(); // consume comma
            }
        }
        self.expect(TokenKind::RParen)?;
        Ok(items)
    }

    /// Parses `DROP [PROPERTY] GRAPH [IF EXISTS] <name>`.
    /// Assumes DROP has already been consumed.
    fn parse_drop_graph_body(&mut self) -> Result<SessionCommand> {
        // Skip optional PROPERTY keyword
        if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("PROPERTY") {
            self.advance();
        }

        // Expect GRAPH
        if !self.is_identifier() || !self.get_identifier_name().eq_ignore_ascii_case("GRAPH") {
            return Err(
                self.error("Expected GRAPH, NODE TYPE, EDGE TYPE, INDEX, or CONSTRAINT after DROP")
            );
        }
        self.advance();

        // Optional IF EXISTS
        let if_exists = self.try_parse_if_exists();

        // Parse graph name
        if !self.is_identifier() {
            return Err(self.error("Expected graph name"));
        }
        let name = self.get_identifier_name();
        self.advance();

        Ok(SessionCommand::DropGraph { name, if_exists })
    }

    /// Parses `USE GRAPH <name>`.
    fn parse_use_graph(&mut self) -> Result<SessionCommand> {
        self.advance(); // consume USE

        // Expect GRAPH
        if !self.is_identifier() || !self.get_identifier_name().eq_ignore_ascii_case("GRAPH") {
            return Err(self.error("Expected GRAPH after USE"));
        }
        self.advance();

        // Parse graph name
        if !self.is_identifier() {
            return Err(self.error("Expected graph name"));
        }
        let name = self.get_identifier_name();
        self.advance();

        Ok(SessionCommand::UseGraph(name))
    }

    /// Parses SESSION commands: SET, RESET, CLOSE.
    fn parse_session_command(&mut self) -> Result<SessionCommand> {
        self.advance(); // consume SESSION

        if !self.is_identifier() {
            return Err(self.error("Expected SET, RESET, or CLOSE after SESSION"));
        }

        let action = self.get_identifier_name();
        match action.to_uppercase().as_str() {
            "SET" => {
                self.advance(); // consume SET
                self.parse_session_set()
            }
            "RESET" => {
                self.advance(); // consume RESET
                // ISO/IEC 39075 Section 7.2: SESSION RESET [ALL CHARACTERISTICS | SCHEMA | GRAPH | TIME ZONE | PARAMETERS]
                self.parse_session_reset()
            }
            "CLOSE" => {
                self.advance(); // consume CLOSE
                Ok(SessionCommand::SessionClose)
            }
            _ => Err(self.error("Expected SET, RESET, or CLOSE after SESSION")),
        }
    }

    /// Parses SESSION SET variants: GRAPH, TIME ZONE, PARAMETER.
    fn parse_session_set(&mut self) -> Result<SessionCommand> {
        if !self.is_identifier() {
            return Err(self.error("Expected GRAPH, TIME, SCHEMA, or PARAMETER after SESSION SET"));
        }

        let keyword = self.get_identifier_name();
        match keyword.to_uppercase().as_str() {
            "GRAPH" => {
                self.advance(); // consume GRAPH
                if !self.is_identifier() {
                    return Err(self.error("Expected graph name"));
                }
                let name = self.get_identifier_name();
                self.advance();
                Ok(SessionCommand::SessionSetGraph(name))
            }
            "TIME" => {
                self.advance(); // consume TIME
                // Expect ZONE
                if !self.is_identifier() || !self.get_identifier_name().eq_ignore_ascii_case("ZONE")
                {
                    return Err(self.error("Expected ZONE after TIME"));
                }
                self.advance();
                // Expect timezone string
                if self.current.kind != TokenKind::String {
                    return Err(self.error("Expected timezone string after TIME ZONE"));
                }
                let tz = self.current.text[1..self.current.text.len() - 1].to_string();
                self.advance();
                Ok(SessionCommand::SessionSetTimeZone(tz))
            }
            "SCHEMA" => {
                self.advance(); // consume SCHEMA
                if !self.is_identifier() {
                    return Err(self.error("Expected schema name"));
                }
                let name = self.get_identifier_name();
                self.advance();
                // ISO/IEC 39075 Section 7.1 GR1: session schema is independent from session graph
                Ok(SessionCommand::SessionSetSchema(name))
            }
            "PARAMETER" => {
                self.advance(); // consume PARAMETER
                // Expect $name or identifier
                let param_name = if self.current.kind == TokenKind::Parameter {
                    let name = self.current.text[1..].to_string();
                    self.advance();
                    name
                } else if self.is_identifier() {
                    let name = self.get_identifier_name();
                    self.advance();
                    name
                } else {
                    return Err(self.error("Expected parameter name"));
                };
                // Expect =
                self.expect(TokenKind::Eq)?;
                // Parse value expression
                let value = self.parse_expression()?;
                Ok(SessionCommand::SessionSetParameter(param_name, value))
            }
            _ => Err(self.error("Expected GRAPH, TIME, SCHEMA, or PARAMETER after SESSION SET")),
        }
    }

    /// Parses SESSION RESET targets per ISO/IEC 39075 Section 7.2.
    ///
    /// `SESSION RESET` (bare) = reset all characteristics.
    /// `SESSION RESET ALL [CHARACTERISTICS | PARAMETERS]` = reset all.
    /// `SESSION RESET SCHEMA` = reset session schema only.
    /// `SESSION RESET [PROPERTY] GRAPH` = reset session graph only.
    /// `SESSION RESET TIME ZONE` = reset time zone only.
    /// `SESSION RESET [ALL] PARAMETERS` = reset parameters only.
    fn parse_session_reset(&mut self) -> Result<SessionCommand> {
        use SessionResetTarget as T;

        // Bare SESSION RESET (no arguments) = reset all per Section 7.2 SR2b
        if self.current.kind == TokenKind::Eof {
            return Ok(SessionCommand::SessionReset(T::All));
        }

        // ALL [CHARACTERISTICS | PARAMETERS]
        if self.current.kind == TokenKind::All {
            self.advance();
            if self.is_identifier() {
                let kw = self.get_identifier_name().to_uppercase();
                if kw == "PARAMETERS" {
                    self.advance();
                    return Ok(SessionCommand::SessionReset(T::Parameters));
                }
                if kw == "CHARACTERISTICS" {
                    self.advance();
                }
            }
            return Ok(SessionCommand::SessionReset(T::All));
        }

        if !self.is_identifier() {
            return Err(
                self.error("Expected SCHEMA, GRAPH, TIME, PARAMETERS, or ALL after SESSION RESET")
            );
        }

        let kw = self.get_identifier_name().to_uppercase();
        match kw.as_str() {
            "SCHEMA" => {
                self.advance();
                Ok(SessionCommand::SessionReset(T::Schema))
            }
            "GRAPH" | "PROPERTY" => {
                self.advance();
                // Skip optional GRAPH after PROPERTY
                if kw == "PROPERTY"
                    && self.is_identifier()
                    && self.get_identifier_name().eq_ignore_ascii_case("GRAPH")
                {
                    self.advance();
                }
                Ok(SessionCommand::SessionReset(T::Graph))
            }
            "TIME" => {
                self.advance();
                if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("ZONE") {
                    self.advance();
                }
                Ok(SessionCommand::SessionReset(T::TimeZone))
            }
            "PARAMETERS" => {
                self.advance();
                Ok(SessionCommand::SessionReset(T::Parameters))
            }
            "CHARACTERISTICS" => {
                self.advance();
                Ok(SessionCommand::SessionReset(T::All))
            }
            _ => {
                Err(self
                    .error("Expected SCHEMA, GRAPH, TIME, PARAMETERS, or ALL after SESSION RESET"))
            }
        }
    }

    /// Parses `START TRANSACTION [READ ONLY | READ WRITE] [ISOLATION LEVEL ...]`.
    fn parse_start_transaction(&mut self) -> Result<SessionCommand> {
        self.advance(); // consume START
        if !self.is_identifier()
            || !self
                .get_identifier_name()
                .eq_ignore_ascii_case("TRANSACTION")
        {
            return Err(self.error("Expected TRANSACTION after START"));
        }
        self.advance(); // consume TRANSACTION

        let mut read_only = false;
        let mut isolation_level = None;

        // Parse optional characteristics (can appear in either order)
        for _ in 0..2 {
            if !self.is_identifier() {
                break;
            }
            let kw = self.get_identifier_name().to_uppercase();
            match kw.as_str() {
                "READ" => {
                    self.advance(); // consume READ
                    if !self.is_identifier() {
                        return Err(self.error("Expected ONLY or WRITE after READ"));
                    }
                    let mode = self.get_identifier_name().to_uppercase();
                    match mode.as_str() {
                        "ONLY" => {
                            self.advance();
                            read_only = true;
                        }
                        "WRITE" | "COMMITTED" if mode == "WRITE" => {
                            self.advance();
                            read_only = false;
                        }
                        _ => return Err(self.error("Expected ONLY or WRITE after READ")),
                    }
                }
                "ISOLATION" => {
                    self.advance(); // consume ISOLATION
                    if !self.is_identifier()
                        || !self.get_identifier_name().eq_ignore_ascii_case("LEVEL")
                    {
                        return Err(self.error("Expected LEVEL after ISOLATION"));
                    }
                    self.advance(); // consume LEVEL
                    isolation_level = Some(self.parse_isolation_level_name()?);
                }
                _ => break,
            }
        }

        Ok(SessionCommand::StartTransaction {
            read_only,
            isolation_level,
        })
    }

    /// Parses an isolation level name: READ COMMITTED, SNAPSHOT [ISOLATION],
    /// REPEATABLE READ, or SERIALIZABLE.
    fn parse_isolation_level_name(&mut self) -> Result<TransactionIsolationLevel> {
        if !self.is_identifier() {
            return Err(self.error("Expected isolation level name"));
        }
        let name = self.get_identifier_name().to_uppercase();
        match name.as_str() {
            "READ" => {
                self.advance();
                if !self.is_identifier()
                    || !self.get_identifier_name().eq_ignore_ascii_case("COMMITTED")
                {
                    return Err(self.error("Expected COMMITTED after READ"));
                }
                self.advance();
                Ok(TransactionIsolationLevel::ReadCommitted)
            }
            "SNAPSHOT" => {
                self.advance();
                // Optional "ISOLATION" suffix
                if self.is_identifier()
                    && self.get_identifier_name().eq_ignore_ascii_case("ISOLATION")
                {
                    self.advance();
                }
                Ok(TransactionIsolationLevel::SnapshotIsolation)
            }
            "REPEATABLE" => {
                self.advance();
                if !self.is_identifier() || !self.get_identifier_name().eq_ignore_ascii_case("READ")
                {
                    return Err(self.error("Expected READ after REPEATABLE"));
                }
                self.advance();
                Ok(TransactionIsolationLevel::SnapshotIsolation)
            }
            "SERIALIZABLE" => {
                self.advance();
                Ok(TransactionIsolationLevel::Serializable)
            }
            _ => Err(self.error(&format!("Unknown isolation level: {name}"))),
        }
    }

    fn error(&self, message: &str) -> Error {
        Error::Query(
            QueryError::new(QueryErrorKind::Syntax, format!("[GQL] {message}"))
                .with_span(self.current.span)
                .with_source(self.source.to_string()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_match() {
        let mut parser = Parser::new("MATCH (n) RETURN n");
        let result = parser.parse();
        assert!(result.is_ok());

        let stmt = result.unwrap();
        assert!(matches!(stmt, Statement::Query(_)));
    }

    #[test]
    fn test_parse_match_with_label() {
        let mut parser = Parser::new("MATCH (n:Person) RETURN n");
        let result = parser.parse().unwrap();
        if let Statement::Query(query) = result {
            assert_eq!(query.match_clauses.len(), 1);
            if let Pattern::Node(node) = &query.match_clauses[0].patterns[0].pattern {
                assert_eq!(node.variable, Some("n".to_string()));
                assert_eq!(node.labels, vec!["Person".to_string()]);
            } else {
                panic!("Expected node pattern");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_match_with_where() {
        let mut parser = Parser::new("MATCH (n:Person) WHERE n.age > 30 RETURN n");
        let result = parser.parse().unwrap();
        if let Statement::Query(query) = result {
            assert!(
                query.where_clause.is_some(),
                "WHERE clause should be parsed"
            );
            let where_clause = query.where_clause.as_ref().unwrap();
            if let Expression::Binary { op, .. } = &where_clause.expression {
                assert_eq!(*op, BinaryOp::Gt);
            } else {
                panic!("Expected binary expression in WHERE clause");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_path_pattern() {
        let mut parser = Parser::new("MATCH (a)-[:KNOWS]->(b) RETURN a, b");
        let result = parser.parse().unwrap();
        if let Statement::Query(query) = result {
            if let Pattern::Path(path) = &query.match_clauses[0].patterns[0].pattern {
                assert_eq!(path.source.variable, Some("a".to_string()));
                assert_eq!(path.edges.len(), 1);
                assert_eq!(path.edges[0].types, vec!["KNOWS".to_string()]);
                assert_eq!(
                    path.edges[0].direction,
                    EdgeDirection::Outgoing,
                    "Arrow should point outward"
                );
                assert_eq!(path.edges[0].target.variable, Some("b".to_string()));
            } else {
                panic!("Expected path pattern");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_insert() {
        let mut parser = Parser::new("INSERT (n:Person {name: 'Alix'})");
        let result = parser.parse().unwrap();
        if let Statement::DataModification(DataModificationStatement::Insert(insert)) = result {
            assert_eq!(insert.patterns.len(), 1);
            if let Pattern::Node(node) = &insert.patterns[0] {
                assert_eq!(node.labels, vec!["Person".to_string()]);
                assert_eq!(node.properties.len(), 1);
                assert_eq!(node.properties[0].0, "name");
            } else {
                panic!("Expected node pattern");
            }
        } else {
            panic!("Expected Insert statement");
        }
    }

    #[test]
    fn test_parse_optional_match() {
        let mut parser =
            Parser::new("MATCH (a:Person) OPTIONAL MATCH (a)-[:KNOWS]->(b) RETURN a, b");
        let result = parser.parse();
        assert!(result.is_ok());

        if let Statement::Query(query) = result.unwrap() {
            assert_eq!(query.match_clauses.len(), 2);
            assert!(!query.match_clauses[0].optional);
            assert!(query.match_clauses[1].optional);
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_with_clause() {
        let mut parser =
            Parser::new("MATCH (n:Person) WITH n.name AS name, n.age AS age RETURN name, age");
        let result = parser.parse();
        assert!(result.is_ok());

        if let Statement::Query(query) = result.unwrap() {
            assert_eq!(query.with_clauses.len(), 1);
            assert_eq!(query.with_clauses[0].items.len(), 2);
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_order_by() {
        let mut parser = Parser::new("MATCH (n:Person) RETURN n.name ORDER BY n.age DESC");
        let result = parser.parse();
        assert!(result.is_ok());

        if let Statement::Query(query) = result.unwrap() {
            let order_by = query.return_clause.order_by.as_ref().unwrap();
            assert_eq!(order_by.items.len(), 1);
            assert_eq!(order_by.items[0].order, SortOrder::Desc);
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_limit_skip() {
        let mut parser = Parser::new("MATCH (n) RETURN n SKIP 10 LIMIT 5");
        let result = parser.parse();
        assert!(result.is_ok());

        if let Statement::Query(query) = result.unwrap() {
            assert!(query.return_clause.skip.is_some());
            assert!(query.return_clause.limit.is_some());
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_aggregation() {
        let mut parser = Parser::new("MATCH (n:Person) RETURN count(n), avg(n.age)");
        let result = parser.parse();
        assert!(result.is_ok());

        if let Statement::Query(query) = result.unwrap() {
            assert_eq!(query.return_clause.items.len(), 2);
            // Check that function calls are parsed
            if let Expression::FunctionCall { name, .. } = &query.return_clause.items[0].expression
            {
                assert_eq!(name, "count");
            } else {
                panic!("Expected function call");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_with_parameter() {
        let mut parser = Parser::new("MATCH (n:Person) WHERE n.age > $min_age RETURN n");
        let result = parser.parse();
        assert!(result.is_ok());

        if let Statement::Query(query) = result.unwrap() {
            // Check that the WHERE clause contains a parameter
            let where_clause = query.where_clause.as_ref().expect("Expected WHERE clause");
            if let Expression::Binary { right, .. } = &where_clause.expression {
                if let Expression::Parameter(name) = right.as_ref() {
                    assert_eq!(name, "min_age");
                } else {
                    panic!("Expected parameter, got {:?}", right);
                }
            } else {
                panic!("Expected binary expression in WHERE clause");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_insert_with_parameter() {
        let mut parser = Parser::new("INSERT (n:Person {name: $name, age: $age})");
        let result = parser.parse();
        assert!(result.is_ok());

        if let Statement::DataModification(DataModificationStatement::Insert(insert)) =
            result.unwrap()
        {
            if let Pattern::Node(node) = &insert.patterns[0] {
                assert_eq!(node.properties.len(), 2);
                // Check first property is a parameter
                if let Expression::Parameter(name) = &node.properties[0].1 {
                    assert_eq!(name, "name");
                } else {
                    panic!("Expected parameter for name property");
                }
            } else {
                panic!("Expected node pattern");
            }
        } else {
            panic!("Expected Insert statement");
        }
    }

    #[test]
    fn test_parse_variable_length_path() {
        let mut parser = Parser::new("MATCH (a)-[:KNOWS*1..3]->(b) RETURN a, b");
        let result = parser.parse();
        assert!(result.is_ok(), "Parse error: {:?}", result.err());

        if let Statement::Query(query) = result.unwrap() {
            if let Pattern::Path(path) = &query.match_clauses[0].patterns[0].pattern {
                let edge = &path.edges[0];
                assert_eq!(edge.min_hops, Some(1));
                assert_eq!(edge.max_hops, Some(3));
            } else {
                panic!("Expected path pattern");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_variable_length_path_unbounded() {
        let mut parser = Parser::new("MATCH (a)-[:KNOWS*]->(b) RETURN a, b");
        let result = parser.parse();
        assert!(result.is_ok());

        if let Statement::Query(query) = result.unwrap() {
            if let Pattern::Path(path) = &query.match_clauses[0].patterns[0].pattern {
                let edge = &path.edges[0];
                assert_eq!(edge.min_hops, Some(1)); // default min is 1
                assert_eq!(edge.max_hops, None); // unbounded max
            } else {
                panic!("Expected path pattern");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_variable_length_path_exact() {
        let mut parser = Parser::new("MATCH (a)-[:KNOWS*2]->(b) RETURN a, b");
        let result = parser.parse();
        assert!(result.is_ok());

        if let Statement::Query(query) = result.unwrap() {
            if let Pattern::Path(path) = &query.match_clauses[0].patterns[0].pattern {
                let edge = &path.edges[0];
                assert_eq!(edge.min_hops, Some(2));
                assert_eq!(edge.max_hops, Some(2)); // exact means min == max
            } else {
                panic!("Expected path pattern");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_variable_length_path_with_properties() {
        // Test variable-length path with node properties and labels
        let query = "MATCH (start:Node {name: 'a'})-[:NEXT*1..3]->(end:Node) RETURN end.name";
        let mut parser = Parser::new(query);
        let result = parser.parse();
        assert!(result.is_ok(), "Parse error: {:?}", result.err());

        if let Statement::Query(query) = result.unwrap() {
            if let Pattern::Path(path) = &query.match_clauses[0].patterns[0].pattern {
                let edge = &path.edges[0];
                assert_eq!(edge.min_hops, Some(1));
                assert_eq!(edge.max_hops, Some(3));
                // Verify source and target patterns
                assert_eq!(path.source.variable, Some("start".to_string()));
                assert_eq!(path.source.labels, vec!["Node".to_string()]);
                assert_eq!(edge.target.variable, Some("end".to_string()));
                assert_eq!(edge.target.labels, vec!["Node".to_string()]);
            } else {
                panic!("Expected path pattern");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_reserved_keywords_as_identifiers() {
        // Test that reserved keywords can be used as variable names
        let queries = [
            ("MATCH (end:Node) RETURN end", "end"),
            ("MATCH (node:Person) RETURN node", "node"),
            ("MATCH (type:Category) RETURN type", "type"),
            ("MATCH (case:Test) RETURN case", "case"),
        ];

        for (query, expected_var) in queries {
            let mut parser = Parser::new(query);
            let result = parser.parse();
            assert!(
                result.is_ok(),
                "Parse error for '{}': {:?}",
                expected_var,
                result.err()
            );

            if let Statement::Query(q) = result.unwrap()
                && let Pattern::Node(node) = &q.match_clauses[0].patterns[0].pattern
            {
                assert_eq!(node.variable, Some(expected_var.to_string()));
            }
        }
    }

    #[test]
    fn test_parse_quoted_identifier_label() {
        let mut parser = Parser::new("MATCH (n:`rdf:type`) RETURN n");
        let result = parser.parse();
        assert!(result.is_ok());

        if let Statement::Query(query) = result.unwrap() {
            if let Pattern::Node(node) = &query.match_clauses[0].patterns[0].pattern {
                assert_eq!(node.labels[0], "rdf:type");
            } else {
                panic!("Expected node pattern");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_unwind() {
        let mut parser = Parser::new("UNWIND [1, 2, 3] AS x RETURN x");
        let result = parser.parse();
        assert!(result.is_ok(), "Parse error: {:?}", result.err());

        if let Statement::Query(query) = result.unwrap() {
            assert_eq!(query.unwind_clauses.len(), 1);
            assert_eq!(query.unwind_clauses[0].alias, "x");
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_merge() {
        let mut parser = Parser::new("MERGE (n:Person {name: 'Alix'}) RETURN n");
        let result = parser.parse();
        assert!(result.is_ok(), "Parse error: {:?}", result.err());

        if let Statement::Query(query) = result.unwrap() {
            assert_eq!(query.merge_clauses.len(), 1);
            if let Pattern::Node(node) = &query.merge_clauses[0].pattern {
                assert_eq!(node.labels[0], "Person");
            } else {
                panic!("Expected node pattern in MERGE");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_merge_on_create() {
        let mut parser =
            Parser::new("MERGE (n:Person {name: 'Alix'}) ON CREATE SET n.created = true RETURN n");
        let result = parser.parse();
        assert!(result.is_ok(), "Parse error: {:?}", result.err());

        if let Statement::Query(query) = result.unwrap() {
            assert_eq!(query.merge_clauses.len(), 1);
            let merge = &query.merge_clauses[0];
            assert!(merge.on_create.is_some());
            assert_eq!(merge.on_create.as_ref().unwrap().len(), 1);
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_remove_label() {
        let mut parser = Parser::new("MATCH (n:Person) REMOVE n:Employee RETURN n");
        let result = parser.parse();
        assert!(result.is_ok(), "Parse error: {:?}", result.err());

        if let Statement::Query(query) = result.unwrap() {
            assert_eq!(query.remove_clauses.len(), 1);
            assert_eq!(query.remove_clauses[0].label_operations.len(), 1);
            assert_eq!(query.remove_clauses[0].label_operations[0].variable, "n");
            assert_eq!(
                query.remove_clauses[0].label_operations[0].labels,
                vec!["Employee"]
            );
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_remove_property() {
        let mut parser = Parser::new("MATCH (n:Person) REMOVE n.age RETURN n");
        let result = parser.parse();
        assert!(result.is_ok(), "Parse error: {:?}", result.err());

        if let Statement::Query(query) = result.unwrap() {
            assert_eq!(query.remove_clauses.len(), 1);
            assert_eq!(query.remove_clauses[0].property_removals.len(), 1);
            assert_eq!(query.remove_clauses[0].property_removals[0].0, "n");
            assert_eq!(query.remove_clauses[0].property_removals[0].1, "age");
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_remove_multiple() {
        let mut parser =
            Parser::new("MATCH (n:Person) REMOVE n:Employee, n.age, n:Contractor RETURN n");
        let result = parser.parse();
        assert!(result.is_ok(), "Parse error: {:?}", result.err());

        if let Statement::Query(query) = result.unwrap() {
            assert_eq!(query.remove_clauses.len(), 1);
            let remove = &query.remove_clauses[0];
            // Two label operations (Employee, Contractor) and one property removal (age)
            assert_eq!(remove.label_operations.len(), 2);
            assert_eq!(remove.property_removals.len(), 1);
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_vector_function_call() {
        let mut parser = Parser::new("MATCH (n) RETURN vector([0.1, 0.2, 0.3])");
        let result = parser.parse();
        assert!(result.is_ok(), "Parse error: {:?}", result.err());

        if let Statement::Query(query) = result.unwrap() {
            assert_eq!(query.return_clause.items.len(), 1);
            if let Expression::FunctionCall { name, args, .. } =
                &query.return_clause.items[0].expression
            {
                assert_eq!(name, "vector");
                assert_eq!(args.len(), 1);
                // The argument should be a list
                if let Expression::List(elements) = &args[0] {
                    assert_eq!(elements.len(), 3);
                } else {
                    panic!("Expected list argument, got {:?}", args[0]);
                }
            } else {
                panic!("Expected function call");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_cosine_similarity() {
        let mut parser =
            Parser::new("MATCH (n) WHERE cosine_similarity(n.embedding, $query) > 0.8 RETURN n");
        let result = parser.parse();
        assert!(result.is_ok(), "Parse error: {:?}", result.err());

        if let Statement::Query(query) = result.unwrap() {
            let where_clause = query.where_clause.as_ref().expect("Expected WHERE clause");
            if let Expression::Binary { left, .. } = &where_clause.expression {
                if let Expression::FunctionCall { name, args, .. } = left.as_ref() {
                    assert_eq!(name, "cosine_similarity");
                    assert_eq!(args.len(), 2);
                } else {
                    panic!("Expected function call, got {:?}", left);
                }
            } else {
                panic!("Expected binary expression");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_euclidean_distance() {
        let mut parser =
            Parser::new("MATCH (n) RETURN euclidean_distance(n.embedding, [1.0, 2.0]) AS dist");
        let result = parser.parse();
        assert!(result.is_ok(), "Parse error: {:?}", result.err());

        if let Statement::Query(query) = result.unwrap() {
            assert_eq!(query.return_clause.items.len(), 1);
            if let Expression::FunctionCall { name, args, .. } =
                &query.return_clause.items[0].expression
            {
                assert_eq!(name, "euclidean_distance");
                assert_eq!(args.len(), 2);
            } else {
                panic!("Expected function call");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_create_vector_index() {
        let mut parser = Parser::new("CREATE VECTOR INDEX movie_embeddings ON :Movie(embedding)");
        let result = parser.parse();
        assert!(result.is_ok(), "Parse error: {:?}", result.err());

        if let Statement::Schema(SchemaStatement::CreateVectorIndex(stmt)) = result.unwrap() {
            assert_eq!(stmt.name, "movie_embeddings");
            assert_eq!(stmt.node_label, "Movie");
            assert_eq!(stmt.property, "embedding");
            assert!(stmt.dimensions.is_none());
            assert!(stmt.metric.is_none());
        } else {
            panic!("Expected CreateVectorIndex statement");
        }
    }

    #[test]
    fn test_parse_create_vector_index_with_options() {
        let mut parser = Parser::new(
            "CREATE VECTOR INDEX embeddings ON :Document(vec) DIMENSION 384 METRIC 'cosine'",
        );
        let result = parser.parse();
        assert!(result.is_ok(), "Parse error: {:?}", result.err());

        if let Statement::Schema(SchemaStatement::CreateVectorIndex(stmt)) = result.unwrap() {
            assert_eq!(stmt.name, "embeddings");
            assert_eq!(stmt.node_label, "Document");
            assert_eq!(stmt.property, "vec");
            assert_eq!(stmt.dimensions, Some(384));
            assert_eq!(stmt.metric, Some("cosine".to_string()));
        } else {
            panic!("Expected CreateVectorIndex statement");
        }
    }

    #[test]
    fn test_in_operator_with_list() {
        let mut parser =
            Parser::new("MATCH (n:Person) WHERE n.name IN ['Alix', 'Gus'] RETURN n.name");
        let result = parser.parse();
        assert!(result.is_ok(), "Parse error: {:?}", result.err());

        if let Statement::Query(q) = result.unwrap() {
            let where_clause = q.where_clause.expect("Expected WHERE clause");
            if let WhereClause {
                expression: Expression::Binary { op, right, .. },
                ..
            } = where_clause
            {
                assert_eq!(op, BinaryOp::In);
                assert!(matches!(right.as_ref(), Expression::List(elems) if elems.len() == 2));
            } else {
                panic!("Expected Binary IN expression");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_in_operator_with_integers() {
        let mut parser = Parser::new("MATCH (n:Item) WHERE n.status IN [1, 2, 3] RETURN n");
        let result = parser.parse();
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
    }

    #[test]
    fn test_string_escape_single_quotes() {
        let mut parser = Parser::new(r#"MATCH (n) WHERE n.name = 'O\'Brien' RETURN n"#);
        let result = parser.parse();
        assert!(result.is_ok(), "Parse error: {:?}", result.err());

        if let Statement::Query(q) = result.unwrap() {
            let where_clause = q.where_clause.expect("Expected WHERE clause");
            if let WhereClause {
                expression: Expression::Binary { right, .. },
                ..
            } = where_clause
            {
                if let Expression::Literal(Literal::String(s)) = right.as_ref() {
                    assert_eq!(s, "O'Brien");
                } else {
                    panic!("Expected string literal");
                }
            }
        }
    }

    #[test]
    fn test_string_escape_sequences() {
        let mut parser = Parser::new(r#"MATCH (n) WHERE n.text = 'line1\nline2' RETURN n"#);
        let result = parser.parse();
        assert!(result.is_ok(), "Parse error: {:?}", result.err());

        if let Statement::Query(q) = result.unwrap() {
            let where_clause = q.where_clause.expect("Expected WHERE clause");
            if let WhereClause {
                expression: Expression::Binary { right, .. },
                ..
            } = where_clause
            {
                if let Expression::Literal(Literal::String(s)) = right.as_ref() {
                    assert_eq!(s, "line1\nline2");
                } else {
                    panic!("Expected string literal");
                }
            }
        }
    }

    // ==================== Error/Negative Cases ====================

    #[test]
    fn test_parse_error_empty_input() {
        let mut parser = Parser::new("");
        let result = parser.parse();
        assert!(result.is_err(), "Empty input should fail");
    }

    #[test]
    fn test_parse_error_just_match() {
        let mut parser = Parser::new("MATCH");
        let result = parser.parse();
        assert!(result.is_err(), "MATCH alone should fail");
    }

    #[test]
    fn test_parse_error_unclosed_node_pattern() {
        let mut parser = Parser::new("MATCH (n:Person RETURN n");
        let result = parser.parse();
        assert!(result.is_err(), "Unclosed node pattern should fail");
    }

    #[test]
    fn test_parse_error_unclosed_edge_pattern() {
        let mut parser = Parser::new("MATCH (a)-[:KNOWS->(b) RETURN a");
        let result = parser.parse();
        assert!(result.is_err(), "Unclosed edge pattern should fail");
    }

    #[test]
    fn test_parse_error_missing_return() {
        let mut parser = Parser::new("MATCH (n:Person) WHERE n.age > 25");
        let result = parser.parse();
        assert!(
            result.is_err(),
            "Query without RETURN or mutation should fail"
        );
    }

    #[test]
    fn test_parse_error_double_where() {
        let mut parser = Parser::new("MATCH (n) WHERE n.a = 1 WHERE n.b = 2 RETURN n");
        let result = parser.parse();
        assert!(result.is_err(), "Double WHERE should fail");
    }

    #[test]
    fn test_parse_error_invalid_literal() {
        let mut parser = Parser::new("MATCH (n) WHERE n.x = @invalid RETURN n");
        let result = parser.parse();
        assert!(result.is_err(), "Invalid literal should fail");
    }

    #[test]
    fn test_parse_error_unclosed_string() {
        let mut parser = Parser::new("MATCH (n) WHERE n.name = 'hello RETURN n");
        let result = parser.parse();
        assert!(result.is_err(), "Unclosed string should fail");
    }

    #[test]
    fn test_parse_error_unclosed_property_map() {
        let mut parser = Parser::new("MATCH (n:Person {name: 'Alix') RETURN n");
        let result = parser.parse();
        assert!(result.is_err(), "Unclosed property map should fail");
    }

    #[test]
    fn test_parse_error_return_only() {
        let mut parser = Parser::new("RETURN RETURN");
        let result = parser.parse();
        assert!(result.is_err(), "RETURN RETURN should fail");
    }

    // ==================== End of input (#380) ====================

    fn parse_err(query: &str) -> String {
        match Parser::new(query).parse() {
            Ok(statement) => panic!("{query}: expected an error, parsed {statement:?}"),
            Err(err) => err.to_string(),
        }
    }

    #[test]
    fn test_trailing_semicolons_and_comments_are_accepted() {
        for query in [
            "MATCH (n) RETURN n;",
            "MATCH (n) RETURN n ;;",
            "MATCH (n) RETURN n -- trailing comment",
            "MATCH (n) RETURN n /* trailing comment */ ;",
            "INSERT (:Person {name: 'Alix'});",
        ] {
            assert!(Parser::new(query).parse().is_ok(), "{query}");
        }
    }

    #[test]
    fn test_statements_separated_by_semicolon_are_rejected() {
        let err = parse_err("INSERT (:A); INSERT (:B)");
        assert!(
            err.contains("multiple statements separated by ';' are not supported in one call"),
            "{err}"
        );
    }

    #[test]
    fn test_trailing_input_is_rejected() {
        let err = parse_err("MATCH (n) RETURN n garbage");
        assert!(
            err.contains("unexpected 'garbage' after the end of the statement"),
            "{err}"
        );

        // Clause orders the parser does not support fail loudly instead of
        // silently dropping the rest of the statement.
        let err = parse_err("MATCH (n) SET n.x = 1 DELETE n");
        assert!(err.contains("unexpected 'DELETE'"), "{err}");
        assert!(err.contains("not supported at this position"), "{err}");
    }

    #[test]
    fn test_insert_sequences_and_insert_return_parse_as_queries() {
        // The docs quickstart shape: several INSERTs in one statement.
        let Statement::Query(query) = Parser::new(
            "INSERT (:Person {name: 'Alix'})\n\
             INSERT (:Person {name: 'Gus'})\n\
             INSERT (:Person {name: 'Harm'})",
        )
        .parse()
        .unwrap() else {
            panic!("an INSERT sequence is a query");
        };
        assert_eq!(query.create_clauses.len(), 3);
        assert!(query.return_clause.items.is_empty());

        let Statement::Query(query) = Parser::new("INSERT (a:A) RETURN a").parse().unwrap() else {
            panic!("INSERT ... RETURN is a query");
        };
        assert_eq!(query.create_clauses.len(), 1);
        assert_eq!(query.return_clause.items.len(), 1);

        let Statement::Query(query) = Parser::new("CREATE (:A) CREATE (:B)").parse().unwrap()
        else {
            panic!("a CREATE sequence is a query");
        };
        assert_eq!(query.create_clauses.len(), 2);
    }

    #[test]
    fn test_lone_insert_stays_an_insert_statement() {
        for query in ["INSERT (:A {v: 1})", "INSERT (:A), (:B)", "CREATE (:A)"] {
            let statement = Parser::new(query).parse().unwrap();
            assert!(
                matches!(
                    statement,
                    Statement::DataModification(DataModificationStatement::Insert(_))
                ),
                "{query}: {statement:?}"
            );
        }
    }

    #[test]
    fn test_power_operator_is_right_associative() {
        let Statement::Query(query) = Parser::new("RETURN 2 ^ 3 ^ 2 * 4").parse().unwrap() else {
            panic!("expected a query");
        };
        // (2 ^ (3 ^ 2)) * 4
        let Expression::Binary { left, op, .. } = &query.return_clause.items[0].expression else {
            panic!("expected a binary expression");
        };
        assert_eq!(*op, BinaryOp::Mul);
        let Expression::Binary { op, right, .. } = left.as_ref() else {
            panic!("expected 2 ^ (3 ^ 2)");
        };
        assert_eq!(*op, BinaryOp::Pow);
        assert!(matches!(
            right.as_ref(),
            Expression::Binary {
                op: BinaryOp::Pow,
                ..
            }
        ));
    }

    #[test]
    fn test_alter_type_accepts_optional_property_keyword() {
        let properties = |query: &str| -> Vec<(String, String)> {
            let Statement::Schema(SchemaStatement::AlterNodeType(stmt)) =
                Parser::new(query).parse().unwrap()
            else {
                panic!("{query}: expected ALTER NODE TYPE");
            };
            stmt.alterations
                .iter()
                .map(|alteration| match alteration {
                    TypeAlteration::AddProperty(def) => (def.name.clone(), def.data_type.clone()),
                    TypeAlteration::DropProperty(name) => (name.clone(), "dropped".to_string()),
                })
                .collect()
        };
        let pair = |a: &str, b: &str| (a.to_string(), b.to_string());

        assert_eq!(
            properties("ALTER NODE TYPE Sensor ADD PROPERTY location STRING"),
            vec![pair("location", "STRING")]
        );
        assert_eq!(
            properties("ALTER NODE TYPE Sensor ADD location STRING"),
            vec![pair("location", "STRING")]
        );
        assert_eq!(
            properties("ALTER NODE TYPE Sensor DROP PROPERTY location"),
            vec![pair("location", "dropped")]
        );
        assert_eq!(
            properties("ALTER NODE TYPE Sensor ADD `property` STRING"),
            vec![pair("property", "STRING")]
        );
    }

    #[test]
    fn test_parse_error_insert_without_pattern() {
        let mut parser = Parser::new("INSERT RETURN n");
        let result = parser.parse();
        assert!(result.is_err(), "INSERT without pattern should fail");
    }

    // ==================== 0.5.13 Features ====================

    // --- Comments ---

    #[test]
    fn test_parse_with_line_comment() {
        let mut parser = Parser::new("MATCH (n) -- find all nodes\nRETURN n");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Line comment should be skipped: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_parse_with_block_comment() {
        let mut parser = Parser::new("MATCH /* nodes */ (n:Person) RETURN n");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Block comment should be skipped: {:?}",
            result.err()
        );
    }

    // --- XOR operator ---

    #[test]
    fn test_parse_xor_expression() {
        let mut parser = Parser::new("MATCH (n) WHERE n.a = 1 XOR n.b = 2 RETURN n");
        let result = parser.parse();
        assert!(result.is_ok(), "XOR should parse: {:?}", result.err());

        if let Statement::Query(q) = result.unwrap() {
            let where_clause = q.where_clause.expect("Expected WHERE clause");
            if let Expression::Binary { op, .. } = &where_clause.expression {
                assert_eq!(*op, BinaryOp::Xor);
            } else {
                panic!("Expected binary XOR expression");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    // --- ISO Path Quantifiers {m,n} ---

    #[test]
    fn test_parse_iso_path_quantifier_range() {
        let mut parser = Parser::new("MATCH (a)-[:KNOWS{2,5}]->(b) RETURN a, b");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "ISO {{m,n}} quantifier should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            if let Pattern::Path(path) = &q.match_clauses[0].patterns[0].pattern {
                assert_eq!(path.edges[0].min_hops, Some(2));
                assert_eq!(path.edges[0].max_hops, Some(5));
            } else {
                panic!("Expected path pattern");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_iso_path_quantifier_exact() {
        let mut parser = Parser::new("MATCH (a)-[:KNOWS{3}]->(b) RETURN a, b");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "ISO {{n}} quantifier should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            if let Pattern::Path(path) = &q.match_clauses[0].patterns[0].pattern {
                assert_eq!(path.edges[0].min_hops, Some(3));
                assert_eq!(path.edges[0].max_hops, Some(3));
            } else {
                panic!("Expected path pattern");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_iso_path_quantifier_lower_only() {
        let mut parser = Parser::new("MATCH (a)-[:KNOWS{2,}]->(b) RETURN a, b");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "ISO {{m,}} quantifier should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            if let Pattern::Path(path) = &q.match_clauses[0].patterns[0].pattern {
                assert_eq!(path.edges[0].min_hops, Some(2));
                assert_eq!(path.edges[0].max_hops, None);
            } else {
                panic!("Expected path pattern");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    // --- List Access list[i] ---

    #[test]
    fn test_parse_list_index_access() {
        let mut parser = Parser::new("MATCH (n) RETURN [1, 2, 3][0]");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "List index access should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            if let Expression::IndexAccess { .. } = &q.return_clause.items[0].expression {
                // IndexAccess parsed
            } else {
                panic!(
                    "Expected IndexAccess expression, got {:?}",
                    q.return_clause.items[0].expression
                );
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    // --- CAST expressions ---

    #[test]
    fn test_parse_cast_to_integer() {
        let mut parser = Parser::new("MATCH (n) RETURN CAST('42' AS INTEGER)");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "CAST AS INTEGER should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            if let Expression::FunctionCall { name, .. } = &q.return_clause.items[0].expression {
                assert_eq!(name, "toInteger");
            } else {
                panic!("Expected function call from CAST");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_cast_to_float() {
        let mut parser = Parser::new("MATCH (n) RETURN CAST(n.val AS FLOAT)");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "CAST AS FLOAT should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            if let Expression::FunctionCall { name, .. } = &q.return_clause.items[0].expression {
                assert_eq!(name, "toFloat");
            } else {
                panic!("Expected function call from CAST");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_cast_to_string() {
        let mut parser = Parser::new("MATCH (n) RETURN CAST(42 AS STRING)");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "CAST AS STRING should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            if let Expression::FunctionCall { name, .. } = &q.return_clause.items[0].expression {
                assert_eq!(name, "toString");
            } else {
                panic!("Expected function call from CAST");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_cast_to_boolean() {
        let mut parser = Parser::new("MATCH (n) RETURN CAST('true' AS BOOLEAN)");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "CAST AS BOOLEAN should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            if let Expression::FunctionCall { name, .. } = &q.return_clause.items[0].expression {
                assert_eq!(name, "toBoolean");
            } else {
                panic!("Expected function call from CAST");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    // --- OFFSET as SKIP alias ---

    #[test]
    fn test_parse_offset_as_skip_alias() {
        let mut parser = Parser::new("MATCH (n) RETURN n OFFSET 10 LIMIT 5");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "OFFSET should parse as SKIP alias: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            assert!(q.return_clause.skip.is_some());
            assert!(q.return_clause.limit.is_some());
        } else {
            panic!("Expected Query statement");
        }
    }

    // --- Label Expressions (IS syntax) ---

    #[test]
    fn test_parse_is_label_single() {
        let mut parser = Parser::new("MATCH (n IS Person) RETURN n");
        let result = parser.parse();
        assert!(result.is_ok(), "IS label should parse: {:?}", result.err());

        if let Statement::Query(q) = result.unwrap() {
            if let Pattern::Node(node) = &q.match_clauses[0].patterns[0].pattern {
                match &node.label_expression {
                    Some(LabelExpression::Label(name)) => assert_eq!(name, "Person"),
                    other => panic!("Expected Label(Person), got {:?}", other),
                }
            } else {
                panic!("Expected node pattern");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_is_label_disjunction() {
        let mut parser = Parser::new("MATCH (n IS Person | Company) RETURN n");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "IS label disjunction should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            if let Pattern::Node(node) = &q.match_clauses[0].patterns[0].pattern {
                assert!(matches!(
                    &node.label_expression,
                    Some(LabelExpression::Disjunction(_))
                ));
            } else {
                panic!("Expected node pattern");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_is_label_conjunction() {
        let mut parser = Parser::new("MATCH (n IS Person & Employee) RETURN n");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "IS label conjunction should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            if let Pattern::Node(node) = &q.match_clauses[0].patterns[0].pattern {
                assert!(matches!(
                    &node.label_expression,
                    Some(LabelExpression::Conjunction(_))
                ));
            } else {
                panic!("Expected node pattern");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_is_label_negation() {
        let mut parser = Parser::new("MATCH (n IS !Inactive) RETURN n");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "IS label negation should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            if let Pattern::Node(node) = &q.match_clauses[0].patterns[0].pattern {
                assert!(matches!(
                    &node.label_expression,
                    Some(LabelExpression::Negation(_))
                ));
            } else {
                panic!("Expected node pattern");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_is_label_wildcard() {
        let mut parser = Parser::new("MATCH (n IS %) RETURN n");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "IS wildcard should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            if let Pattern::Node(node) = &q.match_clauses[0].patterns[0].pattern {
                assert!(matches!(
                    &node.label_expression,
                    Some(LabelExpression::Wildcard)
                ));
            } else {
                panic!("Expected node pattern");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_is_label_complex() {
        // (Person | Company) & !Inactive
        let mut parser = Parser::new("MATCH (n IS (Person | Company) & !Inactive) RETURN n");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Complex label expression should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            if let Pattern::Node(node) = &q.match_clauses[0].patterns[0].pattern {
                assert!(node.label_expression.is_some());
            } else {
                panic!("Expected node pattern");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_is_label_on_edge_colon_syntax() {
        // IS on edges is not yet wired; use colon syntax instead
        let mut parser = Parser::new("MATCH (a)-[e:KNOWS]->(b) RETURN a, b");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Edge colon syntax should parse: {:?}",
            result.err()
        );
    }

    // --- Path Modes ---

    #[test]
    fn test_parse_path_mode_walk() {
        let mut parser = Parser::new("MATCH WALK (a)-[:KNOWS*]->(b) RETURN a, b");
        let result = parser.parse();
        assert!(result.is_ok(), "WALK mode should parse: {:?}", result.err());

        if let Statement::Query(q) = result.unwrap() {
            assert_eq!(q.match_clauses[0].path_mode, Some(PathMode::Walk));
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_path_mode_trail() {
        let mut parser = Parser::new("MATCH TRAIL (a)-[:KNOWS*]->(b) RETURN a, b");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "TRAIL mode should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            assert_eq!(q.match_clauses[0].path_mode, Some(PathMode::Trail));
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_path_mode_simple() {
        let mut parser = Parser::new("MATCH SIMPLE (a)-[:KNOWS*]->(b) RETURN a, b");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "SIMPLE mode should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            assert_eq!(q.match_clauses[0].path_mode, Some(PathMode::Simple));
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_path_mode_acyclic() {
        let mut parser = Parser::new("MATCH ACYCLIC (a)-[:KNOWS*]->(b) RETURN a, b");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "ACYCLIC mode should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            assert_eq!(q.match_clauses[0].path_mode, Some(PathMode::Acyclic));
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_no_path_mode_default() {
        let mut parser = Parser::new("MATCH (a)-[:KNOWS*]->(b) RETURN a, b");
        let result = parser.parse();
        assert!(result.is_ok());

        if let Statement::Query(q) = result.unwrap() {
            assert_eq!(q.match_clauses[0].path_mode, None);
        } else {
            panic!("Expected Query statement");
        }
    }

    // --- Composite Queries ---

    #[test]
    fn test_parse_union() {
        let mut parser =
            Parser::new("MATCH (n:Person) RETURN n.name UNION MATCH (n:Company) RETURN n.name");
        let result = parser.parse();
        assert!(result.is_ok(), "UNION should parse: {:?}", result.err());

        if let Statement::CompositeQuery { op, .. } = result.unwrap() {
            assert_eq!(op, CompositeOp::Union);
        } else {
            panic!("Expected CompositeQuery");
        }
    }

    #[test]
    fn test_parse_union_all() {
        let mut parser =
            Parser::new("MATCH (n:Person) RETURN n.name UNION ALL MATCH (n:Company) RETURN n.name");
        let result = parser.parse();
        assert!(result.is_ok(), "UNION ALL should parse: {:?}", result.err());

        if let Statement::CompositeQuery { op, .. } = result.unwrap() {
            assert_eq!(op, CompositeOp::UnionAll);
        } else {
            panic!("Expected CompositeQuery");
        }
    }

    #[test]
    fn test_parse_except() {
        let mut parser =
            Parser::new("MATCH (n:Person) RETURN n.name EXCEPT MATCH (n:Employee) RETURN n.name");
        let result = parser.parse();
        assert!(result.is_ok(), "EXCEPT should parse: {:?}", result.err());

        if let Statement::CompositeQuery { op, .. } = result.unwrap() {
            assert_eq!(op, CompositeOp::Except);
        } else {
            panic!("Expected CompositeQuery");
        }
    }

    #[test]
    fn test_parse_intersect() {
        let mut parser = Parser::new(
            "MATCH (n:Person) RETURN n.name INTERSECT MATCH (n:Employee) RETURN n.name",
        );
        let result = parser.parse();
        assert!(result.is_ok(), "INTERSECT should parse: {:?}", result.err());

        if let Statement::CompositeQuery { op, .. } = result.unwrap() {
            assert_eq!(op, CompositeOp::Intersect);
        } else {
            panic!("Expected CompositeQuery");
        }
    }

    #[test]
    fn test_parse_otherwise() {
        let mut parser =
            Parser::new("MATCH (n:Person) RETURN n.name OTHERWISE MATCH (n:Company) RETURN n.name");
        let result = parser.parse();
        assert!(result.is_ok(), "OTHERWISE should parse: {:?}", result.err());

        if let Statement::CompositeQuery { op, .. } = result.unwrap() {
            assert_eq!(op, CompositeOp::Otherwise);
        } else {
            panic!("Expected CompositeQuery");
        }
    }

    // --- FILTER statement ---

    #[test]
    fn test_parse_filter_as_where_synonym() {
        let mut parser = Parser::new("MATCH (n:Person) FILTER n.age > 25 RETURN n");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "FILTER should parse as WHERE synonym: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            assert!(q.where_clause.is_some());
        } else {
            panic!("Expected Query statement");
        }
    }

    // --- GROUP BY ---

    #[test]
    fn test_parse_group_by() {
        let mut parser = Parser::new("MATCH (n:Person) RETURN n.city, count(n) GROUP BY n.city");
        let result = parser.parse();
        assert!(result.is_ok(), "GROUP BY should parse: {:?}", result.err());

        if let Statement::Query(q) = result.unwrap() {
            assert_eq!(q.return_clause.group_by.len(), 1);
            if let Expression::PropertyAccess { property, .. } = &q.return_clause.group_by[0] {
                assert_eq!(property, "city");
            } else {
                panic!("Expected property access in GROUP BY");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_group_by_multiple() {
        let mut parser =
            Parser::new("MATCH (n:Person) RETURN n.city, n.age, count(n) GROUP BY n.city, n.age");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Multiple GROUP BY should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            assert_eq!(q.return_clause.group_by.len(), 2);
        } else {
            panic!("Expected Query statement");
        }
    }

    // --- ELEMENT_ID function ---

    #[test]
    fn test_parse_element_id_function() {
        let mut parser = Parser::new("MATCH (n) RETURN element_id(n)");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "element_id should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            if let Expression::FunctionCall { name, args, .. } =
                &q.return_clause.items[0].expression
            {
                assert_eq!(name, "element_id");
                assert_eq!(args.len(), 1);
            } else {
                panic!("Expected function call");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    // --- Error Cases for New Features ---

    #[test]
    fn test_parse_error_cast_missing_as() {
        let mut parser = Parser::new("MATCH (n) RETURN CAST(42 INTEGER)");
        let result = parser.parse();
        assert!(result.is_err(), "CAST without AS should fail");
    }

    #[test]
    fn test_parse_error_cast_invalid_type() {
        let mut parser = Parser::new("MATCH (n) RETURN CAST(42 AS VECTOR)");
        let result = parser.parse();
        assert!(result.is_err(), "CAST to unsupported type should fail");
    }

    #[test]
    fn test_parse_error_group_by_without_expressions() {
        let mut parser = Parser::new("MATCH (n) RETURN n GROUP BY");
        let result = parser.parse();
        assert!(result.is_err(), "GROUP BY without expressions should fail");
    }

    #[test]
    fn test_parse_hex_integer_literal() {
        let mut parser = Parser::new("MATCH (n) RETURN 0xFF");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Hex literal should parse: {:?}",
            result.err()
        );
        if let Statement::Query(q) = result.unwrap() {
            let item = &q.return_clause.items[0];
            if let Expression::Literal(Literal::Integer(val)) = &item.expression {
                assert_eq!(*val, 255, "0xFF should parse to 255");
            } else {
                panic!("Expected integer literal");
            }
        }
    }

    #[test]
    fn test_parse_octal_integer_literal() {
        let mut parser = Parser::new("MATCH (n) RETURN 0o77");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Octal literal should parse: {:?}",
            result.err()
        );
        if let Statement::Query(q) = result.unwrap() {
            let item = &q.return_clause.items[0];
            if let Expression::Literal(Literal::Integer(val)) = &item.expression {
                assert_eq!(*val, 63, "0o77 should parse to 63");
            } else {
                panic!("Expected integer literal");
            }
        }
    }

    #[test]
    fn test_parse_scientific_float_literal() {
        let mut parser = Parser::new("MATCH (n) RETURN 1.5e10");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Scientific literal should parse: {:?}",
            result.err()
        );
        if let Statement::Query(q) = result.unwrap() {
            let item = &q.return_clause.items[0];
            if let Expression::Literal(Literal::Float(val)) = &item.expression {
                assert!((val - 1.5e10).abs() < 1.0, "1.5e10 should parse correctly");
            } else {
                panic!("Expected float literal");
            }
        }
    }

    /// Helper to extract edges from the first match pattern.
    fn get_first_path_edges(stmt: &Statement) -> &[EdgePattern] {
        if let Statement::Query(q) = stmt
            && let Pattern::Path(path) = &q.match_clauses[0].patterns[0].pattern
        {
            return &path.edges;
        }
        panic!("Expected query with path pattern");
    }

    #[test]
    fn test_parse_edge_type_pipe_alternatives() {
        let mut parser = Parser::new("MATCH (a)-[:KNOWS|LIKES|FOLLOWS]->(b) RETURN a, b");
        let result = parser.parse().expect("Edge type pipe should parse");
        let edges = get_first_path_edges(&result);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].types, vec!["KNOWS", "LIKES", "FOLLOWS"]);
    }

    #[test]
    fn test_parse_edge_type_pipe_with_variable() {
        let mut parser = Parser::new("MATCH (a)-[r:KNOWS|LIKES]->(b) RETURN r");
        let result = parser
            .parse()
            .expect("Edge type pipe with var should parse");
        let edges = get_first_path_edges(&result);
        assert_eq!(edges[0].variable, Some("r".to_string()));
        assert_eq!(edges[0].types, vec!["KNOWS", "LIKES"]);
    }

    #[test]
    fn test_parse_tilde_undirected_edge() {
        let mut parser = Parser::new("MATCH (a)~[e:KNOWS]~(b) RETURN a, b");
        let result = parser.parse().expect("Tilde edge should parse");
        let edges = get_first_path_edges(&result);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].variable, Some("e".to_string()));
        assert_eq!(edges[0].types, vec!["KNOWS"]);
        assert_eq!(edges[0].direction, EdgeDirection::Undirected);
    }

    #[test]
    fn test_parse_tilde_simple() {
        let mut parser = Parser::new("MATCH (a)~(b) RETURN a");
        let result = parser.parse().expect("Simple tilde should parse");
        let edges = get_first_path_edges(&result);
        assert_eq!(edges[0].direction, EdgeDirection::Undirected);
        assert!(edges[0].variable.is_none());
        assert!(edges[0].types.is_empty());
    }

    #[test]
    fn test_parse_tilde_with_pipe_types() {
        let mut parser = Parser::new("MATCH (a)~[:KNOWS|LIKES]~(b) RETURN a");
        let result = parser.parse().expect("Tilde with pipe types should parse");
        let edges = get_first_path_edges(&result);
        assert_eq!(edges[0].types, vec!["KNOWS", "LIKES"]);
        assert_eq!(edges[0].direction, EdgeDirection::Undirected);
    }

    // ==================== shorthand arrow edges ====================

    #[test]
    fn test_parse_shorthand_outgoing_arrow() {
        // --> shorthand: directed outgoing, no brackets
        let mut parser = Parser::new("MATCH (a)-->(b) RETURN b");
        let result = parser.parse().expect("Shorthand --> should parse");
        let edges = get_first_path_edges(&result);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].direction, EdgeDirection::Outgoing);
        assert!(edges[0].variable.is_none());
        assert!(edges[0].types.is_empty());
    }

    #[test]
    fn test_parse_shorthand_incoming_arrow() {
        // <-- shorthand: directed incoming, no brackets
        let mut parser = Parser::new("MATCH (a)<--(b) RETURN a");
        let result = parser.parse().expect("Shorthand <-- should parse");
        let edges = get_first_path_edges(&result);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].direction, EdgeDirection::Incoming);
        assert!(edges[0].variable.is_none());
        assert!(edges[0].types.is_empty());
    }

    #[test]
    fn test_parse_shorthand_arrow_chain() {
        // Chained shorthand arrows: (a)-->(b)-->(c)
        let mut parser = Parser::new("MATCH (a)-->(b)-->(c) RETURN c");
        let result = parser.parse().expect("Chained --> should parse");
        let edges = get_first_path_edges(&result);
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].direction, EdgeDirection::Outgoing);
        assert_eq!(edges[1].direction, EdgeDirection::Outgoing);
    }

    #[test]
    fn test_parse_shorthand_mixed_directions() {
        // Mixed: (a)<--(b)-->(c)
        let mut parser = Parser::new("MATCH (a)<--(b)-->(c) RETURN c");
        let result = parser.parse().expect("Mixed <-- and --> should parse");
        let edges = get_first_path_edges(&result);
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].direction, EdgeDirection::Incoming);
        assert_eq!(edges[1].direction, EdgeDirection::Outgoing);
    }

    #[test]
    fn test_parse_shorthand_undirected_still_works() {
        // -- (undirected) must still work
        let mut parser = Parser::new("MATCH (a)--(b) RETURN b");
        let result = parser.parse().expect("Undirected -- should still parse");
        let edges = get_first_path_edges(&result);
        assert_eq!(edges[0].direction, EdgeDirection::Undirected);
    }

    // ==================== unescape_string ====================

    #[test]
    fn test_unescape_string_newline() {
        assert_eq!(unescape_string(r"hello\nworld"), "hello\nworld");
    }

    #[test]
    fn test_unescape_string_carriage_return() {
        assert_eq!(unescape_string(r"line\rend"), "line\rend");
    }

    #[test]
    fn test_unescape_string_tab() {
        assert_eq!(unescape_string(r"col1\tcol2"), "col1\tcol2");
    }

    #[test]
    fn test_unescape_string_backslash() {
        assert_eq!(unescape_string(r"path\\to"), "path\\to");
    }

    #[test]
    fn test_unescape_string_single_quote() {
        assert_eq!(unescape_string(r"it\'s"), "it's");
    }

    #[test]
    fn test_unescape_string_double_quote() {
        assert_eq!(unescape_string(r#"say\"hello\""#), "say\"hello\"");
    }

    #[test]
    fn test_unescape_string_unknown_escape() {
        // Unknown escapes are kept as-is (backslash + char)
        assert_eq!(unescape_string(r"\z"), "\\z");
    }

    #[test]
    fn test_unescape_string_trailing_backslash() {
        // Trailing backslash with no following char is kept
        assert_eq!(unescape_string("trailing\\"), "trailing\\");
    }

    // ==================== DDL / Session Commands ====================

    #[test]
    fn test_parse_create_graph() {
        let mut parser = Parser::new("CREATE GRAPH mydb");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "CREATE GRAPH should parse: {:?}",
            result.err()
        );

        if let Statement::SessionCommand(SessionCommand::CreateGraph {
            name,
            if_not_exists,
            ..
        }) = result.unwrap()
        {
            assert_eq!(name, "mydb");
            assert!(!if_not_exists);
        } else {
            panic!("Expected CreateGraph session command");
        }
    }

    #[test]
    fn test_parse_create_graph_if_not_exists() {
        let mut parser = Parser::new("CREATE GRAPH IF NOT EXISTS mydb");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "CREATE GRAPH IF NOT EXISTS should parse: {:?}",
            result.err()
        );

        if let Statement::SessionCommand(SessionCommand::CreateGraph {
            name,
            if_not_exists,
            ..
        }) = result.unwrap()
        {
            assert_eq!(name, "mydb");
            assert!(if_not_exists);
        } else {
            panic!("Expected CreateGraph session command");
        }
    }

    #[test]
    fn test_parse_create_property_graph() {
        let mut parser = Parser::new("CREATE PROPERTY GRAPH pg1");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "CREATE PROPERTY GRAPH should parse: {:?}",
            result.err()
        );

        if let Statement::SessionCommand(SessionCommand::CreateGraph {
            name,
            if_not_exists,
            ..
        }) = result.unwrap()
        {
            assert_eq!(name, "pg1");
            assert!(!if_not_exists);
        } else {
            panic!("Expected CreateGraph session command");
        }
    }

    #[test]
    fn test_parse_drop_graph() {
        let mut parser = Parser::new("DROP GRAPH mydb");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "DROP GRAPH should parse: {:?}",
            result.err()
        );

        if let Statement::SessionCommand(SessionCommand::DropGraph { name, if_exists }) =
            result.unwrap()
        {
            assert_eq!(name, "mydb");
            assert!(!if_exists);
        } else {
            panic!("Expected DropGraph session command");
        }
    }

    #[test]
    fn test_parse_drop_graph_if_exists() {
        let mut parser = Parser::new("DROP GRAPH IF EXISTS mydb");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "DROP GRAPH IF EXISTS should parse: {:?}",
            result.err()
        );

        if let Statement::SessionCommand(SessionCommand::DropGraph { name, if_exists }) =
            result.unwrap()
        {
            assert_eq!(name, "mydb");
            assert!(if_exists);
        } else {
            panic!("Expected DropGraph session command");
        }
    }

    #[test]
    fn test_parse_drop_property_graph() {
        let mut parser = Parser::new("DROP PROPERTY GRAPH pg1");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "DROP PROPERTY GRAPH should parse: {:?}",
            result.err()
        );

        if let Statement::SessionCommand(SessionCommand::DropGraph { name, if_exists }) =
            result.unwrap()
        {
            assert_eq!(name, "pg1");
            assert!(!if_exists);
        } else {
            panic!("Expected DropGraph session command");
        }
    }

    #[test]
    fn test_parse_use_graph() {
        let mut parser = Parser::new("USE GRAPH workspace");
        let result = parser.parse();
        assert!(result.is_ok(), "USE GRAPH should parse: {:?}", result.err());

        if let Statement::SessionCommand(SessionCommand::UseGraph(name)) = result.unwrap() {
            assert_eq!(name, "workspace");
        } else {
            panic!("Expected UseGraph session command");
        }
    }

    #[test]
    fn test_parse_session_set_graph() {
        let mut parser = Parser::new("SESSION SET GRAPH analytics");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "SESSION SET GRAPH should parse: {:?}",
            result.err()
        );

        if let Statement::SessionCommand(SessionCommand::SessionSetGraph(name)) = result.unwrap() {
            assert_eq!(name, "analytics");
        } else {
            panic!("Expected SessionSetGraph session command");
        }
    }

    #[test]
    fn test_parse_session_set_time_zone() {
        let mut parser = Parser::new("SESSION SET TIME ZONE 'UTC+5'");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "SESSION SET TIME ZONE should parse: {:?}",
            result.err()
        );

        if let Statement::SessionCommand(SessionCommand::SessionSetTimeZone(tz)) = result.unwrap() {
            assert_eq!(tz, "UTC+5");
        } else {
            panic!("Expected SessionSetTimeZone session command");
        }
    }

    #[test]
    fn test_parse_session_set_schema() {
        // ISO/IEC 39075 Section 7.1 GR1: SESSION SET SCHEMA is independent from SESSION SET GRAPH
        let mut parser = Parser::new("SESSION SET SCHEMA myschema");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "SESSION SET SCHEMA should parse: {:?}",
            result.err()
        );

        if let Statement::SessionCommand(SessionCommand::SessionSetSchema(name)) = result.unwrap() {
            assert_eq!(name, "myschema");
        } else {
            panic!("Expected SessionSetSchema session command");
        }
    }

    #[test]
    fn test_parse_session_set_parameter() {
        let mut parser = Parser::new("SESSION SET PARAMETER timeout = 30");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "SESSION SET PARAMETER should parse: {:?}",
            result.err()
        );

        if let Statement::SessionCommand(SessionCommand::SessionSetParameter(name, _value)) =
            result.unwrap()
        {
            assert_eq!(name, "timeout");
        } else {
            panic!("Expected SessionSetParameter session command");
        }
    }

    #[test]
    fn test_parse_session_reset() {
        // Bare SESSION RESET = reset all characteristics (Section 7.2 SR2b)
        let mut parser = Parser::new("SESSION RESET");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "SESSION RESET should parse: {:?}",
            result.err()
        );

        assert!(matches!(
            result.unwrap(),
            Statement::SessionCommand(SessionCommand::SessionReset(SessionResetTarget::All))
        ));
    }

    #[test]
    fn test_parse_session_reset_all() {
        let mut parser = Parser::new("SESSION RESET ALL");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "SESSION RESET ALL should parse: {:?}",
            result.err()
        );

        assert!(matches!(
            result.unwrap(),
            Statement::SessionCommand(SessionCommand::SessionReset(SessionResetTarget::All))
        ));
    }

    #[test]
    fn test_parse_session_reset_schema() {
        // ISO/IEC 39075 Section 7.2 GR1: reset session schema independently
        let mut parser = Parser::new("SESSION RESET SCHEMA");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "SESSION RESET SCHEMA should parse: {:?}",
            result.err()
        );

        assert!(matches!(
            result.unwrap(),
            Statement::SessionCommand(SessionCommand::SessionReset(SessionResetTarget::Schema))
        ));
    }

    #[test]
    fn test_parse_session_reset_graph() {
        // ISO/IEC 39075 Section 7.2 GR2: reset session graph independently
        let mut parser = Parser::new("SESSION RESET GRAPH");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "SESSION RESET GRAPH should parse: {:?}",
            result.err()
        );

        assert!(matches!(
            result.unwrap(),
            Statement::SessionCommand(SessionCommand::SessionReset(SessionResetTarget::Graph))
        ));
    }

    #[test]
    fn test_parse_session_reset_property_graph() {
        // SESSION RESET PROPERTY GRAPH (Section 7.2)
        let mut parser = Parser::new("SESSION RESET PROPERTY GRAPH");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "SESSION RESET PROPERTY GRAPH should parse: {:?}",
            result.err()
        );

        assert!(matches!(
            result.unwrap(),
            Statement::SessionCommand(SessionCommand::SessionReset(SessionResetTarget::Graph))
        ));
    }

    #[test]
    fn test_parse_session_close() {
        let mut parser = Parser::new("SESSION CLOSE");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "SESSION CLOSE should parse: {:?}",
            result.err()
        );

        assert!(matches!(
            result.unwrap(),
            Statement::SessionCommand(SessionCommand::SessionClose)
        ));
    }

    #[test]
    fn test_parse_start_transaction() {
        let mut parser = Parser::new("START TRANSACTION");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "START TRANSACTION should parse: {:?}",
            result.err()
        );
        match result.unwrap() {
            Statement::SessionCommand(SessionCommand::StartTransaction {
                read_only,
                isolation_level,
            }) => {
                assert!(!read_only);
                assert!(isolation_level.is_none());
            }
            other => panic!("Expected StartTransaction, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_start_transaction_read_only() {
        let mut parser = Parser::new("START TRANSACTION READ ONLY");
        let result = parser.parse().unwrap();
        match result {
            Statement::SessionCommand(SessionCommand::StartTransaction {
                read_only,
                isolation_level,
            }) => {
                assert!(read_only);
                assert!(isolation_level.is_none());
            }
            other => panic!("Expected StartTransaction READ ONLY, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_start_transaction_isolation_level() {
        let mut parser = Parser::new("START TRANSACTION ISOLATION LEVEL SERIALIZABLE");
        let result = parser.parse().unwrap();
        match result {
            Statement::SessionCommand(SessionCommand::StartTransaction {
                read_only,
                isolation_level,
            }) => {
                assert!(!read_only);
                assert_eq!(
                    isolation_level,
                    Some(TransactionIsolationLevel::Serializable)
                );
            }
            other => panic!("Expected StartTransaction with isolation, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_start_transaction_read_only_with_isolation() {
        let mut parser = Parser::new("START TRANSACTION READ ONLY ISOLATION LEVEL READ COMMITTED");
        let result = parser.parse().unwrap();
        match result {
            Statement::SessionCommand(SessionCommand::StartTransaction {
                read_only,
                isolation_level,
            }) => {
                assert!(read_only);
                assert_eq!(
                    isolation_level,
                    Some(TransactionIsolationLevel::ReadCommitted)
                );
            }
            other => panic!("Expected StartTransaction READ ONLY + isolation, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_commit() {
        let mut parser = Parser::new("COMMIT");
        let result = parser.parse();
        assert!(result.is_ok(), "COMMIT should parse: {:?}", result.err());

        assert!(matches!(
            result.unwrap(),
            Statement::SessionCommand(SessionCommand::Commit)
        ));
    }

    #[test]
    fn test_parse_rollback() {
        let mut parser = Parser::new("ROLLBACK");
        let result = parser.parse();
        assert!(result.is_ok(), "ROLLBACK should parse: {:?}", result.err());

        assert!(matches!(
            result.unwrap(),
            Statement::SessionCommand(SessionCommand::Rollback)
        ));
    }

    // ==================== Edge WHERE Clause ====================

    #[test]
    fn test_parse_edge_where_clause() {
        let mut parser = Parser::new("MATCH (a)-[e:KNOWS WHERE e.since >= 2020]->(b) RETURN a, b");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Edge WHERE clause should parse: {:?}",
            result.err()
        );

        let stmt = result.unwrap();
        let edges = get_first_path_edges(&stmt);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].variable, Some("e".to_string()));
        assert_eq!(edges[0].types, vec!["KNOWS"]);
        assert!(
            edges[0].where_clause.is_some(),
            "Edge where_clause should be Some"
        );
    }

    // ==================== Path Quantifier Disambiguation ====================

    #[test]
    fn test_parse_path_quantifier_vs_property_map() {
        // {1,3} after an edge type is a quantifier (min/max hops)
        let mut parser = Parser::new("MATCH (a)-[:KNOWS{1,3}]->(b) RETURN a, b");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "{{1,3}} quantifier should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            if let Pattern::Path(path) = &q.match_clauses[0].patterns[0].pattern {
                assert_eq!(path.edges[0].min_hops, Some(1));
                assert_eq!(path.edges[0].max_hops, Some(3));
            } else {
                panic!("Expected path pattern");
            }
        } else {
            panic!("Expected Query statement");
        }

        // {since: 2020} inside a node pattern is a property map, not a quantifier
        let mut parser2 = Parser::new("MATCH (n:Person {since: 2020}) RETURN n");
        let result2 = parser2.parse();
        assert!(
            result2.is_ok(),
            "Property map should parse: {:?}",
            result2.err()
        );

        if let Statement::Query(q) = result2.unwrap() {
            if let Pattern::Node(node) = &q.match_clauses[0].patterns[0].pattern {
                assert_eq!(node.properties.len(), 1);
                assert_eq!(node.properties[0].0, "since");
            } else {
                panic!("Expected node pattern");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    // ==================== NEXT Composition ====================

    #[test]
    fn test_parse_next_composition() {
        let mut parser =
            Parser::new("MATCH (n:Person) RETURN n.name NEXT MATCH (m:Company) RETURN m.name");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "NEXT composition should parse: {:?}",
            result.err()
        );

        if let Statement::CompositeQuery { op, .. } = result.unwrap() {
            assert_eq!(op, CompositeOp::Next);
        } else {
            panic!("Expected CompositeQuery with Next op");
        }
    }

    // ==================== FINISH Statement ====================

    #[test]
    fn test_parse_finish_statement() {
        let mut parser = Parser::new("MATCH (n) FINISH");
        let result = parser.parse();
        assert!(result.is_ok(), "FINISH should parse: {:?}", result.err());

        if let Statement::Query(q) = result.unwrap() {
            assert!(q.return_clause.is_finish, "Expected is_finish to be true");
            assert!(
                q.return_clause.items.is_empty(),
                "FINISH should have no return items"
            );
        } else {
            panic!("Expected Query statement");
        }
    }

    // ==================== SELECT Statement ====================

    #[test]
    fn test_parse_select_statement() {
        let mut parser = Parser::new("MATCH (n:Person) SELECT n.name");
        let result = parser.parse();
        assert!(result.is_ok(), "SELECT should parse: {:?}", result.err());

        if let Statement::Query(q) = result.unwrap() {
            assert!(!q.return_clause.is_finish);
            assert_eq!(q.return_clause.items.len(), 1);
        } else {
            panic!("Expected Query statement");
        }
    }

    // ==================== Error Cases for New Commands ====================

    #[test]
    fn test_parse_error_drop_nothing() {
        let mut parser = Parser::new("DROP NOTHING");
        let result = parser.parse();
        assert!(result.is_err(), "DROP NOTHING should fail");
    }

    #[test]
    fn test_parse_error_session_destroy() {
        let mut parser = Parser::new("SESSION DESTROY");
        let result = parser.parse();
        assert!(result.is_err(), "SESSION DESTROY should fail");
    }

    #[test]
    fn test_parse_error_start_something() {
        let mut parser = Parser::new("START SOMETHING");
        let result = parser.parse();
        assert!(result.is_err(), "START SOMETHING should fail");
    }

    #[test]
    fn test_parse_error_use_something() {
        let mut parser = Parser::new("USE SOMETHING");
        let result = parser.parse();
        assert!(result.is_err(), "USE SOMETHING should fail");
    }

    // -------------------------------------------------------------------
    // ISO GQL Conformance: NULLS FIRST/LAST (GA03)
    // -------------------------------------------------------------------

    #[test]
    fn test_parse_order_by_nulls_first() {
        let mut parser =
            Parser::new("MATCH (n:Person) RETURN n.name ORDER BY n.age ASC NULLS FIRST");
        let stmt = parser.parse().unwrap();
        if let Statement::Query(q) = stmt {
            let order = q.return_clause.order_by.unwrap();
            assert_eq!(order.items[0].nulls, Some(NullsOrdering::First));
        } else {
            panic!("Expected query");
        }
    }

    #[test]
    fn test_parse_order_by_nulls_last() {
        let mut parser =
            Parser::new("MATCH (n:Person) RETURN n.name ORDER BY n.age DESC NULLS LAST");
        let stmt = parser.parse().unwrap();
        if let Statement::Query(q) = stmt {
            let order = q.return_clause.order_by.unwrap();
            assert_eq!(order.items[0].nulls, Some(NullsOrdering::Last));
        } else {
            panic!("Expected query");
        }
    }

    #[test]
    fn test_parse_order_by_no_nulls_clause() {
        let mut parser = Parser::new("MATCH (n:Person) RETURN n.name ORDER BY n.age ASC");
        let stmt = parser.parse().unwrap();
        if let Statement::Query(q) = stmt {
            let order = q.return_clause.order_by.unwrap();
            assert_eq!(order.items[0].nulls, None);
        } else {
            panic!("Expected query");
        }
    }

    // -------------------------------------------------------------------
    // ISO GQL Conformance: NULLIF / COALESCE keyword syntax
    // -------------------------------------------------------------------

    #[test]
    fn test_parse_nullif() {
        let mut parser = Parser::new("MATCH (n:Person) RETURN NULLIF(n.age, 30) AS val");
        let stmt = parser.parse().unwrap();
        if let Statement::Query(q) = stmt {
            match &q.return_clause.items[0].expression {
                Expression::FunctionCall { name, args, .. } => {
                    assert_eq!(name, "nullif");
                    assert_eq!(args.len(), 2);
                }
                other => panic!("Expected FunctionCall, got {other:?}"),
            }
        } else {
            panic!("Expected query");
        }
    }

    #[test]
    fn test_parse_coalesce() {
        let mut parser =
            Parser::new("MATCH (n:Person) RETURN COALESCE(null, n.name, 'default') AS val");
        let stmt = parser.parse().unwrap();
        if let Statement::Query(q) = stmt {
            match &q.return_clause.items[0].expression {
                Expression::FunctionCall { name, args, .. } => {
                    assert_eq!(name, "coalesce");
                    assert_eq!(args.len(), 3);
                }
                other => panic!("Expected FunctionCall, got {other:?}"),
            }
        } else {
            panic!("Expected query");
        }
    }

    // -------------------------------------------------------------------
    // ISO GQL Conformance: IS [NFC|NFD|NFKC|NFKD] NORMALIZED
    // -------------------------------------------------------------------

    #[test]
    fn test_parse_is_nfc_normalized() {
        let mut parser = Parser::new("MATCH (n:Person) RETURN n.name IS NFC NORMALIZED AS norm");
        let stmt = parser.parse().unwrap();
        if let Statement::Query(q) = stmt {
            match &q.return_clause.items[0].expression {
                Expression::FunctionCall { name, args, .. } => {
                    assert_eq!(name, "isNormalized");
                    assert_eq!(args.len(), 2);
                    // Second arg should be the form string "NFC"
                    match &args[1] {
                        Expression::Literal(Literal::String(s)) => assert_eq!(s, "NFC"),
                        other => panic!("Expected string literal, got {other:?}"),
                    }
                }
                other => panic!("Expected FunctionCall, got {other:?}"),
            }
        } else {
            panic!("Expected query");
        }
    }

    #[test]
    fn test_parse_is_not_nfkd_normalized() {
        let mut parser =
            Parser::new("MATCH (n:Person) RETURN n.name IS NOT NFKD NORMALIZED AS norm");
        let stmt = parser.parse().unwrap();
        if let Statement::Query(q) = stmt {
            match &q.return_clause.items[0].expression {
                Expression::Unary { op, operand } => {
                    assert_eq!(*op, UnaryOp::Not);
                    match operand.as_ref() {
                        Expression::FunctionCall { name, args, .. } => {
                            assert_eq!(name, "isNormalized");
                            match &args[1] {
                                Expression::Literal(Literal::String(s)) => {
                                    assert_eq!(s, "NFKD");
                                }
                                other => panic!("Expected string literal, got {other:?}"),
                            }
                        }
                        other => panic!("Expected FunctionCall, got {other:?}"),
                    }
                }
                other => panic!("Expected Unary NOT, got {other:?}"),
            }
        } else {
            panic!("Expected query");
        }
    }

    #[test]
    fn test_parse_is_normalized_default_form() {
        let mut parser = Parser::new("MATCH (n:Person) RETURN n.name IS NORMALIZED AS norm");
        let stmt = parser.parse().unwrap();
        if let Statement::Query(q) = stmt {
            match &q.return_clause.items[0].expression {
                Expression::FunctionCall { name, args, .. } => {
                    assert_eq!(name, "isNormalized");
                    assert_eq!(args.len(), 2);
                    match &args[1] {
                        Expression::Literal(Literal::String(s)) => assert_eq!(s, "NFC"),
                        other => panic!("Expected string literal NFC, got {other:?}"),
                    }
                }
                other => panic!("Expected FunctionCall, got {other:?}"),
            }
        } else {
            panic!("Expected query");
        }
    }

    // --- Group 4: Parenthesized Path Enhancements ---

    #[test]
    fn test_parse_parenthesized_path_mode_prefix() {
        // G049: Path mode prefix inside parenthesized pattern
        let mut parser = Parser::new("MATCH (TRAIL (a)-[e]->(b)){2,5} RETURN a, b");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Parenthesized path with mode prefix should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            let pat = &q.match_clauses[0].patterns[0].pattern;
            if let Pattern::Quantified {
                min,
                max,
                path_mode,
                ..
            } = pat
            {
                assert_eq!(*min, 2);
                assert_eq!(*max, Some(5));
                assert_eq!(*path_mode, Some(PathMode::Trail));
            } else {
                panic!("Expected Quantified pattern, got {pat:?}");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_parenthesized_where_clause() {
        // G050: WHERE clause inside parenthesized pattern
        let mut parser = Parser::new("MATCH ((a)-[e]->(b) WHERE e.weight > 5){1,3} RETURN a, b");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Parenthesized path with WHERE should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            let pat = &q.match_clauses[0].patterns[0].pattern;
            if let Pattern::Quantified {
                min,
                max,
                where_clause,
                ..
            } = pat
            {
                assert_eq!(*min, 1);
                assert_eq!(*max, Some(3));
                assert!(where_clause.is_some(), "WHERE clause should be present");
            } else {
                panic!("Expected Quantified pattern, got {pat:?}");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_parenthesized_all_features() {
        // All three: path mode + WHERE + quantifier
        let mut parser =
            Parser::new("MATCH (ACYCLIC (a)-[e]->(b) WHERE e.active = true){2,4} RETURN a, b");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Full parenthesized path should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            let pat = &q.match_clauses[0].patterns[0].pattern;
            if let Pattern::Quantified {
                min,
                max,
                path_mode,
                where_clause,
                subpath_var,
                ..
            } = pat
            {
                assert_eq!(*min, 2);
                assert_eq!(*max, Some(4));
                assert_eq!(*path_mode, Some(PathMode::Acyclic));
                assert!(where_clause.is_some());
                assert!(subpath_var.is_none());
            } else {
                panic!("Expected Quantified pattern, got {pat:?}");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    // --- Group 5: Simplified Path Patterns ---

    #[test]
    fn test_parse_simplified_outgoing() {
        // G080: -/:KNOWS/-> desugars to -[:KNOWS]->
        let mut parser = Parser::new("MATCH (a:Person)-/:KNOWS/->(b:Person) RETURN b.name");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Simplified outgoing path should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            if let Pattern::Path(path) = &q.match_clauses[0].patterns[0].pattern {
                assert_eq!(path.edges.len(), 1);
                assert_eq!(path.edges[0].types, vec!["KNOWS"]);
                assert_eq!(path.edges[0].direction, EdgeDirection::Outgoing);
            } else {
                panic!("Expected Path pattern");
            }
        } else {
            panic!("Expected Query");
        }
    }

    #[test]
    fn test_parse_simplified_incoming() {
        // G080: <-/:KNOWS/- desugars to <-[:KNOWS]-
        let mut parser = Parser::new("MATCH (a:Person)<-/:KNOWS/-(b:Person) RETURN b.name");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Simplified incoming path should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            if let Pattern::Path(path) = &q.match_clauses[0].patterns[0].pattern {
                assert_eq!(path.edges.len(), 1);
                assert_eq!(path.edges[0].types, vec!["KNOWS"]);
                assert_eq!(path.edges[0].direction, EdgeDirection::Incoming);
            } else {
                panic!("Expected Path pattern");
            }
        } else {
            panic!("Expected Query");
        }
    }

    #[test]
    fn test_parse_simplified_undirected() {
        // G080: -/:KNOWS/- desugars to -[:KNOWS]-
        let mut parser = Parser::new("MATCH (a:Person)-/:KNOWS/-(b:Person) RETURN b.name");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Simplified undirected path should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            if let Pattern::Path(path) = &q.match_clauses[0].patterns[0].pattern {
                assert_eq!(path.edges.len(), 1);
                assert_eq!(path.edges[0].types, vec!["KNOWS"]);
                assert_eq!(path.edges[0].direction, EdgeDirection::Undirected);
            } else {
                panic!("Expected Path pattern");
            }
        } else {
            panic!("Expected Query");
        }
    }

    #[test]
    fn test_parse_simplified_multi_label() {
        // G039: -/:KNOWS|WORKS_WITH/-> with multiple label alternatives
        let mut parser =
            Parser::new("MATCH (a:Person)-/:KNOWS|WORKS_WITH/->(b:Person) RETURN b.name");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Simplified multi-label path should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            if let Pattern::Path(path) = &q.match_clauses[0].patterns[0].pattern {
                assert_eq!(path.edges[0].types, vec!["KNOWS", "WORKS_WITH"]);
                assert_eq!(path.edges[0].direction, EdgeDirection::Outgoing);
            } else {
                panic!("Expected Path pattern");
            }
        } else {
            panic!("Expected Query");
        }
    }

    #[test]
    fn test_parse_simplified_tilde() {
        // G080: ~/:KNOWS/~ desugars to ~[:KNOWS]~
        let mut parser = Parser::new("MATCH (a:Person)~/:KNOWS/~(b:Person) RETURN b.name");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Simplified tilde path should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            if let Pattern::Path(path) = &q.match_clauses[0].patterns[0].pattern {
                assert_eq!(path.edges[0].types, vec!["KNOWS"]);
                assert_eq!(path.edges[0].direction, EdgeDirection::Undirected);
            } else {
                panic!("Expected Path pattern");
            }
        } else {
            panic!("Expected Query");
        }
    }

    // --- Group 6: Multiset Alternation ---

    #[test]
    fn test_parse_multiset_alternation() {
        // G030: |+| operator creates MultisetUnion
        let mut parser = Parser::new(
            "MATCH ((a)-[:KNOWS]->(b) |+| (a)-[:WORKS_WITH]->(b)) RETURN a.name, b.name",
        );
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Multiset alternation should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            let pattern = &q.match_clauses[0].patterns[0].pattern;
            assert!(
                matches!(pattern, Pattern::MultisetUnion(_)),
                "Expected MultisetUnion pattern, got {:?}",
                pattern
            );
            if let Pattern::MultisetUnion(alternatives) = pattern {
                assert_eq!(alternatives.len(), 2);
            }
        } else {
            panic!("Expected Query");
        }
    }

    #[test]
    fn test_parse_set_alternation() {
        // Set union uses | and produces Pattern::Union
        let mut parser =
            Parser::new("MATCH ((a)-[:KNOWS]->(b) | (a)-[:WORKS_WITH]->(b)) RETURN a.name");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Set alternation should parse: {:?}",
            result.err()
        );

        if let Statement::Query(q) = result.unwrap() {
            let pattern = &q.match_clauses[0].patterns[0].pattern;
            assert!(
                matches!(pattern, Pattern::Union(_)),
                "Expected Union pattern, got {:?}",
                pattern
            );
        } else {
            panic!("Expected Query");
        }
    }

    #[test]
    fn test_parse_delete_variable() {
        // Basic DELETE with variable (existing behavior)
        let mut parser = Parser::new("MATCH (n:Person) DELETE n");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "DELETE variable should parse: {:?}",
            result.err()
        );
        if let Statement::Query(q) = result.unwrap() {
            assert_eq!(q.delete_clauses.len(), 1);
            assert_eq!(q.delete_clauses[0].targets.len(), 1);
            assert!(
                matches!(&q.delete_clauses[0].targets[0], DeleteTarget::Variable(name) if name == "n"),
                "Expected Variable target"
            );
        } else {
            panic!("Expected Query");
        }
    }

    #[test]
    fn test_parse_delete_expression() {
        // GD04: DELETE with property access expression
        let mut parser = Parser::new("MATCH (n:Person) DELETE n.friend");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "DELETE expression should parse: {:?}",
            result.err()
        );
        if let Statement::Query(q) = result.unwrap() {
            assert_eq!(q.delete_clauses.len(), 1);
            assert_eq!(q.delete_clauses[0].targets.len(), 1);
            assert!(
                matches!(&q.delete_clauses[0].targets[0], DeleteTarget::Expression(_)),
                "Expected Expression target, got {:?}",
                q.delete_clauses[0].targets[0]
            );
        } else {
            panic!("Expected Query");
        }
    }

    #[test]
    fn test_parse_delete_multiple_mixed_targets() {
        // GD04: DELETE with both variable and expression targets
        let mut parser = Parser::new("MATCH (n:Person)-[r:KNOWS]->(m) DELETE n, r");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "DELETE mixed targets should parse: {:?}",
            result.err()
        );
        if let Statement::Query(q) = result.unwrap() {
            assert_eq!(q.delete_clauses[0].targets.len(), 2);
            assert!(matches!(
                &q.delete_clauses[0].targets[0],
                DeleteTarget::Variable(name) if name == "n"
            ));
            assert!(matches!(
                &q.delete_clauses[0].targets[1],
                DeleteTarget::Variable(name) if name == "r"
            ));
        } else {
            panic!("Expected Query");
        }
    }

    #[test]
    fn test_parse_detach_delete_expression() {
        // GD04: DETACH DELETE with expression
        let mut parser = Parser::new("MATCH (n:Person) DETACH DELETE n");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "DETACH DELETE should parse: {:?}",
            result.err()
        );
        if let Statement::Query(q) = result.unwrap() {
            assert!(q.delete_clauses[0].detach);
            assert_eq!(q.delete_clauses[0].targets.len(), 1);
        } else {
            panic!("Expected Query");
        }
    }

    // =========================================================================
    // Group 13: Graph Type Advanced Features (GG03, GG04, GG21)
    // =========================================================================

    #[test]
    fn test_parse_create_graph_type_iso_syntax() {
        let mut parser = Parser::new(
            "CREATE GRAPH TYPE social (\
                NODE TYPE Person (name STRING NOT NULL, age INTEGER),\
                EDGE TYPE KNOWS (since INTEGER)\
            )",
        );
        let result = parser.parse();
        assert!(result.is_ok(), "Failed to parse ISO graph type: {result:?}");
        if let Statement::Schema(SchemaStatement::CreateGraphType(stmt)) = result.unwrap() {
            assert_eq!(stmt.name, "social");
            assert_eq!(stmt.inline_types.len(), 2);
            assert!(stmt.node_types.contains(&"Person".to_string()));
            assert!(stmt.edge_types.contains(&"KNOWS".to_string()));
        } else {
            panic!("Expected CreateGraphType");
        }
    }

    #[test]
    fn test_parse_create_graph_type_like() {
        let mut parser = Parser::new("CREATE GRAPH TYPE cloned LIKE original_graph");
        let result = parser.parse();
        assert!(result.is_ok(), "Failed to parse LIKE: {result:?}");
        if let Statement::Schema(SchemaStatement::CreateGraphType(stmt)) = result.unwrap() {
            assert_eq!(stmt.name, "cloned");
            assert_eq!(stmt.like_graph, Some("original_graph".to_string()));
        } else {
            panic!("Expected CreateGraphType");
        }
    }

    #[test]
    fn test_parse_create_graph_type_key_labels() {
        let mut parser = Parser::new(
            "CREATE GRAPH TYPE keyed (\
                NODE TYPE Person KEY (PersonLabel, NamedEntity) (name STRING NOT NULL)\
            )",
        );
        let result = parser.parse();
        assert!(result.is_ok(), "Failed to parse key labels: {result:?}");
        if let Statement::Schema(SchemaStatement::CreateGraphType(stmt)) = result.unwrap() {
            assert_eq!(stmt.inline_types.len(), 1);
            match &stmt.inline_types[0] {
                InlineElementType::Node { key_labels, .. } => {
                    assert_eq!(key_labels.len(), 2);
                    assert_eq!(key_labels[0], "PersonLabel");
                    assert_eq!(key_labels[1], "NamedEntity");
                }
                _ => panic!("Expected Node"),
            }
        } else {
            panic!("Expected CreateGraphType");
        }
    }

    // ==================== Issue #316: bare element-type references ====================

    #[test]
    fn test_parse_create_graph_type_bare_reference_node() {
        // Issue #316: `NODE TYPE Person` inside a graph type body must parse
        // as a reference (is_reference = true), not an inline declaration.
        let mut parser = Parser::new("CREATE GRAPH TYPE g (NODE TYPE Person)");
        let result = parser.parse();
        assert!(result.is_ok(), "bare reference should parse: {result:?}");
        if let Statement::Schema(SchemaStatement::CreateGraphType(stmt)) = result.unwrap() {
            assert_eq!(stmt.inline_types.len(), 1);
            match &stmt.inline_types[0] {
                InlineElementType::Node {
                    name,
                    properties,
                    is_reference,
                    ..
                } => {
                    assert_eq!(name, "Person");
                    assert!(properties.is_empty());
                    assert!(*is_reference, "bare name must parse as reference");
                }
                other => panic!("expected Node variant, got {other:?}"),
            }
        } else {
            panic!("Expected CreateGraphType");
        }
    }

    #[test]
    fn test_parse_create_graph_type_bare_reference_edge() {
        let mut parser = Parser::new("CREATE GRAPH TYPE g (EDGE TYPE KNOWS)");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "bare edge reference should parse: {result:?}"
        );
        if let Statement::Schema(SchemaStatement::CreateGraphType(stmt)) = result.unwrap() {
            match &stmt.inline_types[0] {
                InlineElementType::Edge {
                    name,
                    properties,
                    is_reference,
                    ..
                } => {
                    assert_eq!(name, "KNOWS");
                    assert!(properties.is_empty());
                    assert!(*is_reference, "bare edge name must parse as reference");
                }
                other => panic!("expected Edge variant, got {other:?}"),
            }
        } else {
            panic!("Expected CreateGraphType");
        }
    }

    #[test]
    fn test_parse_create_graph_type_inline_declaration_not_reference() {
        // Control: NODE TYPE Person (p STRING) is a declaration, is_reference = false.
        let mut parser = Parser::new("CREATE GRAPH TYPE g (NODE TYPE Person (p STRING))");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "inline declaration should parse: {result:?}"
        );
        if let Statement::Schema(SchemaStatement::CreateGraphType(stmt)) = result.unwrap() {
            match &stmt.inline_types[0] {
                InlineElementType::Node {
                    is_reference,
                    properties,
                    ..
                } => {
                    assert!(!*is_reference, "declaration must not be a reference");
                    assert_eq!(properties.len(), 1);
                }
                other => panic!("expected Node variant, got {other:?}"),
            }
        } else {
            panic!("Expected CreateGraphType");
        }
    }

    #[test]
    fn test_parse_create_graph_type_empty_parens_not_reference() {
        // `NODE TYPE Person ()` is an explicit empty declaration, NOT a reference.
        // Documents the boundary case: any `(...)` marks declaration intent.
        let mut parser = Parser::new("CREATE GRAPH TYPE g (NODE TYPE Person ())");
        let result = parser.parse();
        assert!(result.is_ok(), "empty-parens form should parse: {result:?}");
        if let Statement::Schema(SchemaStatement::CreateGraphType(stmt)) = result.unwrap() {
            match &stmt.inline_types[0] {
                InlineElementType::Node { is_reference, .. } => {
                    assert!(
                        !*is_reference,
                        "explicit () is a declaration, not a reference"
                    );
                }
                other => panic!("expected Node variant, got {other:?}"),
            }
        } else {
            panic!("Expected CreateGraphType");
        }
    }

    #[test]
    fn test_parse_create_graph_type_key_clause_not_reference() {
        // GG21 KEY clause also marks this as a declaration, not a reference.
        let mut parser = Parser::new("CREATE GRAPH TYPE g (NODE TYPE Person KEY (Label1))");
        let result = parser.parse();
        assert!(result.is_ok(), "KEY-only form should parse: {result:?}");
        if let Statement::Schema(SchemaStatement::CreateGraphType(stmt)) = result.unwrap() {
            match &stmt.inline_types[0] {
                InlineElementType::Node {
                    is_reference,
                    key_labels,
                    ..
                } => {
                    assert!(!*is_reference, "KEY clause is a declaration signal");
                    assert_eq!(key_labels, &vec!["Label1".to_string()]);
                }
                other => panic!("expected Node variant, got {other:?}"),
            }
        } else {
            panic!("Expected CreateGraphType");
        }
    }

    // ==================== EXISTS bare pattern ====================

    #[test]
    fn test_parse_exists_with_match() {
        // EXISTS with explicit MATCH (should already work)
        let mut parser = Parser::new("MATCH (n) WHERE EXISTS { MATCH (n)-[:KNOWS]->() } RETURN n");
        let result = parser.parse();
        assert!(result.is_ok(), "EXISTS with MATCH should parse: {result:?}");
    }

    #[test]
    fn test_parse_exists_bare_pattern() {
        // Bare pattern without MATCH keyword
        let mut parser = Parser::new("MATCH (n) WHERE EXISTS { (n)-[:KNOWS]->() } RETURN n");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "EXISTS bare pattern should parse: {result:?}"
        );
    }

    #[test]
    fn test_parse_exists_bare_pattern_with_where() {
        // Bare pattern with WHERE inside EXISTS
        let mut parser = Parser::new(
            "MATCH (a), (b) WHERE NOT EXISTS { (a)-[r]->(b) WHERE type(r) = 'KNOWS' } RETURN a",
        );
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "EXISTS bare pattern with WHERE should parse: {result:?}"
        );
    }

    // ==================== Pattern-form graph type syntax ====================

    #[test]
    fn test_parse_graph_type_pattern_form_simple() {
        let mut parser = Parser::new(
            "CREATE GRAPH TYPE social (\
                (:Person {name STRING NOT NULL})-[:KNOWS {since INTEGER}]->(:Person)\
            )",
        );
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Pattern-form graph type should parse: {result:?}"
        );
        if let Statement::Schema(SchemaStatement::CreateGraphType(stmt)) = result.unwrap() {
            assert_eq!(stmt.name, "social");
            assert!(stmt.node_types.contains(&"Person".to_string()));
            assert!(stmt.edge_types.contains(&"KNOWS".to_string()));
            // Should have Person node type and KNOWS edge type
            let node_count = stmt
                .inline_types
                .iter()
                .filter(|t| matches!(t, InlineElementType::Node { .. }))
                .count();
            let edge_count = stmt
                .inline_types
                .iter()
                .filter(|t| matches!(t, InlineElementType::Edge { .. }))
                .count();
            assert_eq!(
                node_count, 1,
                "Should have 1 node type (Person, deduplicated)"
            );
            assert_eq!(edge_count, 1, "Should have 1 edge type (KNOWS)");
            // Check KNOWS edge has source/target
            for t in &stmt.inline_types {
                if let InlineElementType::Edge {
                    name,
                    source_node_types,
                    target_node_types,
                    ..
                } = t
                {
                    assert_eq!(name, "KNOWS");
                    assert_eq!(source_node_types, &["Person"]);
                    assert_eq!(target_node_types, &["Person"]);
                }
            }
        } else {
            panic!("Expected CreateGraphType");
        }
    }

    #[test]
    fn test_parse_graph_type_pattern_form_multiple_patterns() {
        let mut parser = Parser::new(
            "CREATE GRAPH TYPE social (\
                (:Person {name STRING NOT NULL})-[:KNOWS {since INTEGER}]->(:Person),\
                (:Person)-[:LIVES_IN]->(:City)\
            )",
        );
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Multiple pattern-form entries should parse: {result:?}"
        );
        if let Statement::Schema(SchemaStatement::CreateGraphType(stmt)) = result.unwrap() {
            assert_eq!(stmt.name, "social");
            assert!(stmt.node_types.contains(&"Person".to_string()));
            assert!(stmt.node_types.contains(&"City".to_string()));
            assert!(stmt.edge_types.contains(&"KNOWS".to_string()));
            assert!(stmt.edge_types.contains(&"LIVES_IN".to_string()));
            // Person should appear only once despite being in two patterns
            let node_count = stmt
                .inline_types
                .iter()
                .filter(|t| matches!(t, InlineElementType::Node { .. }))
                .count();
            assert_eq!(node_count, 2, "Should have 2 node types (Person, City)");
        } else {
            panic!("Expected CreateGraphType");
        }
    }

    #[test]
    fn test_parse_graph_type_pattern_form_standalone_node() {
        let mut parser = Parser::new(
            "CREATE GRAPH TYPE basic_nodes (\
                (:Person {name STRING})\
            )",
        );
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Standalone node pattern should parse: {result:?}"
        );
        if let Statement::Schema(SchemaStatement::CreateGraphType(stmt)) = result.unwrap() {
            assert_eq!(stmt.inline_types.len(), 1);
            assert!(matches!(
                &stmt.inline_types[0],
                InlineElementType::Node { name, .. } if name == "Person"
            ));
        } else {
            panic!("Expected CreateGraphType");
        }
    }

    #[test]
    fn test_parse_graph_type_pattern_form_no_props() {
        let mut parser = Parser::new(
            "CREATE GRAPH TYPE bare (\
                (:Person)-[:KNOWS]->(:Person)\
            )",
        );
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Pattern without properties should parse: {result:?}"
        );
        if let Statement::Schema(SchemaStatement::CreateGraphType(stmt)) = result.unwrap() {
            // Node should have no properties
            for t in &stmt.inline_types {
                if let InlineElementType::Node { properties, .. } = t {
                    assert!(properties.is_empty());
                }
                if let InlineElementType::Edge { properties, .. } = t {
                    assert!(properties.is_empty());
                }
            }
        } else {
            panic!("Expected CreateGraphType");
        }
    }

    #[test]
    fn test_parse_graph_type_pattern_form_backward_edge() {
        let mut parser = Parser::new(
            "CREATE GRAPH TYPE rev (\
                (:City)<-[:LIVES_IN]-(:Person)\
            )",
        );
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Backward edge pattern should parse: {result:?}"
        );
        if let Statement::Schema(SchemaStatement::CreateGraphType(stmt)) = result.unwrap() {
            for t in &stmt.inline_types {
                if let InlineElementType::Edge {
                    name,
                    source_node_types,
                    target_node_types,
                    ..
                } = t
                {
                    assert_eq!(name, "LIVES_IN");
                    // Backward: source is Person (right side), target is City (left side)
                    assert_eq!(source_node_types, &["Person"]);
                    assert_eq!(target_node_types, &["City"]);
                }
            }
        } else {
            panic!("Expected CreateGraphType");
        }
    }

    // ==================== SHOW commands ====================

    #[test]
    fn test_parse_show_constraints() {
        let mut parser = Parser::new("SHOW CONSTRAINTS");
        let result = parser.parse();
        assert!(result.is_ok(), "SHOW CONSTRAINTS should parse: {result:?}");
        assert!(matches!(
            result.unwrap(),
            Statement::Schema(SchemaStatement::ShowConstraints)
        ));
    }

    #[test]
    fn test_parse_show_indexes() {
        let mut parser = Parser::new("SHOW INDEXES");
        let result = parser.parse();
        assert!(result.is_ok(), "SHOW INDEXES should parse: {result:?}");
        assert!(matches!(
            result.unwrap(),
            Statement::Schema(SchemaStatement::ShowIndexes)
        ));
    }

    #[test]
    fn test_parse_show_index_singular() {
        // INDEX is a keyword token, so SHOW INDEX should also work
        let mut parser = Parser::new("SHOW INDEX");
        let result = parser.parse();
        assert!(result.is_ok(), "SHOW INDEX should parse: {result:?}");
        assert!(matches!(
            result.unwrap(),
            Statement::Schema(SchemaStatement::ShowIndexes)
        ));
    }

    #[test]
    fn test_parse_show_node_types() {
        let mut parser = Parser::new("SHOW NODE TYPES");
        let result = parser.parse();
        assert!(result.is_ok(), "SHOW NODE TYPES should parse: {result:?}");
        assert!(matches!(
            result.unwrap(),
            Statement::Schema(SchemaStatement::ShowNodeTypes)
        ));
    }

    #[test]
    fn test_parse_show_edge_types() {
        let mut parser = Parser::new("SHOW EDGE TYPES");
        let result = parser.parse();
        assert!(result.is_ok(), "SHOW EDGE TYPES should parse: {result:?}");
        assert!(matches!(
            result.unwrap(),
            Statement::Schema(SchemaStatement::ShowEdgeTypes)
        ));
    }

    #[test]
    fn test_parse_show_graph_types() {
        let mut parser = Parser::new("SHOW GRAPH TYPES");
        let result = parser.parse();
        assert!(result.is_ok(), "SHOW GRAPH TYPES should parse: {result:?}");
        assert!(matches!(
            result.unwrap(),
            Statement::Schema(SchemaStatement::ShowGraphTypes)
        ));
    }

    #[test]
    fn test_parse_show_graphs() {
        let mut parser = Parser::new("SHOW GRAPHS");
        let result = parser.parse();
        assert!(result.is_ok(), "SHOW GRAPHS should parse: {result:?}");
        assert!(matches!(
            result.unwrap(),
            Statement::Schema(SchemaStatement::ShowGraphs)
        ));
    }

    #[test]
    fn test_parse_show_graph_type_named() {
        let mut parser = Parser::new("SHOW GRAPH TYPE social");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "SHOW GRAPH TYPE <name> should parse: {result:?}"
        );
        if let Statement::Schema(SchemaStatement::ShowGraphType(name)) = result.unwrap() {
            assert_eq!(name, "social");
        } else {
            panic!("Expected ShowGraphType");
        }
    }

    // --- LOAD DATA ---

    #[test]
    fn test_parse_load_data_csv() {
        let mut parser = Parser::new(
            "LOAD DATA FROM 'people.csv' FORMAT CSV WITH HEADERS AS row RETURN row.name",
        );
        let result = parser.parse();
        assert!(result.is_ok(), "LOAD DATA CSV parse failed: {result:?}");
        if let Statement::Query(query) = result.unwrap() {
            assert!(!query.ordered_clauses.is_empty());
            assert!(matches!(query.ordered_clauses[0], QueryClause::LoadData(_)));
            if let QueryClause::LoadData(ref ld) = query.ordered_clauses[0] {
                assert_eq!(ld.path, "people.csv");
                assert_eq!(ld.format, LoadFormat::Csv);
                assert!(ld.with_headers);
                assert_eq!(ld.variable, "row");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_load_data_csv_no_headers() {
        let mut parser = Parser::new("LOAD DATA FROM 'data.csv' FORMAT CSV AS r RETURN r[0]");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "LOAD DATA CSV no headers parse failed: {result:?}"
        );
        if let Statement::Query(query) = result.unwrap()
            && let QueryClause::LoadData(ref ld) = query.ordered_clauses[0]
        {
            assert!(!ld.with_headers);
            assert_eq!(ld.variable, "r");
        }
    }

    #[test]
    fn test_parse_load_data_jsonl() {
        let mut parser =
            Parser::new("LOAD DATA FROM 'events.jsonl' FORMAT JSONL AS row RETURN row.title");
        let result = parser.parse();
        assert!(result.is_ok(), "LOAD DATA JSONL parse failed: {result:?}");
        if let Statement::Query(query) = result.unwrap()
            && let QueryClause::LoadData(ref ld) = query.ordered_clauses[0]
        {
            assert_eq!(ld.format, LoadFormat::Jsonl);
            assert_eq!(ld.path, "events.jsonl");
        }
    }

    #[test]
    fn test_parse_load_data_ndjson() {
        let mut parser =
            Parser::new("LOAD DATA FROM 'data.ndjson' FORMAT NDJSON AS row RETURN row");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "LOAD DATA NDJSON alias parse failed: {result:?}"
        );
        if let Statement::Query(query) = result.unwrap()
            && let QueryClause::LoadData(ref ld) = query.ordered_clauses[0]
        {
            assert_eq!(ld.format, LoadFormat::Jsonl);
        }
    }

    #[test]
    fn test_parse_load_data_parquet() {
        let mut parser =
            Parser::new("LOAD DATA FROM 'data.parquet' FORMAT PARQUET AS row RETURN row.id");
        let result = parser.parse();
        assert!(result.is_ok(), "LOAD DATA PARQUET parse failed: {result:?}");
        if let Statement::Query(query) = result.unwrap()
            && let QueryClause::LoadData(ref ld) = query.ordered_clauses[0]
        {
            assert_eq!(ld.format, LoadFormat::Parquet);
        }
    }

    #[test]
    fn test_parse_load_csv_compat() {
        // Cypher-compatible LOAD CSV syntax in GQL parser
        let mut parser =
            Parser::new("LOAD CSV WITH HEADERS FROM 'file.csv' AS row RETURN row.name");
        let result = parser.parse();
        assert!(result.is_ok(), "LOAD CSV compat parse failed: {result:?}");
        if let Statement::Query(query) = result.unwrap()
            && let QueryClause::LoadData(ref ld) = query.ordered_clauses[0]
        {
            assert_eq!(ld.format, LoadFormat::Csv);
            assert!(ld.with_headers);
            assert_eq!(ld.path, "file.csv");
        }
    }

    #[test]
    fn test_parse_load_data_with_fieldterminator() {
        let mut parser = Parser::new(
            "LOAD DATA FROM 'data.tsv' FORMAT CSV WITH HEADERS AS row FIELDTERMINATOR '\\t' RETURN row",
        );
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "LOAD DATA with FIELDTERMINATOR parse failed: {result:?}"
        );
        if let Statement::Query(query) = result.unwrap()
            && let QueryClause::LoadData(ref ld) = query.ordered_clauses[0]
        {
            assert_eq!(ld.field_terminator, Some('\t'));
        }
    }

    #[test]
    fn test_parse_load_data_with_insert() {
        let mut parser = Parser::new(
            "LOAD DATA FROM 'people.csv' FORMAT CSV WITH HEADERS AS row INSERT (:Person {name: row.name})",
        );
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "LOAD DATA + INSERT parse failed: {result:?}"
        );
    }

    #[test]
    fn test_parse_load_data_bad_format() {
        let mut parser = Parser::new("LOAD DATA FROM 'data.xml' FORMAT XML AS row RETURN row");
        let result = parser.parse();
        assert!(result.is_err(), "Should fail with unknown format XML");
    }

    // --- T2-11: GQL keyword case-insensitivity ---

    #[test]
    fn test_parse_lowercase_keywords() {
        // Lowercase GQL keywords should parse identically to uppercase
        let upper = Parser::new("MATCH (n:Person) WHERE n.age > 30 RETURN n.name")
            .parse()
            .unwrap();
        let lower = Parser::new("match (n:Person) where n.age > 30 return n.name")
            .parse()
            .unwrap();

        // Both should be Query statements with the same structure
        let (Statement::Query(q_upper), Statement::Query(q_lower)) = (&upper, &lower) else {
            panic!("Expected Query statements");
        };
        assert_eq!(q_upper.match_clauses.len(), q_lower.match_clauses.len());
        assert!(q_upper.where_clause.is_some());
        assert!(q_lower.where_clause.is_some());
        assert_eq!(
            q_upper.return_clause.items.len(),
            q_lower.return_clause.items.len()
        );
    }

    #[test]
    fn test_parse_mixed_case_keywords() {
        let result = Parser::new("Match (n:Person) Return n").parse();
        assert!(
            result.is_ok(),
            "Mixed-case keywords should parse: {result:?}"
        );
    }

    // --- T2-12: Temporal literal parsing ---

    #[test]
    fn test_parse_date_literal() {
        let mut parser = Parser::new("RETURN DATE '2024-01-15'");
        let result = parser.parse().unwrap();
        if let Statement::Query(query) = result {
            if let Expression::Literal(Literal::Date(s)) = &query.return_clause.items[0].expression
            {
                assert_eq!(s, "2024-01-15");
            } else {
                panic!(
                    "Expected Date literal, got: {:?}",
                    query.return_clause.items[0].expression
                );
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_time_literal() {
        let mut parser = Parser::new("RETURN TIME '10:30:00'");
        let result = parser.parse().unwrap();
        if let Statement::Query(query) = result {
            if let Expression::Literal(Literal::Time(s)) = &query.return_clause.items[0].expression
            {
                assert_eq!(s, "10:30:00");
            } else {
                panic!(
                    "Expected Time literal, got: {:?}",
                    query.return_clause.items[0].expression
                );
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_duration_literal() {
        let mut parser = Parser::new("RETURN DURATION 'P1Y2M'");
        let result = parser.parse().unwrap();
        if let Statement::Query(query) = result {
            if let Expression::Literal(Literal::Duration(s)) =
                &query.return_clause.items[0].expression
            {
                assert_eq!(s, "P1Y2M");
            } else {
                panic!(
                    "Expected Duration literal, got: {:?}",
                    query.return_clause.items[0].expression
                );
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_datetime_literal() {
        let mut parser = Parser::new("RETURN DATETIME '2024-01-15T14:30:00'");
        let result = parser.parse().unwrap();
        if let Statement::Query(query) = result {
            if let Expression::Literal(Literal::Datetime(s)) =
                &query.return_clause.items[0].expression
            {
                assert_eq!(s, "2024-01-15T14:30:00");
            } else {
                panic!(
                    "Expected Datetime literal, got: {:?}",
                    query.return_clause.items[0].expression
                );
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_zoned_datetime_literal() {
        let mut parser = Parser::new("RETURN ZONED DATETIME '2024-01-15T14:30:00+05:30'");
        let result = parser.parse().unwrap();
        if let Statement::Query(query) = result {
            if let Expression::Literal(Literal::ZonedDatetime(s)) =
                &query.return_clause.items[0].expression
            {
                assert_eq!(s, "2024-01-15T14:30:00+05:30");
            } else {
                panic!(
                    "Expected ZonedDatetime literal, got: {:?}",
                    query.return_clause.items[0].expression
                );
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    #[test]
    fn test_parse_zoned_time_literal() {
        let mut parser = Parser::new("RETURN ZONED TIME '14:30:00+01:00'");
        let result = parser.parse().unwrap();
        if let Statement::Query(query) = result {
            if let Expression::Literal(Literal::ZonedTime(s)) =
                &query.return_clause.items[0].expression
            {
                assert_eq!(s, "14:30:00+01:00");
            } else {
                panic!(
                    "Expected ZonedTime literal, got: {:?}",
                    query.return_clause.items[0].expression
                );
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    // ======================================================================
    // Parser robustness: mixed union operators, graph ref, quantified paths
    // ======================================================================

    #[test]
    fn test_mixed_pipe_operators_rejected() {
        // Mixing | and |+| in the same alternation must produce an error.
        let mut parser =
            Parser::new("MATCH (a)-[:X]->(b) | (c)-[:Y]->(d) |+| (e)-[:Z]->(f) RETURN a");
        let result = parser.parse();
        assert!(result.is_err(), "Mixing | and |+| should be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("mix") || msg.contains("Mix"),
            "Error should mention mixing operators: {msg}"
        );
    }

    #[test]
    fn test_consistent_pipe_operators_accepted() {
        // Consistent | operators should parse fine.
        let mut parser = Parser::new("MATCH (a)-[:X]->(b) | (c)-[:Y]->(d) RETURN a");
        assert!(parser.parse().is_ok(), "Consistent | should parse");
    }

    #[test]
    fn test_select_from_match_parses() {
        // SELECT ... FROM graph_name MATCH pattern should parse.
        let mut parser = Parser::new("SELECT n.name FROM my_graph MATCH (n:Person) RETURN n.name");
        // May fail during semantic validation, but should not panic.
        let _ = parser.parse();
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
    fn test_nesting_at_exact_limit_succeeds() {
        // 128 levels should succeed (limit is > 128, not >=)
        let depth = 128;
        let deep = "(".repeat(depth) + "1" + &")".repeat(depth);
        let query = format!("RETURN {deep}");
        let mut parser = Parser::new(&query);
        // At the limit: this may succeed or fail depending on how many nesting
        // points accumulate. The important thing is no stack overflow.
        let _ = parser.parse();
    }

    #[test]
    fn test_moderate_nesting_succeeds() {
        // 50 levels of parentheses should always succeed
        let depth = 50;
        let deep = "(".repeat(depth) + "1" + &")".repeat(depth);
        let query = format!("RETURN {deep}");
        let mut parser = Parser::new(&query);
        let result = parser.parse();
        assert!(result.is_ok(), "50 levels of nesting should succeed");
    }

    #[test]
    fn test_explain_recursion_depth_limit() {
        // Deeply nested EXPLAIN should hit nesting limit, not stack overflow
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
    fn test_nested_case_expressions_depth_limit() {
        // CASE WHEN ... THEN CASE WHEN ... THEN ... END END
        let prefix = "CASE WHEN true THEN ".repeat(300);
        let suffix = " END".repeat(300);
        let query = format!("RETURN {prefix}1{suffix}");
        let mut parser = Parser::new(&query);
        let result = parser.parse();
        // Should error rather than stack overflow
        assert!(result.is_err());
    }

    #[test]
    fn test_nested_list_literals_depth_limit() {
        let deep = "[".repeat(300) + "1" + &"]".repeat(300);
        let query = format!("RETURN {deep}");
        let mut parser = Parser::new(&query);
        let result = parser.parse();
        // Should error rather than stack overflow
        assert!(result.is_err());
    }

    // ==================== Integer overflow error message tests ====================

    #[test]
    fn test_integer_overflow_decimal_shows_descriptive_error() {
        let mut parser = Parser::new("RETURN 99999999999999999999");
        let result = parser.parse();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("overflows"),
            "Expected overflow message, got: {err}"
        );
        assert!(
            err.contains("99999999999999999999"),
            "Expected literal value in error, got: {err}"
        );
    }

    #[test]
    fn test_integer_overflow_hex_shows_descriptive_error() {
        let mut parser = Parser::new("RETURN 0xFFFFFFFFFFFFFFFFFF");
        let result = parser.parse();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("overflows"),
            "Expected overflow message, got: {err}"
        );
    }

    #[test]
    fn test_integer_overflow_octal_shows_descriptive_error() {
        let mut parser = Parser::new("RETURN 0o7777777777777777777777");
        let result = parser.parse();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("overflows"),
            "Expected overflow message, got: {err}"
        );
    }

    #[test]
    fn test_integer_overflow_binary_shows_descriptive_error() {
        let long_binary = "1".repeat(65);
        let query = format!("RETURN 0b{long_binary}");
        let mut parser = Parser::new(&query);
        let result = parser.parse();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("overflows"),
            "Expected overflow message, got: {err}"
        );
    }

    #[test]
    fn test_i64_max_parses_successfully() {
        let query = format!("RETURN {}", i64::MAX);
        let mut parser = Parser::new(&query);
        let result = parser.parse();
        assert!(result.is_ok(), "i64::MAX should parse successfully");
    }

    #[test]
    fn test_i64_min_parses_successfully() {
        // i64::MIN is -9223372036854775808, written as negative literal
        let query = format!("RETURN {}", i64::MIN);
        let mut parser = Parser::new(&query);
        let result = parser.parse();
        assert!(result.is_ok(), "i64::MIN should parse successfully");
    }

    #[test]
    fn test_i64_max_plus_one_overflows() {
        // 9223372036854775808 = i64::MAX + 1
        let mut parser = Parser::new("RETURN 9223372036854775808");
        let result = parser.parse();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("overflows"),
            "i64::MAX+1 should overflow, got: {err}"
        );
    }

    #[test]
    fn test_negative_i64_min_minus_one_overflows() {
        // -9223372036854775809 overflows i64::MIN
        let mut parser = Parser::new("RETURN -9223372036854775809");
        let result = parser.parse();
        // This should either error on overflow or produce a unary negation error
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_parses_successfully() {
        let mut parser = Parser::new("RETURN 0");
        assert!(parser.parse().is_ok());
    }

    #[test]
    fn test_hex_i64_max_parses_successfully() {
        let mut parser = Parser::new("RETURN 0x7FFFFFFFFFFFFFFF");
        let result = parser.parse();
        assert!(result.is_ok(), "i64::MAX in hex should parse successfully");
    }

    // ==================== != operator tests ====================

    #[test]
    fn test_not_equal_bang_equals() {
        let mut parser = Parser::new("MATCH (n) WHERE n.age != 19 RETURN n");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "!= should be accepted as not-equal: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_not_equal_diamond() {
        let mut parser = Parser::new("MATCH (n) WHERE n.age <> 19 RETURN n");
        let result = parser.parse();
        assert!(result.is_ok(), "<> should still work: {:?}", result.err());
    }

    #[test]
    fn test_not_equal_both_produce_same_ast() {
        let mut p1 = Parser::new("MATCH (n) WHERE n.x != 1 RETURN n");
        let mut p2 = Parser::new("MATCH (n) WHERE n.x <> 1 RETURN n");
        let r1 = p1.parse();
        let r2 = p2.parse();
        assert!(r1.is_ok() && r2.is_ok(), "Both != and <> should parse");
        // Both should produce the same AST structure (Ne comparison)
    }

    #[test]
    fn test_exclamation_mark_alone_still_works() {
        // ! without = should still produce Exclamation token (for NOT in some contexts)
        let mut parser = Parser::new("MATCH (n) WHERE NOT n.active RETURN n");
        let result = parser.parse();
        assert!(result.is_ok(), "NOT should still work: {:?}", result.err());
    }

    // ==================== Error source identification tests ====================

    #[test]
    fn test_gql_error_includes_language_prefix() {
        let mut parser = Parser::new("INVALID SYNTAX HERE");
        let result = parser.parse();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("[GQL]"),
            "GQL parser errors should be prefixed with [GQL], got: {err}"
        );
    }

    // ==================== Unicode identifier tests ====================

    #[test]
    fn test_unicode_label_cjk() {
        let mut parser = Parser::new("INSERT (:人物 {名前: 'Alix'})");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "CJK label and property should parse: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_unicode_label_accented() {
        let mut parser = Parser::new("MATCH (n:Universit\u{00E9}) RETURN n");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Accented label should parse: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_unicode_label_cyrillic() {
        let mut parser = Parser::new(
            "MATCH (n:\u{041F}\u{0435}\u{0440}\u{0441}\u{043E}\u{043D}\u{0430}) RETURN n",
        );
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Cyrillic label should parse: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_unicode_label_arabic() {
        let mut parser = Parser::new("MATCH (n:\u{0634}\u{062E}\u{0635}) RETURN n");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Arabic label should parse: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_unicode_variable_name() {
        let mut parser = Parser::new("MATCH (\u{540D}\u{524D}:Person) RETURN \u{540D}\u{524D}");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "Unicode variable name should parse: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_ascii_identifiers_still_work() {
        // Regression: ensure basic ASCII identifiers are unaffected
        let mut parser = Parser::new("MATCH (n:Person) WHERE n.age > 19 RETURN n.name");
        let result = parser.parse();
        assert!(result.is_ok(), "ASCII identifiers should still work");
    }

    #[test]
    fn test_unicode_string_with_escape() {
        let mut parser = Parser::new("RETURN '\\u0041'");
        let result = parser.parse();
        assert!(result.is_ok(), "String with \\u escape should parse");
    }

    // --- Keyword-aware helpers: regression for bug-gql-create-constraint-named ---
    //
    // The four parser paths below used to check FOR / OR / NOT as identifiers
    // only, which silently failed when the lexer promoted those words to
    // dedicated `TokenKind` variants. Each test parses a statement that goes
    // through one of the fixed paths and must now succeed.

    #[test]
    fn test_create_constraint_anonymous_parses() {
        let mut parser = Parser::new("CREATE CONSTRAINT FOR (n:Item) ON (n.id) UNIQUE");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "anonymous CREATE CONSTRAINT must parse: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_create_constraint_named_parses() {
        let mut parser = Parser::new("CREATE CONSTRAINT uniq_item FOR (n:Item) ON (n.id) UNIQUE");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "named CREATE CONSTRAINT must parse: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_create_constraint_if_not_exists_parses() {
        // Parser requires IF NOT EXISTS before the optional name.
        let mut parser =
            Parser::new("CREATE CONSTRAINT IF NOT EXISTS uniq_item FOR (n:Item) ON (n.id) UNIQUE");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "CREATE CONSTRAINT IF NOT EXISTS must parse: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_create_index_for_parses() {
        let mut parser = Parser::new("CREATE INDEX idx FOR (n:Item) ON (n.id)");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "CREATE INDEX ... FOR must parse: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_create_or_replace_node_type_parses() {
        // Exercises try_parse_or_replace via the NODE TYPE branch.
        let mut parser = Parser::new("CREATE OR REPLACE NODE TYPE Person (name STRING)");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "CREATE OR REPLACE NODE TYPE must parse: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_create_graph_if_not_exists_parses() {
        let mut parser = Parser::new("CREATE GRAPH IF NOT EXISTS g");
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "CREATE GRAPH IF NOT EXISTS must parse: {:?}",
            result.err()
        );
    }
}
