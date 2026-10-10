//! GQL Parser.

#[allow(clippy::wildcard_imports)]
use super::ast::*;
use super::lexer::{Lexer, Token, TokenKind};
use crate::query::keywords::unescape_string;
use crate::query::limits::{Chain, Nesting, nesting_error_message};
use grafeo_common::storage::value_codec::MAX_PROPERTY_VALUE_DEPTH;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result, SourceSpan};

/// A graph type body: the names of its node types and of its edge types, the
/// element types it declares, and whether it is open.
type GraphTypeBody = (Vec<String>, Vec<String>, Vec<InlineElementType>, bool);

/// The names of the node types and of the edge types among `elements`, in
/// order.
fn element_type_names(elements: &[InlineElementType]) -> (Vec<String>, Vec<String>) {
    let mut node_types = Vec::new();
    let mut edge_types = Vec::new();
    for element in elements {
        match element {
            InlineElementType::Node { name, .. } => node_types.push(name.clone()),
            InlineElementType::Edge { name, .. } => edge_types.push(name.clone()),
        }
    }
    (node_types, edge_types)
}

/// GQL Parser.
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
    peeked: Option<Token>,
    peeked_second: Option<Token>,
    source: &'a str,
    /// How deep the statement parsed so far nests (see
    /// [`MAX_NESTING_DEPTH`](crate::query::limits::MAX_NESTING_DEPTH)).
    nesting: Nesting,
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
            nesting: Nesting::default(),
        }
    }

    /// Enters one level of nesting, or fails past the nesting limit.
    fn enter_nesting(&mut self) -> Result<()> {
        if self.nesting.enter() {
            Ok(())
        } else {
            Err(self.error(&nesting_error_message()))
        }
    }

    /// Leaves the level [`Self::enter_nesting`] entered.
    fn exit_nesting(&mut self) {
        self.nesting.exit();
    }

    /// Enters a subquery, which nests two levels: translating and planning
    /// one takes more stack than an expression in parentheses.
    fn enter_subquery(&mut self) -> Result<()> {
        self.enter_nesting()?;
        self.enter_nesting()
    }

    /// Leaves the levels [`Self::enter_subquery`] entered.
    fn exit_subquery(&mut self) {
        self.exit_nesting();
        self.exit_nesting();
    }

    /// Begins a chain of binary operators (see [`Nesting::begin_chain`]).
    fn begin_chain(&mut self) -> Chain {
        self.nesting.begin_chain()
    }

    /// Counts one operator of a chain, or fails past the nesting limit.
    fn link_chain(&mut self) -> Result<()> {
        if self.nesting.link() {
            Ok(())
        } else {
            Err(self.error(&nesting_error_message()))
        }
    }

    /// Ends a chain of binary operators.
    fn end_chain(&mut self, chain: Chain) {
        self.nesting.end_chain(chain);
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

        // NEXT and the set operators chain statements: each nests one level.
        let chain = self.begin_chain();
        let mut left = self.parse_single_statement()?;

        // Handle NEXT (linear composition): output of left becomes input of right
        while self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("NEXT") {
            self.advance(); // consume NEXT
            let right = self.parse_single_statement()?;
            self.link_chain()?;
            // NEXT semantics: chain right after left (like WITH pipe).
            // Represent as CompositeQuery with a dedicated op.
            left = Statement::CompositeQuery {
                left: Box::new(left),
                op: CompositeOp::Next,
                right: Box::new(right),
            };
        }

        // Check for composite query operators (UNION, EXCEPT, INTERSECT,
        // OTHERWISE). A run of one associative operator is a balanced tree:
        // a UNION of any number of queries stays shallow.
        let mut pending = self.parse_composite_op();
        while let Some(op) = pending {
            if !is_associative(op) {
                let right = self.parse_single_statement()?;
                self.link_chain()?;
                left = Statement::CompositeQuery {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                };
                pending = self.parse_composite_op();
                continue;
            }
            let mut operands = vec![(left, self.nesting.take_operand())];
            loop {
                let right = self.parse_single_statement()?;
                operands.push((right, self.nesting.take_operand()));
                pending = self.parse_composite_op();
                if pending != Some(op) {
                    break;
                }
            }
            left = self
                .nesting
                .join_balanced(operands, |left, right| Statement::CompositeQuery {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                })
                .ok_or_else(|| self.error(&nesting_error_message()))?;
        }
        self.end_chain(chain);

        Ok(left)
    }

    /// Consumes a set operator between two queries (`UNION [ALL | DISTINCT]`,
    /// `EXCEPT ...`, `INTERSECT ...`, `OTHERWISE`), or returns `None` when
    /// the current token starts none.
    fn parse_composite_op(&mut self) -> Option<CompositeOp> {
        let (distinct, all) = match self.current.kind {
            TokenKind::Union => (CompositeOp::Union, CompositeOp::UnionAll),
            TokenKind::Except => (CompositeOp::Except, CompositeOp::ExceptAll),
            TokenKind::Intersect => (CompositeOp::Intersect, CompositeOp::IntersectAll),
            TokenKind::Otherwise => {
                self.advance();
                return Some(CompositeOp::Otherwise);
            }
            _ => return None,
        };
        self.advance();
        if self.current.kind == TokenKind::All {
            self.advance();
            return Some(all);
        }
        // DISTINCT is the explicit form of the default
        if self.current.kind == TokenKind::Distinct {
            self.advance();
        }
        Some(distinct)
    }

    fn parse_single_statement(&mut self) -> Result<Statement> {
        match self.current.kind {
            // A linear statement may start with any simple query statement:
            // a FILTER or an order by and page statement too (LET below).
            TokenKind::Match
            | TokenKind::Optional
            | TokenKind::Unwind
            | TokenKind::Merge
            | TokenKind::For
            | TokenKind::Filter
            | TokenKind::Order
            | TokenKind::Offset
            | TokenKind::Skip
            | TokenKind::Limit
            | TokenKind::Fetch
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
                if matches!(self.peek_kind(), TokenKind::LBrace | TokenKind::LParen) {
                    // CALL [(scope)] { subquery } RETURN ... : treat as a query
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
                    "LOAD" | "LET" => self.parse_query().map(Statement::Query),
                    "SELECT" => {
                        self.advance(); // consume SELECT
                        self.parse_select_from_statement()
                    }
                    _ => Err(self.error(
                        "Expected MATCH, LET, FILTER, INSERT, DELETE, MERGE, UNWIND, FOR, CREATE, \
                         CALL, DROP, ALTER, SHOW, LOAD, USE, SESSION, START, COMMIT, ROLLBACK, \
                         SELECT, or SAVEPOINT",
                    )),
                }
            }
            _ => Err(self.error(
                "Expected MATCH, LET, FILTER, INSERT, DELETE, MERGE, UNWIND, FOR, CREATE, CALL, \
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

    /// Parses an inline CALL { subquery } and its variable scope clause, if
    /// any: `CALL (a, b) { ... }` sees the outer variables `a` and `b`,
    /// `CALL () { ... }` none, and `CALL { ... }` all of them. The body may
    /// combine queries with `UNION`, `EXCEPT`, `INTERSECT` or `OTHERWISE`.
    ///
    /// ```text
    /// CALL [( [var [, var]*] )] { query_body RETURN ... [UNION ...] }
    /// ```
    fn parse_inline_call(&mut self, optional: bool) -> Result<QueryClause> {
        self.expect(TokenKind::Call)?;
        let scope = if self.current.kind == TokenKind::LParen {
            self.advance();
            let mut names = Vec::new();
            if self.current.kind != TokenKind::RParen {
                loop {
                    if !self.is_identifier() {
                        return Err(self.error("Expected a variable in the variable scope clause"));
                    }
                    names.push(self.get_identifier_name());
                    self.advance();
                    if self.current.kind != TokenKind::Comma {
                        break;
                    }
                    self.advance();
                }
            }
            self.expect(TokenKind::RParen)?;
            Some(names)
        } else {
            None
        };
        self.expect(TokenKind::LBrace)?;

        // Parse the inner query body (MATCH ... RETURN ...)
        self.enter_subquery()?;
        let subquery = self.parse_query()?;
        let mut combined = Vec::new();
        while let Some(op) = self.parse_composite_op() {
            combined.push((op, self.parse_query()?));
        }
        self.exit_subquery();

        self.expect(TokenKind::RBrace)?;
        Ok(QueryClause::InlineCall {
            subquery,
            combined,
            optional,
            scope,
        })
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

    /// Checks that a WHERE (`is_where`) or FILTER may follow `previous`, the
    /// clause before it. Both filter the rows so far, so a second WHERE is an
    /// AND of the first. Right after INSERT, CREATE, MERGE or DELETE either
    /// would filter the rows after the write, while it used to filter the
    /// rows before it (and so limit what the write touched): an error rather
    /// than a silent change, until a WITH or another clause between makes
    /// the order plain. Right after SET or REMOVE (never accepted before) a
    /// FILTER filters the rows after the write, as ISO GQL's filter statement
    /// does; WHERE belongs to a MATCH, so it does not stand there.
    fn check_filter_follows(
        is_where: bool,
        previous: Option<&QueryClause>,
    ) -> std::result::Result<(), &'static str> {
        match previous {
            Some(QueryClause::Filter(_)) if is_where => Err(
                "a WHERE cannot follow another WHERE or FILTER: combine the conditions with AND",
            ),
            Some(QueryClause::Create(_) | QueryClause::Delete(_) | QueryClause::Merge(_)) => Err(
                "a WHERE or FILTER cannot follow INSERT, CREATE, MERGE or DELETE: it used to \
                 filter the rows before the write; put the condition before the write, or \
                 after a WITH (`WITH ... WHERE ...`) to filter the rows after it",
            ),
            Some(QueryClause::Set(_) | QueryClause::Remove(_)) if is_where => Err(
                "a WHERE cannot follow SET or REMOVE: put the condition before the write, \
                 or use FILTER to filter the rows after it",
            ),
            _ => Ok(()),
        }
    }

    /// Whether `clause` is an inline procedure call whose body modifies data
    /// (ISO GQL's call data-modifying procedure statement), which a statement
    /// may end with, as with any other data-modifying statement.
    fn calls_a_data_modifying_procedure(clause: &QueryClause) -> bool {
        match clause {
            QueryClause::InlineCall {
                subquery, combined, ..
            } => {
                Self::modifies_data(subquery)
                    || combined.iter().any(|(_, part)| Self::modifies_data(part))
            }
            _ => false,
        }
    }

    /// Whether `query` modifies data: it has a data-modifying clause, or an
    /// inline procedure call whose body does.
    fn modifies_data(query: &QueryStatement) -> bool {
        !query.set_clauses.is_empty()
            || !query.remove_clauses.is_empty()
            || !query.merge_clauses.is_empty()
            || !query.create_clauses.is_empty()
            || !query.delete_clauses.is_empty()
            || query
                .ordered_clauses
                .iter()
                .any(Self::calls_a_data_modifying_procedure)
    }

    /// A query made of a single INSERT (or `CREATE (...)`) clause and nothing
    /// else is kept as a standalone INSERT statement; anything longer stays a
    /// query. Neither has a result without a RETURN.
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

    /// Parses a linear query or data-modifying statement: its clauses in
    /// source order, then its result statement.
    ///
    /// ISO/IEC 39075:2024 composes a `<simple linear query statement>` of
    /// `<simple query statement>`s (MATCH, LET, FOR, FILTER, an `<order by and
    /// page statement>`, CALL) in any order, and a `<linear data-modifying
    /// statement>` of those and `<simple data-modifying statement>`s (INSERT,
    /// SET, REMOVE, DELETE, a CALL that modifies data), also in any order.
    /// Grafeo adds WITH, UNWIND, MERGE, CREATE and LOAD DATA, and a WHERE
    /// between clauses (a filter of the rows so far). Each clause reads the
    /// rows the ones before it leave; the result statement (RETURN, FINISH or
    /// SELECT) may be left out when the statement modifies data.
    fn parse_query(&mut self) -> Result<QueryStatement> {
        let span_start = self.current.span.start;

        let mut match_clauses = Vec::new();
        let mut unwind_clauses = Vec::new();
        let mut merge_clauses = Vec::new();
        let mut create_clauses = Vec::new();
        let mut delete_clauses = Vec::new();
        let mut set_clauses = Vec::new();
        let mut remove_clauses = Vec::new();
        let mut with_clauses = Vec::new();
        let mut ordered_clauses = Vec::new();

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
                        if matches!(self.peek_kind(), TokenKind::LBrace | TokenKind::LParen) {
                            // OPTIONAL CALL [(scope)] { subquery }
                            ordered_clauses.push(self.parse_inline_call(true)?);
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
                    // CALL [(scope)] { subquery } (inline) or CALL procedure(...)
                    // (within query): a procedure name comes before any `(`
                    if matches!(self.peek_kind(), TokenKind::LBrace | TokenKind::LParen) {
                        ordered_clauses.push(self.parse_inline_call(false)?);
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
                // A FILTER statement, or a WHERE after a clause: a WHERE right
                // after a MATCH is that MATCH's graph pattern WHERE clause.
                TokenKind::Where | TokenKind::Filter => {
                    let is_where = self.current.kind == TokenKind::Where;
                    Self::check_filter_follows(is_where, ordered_clauses.last())
                        .map_err(|message| self.error(message))?;
                    let clause = self.parse_where_or_filter_clause()?;
                    ordered_clauses.push(QueryClause::Filter(clause));
                }
                TokenKind::Set => {
                    let clause = self.parse_set_clause()?;
                    ordered_clauses.push(QueryClause::Set(clause.clone()));
                    set_clauses.push(clause);
                }
                TokenKind::Remove => {
                    let clause = self.parse_remove_clause()?;
                    ordered_clauses.push(QueryClause::Remove(clause.clone()));
                    remove_clauses.push(clause);
                }
                TokenKind::With => {
                    let mut clause = self.parse_with_clause()?;
                    // LET bindings right after a WITH belong to it.
                    if self.is_identifier()
                        && self.get_identifier_name().eq_ignore_ascii_case("LET")
                    {
                        clause.let_bindings = self.parse_let_clause()?;
                    }
                    ordered_clauses.push(QueryClause::With(clause.clone()));
                    with_clauses.push(clause);
                }
                TokenKind::Order
                | TokenKind::Offset
                | TokenKind::Skip
                | TokenKind::Limit
                | TokenKind::Fetch => {
                    let page = self.parse_order_by_and_page()?;
                    ordered_clauses.push(QueryClause::OrderByAndPage(page));
                }
                _ => break,
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
            || ordered_clauses
                .iter()
                .any(Self::calls_a_data_modifying_procedure)
        {
            // A statement that modifies data needs no result statement (ISO
            // GQL's linear data-modifying statement): it has no result
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
            where_clause: None,
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

    /// Parses an `<order by and page statement>` (ISO/IEC 39075:2024):
    /// `[ORDER BY ...] [OFFSET n | SKIP n] [LIMIT n | FETCH FIRST n ROWS ONLY]`
    /// with at least one part, as a statement of its own or after RETURN.
    fn parse_order_by_and_page(&mut self) -> Result<OrderByAndPage> {
        let span_start = self.current.span.start;
        let order_by = if self.current.kind == TokenKind::Order {
            Some(self.parse_order_by()?)
        } else {
            None
        };
        let offset = if matches!(self.current.kind, TokenKind::Skip | TokenKind::Offset) {
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
        if order_by.is_none() && offset.is_none() && limit.is_none() {
            return Err(self.error("Expected ORDER BY, OFFSET, SKIP, LIMIT or FETCH"));
        }
        Ok(OrderByAndPage {
            order_by,
            offset,
            limit,
            span: Some(SourceSpan::new(span_start, self.current.span.start, 1, 1)),
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

        // Parse optional WITH ORDINALITY/OFFSET; any other WITH is a WITH
        // clause after the FOR.
        let (ordinality_var, offset_var) = if self.current.kind == TokenKind::With
            && matches!(self.peek_kind(), TokenKind::Ordinality | TokenKind::Offset)
        {
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
        self.check_one_type_each("MERGE", &pattern)?;

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

        // Check for path mode (WALK, TRAIL, SIMPLE, ACYCLIC), which may also
        // come before the match mode here
        let path_mode = self.parse_path_mode();
        if path_mode.is_some() {
            self.skip_path_or_paths();
        }

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

        // The path pattern prefix: a path mode (the standard puts the match
        // mode first, `MATCH DIFFERENT EDGES TRAIL`), or a path search prefix
        // with its own path mode (`MATCH ANY SHORTEST TRAIL`). They apply to
        // every path pattern of the clause without a prefix of its own.
        let (search_prefix, prefix_mode) = self.parse_path_pattern_prefix()?;
        let path_mode = match (path_mode, prefix_mode) {
            (Some(_), Some(_)) => {
                return Err(self.error("A path pattern has one path mode, but two are given"));
            }
            (mode, None) | (None, mode) => mode,
        };

        let mut patterns = Vec::new();
        let first = self.parse_aliased_pattern()?;
        // The prefix after MATCH is the first pattern's (`MATCH ANY p = ...`)
        if search_prefix.is_some() && first.search_prefix.is_some() {
            return Err(self.error("A path pattern has one path search prefix, but two are given"));
        }
        if path_mode.is_some() && first.path_mode.is_some() {
            return Err(self.error("A path pattern has one path mode, but two are given"));
        }
        patterns.push(first);

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
                let alternative = self.parse_aliased_pattern()?;
                if alternative.search_prefix.is_some() || alternative.path_mode.is_some() {
                    return Err(self.error(
                        "A path pattern prefix goes before the whole alternation, not before \
                         one of its operands",
                    ));
                }
                alt_patterns.push(alternative.pattern);
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
                path_mode: first.path_mode,
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

    /// Parses an optional path mode keyword: WALK, TRAIL, SIMPLE or ACYCLIC.
    fn parse_path_mode(&mut self) -> Option<PathMode> {
        let mode = match self.current.kind {
            TokenKind::Walk => PathMode::Walk,
            TokenKind::Trail => PathMode::Trail,
            TokenKind::Simple => PathMode::Simple,
            TokenKind::Acyclic => PathMode::Acyclic,
            _ => return None,
        };
        self.advance();
        Some(mode)
    }

    /// Skips the optional `PATH` or `PATHS` of a path pattern prefix (ISO/IEC
    /// 39075:2024 16.6 `<path or paths>`), which changes nothing. A variable
    /// named so (`path = (a)-->(b)`) is followed by `=`.
    fn skip_path_or_paths(&mut self) {
        if self.current.kind == TokenKind::Identifier
            && (self.current.text.eq_ignore_ascii_case("PATH")
                || self.current.text.eq_ignore_ascii_case("PATHS"))
            && self.peek_kind() != TokenKind::Eq
        {
            self.advance();
        }
    }

    /// The number of paths or groups (`what`) of a path search prefix, from
    /// the text of its integer: at least 1.
    fn number_of_paths(&self, text: &str, what: &str) -> Result<usize> {
        let count: usize = text
            .parse()
            .map_err(|_| self.error(&format!("The number of {what} {text} is too large")))?;
        if count == 0 {
            return Err(self.error(&format!(
                "The number of {what} of a path search prefix must be at least 1"
            )));
        }
        Ok(count)
    }

    /// The text of the integer at the current token, which it consumes.
    fn take_integer_text(&mut self) -> String {
        let text = self.current.text.clone();
        self.advance();
        text
    }

    /// Parses an optional path pattern prefix (ISO/IEC 39075:2024 16.6
    /// `<path pattern prefix>`): a path mode, or a path search prefix with an
    /// optional path mode of its own. Returns the search prefix and the path
    /// mode.
    ///
    /// ```text
    /// WALK | TRAIL | SIMPLE | ACYCLIC [PATH | PATHS]
    /// ALL [mode] [PATH | PATHS]
    /// ANY [k] [mode] [PATH | PATHS]
    /// ALL SHORTEST [mode] [PATH | PATHS]
    /// ANY SHORTEST [mode] [PATH | PATHS]
    /// SHORTEST k [mode] [PATH | PATHS]
    /// SHORTEST [k] [mode] [PATH | PATHS] GROUP | GROUPS
    /// ```
    ///
    /// `SHORTEST` alone is taken as `SHORTEST 1`.
    fn parse_path_pattern_prefix(
        &mut self,
    ) -> Result<(Option<PathSearchPrefix>, Option<PathMode>)> {
        if let Some(mode) = self.parse_path_mode() {
            self.skip_path_or_paths();
            return Ok((None, Some(mode)));
        }
        // What may follow ALL or ANY when it starts a prefix: a path
        // pattern, its mode, `PATH`, `PATHS`, or the path variable of the
        // legacy `MATCH ANY p = ...`
        let starts_prefix = |kind: TokenKind| {
            matches!(
                kind,
                TokenKind::LParen
                    | TokenKind::Identifier
                    | TokenKind::QuotedIdentifier
                    | TokenKind::Walk
                    | TokenKind::Trail
                    | TokenKind::Simple
                    | TokenKind::Acyclic
            )
        };
        let prefix = if self.current.kind == TokenKind::All {
            match self.peek_kind() {
                TokenKind::Shortest => {
                    self.advance(); // consume ALL
                    self.advance(); // consume SHORTEST
                    PathSearchPrefix::AllShortest
                }
                next if starts_prefix(next) => {
                    self.advance(); // consume ALL
                    PathSearchPrefix::All
                }
                _ => return Ok((None, None)),
            }
        } else if self.is_identifier() && self.get_identifier_name().eq_ignore_ascii_case("ANY") {
            match self.peek_kind() {
                TokenKind::Shortest => {
                    self.advance(); // consume ANY
                    self.advance(); // consume SHORTEST
                    PathSearchPrefix::AnyShortest
                }
                TokenKind::Integer => {
                    self.advance(); // consume ANY
                    let text = self.take_integer_text();
                    PathSearchPrefix::AnyK(self.number_of_paths(&text, "paths")?)
                }
                next if starts_prefix(next) => {
                    self.advance(); // consume ANY
                    PathSearchPrefix::Any
                }
                _ => return Ok((None, None)),
            }
        } else if self.current.kind == TokenKind::Shortest {
            self.advance(); // consume SHORTEST
            let count = (self.current.kind == TokenKind::Integer).then(|| self.take_integer_text());
            let mode = self.parse_path_mode();
            self.skip_path_or_paths();
            let groups = matches!(self.current.kind, TokenKind::Group | TokenKind::Groups);
            let what = if groups { "groups" } else { "paths" };
            let count = match count {
                Some(text) => self.number_of_paths(&text, what)?,
                None => 1,
            };
            if groups {
                self.advance(); // consume GROUP or GROUPS
                return Ok((Some(PathSearchPrefix::ShortestKGroups(count)), mode));
            }
            return Ok((Some(PathSearchPrefix::ShortestK(count)), mode));
        } else {
            return Ok((None, None));
        };
        let mode = self.parse_path_mode();
        self.skip_path_or_paths();
        Ok((Some(prefix), mode))
    }

    /// Parses a pattern with optional alias and path function.
    /// Supports: `p = shortestPath((a)-[*]-(b))` and `p = (a)-[*]-(b)` and `(a)-[*]-(b)`,
    /// and a path pattern prefix after the alias, or before a pattern
    /// without one (ISO/IEC 39075:2024 16.4 `<path pattern>`:
    /// `p = ANY SHORTEST TRAIL (a)-[*]-(b)`).
    fn parse_aliased_pattern(&mut self) -> Result<AliasedPattern> {
        let mut alias = None;
        let mut path_function = None;
        let mut search_prefix = None;
        let mut path_mode = None;

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
        }

        // Per-pattern path pattern prefix: p = ANY SHORTEST TRAIL (...)
        if path_function.is_none() {
            (search_prefix, path_mode) = self.parse_path_pattern_prefix()?;
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
            path_mode,
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

        // Parse the inner pattern(s), potentially with union via | or multiset union via |+|.
        // A pattern in parentheses nests as a subquery: it plans as one.
        self.enter_subquery()?;
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
        self.exit_subquery();

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
            self.enter_nesting()?;
            let expr = self.parse_label_expression()?;
            self.exit_nesting();
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

    /// Parses the types of an edge pattern, after its variable: none, one
    /// (`:KNOWS`), or alternatives (`:KNOWS|LIKES`), the disjunction of a
    /// label expression (ISO/IEC 39075:2024, 16.8 `<label expression>`).
    ///
    /// An edge has exactly one type, so a conjunction (`:A&B`, an edge with
    /// both labels) would match no edge, and `:A:B` is no label expression:
    /// both are errors that name what to write, the alternatives or the
    /// quoted name of one type.
    fn parse_edge_types(&mut self) -> Result<Vec<String>> {
        let mut types = Vec::new();
        if self.current.kind != TokenKind::Colon {
            return Ok(types);
        }
        self.advance();
        loop {
            if !self.is_label_or_type_name() {
                return Err(self.error(if types.is_empty() {
                    "Expected edge type"
                } else {
                    "Expected edge type after |"
                }));
            }
            types.push(self.get_identifier_name());
            self.advance();
            if self.current.kind != TokenKind::Pipe {
                break;
            }
            self.advance();
        }
        let joint = match self.current.kind {
            TokenKind::Colon => ":",
            TokenKind::Ampersand => "&",
            _ => return Ok(types),
        };
        let at = self.current.span;
        self.advance();
        let first = types.last().map_or("A", String::as_str);
        let second = if self.is_label_or_type_name() {
            self.get_identifier_name()
        } else {
            "B".to_string()
        };
        let what = if joint == "&" {
            format!("An edge has one type, so :{first}&{second}, an edge of both, matches no edge")
        } else {
            "An edge has one type".to_string()
        };
        Err(Error::Query(
            QueryError::new(
                QueryErrorKind::Syntax,
                format!(
                    "[GQL] {what}: write :`{first}{joint}{second}` for the type named \
                     {first}{joint}{second}, or :{first}|{second} to match either type"
                ),
            )
            .with_span(at)
            .with_source(self.source.to_string()),
        ))
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

                        let tps = self.parse_edge_types()?;

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

                        let tps = self.parse_edge_types()?;

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

                        let tps = self.parse_edge_types()?;

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
            filter: false,
            span: None,
        })
    }

    /// Parses either a WHERE or a FILTER clause (see [`WhereClause::filter`]).
    fn parse_where_or_filter_clause(&mut self) -> Result<WhereClause> {
        // Accept both WHERE and FILTER
        let filter = self.current.kind == TokenKind::Filter;
        if filter {
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
            filter,
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

    /// Parses an expression, which nests one level in the expression,
    /// list, call or clause it is part of.
    fn parse_expression(&mut self) -> Result<Expression> {
        self.enter_nesting()?;
        let expression = self.parse_or_expression()?;
        self.exit_nesting();
        Ok(expression)
    }

    fn parse_or_expression(&mut self) -> Result<Expression> {
        self.parse_balanced_chain(TokenKind::Or, BinaryOp::Or, Self::parse_xor_expression)
    }

    fn parse_xor_expression(&mut self) -> Result<Expression> {
        self.parse_balanced_chain(TokenKind::Xor, BinaryOp::Xor, Self::parse_and_expression)
    }

    fn parse_and_expression(&mut self) -> Result<Expression> {
        self.parse_balanced_chain(TokenKind::And, BinaryOp::And, Self::parse_not_expression)
    }

    /// Parses `operand` joined by `token`, an associative operator, into a
    /// balanced tree of `op` (see [`Nesting::join_balanced`]): a chain of any
    /// length stays shallow.
    fn parse_balanced_chain(
        &mut self,
        token: TokenKind,
        op: BinaryOp,
        operand: fn(&mut Self) -> Result<Expression>,
    ) -> Result<Expression> {
        let chain = self.begin_chain();
        let first = operand(self)?;
        if self.current.kind != token {
            self.end_chain(chain);
            return Ok(first);
        }
        let mut operands = vec![(first, self.nesting.take_operand())];
        while self.current.kind == token {
            self.advance();
            let next = operand(self)?;
            operands.push((next, self.nesting.take_operand()));
        }
        let tree = self
            .nesting
            .join_balanced(operands, |left, right| Expression::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            })
            .ok_or_else(|| self.error(&nesting_error_message()));
        self.end_chain(chain);
        tree
    }

    fn parse_not_expression(&mut self) -> Result<Expression> {
        if self.current.kind == TokenKind::Not {
            self.advance();
            self.enter_nesting()?;
            let operand = self.parse_not_expression();
            self.exit_nesting();
            let operand = operand?;
            return Ok(Expression::Unary {
                op: UnaryOp::Not,
                operand: Box::new(operand),
            });
        }
        self.parse_comparison_expression()
    }

    fn parse_comparison_expression(&mut self) -> Result<Expression> {
        let chain = self.begin_chain();
        let expression = self.parse_comparison_operands()?;
        self.end_chain(chain);
        Ok(expression)
    }

    /// The operands of a comparison and its operator, which nests one level
    /// above them (see [`Self::parse_comparison_expression`]).
    fn parse_comparison_operands(&mut self) -> Result<Expression> {
        let left = self.parse_additive_expression()?;

        // `=~`, a regular expression match: a Grafeo extension with the
        // precedence and meaning of Cypher's `=~`.
        if self.at_regex_match() {
            self.advance(); // consume =
            self.advance(); // consume ~
            let right = self.parse_additive_expression()?;
            self.link_chain()?;
            return Ok(Expression::Binary {
                left: Box::new(left),
                op: BinaryOp::RegexMatch,
                right: Box::new(right),
            });
        }

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
            self.link_chain()?;
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
                self.link_chain()?;
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
                self.link_chain()?;
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
                self.link_chain()?;
                return Ok(Expression::Binary {
                    left: Box::new(left),
                    op: BinaryOp::EndsWith,
                    right: Box::new(right),
                });
            }
            TokenKind::Contains => {
                self.advance(); // consume CONTAINS
                let right = self.parse_additive_expression()?;
                self.link_chain()?;
                return Ok(Expression::Binary {
                    left: Box::new(left),
                    op: BinaryOp::Contains,
                    right: Box::new(right),
                });
            }
            TokenKind::Like => {
                self.advance(); // consume LIKE
                let right = self.parse_additive_expression()?;
                self.link_chain()?;
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
                    // `IS NOT ...` may wrap the predicate in a NOT.
                    self.link_chain()?;
                }
                self.link_chain()?;

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
        let chain = self.begin_chain();
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
            self.link_chain()?;
            left = Expression::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }
        self.end_chain(chain);

        Ok(left)
    }

    fn parse_multiplicative_expression(&mut self) -> Result<Expression> {
        let chain = self.begin_chain();
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
            self.link_chain()?;
            left = Expression::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }
        self.end_chain(chain);

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
        // Each `^` nests one level (right associative): bound the depth so a
        // long chain is an error, not a stack overflow.
        self.enter_nesting()?;
        let right = self.parse_power_expression();
        self.exit_nesting();
        let right = right?;
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
                self.enter_nesting()?;
                let operand = self.parse_unary_expression();
                self.exit_nesting();
                let operand = operand?;
                Ok(Expression::Unary {
                    op: UnaryOp::Neg,
                    operand: Box::new(operand),
                })
            }
            TokenKind::Plus => {
                self.advance();
                self.enter_nesting()?;
                let operand = self.parse_unary_expression();
                self.exit_nesting();
                let operand = operand?;
                Ok(Expression::Unary {
                    op: UnaryOp::Pos,
                    operand: Box::new(operand),
                })
            }
            _ => self.parse_postfix_expression(),
        }
    }

    fn parse_postfix_expression(&mut self) -> Result<Expression> {
        // Each subscript, key access and label check nests one level.
        let chain = self.begin_chain();
        // Parentheses, lists and maps go around the primary expression's
        // parser: its frame is large, and each level of a deeply nested
        // expression would keep one on the stack.
        // A call (a function, CAST, or a form such as TRIM, reduce or a list
        // predicate) nests one level more than its arguments: it is parsed in
        // that large frame.
        let call = (self.is_identifier() || self.current.kind == TokenKind::Cast)
            && self.peek_kind() == TokenKind::LParen;
        let mut expr = match self.current.kind {
            TokenKind::LParen => self.parse_parenthesized_expression()?,
            TokenKind::LBracket => self.parse_list_expression()?,
            TokenKind::LBrace => Expression::Map(self.parse_property_map()?),
            _ if call => {
                self.enter_nesting()?;
                let call = self.parse_primary_expression();
                self.exit_nesting();
                call?
            }
            _ => self.parse_primary_expression()?,
        };

        loop {
            match self.current.kind {
                TokenKind::LBracket => {
                    self.advance();
                    self.link_chain()?;

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
                // Key access into a map value: `n.meta.route` reads like
                // `n.meta['route']` (the primary expression already consumed
                // `n.meta`).
                TokenKind::Dot => {
                    self.advance();
                    if !self.is_label_or_type_name() {
                        return Err(self.error("Expected map key after '.'"));
                    }
                    let key = self.get_identifier_name();
                    self.advance();
                    self.link_chain()?;
                    expr = Expression::MapAccess {
                        base: Box::new(expr),
                        key,
                    };
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
                        self.link_chain()?;
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
        self.end_chain(chain);

        Ok(expr)
    }

    /// Parses `( expression )`: the expression nests one level deeper.
    fn parse_parenthesized_expression(&mut self) -> Result<Expression> {
        self.expect(TokenKind::LParen)?;
        let expr = self.parse_expression()?;
        self.expect(TokenKind::RParen)?;
        Ok(expr)
    }

    /// Parses the bindings and the body of `LET ... IN ... END`, after `LET`.
    fn parse_let_in_expression(&mut self) -> Result<Expression> {
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
        Ok(Expression::LetIn {
            bindings,
            body: Box::new(body),
        })
    }

    /// Parses the arguments of a call of the function `name`, from its `(`.
    fn parse_function_call(&mut self, name: String) -> Result<Expression> {
        self.expect(TokenKind::LParen)?;
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
    }

    /// Parses a list literal or a list comprehension, from its `[`: each
    /// element nests one level deeper.
    fn parse_list_expression(&mut self) -> Result<Expression> {
        self.expect(TokenKind::LBracket)?;
        // Disambiguate: [x IN list WHERE ... | expr] vs [elem, ...]
        // List comprehension if: identifier followed by IN keyword
        if self.is_identifier() && self.peek_kind() == TokenKind::In {
            return self.parse_list_comprehension_inner();
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
        Ok(Expression::List(elements))
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

                // LET ... IN ... END expression: it nests one level, as its
                // bindings are parsed below the expression level.
                if name.eq_ignore_ascii_case("LET") {
                    self.advance(); // consume LET
                    self.enter_nesting()?;
                    let let_in = self.parse_let_in_expression();
                    self.exit_nesting();
                    return let_in;
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
                    self.enter_subquery()?;
                    let inner_query = self.parse_exists_inner_query()?;
                    self.exit_subquery();
                    self.expect(TokenKind::RBrace)?;
                    Ok(Expression::CountSubquery {
                        query: Box::new(inner_query),
                    })
                } else if self.current.kind == TokenKind::LBrace
                    && name.eq_ignore_ascii_case("value")
                {
                    // VALUE { subquery } expression
                    self.advance(); // consume {
                    self.enter_subquery()?;
                    let inner_query = self.parse_value_subquery_inner()?;
                    self.exit_subquery();
                    self.expect(TokenKind::RBrace)?;
                    Ok(Expression::ValueSubquery {
                        query: Box::new(inner_query),
                    })
                } else if self.current.kind == TokenKind::LParen {
                    // Function call (its extra level of nesting is counted
                    // where the postfix expression starts).
                    self.parse_function_call(name)
                } else {
                    Ok(Expression::Variable(name))
                }
            }
            TokenKind::LParen => self.parse_parenthesized_expression(),
            TokenKind::LBracket => self.parse_list_expression(),
            TokenKind::Parameter => {
                // Parameter token includes the $ prefix, so we extract just the name
                let full_text = &self.current.text;
                let name = full_text.trim_start_matches('$').to_string();
                self.advance();
                Ok(Expression::Parameter(name))
            }
            TokenKind::Exists => {
                self.advance();
                if self.current.kind == TokenKind::LBrace {
                    // EXISTS { MATCH ... } subquery form
                    self.advance(); // consume {
                    self.enter_subquery()?;
                    let inner_query = self.parse_exists_inner_query()?;
                    self.exit_subquery();
                    self.expect(TokenKind::RBrace)?;
                    Ok(Expression::ExistsSubquery {
                        query: Box::new(inner_query),
                    })
                } else {
                    // exists(expr) function form, which nests as a call
                    self.expect(TokenKind::LParen)?;
                    self.enter_nesting()?;
                    let arg = self.parse_expression()?;
                    self.exit_nesting();
                    self.expect(TokenKind::RParen)?;
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
    ///
    /// Each operand is an expression, which nests one level deeper, and the
    /// `CASE` itself counts one more: its parser keeps a larger frame on the
    /// stack.
    fn parse_case_expression(&mut self) -> Result<Expression> {
        self.enter_nesting()?;
        let case = self.parse_case_operands();
        self.exit_nesting();
        case
    }

    /// The operands of a `CASE`, from the `CASE` keyword to `END`.
    fn parse_case_operands(&mut self) -> Result<Expression> {
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
        for pattern in &patterns {
            self.check_one_type_each("INSERT", pattern)?;
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
        for pattern in &patterns {
            self.check_one_type_each("CREATE", pattern)?;
        }

        Ok(InsertStatement {
            patterns,
            span: None,
        })
    }

    /// Refuses an edge with alternative types (`:A|B`) in `pattern` of
    /// `clause`, which creates edges: each gets one type (the label set of
    /// an inserted edge has no disjunction, ISO/IEC 39075:2024 16.5
    /// `<insert graph pattern>`).
    fn check_one_type_each(&self, clause: &str, pattern: &Pattern) -> Result<()> {
        match pattern {
            Pattern::Node(_) => Ok(()),
            Pattern::Path(path) => match path.edges.iter().find(|edge| edge.types.len() > 1) {
                Some(edge) => Err(self.error(&format!(
                    "{clause} gives an edge one type: :{} names alternatives, which only a \
                     pattern to match can have",
                    edge.types.join("|")
                ))),
                None => Ok(()),
            },
            Pattern::Quantified { pattern, .. } => self.check_one_type_each(clause, pattern),
            Pattern::Union(patterns) | Pattern::MultisetUnion(patterns) => patterns
                .iter()
                .try_for_each(|pattern| self.check_one_type_each(clause, pattern)),
        }
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
                        self.parse_graph_type_body()?
                    } else if self.current.kind == TokenKind::LParen {
                        let inline = self.parse_graph_type_iso_body()?;
                        let (nt, et) = element_type_names(&inline);
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

    /// Parses the options of a text index, `{k1: 1.2, b: 0.75, tokenizer:
    /// 'standard', stop_words: ['and', 'or']}` (each optional, names in any
    /// case), into `options`.
    fn parse_text_index_options(&mut self, options: &mut IndexOptions) -> Result<()> {
        self.expect(TokenKind::LBrace)?;
        while self.current.kind != TokenKind::RBrace {
            if !self.is_identifier() {
                return Err(self.error("Expected option name"));
            }
            let opt_name = self.get_identifier_name();
            self.advance();
            self.expect(TokenKind::Colon)?;
            match opt_name.to_uppercase().as_str() {
                "K1" => options.k1 = Some(self.parse_index_option_number("k1")?),
                "B" => options.b = Some(self.parse_index_option_number("b")?),
                "TOKENIZER" => {
                    if self.current.kind != TokenKind::String {
                        return Err(self.error("Expected string for tokenizer"));
                    }
                    options.tokenizer = Some(self.index_option_string());
                    self.advance();
                }
                "STOP_WORDS" => {
                    self.expect(TokenKind::LBracket)?;
                    let mut words = Vec::new();
                    while self.current.kind != TokenKind::RBracket {
                        if self.current.kind != TokenKind::String {
                            return Err(self.error("Expected string in stop_words"));
                        }
                        words.push(self.index_option_string());
                        self.advance();
                        if self.current.kind == TokenKind::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    self.expect(TokenKind::RBracket)?;
                    options.stop_words = Some(words);
                }
                _ => {
                    return Err(self.error(&format!(
                        "Unknown text index option '{opt_name}'. Use: k1, b, tokenizer, stop_words"
                    )));
                }
            }
            if self.current.kind == TokenKind::Comma {
                self.advance();
            }
        }
        self.expect(TokenKind::RBrace)?;
        Ok(())
    }

    /// Parses the number of the index option `name`: an integer or a float,
    /// with an optional leading minus (which the engine then refuses as out
    /// of range, naming the option).
    fn parse_index_option_number(&mut self, name: &str) -> Result<f64> {
        let negative = self.current.kind == TokenKind::Minus;
        if negative {
            self.advance();
        }
        if !matches!(self.current.kind, TokenKind::Integer | TokenKind::Float) {
            return Err(self.error(&format!("Expected a number for {name}")));
        }
        let value: f64 = self
            .current
            .text
            .parse()
            .map_err(|_| self.error(&format!("Invalid number for {name}")))?;
        self.advance();
        Ok(if negative { -value } else { value })
    }

    /// The text of the current string literal, without its quotes.
    fn index_option_string(&self) -> String {
        self.current
            .text
            .trim_matches('\'')
            .trim_matches('"')
            .to_string()
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
                "TEXT" => {
                    index_kind = IndexKind::Text;
                    // Parse optional {k1: 1.2, b: 0.75, tokenizer: 'name', stop_words: [...]}
                    if self.current.kind == TokenKind::LBrace {
                        self.parse_text_index_options(&mut options)?;
                    }
                }
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

    /// Parses the body of CREATE GRAPH TYPE in braces, in one of two forms:
    ///
    /// - The element patterns of ISO/IEC 39075:2024 (`<nested graph type
    ///   specification>`): `{ (:Person {name STRING})-[:KNOWS]->(:Person) }`,
    ///   which declare their node and edge types as the paren form does.
    /// - Lists of type names: `{ node_types: [A, B], edge_types: [E1], open: true }`,
    ///   which declare no types.
    ///
    /// Returns (node_types, edge_types, inline_types, open).
    fn parse_graph_type_body(&mut self) -> Result<GraphTypeBody> {
        self.expect(TokenKind::LBrace)?;

        if self.current.kind == TokenKind::LParen {
            let inline = self.parse_graph_type_pattern_elements(TokenKind::RBrace)?;
            let (node_types, edge_types) = element_type_names(&inline);
            return Ok((node_types, edge_types, inline, false));
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
        Ok((node_types, edge_types, Vec::new(), open))
    }

    /// Parses the paren body of CREATE GRAPH TYPE.
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
            self.parse_graph_type_pattern_elements(TokenKind::RParen)
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

    /// Parses the element patterns of a graph type body up to and including
    /// `close`, the `)` of the paren form or the `}` of the brace form:
    /// `(:Person {name STRING})-[:KNOWS {since INT64}]->(:Person), ...`.
    fn parse_graph_type_pattern_elements(
        &mut self,
        close: TokenKind,
    ) -> Result<Vec<InlineElementType>> {
        let mut types = Vec::new();
        // Track which node type names we've already emitted so we don't duplicate them.
        let mut seen_node_types = std::collections::HashSet::new();

        while self.current.kind != close && self.current.kind != TokenKind::Eof {
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

        self.expect(close)?;
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
    /// `DROP PROPERTY name`. It is a keyword only when a property name
    /// follows (see [`Self::next_names_property`]), so a property that is
    /// itself called `property` still works as a backtick-quoted identifier.
    fn skip_property_keyword(&mut self) {
        if self.current.kind == TokenKind::Identifier
            && self.current.text.eq_ignore_ascii_case("PROPERTY")
            && self.next_names_property()
        {
            self.advance();
        }
    }

    /// Whether the token after the current one can name a property, as a
    /// key of a property map can (see [`Self::is_label_or_type_name`]):
    /// keywords such as `starts` or `match` included.
    fn next_names_property(&mut self) -> bool {
        let _ = self.peek_kind();
        let Some(next) = self.peeked.take() else {
            return false;
        };
        let current = std::mem::replace(&mut self.current, next);
        let names_property = self.is_label_or_type_name();
        self.peeked = Some(std::mem::replace(&mut self.current, current));
        names_property
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
                    // ADD [PROPERTY] property_name type [NOT NULL] [DEFAULT literal]
                    self.skip_property_keyword();
                    alterations.push(TypeAlteration::AddProperty(
                        self.parse_property_definition()?,
                    ));
                }
                "DROP" => {
                    self.advance();
                    // DROP [PROPERTY] property_name
                    self.skip_property_keyword();
                    if !self.is_label_or_type_name() {
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

    /// Reads a property type: a name, `ZONED DATETIME`, `LOCAL DATETIME`, or
    /// `LIST<type>` (nested), returned in the canonical spelling
    /// `PropertyDataType::from_type_name` reads: a single name as written (the
    /// catalog ignores its case), the two-word types in capitals with one
    /// space, and `LIST<...>` around its element type. The same string goes
    /// into the WAL record of the statement, so replay reads the same type
    /// (#569).
    ///
    /// The `LIST<` levels are counted, not recursed into, and at most
    /// [`MAX_PROPERTY_VALUE_DEPTH`] are accepted, as deep as a property value
    /// may be: a type of any depth is refused without a stack overflow.
    ///
    /// A name that is not a property type Grafeo supports is refused
    /// (ISO/IEC 39075:2024 18.7 `<property value type>`, Syntax Rule 4); it
    /// used to declare an `ANY` property. Returns the type's spelling and
    /// what it takes as a `DEFAULT`.
    fn parse_property_type_name(&mut self) -> Result<(String, PropertyTypeKind)> {
        use crate::query::schema::{property_type_kind, unknown_property_type};

        let mut levels = 0;
        let (mut element, kind) = loop {
            if !self.is_identifier() {
                return Err(self.error("Expected type name"));
            }
            let name = self.get_identifier_name();

            if name.eq_ignore_ascii_case("LIST") && self.peek_kind() == TokenKind::Lt {
                levels += 1;
                if levels > MAX_PROPERTY_VALUE_DEPTH {
                    return Err(self.error(&format!(
                        "A property type nests at most {MAX_PROPERTY_VALUE_DEPTH} LIST<...> levels"
                    )));
                }
                self.advance(); // LIST
                self.advance(); // <
                continue;
            }
            if name.eq_ignore_ascii_case("ZONED") || name.eq_ignore_ascii_case("LOCAL") {
                self.advance();
                let prefix = name.to_ascii_uppercase();
                if !self.try_accept_identifier_keyword("DATETIME") {
                    return Err(self.error(&format!("Expected DATETIME after {prefix}")));
                }
                break (format!("{prefix} DATETIME"), PropertyTypeKind::Other);
            }
            let Some(kind) = property_type_kind(&name) else {
                return Err(self.error(&unknown_property_type(&name)));
            };
            self.advance();
            break (name, kind);
        };

        for _ in 0..levels {
            if self.current.kind != TokenKind::Gt {
                return Err(self.error(&format!("Expected '>' to close LIST<{element}")));
            }
            self.advance();
            element = format!("LIST<{element}>");
        }
        let kind = if levels == 0 {
            kind
        } else {
            PropertyTypeKind::Other
        };
        Ok((element, kind))
    }

    /// Reads one property of type DDL (ISO/IEC 39075:2024 18.6
    /// `<property type>`, and Grafeo's `DEFAULT`):
    /// `name TYPE [NOT NULL] [DEFAULT literal]`.
    ///
    /// The name may be any name a property map takes as a key, keywords such
    /// as `starts` or `match` included: an INSERT can write such a property,
    /// so its type can declare it.
    fn parse_property_definition(&mut self) -> Result<PropertyDefinition> {
        if !self.is_label_or_type_name() {
            return Err(self.error("Expected property name"));
        }
        let name = self.get_identifier_name();
        self.advance();

        let (data_type, kind) = self.parse_property_type_name()?;

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

        let default_value = if self.try_accept_identifier_keyword("DEFAULT") {
            Some(self.parse_default_literal(&data_type, kind, nullable)?)
        } else {
            None
        };

        Ok(PropertyDefinition {
            name,
            data_type,
            nullable,
            default_value,
        })
    }

    /// Reads the literal after `DEFAULT` for a property of type `data_type`
    /// (taking literals of `kind`): a string, a signed number (ISO/IEC
    /// 39075:2024 21.2 `<signed numeric literal>`), `TRUE`, `FALSE` or
    /// `NULL`. Returns it as [`PropertyDefinition::default_value`] spells it:
    /// an integer default of a `FLOAT64` property as the float.
    ///
    /// A literal that is not a value of the type is refused, and so is `NULL`
    /// for a `NOT NULL` property: they used to be accepted, and then every
    /// insert that relied on the default failed.
    fn parse_default_literal(
        &mut self,
        data_type: &str,
        kind: PropertyTypeKind,
        nullable: bool,
    ) -> Result<String> {
        let start = self.current.span.start;
        let negative = match self.current.kind {
            TokenKind::Minus | TokenKind::Plus => {
                let negative = self.current.kind == TokenKind::Minus;
                self.advance();
                if !matches!(self.current.kind, TokenKind::Integer | TokenKind::Float) {
                    return Err(self.error("Expected a number after the sign of a DEFAULT"));
                }
                Some(negative)
            }
            _ => None,
        };
        let written = self
            .source
            .get(start..self.current.span.end)
            .unwrap_or(&self.current.text)
            .to_string();
        let not_of_type = |parser: &Self| {
            parser.error(&format!(
                "DEFAULT {written} is not a value of type {data_type}"
            ))
        };

        let spelled = match self.current.kind {
            TokenKind::Null if nullable => "NULL".to_string(),
            TokenKind::Null => {
                return Err(self.error("DEFAULT NULL is not a value of a NOT NULL property"));
            }
            TokenKind::True | TokenKind::False => match kind {
                PropertyTypeKind::Boolean | PropertyTypeKind::Any => {
                    self.current.text.to_ascii_uppercase()
                }
                _ => return Err(not_of_type(self)),
            },
            TokenKind::String => match kind {
                PropertyTypeKind::String | PropertyTypeKind::Any => {
                    let text = &self.current.text;
                    format!("'{}'", unescape_string(&text[1..text.len() - 1]))
                }
                _ => return Err(not_of_type(self)),
            },
            TokenKind::Integer => {
                let Some(value) = integer_literal(&self.current.text, negative == Some(true))
                else {
                    return Err(self.error(&format!(
                        "DEFAULT {written} is out of the INT64 range ({} to {})",
                        i64::MIN,
                        i64::MAX
                    )));
                };
                match kind {
                    PropertyTypeKind::Integer | PropertyTypeKind::Any => value.to_string(),
                    PropertyTypeKind::Float => match exact_float(value) {
                        Some(float) => format!("{float:?}"),
                        None => {
                            return Err(self.error(&format!(
                                "DEFAULT {written} has no exact {data_type} value"
                            )));
                        }
                    },
                    _ => return Err(not_of_type(self)),
                }
            }
            TokenKind::Float => match kind {
                PropertyTypeKind::Float | PropertyTypeKind::Any => {
                    let value: f64 = self
                        .current
                        .text
                        .parse()
                        .map_err(|_| self.error("Invalid float"))?;
                    let value = if negative == Some(true) {
                        -value
                    } else {
                        value
                    };
                    format!("{value:?}")
                }
                _ => return Err(not_of_type(self)),
            },
            _ => return Err(self.error("Expected literal value after DEFAULT")),
        };
        self.advance();
        Ok(spelled)
    }

    /// Parses the property definitions of a type in parentheses:
    /// `( name TYPE [NOT NULL] [DEFAULT literal], ... )`.
    fn parse_property_definitions(&mut self) -> Result<Vec<PropertyDefinition>> {
        self.parse_property_definition_list(TokenKind::LParen, TokenKind::RParen)
    }

    /// Parses property definitions inside braces: `{ name TYPE [NOT NULL], ... }`.
    ///
    /// Used by pattern-form graph type syntax where properties use `{}` instead of `()`.
    fn parse_property_definitions_braces(&mut self) -> Result<Vec<PropertyDefinition>> {
        self.parse_property_definition_list(TokenKind::LBrace, TokenKind::RBrace)
    }

    /// Parses property definitions separated by commas between `open` and
    /// `close` (see [`Self::parse_property_definition`]).
    fn parse_property_definition_list(
        &mut self,
        open: TokenKind,
        close: TokenKind,
    ) -> Result<Vec<PropertyDefinition>> {
        self.expect(open)?;

        let mut defs = Vec::new();

        if self.current.kind != close {
            loop {
                defs.push(self.parse_property_definition()?);
                if self.current.kind != TokenKind::Comma {
                    break;
                }
                self.advance();
            }
        }

        self.expect(close)?;
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

    /// Whether the current token starts `=~`: a `=` with a `~` right after
    /// it. An expression never starts with `~`, so after an operand this is
    /// the regular expression match; with a space between (`= ~`) it is
    /// not, as in Cypher, where `=~` is one token. A path variable before
    /// an undirected edge (`p=~[e]~(b)`) is not read as an expression.
    fn at_regex_match(&mut self) -> bool {
        if self.current.kind != TokenKind::Eq || self.peek_kind() != TokenKind::Tilde {
            return false;
        }
        let equals_end = self.current.span.end;
        self.peeked
            .as_ref()
            .is_some_and(|tilde| tilde.span.start == equals_end)
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

/// Whether `a op b op c` means the same however it is grouped, so a chain of
/// `op` can be a balanced tree: the unions, the intersections and
/// `OTHERWISE`, not `EXCEPT` or `NEXT`.
fn is_associative(op: CompositeOp) -> bool {
    match op {
        CompositeOp::Union
        | CompositeOp::UnionAll
        | CompositeOp::Intersect
        | CompositeOp::IntersectAll
        | CompositeOp::Otherwise => true,
        CompositeOp::Except | CompositeOp::ExceptAll | CompositeOp::Next => false,
    }
}

/// What a property type of type DDL takes as a `DEFAULT`.
type PropertyTypeKind = crate::query::schema::PropertyTypeKind;

/// The value of the integer literal `text` (decimal, or `0x`, `0o`, `0b`
/// digits), negated when `negative`; `None` when it is out of the INT64
/// range. The sign counts before the range check, so the smallest INT64,
/// `-9223372036854775808`, is a literal.
fn integer_literal(text: &str, negative: bool) -> Option<i64> {
    let (digits, radix) = match text.get(..2) {
        Some("0x" | "0X") => (&text[2..], 16),
        Some("0o" | "0O") => (&text[2..], 8),
        Some("0b" | "0B") => (&text[2..], 2),
        _ => (text, 10),
    };
    let magnitude = i128::from(u64::from_str_radix(digits, radix).ok()?);
    i64::try_from(if negative { -magnitude } else { magnitude }).ok()
}

/// `value` as a float, when the float is exactly `value`: every integer up to
/// 2^53 in magnitude, and the larger ones a float holds.
fn exact_float(value: i64) -> Option<f64> {
    let decimal = value.to_string();
    let float: f64 = decimal.parse().ok()?;
    // A float that holds an integer prints as that integer without
    // fractional digits; the nearest float to another integer prints as a
    // different one.
    (format!("{float:.0}") == decimal).then_some(float)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first WHERE or FILTER among the clauses of `query`.
    fn filter_of(query: &QueryStatement) -> Option<WhereClause> {
        query
            .ordered_clauses
            .iter()
            .find_map(|clause| match clause {
                QueryClause::Filter(filter) => Some(filter.clone()),
                _ => None,
            })
    }

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
            assert!(filter_of(&query).is_some(), "WHERE clause should be parsed");
            let where_clause = filter_of(&query).unwrap();
            if let Expression::Binary { op, .. } = &where_clause.expression {
                assert_eq!(*op, BinaryOp::Gt);
            } else {
                panic!("Expected binary expression in WHERE clause");
            }
        } else {
            panic!("Expected Query statement");
        }
    }

    /// The WHERE expression of `query`, a GQL query statement.
    fn where_expression(query: &str) -> Expression {
        let Statement::Query(statement) = Parser::new(query).parse().unwrap() else {
            panic!("{query} is not a query statement");
        };
        filter_of(&statement)
            .unwrap_or_else(|| panic!("{query} has no WHERE"))
            .expression
    }

    /// Whether `expression` is `left =~ right` with a string literal `right`.
    fn is_regex_match(expression: &Expression, pattern: &str) -> bool {
        matches!(
            expression,
            Expression::Binary { op: BinaryOp::RegexMatch, right, .. }
                if matches!(right.as_ref(), Expression::Literal(Literal::String(p)) if p == pattern)
        )
    }

    #[test]
    fn regex_match_parses_as_a_comparison() {
        let expression =
            where_expression("MATCH (n) WHERE n.path =~ '.*(test|spec).*' RETURN n.path");
        assert!(
            is_regex_match(&expression, ".*(test|spec).*"),
            "{expression:?}"
        );
        let Expression::Binary { left, .. } = &expression else {
            unreachable!()
        };
        assert!(
            matches!(left.as_ref(), Expression::PropertyAccess { variable, property }
                if variable == "n" && property == "path"),
            "{left:?}"
        );
    }

    #[test]
    fn regex_match_binds_like_the_other_comparisons() {
        // Looser than `||`, tighter than AND and NOT, as `=~` in Cypher.
        let expression =
            where_expression("MATCH (n) WHERE n.a || 'x' =~ 'ax' AND n.b = 3 RETURN n");
        let Expression::Binary {
            left,
            op: BinaryOp::And,
            ..
        } = &expression
        else {
            panic!("AND is not the top operator: {expression:?}");
        };
        assert!(is_regex_match(left, "ax"), "{left:?}");
        let Expression::Binary { left: operand, .. } = left.as_ref() else {
            unreachable!()
        };
        assert!(
            matches!(
                operand.as_ref(),
                Expression::Binary {
                    op: BinaryOp::Concat,
                    ..
                }
            ),
            "the left operand is not `n.a || 'x'`: {operand:?}"
        );

        let negated = where_expression("MATCH (n) WHERE NOT n.a =~ 'x.*' RETURN n");
        assert!(
            matches!(&negated, Expression::Unary { op: UnaryOp::Not, operand }
                if is_regex_match(operand, "x.*")),
            "{negated:?}"
        );
    }

    #[test]
    fn regex_match_parses_in_return_and_set() {
        let Statement::Query(query) = Parser::new("MATCH (n) RETURN n.path =~ 'src/.*' AS in_src")
            .parse()
            .unwrap()
        else {
            panic!("not a query");
        };
        let item = &query.return_clause.items[0];
        assert!(is_regex_match(&item.expression, "src/.*"), "{item:?}");

        let set = Parser::new("MATCH (n) SET n.in_src = n.path =~ 'src/.*'").parse();
        assert!(set.is_ok(), "{set:?}");
    }

    #[test]
    fn equals_then_a_tilde_with_a_space_is_not_a_regex_match() {
        // `=~` is one operator, as in Cypher: `= ~` is an equality with no
        // right operand.
        let result = Parser::new("MATCH (n) WHERE n.a = ~'x' RETURN n").parse();
        assert!(result.is_err(), "{result:?}");
    }

    #[test]
    fn a_path_variable_before_an_undirected_edge_still_parses() {
        // `p=~[e]~(b)` is a path variable and an undirected edge, not `=~`.
        let result = Parser::new("MATCH p=~[e]~(b) RETURN p").parse();
        assert!(result.is_ok(), "{result:?}");
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
            let where_clause = filter_of(&query).expect("Expected WHERE clause");
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
            let where_clause = filter_of(&query).expect("Expected WHERE clause");
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
            let where_clause = filter_of(&q).expect("Expected WHERE clause");
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
            let where_clause = filter_of(&q).expect("Expected WHERE clause");
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
            let where_clause = filter_of(&q).expect("Expected WHERE clause");
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
        let err = parse_err("MATCH (n) FILTER n.a = 1 WHERE n.b = 2 RETURN n");
        assert!(err.contains("combine the conditions with AND"), "{err}");
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

    /// The types of the first edge of `query`'s first MATCH.
    fn first_edge_types(query: &str) -> Vec<String> {
        let Ok(Statement::Query(query)) = Parser::new(query).parse() else {
            panic!("{query}: expected a query");
        };
        let Pattern::Path(path) = &query.match_clauses[0].patterns[0].pattern else {
            panic!("expected a path");
        };
        path.edges[0].types.clone()
    }

    #[test]
    fn an_edge_has_one_type_or_alternatives() {
        // A disjunction of label names (ISO/IEC 39075:2024, 16.8)
        assert_eq!(first_edge_types("MATCH (a)-[:A]->(b) RETURN b"), ["A"]);
        assert_eq!(
            first_edge_types("MATCH (a)<-[e:A|`Graph:CONTAINS`*1..3]-(b) RETURN b"),
            ["A", "Graph:CONTAINS"]
        );
        assert_eq!(
            first_edge_types("MATCH (a)~[:A|B]~(b) RETURN b"),
            ["A", "B"]
        );
        // A second `:` is no label expression, and an edge with two labels
        // (`&`) would match none: in every bracket form, MATCH and INSERT
        for (query, joint) in [
            ("MATCH (a)-[:Graph:CONTAINS]->(b) RETURN b", ':'),
            ("MATCH (a)<-[:Graph:CONTAINS*]-(b) RETURN b", ':'),
            ("MATCH (a)~[:Graph:CONTAINS]~(b) RETURN b", ':'),
            ("MATCH (a)-[:A|Graph:CONTAINS]->{1,3}(b) RETURN b", ':'),
            ("INSERT (a)-[:Graph:CONTAINS]->(b)", ':'),
            ("MATCH (a)-[:Graph&CONTAINS]->(b) RETURN b", '&'),
            ("INSERT (a)-[:Graph&CONTAINS]->(b)", '&'),
        ] {
            let err = parse_err(query);
            for advice in [
                format!(":`Graph{joint}CONTAINS`"),
                ":Graph|CONTAINS".to_string(),
            ] {
                assert!(err.contains(&advice), "{query}: names {advice}: {err}");
            }
        }
    }

    #[test]
    fn an_inserted_edge_has_one_type() {
        for query in [
            "INSERT (a)-[:A|B]->(b)",
            "MATCH (a), (b) INSERT (a)-[:A]->(b), (b)-[:A|B]->(a)",
            "MATCH (a), (b) CREATE (a)-[:A|B]->(b)",
            "MATCH (a) MERGE (a)-[:A|B]->(b:N)",
        ] {
            let err = parse_err(query);
            assert!(
                err.contains("one type") && err.contains(":A|B"),
                "{query}: {err}"
            );
        }
        assert_eq!(
            first_edge_types("MATCH (a)-[:A|B]->(b) INSERT (a)-[:A]->(b)"),
            ["A", "B"]
        );
    }

    #[test]
    fn test_trailing_input_is_rejected() {
        let err = parse_err("MATCH (n) RETURN n garbage");
        assert!(
            err.contains("unexpected 'garbage' after the end of the statement"),
            "{err}"
        );

        // A clause after the result statement fails loudly instead of
        // silently dropping the rest of the statement.
        let err = parse_err("MATCH (n) RETURN n DELETE n");
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
        assert_eq!(
            properties("ALTER NODE TYPE Sensor ADD seen local datetime NOT NULL"),
            vec![pair("seen", "LOCAL DATETIME")]
        );
    }

    /// The property types of #569: `ZONED DATETIME`, `LOCAL DATETIME` and
    /// `LIST<type>`, nested, in the spelling the catalog reads back.
    #[test]
    fn test_property_types_with_two_words_and_typed_lists() {
        let types = |query: &str| -> Vec<(String, String, bool)> {
            let Statement::Schema(SchemaStatement::CreateNodeType(stmt)) =
                Parser::new(query).parse().unwrap()
            else {
                panic!("{query}: expected CREATE NODE TYPE");
            };
            stmt.properties
                .into_iter()
                .map(|def| (def.name, def.data_type, def.nullable))
                .collect()
        };
        let property = |name: &str, data_type: &str, nullable: bool| {
            (name.to_string(), data_type.to_string(), nullable)
        };

        assert_eq!(
            types("CREATE NODE TYPE T (a ZONED DATETIME, b LIST<LIST<STRING>> NOT NULL)"),
            vec![
                property("a", "ZONED DATETIME", true),
                property("b", "LIST<LIST<STRING>>", false),
            ]
        );
        assert_eq!(
            types(
                "CREATE NODE TYPE T (a local datetime DEFAULT NULL, \
                 b list<zoned datetime>, c LIST, d STRING)"
            ),
            vec![
                property("a", "LOCAL DATETIME", true),
                property("b", "LIST<ZONED DATETIME>", true),
                property("c", "LIST", true),
                property("d", "STRING", true),
            ]
        );

        let Statement::Schema(SchemaStatement::CreateGraphType(stmt)) = Parser::new(
            "CREATE GRAPH TYPE g ((:Stop {at ZONED DATETIME})-[:LEG {at LIST<LOCAL DATETIME>}]->(:Stop))",
        )
        .parse()
        .unwrap() else {
            panic!("expected CREATE GRAPH TYPE");
        };
        let inline_types: Vec<String> = stmt
            .inline_types
            .iter()
            .flat_map(|inline| match inline {
                InlineElementType::Node { properties, .. }
                | InlineElementType::Edge { properties, .. } => properties,
            })
            .map(|def| def.data_type.clone())
            .collect();
        assert_eq!(inline_types, ["ZONED DATETIME", "LIST<LOCAL DATETIME>"]);

        for (query, expected) in [
            ("CREATE NODE TYPE T (a ZONED TIME)", "Expected DATETIME"),
            ("CREATE NODE TYPE T (a LOCAL)", "Expected DATETIME"),
            ("CREATE NODE TYPE T (a LIST<STRING)", "Expected '>'"),
            ("CREATE NODE TYPE T (a LIST<)", "Expected type name"),
            ("CREATE NODE TYPE T (a LIST<LIST<STRING>)", "Expected '>'"),
        ] {
            let error = Parser::new(query).parse().unwrap_err().to_string();
            assert!(error.contains(expected), "{query}: {error}");
        }
    }

    /// A property type nests at most 128 `LIST<...>` levels, as deep as a
    /// property value may be; a deeper one, however deep, is refused with the
    /// limit named, without a stack overflow.
    #[test]
    fn test_property_types_nest_at_most_128_lists() {
        let nested =
            |levels: usize| format!("{}STRING{}", "LIST<".repeat(levels), ">".repeat(levels));
        let query = |levels: usize| format!("CREATE NODE TYPE T (a {} NOT NULL)", nested(levels));

        let Statement::Schema(SchemaStatement::CreateNodeType(stmt)) =
            Parser::new(&query(128)).parse().unwrap()
        else {
            panic!("expected CREATE NODE TYPE");
        };
        assert_eq!(stmt.properties[0].data_type, nested(128));
        assert!(!stmt.properties[0].nullable);

        for levels in [129, 100_000] {
            let error = Parser::new(&query(levels)).parse().unwrap_err().to_string();
            assert!(
                error.contains("at most 128 LIST<...> levels"),
                "{levels} levels: {error}"
            );
        }
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
            let where_clause = filter_of(&q).expect("Expected WHERE clause");
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

    #[test]
    fn test_parse_dotted_map_key_access() {
        let return_expression = |query: &str| {
            let Statement::Query(q) = Parser::new(query).parse().unwrap() else {
                panic!("Expected Query statement");
            };
            format!("{:?}", q.return_clause.items[0].expression)
        };
        let meta = Expression::PropertyAccess {
            variable: "n".to_string(),
            property: "meta".to_string(),
        };
        let key = |base: Expression, key: &str| Expression::MapAccess {
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
            format!("{:?}", key(key(meta.clone(), "a"), "b"))
        );
        // Keywords are valid keys, as they are valid property names.
        assert_eq!(
            return_expression("MATCH (n) RETURN n.meta.type"),
            format!("{:?}", key(meta, "type"))
        );
        let err = Parser::new("MATCH (n) RETURN n.meta.")
            .parse()
            .unwrap_err()
            .to_string();
        assert!(err.contains("Expected map key"), "got: {err}");
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

    /// ISO/IEC 39075:2024 puts the match mode before the path pattern prefix:
    /// `MATCH DIFFERENT EDGES TRAIL (...)`. The path mode first still parses.
    #[test]
    fn a_match_mode_and_a_path_mode_parse_in_either_order() {
        for (query, mode) in [
            (
                "MATCH DIFFERENT EDGES TRAIL (a)-[]->(b) RETURN a",
                MatchMode::DifferentEdges,
            ),
            (
                "MATCH TRAIL DIFFERENT EDGES (a)-[]->(b) RETURN a",
                MatchMode::DifferentEdges,
            ),
            (
                "MATCH REPEATABLE ELEMENTS TRAIL (a)-[]->(b) RETURN a",
                MatchMode::RepeatableElements,
            ),
        ] {
            let Statement::Query(q) = Parser::new(query).parse().unwrap() else {
                panic!("expected a query: {query}");
            };
            assert_eq!(
                q.match_clauses[0].path_mode,
                Some(PathMode::Trail),
                "{query}"
            );
            assert_eq!(q.match_clauses[0].match_mode, Some(mode), "{query}");
        }
    }

    /// The path pattern prefixes of ISO/IEC 39075:2024 16.6: each search
    /// prefix with its optional path mode and `PATH` or `PATHS`, after the
    /// path variable or after MATCH, where they are the clause's.
    #[test]
    fn path_pattern_prefixes_parse_with_their_path_modes() {
        use PathSearchPrefix as Prefix;
        for (prefix, search, mode) in [
            ("TRAIL", None, Some(PathMode::Trail)),
            ("ACYCLIC PATHS", None, Some(PathMode::Acyclic)),
            ("ALL", Some(Prefix::All), None),
            ("ALL SIMPLE PATH", Some(Prefix::All), Some(PathMode::Simple)),
            ("ANY", Some(Prefix::Any), None),
            ("ANY TRAIL", Some(Prefix::Any), Some(PathMode::Trail)),
            (
                "ANY 3 ACYCLIC PATHS",
                Some(Prefix::AnyK(3)),
                Some(PathMode::Acyclic),
            ),
            (
                "ANY SHORTEST WALK",
                Some(Prefix::AnyShortest),
                Some(PathMode::Walk),
            ),
            (
                "ALL SHORTEST TRAIL PATHS",
                Some(Prefix::AllShortest),
                Some(PathMode::Trail),
            ),
            ("SHORTEST 19", Some(Prefix::ShortestK(19)), None),
            (
                "SHORTEST 3 SIMPLE PATHS",
                Some(Prefix::ShortestK(3)),
                Some(PathMode::Simple),
            ),
            ("SHORTEST GROUPS", Some(Prefix::ShortestKGroups(1)), None),
            (
                "SHORTEST 2 TRAIL PATH GROUP",
                Some(Prefix::ShortestKGroups(2)),
                Some(PathMode::Trail),
            ),
        ] {
            let query = format!("MATCH p = {prefix} (a)-[:KNOWS]->+(b) RETURN p");
            let Statement::Query(q) = Parser::new(&query).parse().unwrap() else {
                panic!("expected a query: {query}");
            };
            let pattern = &q.match_clauses[0].patterns[0];
            assert_eq!(pattern.alias.as_deref(), Some("p"), "{query}");
            assert_eq!(pattern.search_prefix, search, "{query}");
            assert_eq!(pattern.path_mode, mode, "{query}");

            let query = format!("MATCH {prefix} (a)-[:KNOWS]->+(b) RETURN a");
            let Statement::Query(q) = Parser::new(&query).parse().unwrap() else {
                panic!("expected a query: {query}");
            };
            assert_eq!(q.match_clauses[0].search_prefix, search, "{query}");
            assert_eq!(q.match_clauses[0].path_mode, mode, "{query}");
        }
    }

    /// A path variable may be named `path`, and a later pattern of a MATCH
    /// takes a prefix of its own.
    #[test]
    fn a_prefix_keyword_is_not_a_path_variable() {
        let query = "MATCH ANY path = (a)-[:KNOWS]->+(b), ANY SHORTEST TRAIL (b)-[:KNOWS]->+(c) \
                     RETURN path";
        let Statement::Query(q) = Parser::new(query).parse().unwrap() else {
            panic!("expected a query");
        };
        let clause = &q.match_clauses[0];
        assert_eq!(clause.search_prefix, Some(PathSearchPrefix::Any));
        assert_eq!(clause.patterns[0].alias.as_deref(), Some("path"));
        assert_eq!(clause.patterns[0].search_prefix, None);
        assert_eq!(
            clause.patterns[1].search_prefix,
            Some(PathSearchPrefix::AnyShortest)
        );
        assert_eq!(clause.patterns[1].path_mode, Some(PathMode::Trail));
    }

    #[test]
    fn a_wrong_path_pattern_prefix_is_a_syntax_error() {
        for (query, message) in [
            ("MATCH ANY 0 (a)-->(b) RETURN a", "at least 1"),
            (
                "MATCH SHORTEST 0 GROUPS (a)-->(b) RETURN a",
                "number of groups",
            ),
            (
                "MATCH ANY 99999999999999999999999 (a)-->(b) RETURN a",
                "too large",
            ),
            (
                "MATCH TRAIL ANY SHORTEST ACYCLIC (a)-->(b) RETURN a",
                "path mode",
            ),
            ("MATCH TRAIL p = ACYCLIC (a)-->(b) RETURN a", "path mode"),
            (
                "MATCH ANY p = ANY SHORTEST (a)-->(b) RETURN a",
                "search prefix",
            ),
            (
                "MATCH (a)-->(b) | ANY (a)-->(c) RETURN a",
                "whole alternation",
            ),
        ] {
            let error = Parser::new(query).parse().unwrap_err().to_string();
            assert!(error.contains(message), "{query}: {error}");
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
            assert!(filter_of(&q).is_some());
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
        assert!(edges[0].types.is_empty(), "expected empty");
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
        assert!(edges[0].types.is_empty(), "expected empty");
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
        assert!(edges[0].types.is_empty(), "expected empty");
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

    /// ISO/IEC 39075:2024 writes a graph type's elements in braces
    /// (`<nested graph type specification>`): the brace form declares the
    /// same element types, with their properties and endpoints, as Grafeo's
    /// paren form. It used to keep the type names only.
    #[test]
    fn test_parse_graph_type_brace_form_declares_what_the_paren_form_declares() {
        let elements = "(:City {name STRING NOT NULL, country STRING DEFAULT 'NL'})\
             -[:ROUTE {km INT64}]->(:City), \
             (:Country {code STRING}), (:City)<-[:CAPITAL]-(:Country)";
        let parse = |query: String| match Parser::new(&query).parse() {
            Ok(Statement::Schema(SchemaStatement::CreateGraphType(stmt))) => stmt,
            other => panic!("{query}: expected CreateGraphType, got {other:?}"),
        };
        let brace = parse(format!("CREATE GRAPH TYPE routes {{ {elements} }}"));
        let paren = parse(format!("CREATE GRAPH TYPE routes ( {elements} )"));

        assert_eq!(format!("{brace:?}"), format!("{paren:?}"));
        assert_eq!(brace.node_types, ["City", "Country"]);
        assert_eq!(brace.edge_types, ["ROUTE", "CAPITAL"]);
        assert!(!brace.open, "an element list closes the graph type");
        let declared: Vec<String> = brace
            .inline_types
            .iter()
            .map(|element| match element {
                InlineElementType::Node {
                    name, properties, ..
                } => format!("{name} {properties:?}"),
                InlineElementType::Edge {
                    name,
                    properties,
                    source_node_types,
                    target_node_types,
                    ..
                } => format!("{source_node_types:?}-{name}->{target_node_types:?} {properties:?}"),
            })
            .collect();
        assert_eq!(declared.len(), 4, "{declared:#?}");
        assert!(
            declared[0].starts_with("City ")
                && declared[0].contains("name: \"name\"")
                && declared[0].contains("default_value: Some(\"'NL'\")"),
            "{declared:#?}"
        );
        assert!(
            declared[1].starts_with("[\"City\"]-ROUTE->[\"City\"]")
                && declared[1].contains("name: \"km\""),
            "{declared:#?}"
        );
        assert!(
            declared[2].starts_with("Country ") && declared[2].contains("name: \"code\""),
            "{declared:#?}"
        );
        assert!(
            declared[3].starts_with("[\"Country\"]-CAPITAL->[\"City\"]"),
            "{declared:#?}"
        );
    }

    /// The brace form with the keys `node_types`, `edge_types` and `open`
    /// still lists type names without declaring them.
    #[test]
    fn test_parse_graph_type_brace_form_with_type_lists_declares_nothing() {
        let query = "CREATE GRAPH TYPE g { node_types: [City], edge_types: [ROUTE], open: true }";
        match Parser::new(query).parse() {
            Ok(Statement::Schema(SchemaStatement::CreateGraphType(stmt))) => {
                assert_eq!(stmt.node_types, ["City"]);
                assert_eq!(stmt.edge_types, ["ROUTE"]);
                assert!(stmt.open);
                assert!(stmt.inline_types.is_empty(), "{:?}", stmt.inline_types);
            }
            other => panic!("{query}: expected CreateGraphType, got {other:?}"),
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

    // --- Clause order ---

    #[test]
    fn test_ordered_clauses_keep_remove_and_with_in_place() {
        let mut parser = Parser::new("MATCH (a:P) REMOVE a.w WITH a MATCH (a)-[:K]->(b) RETURN b");
        let Statement::Query(query) = parser.parse().unwrap() else {
            panic!("Expected Query statement");
        };
        let kinds: Vec<&str> = query
            .ordered_clauses
            .iter()
            .map(|clause| match clause {
                QueryClause::Match(_) => "MATCH",
                QueryClause::Remove(_) => "REMOVE",
                QueryClause::With(_) => "WITH",
                _ => "other",
            })
            .collect();
        assert_eq!(kinds, ["MATCH", "REMOVE", "WITH", "MATCH"]);
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
        assert!(filter_of(q_upper).is_some());
        assert!(filter_of(q_lower).is_some());
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
    fn test_long_power_and_sign_chains_error_not_stack_overflow() {
        for query in [
            format!("RETURN 2{}", " ^ 2".repeat(50_000)),
            format!("RETURN {}1", "- ".repeat(50_000)),
            format!("RETURN {}1", "+ ".repeat(50_000)),
            format!("RETURN {}true", "NOT ".repeat(50_000)),
        ] {
            let err = Parser::new(&query).parse().unwrap_err().to_string();
            assert!(err.contains("nesting depth"), "got: {err}");
        }
        // Short chains are unaffected.
        assert!(
            Parser::new(&format!("RETURN 2{}", " ^ 1".repeat(20)))
                .parse()
                .is_ok()
        );
        assert!(Parser::new("RETURN - - 1").parse().is_ok());
        assert!(Parser::new("RETURN NOT NOT true").parse().is_ok());
    }

    #[test]
    fn test_nesting_at_exact_limit_succeeds() {
        use crate::query::limits::MAX_NESTING_DEPTH;
        // The RETURN item is an expression (one level), each pair of
        // parentheses one more: the limit itself is accepted, one more not.
        let at_limit = MAX_NESTING_DEPTH as usize - 1;
        let parse = |depth: usize| {
            let deep = "(".repeat(depth) + "1" + &")".repeat(depth);
            Parser::new(&format!("RETURN {deep}")).parse()
        };
        assert!(parse(at_limit).is_ok(), "the limit is inclusive");
        let error = parse(at_limit + 1).unwrap_err().to_string();
        assert!(
            error.contains(&format!("nesting depth of {MAX_NESTING_DEPTH}")),
            "{error}"
        );
    }

    #[test]
    fn test_chains_of_operators_count_toward_the_nesting_limit() {
        use crate::query::limits::MAX_NESTING_DEPTH;
        // `1 + 1 + ... + 1` with n operators nests n levels in the expression
        // of the RETURN item, which nests one: a chain is parsed in a loop,
        // but every later stage recurses into the tree it builds.
        let operators = MAX_NESTING_DEPTH as usize - 1;
        let sum = |operators: usize| {
            let terms = vec!["1"; operators + 1].join(" + ");
            Parser::new(&format!("RETURN {terms}")).parse()
        };
        assert!(sum(operators).is_ok());
        assert!(sum(operators + 1).is_err());
        // A chain over parenthesized operands counts the deeper side once:
        // (1 + 1) + (1 + 1) + ... nests one level per operator plus two.
        let pairs = |count: usize| {
            let terms = vec!["(1 + 1)"; count].join(" + ");
            Parser::new(&format!("RETURN {terms}")).parse()
        };
        assert!(pairs(operators - 2).is_ok());
        assert!(pairs(operators).is_err());
        // A long chain fails with the limit, without a stack overflow.
        let error = sum(100_000).unwrap_err().to_string();
        assert!(error.contains("nesting depth"), "{error}");
    }

    #[test]
    fn test_and_or_xor_and_union_chains_are_balanced_trees() {
        /// The depth of the tree of `op` at `expression`, and its leaves.
        fn shape(expression: &Expression, op: BinaryOp, leaves: &mut Vec<i64>) -> usize {
            match expression {
                Expression::Binary {
                    left,
                    op: found,
                    right,
                } if *found == op => 1 + shape(left, op, leaves).max(shape(right, op, leaves)),
                Expression::Literal(Literal::Integer(leaf)) => {
                    leaves.push(*leaf);
                    0
                }
                other => panic!("an operand of {op:?} or a leaf: {other:?}"),
            }
        }
        for (keyword, op) in [
            ("OR", BinaryOp::Or),
            ("AND", BinaryOp::And),
            ("XOR", BinaryOp::Xor),
        ] {
            let terms: Vec<String> = (0..100_000).map(|i| i.to_string()).collect();
            let query = format!("RETURN {} AS v", terms.join(&format!(" {keyword} ")));
            let statement = Parser::new(&query).parse().unwrap();
            let Statement::Query(query) = statement else {
                panic!("a query")
            };
            let mut leaves = Vec::new();
            let depth = shape(&query.return_clause.items[0].expression, op, &mut leaves);
            assert_eq!(depth, 17, "{keyword}: ceil(log2(100,000)) levels");
            assert_eq!(
                leaves,
                (0..100_000).collect::<Vec<i64>>(),
                "{keyword}: in order"
            );
        }
        // A chain of one associative set operator is balanced too; EXCEPT
        // still nests one level per operator.
        let union = vec!["RETURN 3 AS v"; 10_000].join(" UNION ALL ");
        assert!(Parser::new(&union).parse().is_ok());
        let except = |count: usize| {
            let query = vec!["RETURN 3 AS v"; count].join(" EXCEPT ");
            Parser::new(&query).parse()
        };
        let limit = crate::query::limits::MAX_NESTING_DEPTH as usize;
        assert!(except(limit / 2).is_ok());
        assert!(except(limit + 2).is_err());
        // An operand deeper than the rest counts where it sits in the tree.
        let deep = format!("{}3{}", "(".repeat(55), ")".repeat(55));
        let chain = |count: usize| {
            let mut terms = vec!["3".to_string(); count];
            terms[0] = deep.clone();
            Parser::new(&format!("RETURN {}", terms.join(" OR "))).parse()
        };
        // The expression, 55 parentheses and the joins above the operand.
        assert!(chain(64).is_ok(), "six joins above the operand");
        assert!(chain(1_000).is_err(), "ten joins above the operand");
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

    /// The options of `CREATE INDEX ... USING TEXT {...}` reach the statement.
    #[test]
    fn create_text_index_takes_bm25_tokenizer_and_stop_word_options() {
        let parsed = Parser::new(
            "CREATE INDEX notes FOR (n:Note) ON (n.body) USING TEXT \
             {K1: 1.5, b: 0, tokenizer: 'cjk_bigram', stop_words: ['и', \"в\"]}",
        )
        .parse()
        .expect("text index options parse");
        let Statement::Schema(SchemaStatement::CreateIndex(stmt)) = parsed else {
            panic!("a CREATE INDEX statement: {parsed:?}");
        };
        assert_eq!(stmt.index_kind, IndexKind::Text);
        assert_eq!(stmt.options.k1, Some(1.5));
        assert_eq!(stmt.options.b, Some(0.0), "an integer is a number too");
        assert_eq!(stmt.options.tokenizer.as_deref(), Some("cjk_bigram"));
        assert_eq!(
            stmt.options.stop_words,
            Some(vec!["и".to_string(), "в".to_string()])
        );

        let negative = Parser::new(
            "CREATE INDEX notes FOR (n:Note) ON (n.body) USING TEXT {k1: -3, stop_words: []}",
        )
        .parse()
        .expect("a negative k1 parses; the engine refuses it");
        let Statement::Schema(SchemaStatement::CreateIndex(stmt)) = negative else {
            panic!("a CREATE INDEX statement: {negative:?}");
        };
        assert_eq!(stmt.options.k1, Some(-3.0));
        assert_eq!(stmt.options.stop_words, Some(Vec::new()));

        let plain = Parser::new("CREATE INDEX notes FOR (n:Note) ON (n.body) USING TEXT")
            .parse()
            .expect("no options");
        let Statement::Schema(SchemaStatement::CreateIndex(stmt)) = plain else {
            panic!("a CREATE INDEX statement: {plain:?}");
        };
        assert_eq!(
            (
                stmt.options.k1,
                stmt.options.b,
                stmt.options.tokenizer,
                stmt.options.stop_words
            ),
            (None, None, None, None)
        );
    }

    #[test]
    fn unknown_or_mistyped_text_index_options_are_errors() {
        for (options, says) in [
            (
                "{analyzer: 'standard'}",
                "Unknown text index option 'analyzer'",
            ),
            ("{k1: 'high'}", "Expected a number for k1"),
            ("{tokenizer: standard}", "Expected string for tokenizer"),
            ("{stop_words: ['and', 3]}", "Expected string in stop_words"),
        ] {
            let error = Parser::new(&format!(
                "CREATE INDEX notes FOR (n:Note) ON (n.body) USING TEXT {options}"
            ))
            .parse()
            .expect_err(options);
            assert!(error.to_string().contains(says), "{options}: {error}");
        }
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

    /// An inline CALL body combines queries with set operators, left to right.
    #[test]
    fn test_parse_inline_call_combined_body() {
        let combined_of = |query: &str| {
            let Statement::Query(statement) = Parser::new(query).parse().unwrap() else {
                panic!("expected a query: {query}");
            };
            statement
                .ordered_clauses
                .into_iter()
                .find_map(|clause| match clause {
                    QueryClause::InlineCall { combined, .. } => {
                        Some(combined.into_iter().map(|(op, _)| op).collect::<Vec<_>>())
                    }
                    _ => None,
                })
                .unwrap_or_else(|| panic!("expected an inline CALL: {query}"))
        };
        assert_eq!(combined_of("CALL { RETURN 1 AS x } RETURN x"), []);
        assert_eq!(
            combined_of(
                "MATCH (a) CALL (a) { RETURN 1 AS x UNION ALL RETURN 2 AS x EXCEPT RETURN 3 AS x } RETURN x"
            ),
            [CompositeOp::UnionAll, CompositeOp::Except]
        );
        assert_eq!(
            combined_of("CALL { RETURN 1 AS x UNION DISTINCT RETURN 1 AS x } RETURN x"),
            [CompositeOp::Union]
        );
    }

    /// The variable scope clause of an inline CALL: `(a, b)` names the outer
    /// variables the subquery sees, `()` none, and no clause all of them. A
    /// procedure call has its name before any parenthesis.
    #[test]
    fn test_parse_inline_call_scope_clause() {
        let scope_of = |query: &str| {
            let Statement::Query(statement) = Parser::new(query).parse().unwrap() else {
                panic!("expected a query: {query}");
            };
            statement
                .ordered_clauses
                .iter()
                .find_map(|clause| match clause {
                    QueryClause::InlineCall {
                        scope, optional, ..
                    } => Some((scope.clone(), *optional)),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("expected an inline CALL: {query}"))
        };
        let names = |names: &[&str]| Some(names.iter().map(ToString::to_string).collect());
        assert_eq!(
            scope_of("MATCH (a), (b) CALL (a, b) { RETURN 1 AS x } RETURN x"),
            (names(&["a", "b"]), false)
        );
        assert_eq!(
            scope_of("MATCH (a) CALL () { RETURN 1 AS x } RETURN x"),
            (names(&[]), false)
        );
        assert_eq!(
            scope_of("MATCH (a) CALL { RETURN 1 AS x } RETURN x"),
            (None, false)
        );
        assert_eq!(
            scope_of("MATCH (a) OPTIONAL CALL (a) { RETURN 1 AS x } RETURN x"),
            (names(&["a"]), true)
        );
        assert_eq!(
            scope_of("CALL () { RETURN 1 AS x } RETURN x"),
            (names(&[]), false)
        );
        assert!(
            Parser::new("MATCH (a) CALL (1) { RETURN 1 AS x } RETURN x")
                .parse()
                .is_err()
        );
        assert!(matches!(
            Parser::new("CALL db.labels()").parse().unwrap(),
            Statement::Call(_)
        ));
    }

    /// A statement that modifies data needs no result statement (ISO GQL's
    /// linear data-modifying statement), and an inline CALL whose body
    /// modifies data is such a statement (a call of a data-modifying
    /// procedure): a query may end with it. A query that only reads still
    /// needs one, also when it ends with a CALL that only reads.
    #[test]
    fn test_parse_query_ending_with_a_data_modifying_call() {
        for query in [
            "MATCH (n) CALL { INSERT (:X) }",
            "MATCH (n) CALL (n) { SET n.k = 3 }",
            "FOR i IN [3, 19] CALL (i) { FOR x IN [i] INSERT (:X {x: x}) }",
            "MATCH (n) OPTIONAL CALL (n) { MATCH (n)-[e]->() DELETE e }",
            "MATCH (n) CALL { CALL { INSERT (:X) } }",
            "MATCH (n) CALL (n) { INSERT (x:X) RETURN x }",
            "MATCH (n) CALL { INSERT (:X) } MATCH (m)",
        ] {
            let Statement::Query(statement) = Parser::new(query)
                .parse()
                .unwrap_or_else(|error| panic!("`{query}` must parse: {error}"))
            else {
                panic!("expected a query: {query}");
            };
            let result = &statement.return_clause;
            assert!(
                result.items.is_empty() && !result.is_wildcard && !result.is_finish,
                "`{query}` has no result statement: {result:?}"
            );
        }
        for query in [
            "MATCH (n) CALL { MATCH (m) RETURN m }",
            "MATCH (n) CALL (n) { MATCH (n)-[]->(m) FINISH }",
            "MATCH (n) CALL { RETURN 1 AS x UNION RETURN 2 AS x }",
        ] {
            let error = Parser::new(query)
                .parse()
                .expect_err(&format!("`{query}` reads only and has no result"));
            assert!(
                error
                    .to_string()
                    .contains("Expected RETURN, FINISH, or SELECT"),
                "`{query}`: {error}"
            );
        }
    }

    // ==================== Statements in any order (#483) ====================
    //
    // ISO/IEC 39075:2024: a <simple linear query statement> is a sequence of
    // <simple query statement>s (<match statement>, <let statement>, <for
    // statement>, <filter statement>, <order by and page statement>, <call
    // query statement>) in any order, and a <linear data-modifying statement>
    // mixes them with <simple data-modifying statement>s (<insert statement>,
    // <set statement>, <remove statement>, <delete statement>). Grafeo's WITH
    // and a WHERE between statements (a filter) are statements of the same
    // sequence.

    /// The clauses of `query` in source order, by the name of their
    /// `QueryClause` variant (`OPTIONAL MATCH` for an optional match).
    fn clause_names(query: &str) -> Vec<String> {
        let Statement::Query(statement) = Parser::new(query)
            .parse()
            .unwrap_or_else(|error| panic!("`{query}` must parse: {error}"))
        else {
            panic!("expected a query: {query}");
        };
        statement
            .ordered_clauses
            .iter()
            .map(|clause| match clause {
                QueryClause::Match(clause) if clause.optional => "OPTIONAL MATCH".to_string(),
                other => {
                    let debug = format!("{other:?}");
                    let end = debug
                        .find(|c: char| !c.is_ascii_alphanumeric())
                        .unwrap_or(debug.len());
                    debug[..end].to_string()
                }
            })
            .collect()
    }

    /// <order by and page statement> between other statements: after a
    /// MATCH, after a WITH, before a MATCH, a write or another page statement.
    #[test]
    fn test_parse_order_by_and_page_statement_between_statements() {
        for (query, expected) in [
            (
                "MATCH (p:Person) ORDER BY p.id LIMIT 2 RETURN p.id",
                &["Match", "OrderByAndPage"][..],
            ),
            (
                "MATCH (p:Person) WITH p ORDER BY p.id LIMIT 2 RETURN p.id",
                &["Match", "With", "OrderByAndPage"],
            ),
            (
                "MATCH (p:Person) WITH p LIMIT 1 RETURN p.id",
                &["Match", "With", "OrderByAndPage"],
            ),
            (
                "MATCH (p) ORDER BY p.id DESC OFFSET 1 LIMIT 2 MATCH (p)-[:R]->(q) RETURN q",
                &["Match", "OrderByAndPage", "Match"],
            ),
            ("MATCH (p) SKIP 3 RETURN p", &["Match", "OrderByAndPage"]),
            (
                "MATCH (p) OFFSET 3 LIMIT 19 ORDER BY p.name LIMIT 1 RETURN p",
                &["Match", "OrderByAndPage", "OrderByAndPage"],
            ),
            (
                "MATCH (p) WITH p ORDER BY p.age LIMIT 3 SET p.top = true",
                &["Match", "With", "OrderByAndPage", "Set"],
            ),
            (
                "MATCH (p) WITH p ORDER BY p.age LIMIT 3 WHERE p.age > 19 RETURN p",
                &["Match", "With", "OrderByAndPage", "Filter"],
            ),
        ] {
            assert_eq!(clause_names(query), expected, "{query}");
        }
    }

    /// The page statement holds its own keys and counts; the RETURN after it
    /// keeps its own ORDER BY and LIMIT.
    #[test]
    fn test_parse_order_by_and_page_statement_parts() {
        let Statement::Query(query) = Parser::new(
            "MATCH (p) ORDER BY p.age DESC, p.name OFFSET 3 LIMIT 19 RETURN p.name ORDER BY p.name LIMIT 1",
        )
        .parse()
        .unwrap() else {
            panic!("expected a query");
        };
        let pages: Vec<&OrderByAndPage> = query
            .ordered_clauses
            .iter()
            .filter_map(|clause| match clause {
                QueryClause::OrderByAndPage(page) => Some(page),
                _ => None,
            })
            .collect();
        let [page] = pages.as_slice() else {
            panic!("expected one page statement: {pages:?}");
        };
        let keys = &page.order_by.as_ref().expect("the ORDER BY").items;
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].order, SortOrder::Desc);
        assert_eq!(keys[1].order, SortOrder::Asc);
        assert!(matches!(
            page.offset,
            Some(Expression::Literal(Literal::Integer(3)))
        ));
        assert!(matches!(
            page.limit,
            Some(Expression::Literal(Literal::Integer(19)))
        ));
        let result = &query.return_clause;
        assert_eq!(result.order_by.as_ref().map(|o| o.items.len()), Some(1));
        assert!(matches!(
            result.limit,
            Some(Expression::Literal(Literal::Integer(1)))
        ));
        assert!(result.skip.is_none());

        // Each part is optional, but one of them must be there.
        for (query, order_by, offset, limit) in [
            ("MATCH (p) LIMIT 3 RETURN p", false, false, true),
            ("MATCH (p) OFFSET 3 RETURN p", false, true, false),
            ("MATCH (p) ORDER BY p.age RETURN p", true, false, false),
            (
                "MATCH (p) ORDER BY p.age SKIP 3 RETURN p",
                true,
                true,
                false,
            ),
        ] {
            let Statement::Query(statement) = Parser::new(query).parse().unwrap() else {
                panic!("expected a query: {query}");
            };
            let Some(QueryClause::OrderByAndPage(page)) = statement.ordered_clauses.last() else {
                panic!("expected a page statement last: {query}");
            };
            assert_eq!(
                (
                    page.order_by.is_some(),
                    page.offset.is_some(),
                    page.limit.is_some()
                ),
                (order_by, offset, limit),
                "{query}"
            );
        }
    }

    /// A WHERE or FILTER between MATCH statements is a filter of the rows so
    /// far, kept in its place among the clauses (no longer one WHERE for the
    /// whole statement).
    #[test]
    fn test_parse_where_and_filter_between_statements() {
        for (query, expected) in [
            (
                "MATCH (a:Person) WHERE a.age > 30 MATCH (c:City) RETURN a, c",
                &["Match", "Filter", "Match"][..],
            ),
            (
                "MATCH (a) FILTER a.age > 30 OPTIONAL MATCH (a)-[:R]->(b) RETURN a, b",
                &["Match", "Filter", "OPTIONAL MATCH"],
            ),
            (
                "MATCH (a) WITH a MATCH (b) WHERE a.age < b.age RETURN a, b",
                &["Match", "With", "Match", "Filter"],
            ),
            (
                "MATCH (a) WHERE a.age > 30 CALL { WITH a RETURN a.name AS m } RETURN m",
                &["Match", "Filter", "InlineCall"],
            ),
            (
                "MATCH (s) WHERE s.grade IN ['C'] LET x = s.grade RETURN x",
                &["Match", "Filter", "Let"],
            ),
            (
                "MATCH (s) FILTER s.grade IN ['C'] LET x = s.grade RETURN x",
                &["Match", "Filter", "Let"],
            ),
            (
                "MATCH (n) WHERE n.id IN [1] MATCH (n)-[*1..2]-(m) RETURN m.id",
                &["Match", "Filter", "Match"],
            ),
            (
                "MATCH (f) MATCH (t) WHERE t.path = f.path MATCH (x) RETURN x",
                &["Match", "Match", "Filter", "Match"],
            ),
            (
                "MATCH (a) WHERE a.x = 3 FILTER WHERE a.y = 19 RETURN a",
                &["Match", "Filter", "Filter"],
            ),
        ] {
            assert_eq!(clause_names(query), expected, "{query}");
            let Statement::Query(statement) = Parser::new(query).parse().unwrap() else {
                unreachable!("checked above");
            };
            assert!(
                statement.where_clause.is_none(),
                "{query}: each WHERE is among the ordered clauses"
            );
        }
    }

    /// <linear data-modifying statement>: data-modifying statements after a
    /// WITH, a SET, a LET or a FOR, and query statements after them.
    #[test]
    fn test_parse_data_modifying_statements_in_any_order() {
        for (query, expected) in [
            (
                "MATCH (n) WITH n SET n.x = 3 RETURN n",
                &["Match", "With", "Set"][..],
            ),
            (
                "MATCH (n) WITH n DETACH DELETE n",
                &["Match", "With", "Delete"],
            ),
            (
                "MATCH (n) WITH n REMOVE n.x RETURN n",
                &["Match", "With", "Remove"],
            ),
            (
                "MATCH (n) WITH n INSERT (n)-[:R]->(:T) RETURN n",
                &["Match", "With", "Create"],
            ),
            (
                "MATCH (n) WITH n CALL { WITH n RETURN n.x AS a } RETURN a",
                &["Match", "With", "InlineCall"],
            ),
            (
                "MATCH (n) WITH n OPTIONAL CALL (n) { MATCH (n)-[]->(m) RETURN m } RETURN m",
                &["Match", "With", "InlineCall"],
            ),
            (
                "MATCH (n) SET n.x = 3 CALL { WITH n RETURN n.x AS a } RETURN a",
                &["Match", "Set", "InlineCall"],
            ),
            (
                "MATCH (n) SET n.x = 3 DELETE n",
                &["Match", "Set", "Delete"],
            ),
            (
                "MATCH (n) SET n.x = 3 FILTER n.y > 3 RETURN n",
                &["Match", "Set", "Filter"],
            ),
            (
                "MATCH (n) REMOVE n.x SET n.y = 3 RETURN n",
                &["Match", "Remove", "Set"],
            ),
            (
                "MATCH (n) SET n.y = 3 REMOVE n.x SET n.z = 19",
                &["Match", "Set", "Remove", "Set"],
            ),
            ("MATCH (n) LET y = 3 DELETE n", &["Match", "Let", "Delete"]),
            (
                "MATCH (n) WITH n FOR x IN [1] DELETE n",
                &["Match", "With", "For", "Delete"],
            ),
            (
                "MATCH (n) INSERT (m:T) WITH n, m FOR x IN [3] MATCH (k) INSERT (k)-[:R]->(m)",
                &["Match", "Create", "With", "For", "Match", "Create"],
            ),
            (
                "MATCH (n) DELETE n WITH count(*) AS c RETURN c",
                &["Match", "Delete", "With"],
            ),
            (
                "MATCH (n) SET n.x = 3 MERGE (m:T) RETURN m",
                &["Match", "Set", "Merge"],
            ),
        ] {
            assert_eq!(clause_names(query), expected, "{query}");
        }
    }

    /// A statement may start with any simple query statement: LET, FILTER,
    /// FOR, an order by and page statement, a CALL, or a MATCH; also after
    /// NEXT.
    #[test]
    fn test_parse_statement_starting_with_let_filter_or_page() {
        for (query, expected) in [
            ("LET x = 3 RETURN x", &["Let"][..]),
            ("FILTER 3 > 1 RETURN 19", &["Filter"]),
            ("FILTER WHERE 3 > 1 RETURN 19", &["Filter"]),
            ("LET x = 3 FILTER x > 1 RETURN x", &["Let", "Filter"]),
            (
                "LET who = 'Alix' MATCH (p {name: who}) RETURN p",
                &["Let", "Match"],
            ),
            ("LIMIT 1 RETURN 3", &["OrderByAndPage"]),
            ("ORDER BY 3 RETURN 3", &["OrderByAndPage"]),
            ("OFFSET 0 MATCH (n) RETURN n", &["OrderByAndPage", "Match"]),
        ] {
            assert_eq!(clause_names(query), expected, "{query}");
        }
        let Statement::CompositeQuery { right, .. } =
            Parser::new("RETURN 3 AS x NEXT LET y = x + 19 RETURN y")
                .parse()
                .unwrap()
        else {
            panic!("expected NEXT");
        };
        let Statement::Query(right) = *right else {
            panic!("expected a query after NEXT");
        };
        assert!(matches!(
            right.ordered_clauses.as_slice(),
            [QueryClause::Let(_)]
        ));
    }

    /// A filter among the clauses keeps the keyword it was written with: a
    /// FILTER filters every row, also right after an OPTIONAL MATCH, whose
    /// WHERE belongs to its graph pattern.
    #[test]
    fn test_parse_filter_clause_keeps_its_keyword() {
        let Statement::Query(statement) = Parser::new(
            "MATCH (a) OPTIONAL MATCH (a)-[:R]->(b) WHERE b.x = 3 FILTER b.y = 19 \
             FILTER WHERE b.z = 88 RETURN a",
        )
        .parse()
        .unwrap() else {
            panic!("expected a query");
        };
        let keywords: Vec<bool> = statement
            .ordered_clauses
            .iter()
            .filter_map(|clause| match clause {
                QueryClause::Filter(filter) => Some(filter.filter),
                _ => None,
            })
            .collect();
        assert_eq!(keywords, [false, true, true], "WHERE, FILTER, FILTER WHERE");
    }

    /// A WITH may follow a FOR: the FOR takes `WITH` only for `WITH
    /// ORDINALITY` or `WITH OFFSET`.
    #[test]
    fn test_parse_with_after_for() {
        assert_eq!(
            clause_names("FOR x IN [3, 19] WITH x RETURN x"),
            ["For", "With"]
        );
        assert_eq!(
            clause_names("FOR x IN [3, 19] WITH ORDINALITY i WITH x, i RETURN x"),
            ["For", "With"]
        );
        let Statement::Query(statement) = Parser::new("FOR x IN [3, 19] WITH OFFSET i RETURN x, i")
            .parse()
            .unwrap()
        else {
            panic!("expected a query");
        };
        let [QueryClause::For(for_clause)] = statement.ordered_clauses.as_slice() else {
            panic!("expected one FOR: {:?}", statement.ordered_clauses);
        };
        assert_eq!(for_clause.offset_var.as_deref(), Some("i"));
    }

    /// What the standard does not allow still fails: a query that only reads
    /// needs a result statement, nothing follows the result statement, and a
    /// page statement needs its count.
    #[test]
    fn test_parse_statement_order_errors() {
        for (query, message) in [
            (
                "MATCH (n) WHERE n.x = 3",
                "Expected RETURN, FINISH, or SELECT",
            ),
            (
                "MATCH (n) ORDER BY n.x LIMIT 3",
                "Expected RETURN, FINISH, or SELECT",
            ),
            (
                "MATCH (n) RETURN n MATCH (m) RETURN m",
                "unexpected 'MATCH' after the end of the statement",
            ),
            (
                "MATCH (n) RETURN n WHERE n.x = 3",
                "unexpected 'WHERE' after the end of the statement",
            ),
            ("MATCH (n) LIMIT RETURN n", ""),
            ("MATCH (n) ORDER n.x RETURN n", ""),
        ] {
            let error = parse_err(query);
            assert!(error.contains(message), "`{query}`: {error}");
        }
    }

    /// A WHERE or FILTER right after INSERT, CREATE, MERGE or DELETE would
    /// filter the rows after the write; it used to filter the rows before it,
    /// so it is an error instead of a silent change of what the write
    /// touches. After SET or REMOVE a FILTER filters the rows after the
    /// write; a WHERE does not stand there. A WHERE on a later MATCH, or one
    /// of a WITH, filters the rows after the write.
    #[test]
    fn test_parse_filter_after_a_write() {
        for query in [
            "MATCH (n) DETACH DELETE n WHERE n.age > 30",
            "MATCH (n) DETACH DELETE n FILTER n.age > 30",
            "MATCH (a) INSERT (a)-[:R]->(:T) WHERE a.age > 30 RETURN a",
            "MATCH (a) INSERT (a)-[:R]->(:T) FILTER a.age > 30 RETURN a",
            "MATCH (a) CREATE (a)-[:R]->(:T) WHERE a.age > 30",
            "MATCH (a) MERGE (c:City {name: 'Paris'}) FILTER a.age > 30 RETURN a",
        ] {
            let error = parse_err(query);
            assert!(
                error.contains("a WHERE or FILTER cannot follow INSERT, CREATE, MERGE or DELETE"),
                "`{query}`: {error}"
            );
        }
        for query in [
            "MATCH (a) SET a.x = 3 WHERE a.y = 19 RETURN a",
            "MATCH (a) REMOVE a.x WHERE a.y = 19 RETURN a",
        ] {
            let error = parse_err(query);
            assert!(
                error.contains("a WHERE cannot follow SET or REMOVE"),
                "`{query}`: {error}"
            );
        }
        for (query, expected) in [
            (
                "MATCH (a) SET a.x = 3 FILTER a.y = 19 RETURN a",
                &["Match", "Set", "Filter"][..],
            ),
            (
                "MATCH (a) INSERT (a)-[:R]->(t:T) WITH a, t WHERE a.x > 3 RETURN t",
                &["Match", "Create", "With"],
            ),
            (
                "MATCH (a) INSERT (:T) MATCH (b:T) WHERE a.x > 3 RETURN b",
                &["Match", "Create", "Match", "Filter"],
            ),
        ] {
            assert_eq!(clause_names(query), expected, "{query}");
        }
    }
}
