//! GQL Parser.

use super::ast::*;
use super::lexer::{Lexer, Token, TokenKind};
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result, SourceSpan};

/// GQL Parser.
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
    peeked: Option<Token>,
    source: &'a str,
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
            source: input,
        }
    }

    /// Checks if the current token can be used as a label or type name.
    /// This includes identifiers, quoted identifiers, and certain reserved keywords that are
    /// commonly used as labels (Node, Edge, Type, etc.)
    fn is_label_or_type_name(&self) -> bool {
        matches!(
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
                | TokenKind::Case
                | TokenKind::When
                | TokenKind::Then
                | TokenKind::Else
                | TokenKind::End
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
                | TokenKind::False // FALSE literal
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

    /// Parses the input into a statement.
    pub fn parse(&mut self) -> Result<Statement> {
        match self.current.kind {
            TokenKind::Match | TokenKind::Optional | TokenKind::Unwind | TokenKind::Merge => {
                self.parse_query().map(Statement::Query)
            }
            TokenKind::Insert => self
                .parse_insert()
                .map(|s| Statement::DataModification(DataModificationStatement::Insert(s))),
            TokenKind::Delete => self
                .parse_delete()
                .map(|s| Statement::DataModification(DataModificationStatement::Delete(s))),
            TokenKind::Create => {
                // Check if CREATE is followed by a pattern (Cypher-style) or NODE/EDGE (GQL schema)
                let next = self.peek_kind();
                if next == TokenKind::LParen {
                    // Cypher-style: CREATE (n:Label {...}) - treat as INSERT
                    self.parse_create_as_insert()
                        .map(|s| Statement::DataModification(DataModificationStatement::Insert(s)))
                } else {
                    // GQL schema: CREATE NODE TYPE / CREATE EDGE TYPE
                    self.parse_create_schema().map(Statement::Schema)
                }
            }
            _ => Err(self.error("Expected MATCH, INSERT, DELETE, MERGE, UNWIND, or CREATE")),
        }
    }

    fn parse_query(&mut self) -> Result<QueryStatement> {
        let span_start = self.current.span.start;

        // Parse MATCH clauses (including OPTIONAL MATCH)
        let mut match_clauses = Vec::new();
        let mut unwind_clauses = Vec::new();
        let mut merge_clauses = Vec::new();
        let mut create_clauses = Vec::new();
        let mut delete_clauses = Vec::new();

        // Parse initial clauses (MATCH, OPTIONAL MATCH, UNWIND, MERGE)
        loop {
            match self.current.kind {
                TokenKind::Match | TokenKind::Optional => {
                    match_clauses.push(self.parse_match_clause()?);
                }
                TokenKind::Unwind => {
                    unwind_clauses.push(self.parse_unwind_clause()?);
                }
                TokenKind::Merge => {
                    merge_clauses.push(self.parse_merge_clause()?);
                }
                _ => break,
            }
        }

        // Parse WHERE clause (after all MATCH clauses)
        let where_clause = if self.current.kind == TokenKind::Where {
            Some(self.parse_where_clause()?)
        } else {
            None
        };

        // Parse SET clauses
        let mut set_clauses = Vec::new();
        while self.current.kind == TokenKind::Set {
            set_clauses.push(self.parse_set_clause()?);
        }

        // Parse REMOVE clauses
        let mut remove_clauses = Vec::new();
        while self.current.kind == TokenKind::Remove {
            remove_clauses.push(self.parse_remove_clause()?);
        }

        // Parse CREATE clauses (Cypher-style: MATCH ... CREATE ...)
        while self.current.kind == TokenKind::Create {
            create_clauses.push(self.parse_create_clause_in_query()?);
        }

        // Parse DELETE clauses (Cypher-style: MATCH ... DELETE ...)
        while self.current.kind == TokenKind::Delete || self.current.kind == TokenKind::Detach {
            delete_clauses.push(self.parse_delete_clause_in_query()?);
        }

        // Parse WITH clauses
        let mut with_clauses = Vec::new();
        while self.current.kind == TokenKind::With {
            with_clauses.push(self.parse_with_clause()?);

            // After WITH, we can have more MATCH/UNWIND/MERGE clauses
            loop {
                match self.current.kind {
                    TokenKind::Match | TokenKind::Optional => {
                        match_clauses.push(self.parse_match_clause()?);
                    }
                    TokenKind::Unwind => {
                        unwind_clauses.push(self.parse_unwind_clause()?);
                    }
                    TokenKind::Merge => {
                        merge_clauses.push(self.parse_merge_clause()?);
                    }
                    _ => break,
                }
            }
        }

        // Parse RETURN clause (optional if we have SET, REMOVE, MERGE, CREATE, or DELETE clauses)
        let return_clause = if self.current.kind == TokenKind::Return {
            self.parse_return_clause()?
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
                order_by: None,
                skip: None,
                limit: None,
                span: None,
            }
        } else {
            return Err(self.error("Expected RETURN"));
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
            span: Some(SourceSpan::new(span_start, self.current.span.end, 1, 1)),
        })
    }

    fn parse_set_clause(&mut self) -> Result<SetClause> {
        let span_start = self.current.span.start;
        self.expect(TokenKind::Set)?;

        let mut assignments = Vec::new();
        let mut label_operations = Vec::new();

        loop {
            // Parse variable name
            if !self.is_identifier() {
                return Err(self.error("Expected variable name in SET"));
            }
            let variable = self.current.text.clone();
            self.advance();

            // Check if this is a label operation (n:Label) or property assignment (n.prop = value)
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
            } else {
                return Err(self.error("Expected '.' or ':' after variable in SET"));
            }

            // Check for more assignments/operations
            if self.current.kind != TokenKind::Comma {
                break;
            }
            self.advance();
        }

        Ok(SetClause {
            assignments,
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
            span: Some(SourceSpan::new(span_start, self.current.span.end, 1, 1)),
        })
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

        let mut patterns = Vec::new();
        patterns.push(self.parse_aliased_pattern()?);

        while self.current.kind == TokenKind::Comma {
            self.advance();
            patterns.push(self.parse_aliased_pattern()?);
        }

        Ok(MatchClause {
            optional,
            patterns,
            span: Some(SourceSpan::new(span_start, self.current.span.end, 1, 1)),
        })
    }

    /// Parses a pattern with optional alias and path function.
    /// Supports: `p = shortestPath((a)-[*]-(b))` and `p = (a)-[*]-(b)` and `(a)-[*]-(b)`
    fn parse_aliased_pattern(&mut self) -> Result<AliasedPattern> {
        let mut alias = None;
        let mut path_function = None;

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

        let pattern = self.parse_pattern()?;

        if path_function.is_some() {
            self.expect(TokenKind::RParen)?;
        }

        Ok(AliasedPattern {
            alias,
            path_function,
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

        let mut items = Vec::new();
        items.push(self.parse_return_item()?);

        while self.current.kind == TokenKind::Comma {
            self.advance();
            items.push(self.parse_return_item()?);
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
            where_clause,
            span: Some(SourceSpan::new(span_start, self.current.span.end, 1, 1)),
        })
    }

    fn parse_pattern(&mut self) -> Result<Pattern> {
        let node = self.parse_node_pattern()?;

        // Check for path continuation
        // Handle both `-[...]->`/`<-[...]-` style and `->` style
        if matches!(
            self.current.kind,
            TokenKind::Arrow | TokenKind::LeftArrow | TokenKind::DoubleDash | TokenKind::Minus
        ) {
            let mut edges = Vec::new();

            while matches!(
                self.current.kind,
                TokenKind::Arrow | TokenKind::LeftArrow | TokenKind::DoubleDash | TokenKind::Minus
            ) {
                edges.push(self.parse_edge_pattern()?);
            }

            Ok(Pattern::Path(PathPattern {
                source: node,
                edges,
                span: None,
            }))
        } else {
            Ok(Pattern::Node(node))
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
        while self.current.kind == TokenKind::Colon {
            self.advance();
            if !self.is_label_or_type_name() {
                return Err(self.error("Expected label name"));
            }
            labels.push(self.get_identifier_name());
            self.advance();
        }

        // Parse properties { key: value, ... }
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

    fn parse_edge_pattern(&mut self) -> Result<EdgePattern> {
        // Handle both styles:
        // 1. `-[...]->` or `-[:TYPE]->` or `-[:TYPE*1..3]->` (direction determined by trailing arrow)
        // 2. `->` or `<-` or `--` (direction determined by leading arrow)

        let (variable, types, min_hops, max_hops, properties, direction) =
            if self.current.kind == TokenKind::Minus {
                // Pattern: -[...]->(target) or -[...]-(target)
                self.advance();

                // Parse [variable:TYPE*min..max {props}]
                let (var, edge_types, min_h, max_h, props) =
                    if self.current.kind == TokenKind::LBracket {
                        self.advance();

                        // Parse variable name if present
                        // Variable is followed by : (type), * (quantifier), { (properties), or ] (end)
                        let v = if self.is_identifier() {
                            let peek = self.peek_kind();
                            if matches!(
                                peek,
                                TokenKind::Colon
                                    | TokenKind::Star
                                    | TokenKind::LBrace
                                    | TokenKind::RBracket
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
                        }

                        // Parse variable-length path quantifier: *min..max
                        let (min_h, max_h) = self.parse_path_quantifier()?;

                        // Parse edge properties: {key: value, ...}
                        let edge_props = if self.current.kind == TokenKind::LBrace {
                            self.parse_property_map()?
                        } else {
                            Vec::new()
                        };

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

                let (var, edge_types, min_h, max_h, props) =
                    if self.current.kind == TokenKind::LBracket {
                        self.advance();

                        // Parse variable name if present
                        // Variable is followed by : (type), * (quantifier), { (properties), or ] (end)
                        let v = if self.is_identifier() {
                            let peek = self.peek_kind();
                            if matches!(
                                peek,
                                TokenKind::Colon
                                    | TokenKind::Star
                                    | TokenKind::LBrace
                                    | TokenKind::RBracket
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
                        }

                        // Parse variable-length path quantifier
                        let (min_h, max_h) = self.parse_path_quantifier()?;

                        // Parse edge properties: {key: value, ...}
                        let edge_props = if self.current.kind == TokenKind::LBrace {
                            self.parse_property_map()?
                        } else {
                            Vec::new()
                        };

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
                // Simple --
                self.advance();
                (
                    None,
                    Vec::new(),
                    None,
                    None,
                    Vec::new(),
                    EdgeDirection::Undirected,
                )
            } else {
                return Err(self.error("Expected edge pattern"));
            };

        let target = self.parse_node_pattern()?;

        Ok(EdgePattern {
            variable,
            types,
            direction,
            target,
            min_hops,
            max_hops,
            properties,
            span: None,
        })
    }

    /// Parses a path quantifier like `*`, `*2`, `*1..3`, `*..5`, `*2..`.
    /// Returns (min_hops, max_hops) where None means no quantifier was present.
    fn parse_path_quantifier(&mut self) -> Result<(Option<u32>, Option<u32>)> {
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

            if self.current.kind == TokenKind::Dot {
                self.advance();
                self.expect(TokenKind::Dot)?; // expect second dot for ..

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
        } else if self.current.kind == TokenKind::Dot {
            self.advance();
            self.expect(TokenKind::Dot)?; // expect second dot for ..

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

        let mut items = Vec::new();
        items.push(self.parse_return_item()?);

        while self.current.kind == TokenKind::Comma {
            self.advance();
            items.push(self.parse_return_item()?);
        }

        let order_by = if self.current.kind == TokenKind::Order {
            Some(self.parse_order_by()?)
        } else {
            None
        };

        let skip = if self.current.kind == TokenKind::Skip {
            self.advance();
            Some(self.parse_expression()?)
        } else {
            None
        };

        let limit = if self.current.kind == TokenKind::Limit {
            self.advance();
            Some(self.parse_expression()?)
        } else {
            None
        };

        Ok(ReturnClause {
            distinct,
            items,
            order_by,
            skip,
            limit,
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

        Ok(OrderByItem { expression, order })
    }

    fn parse_expression(&mut self) -> Result<Expression> {
        self.parse_or_expression()
    }

    fn parse_or_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_and_expression()?;

        while self.current.kind == TokenKind::Or {
            self.advance();
            let right = self.parse_and_expression()?;
            left = Expression::Binary {
                left: Box::new(left),
                op: BinaryOp::Or,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_and_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_comparison_expression()?;

        while self.current.kind == TokenKind::And {
            self.advance();
            let right = self.parse_comparison_expression()?;
            left = Expression::Binary {
                left: Box::new(left),
                op: BinaryOp::And,
                right: Box::new(right),
            };
        }

        Ok(left)
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

        // Check for STARTS WITH, ENDS WITH, CONTAINS
        match self.current.kind {
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
        let mut left = self.parse_unary_expression()?;

        loop {
            let op = match self.current.kind {
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::Slash => BinaryOp::Div,
                TokenKind::Percent => BinaryOp::Mod,
                _ => break,
            };
            self.advance();
            let right = self.parse_unary_expression()?;
            left = Expression::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_unary_expression(&mut self) -> Result<Expression> {
        match self.current.kind {
            TokenKind::Not => {
                self.advance();
                let operand = self.parse_unary_expression()?;
                Ok(Expression::Unary {
                    op: UnaryOp::Not,
                    operand: Box::new(operand),
                })
            }
            TokenKind::Minus => {
                self.advance();
                let operand = self.parse_unary_expression()?;
                Ok(Expression::Unary {
                    op: UnaryOp::Neg,
                    operand: Box::new(operand),
                })
            }
            _ => self.parse_primary_expression(),
        }
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
                let value = self
                    .current
                    .text
                    .parse()
                    .map_err(|_| self.error("Invalid integer"))?;
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
                let value = text[1..text.len() - 1].to_string(); // Remove quotes
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
                self.advance();

                if self.current.kind == TokenKind::Dot {
                    self.advance();
                    if !self.is_identifier() {
                        return Err(self.error("Expected property name"));
                    }
                    let property = self.get_identifier_name();
                    self.advance();
                    Ok(Expression::PropertyAccess {
                        variable: name,
                        property,
                    })
                } else if self.current.kind == TokenKind::LParen {
                    // Function call
                    self.advance();
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
                self.advance();
                let expr = self.parse_expression()?;
                self.expect(TokenKind::RParen)?;
                Ok(expr)
            }
            TokenKind::LBracket => {
                self.advance();
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
            TokenKind::Parameter => {
                // Parameter token includes the $ prefix, so we extract just the name
                let full_text = &self.current.text;
                let name = full_text.trim_start_matches('$').to_string();
                self.advance();
                Ok(Expression::Parameter(name))
            }
            TokenKind::Exists => {
                self.advance();
                self.expect(TokenKind::LBrace)?;
                let inner_query = self.parse_exists_inner_query()?;
                self.expect(TokenKind::RBrace)?;
                Ok(Expression::ExistsSubquery {
                    query: Box::new(inner_query),
                })
            }
            _ => Err(self.error("Expected expression")),
        }
    }

    /// Parses a CASE expression.
    /// CASE [input] WHEN condition THEN result [WHEN ...] [ELSE default] END
    fn parse_case_expression(&mut self) -> Result<Expression> {
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
                order_by: None,
                skip: None,
                limit: None,
                span: None,
            },
            having_clause: None,
            span: None,
        })
    }

    fn parse_property_map(&mut self) -> Result<Vec<(String, Expression)>> {
        self.expect(TokenKind::LBrace)?;

        let mut properties = Vec::new();

        if self.current.kind != TokenKind::RBrace {
            loop {
                if !self.is_identifier() {
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

    /// Parses CREATE as INSERT (Cypher-style data modification).
    fn parse_create_as_insert(&mut self) -> Result<InsertStatement> {
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

    /// Parses DELETE clause within a query (e.g., MATCH ... DELETE ...).
    fn parse_delete_clause_in_query(&mut self) -> Result<DeleteStatement> {
        let detach = if self.current.kind == TokenKind::Detach {
            self.advance();
            true
        } else {
            false
        };

        self.expect(TokenKind::Delete)?;

        let mut variables = Vec::new();
        if !self.is_identifier() {
            return Err(self.error("Expected variable name in DELETE"));
        }
        variables.push(self.get_identifier_name());
        self.advance();

        while self.current.kind == TokenKind::Comma {
            self.advance();
            if !self.is_identifier() {
                return Err(self.error("Expected variable name in DELETE"));
            }
            variables.push(self.get_identifier_name());
            self.advance();
        }

        Ok(DeleteStatement {
            variables,
            detach,
            span: None,
        })
    }

    fn parse_delete(&mut self) -> Result<DeleteStatement> {
        let detach = if self.current.kind == TokenKind::Detach {
            self.advance();
            true
        } else {
            false
        };

        self.expect(TokenKind::Delete)?;

        let mut variables = Vec::new();
        if self.current.kind != TokenKind::Identifier {
            return Err(self.error("Expected variable name"));
        }
        variables.push(self.current.text.clone());
        self.advance();

        while self.current.kind == TokenKind::Comma {
            self.advance();
            if self.current.kind != TokenKind::Identifier {
                return Err(self.error("Expected variable name"));
            }
            variables.push(self.current.text.clone());
            self.advance();
        }

        Ok(DeleteStatement {
            variables,
            detach,
            span: None,
        })
    }

    fn parse_create_schema(&mut self) -> Result<SchemaStatement> {
        self.expect(TokenKind::Create)?;

        match self.current.kind {
            TokenKind::Node => {
                self.advance();
                self.expect(TokenKind::Type)?;

                if !self.is_identifier() {
                    return Err(self.error("Expected type name"));
                }
                let name = self.get_identifier_name();
                self.advance();

                // Parse property definitions
                let properties = if self.current.kind == TokenKind::LParen {
                    self.parse_property_definitions()?
                } else {
                    Vec::new()
                };

                Ok(SchemaStatement::CreateNodeType(CreateNodeTypeStatement {
                    name,
                    properties,
                    span: None,
                }))
            }
            TokenKind::Edge => {
                self.advance();
                self.expect(TokenKind::Type)?;

                if !self.is_identifier() {
                    return Err(self.error("Expected type name"));
                }
                let name = self.get_identifier_name();
                self.advance();

                let properties = if self.current.kind == TokenKind::LParen {
                    self.parse_property_definitions()?
                } else {
                    Vec::new()
                };

                Ok(SchemaStatement::CreateEdgeType(CreateEdgeTypeStatement {
                    name,
                    properties,
                    span: None,
                }))
            }
            _ => Err(self.error("Expected NODE or EDGE")),
        }
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

                defs.push(PropertyDefinition {
                    name,
                    data_type,
                    nullable,
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

    fn advance(&mut self) {
        if let Some(peeked) = self.peeked.take() {
            self.current = peeked;
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
        self.peeked.as_ref().unwrap().kind
    }

    fn error(&self, message: &str) -> Error {
        Error::Query(
            QueryError::new(QueryErrorKind::Syntax, message)
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
        let result = parser.parse();
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_match_with_where() {
        let mut parser = Parser::new("MATCH (n:Person) WHERE n.age > 30 RETURN n");
        let result = parser.parse();
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_path_pattern() {
        let mut parser = Parser::new("MATCH (a)-[:KNOWS]->(b) RETURN a, b");
        let result = parser.parse();
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_insert() {
        let mut parser = Parser::new("INSERT (n:Person {name: 'Alice'})");
        let result = parser.parse();
        assert!(result.is_ok());
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

            if let Statement::Query(q) = result.unwrap() {
                if let Pattern::Node(node) = &q.match_clauses[0].patterns[0].pattern {
                    assert_eq!(node.variable, Some(expected_var.to_string()));
                }
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
        let mut parser = Parser::new("MERGE (n:Person {name: 'Alice'}) RETURN n");
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
            Parser::new("MERGE (n:Person {name: 'Alice'}) ON CREATE SET n.created = true RETURN n");
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
}
