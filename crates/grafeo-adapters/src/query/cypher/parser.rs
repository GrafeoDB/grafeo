//! Cypher Parser.
//!
//! Parses Cypher queries into an AST.

use super::ast::*;
use super::lexer::{Lexer, Token, TokenKind};
use grafeo_common::utils::error::{QueryError, QueryErrorKind, Result};

/// Cypher query parser.
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
    previous: Token,
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
        }
    }

    /// Parses the query into a statement.
    pub fn parse(&mut self) -> Result<Statement> {
        let stmt = self.parse_statement()?;
        if self.current.kind != TokenKind::Eof {
            return Err(self.error("Expected end of query"));
        }
        Ok(stmt)
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
                _ => break,
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

        let items = self.parse_projection_items()?;

        let where_clause = if self.current.kind == TokenKind::Where {
            Some(Box::new(self.parse_where_clause()?))
        } else {
            None
        };

        Ok(WithClause {
            distinct,
            items,
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
            TokenKind::Arrow | TokenKind::LeftArrow | TokenKind::Minus
        ) {
            let mut chain = Vec::new();
            while matches!(
                self.current.kind,
                TokenKind::Arrow | TokenKind::LeftArrow | TokenKind::Minus
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
            TokenKind::Arrow | TokenKind::LeftArrow | TokenKind::Minus
        ) {
            let mut chain = Vec::new();
            while matches!(
                self.current.kind,
                TokenKind::Arrow | TokenKind::LeftArrow | TokenKind::Minus
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
                // <-
                self.advance();
                (Direction::Incoming, false)
            }
            TokenKind::Minus => {
                // - followed by [ or - or >
                self.advance();

                if self.current.kind == TokenKind::LBracket {
                    // -[...]- or -[...]->
                    (Direction::Undirected, true) // Direction will be updated based on closing
                } else if self.current.kind == TokenKind::Gt {
                    // ->
                    self.advance();
                    (Direction::Outgoing, false)
                } else if self.current.kind == TokenKind::Minus {
                    // --
                    self.advance();
                    (Direction::Undirected, false)
                } else {
                    return Err(self.error("Expected relationship pattern"));
                }
            }
            _ => return Err(self.error("Expected relationship pattern")),
        };

        // Parse relationship details [r:TYPE*1..3 {props}]
        let (variable, types, length, properties, final_direction) =
            if has_bracket || self.current.kind == TokenKind::LBracket {
                if self.current.kind == TokenKind::LBracket {
                    self.advance();
                }

                // Parse optional variable name - could be followed by : for type
                // Allow contextual keywords like 'end' to be used as variable names
                let var = if self.can_be_identifier() {
                    // Check if this is a variable (followed by : or ] or { or *)
                    let is_variable = self.peek_kind() == TokenKind::Colon
                        || self.peek_kind() == TokenKind::RBracket
                        || self.peek_kind() == TokenKind::LBrace
                        || self.peek_kind() == TokenKind::Star;
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

                (var, rel_types, len, props, dir)
            } else {
                (None, Vec::new(), None, Vec::new(), direction)
            };

        let target = self.parse_node_pattern()?;

        Ok(RelationshipPattern {
            variable,
            types,
            direction: final_direction,
            length,
            properties,
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
                    let index = self.parse_expression()?;
                    self.expect(TokenKind::RBracket)?;
                    expr = Expression::IndexAccess {
                        base: Box::new(expr),
                        index: Box::new(index),
                    };
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
                let value = self.current.text.parse().unwrap_or(0);
                self.advance();
                Ok(Expression::Literal(Literal::Integer(value)))
            }
            TokenKind::Float => {
                let value = self.current.text.parse().unwrap_or(0.0);
                self.advance();
                Ok(Expression::Literal(Literal::Float(value)))
            }
            TokenKind::String => {
                // Remove quotes
                let text = &self.current.text;
                let value = text[1..text.len() - 1].to_string();
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
                self.advance();

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
                        distinct,
                        args,
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
                // List literal
                self.advance();
                let mut items = Vec::new();
                if self.current.kind != TokenKind::RBracket {
                    items.push(self.parse_expression()?);
                    while self.current.kind == TokenKind::Comma {
                        self.advance();
                        items.push(self.parse_expression()?);
                    }
                }
                self.expect(TokenKind::RBracket)?;
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
            // Aggregate functions (COUNT, SUM, AVG, MIN, MAX, COLLECT)
            TokenKind::Count => {
                self.advance();
                self.parse_aggregate_function("count")
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

    fn error(&self, message: &str) -> grafeo_common::utils::error::Error {
        QueryError::new(QueryErrorKind::Syntax, message)
            .with_span(self.current.span)
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
        let stmt = parse_ok("MATCH (n:Person {name: 'Alice', age: 30}) RETURN n");
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
        let stmt = parse_ok("MATCH (n) WHERE n.age > 30 AND n.name = 'Alice' RETURN n");
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
        let stmt = parse_ok("CREATE (n:Person {name: 'Alice'})");
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
        let stmt = parse_ok("MERGE (n:Person {name: 'Alice'})");
        if let Statement::Query(Query { clauses, .. }) = stmt {
            assert!(matches!(&clauses[0], Clause::Merge(_)));
        }
    }

    #[test]
    fn test_parse_merge_on_create() {
        let stmt =
            parse_ok("MERGE (n:Person {name: 'Alice'}) ON CREATE SET n.created = timestamp()");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Merge(MergeClause { on_create, .. }) = &clauses[0]
        {
            assert!(on_create.is_some());
        }
    }

    #[test]
    fn test_parse_merge_on_match() {
        let stmt = parse_ok("MERGE (n:Person {name: 'Alice'}) ON MATCH SET n.seen = true");
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
        let stmt = parse_ok("MATCH (n) SET n.name = 'Bob'");
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
        let stmt = parse_ok("MATCH (n) SET n = {name: 'Alice', age: 30}");
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
        let stmt = parse_ok("RETURN 3.14");
        if let Statement::Query(Query { clauses, .. }) = stmt
            && let Clause::Return(ReturnClause {
                items: ReturnItems::Explicit(items),
                ..
            }) = &clauses[0]
            && let Expression::Literal(Literal::Float(val)) = &items[0].expression
        {
            assert!((val - 3.14).abs() < 0.001);
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
        let stmt = parse_ok("RETURN {name: 'Alice', age: 30}");
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
}
