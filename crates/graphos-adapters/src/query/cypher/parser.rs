//! Cypher Parser.
//!
//! Parses Cypher queries into an AST.

use super::ast::*;
use super::lexer::{Lexer, Token, TokenKind};
use graphos_common::utils::error::{QueryError, QueryErrorKind, Result};

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
            span: current.span.clone(),
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
        if self.current.kind == TokenKind::Identifier && self.peek_kind() == TokenKind::Eq {
            let name = self.expect_identifier()?;
            self.expect(TokenKind::Eq)?;
            let pattern = self.parse_pattern()?;
            return Ok(Pattern::NamedPath {
                name,
                pattern: Box::new(pattern),
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

    fn parse_node_pattern(&mut self) -> Result<NodePattern> {
        self.expect(TokenKind::LParen)?;

        let variable = if self.current.kind == TokenKind::Identifier {
            let name = self.current.text.clone();
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
        let (variable, types, length, properties, final_direction) = if has_bracket
            || self.current.kind == TokenKind::LBracket
        {
            if self.current.kind == TokenKind::LBracket {
                self.advance();
            }

            let var = if self.current.kind == TokenKind::Identifier
                && self.peek_kind() != TokenKind::Colon
            {
                let name = self.current.text.clone();
                self.advance();
                Some(name)
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
            TokenKind::Identifier => {
                let name = self.current.text.clone();
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
            _ => Err(self.error("Expected expression")),
        }
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
        match self.current.kind {
            TokenKind::Identifier | TokenKind::QuotedIdentifier => {
                let mut text = self.current.text.clone();
                // Remove backticks from quoted identifier
                if self.current.kind == TokenKind::QuotedIdentifier {
                    text = text[1..text.len() - 1].to_string();
                }
                self.advance();
                Ok(text)
            }
            _ => Err(self.error("Expected identifier")),
        }
    }

    fn peek_kind(&mut self) -> TokenKind {
        // Lookahead - we need to save and restore state
        let saved_pos = self.lexer.clone();
        let token = self.lexer.next_token();
        let kind = token.kind;
        self.lexer = saved_pos;
        kind
    }

    fn error(&self, message: &str) -> graphos_common::utils::error::Error {
        QueryError::new(QueryErrorKind::Syntax, message)
            .with_span(self.current.span.clone())
            .into()
    }
}
