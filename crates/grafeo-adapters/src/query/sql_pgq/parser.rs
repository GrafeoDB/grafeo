//! SQL/PGQ Parser.
//!
//! Parses SQL:2023 GRAPH_TABLE queries into an AST. The inner MATCH clause
//! uses GQL pattern syntax, producing GQL AST types.

#[allow(clippy::wildcard_imports)]
use super::ast::*;
use super::lexer::{Lexer, Token, TokenKind};
use crate::query::keywords::unescape_string;
use crate::query::limits::{Chain, Nesting, nesting_error_message};
use grafeo_common::utils::error::{QueryError, QueryErrorKind, Result};

/// SQL/PGQ query parser.
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
    previous: Token,
    source: &'a str,
    /// How deep the query parsed so far nests (see
    /// [`MAX_NESTING_DEPTH`](crate::query::limits::MAX_NESTING_DEPTH)).
    nesting: Nesting,
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
            nesting: Nesting::default(),
        }
    }

    /// Parses the query into a statement.
    ///
    /// # Errors
    ///
    /// Returns an error if the input contains invalid or unexpected SQL/PGQ syntax.
    pub fn parse(&mut self) -> Result<Statement> {
        let stmt = match self.current.kind {
            TokenKind::Create => self.parse_create_statement()?,
            TokenKind::Call => self.parse_call_statement()?,
            _ => {
                let select = self.parse_select_statement()?;

                // Check for set operations: UNION, INTERSECT, EXCEPT
                if let Some(op_kind) = self.current_set_operation() {
                    self.advance(); // consume UNION/INTERSECT/EXCEPT

                    // Check for optional ALL
                    let all = if self.current.kind == TokenKind::All {
                        self.advance();
                        true
                    } else {
                        false
                    };

                    let right = self.parse_select_statement()?;

                    Statement::SetOperation(SetOperationStatement {
                        left: select,
                        operation: op_kind,
                        all,
                        right,
                        span: None,
                    })
                } else {
                    Statement::Select(select)
                }
            }
        };

        // Allow optional trailing semicolon
        if self.current.kind == TokenKind::Semicolon {
            self.advance();
        }

        if self.current.kind != TokenKind::Eof {
            return Err(self.error("Expected end of query"));
        }

        Ok(stmt)
    }

    /// Returns the set operation kind if the current token is a set operation keyword.
    fn current_set_operation(&self) -> Option<SetOperationKind> {
        match self.current.kind {
            TokenKind::Union => Some(SetOperationKind::Union),
            TokenKind::Intersect => Some(SetOperationKind::Intersect),
            TokenKind::Except => Some(SetOperationKind::Except),
            _ => None,
        }
    }

    // ==================== Statement Parsing ====================

    fn parse_select_statement(&mut self) -> Result<SelectStatement> {
        self.expect(TokenKind::Select)?;

        // Optional DISTINCT
        let distinct = if self.current.kind == TokenKind::Distinct {
            self.advance();
            true
        } else {
            false
        };

        let select_list = self.parse_select_list()?;

        self.expect(TokenKind::From)?;

        let graph_table = self.parse_graph_table_expression()?;

        // Optional table alias: `AS alias` or just `alias`
        let table_alias = if self.current.kind == TokenKind::As {
            self.advance();
            Some(self.expect_identifier()?)
        } else if self.current.kind == TokenKind::Identifier {
            // Bare alias (no AS), but only if it's not a SQL keyword
            Some(self.expect_identifier()?)
        } else {
            None
        };

        // Optional WHERE clause
        let where_clause = if self.current.kind == TokenKind::Where {
            self.advance();
            Some(self.parse_expression()?)
        } else {
            None
        };

        // Optional GROUP BY clause
        let group_by = if self.current.kind == TokenKind::Group {
            self.advance();
            self.expect(TokenKind::By)?;
            let mut exprs = vec![self.parse_expression()?];
            while self.current.kind == TokenKind::Comma {
                self.advance();
                exprs.push(self.parse_expression()?);
            }
            Some(exprs)
        } else {
            None
        };

        // Optional HAVING clause
        let having = if self.current.kind == TokenKind::Having {
            self.advance();
            Some(self.parse_expression()?)
        } else {
            None
        };

        // Optional ORDER BY clause
        let order_by = if self.current.kind == TokenKind::Order {
            self.advance();
            self.expect(TokenKind::By)?;
            Some(self.parse_sort_items()?)
        } else {
            None
        };

        // Optional LIMIT
        let limit = if self.current.kind == TokenKind::Limit {
            self.advance();
            Some(self.parse_integer_literal()?)
        } else {
            None
        };

        // Optional OFFSET
        let offset = if self.current.kind == TokenKind::Offset {
            self.advance();
            Some(self.parse_integer_literal()?)
        } else {
            None
        };

        Ok(SelectStatement {
            distinct,
            select_list,
            graph_table,
            table_alias,
            where_clause,
            group_by,
            having,
            order_by,
            limit,
            offset,
            span: None,
        })
    }

    fn parse_select_list(&mut self) -> Result<SelectList> {
        if self.current.kind == TokenKind::Star {
            self.advance();
            return Ok(SelectList::All);
        }

        let mut items = vec![self.parse_select_item()?];
        while self.current.kind == TokenKind::Comma {
            self.advance();
            items.push(self.parse_select_item()?);
        }

        Ok(SelectList::Columns(items))
    }

    fn parse_select_item(&mut self) -> Result<SelectItem> {
        let expression = self.parse_expression()?;
        let alias = if self.current.kind == TokenKind::As {
            self.advance();
            Some(self.expect_identifier()?)
        } else {
            None
        };

        Ok(SelectItem {
            expression,
            alias,
            span: None,
        })
    }

    // ==================== CALL Statement Parsing ====================

    fn parse_call_statement(&mut self) -> Result<Statement> {
        let span_start = self.current.span.start;
        self.expect(TokenKind::Call)?;

        // Parse dotted procedure name: grafeo.pagerank
        let mut name_parts = vec![self.expect_identifier()?];
        while self.current.kind == TokenKind::Dot {
            self.advance();
            name_parts.push(self.expect_identifier()?);
        }

        // Parse argument list: (arg1, arg2, ...)
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
            let mut items = vec![self.parse_yield_item()?];
            while self.current.kind == TokenKind::Comma {
                self.advance();
                items.push(self.parse_yield_item()?);
            }
            Some(items)
        } else {
            None
        };

        // Parse optional WHERE clause (only valid after YIELD)
        let where_clause = if yield_items.is_some() && self.current.kind == TokenKind::Where {
            self.advance();
            let expression = self.parse_expression()?;
            Some(WhereClause {
                expression,
                filter: false,
                span: None,
            })
        } else {
            None
        };

        // Parse optional ORDER BY / LIMIT (SQL-style, no RETURN keyword)
        let return_clause = if yield_items.is_some()
            && (self.current.kind == TokenKind::Order || self.current.kind == TokenKind::Limit)
        {
            let order_by = if self.current.kind == TokenKind::Order {
                Some(self.parse_call_order_by()?)
            } else {
                None
            };

            let limit = if self.current.kind == TokenKind::Limit {
                self.advance();
                Some(self.parse_expression()?)
            } else {
                None
            };

            Some(ReturnClause {
                distinct: false,
                items: vec![],
                is_wildcard: false,
                group_by: vec![],
                order_by,
                skip: None,
                limit,
                is_finish: false,
                span: None,
            })
        } else {
            None
        };

        Ok(Statement::Call(CallStatement {
            procedure_name: name_parts,
            arguments,
            yield_items,
            where_clause,
            return_clause,
            span: Some(grafeo_common::utils::error::SourceSpan::new(
                span_start,
                self.current.span.start,
                1,
                1,
            )),
        }))
    }

    fn parse_yield_item(&mut self) -> Result<YieldItem> {
        let field_name = self.expect_identifier()?;
        let alias = if self.current.kind == TokenKind::As {
            self.advance();
            Some(self.expect_identifier()?)
        } else {
            None
        };
        Ok(YieldItem {
            field_name,
            alias,
            span: None,
        })
    }

    /// Parses ORDER BY for CALL statements: `ORDER BY expr [ASC|DESC] { , expr [ASC|DESC] }`.
    fn parse_call_order_by(&mut self) -> Result<OrderByClause> {
        self.expect(TokenKind::Order)?;
        self.expect(TokenKind::By)?;

        let mut items = vec![self.parse_call_order_item()?];
        while self.current.kind == TokenKind::Comma {
            self.advance();
            items.push(self.parse_call_order_item()?);
        }

        Ok(OrderByClause { items, span: None })
    }

    fn parse_call_order_item(&mut self) -> Result<OrderByItem> {
        let expression = self.parse_expression()?;
        let order = match self.current.kind {
            TokenKind::Asc => {
                self.advance();
                GqlSortOrder::Asc
            }
            TokenKind::Desc => {
                self.advance();
                GqlSortOrder::Desc
            }
            _ => GqlSortOrder::Asc,
        };
        Ok(OrderByItem {
            expression,
            order,
            nulls: None,
        })
    }

    // ==================== GRAPH_TABLE Parsing ====================

    fn parse_graph_table_expression(&mut self) -> Result<GraphTableExpression> {
        self.expect(TokenKind::GraphTable)?;
        self.expect(TokenKind::LParen)?;

        // Optional graph name reference: GRAPH_TABLE(my_graph, MATCH ...)
        let graph_name =
            if self.current.kind == TokenKind::Identifier && self.peek_kind() == TokenKind::Comma {
                let name = self.expect_identifier()?;
                self.advance(); // consume comma
                Some(name)
            } else {
                None
            };

        let match_clause = self.parse_match_clause()?;

        // Parse optional matches: OPTIONAL MATCH or LEFT [OUTER] JOIN MATCH
        let mut optional_matches = Vec::new();
        loop {
            if self.current.kind == TokenKind::Optional {
                // OPTIONAL MATCH ...
                self.advance(); // consume OPTIONAL
                let mut opt = self.parse_match_clause()?;
                opt.optional = true;
                optional_matches.push(opt);
            } else if self.current.kind == TokenKind::Left {
                // LEFT [OUTER] JOIN MATCH ...
                self.advance(); // consume LEFT
                if self.current.kind == TokenKind::Outer {
                    self.advance(); // consume OUTER (optional)
                }
                self.expect(TokenKind::Join)?;
                let mut opt = self.parse_match_clause()?;
                opt.optional = true;
                optional_matches.push(opt);
            } else {
                break;
            }
        }

        // Optional WHERE clause inside GRAPH_TABLE (between MATCH and COLUMNS)
        let where_clause = if self.current.kind == TokenKind::Where {
            self.advance();
            Some(self.parse_expression()?)
        } else {
            None
        };

        let columns = self.parse_columns_clause()?;

        self.expect(TokenKind::RParen)?;

        Ok(GraphTableExpression {
            graph_name,
            match_clause,
            optional_matches,
            where_clause,
            columns,
            span: None,
        })
    }

    fn parse_match_clause(&mut self) -> Result<MatchClause> {
        self.expect(TokenKind::Match)?;

        let patterns = self.parse_pattern_list()?;

        Ok(MatchClause {
            optional: false,
            path_mode: None,
            search_prefix: None,
            match_mode: None,
            patterns,
            span: None,
        })
    }

    fn parse_columns_clause(&mut self) -> Result<ColumnsClause> {
        self.expect(TokenKind::Columns)?;
        self.expect(TokenKind::LParen)?;

        let mut items = vec![self.parse_column_item()?];
        while self.current.kind == TokenKind::Comma {
            self.advance();
            items.push(self.parse_column_item()?);
        }

        self.expect(TokenKind::RParen)?;

        Ok(ColumnsClause { items, span: None })
    }

    fn parse_column_item(&mut self) -> Result<ColumnItem> {
        let expression = self.parse_expression()?;
        self.expect(TokenKind::As)?;
        let alias = self.expect_identifier()?;

        Ok(ColumnItem {
            expression,
            alias,
            span: None,
        })
    }

    // ==================== Pattern Parsing ====================

    fn parse_pattern_list(&mut self) -> Result<Vec<AliasedPattern>> {
        let mut patterns = vec![self.parse_aliased_pattern()?];
        while self.current.kind == TokenKind::Comma {
            self.advance();
            patterns.push(self.parse_aliased_pattern()?);
        }
        Ok(patterns)
    }

    fn parse_aliased_pattern(&mut self) -> Result<AliasedPattern> {
        let pattern = self.parse_pattern()?;
        Ok(AliasedPattern {
            alias: None,
            path_function: None,
            search_prefix: None,
            path_mode: None,
            keep: None,
            pattern,
        })
    }

    fn parse_pattern(&mut self) -> Result<Pattern> {
        let start = self.parse_node_pattern()?;

        // Check for edge continuation
        if self.is_edge_start() {
            let mut edges = Vec::new();
            while self.is_edge_start() {
                edges.push(self.parse_edge_pattern()?);
            }
            Ok(Pattern::Path(PathPattern {
                source: start,
                edges,
                span: None,
            }))
        } else {
            Ok(Pattern::Node(start))
        }
    }

    fn parse_node_pattern(&mut self) -> Result<NodePattern> {
        self.expect(TokenKind::LParen)?;

        // Optional variable name
        let variable = if self.can_be_identifier() && self.current.kind != TokenKind::Colon {
            let name = self.get_identifier_text();
            self.advance();
            Some(name)
        } else {
            None
        };

        // Optional labels
        let mut labels = Vec::new();
        while self.current.kind == TokenKind::Colon {
            self.advance();
            labels.push(self.expect_identifier()?);
        }

        // Optional property map
        let properties = if self.current.kind == TokenKind::LBrace {
            self.parse_property_map()?
        } else {
            Vec::new()
        };

        self.expect(TokenKind::RParen)?;

        Ok(NodePattern {
            variable,
            labels,
            label_expression: None,
            properties,
            where_clause: None,
            span: None,
        })
    }

    fn parse_edge_pattern(&mut self) -> Result<EdgePattern> {
        // Determine direction and whether there's a bracket
        let (leading_direction, has_bracket) = match self.current.kind {
            TokenKind::Arrow => {
                // ->
                self.advance();
                if self.current.kind == TokenKind::LBracket {
                    // Unusual but handle: ->[...]->
                    (Some(EdgeDirection::Outgoing), true)
                } else {
                    (Some(EdgeDirection::Outgoing), false)
                }
            }
            TokenKind::LeftArrow => {
                // <-  which may be followed by [...]-
                self.advance();
                if self.current.kind == TokenKind::LBracket {
                    // <-[...]-  incoming edge with details
                    (Some(EdgeDirection::Incoming), true)
                } else {
                    (Some(EdgeDirection::Incoming), false)
                }
            }
            TokenKind::DoubleDash => {
                // --
                self.advance();
                (Some(EdgeDirection::Undirected), false)
            }
            TokenKind::Minus => {
                // - followed by [ or other
                self.advance();
                if self.current.kind == TokenKind::LBracket {
                    (None, true) // Direction determined by trailing symbol
                } else {
                    return Err(self.error("Expected '[' or arrow after '-'"));
                }
            }
            _ => return Err(self.error("Expected edge pattern")),
        };

        // Parse edge details if there's a bracket
        let (variable, types, min_hops, max_hops, properties, direction) = if has_bracket {
            self.advance(); // consume [

            // Optional variable
            let var = if self.can_be_identifier() && self.peek_kind() != TokenKind::LParen {
                let is_var = matches!(
                    self.peek_kind(),
                    TokenKind::Colon | TokenKind::RBracket | TokenKind::Star
                );
                if is_var {
                    let name = self.get_identifier_text();
                    self.advance();
                    Some(name)
                } else {
                    None
                }
            } else {
                None
            };

            // Optional :Type, or alternatives :Type1|Type2. An edge has one
            // type, so a second `:` is an error naming both ways to write
            // what was meant.
            let mut edge_types = Vec::new();
            if self.current.kind == TokenKind::Colon {
                self.advance();
                edge_types.push(self.expect_identifier()?);
                while self.current.kind == TokenKind::Pipe {
                    self.advance();
                    edge_types.push(self.expect_identifier()?);
                }
                if self.current.kind == TokenKind::Colon {
                    let colon = self.current.span;
                    self.advance();
                    let first = edge_types.last().map_or("A", String::as_str);
                    let second = if self.can_be_identifier() {
                        self.get_identifier_text()
                    } else {
                        "B".to_string()
                    };
                    return Err(QueryError::new(
                        QueryErrorKind::Syntax,
                        format!(
                            "[SQL/PGQ] An edge has one type: write :\"{first}:{second}\" for the \
                             type named {first}:{second}, or :{first}|{second} to match either type"
                        ),
                    )
                    .with_span(colon)
                    .with_source(self.source.to_string())
                    .into());
                }
            }

            // Optional *min..max for variable-length
            let (min, max) = if self.current.kind == TokenKind::Star {
                self.advance();
                self.parse_hop_range()?
            } else {
                (None, None)
            };

            // Optional properties
            let props = if self.current.kind == TokenKind::LBrace {
                self.parse_property_map()?
            } else {
                Vec::new()
            };

            self.expect(TokenKind::RBracket)?;

            // Determine direction from trailing symbol + leading context
            let dir = if self.current.kind == TokenKind::Arrow {
                self.advance();
                EdgeDirection::Outgoing
            } else if self.current.kind == TokenKind::Minus {
                self.advance();
                // If we had a leading <-, this is <-[...]-  → Incoming
                if leading_direction == Some(EdgeDirection::Incoming) {
                    EdgeDirection::Incoming
                } else {
                    EdgeDirection::Undirected
                }
            } else {
                leading_direction.unwrap_or(EdgeDirection::Undirected)
            };

            (var, edge_types, min, max, props, dir)
        } else {
            // No bracket - simple arrow connection
            let dir = leading_direction.unwrap_or(EdgeDirection::Undirected);
            (None, Vec::new(), None, None, Vec::new(), dir)
        };

        // Parse the target node
        let target = self.parse_node_pattern()?;

        Ok(EdgePattern {
            variable,
            types,
            direction,
            target,
            min_hops,
            max_hops,
            properties,
            where_clause: None,
            questioned: false,
            span: None,
        })
    }

    fn parse_hop_range(&mut self) -> Result<(Option<u32>, Option<u32>)> {
        let min = if self.current.kind == TokenKind::Integer {
            let val = self.current.text.parse().unwrap_or(1);
            self.advance();
            Some(val)
        } else {
            None
        };

        let max = if self.current.kind == TokenKind::Dot {
            self.advance();
            // Expect second dot
            if self.current.kind != TokenKind::Dot {
                return Err(self.error("Expected '..' in hop range"));
            }
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

        Ok((min, max))
    }

    fn parse_property_map(&mut self) -> Result<Vec<(String, Expression)>> {
        self.expect(TokenKind::LBrace)?;

        let mut props = Vec::new();
        if self.current.kind != TokenKind::RBrace {
            let key = self.expect_identifier()?;
            self.expect(TokenKind::Colon)?;
            let value = self.parse_expression()?;
            props.push((key, value));

            while self.current.kind == TokenKind::Comma {
                self.advance();
                let key = self.expect_identifier()?;
                self.expect(TokenKind::Colon)?;
                let value = self.parse_expression()?;
                props.push((key, value));
            }
        }

        self.expect(TokenKind::RBrace)?;
        Ok(props)
    }

    // ==================== Expression Parsing ====================

    fn parse_expression(&mut self) -> Result<Expression> {
        self.enter_nesting()?;
        let result = self.parse_or_expression();
        self.exit_nesting();
        result
    }

    fn parse_or_expression(&mut self) -> Result<Expression> {
        self.parse_balanced_chain(TokenKind::Or, BinaryOp::Or, Self::parse_and_expression)
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
            Ok(Expression::Unary {
                op: UnaryOp::Not,
                operand: Box::new(operand),
            })
        } else {
            self.parse_comparison_expression()
        }
    }

    fn parse_comparison_expression(&mut self) -> Result<Expression> {
        let chain = self.begin_chain();
        let expression = self.parse_comparison_chain()?;
        self.end_chain(chain);
        Ok(expression)
    }

    /// The operands and operators of a chain of comparisons, each operator
    /// nesting one level (see [`Self::parse_comparison_expression`]).
    fn parse_comparison_chain(&mut self) -> Result<Expression> {
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
                TokenKind::Like => BinaryOp::Like,
                TokenKind::Is => {
                    self.advance();
                    let not = self.current.kind == TokenKind::Not;
                    if not {
                        self.advance();
                    }
                    self.expect(TokenKind::Null)?;
                    self.link_chain()?;
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
                TokenKind::Between => {
                    // BETWEEN low AND high → left >= low AND left <= high,
                    // two levels above its operands
                    self.advance();
                    let low = self.parse_additive_expression()?;
                    self.expect(TokenKind::And)?;
                    let high = self.parse_additive_expression()?;
                    self.link_chain()?;
                    self.link_chain()?;
                    left = Expression::Binary {
                        left: Box::new(Expression::Binary {
                            left: Box::new(left.clone()),
                            op: BinaryOp::Ge,
                            right: Box::new(low),
                        }),
                        op: BinaryOp::And,
                        right: Box::new(Expression::Binary {
                            left: Box::new(left),
                            op: BinaryOp::Le,
                            right: Box::new(high),
                        }),
                    };
                    // BETWEEN consumes both operands, return directly
                    return Ok(left);
                }
                _ => break,
            };

            self.advance();
            let right = self.parse_additive_expression()?;
            self.link_chain()?;
            left = Expression::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
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

    fn parse_unary_expression(&mut self) -> Result<Expression> {
        match self.current.kind {
            TokenKind::Minus => {
                self.advance();
                self.enter_nesting()?;
                let operand = self.parse_unary_expression();
                self.exit_nesting();
                let operand = operand?;
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
            // A CASE nests one level more than its operands: parsing and
            // planning one takes more stack.
            TokenKind::Case => {
                self.enter_nesting()?;
                let case = self.parse_case_expression();
                self.exit_nesting();
                case
            }
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
                let value = text.parse().map_err(|e: std::num::ParseIntError| {
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
                self.advance();

                // Check for function call: its arguments nest one level more
                // than an expression in parentheses, as the call keeps the
                // large frame of this parser on the stack.
                if self.current.kind == TokenKind::LParen {
                    self.advance();
                    self.enter_nesting()?;
                    let call = self.parse_function_arguments(name);
                    self.exit_nesting();
                    call
                } else if self.current.kind == TokenKind::Dot {
                    // Property access: name.property
                    self.advance();
                    let property = self.expect_identifier()?;
                    Ok(Expression::PropertyAccess {
                        variable: name,
                        property,
                    })
                } else {
                    Ok(Expression::Variable(name))
                }
            }
            // The expression nests one level deeper (see `parse_expression`).
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
                // Map literal: {key: value, ...}
                let props = self.parse_property_map()?;
                Ok(Expression::Map(props))
            }
            _ => Err(self.error(&format!(
                "Expected expression, found {:?}",
                self.current.kind
            ))),
        }
    }

    // ==================== CASE Expression ====================

    /// Parses a CASE expression (simple or searched).
    ///
    /// Simple:   `CASE expr WHEN val THEN result [ELSE default] END`
    /// Searched: `CASE WHEN condition THEN result [ELSE default] END`
    /// Parses the arguments of a call of the function `name`, after its `(`.
    fn parse_function_arguments(&mut self, name: String) -> Result<Expression> {
        let distinct = if self.current.kind == TokenKind::Distinct {
            self.advance();
            true
        } else {
            false
        };

        let mut args = Vec::new();
        if self.current.kind == TokenKind::Star {
            // Handle COUNT(*)
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
    }

    fn parse_case_expression(&mut self) -> Result<Expression> {
        self.expect(TokenKind::Case)?;

        // Determine simple vs searched CASE.
        // If the next token is not WHEN, we have a simple CASE with an input expression.
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

    // ==================== ORDER BY / LIMIT / OFFSET ====================

    fn parse_sort_items(&mut self) -> Result<Vec<SortItem>> {
        let mut items = vec![self.parse_sort_item()?];
        while self.current.kind == TokenKind::Comma {
            self.advance();
            items.push(self.parse_sort_item()?);
        }
        Ok(items)
    }

    fn parse_sort_item(&mut self) -> Result<SortItem> {
        let expression = self.parse_expression()?;
        let direction = match self.current.kind {
            TokenKind::Asc => {
                self.advance();
                SortDirection::Asc
            }
            TokenKind::Desc => {
                self.advance();
                SortDirection::Desc
            }
            _ => SortDirection::Asc, // Default ascending
        };
        Ok(SortItem {
            expression,
            direction,
        })
    }

    fn parse_integer_literal(&mut self) -> Result<u64> {
        if self.current.kind == TokenKind::Integer {
            let text = &self.current.text;
            let value = text.parse().map_err(|e: std::num::ParseIntError| {
                if *e.kind() == std::num::IntErrorKind::PosOverflow
                    || *e.kind() == std::num::IntErrorKind::NegOverflow
                {
                    self.error(&format!(
                        "Integer literal '{}' overflows the valid range (0 to {})",
                        text,
                        u64::MAX
                    ))
                } else {
                    self.error(&format!("Invalid integer literal: '{}'", text))
                }
            })?;
            self.advance();
            Ok(value)
        } else {
            Err(self.error("Expected integer"))
        }
    }

    // ==================== Helpers ====================

    fn advance(&mut self) {
        self.previous = std::mem::replace(&mut self.current, self.lexer.next_token());
    }

    fn expect(&mut self, kind: TokenKind) -> Result<()> {
        if self.current.kind == kind {
            self.advance();
            Ok(())
        } else {
            Err(self.error(&format!(
                "Expected {:?}, found {:?}",
                kind, self.current.kind
            )))
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

    fn can_be_identifier(&self) -> bool {
        matches!(
            self.current.kind,
            TokenKind::Identifier
                | TokenKind::QuotedIdentifier
                // Contextual keywords that can be used as identifiers
                | TokenKind::Asc
                | TokenKind::Desc
                | TokenKind::Offset
                | TokenKind::Columns
                | TokenKind::Property
                | TokenKind::Graph
                | TokenKind::Node
                | TokenKind::Edge
                | TokenKind::Tables
                | TokenKind::Key
                | TokenKind::References
                | TokenKind::Call
                | TokenKind::Yield
                | TokenKind::When
                | TokenKind::Then
                | TokenKind::Else
                | TokenKind::End
        )
    }

    fn get_identifier_text(&self) -> String {
        let mut text = self.current.text.clone();
        // Remove quotes from quoted identifier
        if self.current.kind == TokenKind::QuotedIdentifier {
            text = text[1..text.len() - 1].to_string();
        }
        text
    }

    fn peek_kind(&mut self) -> TokenKind {
        let saved = self.lexer.clone();
        let token = self.lexer.next_token();
        let kind = token.kind;
        self.lexer = saved;
        kind
    }

    fn is_edge_start(&self) -> bool {
        matches!(
            self.current.kind,
            TokenKind::Arrow | TokenKind::LeftArrow | TokenKind::DoubleDash | TokenKind::Minus
        )
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

    fn error(&self, message: &str) -> grafeo_common::utils::error::Error {
        QueryError::new(QueryErrorKind::Syntax, format!("[SQL/PGQ] {message}"))
            .with_span(self.current.span)
            .with_source(self.source.to_string())
            .into()
    }

    // ==================== DDL Parsing ====================

    fn parse_create_statement(&mut self) -> Result<Statement> {
        self.expect(TokenKind::Create)?;
        self.expect(TokenKind::Property)?;
        self.expect(TokenKind::Graph)?;

        let name = self.expect_identifier()?;

        let mut node_tables = Vec::new();
        let mut edge_tables = Vec::new();

        // Parse NODE TABLES and EDGE TABLES in any order
        loop {
            if self.current.kind == TokenKind::Node {
                self.advance();
                self.expect(TokenKind::Tables)?;
                self.expect(TokenKind::LParen)?;
                node_tables.extend(self.parse_node_table_list()?);
                self.expect(TokenKind::RParen)?;
            } else if self.current.kind == TokenKind::Edge {
                self.advance();
                self.expect(TokenKind::Tables)?;
                self.expect(TokenKind::LParen)?;
                edge_tables.extend(self.parse_edge_table_list()?);
                self.expect(TokenKind::RParen)?;
            } else {
                break;
            }
        }

        if node_tables.is_empty() && edge_tables.is_empty() {
            return Err(self.error("Expected NODE TABLES or EDGE TABLES"));
        }

        Ok(Statement::CreatePropertyGraph(
            CreatePropertyGraphStatement {
                name,
                node_tables,
                edge_tables,
                span: None,
            },
        ))
    }

    fn parse_node_table_list(&mut self) -> Result<Vec<NodeTableDefinition>> {
        let mut tables = vec![self.parse_node_table()?];
        while self.current.kind == TokenKind::Comma {
            self.advance();
            tables.push(self.parse_node_table()?);
        }
        Ok(tables)
    }

    fn parse_node_table(&mut self) -> Result<NodeTableDefinition> {
        let name = self.expect_identifier()?;
        self.expect(TokenKind::LParen)?;
        let columns = self.parse_column_definition_list()?;
        self.expect(TokenKind::RParen)?;

        Ok(NodeTableDefinition {
            name,
            columns,
            span: None,
        })
    }

    fn parse_edge_table_list(&mut self) -> Result<Vec<EdgeTableDefinition>> {
        let mut tables = vec![self.parse_edge_table()?];
        while self.current.kind == TokenKind::Comma {
            self.advance();
            tables.push(self.parse_edge_table()?);
        }
        Ok(tables)
    }

    fn parse_edge_table(&mut self) -> Result<EdgeTableDefinition> {
        let name = self.expect_identifier()?;
        self.expect(TokenKind::LParen)?;
        let columns = self.parse_column_definition_list()?;
        self.expect(TokenKind::RParen)?;

        // Extract source and target references from columns
        let mut source_table = None;
        let mut source_column = None;
        let mut target_table = None;
        let mut target_column = None;

        for col in &columns {
            if let Some(ref fk) = col.references {
                if source_table.is_none() {
                    source_table = Some(fk.table.clone());
                    source_column = Some(fk.column.clone());
                } else if target_table.is_none() {
                    target_table = Some(fk.table.clone());
                    target_column = Some(fk.column.clone());
                }
            }
        }

        Ok(EdgeTableDefinition {
            name,
            columns,
            source_table: source_table.unwrap_or_default(),
            source_column: source_column.unwrap_or_default(),
            target_table: target_table.unwrap_or_default(),
            target_column: target_column.unwrap_or_default(),
            span: None,
        })
    }

    fn parse_column_definition_list(&mut self) -> Result<Vec<ColumnDefinition>> {
        let mut columns = vec![self.parse_column_definition()?];
        while self.current.kind == TokenKind::Comma {
            self.advance();
            columns.push(self.parse_column_definition()?);
        }
        Ok(columns)
    }

    fn parse_column_definition(&mut self) -> Result<ColumnDefinition> {
        let name = self.expect_identifier()?;
        let data_type = self.parse_data_type()?;

        let mut primary_key = false;
        let mut references = None;

        // Parse optional constraints
        loop {
            if self.current.kind == TokenKind::Primary {
                self.advance();
                self.expect(TokenKind::Key)?;
                primary_key = true;
            } else if self.current.kind == TokenKind::References {
                self.advance();
                let table = self.expect_identifier()?;
                self.expect(TokenKind::LParen)?;
                let column = self.expect_identifier()?;
                self.expect(TokenKind::RParen)?;
                references = Some(ForeignKeyRef { table, column });
            } else {
                break;
            }
        }

        Ok(ColumnDefinition {
            name,
            data_type,
            primary_key,
            references,
            span: None,
        })
    }

    fn parse_data_type(&mut self) -> Result<SqlDataType> {
        if !self.can_be_identifier() {
            return Err(self.error("Expected data type"));
        }
        let type_name = self.get_identifier_text().to_uppercase();
        self.advance();

        match type_name.as_str() {
            "BIGINT" => Ok(SqlDataType::Bigint),
            "INT" | "INTEGER" => Ok(SqlDataType::Int),
            "VARCHAR" => {
                let length = if self.current.kind == TokenKind::LParen {
                    self.advance();
                    // reason: VARCHAR length is a small literal, fits in usize on all targets
                    #[allow(clippy::cast_possible_truncation)]
                    let len = self.parse_integer_literal()? as usize;
                    self.expect(TokenKind::RParen)?;
                    Some(len)
                } else {
                    None
                };
                Ok(SqlDataType::Varchar(length))
            }
            "BOOLEAN" | "BOOL" => Ok(SqlDataType::Boolean),
            "FLOAT" | "REAL" => Ok(SqlDataType::Float),
            "DOUBLE" => Ok(SqlDataType::Double),
            "TIMESTAMP" => Ok(SqlDataType::Timestamp),
            _ => Err(self.error(&format!("Unknown data type: {type_name}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(query: &str) -> Statement {
        let mut parser = Parser::new(query);
        parser
            .parse()
            .unwrap_or_else(|e| panic!("Failed to parse: {query}\nError: {e}"))
    }

    fn parse_err(query: &str) {
        let mut parser = Parser::new(query);
        assert!(
            parser.parse().is_err(),
            "Expected parse error for: {}",
            query
        );
    }

    #[test]
    fn an_edge_has_one_type_or_alternatives() {
        parse_ok(
            "SELECT * FROM GRAPH_TABLE (MATCH (a)-[:A|\"Graph:CONTAINS\"]->(b) COLUMNS (b.id AS id))",
        );
        let query =
            "SELECT * FROM GRAPH_TABLE (MATCH (a)-[:Graph:CONTAINS]->(b) COLUMNS (b.id AS id))";
        let err = Parser::new(query).parse().expect_err(query).to_string();
        for advice in [":\"Graph:CONTAINS\"", ":Graph|CONTAINS"] {
            assert!(err.contains(advice), "names {advice}: {err}");
        }
        parse_err("SELECT * FROM GRAPH_TABLE (MATCH (a)-[:A|B:C*]->(b) COLUMNS (b.id AS id))");
    }

    fn select(stmt: Statement) -> SelectStatement {
        match stmt {
            Statement::Select(s) => s,
            other => panic!("Expected Select, got {other:?}"),
        }
    }

    fn create_pg(stmt: Statement) -> CreatePropertyGraphStatement {
        match stmt {
            Statement::CreatePropertyGraph(cpg) => cpg,
            other => panic!("Expected CreatePropertyGraph, got {other:?}"),
        }
    }

    // ==================== Basic Queries ====================

    #[test]
    fn test_basic_select_star() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            )",
        ));
        assert!(matches!(s.select_list, SelectList::All));
        assert!(s.table_alias.is_none());
        assert!(s.where_clause.is_none());
    }

    #[test]
    fn test_select_columns() {
        let s = select(parse_ok(
            "SELECT g.person, g.friend FROM GRAPH_TABLE (
                MATCH (a:Person)-[e:KNOWS]->(b:Person)
                COLUMNS (a.name AS person, b.name AS friend)
            ) AS g",
        ));
        if let SelectList::Columns(items) = &s.select_list {
            assert_eq!(items.len(), 2);
        } else {
            panic!("Expected Columns");
        }
        assert_eq!(s.table_alias, Some("g".to_string()));
    }

    // ==================== Pattern Tests ====================

    #[test]
    fn test_node_only_pattern() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            )",
        ));
        let pattern = &s.graph_table.match_clause.patterns[0].pattern;
        if let Pattern::Node(node) = pattern {
            assert_eq!(node.variable, Some("n".to_string()));
            assert_eq!(node.labels, vec!["Person"]);
        } else {
            panic!("Expected Node pattern");
        }
    }

    #[test]
    fn test_outgoing_edge() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)-[e:KNOWS]->(b:Person)
                COLUMNS (a.name AS person, b.name AS friend)
            )",
        ));
        let pattern = &s.graph_table.match_clause.patterns[0].pattern;
        if let Pattern::Path(path) = pattern {
            assert_eq!(path.edges.len(), 1);
            assert_eq!(path.edges[0].direction, EdgeDirection::Outgoing);
            assert_eq!(path.edges[0].types, vec!["KNOWS"]);
            assert_eq!(path.edges[0].variable, Some("e".to_string()));
        } else {
            panic!("Expected Path pattern");
        }
    }

    #[test]
    fn test_incoming_edge() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)<-[:KNOWS]-(b:Person)
                COLUMNS (a.name AS name)
            )",
        ));
        let pattern = &s.graph_table.match_clause.patterns[0].pattern;
        if let Pattern::Path(path) = pattern {
            assert_eq!(path.edges[0].direction, EdgeDirection::Incoming);
        } else {
            panic!("Expected Path pattern");
        }
    }

    #[test]
    fn test_multi_hop_pattern() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)-[:KNOWS]->(b:Person)-[:WORKS_AT]->(c:Company)
                COLUMNS (a.name AS person, c.name AS company)
            )",
        ));
        let pattern = &s.graph_table.match_clause.patterns[0].pattern;
        if let Pattern::Path(path) = pattern {
            assert_eq!(path.edges.len(), 2);
            assert_eq!(path.edges[0].types, vec!["KNOWS"]);
            assert_eq!(path.edges[1].types, vec!["WORKS_AT"]);
        } else {
            panic!("Expected Path pattern");
        }
    }

    #[test]
    fn test_node_with_properties() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person {name: 'Alix'})
                COLUMNS (n.name AS name)
            )",
        ));
        let pattern = &s.graph_table.match_clause.patterns[0].pattern;
        if let Pattern::Node(node) = pattern {
            assert_eq!(node.properties.len(), 1);
            assert_eq!(node.properties[0].0, "name");
        } else {
            panic!("Expected Node pattern");
        }
    }

    // ==================== COLUMNS Tests ====================

    #[test]
    fn test_columns_clause() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)-[e:KNOWS]->(b:Person)
                COLUMNS (a.name AS person, e.since AS year, b.name AS friend)
            )",
        ));
        assert_eq!(s.graph_table.columns.items.len(), 3);
        assert_eq!(s.graph_table.columns.items[0].alias, "person");
        assert_eq!(s.graph_table.columns.items[1].alias, "year");
        assert_eq!(s.graph_table.columns.items[2].alias, "friend");
    }

    // ==================== WHERE Clause Tests ====================

    #[test]
    fn test_where_equality() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            ) AS g
            WHERE g.name = 'Alix'",
        ));
        assert!(s.where_clause.is_some());
        if let Some(Expression::Binary { op, .. }) = &s.where_clause {
            assert_eq!(*op, BinaryOp::Eq);
        }
    }

    #[test]
    fn test_where_like() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            ) AS g
            WHERE g.name LIKE 'Al%'",
        ));
        assert!(s.where_clause.is_some());
        if let Some(Expression::Binary { op, .. }) = &s.where_clause {
            assert_eq!(*op, BinaryOp::Like);
        }
    }

    #[test]
    fn test_where_is_null() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            ) AS g
            WHERE g.name IS NULL",
        ));
        assert!(matches!(
            s.where_clause,
            Some(Expression::Unary {
                op: UnaryOp::IsNull,
                ..
            })
        ));
    }

    #[test]
    fn test_where_is_not_null() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            ) AS g
            WHERE g.name IS NOT NULL",
        ));
        assert!(matches!(
            s.where_clause,
            Some(Expression::Unary {
                op: UnaryOp::IsNotNull,
                ..
            })
        ));
    }

    #[test]
    fn test_where_and_or() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name, n.age AS age)
            ) AS g
            WHERE g.age > 20 AND g.age < 60 OR g.name = 'Admin'",
        ));
        // OR has lower precedence than AND
        assert!(matches!(
            s.where_clause,
            Some(Expression::Binary {
                op: BinaryOp::Or,
                ..
            })
        ));
    }

    // ==================== ORDER BY Tests ====================

    #[test]
    fn test_order_by_asc() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            ) AS g
            ORDER BY g.name ASC",
        ));
        let items = s.order_by.unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].direction, SortDirection::Asc);
    }

    #[test]
    fn test_order_by_desc() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name, n.age AS age)
            ) AS g
            ORDER BY g.age DESC",
        ));
        let items = s.order_by.unwrap();
        assert_eq!(items[0].direction, SortDirection::Desc);
    }

    #[test]
    fn test_order_by_multiple() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name, n.age AS age)
            ) AS g
            ORDER BY g.name, g.age DESC",
        ));
        let items = s.order_by.unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].direction, SortDirection::Asc); // default
        assert_eq!(items[1].direction, SortDirection::Desc);
    }

    // ==================== LIMIT / OFFSET Tests ====================

    #[test]
    fn test_limit() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            ) LIMIT 10",
        ));
        assert_eq!(s.limit, Some(10));
    }

    #[test]
    fn test_offset() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            ) OFFSET 5",
        ));
        assert_eq!(s.offset, Some(5));
    }

    #[test]
    fn test_limit_and_offset() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            ) LIMIT 10 OFFSET 5",
        ));
        assert_eq!(s.limit, Some(10));
        assert_eq!(s.offset, Some(5));
    }

    // ==================== Full Query Tests ====================

    #[test]
    fn test_full_query() {
        let s = select(parse_ok(
            "SELECT g.person, g.friend
             FROM GRAPH_TABLE (
                MATCH (a:Person)-[e:KNOWS]->(b:Person)
                COLUMNS (a.name AS person, e.since AS year, b.name AS friend)
             ) AS g
             WHERE g.person LIKE 'Al%'
             ORDER BY g.year DESC
             LIMIT 10
             OFFSET 5",
        ));
        if let SelectList::Columns(items) = &s.select_list {
            assert_eq!(items.len(), 2);
        }
        assert_eq!(s.table_alias, Some("g".to_string()));
        assert!(s.where_clause.is_some());
        assert!(s.order_by.is_some());
        assert_eq!(s.limit, Some(10));
        assert_eq!(s.offset, Some(5));
    }

    #[test]
    fn test_trailing_semicolon() {
        let stmt = parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            );",
        );
        assert!(matches!(stmt, Statement::Select(_)));
    }

    // ==================== Error Cases ====================

    // ==================== Variable-Length Path Tests ====================

    #[test]
    fn test_variable_length_edge() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)-[p:KNOWS*1..5]->(b:Person)
                COLUMNS (a.name AS source, LENGTH(p) AS distance, b.name AS target)
            )",
        ));
        let pattern = &s.graph_table.match_clause.patterns[0].pattern;
        if let Pattern::Path(path) = pattern {
            assert_eq!(path.edges.len(), 1);
            assert_eq!(path.edges[0].variable, Some("p".to_string()));
            assert_eq!(path.edges[0].min_hops, Some(1));
            assert_eq!(path.edges[0].max_hops, Some(5));
        } else {
            panic!("Expected Path pattern");
        }

        // Verify COLUMNS has 3 items including the LENGTH function
        assert_eq!(s.graph_table.columns.items.len(), 3);
        assert_eq!(s.graph_table.columns.items[1].alias, "distance");
        assert!(matches!(
            &s.graph_table.columns.items[1].expression,
            Expression::FunctionCall { name, args, .. } if name == "LENGTH" && args.len() == 1
        ));
    }

    #[test]
    fn test_path_functions_in_columns() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)-[p:KNOWS*1..3]->(b:Person)
                COLUMNS (NODES(p) AS path_nodes, EDGES(p) AS path_edges)
            )",
        ));
        assert_eq!(s.graph_table.columns.items.len(), 2);
        assert!(matches!(
            &s.graph_table.columns.items[0].expression,
            Expression::FunctionCall { name, .. } if name == "NODES"
        ));
        assert!(matches!(
            &s.graph_table.columns.items[1].expression,
            Expression::FunctionCall { name, .. } if name == "EDGES"
        ));
    }

    // ==================== Error Cases ====================

    #[test]
    fn test_error_missing_from() {
        parse_err("SELECT * GRAPH_TABLE ( MATCH (n) COLUMNS (n.name AS name) )");
    }

    #[test]
    fn test_error_missing_graph_table() {
        parse_err("SELECT * FROM ( MATCH (n) COLUMNS (n.name AS name) )");
    }

    #[test]
    fn test_error_missing_match() {
        parse_err("SELECT * FROM GRAPH_TABLE ( COLUMNS (n.name AS name) )");
    }

    #[test]
    fn test_error_missing_columns() {
        parse_err("SELECT * FROM GRAPH_TABLE ( MATCH (n:Person) )");
    }

    #[test]
    fn test_error_missing_column_alias() {
        parse_err("SELECT * FROM GRAPH_TABLE ( MATCH (n:Person) COLUMNS (n.name) )");
    }

    #[test]
    fn test_error_empty_query() {
        parse_err("");
    }

    #[test]
    fn test_error_select_from_only() {
        parse_err("SELECT FROM");
    }

    // ==================== CREATE PROPERTY GRAPH Tests ====================

    #[test]
    fn test_create_property_graph_basic() {
        let cpg = create_pg(parse_ok(
            "CREATE PROPERTY GRAPH SocialGraph
             NODE TABLES (
                 Person (id BIGINT PRIMARY KEY, name VARCHAR, age INT)
             )
             EDGE TABLES (
                 KNOWS (
                     id BIGINT PRIMARY KEY,
                     source BIGINT REFERENCES Person(id),
                     target BIGINT REFERENCES Person(id),
                     since INT
                 )
             )",
        ));
        assert_eq!(cpg.name, "SocialGraph");
        assert_eq!(cpg.node_tables.len(), 1);
        assert_eq!(cpg.node_tables[0].name, "Person");
        assert_eq!(cpg.node_tables[0].columns.len(), 3);

        // Verify column types
        assert_eq!(cpg.node_tables[0].columns[0].name, "id");
        assert_eq!(cpg.node_tables[0].columns[0].data_type, SqlDataType::Bigint);
        assert!(cpg.node_tables[0].columns[0].primary_key);

        assert_eq!(cpg.node_tables[0].columns[1].name, "name");
        assert_eq!(
            cpg.node_tables[0].columns[1].data_type,
            SqlDataType::Varchar(None)
        );

        assert_eq!(cpg.node_tables[0].columns[2].name, "age");
        assert_eq!(cpg.node_tables[0].columns[2].data_type, SqlDataType::Int);

        // Edge table
        assert_eq!(cpg.edge_tables.len(), 1);
        assert_eq!(cpg.edge_tables[0].name, "KNOWS");
        assert_eq!(cpg.edge_tables[0].columns.len(), 4);
        assert_eq!(cpg.edge_tables[0].source_table, "Person");
        assert_eq!(cpg.edge_tables[0].source_column, "id");
        assert_eq!(cpg.edge_tables[0].target_table, "Person");
        assert_eq!(cpg.edge_tables[0].target_column, "id");
    }

    #[test]
    fn test_create_property_graph_multiple_node_tables() {
        let cpg = create_pg(parse_ok(
            "CREATE PROPERTY GRAPH CompanyGraph
             NODE TABLES (
                 Person (id BIGINT PRIMARY KEY, name VARCHAR),
                 Company (id BIGINT PRIMARY KEY, name VARCHAR(255))
             )
             EDGE TABLES (
                 WORKS_AT (
                     id BIGINT PRIMARY KEY,
                     employee BIGINT REFERENCES Person(id),
                     employer BIGINT REFERENCES Company(id),
                     since INT
                 )
             )",
        ));
        assert_eq!(cpg.name, "CompanyGraph");
        assert_eq!(cpg.node_tables.len(), 2);
        assert_eq!(cpg.node_tables[0].name, "Person");
        assert_eq!(cpg.node_tables[1].name, "Company");

        // VARCHAR with length
        assert_eq!(
            cpg.node_tables[1].columns[1].data_type,
            SqlDataType::Varchar(Some(255))
        );
    }

    #[test]
    fn test_create_property_graph_node_only() {
        let cpg = create_pg(parse_ok(
            "CREATE PROPERTY GRAPH SimpleGraph
             NODE TABLES (
                 Item (id BIGINT PRIMARY KEY, label VARCHAR)
             )",
        ));
        assert_eq!(cpg.name, "SimpleGraph");
        assert_eq!(cpg.node_tables.len(), 1);
        assert!(cpg.edge_tables.is_empty());
    }

    #[test]
    fn test_create_property_graph_with_semicolon() {
        let cpg = create_pg(parse_ok(
            "CREATE PROPERTY GRAPH G
             NODE TABLES (
                 N (id BIGINT PRIMARY KEY)
             );",
        ));
        assert_eq!(cpg.name, "G");
    }

    #[test]
    fn test_create_property_graph_all_data_types() {
        let cpg = create_pg(parse_ok(
            "CREATE PROPERTY GRAPH TypeTest
             NODE TABLES (
                 AllTypes (
                     a BIGINT PRIMARY KEY,
                     b INT,
                     c VARCHAR,
                     d VARCHAR(100),
                     e BOOLEAN,
                     f FLOAT,
                     g DOUBLE,
                     h TIMESTAMP
                 )
             )",
        ));
        let cols = &cpg.node_tables[0].columns;
        assert_eq!(cols.len(), 8);
        assert_eq!(cols[0].data_type, SqlDataType::Bigint);
        assert_eq!(cols[1].data_type, SqlDataType::Int);
        assert_eq!(cols[2].data_type, SqlDataType::Varchar(None));
        assert_eq!(cols[3].data_type, SqlDataType::Varchar(Some(100)));
        assert_eq!(cols[4].data_type, SqlDataType::Boolean);
        assert_eq!(cols[5].data_type, SqlDataType::Float);
        assert_eq!(cols[6].data_type, SqlDataType::Double);
        assert_eq!(cols[7].data_type, SqlDataType::Timestamp);
    }

    #[test]
    fn test_error_create_property_graph_no_tables() {
        parse_err("CREATE PROPERTY GRAPH EmptyGraph");
    }

    #[test]
    fn test_error_create_missing_property() {
        parse_err("CREATE GRAPH G NODE TABLES (N (id BIGINT PRIMARY KEY))");
    }

    // ==================== Additional Error Cases ====================

    #[test]
    fn test_error_unclosed_graph_table_paren() {
        parse_err(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)",
        );
    }

    #[test]
    fn test_error_unclosed_node_pattern() {
        parse_err(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person
                COLUMNS (n.name AS name)
            )",
        );
    }

    #[test]
    fn test_error_empty_columns_clause() {
        parse_err(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS ()
            )",
        );
    }

    #[test]
    fn test_error_trailing_garbage() {
        parse_err(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            ) EXTRA STUFF HERE",
        );
    }

    #[test]
    fn test_error_unknown_data_type() {
        parse_err(
            "CREATE PROPERTY GRAPH G
             NODE TABLES (
                 T (id UNKNOWN_TYPE PRIMARY KEY)
             )",
        );
    }

    #[test]
    fn test_error_between_without_and() {
        parse_err(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.age AS age)
            ) AS g
            WHERE g.age BETWEEN 10",
        );
    }

    #[test]
    fn test_error_limit_non_integer() {
        parse_err(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            ) LIMIT abc",
        );
    }

    #[test]
    fn test_error_order_by_missing_by() {
        parse_err(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            ) ORDER name",
        );
    }

    #[test]
    fn test_error_edge_missing_bracket() {
        parse_err(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)-(b:Person)
                COLUMNS (a.name AS name)
            )",
        );
    }

    #[test]
    fn test_error_select_without_from() {
        parse_err("SELECT *");
    }

    #[test]
    fn test_error_double_semicolon() {
        parse_err(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            );;",
        );
    }

    // ==================== CASE Expression Tests ====================

    #[test]
    fn test_parse_case_searched() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (
                    CASE WHEN n.age > 30 THEN 'senior' ELSE 'junior' END AS category
                )
            )",
        ));
        let expr = &s.graph_table.columns.items[0].expression;
        if let Expression::Case {
            input,
            whens,
            else_clause,
        } = expr
        {
            // Searched CASE has no input expression
            assert!(input.is_none());
            assert_eq!(whens.len(), 1);
            // WHEN condition: n.age > 30
            assert!(matches!(
                &whens[0].0,
                Expression::Binary {
                    op: BinaryOp::Gt,
                    ..
                }
            ));
            // THEN result: 'senior'
            assert!(matches!(
                &whens[0].1,
                Expression::Literal(Literal::String(s)) if s == "senior"
            ));
            // ELSE result: 'junior'
            let else_expr = else_clause.as_ref().expect("should have ELSE clause");
            assert!(matches!(
                else_expr.as_ref(),
                Expression::Literal(Literal::String(s)) if s == "junior"
            ));
        } else {
            panic!("Expected Case expression, got {expr:?}");
        }
        assert_eq!(s.graph_table.columns.items[0].alias, "category");
    }

    #[test]
    fn test_parse_case_simple() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (
                    CASE n.city WHEN 'Amsterdam' THEN 'NL' WHEN 'Berlin' THEN 'DE' END AS country
                )
            )",
        ));
        let expr = &s.graph_table.columns.items[0].expression;
        if let Expression::Case {
            input,
            whens,
            else_clause,
        } = expr
        {
            // Simple CASE has an input expression
            assert!(
                input.is_some(),
                "simple CASE should have an input expression"
            );
            // Two WHEN clauses
            assert_eq!(whens.len(), 2);
            // First WHEN: 'Amsterdam' -> 'NL'
            assert!(matches!(
                &whens[0].0,
                Expression::Literal(Literal::String(s)) if s == "Amsterdam"
            ));
            assert!(matches!(
                &whens[0].1,
                Expression::Literal(Literal::String(s)) if s == "NL"
            ));
            // Second WHEN: 'Berlin' -> 'DE'
            assert!(matches!(
                &whens[1].0,
                Expression::Literal(Literal::String(s)) if s == "Berlin"
            ));
            assert!(matches!(
                &whens[1].1,
                Expression::Literal(Literal::String(s)) if s == "DE"
            ));
            // No ELSE clause
            assert!(else_clause.is_none());
        } else {
            panic!("Expected Case expression, got {expr:?}");
        }
        assert_eq!(s.graph_table.columns.items[0].alias, "country");
    }

    #[test]
    fn test_parse_case_without_else() {
        let s = select(parse_ok(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (
                    CASE WHEN n.age > 0 THEN 'positive' END AS sign
                )
            )",
        ));
        let expr = &s.graph_table.columns.items[0].expression;
        if let Expression::Case {
            input,
            whens,
            else_clause,
        } = expr
        {
            assert!(input.is_none(), "searched CASE should have no input");
            assert_eq!(whens.len(), 1);
            assert!(
                else_clause.is_none(),
                "CASE without ELSE should produce None"
            );
        } else {
            panic!("Expected Case expression, got {expr:?}");
        }
    }

    #[test]
    fn test_parse_case_nested_in_select() {
        let s = select(parse_ok(
            "SELECT g.person_name, g.tier
             FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (
                    n.name AS person_name,
                    CASE WHEN n.age >= 65 THEN 'retired'
                         WHEN n.age >= 30 THEN 'senior'
                         ELSE 'junior'
                    END AS tier
                )
             ) AS g
             WHERE g.tier = 'senior'
             ORDER BY g.person_name
             LIMIT 10",
        ));
        // Verify the full statement structure
        if let SelectList::Columns(items) = &s.select_list {
            assert_eq!(items.len(), 2);
        } else {
            panic!("Expected explicit SELECT columns");
        }
        assert_eq!(s.table_alias, Some("g".to_string()));
        assert!(s.where_clause.is_some());
        assert!(s.order_by.is_some());
        assert_eq!(s.limit, Some(10));

        // Verify the CASE expression in COLUMNS
        let case_expr = &s.graph_table.columns.items[1].expression;
        if let Expression::Case {
            input,
            whens,
            else_clause,
        } = case_expr
        {
            assert!(input.is_none());
            assert_eq!(whens.len(), 2);
            // First WHEN result: 'retired'
            assert!(matches!(
                &whens[0].1,
                Expression::Literal(Literal::String(s)) if s == "retired"
            ));
            // Second WHEN result: 'senior'
            assert!(matches!(
                &whens[1].1,
                Expression::Literal(Literal::String(s)) if s == "senior"
            ));
            // ELSE: 'junior'
            let else_expr = else_clause.as_ref().expect("should have ELSE");
            assert!(matches!(
                else_expr.as_ref(),
                Expression::Literal(Literal::String(s)) if s == "junior"
            ));
        } else {
            panic!("Expected Case expression, got {case_expr:?}");
        }
        assert_eq!(s.graph_table.columns.items[1].alias, "tier");
    }

    // ==================== Recursion depth limit tests ====================

    #[test]
    fn test_deeply_nested_parentheses_errors_not_stack_overflow() {
        let deep = "(".repeat(300) + "1" + &")".repeat(300);
        let query = format!("SELECT * FROM GRAPH_TABLE (MATCH (n) COLUMNS ({deep} AS val))");
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
    fn test_long_not_and_sign_chains_error_not_stack_overflow() {
        let in_columns =
            |expr: &str| format!("SELECT * FROM GRAPH_TABLE (MATCH (n) COLUMNS ({expr} AS val))");
        for expr in [
            format!("{}TRUE", "NOT ".repeat(50_000)),
            format!("{}1", "- ".repeat(50_000)),
        ] {
            let err = Parser::new(&in_columns(&expr))
                .parse()
                .unwrap_err()
                .to_string();
            assert!(err.contains("nesting depth"), "got: {err}");
        }
        // Short chains are unaffected.
        assert!(Parser::new(&in_columns("NOT NOT TRUE")).parse().is_ok());
        assert!(Parser::new(&in_columns("- - 1")).parse().is_ok());
    }

    #[test]
    fn test_moderate_nesting_succeeds() {
        let depth = 50;
        let deep = "(".repeat(depth) + "1" + &")".repeat(depth);
        let query = format!("SELECT * FROM GRAPH_TABLE (MATCH (n) COLUMNS ({deep} AS val))");
        let mut parser = Parser::new(&query);
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "50 levels of nesting should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_deeply_nested_case_in_columns() {
        let prefix = "CASE WHEN true THEN ".repeat(300);
        let suffix = " END".repeat(300);
        let query =
            format!("SELECT * FROM GRAPH_TABLE (MATCH (n) COLUMNS ({prefix}1{suffix} AS val))");
        let mut parser = Parser::new(&query);
        let result = parser.parse();
        assert!(
            result.is_err(),
            "300 levels of CASE should error, not stack overflow"
        );
    }
}
