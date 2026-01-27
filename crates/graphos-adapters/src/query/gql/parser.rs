//! GQL Parser.

use super::ast::*;
use super::lexer::{Lexer, Token, TokenKind};
use graphos_common::utils::error::{Error, QueryError, QueryErrorKind, Result, SourceSpan};

/// GQL Parser.
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
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
            source: input,
        }
    }

    /// Parses the input into a statement.
    pub fn parse(&mut self) -> Result<Statement> {
        match self.current.kind {
            TokenKind::Match => self.parse_query().map(Statement::Query),
            TokenKind::Insert => self
                .parse_insert()
                .map(|s| Statement::DataModification(DataModificationStatement::Insert(s))),
            TokenKind::Delete => self
                .parse_delete()
                .map(|s| Statement::DataModification(DataModificationStatement::Delete(s))),
            TokenKind::Create => self.parse_create_schema().map(Statement::Schema),
            _ => Err(self.error("Expected MATCH, INSERT, DELETE, or CREATE")),
        }
    }

    fn parse_query(&mut self) -> Result<QueryStatement> {
        let span_start = self.current.span.start;

        // Parse MATCH clauses (including OPTIONAL MATCH)
        let mut match_clauses = Vec::new();
        while self.current.kind == TokenKind::Match || self.current.kind == TokenKind::Optional {
            match_clauses.push(self.parse_match_clause()?);
        }

        // Parse WHERE clause (after all MATCH clauses)
        let where_clause = if self.current.kind == TokenKind::Where {
            Some(self.parse_where_clause()?)
        } else {
            None
        };

        // Parse WITH clauses
        let mut with_clauses = Vec::new();
        while self.current.kind == TokenKind::With {
            with_clauses.push(self.parse_with_clause()?);

            // After WITH, we can have more MATCH clauses
            while self.current.kind == TokenKind::Match || self.current.kind == TokenKind::Optional
            {
                match_clauses.push(self.parse_match_clause()?);
            }
        }

        // Parse RETURN clause
        if self.current.kind != TokenKind::Return {
            return Err(self.error("Expected RETURN"));
        }
        let return_clause = self.parse_return_clause()?;

        Ok(QueryStatement {
            match_clauses,
            where_clause,
            with_clauses,
            return_clause,
            span: Some(SourceSpan::new(span_start, self.current.span.end, 1, 1)),
        })
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
        patterns.push(self.parse_pattern()?);

        while self.current.kind == TokenKind::Comma {
            self.advance();
            patterns.push(self.parse_pattern()?);
        }

        Ok(MatchClause {
            optional,
            patterns,
            span: Some(SourceSpan::new(span_start, self.current.span.end, 1, 1)),
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
            if self.current.kind != TokenKind::Identifier {
                return Err(self.error("Expected label name"));
            }
            labels.push(self.current.text.clone());
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
        // 1. `-[...]->` or `-[:TYPE]->` (direction determined by trailing arrow)
        // 2. `->` or `<-` or `--` (direction determined by leading arrow)

        let (variable, types, direction) = if self.current.kind == TokenKind::Minus {
            // Pattern: -[...]->(target) or -[...]-(target)
            self.advance();

            // Parse [variable:TYPE]
            let (var, edge_types) = if self.current.kind == TokenKind::LBracket {
                self.advance();

                let v = if self.current.kind == TokenKind::Identifier
                    && self.peek_kind() != TokenKind::Colon
                {
                    let name = self.current.text.clone();
                    self.advance();
                    Some(name)
                } else {
                    None
                };

                let mut tps = Vec::new();
                while self.current.kind == TokenKind::Colon {
                    self.advance();
                    if self.current.kind != TokenKind::Identifier {
                        return Err(self.error("Expected edge type"));
                    }
                    tps.push(self.current.text.clone());
                    self.advance();
                }

                self.expect(TokenKind::RBracket)?;
                (v, tps)
            } else {
                (None, Vec::new())
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

            (var, edge_types, dir)
        } else if self.current.kind == TokenKind::LeftArrow {
            // Pattern: <-[...]-(target)
            self.advance();

            let (var, edge_types) = if self.current.kind == TokenKind::LBracket {
                self.advance();

                let v = if self.current.kind == TokenKind::Identifier
                    && self.peek_kind() != TokenKind::Colon
                {
                    let name = self.current.text.clone();
                    self.advance();
                    Some(name)
                } else {
                    None
                };

                let mut tps = Vec::new();
                while self.current.kind == TokenKind::Colon {
                    self.advance();
                    if self.current.kind != TokenKind::Identifier {
                        return Err(self.error("Expected edge type"));
                    }
                    tps.push(self.current.text.clone());
                    self.advance();
                }

                self.expect(TokenKind::RBracket)?;
                (v, tps)
            } else {
                (None, Vec::new())
            };

            // Consume trailing -
            if self.current.kind == TokenKind::Minus {
                self.advance();
            }

            (var, edge_types, EdgeDirection::Incoming)
        } else if self.current.kind == TokenKind::Arrow {
            // Simple ->
            self.advance();
            (None, Vec::new(), EdgeDirection::Outgoing)
        } else if self.current.kind == TokenKind::DoubleDash {
            // Simple --
            self.advance();
            (None, Vec::new(), EdgeDirection::Undirected)
        } else {
            return Err(self.error("Expected edge pattern"));
        };

        let target = self.parse_node_pattern()?;

        Ok(EdgePattern {
            variable,
            types,
            direction,
            target,
            span: None,
        })
    }

    fn parse_where_clause(&mut self) -> Result<WhereClause> {
        self.expect(TokenKind::Where)?;
        let expression = self.parse_expression()?;

        Ok(WhereClause {
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
            if self.current.kind != TokenKind::Identifier {
                return Err(self.error("Expected alias name"));
            }
            let name = self.current.text.clone();
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
            Ok(Expression::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            })
        } else {
            Ok(left)
        }
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
            TokenKind::Identifier => {
                let name = self.current.text.clone();
                self.advance();

                if self.current.kind == TokenKind::Dot {
                    self.advance();
                    if self.current.kind != TokenKind::Identifier {
                        return Err(self.error("Expected property name"));
                    }
                    let property = self.current.text.clone();
                    self.advance();
                    Ok(Expression::PropertyAccess {
                        variable: name,
                        property,
                    })
                } else if self.current.kind == TokenKind::LParen {
                    // Function call
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
                    Ok(Expression::FunctionCall { name, args })
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
            _ => Err(self.error("Expected expression")),
        }
    }

    fn parse_property_map(&mut self) -> Result<Vec<(String, Expression)>> {
        self.expect(TokenKind::LBrace)?;

        let mut properties = Vec::new();

        if self.current.kind != TokenKind::RBrace {
            loop {
                if self.current.kind != TokenKind::Identifier {
                    return Err(self.error("Expected property name"));
                }
                let key = self.current.text.clone();
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

                if self.current.kind != TokenKind::Identifier {
                    return Err(self.error("Expected type name"));
                }
                let name = self.current.text.clone();
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

                if self.current.kind != TokenKind::Identifier {
                    return Err(self.error("Expected type name"));
                }
                let name = self.current.text.clone();
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
                if self.current.kind != TokenKind::Identifier {
                    return Err(self.error("Expected property name"));
                }
                let name = self.current.text.clone();
                self.advance();

                if self.current.kind != TokenKind::Identifier {
                    return Err(self.error("Expected type name"));
                }
                let data_type = self.current.text.clone();
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
        self.current = self.lexer.next_token();
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
        // Simple lookahead by creating a temporary lexer
        // In a production implementation, we'd buffer tokens
        self.current.kind
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
}
