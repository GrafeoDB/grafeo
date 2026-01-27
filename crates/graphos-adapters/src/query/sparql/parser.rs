//! SPARQL Parser.
//!
//! Implements a recursive descent parser for SPARQL 1.1 Query Language.

use super::ast::*;
use super::lexer::{Lexer, Token, TokenKind};
use graphos_common::utils::error::{Error, QueryError, QueryErrorKind, Result};

/// SPARQL Parser.
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
    source: &'a str,
}

impl<'a> Parser<'a> {
    /// Creates a new parser for the given query string.
    pub fn new(source: &'a str) -> Self {
        let mut lexer = Lexer::new(source);
        let current = lexer.next_token();
        Self {
            lexer,
            current,
            source,
        }
    }

    /// Parses the entire query.
    pub fn parse(&mut self) -> Result<Query> {
        let base = self.parse_base()?;
        let prefixes = self.parse_prefixes()?;
        let query_form = self.parse_query_form()?;

        if self.current.kind != TokenKind::Eof {
            return Err(self.error(&format!(
                "unexpected token '{}' after query",
                self.current.text
            )));
        }

        Ok(Query {
            base,
            prefixes,
            query_form,
        })
    }

    // ==================== Prologue ====================

    fn parse_base(&mut self) -> Result<Option<Iri>> {
        if self.current.kind == TokenKind::Base {
            self.advance();
            let iri = self.expect_iri()?;
            Ok(Some(iri))
        } else {
            Ok(None)
        }
    }

    fn parse_prefixes(&mut self) -> Result<Vec<PrefixDeclaration>> {
        let mut prefixes = Vec::new();
        while self.current.kind == TokenKind::Prefix {
            self.advance();

            // Parse prefix name (can be empty for default prefix)
            let prefix = if self.current.kind == TokenKind::PrefixedName {
                let text = self.current.text.clone();
                self.advance();
                // Remove trailing colon if present
                text.trim_end_matches(':').to_string()
            } else if self.current.kind == TokenKind::Colon {
                self.advance();
                String::new() // Default prefix
            } else {
                return Err(self.error("expected prefix name or ':'"));
            };

            let namespace = self.expect_iri()?;
            prefixes.push(PrefixDeclaration { prefix, namespace });
        }
        Ok(prefixes)
    }

    // ==================== Query Forms ====================

    fn parse_query_form(&mut self) -> Result<QueryForm> {
        match self.current.kind {
            TokenKind::Select => Ok(QueryForm::Select(self.parse_select_query()?)),
            TokenKind::Construct => Ok(QueryForm::Construct(self.parse_construct_query()?)),
            TokenKind::Ask => Ok(QueryForm::Ask(self.parse_ask_query()?)),
            TokenKind::Describe => Ok(QueryForm::Describe(self.parse_describe_query()?)),
            _ => Err(self.error("expected SELECT, CONSTRUCT, ASK, or DESCRIBE")),
        }
    }

    fn parse_select_query(&mut self) -> Result<SelectQuery> {
        self.expect(TokenKind::Select)?;

        let modifier = self.parse_select_modifier();
        let projection = self.parse_projection()?;
        let dataset = self.parse_dataset_clause()?;
        let where_clause = self.parse_where_clause()?;
        let solution_modifiers = self.parse_solution_modifiers()?;

        Ok(SelectQuery {
            modifier,
            projection,
            dataset,
            where_clause,
            solution_modifiers,
        })
    }

    fn parse_construct_query(&mut self) -> Result<ConstructQuery> {
        self.expect(TokenKind::Construct)?;

        let template = self.parse_construct_template()?;
        let dataset = self.parse_dataset_clause()?;
        let where_clause = self.parse_where_clause()?;
        let solution_modifiers = self.parse_solution_modifiers()?;

        Ok(ConstructQuery {
            template,
            dataset,
            where_clause,
            solution_modifiers,
        })
    }

    fn parse_ask_query(&mut self) -> Result<AskQuery> {
        self.expect(TokenKind::Ask)?;

        let dataset = self.parse_dataset_clause()?;
        let where_clause = self.parse_where_clause()?;

        Ok(AskQuery {
            dataset,
            where_clause,
        })
    }

    fn parse_describe_query(&mut self) -> Result<DescribeQuery> {
        self.expect(TokenKind::Describe)?;

        let resources = self.parse_describe_resources()?;
        let dataset = self.parse_dataset_clause()?;

        let where_clause = if self.current.kind == TokenKind::Where
            || self.current.kind == TokenKind::LeftBrace
        {
            Some(self.parse_where_clause()?)
        } else {
            None
        };

        Ok(DescribeQuery {
            resources,
            dataset,
            where_clause,
        })
    }

    fn parse_describe_resources(&mut self) -> Result<Vec<VariableOrIri>> {
        let mut resources = Vec::new();

        if self.current.kind == TokenKind::Star {
            // DESCRIBE * - will be handled specially
            self.advance();
            return Ok(resources);
        }

        while matches!(
            self.current.kind,
            TokenKind::Variable | TokenKind::Iri | TokenKind::PrefixedName
        ) {
            resources.push(self.parse_variable_or_iri()?);
        }

        if resources.is_empty() {
            return Err(self.error("expected variable or IRI to describe"));
        }

        Ok(resources)
    }

    // ==================== Select Clause ====================

    fn parse_select_modifier(&mut self) -> SelectModifier {
        match self.current.kind {
            TokenKind::Distinct => {
                self.advance();
                SelectModifier::Distinct
            }
            TokenKind::Reduced => {
                self.advance();
                SelectModifier::Reduced
            }
            _ => SelectModifier::None,
        }
    }

    fn parse_projection(&mut self) -> Result<Projection> {
        if self.current.kind == TokenKind::Star {
            self.advance();
            return Ok(Projection::Wildcard);
        }

        let mut variables = Vec::new();
        while self.current.kind == TokenKind::Variable
            || self.current.kind == TokenKind::LeftParen
        {
            variables.push(self.parse_projection_variable()?);
        }

        if variables.is_empty() {
            return Err(self.error("expected '*' or variable list in SELECT"));
        }

        Ok(Projection::Variables(variables))
    }

    fn parse_projection_variable(&mut self) -> Result<ProjectionVariable> {
        if self.current.kind == TokenKind::LeftParen {
            // Expression with alias: (expr AS ?var)
            self.advance();
            let expression = self.parse_expression()?;
            self.expect(TokenKind::As)?;
            let alias = self.expect_variable_name()?;
            self.expect(TokenKind::RightParen)?;

            Ok(ProjectionVariable {
                expression,
                alias: Some(alias),
            })
        } else {
            // Simple variable
            let var_name = self.expect_variable_name()?;
            Ok(ProjectionVariable {
                expression: Expression::Variable(var_name),
                alias: None,
            })
        }
    }

    // ==================== Dataset Clause ====================

    fn parse_dataset_clause(&mut self) -> Result<Option<DatasetClause>> {
        let mut default_graphs = Vec::new();
        let mut named_graphs = Vec::new();

        while self.current.kind == TokenKind::From {
            self.advance();

            if self.current.kind == TokenKind::Named {
                self.advance();
                named_graphs.push(self.expect_iri()?);
            } else {
                default_graphs.push(self.expect_iri()?);
            }
        }

        if default_graphs.is_empty() && named_graphs.is_empty() {
            Ok(None)
        } else {
            Ok(Some(DatasetClause {
                default_graphs,
                named_graphs,
            }))
        }
    }

    // ==================== WHERE Clause ====================

    fn parse_where_clause(&mut self) -> Result<GraphPattern> {
        // WHERE is optional
        if self.current.kind == TokenKind::Where {
            self.advance();
        }

        self.parse_group_graph_pattern()
    }

    fn parse_group_graph_pattern(&mut self) -> Result<GraphPattern> {
        self.expect(TokenKind::LeftBrace)?;

        let mut patterns = Vec::new();

        while self.current.kind != TokenKind::RightBrace {
            if self.current.kind == TokenKind::Eof {
                return Err(self.error("unexpected end of input in graph pattern"));
            }

            patterns.push(self.parse_graph_pattern_element()?);
        }

        self.expect(TokenKind::RightBrace)?;

        if patterns.len() == 1 {
            Ok(patterns.into_iter().next().unwrap())
        } else {
            Ok(GraphPattern::Group(patterns))
        }
    }

    fn parse_graph_pattern_element(&mut self) -> Result<GraphPattern> {
        match self.current.kind {
            TokenKind::Optional => {
                self.advance();
                let pattern = self.parse_group_graph_pattern()?;
                Ok(GraphPattern::Optional(Box::new(pattern)))
            }
            TokenKind::Minus => {
                self.advance();
                let pattern = self.parse_group_graph_pattern()?;
                Ok(GraphPattern::Minus(Box::new(pattern)))
            }
            TokenKind::Graph => {
                self.advance();
                let graph = self.parse_variable_or_iri()?;
                let pattern = self.parse_group_graph_pattern()?;
                Ok(GraphPattern::NamedGraph {
                    graph,
                    pattern: Box::new(pattern),
                })
            }
            TokenKind::Service => {
                self.advance();
                let silent = if self.current.kind == TokenKind::Silent {
                    self.advance();
                    true
                } else {
                    false
                };
                let endpoint = self.parse_variable_or_iri()?;
                let pattern = self.parse_group_graph_pattern()?;
                Ok(GraphPattern::Service {
                    silent,
                    endpoint,
                    pattern: Box::new(pattern),
                })
            }
            TokenKind::Filter => {
                self.advance();
                let expression = self.parse_constraint()?;
                Ok(GraphPattern::Filter(expression))
            }
            TokenKind::Bind => {
                self.advance();
                self.expect(TokenKind::LeftParen)?;
                let expression = self.parse_expression()?;
                self.expect(TokenKind::As)?;
                let variable = self.expect_variable_name()?;
                self.expect(TokenKind::RightParen)?;
                Ok(GraphPattern::Bind {
                    expression,
                    variable,
                })
            }
            TokenKind::Values => {
                self.parse_inline_data()
            }
            TokenKind::LeftBrace => {
                // Nested group or subquery
                let pattern = self.parse_group_or_subquery()?;
                // Check for UNION
                self.parse_union_continuation(pattern)
            }
            TokenKind::Select => {
                // Subquery
                let subquery = self.parse_select_query()?;
                Ok(GraphPattern::SubSelect(Box::new(subquery)))
            }
            _ => {
                // Triple patterns
                let triples = self.parse_triples_block()?;
                Ok(GraphPattern::Basic(triples))
            }
        }
    }

    fn parse_group_or_subquery(&mut self) -> Result<GraphPattern> {
        // Save position to potentially backtrack
        let saved_kind = self.current.kind.clone();

        if saved_kind == TokenKind::LeftBrace {
            self.advance();

            if self.current.kind == TokenKind::Select {
                // It's a subquery
                let subquery = self.parse_select_query()?;
                self.expect(TokenKind::RightBrace)?;
                return Ok(GraphPattern::SubSelect(Box::new(subquery)));
            }

            // Regular group pattern - parse contents
            let mut patterns = Vec::new();
            while self.current.kind != TokenKind::RightBrace {
                if self.current.kind == TokenKind::Eof {
                    return Err(self.error("unexpected end of input in graph pattern"));
                }
                patterns.push(self.parse_graph_pattern_element()?);
            }
            self.expect(TokenKind::RightBrace)?;

            if patterns.len() == 1 {
                Ok(patterns.into_iter().next().unwrap())
            } else {
                Ok(GraphPattern::Group(patterns))
            }
        } else {
            Err(self.error("expected '{'"))
        }
    }

    fn parse_union_continuation(&mut self, first: GraphPattern) -> Result<GraphPattern> {
        if self.current.kind == TokenKind::Union {
            let mut alternatives = vec![first];
            while self.current.kind == TokenKind::Union {
                self.advance();
                let pattern = self.parse_group_graph_pattern()?;
                alternatives.push(pattern);
            }
            Ok(GraphPattern::Union(alternatives))
        } else {
            Ok(first)
        }
    }

    fn parse_constraint(&mut self) -> Result<Expression> {
        if self.current.kind == TokenKind::LeftParen {
            self.advance();
            let expr = self.parse_expression()?;
            self.expect(TokenKind::RightParen)?;
            Ok(expr)
        } else {
            self.parse_built_in_call()
        }
    }

    fn parse_inline_data(&mut self) -> Result<GraphPattern> {
        self.expect(TokenKind::Values)?;

        let mut variables = Vec::new();

        // Parse variable list
        if self.current.kind == TokenKind::LeftParen {
            self.advance();
            while self.current.kind == TokenKind::Variable {
                variables.push(self.expect_variable_name()?);
            }
            self.expect(TokenKind::RightParen)?;
        } else if self.current.kind == TokenKind::Variable {
            variables.push(self.expect_variable_name()?);
        }

        // Parse data block
        self.expect(TokenKind::LeftBrace)?;
        let mut values = Vec::new();

        while self.current.kind != TokenKind::RightBrace {
            if variables.len() == 1 {
                // Single variable - values listed directly
                let value = self.parse_data_value()?;
                values.push(vec![value]);
            } else {
                // Multiple variables - parenthesized rows
                self.expect(TokenKind::LeftParen)?;
                let mut row = Vec::new();
                for _ in 0..variables.len() {
                    row.push(self.parse_data_value()?);
                }
                self.expect(TokenKind::RightParen)?;
                values.push(row);
            }
        }

        self.expect(TokenKind::RightBrace)?;

        Ok(GraphPattern::InlineData(InlineDataClause {
            variables,
            values,
        }))
    }

    fn parse_data_value(&mut self) -> Result<Option<DataValue>> {
        match self.current.kind {
            TokenKind::Undef => {
                self.advance();
                Ok(None)
            }
            TokenKind::Iri => {
                let iri = self.parse_iri()?;
                Ok(Some(DataValue::Iri(iri)))
            }
            TokenKind::PrefixedName => {
                let iri = self.parse_prefixed_iri()?;
                Ok(Some(DataValue::Iri(iri)))
            }
            TokenKind::String | TokenKind::LongString => {
                let literal = self.parse_literal()?;
                Ok(Some(DataValue::Literal(literal)))
            }
            TokenKind::Integer | TokenKind::Decimal | TokenKind::Double => {
                let literal = self.parse_numeric_literal()?;
                Ok(Some(DataValue::Literal(literal)))
            }
            TokenKind::True | TokenKind::False => {
                let literal = self.parse_boolean_literal()?;
                Ok(Some(DataValue::Literal(literal)))
            }
            _ => Err(self.error("expected data value or UNDEF")),
        }
    }

    // ==================== Triple Patterns ====================

    fn parse_triples_block(&mut self) -> Result<Vec<TriplePattern>> {
        let mut triples = Vec::new();

        loop {
            if !self.is_triple_start() {
                break;
            }

            self.parse_triples_same_subject(&mut triples)?;

            // Optional trailing dot
            if self.current.kind == TokenKind::Dot {
                self.advance();
            }
        }

        Ok(triples)
    }

    fn is_triple_start(&self) -> bool {
        matches!(
            self.current.kind,
            TokenKind::Variable
                | TokenKind::Iri
                | TokenKind::PrefixedName
                | TokenKind::BlankNodeLabel
                | TokenKind::LeftBracket
                | TokenKind::LeftParen
                | TokenKind::String
                | TokenKind::LongString
                | TokenKind::Integer
                | TokenKind::Decimal
                | TokenKind::Double
                | TokenKind::True
                | TokenKind::False
        )
    }

    fn parse_triples_same_subject(&mut self, triples: &mut Vec<TriplePattern>) -> Result<()> {
        let subject = self.parse_var_or_term()?;
        self.parse_property_list_not_empty(&subject, triples)?;
        Ok(())
    }

    fn parse_property_list_not_empty(
        &mut self,
        subject: &TripleTerm,
        triples: &mut Vec<TriplePattern>,
    ) -> Result<()> {
        loop {
            let predicate = self.parse_verb()?;
            self.parse_object_list(subject, &predicate, triples)?;

            if self.current.kind == TokenKind::Semicolon {
                self.advance();
                // Check if there's another predicate-object list
                if !self.is_verb() {
                    break;
                }
            } else {
                break;
            }
        }
        Ok(())
    }

    fn is_verb(&self) -> bool {
        matches!(
            self.current.kind,
            TokenKind::Variable
                | TokenKind::Iri
                | TokenKind::PrefixedName
                | TokenKind::A
                | TokenKind::Caret
                | TokenKind::Bang
                | TokenKind::LeftParen
        )
    }

    fn parse_verb(&mut self) -> Result<PropertyPath> {
        if self.current.kind == TokenKind::A {
            self.advance();
            Ok(PropertyPath::RdfType)
        } else if self.current.kind == TokenKind::Variable {
            let name = self.expect_variable_name()?;
            Ok(PropertyPath::Variable(name))
        } else {
            self.parse_property_path()
        }
    }

    fn parse_object_list(
        &mut self,
        subject: &TripleTerm,
        predicate: &PropertyPath,
        triples: &mut Vec<TriplePattern>,
    ) -> Result<()> {
        loop {
            let object = self.parse_object(triples)?;
            triples.push(TriplePattern {
                subject: subject.clone(),
                predicate: predicate.clone(),
                object,
            });

            if self.current.kind == TokenKind::Comma {
                self.advance();
            } else {
                break;
            }
        }
        Ok(())
    }

    fn parse_object(&mut self, triples: &mut Vec<TriplePattern>) -> Result<TripleTerm> {
        if self.current.kind == TokenKind::LeftBracket {
            // Blank node with property list
            self.advance();

            if self.current.kind == TokenKind::RightBracket {
                self.advance();
                return Ok(TripleTerm::BlankNode(BlankNode::Anonymous(vec![])));
            }

            // Create an anonymous blank node as subject
            let blank_subject = TripleTerm::BlankNode(BlankNode::Labeled(format!("_:b{}", triples.len())));
            self.parse_property_list_not_empty(&blank_subject, triples)?;
            self.expect(TokenKind::RightBracket)?;

            Ok(blank_subject)
        } else {
            self.parse_var_or_term()
        }
    }

    fn parse_var_or_term(&mut self) -> Result<TripleTerm> {
        match self.current.kind {
            TokenKind::Variable => {
                let name = self.expect_variable_name()?;
                Ok(TripleTerm::Variable(name))
            }
            TokenKind::Iri => {
                let iri = self.parse_iri()?;
                Ok(TripleTerm::Iri(iri))
            }
            TokenKind::PrefixedName => {
                let iri = self.parse_prefixed_iri()?;
                Ok(TripleTerm::Iri(iri))
            }
            TokenKind::BlankNodeLabel => {
                let label = self.current.text[2..].to_string(); // Remove _: prefix
                self.advance();
                Ok(TripleTerm::BlankNode(BlankNode::Labeled(label)))
            }
            TokenKind::LeftBracket => {
                self.advance();
                if self.current.kind == TokenKind::RightBracket {
                    self.advance();
                    Ok(TripleTerm::BlankNode(BlankNode::Anonymous(vec![])))
                } else {
                    // Property list - simplified handling
                    let mut props = Vec::new();
                    while self.current.kind != TokenKind::RightBracket {
                        let pred = self.parse_verb()?;
                        let obj = self.parse_var_or_term()?;
                        props.push((pred, obj));
                        if self.current.kind == TokenKind::Semicolon {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    self.expect(TokenKind::RightBracket)?;
                    Ok(TripleTerm::BlankNode(BlankNode::Anonymous(props)))
                }
            }
            TokenKind::String | TokenKind::LongString => {
                let literal = self.parse_literal()?;
                Ok(TripleTerm::Literal(literal))
            }
            TokenKind::Integer | TokenKind::Decimal | TokenKind::Double => {
                let literal = self.parse_numeric_literal()?;
                Ok(TripleTerm::Literal(literal))
            }
            TokenKind::True | TokenKind::False => {
                let literal = self.parse_boolean_literal()?;
                Ok(TripleTerm::Literal(literal))
            }
            _ => Err(self.error("expected variable, IRI, blank node, or literal")),
        }
    }

    // ==================== Property Paths ====================

    fn parse_property_path(&mut self) -> Result<PropertyPath> {
        self.parse_path_alternative()
    }

    fn parse_path_alternative(&mut self) -> Result<PropertyPath> {
        let mut path = self.parse_path_sequence()?;

        if self.current.kind == TokenKind::Pipe {
            let mut alternatives = vec![path];
            while self.current.kind == TokenKind::Pipe {
                self.advance();
                alternatives.push(self.parse_path_sequence()?);
            }
            path = PropertyPath::Alternative(alternatives);
        }

        Ok(path)
    }

    fn parse_path_sequence(&mut self) -> Result<PropertyPath> {
        let mut path = self.parse_path_elt_or_inverse()?;

        if self.current.kind == TokenKind::Slash {
            let mut sequence = vec![path];
            while self.current.kind == TokenKind::Slash {
                self.advance();
                sequence.push(self.parse_path_elt_or_inverse()?);
            }
            path = PropertyPath::Sequence(sequence);
        }

        Ok(path)
    }

    fn parse_path_elt_or_inverse(&mut self) -> Result<PropertyPath> {
        if self.current.kind == TokenKind::Caret {
            self.advance();
            let path = self.parse_path_elt()?;
            Ok(PropertyPath::Inverse(Box::new(path)))
        } else {
            self.parse_path_elt()
        }
    }

    fn parse_path_elt(&mut self) -> Result<PropertyPath> {
        let mut path = self.parse_path_primary()?;

        // Path modifiers: * + ?
        match self.current.kind {
            TokenKind::Star => {
                self.advance();
                path = PropertyPath::ZeroOrMore(Box::new(path));
            }
            TokenKind::Plus => {
                self.advance();
                path = PropertyPath::OneOrMore(Box::new(path));
            }
            TokenKind::QuestionMark => {
                self.advance();
                path = PropertyPath::ZeroOrOne(Box::new(path));
            }
            _ => {}
        }

        Ok(path)
    }

    fn parse_path_primary(&mut self) -> Result<PropertyPath> {
        match self.current.kind {
            TokenKind::Iri => {
                let iri = self.parse_iri()?;
                Ok(PropertyPath::Predicate(iri))
            }
            TokenKind::PrefixedName => {
                let iri = self.parse_prefixed_iri()?;
                Ok(PropertyPath::Predicate(iri))
            }
            TokenKind::A => {
                self.advance();
                Ok(PropertyPath::RdfType)
            }
            TokenKind::Bang => {
                self.advance();
                self.parse_path_negation()
            }
            TokenKind::LeftParen => {
                self.advance();
                let path = self.parse_property_path()?;
                self.expect(TokenKind::RightParen)?;
                Ok(path)
            }
            _ => Err(self.error("expected property path")),
        }
    }

    fn parse_path_negation(&mut self) -> Result<PropertyPath> {
        if self.current.kind == TokenKind::LeftParen {
            self.advance();
            let mut iris = Vec::new();

            if self.current.kind != TokenKind::RightParen {
                iris.push(self.expect_iri_or_prefixed()?);
                while self.current.kind == TokenKind::Pipe {
                    self.advance();
                    iris.push(self.expect_iri_or_prefixed()?);
                }
            }

            self.expect(TokenKind::RightParen)?;
            Ok(PropertyPath::Negation(iris))
        } else {
            let iri = self.expect_iri_or_prefixed()?;
            Ok(PropertyPath::Negation(vec![iri]))
        }
    }

    // ==================== CONSTRUCT Template ====================

    fn parse_construct_template(&mut self) -> Result<Vec<TriplePattern>> {
        self.expect(TokenKind::LeftBrace)?;
        let mut triples = Vec::new();

        while self.current.kind != TokenKind::RightBrace {
            if self.current.kind == TokenKind::Eof {
                return Err(self.error("unexpected end of input in CONSTRUCT template"));
            }
            self.parse_triples_same_subject(&mut triples)?;
            if self.current.kind == TokenKind::Dot {
                self.advance();
            }
        }

        self.expect(TokenKind::RightBrace)?;
        Ok(triples)
    }

    // ==================== Solution Modifiers ====================

    fn parse_solution_modifiers(&mut self) -> Result<SolutionModifiers> {
        let group_by = self.parse_group_by()?;
        let having = self.parse_having()?;
        let order_by = self.parse_order_by()?;
        let (limit, offset) = self.parse_limit_offset()?;

        Ok(SolutionModifiers {
            group_by,
            having,
            order_by,
            limit,
            offset,
        })
    }

    fn parse_group_by(&mut self) -> Result<Option<Vec<GroupCondition>>> {
        if self.current.kind != TokenKind::Group {
            return Ok(None);
        }
        self.advance();
        self.expect(TokenKind::By)?;

        let mut conditions = Vec::new();
        while self.current.kind == TokenKind::Variable
            || self.current.kind == TokenKind::LeftParen
        {
            conditions.push(self.parse_group_condition()?);
        }

        if conditions.is_empty() {
            return Err(self.error("expected GROUP BY condition"));
        }

        Ok(Some(conditions))
    }

    fn parse_group_condition(&mut self) -> Result<GroupCondition> {
        if self.current.kind == TokenKind::LeftParen {
            self.advance();
            let expression = self.parse_expression()?;

            let alias = if self.current.kind == TokenKind::As {
                self.advance();
                Some(self.expect_variable_name()?)
            } else {
                None
            };

            self.expect(TokenKind::RightParen)?;
            Ok(GroupCondition::Expression { expression, alias })
        } else if self.current.kind == TokenKind::Variable {
            let name = self.expect_variable_name()?;
            Ok(GroupCondition::Variable(name))
        } else {
            let expr = self.parse_built_in_call()?;
            Ok(GroupCondition::BuiltInCall(expr))
        }
    }

    fn parse_having(&mut self) -> Result<Option<Expression>> {
        if self.current.kind != TokenKind::Having {
            return Ok(None);
        }
        self.advance();
        Ok(Some(self.parse_constraint()?))
    }

    fn parse_order_by(&mut self) -> Result<Option<Vec<OrderCondition>>> {
        if self.current.kind != TokenKind::Order {
            return Ok(None);
        }
        self.advance();
        self.expect(TokenKind::By)?;

        let mut conditions = Vec::new();
        loop {
            let (expression, direction) = if self.current.kind == TokenKind::Asc {
                self.advance();
                self.expect(TokenKind::LeftParen)?;
                let expr = self.parse_expression()?;
                self.expect(TokenKind::RightParen)?;
                (expr, SortDirection::Ascending)
            } else if self.current.kind == TokenKind::Desc {
                self.advance();
                self.expect(TokenKind::LeftParen)?;
                let expr = self.parse_expression()?;
                self.expect(TokenKind::RightParen)?;
                (expr, SortDirection::Descending)
            } else if self.current.kind == TokenKind::Variable {
                let name = self.expect_variable_name()?;
                (Expression::Variable(name), SortDirection::Ascending)
            } else if self.current.kind == TokenKind::LeftParen {
                self.advance();
                let expr = self.parse_expression()?;
                self.expect(TokenKind::RightParen)?;
                (expr, SortDirection::Ascending)
            } else {
                break;
            };

            conditions.push(OrderCondition {
                expression,
                direction,
            });
        }

        if conditions.is_empty() {
            return Err(self.error("expected ORDER BY condition"));
        }

        Ok(Some(conditions))
    }

    fn parse_limit_offset(&mut self) -> Result<(Option<u64>, Option<u64>)> {
        let mut limit = None;
        let mut offset = None;

        // Can appear in either order
        for _ in 0..2 {
            if self.current.kind == TokenKind::Limit {
                self.advance();
                limit = Some(self.expect_integer()?);
            } else if self.current.kind == TokenKind::Offset {
                self.advance();
                offset = Some(self.expect_integer()?);
            }
        }

        Ok((limit, offset))
    }

    // ==================== Expressions ====================

    fn parse_expression(&mut self) -> Result<Expression> {
        self.parse_conditional_or_expression()
    }

    fn parse_conditional_or_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_conditional_and_expression()?;

        while self.current.kind == TokenKind::OrOp {
            self.advance();
            let right = self.parse_conditional_and_expression()?;
            expr = Expression::Binary {
                left: Box::new(expr),
                operator: BinaryOperator::Or,
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_conditional_and_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_value_logical()?;

        while self.current.kind == TokenKind::AndOp {
            self.advance();
            let right = self.parse_value_logical()?;
            expr = Expression::Binary {
                left: Box::new(expr),
                operator: BinaryOperator::And,
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_value_logical(&mut self) -> Result<Expression> {
        self.parse_relational_expression()
    }

    fn parse_relational_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_numeric_expression()?;

        let operator = match self.current.kind {
            TokenKind::Equals => Some(BinaryOperator::Equal),
            TokenKind::NotEquals => Some(BinaryOperator::NotEqual),
            TokenKind::LessThan => Some(BinaryOperator::LessThan),
            TokenKind::LessOrEqual => Some(BinaryOperator::LessOrEqual),
            TokenKind::GreaterThan => Some(BinaryOperator::GreaterThan),
            TokenKind::GreaterOrEqual => Some(BinaryOperator::GreaterOrEqual),
            TokenKind::In => {
                self.advance();
                let list = self.parse_expression_list()?;
                return Ok(Expression::In {
                    expression: Box::new(expr),
                    list,
                });
            }
            TokenKind::Not => {
                self.advance();
                self.expect(TokenKind::In)?;
                let list = self.parse_expression_list()?;
                return Ok(Expression::NotIn {
                    expression: Box::new(expr),
                    list,
                });
            }
            _ => None,
        };

        if let Some(op) = operator {
            self.advance();
            let right = self.parse_numeric_expression()?;
            expr = Expression::Binary {
                left: Box::new(expr),
                operator: op,
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_numeric_expression(&mut self) -> Result<Expression> {
        self.parse_additive_expression()
    }

    fn parse_additive_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_multiplicative_expression()?;

        loop {
            let operator = match self.current.kind {
                TokenKind::Plus => BinaryOperator::Add,
                TokenKind::MinusOp => BinaryOperator::Subtract,
                _ => break,
            };
            self.advance();
            let right = self.parse_multiplicative_expression()?;
            expr = Expression::Binary {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_multiplicative_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_unary_expression()?;

        loop {
            let operator = match self.current.kind {
                TokenKind::Star => BinaryOperator::Multiply,
                TokenKind::Slash => BinaryOperator::Divide,
                _ => break,
            };
            self.advance();
            let right = self.parse_unary_expression()?;
            expr = Expression::Binary {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_unary_expression(&mut self) -> Result<Expression> {
        match self.current.kind {
            TokenKind::Bang => {
                self.advance();
                let operand = self.parse_primary_expression()?;
                Ok(Expression::Unary {
                    operator: UnaryOperator::Not,
                    operand: Box::new(operand),
                })
            }
            TokenKind::Plus => {
                self.advance();
                let operand = self.parse_primary_expression()?;
                Ok(Expression::Unary {
                    operator: UnaryOperator::Plus,
                    operand: Box::new(operand),
                })
            }
            TokenKind::MinusOp => {
                self.advance();
                let operand = self.parse_primary_expression()?;
                Ok(Expression::Unary {
                    operator: UnaryOperator::Minus,
                    operand: Box::new(operand),
                })
            }
            _ => self.parse_primary_expression(),
        }
    }

    fn parse_primary_expression(&mut self) -> Result<Expression> {
        match self.current.kind {
            TokenKind::LeftParen => {
                self.advance();
                let expr = self.parse_expression()?;
                self.expect(TokenKind::RightParen)?;
                Ok(Expression::Bracketed(Box::new(expr)))
            }
            TokenKind::Variable => {
                let name = self.expect_variable_name()?;
                Ok(Expression::Variable(name))
            }
            TokenKind::Iri => {
                let iri = self.parse_iri()?;
                // Could be IRI or function call
                if self.current.kind == TokenKind::LeftParen {
                    self.parse_function_call_with_iri(iri)
                } else {
                    Ok(Expression::Iri(iri))
                }
            }
            TokenKind::PrefixedName => {
                let iri = self.parse_prefixed_iri()?;
                if self.current.kind == TokenKind::LeftParen {
                    self.parse_function_call_with_iri(iri)
                } else {
                    Ok(Expression::Iri(iri))
                }
            }
            TokenKind::String | TokenKind::LongString => {
                let literal = self.parse_literal()?;
                Ok(Expression::Literal(literal))
            }
            TokenKind::Integer | TokenKind::Decimal | TokenKind::Double => {
                let literal = self.parse_numeric_literal()?;
                Ok(Expression::Literal(literal))
            }
            TokenKind::True | TokenKind::False => {
                let literal = self.parse_boolean_literal()?;
                Ok(Expression::Literal(literal))
            }
            TokenKind::Not => {
                self.advance();
                self.expect(TokenKind::Exists)?;
                let pattern = self.parse_group_graph_pattern()?;
                Ok(Expression::NotExists(Box::new(pattern)))
            }
            TokenKind::Exists => {
                self.advance();
                let pattern = self.parse_group_graph_pattern()?;
                Ok(Expression::Exists(Box::new(pattern)))
            }
            // Aggregates and built-in functions
            _ => self.parse_built_in_call(),
        }
    }

    fn parse_built_in_call(&mut self) -> Result<Expression> {
        match self.current.kind {
            // Aggregates
            TokenKind::Count => self.parse_count_aggregate(),
            TokenKind::Sum => self.parse_aggregate(|e| AggregateExpression::Sum {
                distinct: false,
                expression: Box::new(e),
            }),
            TokenKind::Avg => self.parse_aggregate(|e| AggregateExpression::Average {
                distinct: false,
                expression: Box::new(e),
            }),
            TokenKind::Min => self.parse_aggregate(|e| AggregateExpression::Minimum {
                expression: Box::new(e),
            }),
            TokenKind::Max => self.parse_aggregate(|e| AggregateExpression::Maximum {
                expression: Box::new(e),
            }),
            TokenKind::Sample => self.parse_aggregate(|e| AggregateExpression::Sample {
                expression: Box::new(e),
            }),
            TokenKind::GroupConcat => self.parse_group_concat(),

            // String functions
            _ if self.is_built_in_function() => self.parse_built_in_function(),

            _ => Err(self.error("expected expression")),
        }
    }

    fn is_built_in_function(&self) -> bool {
        // Check for keywords that can be function names
        matches!(
            self.current.text.to_uppercase().as_str(),
            "STR" | "LANG" | "LANGMATCHES" | "DATATYPE" | "BOUND" | "IRI" | "URI"
                | "BNODE" | "RAND" | "ABS" | "CEIL" | "FLOOR" | "ROUND" | "CONCAT"
                | "STRLEN" | "UCASE" | "LCASE" | "ENCODE_FOR_URI" | "CONTAINS"
                | "STRSTARTS" | "STRENDS" | "STRBEFORE" | "STRAFTER" | "YEAR"
                | "MONTH" | "DAY" | "HOURS" | "MINUTES" | "SECONDS" | "TIMEZONE"
                | "TZ" | "NOW" | "UUID" | "STRUUID" | "MD5" | "SHA1" | "SHA256"
                | "SHA384" | "SHA512" | "COALESCE" | "IF" | "STRLANG" | "STRDT"
                | "SAMETERM" | "ISIRI" | "ISURI" | "ISBLANK" | "ISLITERAL"
                | "ISNUMERIC" | "REGEX" | "SUBSTR" | "REPLACE"
        )
    }

    fn parse_built_in_function(&mut self) -> Result<Expression> {
        let func_name = self.current.text.to_uppercase();
        self.advance();

        // Special cases
        match func_name.as_str() {
            "BOUND" => {
                self.expect(TokenKind::LeftParen)?;
                let var = self.expect_variable_name()?;
                self.expect(TokenKind::RightParen)?;
                return Ok(Expression::Bound(var));
            }
            "IF" => {
                self.expect(TokenKind::LeftParen)?;
                let condition = self.parse_expression()?;
                self.expect(TokenKind::Comma)?;
                let then_expr = self.parse_expression()?;
                self.expect(TokenKind::Comma)?;
                let else_expr = self.parse_expression()?;
                self.expect(TokenKind::RightParen)?;
                return Ok(Expression::Conditional {
                    condition: Box::new(condition),
                    then_expression: Box::new(then_expr),
                    else_expression: Box::new(else_expr),
                });
            }
            "COALESCE" => {
                let args = self.parse_expression_list()?;
                return Ok(Expression::Coalesce(args));
            }
            _ => {}
        }

        // Regular function call
        let builtin = match func_name.as_str() {
            "STR" => BuiltInFunction::Str,
            "LANG" => BuiltInFunction::Lang,
            "LANGMATCHES" => BuiltInFunction::LangMatches,
            "DATATYPE" => BuiltInFunction::Datatype,
            "IRI" | "URI" => BuiltInFunction::Iri,
            "BNODE" => BuiltInFunction::Bnode,
            "RAND" => BuiltInFunction::Rand,
            "ABS" => BuiltInFunction::Abs,
            "CEIL" => BuiltInFunction::Ceil,
            "FLOOR" => BuiltInFunction::Floor,
            "ROUND" => BuiltInFunction::Round,
            "CONCAT" => BuiltInFunction::Concat,
            "STRLEN" => BuiltInFunction::StrLen,
            "UCASE" => BuiltInFunction::Ucase,
            "LCASE" => BuiltInFunction::Lcase,
            "ENCODE_FOR_URI" => BuiltInFunction::EncodeForUri,
            "CONTAINS" => BuiltInFunction::Contains,
            "STRSTARTS" => BuiltInFunction::StrStarts,
            "STRENDS" => BuiltInFunction::StrEnds,
            "STRBEFORE" => BuiltInFunction::StrBefore,
            "STRAFTER" => BuiltInFunction::StrAfter,
            "YEAR" => BuiltInFunction::Year,
            "MONTH" => BuiltInFunction::Month,
            "DAY" => BuiltInFunction::Day,
            "HOURS" => BuiltInFunction::Hours,
            "MINUTES" => BuiltInFunction::Minutes,
            "SECONDS" => BuiltInFunction::Seconds,
            "TIMEZONE" => BuiltInFunction::Timezone,
            "TZ" => BuiltInFunction::Tz,
            "NOW" => BuiltInFunction::Now,
            "UUID" => BuiltInFunction::Uuid,
            "STRUUID" => BuiltInFunction::StrUuid,
            "MD5" => BuiltInFunction::Md5,
            "SHA1" => BuiltInFunction::Sha1,
            "SHA256" => BuiltInFunction::Sha256,
            "SHA384" => BuiltInFunction::Sha384,
            "SHA512" => BuiltInFunction::Sha512,
            "STRLANG" => BuiltInFunction::StrLang,
            "STRDT" => BuiltInFunction::StrDt,
            "SAMETERM" => BuiltInFunction::SameTerm,
            "ISIRI" | "ISURI" => BuiltInFunction::IsIri,
            "ISBLANK" => BuiltInFunction::IsBlank,
            "ISLITERAL" => BuiltInFunction::IsLiteral,
            "ISNUMERIC" => BuiltInFunction::IsNumeric,
            "REGEX" => BuiltInFunction::Regex,
            "SUBSTR" => BuiltInFunction::Substr,
            "REPLACE" => BuiltInFunction::Replace,
            _ => return Err(self.error(&format!("unknown function: {}", func_name))),
        };

        let arguments = self.parse_argument_list()?;
        Ok(Expression::FunctionCall {
            function: FunctionName::BuiltIn(builtin),
            arguments,
        })
    }

    fn parse_count_aggregate(&mut self) -> Result<Expression> {
        self.expect(TokenKind::Count)?;
        self.expect(TokenKind::LeftParen)?;

        let distinct = if self.current.kind == TokenKind::Distinct {
            self.advance();
            true
        } else {
            false
        };

        let expression = if self.current.kind == TokenKind::Star {
            self.advance();
            None
        } else {
            Some(Box::new(self.parse_expression()?))
        };

        self.expect(TokenKind::RightParen)?;

        Ok(Expression::Aggregate(AggregateExpression::Count {
            distinct,
            expression,
        }))
    }

    fn parse_aggregate<F>(&mut self, constructor: F) -> Result<Expression>
    where
        F: FnOnce(Expression) -> AggregateExpression,
    {
        self.advance(); // Skip the aggregate keyword
        self.expect(TokenKind::LeftParen)?;

        let distinct = if self.current.kind == TokenKind::Distinct {
            self.advance();
            true
        } else {
            false
        };

        let expression = self.parse_expression()?;
        self.expect(TokenKind::RightParen)?;

        let mut agg = constructor(expression);

        // Apply distinct if needed
        match &mut agg {
            AggregateExpression::Sum { distinct: d, .. }
            | AggregateExpression::Average { distinct: d, .. } => {
                *d = distinct;
            }
            _ => {}
        }

        Ok(Expression::Aggregate(agg))
    }

    fn parse_group_concat(&mut self) -> Result<Expression> {
        self.expect(TokenKind::GroupConcat)?;
        self.expect(TokenKind::LeftParen)?;

        let distinct = if self.current.kind == TokenKind::Distinct {
            self.advance();
            true
        } else {
            false
        };

        let expression = self.parse_expression()?;

        let separator = if self.current.kind == TokenKind::Semicolon {
            self.advance();
            self.expect(TokenKind::Separator)?;
            self.expect(TokenKind::Equals)?;
            let sep = self.parse_literal()?;
            Some(sep.value)
        } else {
            None
        };

        self.expect(TokenKind::RightParen)?;

        Ok(Expression::Aggregate(AggregateExpression::GroupConcat {
            distinct,
            expression: Box::new(expression),
            separator,
        }))
    }

    fn parse_function_call_with_iri(&mut self, iri: Iri) -> Result<Expression> {
        let arguments = self.parse_argument_list()?;
        Ok(Expression::FunctionCall {
            function: FunctionName::Custom(iri),
            arguments,
        })
    }

    fn parse_argument_list(&mut self) -> Result<Vec<Expression>> {
        self.expect(TokenKind::LeftParen)?;

        if self.current.kind == TokenKind::RightParen {
            self.advance();
            return Ok(Vec::new());
        }

        let mut args = vec![self.parse_expression()?];
        while self.current.kind == TokenKind::Comma {
            self.advance();
            args.push(self.parse_expression()?);
        }

        self.expect(TokenKind::RightParen)?;
        Ok(args)
    }

    fn parse_expression_list(&mut self) -> Result<Vec<Expression>> {
        self.expect(TokenKind::LeftParen)?;

        let mut exprs = Vec::new();
        if self.current.kind != TokenKind::RightParen {
            exprs.push(self.parse_expression()?);
            while self.current.kind == TokenKind::Comma {
                self.advance();
                exprs.push(self.parse_expression()?);
            }
        }

        self.expect(TokenKind::RightParen)?;
        Ok(exprs)
    }

    // ==================== Literals ====================

    fn parse_literal(&mut self) -> Result<Literal> {
        let value = self.parse_string_value()?;

        // Check for language tag or datatype
        if self.current.kind == TokenKind::At {
            self.advance();
            let lang = if self.current.kind == TokenKind::PrefixedName {
                let text = self.current.text.clone();
                self.advance();
                text
            } else {
                return Err(self.error("expected language tag"));
            };
            Ok(Literal::with_language(value, lang))
        } else if self.current.kind == TokenKind::DoubleCaret {
            self.advance();
            let datatype = self.expect_iri_or_prefixed()?;
            Ok(Literal::typed(value, datatype))
        } else {
            Ok(Literal::string(value))
        }
    }

    fn parse_string_value(&mut self) -> Result<String> {
        let text = self.current.text.clone();

        let value = if self.current.kind == TokenKind::LongString {
            self.advance();
            // Remove triple quotes
            let quote = text.chars().next().unwrap();
            let inner = &text[3..text.len() - 3];
            self.unescape_string(inner, quote)
        } else if self.current.kind == TokenKind::String {
            self.advance();
            // Remove single quotes
            let quote = text.chars().next().unwrap();
            let inner = &text[1..text.len() - 1];
            self.unescape_string(inner, quote)
        } else {
            return Err(self.error("expected string literal"));
        };

        Ok(value)
    }

    fn unescape_string(&self, s: &str, _quote: char) -> String {
        let mut result = String::with_capacity(s.len());
        let mut chars = s.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '\\' {
                if let Some(&next) = chars.peek() {
                    chars.next();
                    match next {
                        'n' => result.push('\n'),
                        'r' => result.push('\r'),
                        't' => result.push('\t'),
                        '\\' => result.push('\\'),
                        '"' => result.push('"'),
                        '\'' => result.push('\''),
                        'u' => {
                            // Unicode escape \uXXXX
                            let hex: String = chars.by_ref().take(4).collect();
                            if let Ok(code) = u32::from_str_radix(&hex, 16) {
                                if let Some(ch) = char::from_u32(code) {
                                    result.push(ch);
                                }
                            }
                        }
                        'U' => {
                            // Unicode escape \UXXXXXXXX
                            let hex: String = chars.by_ref().take(8).collect();
                            if let Ok(code) = u32::from_str_radix(&hex, 16) {
                                if let Some(ch) = char::from_u32(code) {
                                    result.push(ch);
                                }
                            }
                        }
                        _ => {
                            result.push('\\');
                            result.push(next);
                        }
                    }
                }
            } else {
                result.push(c);
            }
        }

        result
    }

    fn parse_numeric_literal(&mut self) -> Result<Literal> {
        let text = self.current.text.clone();
        let datatype = match self.current.kind {
            TokenKind::Integer => Iri::new("http://www.w3.org/2001/XMLSchema#integer"),
            TokenKind::Decimal => Iri::new("http://www.w3.org/2001/XMLSchema#decimal"),
            TokenKind::Double => Iri::new("http://www.w3.org/2001/XMLSchema#double"),
            _ => return Err(self.error("expected numeric literal")),
        };
        self.advance();
        Ok(Literal::typed(text, datatype))
    }

    fn parse_boolean_literal(&mut self) -> Result<Literal> {
        let value = match self.current.kind {
            TokenKind::True => "true",
            TokenKind::False => "false",
            _ => return Err(self.error("expected boolean literal")),
        };
        self.advance();
        Ok(Literal::typed(
            value,
            Iri::new("http://www.w3.org/2001/XMLSchema#boolean"),
        ))
    }

    // ==================== IRIs ====================

    fn parse_iri(&mut self) -> Result<Iri> {
        if self.current.kind != TokenKind::Iri {
            return Err(self.error("expected IRI"));
        }
        let text = self.current.text.clone();
        self.advance();
        // Remove angle brackets
        let iri = &text[1..text.len() - 1];
        Ok(Iri::new(iri))
    }

    fn parse_prefixed_iri(&mut self) -> Result<Iri> {
        if self.current.kind != TokenKind::PrefixedName {
            return Err(self.error("expected prefixed name"));
        }
        let text = self.current.text.clone();
        self.advance();
        // For now, store as-is; resolution happens during binding
        Ok(Iri::new(text))
    }

    fn parse_variable_or_iri(&mut self) -> Result<VariableOrIri> {
        match self.current.kind {
            TokenKind::Variable => {
                let name = self.expect_variable_name()?;
                Ok(VariableOrIri::Variable(name))
            }
            TokenKind::Iri => {
                let iri = self.parse_iri()?;
                Ok(VariableOrIri::Iri(iri))
            }
            TokenKind::PrefixedName => {
                let iri = self.parse_prefixed_iri()?;
                Ok(VariableOrIri::Iri(iri))
            }
            _ => Err(self.error("expected variable or IRI")),
        }
    }

    // ==================== Helpers ====================

    fn advance(&mut self) {
        self.current = self.lexer.next_token();
    }

    fn expect(&mut self, kind: TokenKind) -> Result<()> {
        if self.current.kind == kind {
            self.advance();
            Ok(())
        } else {
            Err(self.error(&format!("expected {:?}", kind)))
        }
    }

    fn expect_iri(&mut self) -> Result<Iri> {
        if self.current.kind == TokenKind::Iri {
            self.parse_iri()
        } else if self.current.kind == TokenKind::PrefixedName {
            self.parse_prefixed_iri()
        } else {
            Err(self.error("expected IRI"))
        }
    }

    fn expect_iri_or_prefixed(&mut self) -> Result<Iri> {
        match self.current.kind {
            TokenKind::Iri => self.parse_iri(),
            TokenKind::PrefixedName => self.parse_prefixed_iri(),
            TokenKind::A => {
                self.advance();
                Ok(Iri::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"))
            }
            _ => Err(self.error("expected IRI or prefixed name")),
        }
    }

    fn expect_variable_name(&mut self) -> Result<String> {
        if self.current.kind != TokenKind::Variable {
            return Err(self.error("expected variable"));
        }
        let text = self.current.text.clone();
        self.advance();
        // Remove ? or $ prefix
        Ok(text[1..].to_string())
    }

    fn expect_integer(&mut self) -> Result<u64> {
        if self.current.kind != TokenKind::Integer {
            return Err(self.error("expected integer"));
        }
        let text = self.current.text.clone();
        self.advance();
        text.parse()
            .map_err(|_| self.error(&format!("invalid integer: {}", text)))
    }

    fn error(&self, message: &str) -> Error {
        Error::Query(
            QueryError::new(
                QueryErrorKind::Syntax,
                format!("{} at line {}, column {}", message, self.current.span.line, self.current.span.column),
            )
            .with_span(self.current.span)
            .with_source(self.source.to_string())
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(query: &str) -> Result<Query> {
        let mut parser = Parser::new(query);
        parser.parse()
    }

    #[test]
    fn test_simple_select() {
        let query = parse("SELECT ?x WHERE { ?x ?y ?z }").unwrap();
        assert!(matches!(query.query_form, QueryForm::Select(_)));
    }

    #[test]
    fn test_select_with_prefix() {
        let query = parse(
            r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?name
            WHERE { ?x foaf:name ?name }
        "#,
        )
        .unwrap();
        assert_eq!(query.prefixes.len(), 1);
        assert_eq!(query.prefixes[0].prefix, "foaf");
    }

    #[test]
    fn test_select_distinct() {
        let query = parse("SELECT DISTINCT ?x WHERE { ?x ?y ?z }").unwrap();
        if let QueryForm::Select(select) = query.query_form {
            assert_eq!(select.modifier, SelectModifier::Distinct);
        } else {
            panic!("expected SELECT query");
        }
    }

    #[test]
    fn test_select_star() {
        let query = parse("SELECT * WHERE { ?x ?y ?z }").unwrap();
        if let QueryForm::Select(select) = query.query_form {
            assert!(matches!(select.projection, Projection::Wildcard));
        } else {
            panic!("expected SELECT query");
        }
    }

    #[test]
    fn test_optional() {
        let query = parse("SELECT ?x ?y WHERE { ?x ?p ?y OPTIONAL { ?y ?q ?z } }").unwrap();
        assert!(matches!(query.query_form, QueryForm::Select(_)));
    }

    #[test]
    fn test_filter() {
        let query = parse("SELECT ?x WHERE { ?x ?y ?z FILTER(?z > 10) }").unwrap();
        assert!(matches!(query.query_form, QueryForm::Select(_)));
    }

    #[test]
    fn test_order_by() {
        let query = parse("SELECT ?x WHERE { ?x ?y ?z } ORDER BY ?x").unwrap();
        if let QueryForm::Select(select) = query.query_form {
            assert!(select.solution_modifiers.order_by.is_some());
        } else {
            panic!("expected SELECT query");
        }
    }

    #[test]
    fn test_limit_offset() {
        let query = parse("SELECT ?x WHERE { ?x ?y ?z } LIMIT 10 OFFSET 5").unwrap();
        if let QueryForm::Select(select) = query.query_form {
            assert_eq!(select.solution_modifiers.limit, Some(10));
            assert_eq!(select.solution_modifiers.offset, Some(5));
        } else {
            panic!("expected SELECT query");
        }
    }

    #[test]
    fn test_ask_query() {
        let query = parse("ASK { ?x ?y ?z }").unwrap();
        assert!(matches!(query.query_form, QueryForm::Ask(_)));
    }

    #[test]
    fn test_construct_query() {
        let query = parse("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }").unwrap();
        assert!(matches!(query.query_form, QueryForm::Construct(_)));
    }

    #[test]
    fn test_union() {
        let query = parse("SELECT ?x WHERE { { ?x ?y ?z } UNION { ?x ?a ?b } }").unwrap();
        assert!(matches!(query.query_form, QueryForm::Select(_)));
    }

    #[test]
    fn test_aggregates() {
        let query = parse("SELECT (COUNT(?x) AS ?count) WHERE { ?x ?y ?z } GROUP BY ?z").unwrap();
        if let QueryForm::Select(select) = query.query_form {
            assert!(select.solution_modifiers.group_by.is_some());
        } else {
            panic!("expected SELECT query");
        }
    }

    #[test]
    fn test_property_path() {
        let query = parse("SELECT ?x WHERE { ?x foaf:knows+ ?y }").unwrap();
        assert!(matches!(query.query_form, QueryForm::Select(_)));
    }

    #[test]
    fn test_bind() {
        let query = parse("SELECT ?x ?doubled WHERE { ?x ?y ?z BIND(?z * 2 AS ?doubled) }").unwrap();
        assert!(matches!(query.query_form, QueryForm::Select(_)));
    }

    #[test]
    fn test_literal_with_datatype() {
        let query = parse(r#"SELECT ?x WHERE { ?x ?y "42"^^xsd:integer }"#).unwrap();
        assert!(matches!(query.query_form, QueryForm::Select(_)));
    }

    #[test]
    fn test_literal_with_language() {
        let query = parse(r#"SELECT ?x WHERE { ?x ?y "hello"@en }"#).unwrap();
        assert!(matches!(query.query_form, QueryForm::Select(_)));
    }
}
