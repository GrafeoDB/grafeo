//! Gremlin Parser.
//!
//! Parses tokenized Gremlin queries into an AST.

use super::ast::*;
use super::lexer::{Lexer, Token, TokenKind};
use graphos_common::types::Value;
use graphos_common::utils::error::{Error, Result};

/// Gremlin parser.
pub struct Parser<'a> {
    tokens: Vec<Token>,
    position: usize,
    source: &'a str,
}

impl<'a> Parser<'a> {
    /// Creates a new parser for the given source.
    pub fn new(source: &'a str) -> Self {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        Self {
            tokens,
            position: 0,
            source,
        }
    }

    /// Parses the query into a statement.
    pub fn parse(&mut self) -> Result<Statement> {
        self.parse_statement()
    }

    fn parse_statement(&mut self) -> Result<Statement> {
        // Expect 'g' at the start
        self.expect(TokenKind::G)?;
        self.expect(TokenKind::Dot)?;

        // Parse source (V, E, addV, addE)
        let source = self.parse_source()?;

        // Parse steps
        let mut steps = Vec::new();
        while self.check(TokenKind::Dot) {
            self.advance(); // consume '.'
            let step = self.parse_step()?;
            steps.push(step);
        }

        Ok(Statement { source, steps })
    }

    fn parse_source(&mut self) -> Result<TraversalSource> {
        let token = self.advance_token()?;
        match &token.kind {
            TokenKind::V => {
                self.expect(TokenKind::LParen)?;
                let ids = self.parse_optional_value_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(TraversalSource::V(if ids.is_empty() { None } else { Some(ids) }))
            }
            TokenKind::E => {
                self.expect(TokenKind::LParen)?;
                let ids = self.parse_optional_value_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(TraversalSource::E(if ids.is_empty() { None } else { Some(ids) }))
            }
            TokenKind::AddV => {
                self.expect(TokenKind::LParen)?;
                let label = if self.check_string() {
                    Some(self.parse_string()?)
                } else {
                    None
                };
                self.expect(TokenKind::RParen)?;
                Ok(TraversalSource::AddV(label))
            }
            TokenKind::AddE => {
                self.expect(TokenKind::LParen)?;
                let label = self.parse_string()?;
                self.expect(TokenKind::RParen)?;
                Ok(TraversalSource::AddE(label))
            }
            _ => Err(self.error("Expected V, E, addV, or addE")),
        }
    }

    fn parse_step(&mut self) -> Result<Step> {
        let token = self.advance_token()?;
        match &token.kind {
            // Navigation steps
            TokenKind::Out => {
                self.expect(TokenKind::LParen)?;
                let labels = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Out(labels))
            }
            TokenKind::In => {
                self.expect(TokenKind::LParen)?;
                let labels = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::In(labels))
            }
            TokenKind::Both => {
                self.expect(TokenKind::LParen)?;
                let labels = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Both(labels))
            }
            TokenKind::OutE => {
                self.expect(TokenKind::LParen)?;
                let labels = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::OutE(labels))
            }
            TokenKind::InE => {
                self.expect(TokenKind::LParen)?;
                let labels = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::InE(labels))
            }
            TokenKind::BothE => {
                self.expect(TokenKind::LParen)?;
                let labels = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::BothE(labels))
            }
            TokenKind::OutV => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::OutV)
            }
            TokenKind::InV => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::InV)
            }
            TokenKind::BothV => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::BothV)
            }
            TokenKind::OtherV => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::OtherV)
            }

            // Filter steps
            TokenKind::Has => {
                self.expect(TokenKind::LParen)?;
                let has_step = self.parse_has_args()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Has(has_step))
            }
            TokenKind::HasLabel => {
                self.expect(TokenKind::LParen)?;
                let labels = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::HasLabel(labels))
            }
            TokenKind::HasId => {
                self.expect(TokenKind::LParen)?;
                let ids = self.parse_value_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::HasId(ids))
            }
            TokenKind::HasNot => {
                self.expect(TokenKind::LParen)?;
                let key = self.parse_string()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::HasNot(key))
            }
            TokenKind::Dedup => {
                self.expect(TokenKind::LParen)?;
                let keys = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Dedup(keys))
            }
            TokenKind::Limit => {
                self.expect(TokenKind::LParen)?;
                let n = self.parse_integer()? as usize;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Limit(n))
            }
            TokenKind::Skip => {
                self.expect(TokenKind::LParen)?;
                let n = self.parse_integer()? as usize;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Skip(n))
            }
            TokenKind::Range => {
                self.expect(TokenKind::LParen)?;
                let start = self.parse_integer()? as usize;
                self.expect(TokenKind::Comma)?;
                let end = self.parse_integer()? as usize;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Range(start, end))
            }

            // Map steps
            TokenKind::Values => {
                self.expect(TokenKind::LParen)?;
                let keys = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Values(keys))
            }
            TokenKind::ValueMap => {
                self.expect(TokenKind::LParen)?;
                let keys = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::ValueMap(keys))
            }
            TokenKind::ElementMap => {
                self.expect(TokenKind::LParen)?;
                let keys = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::ElementMap(keys))
            }
            TokenKind::Id => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Id)
            }
            TokenKind::Label => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Label)
            }
            TokenKind::Properties => {
                self.expect(TokenKind::LParen)?;
                let keys = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Properties(keys))
            }
            TokenKind::Constant => {
                self.expect(TokenKind::LParen)?;
                let value = self.parse_value()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Constant(value))
            }
            TokenKind::Count => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Count)
            }
            TokenKind::Sum => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Sum)
            }
            TokenKind::Mean => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Mean)
            }
            TokenKind::Min => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Min)
            }
            TokenKind::Max => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Max)
            }
            TokenKind::Fold => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Fold)
            }
            TokenKind::Unfold => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Unfold)
            }
            TokenKind::Group => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Group(None))
            }
            TokenKind::GroupCount => {
                self.expect(TokenKind::LParen)?;
                let label = if self.check_string() {
                    Some(self.parse_string()?)
                } else {
                    None
                };
                self.expect(TokenKind::RParen)?;
                Ok(Step::GroupCount(label))
            }
            TokenKind::Path => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Path)
            }
            TokenKind::Select => {
                self.expect(TokenKind::LParen)?;
                let keys = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Select(keys))
            }
            TokenKind::Project => {
                self.expect(TokenKind::LParen)?;
                let keys = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Project(keys))
            }
            TokenKind::By => {
                self.expect(TokenKind::LParen)?;
                let modifier = self.parse_by_modifier()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::By(modifier))
            }
            TokenKind::Order => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Order(Vec::new()))
            }

            // Side effect steps
            TokenKind::As => {
                self.expect(TokenKind::LParen)?;
                let label = self.parse_string()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::As(label))
            }
            TokenKind::Aggregate => {
                self.expect(TokenKind::LParen)?;
                let label = self.parse_string()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Aggregate(label))
            }
            TokenKind::Store => {
                self.expect(TokenKind::LParen)?;
                let label = self.parse_string()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Store(label))
            }
            TokenKind::Property => {
                self.expect(TokenKind::LParen)?;
                let prop_step = self.parse_property_args()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Property(prop_step))
            }
            TokenKind::Drop => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Drop)
            }

            // Edge creation
            TokenKind::From => {
                self.expect(TokenKind::LParen)?;
                let from_to = self.parse_from_to()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::From(from_to))
            }
            TokenKind::To => {
                self.expect(TokenKind::LParen)?;
                let from_to = self.parse_from_to()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::To(from_to))
            }
            TokenKind::AddV => {
                self.expect(TokenKind::LParen)?;
                let label = if self.check_string() {
                    Some(self.parse_string()?)
                } else {
                    None
                };
                self.expect(TokenKind::RParen)?;
                Ok(Step::AddV(label))
            }
            TokenKind::AddE => {
                self.expect(TokenKind::LParen)?;
                let label = self.parse_string()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::AddE(label))
            }

            _ => Err(self.error(&format!("Unknown step: {:?}", token.kind))),
        }
    }

    fn parse_has_args(&mut self) -> Result<HasStep> {
        let first = self.parse_string()?;

        if !self.check(TokenKind::Comma) {
            return Ok(HasStep::Key(first));
        }
        self.advance(); // consume ','

        // Check if next is a predicate (P.*)
        if self.check(TokenKind::P) {
            let pred = self.parse_predicate()?;
            return Ok(HasStep::KeyPredicate(first, pred));
        }

        let second = self.parse_value()?;

        if !self.check(TokenKind::Comma) {
            return Ok(HasStep::KeyValue(first, second));
        }
        self.advance(); // consume ','

        // Three arguments: label, key, value
        let third = self.parse_value()?;
        let key = match second {
            Value::String(s) => s.to_string(),
            _ => return Err(self.error("Expected string for key")),
        };
        Ok(HasStep::LabelKeyValue(first, key, third))
    }

    fn parse_predicate(&mut self) -> Result<Predicate> {
        self.expect(TokenKind::P)?;
        self.expect(TokenKind::Dot)?;

        let token = self.advance_token()?;
        match &token.kind {
            TokenKind::Eq => {
                self.expect(TokenKind::LParen)?;
                let value = self.parse_value()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Eq(value))
            }
            TokenKind::Neq => {
                self.expect(TokenKind::LParen)?;
                let value = self.parse_value()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Neq(value))
            }
            TokenKind::Lt => {
                self.expect(TokenKind::LParen)?;
                let value = self.parse_value()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Lt(value))
            }
            TokenKind::Lte => {
                self.expect(TokenKind::LParen)?;
                let value = self.parse_value()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Lte(value))
            }
            TokenKind::Gt => {
                self.expect(TokenKind::LParen)?;
                let value = self.parse_value()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Gt(value))
            }
            TokenKind::Gte => {
                self.expect(TokenKind::LParen)?;
                let value = self.parse_value()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Gte(value))
            }
            TokenKind::Within => {
                self.expect(TokenKind::LParen)?;
                let values = self.parse_value_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Within(values))
            }
            TokenKind::Without => {
                self.expect(TokenKind::LParen)?;
                let values = self.parse_value_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Without(values))
            }
            TokenKind::Between => {
                self.expect(TokenKind::LParen)?;
                let start = self.parse_value()?;
                self.expect(TokenKind::Comma)?;
                let end = self.parse_value()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Between(start, end))
            }
            TokenKind::Containing => {
                self.expect(TokenKind::LParen)?;
                let s = self.parse_string()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Containing(s))
            }
            TokenKind::StartingWith => {
                self.expect(TokenKind::LParen)?;
                let s = self.parse_string()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::StartingWith(s))
            }
            TokenKind::EndingWith => {
                self.expect(TokenKind::LParen)?;
                let s = self.parse_string()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::EndingWith(s))
            }
            _ => Err(self.error("Unknown predicate")),
        }
    }

    fn parse_by_modifier(&mut self) -> Result<ByModifier> {
        if self.check(TokenKind::RParen) {
            return Ok(ByModifier::Identity);
        }

        if self.check(TokenKind::T) {
            self.advance();
            self.expect(TokenKind::Dot)?;
            let token = self.advance_token()?;
            let t = match &token.kind {
                TokenKind::Id => TokenType::Id,
                TokenKind::Label => TokenType::Label,
                _ => return Err(self.error("Expected T.id or T.label")),
            };
            return Ok(ByModifier::Token(t));
        }

        if self.check_string() {
            let key = self.parse_string()?;
            return Ok(ByModifier::Key(key));
        }

        Ok(ByModifier::Identity)
    }

    fn parse_property_args(&mut self) -> Result<PropertyStep> {
        let mut cardinality = None;

        // Check for cardinality
        match self.current_kind() {
            Some(TokenKind::Single) => {
                cardinality = Some(Cardinality::Single);
                self.advance();
                self.expect(TokenKind::Comma)?;
            }
            Some(TokenKind::List) => {
                cardinality = Some(Cardinality::List);
                self.advance();
                self.expect(TokenKind::Comma)?;
            }
            Some(TokenKind::Set) => {
                cardinality = Some(Cardinality::Set);
                self.advance();
                self.expect(TokenKind::Comma)?;
            }
            _ => {}
        }

        let key = self.parse_string()?;
        self.expect(TokenKind::Comma)?;
        let value = self.parse_value()?;

        Ok(PropertyStep {
            cardinality,
            key,
            value,
        })
    }

    fn parse_from_to(&mut self) -> Result<FromTo> {
        if self.check_string() {
            let label = self.parse_string()?;
            return Ok(FromTo::Label(label));
        }
        // Could also be a traversal, but for simplicity just handle labels
        Err(self.error("Expected label for from/to"))
    }

    fn parse_string_list(&mut self) -> Result<Vec<String>> {
        let mut result = Vec::new();
        while self.check_string() {
            result.push(self.parse_string()?);
            if !self.check(TokenKind::Comma) {
                break;
            }
            self.advance();
        }
        Ok(result)
    }

    fn parse_value_list(&mut self) -> Result<Vec<Value>> {
        let mut result = Vec::new();
        loop {
            if self.check(TokenKind::RParen) {
                break;
            }
            result.push(self.parse_value()?);
            if !self.check(TokenKind::Comma) {
                break;
            }
            self.advance();
        }
        Ok(result)
    }

    fn parse_optional_value_list(&mut self) -> Result<Vec<Value>> {
        if self.check(TokenKind::RParen) {
            return Ok(Vec::new());
        }
        self.parse_value_list()
    }

    fn parse_string(&mut self) -> Result<String> {
        let token = self.advance_token()?;
        match token.kind {
            TokenKind::String(s) => Ok(s),
            TokenKind::Identifier(s) => Ok(s),
            _ => Err(self.error("Expected string")),
        }
    }

    fn parse_integer(&mut self) -> Result<i64> {
        let token = self.advance_token()?;
        match token.kind {
            TokenKind::Integer(n) => Ok(n),
            _ => Err(self.error("Expected integer")),
        }
    }

    fn parse_value(&mut self) -> Result<Value> {
        let token = self.advance_token()?;
        match token.kind {
            TokenKind::Integer(n) => Ok(Value::Int64(n)),
            TokenKind::Float(f) => Ok(Value::Float64(f)),
            TokenKind::String(s) => Ok(Value::String(s.into())),
            TokenKind::True => Ok(Value::Bool(true)),
            TokenKind::False => Ok(Value::Bool(false)),
            _ => Err(self.error("Expected value")),
        }
    }

    fn check(&self, kind: TokenKind) -> bool {
        self.current_kind() == Some(&kind)
    }

    fn check_string(&self) -> bool {
        matches!(
            self.current_kind(),
            Some(TokenKind::String(_)) | Some(TokenKind::Identifier(_))
        )
    }

    fn current_kind(&self) -> Option<&TokenKind> {
        self.tokens.get(self.position).map(|t| &t.kind)
    }

    fn advance(&mut self) -> Option<&Token> {
        let token = self.tokens.get(self.position);
        self.position += 1;
        token
    }

    fn advance_token(&mut self) -> Result<Token> {
        self.advance()
            .cloned()
            .ok_or_else(|| self.error("Unexpected end of input"))
    }

    fn expect(&mut self, kind: TokenKind) -> Result<Token> {
        let token = self.advance_token()?;
        if std::mem::discriminant(&token.kind) == std::mem::discriminant(&kind) {
            Ok(token)
        } else {
            Err(self.error(&format!("Expected {:?}, found {:?}", kind, token.kind)))
        }
    }

    fn error(&self, message: &str) -> Error {
        Error::Query(graphos_common::utils::error::QueryError::new(
            graphos_common::utils::error::QueryErrorKind::Syntax,
            message,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_traversal() {
        let mut parser = Parser::new("g.V()");
        let result = parser.parse();
        assert!(result.is_ok());
        let stmt = result.unwrap();
        assert!(matches!(stmt.source, TraversalSource::V(None)));
        assert!(stmt.steps.is_empty());
    }

    #[test]
    fn test_parse_with_steps() {
        let mut parser = Parser::new("g.V().hasLabel('Person').out('knows')");
        let result = parser.parse();
        assert!(result.is_ok());
        let stmt = result.unwrap();
        assert_eq!(stmt.steps.len(), 2);
    }

    #[test]
    fn test_parse_has_with_value() {
        let mut parser = Parser::new("g.V().has('name', 'Alice')");
        let result = parser.parse();
        assert!(result.is_ok());
        let stmt = result.unwrap();
        assert_eq!(stmt.steps.len(), 1);
        if let Step::Has(HasStep::KeyValue(key, value)) = &stmt.steps[0] {
            assert_eq!(key, "name");
            assert_eq!(*value, Value::String("Alice".into()));
        } else {
            panic!("Expected Has step with key-value");
        }
    }

    #[test]
    fn test_parse_limit() {
        let mut parser = Parser::new("g.V().limit(10)");
        let result = parser.parse();
        assert!(result.is_ok());
        let stmt = result.unwrap();
        if let Step::Limit(n) = &stmt.steps[0] {
            assert_eq!(*n, 10);
        } else {
            panic!("Expected Limit step");
        }
    }

    #[test]
    fn test_parse_values() {
        let mut parser = Parser::new("g.V().values('name', 'age')");
        let result = parser.parse();
        assert!(result.is_ok());
        let stmt = result.unwrap();
        if let Step::Values(keys) = &stmt.steps[0] {
            assert_eq!(keys.len(), 2);
            assert_eq!(keys[0], "name");
            assert_eq!(keys[1], "age");
        } else {
            panic!("Expected Values step");
        }
    }
}
