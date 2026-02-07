//! GraphQL Parser.
//!
//! Parses tokenized GraphQL queries into an AST.

use super::ast::*;
use super::lexer::{Lexer, Token, TokenKind};
use grafeo_common::utils::error::{Error, Result};

/// GraphQL parser.
pub struct Parser<'a> {
    tokens: Vec<Token>,
    position: usize,
    #[allow(dead_code)]
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

    /// Parses the document.
    pub fn parse(&mut self) -> Result<Document> {
        self.parse_document()
    }

    fn parse_document(&mut self) -> Result<Document> {
        let mut definitions = Vec::new();

        while !self.is_eof() {
            definitions.push(self.parse_definition()?);
        }

        // If no definitions but we have a selection set, wrap in anonymous query
        if definitions.is_empty() && self.check(TokenKind::LBrace) {
            definitions.push(Definition::Operation(OperationDefinition {
                operation: OperationType::Query,
                name: None,
                variables: Vec::new(),
                directives: Vec::new(),
                selection_set: self.parse_selection_set()?,
            }));
        }

        Ok(Document { definitions })
    }

    fn parse_definition(&mut self) -> Result<Definition> {
        match self.current_kind() {
            Some(TokenKind::Query) | Some(TokenKind::Mutation) | Some(TokenKind::Subscription) => {
                Ok(Definition::Operation(self.parse_operation_definition()?))
            }
            Some(TokenKind::Fragment) => {
                Ok(Definition::Fragment(self.parse_fragment_definition()?))
            }
            Some(TokenKind::LBrace) => {
                // Anonymous query
                Ok(Definition::Operation(OperationDefinition {
                    operation: OperationType::Query,
                    name: None,
                    variables: Vec::new(),
                    directives: Vec::new(),
                    selection_set: self.parse_selection_set()?,
                }))
            }
            _ => Err(self.error("Expected operation or fragment definition")),
        }
    }

    fn parse_operation_definition(&mut self) -> Result<OperationDefinition> {
        let operation = match self.current_kind() {
            Some(TokenKind::Query) => {
                self.advance();
                OperationType::Query
            }
            Some(TokenKind::Mutation) => {
                self.advance();
                OperationType::Mutation
            }
            Some(TokenKind::Subscription) => {
                self.advance();
                OperationType::Subscription
            }
            _ => OperationType::Query,
        };

        // Optional name
        let name = if let Some(TokenKind::Name(n)) = self.current_kind() {
            let name = n.clone();
            self.advance();
            Some(name)
        } else {
            None
        };

        // Variable definitions
        let variables = if self.check(TokenKind::LParen) {
            self.parse_variable_definitions()?
        } else {
            Vec::new()
        };

        // Directives
        let directives = self.parse_directives()?;

        // Selection set
        let selection_set = self.parse_selection_set()?;

        Ok(OperationDefinition {
            operation,
            name,
            variables,
            directives,
            selection_set,
        })
    }

    fn parse_fragment_definition(&mut self) -> Result<FragmentDefinition> {
        self.expect(TokenKind::Fragment)?;

        let name = self.parse_name()?;

        self.expect(TokenKind::On)?;

        let type_condition = self.parse_name()?;

        let directives = self.parse_directives()?;

        let selection_set = self.parse_selection_set()?;

        Ok(FragmentDefinition {
            name,
            type_condition,
            directives,
            selection_set,
        })
    }

    fn parse_variable_definitions(&mut self) -> Result<Vec<VariableDefinition>> {
        self.expect(TokenKind::LParen)?;

        let mut definitions = Vec::new();
        while !self.check(TokenKind::RParen) && !self.is_eof() {
            definitions.push(self.parse_variable_definition()?);
        }

        self.expect(TokenKind::RParen)?;

        Ok(definitions)
    }

    fn parse_variable_definition(&mut self) -> Result<VariableDefinition> {
        self.expect(TokenKind::Dollar)?;
        let name = self.parse_name()?;
        self.expect(TokenKind::Colon)?;
        let variable_type = self.parse_type()?;

        let default_value = if self.check(TokenKind::Eq) {
            self.advance();
            Some(self.parse_input_value()?.to_value())
        } else {
            None
        };

        let directives = self.parse_directives()?;

        Ok(VariableDefinition {
            name,
            variable_type,
            default_value,
            directives,
        })
    }

    fn parse_type(&mut self) -> Result<Type> {
        let base_type = if self.check(TokenKind::LBracket) {
            self.advance();
            let inner = self.parse_type()?;
            self.expect(TokenKind::RBracket)?;
            Type::List(Box::new(inner))
        } else {
            let name = self.parse_name()?;
            Type::Named(name)
        };

        if self.check(TokenKind::Bang) {
            self.advance();
            Ok(Type::NonNull(Box::new(base_type)))
        } else {
            Ok(base_type)
        }
    }

    fn parse_selection_set(&mut self) -> Result<SelectionSet> {
        self.expect(TokenKind::LBrace)?;

        let mut selections = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_eof() {
            selections.push(self.parse_selection()?);
        }

        self.expect(TokenKind::RBrace)?;

        Ok(SelectionSet { selections })
    }

    fn parse_selection(&mut self) -> Result<Selection> {
        if self.check(TokenKind::Spread) {
            self.advance();

            // Check for fragment spread or inline fragment
            if self.check(TokenKind::On)
                || self.check(TokenKind::At)
                || self.check(TokenKind::LBrace)
            {
                // Inline fragment
                let type_condition = if self.check(TokenKind::On) {
                    self.advance();
                    Some(self.parse_name()?)
                } else {
                    None
                };

                let directives = self.parse_directives()?;
                let selection_set = self.parse_selection_set()?;

                Ok(Selection::InlineFragment(InlineFragment {
                    type_condition,
                    directives,
                    selection_set,
                }))
            } else {
                // Fragment spread
                let name = self.parse_name()?;
                let directives = self.parse_directives()?;

                Ok(Selection::FragmentSpread(FragmentSpread {
                    name,
                    directives,
                }))
            }
        } else {
            // Field
            Ok(Selection::Field(self.parse_field()?))
        }
    }

    fn parse_field(&mut self) -> Result<Field> {
        let first_name = self.parse_name()?;

        // Check for alias
        let (alias, name) = if self.check(TokenKind::Colon) {
            self.advance();
            let name = self.parse_name()?;
            (Some(first_name), name)
        } else {
            (None, first_name)
        };

        // Arguments
        let arguments = if self.check(TokenKind::LParen) {
            self.parse_arguments()?
        } else {
            Vec::new()
        };

        // Directives
        let directives = self.parse_directives()?;

        // Selection set
        let selection_set = if self.check(TokenKind::LBrace) {
            Some(self.parse_selection_set()?)
        } else {
            None
        };

        Ok(Field {
            alias,
            name,
            arguments,
            directives,
            selection_set,
        })
    }

    fn parse_arguments(&mut self) -> Result<Vec<Argument>> {
        self.expect(TokenKind::LParen)?;

        let mut arguments = Vec::new();
        while !self.check(TokenKind::RParen) && !self.is_eof() {
            arguments.push(self.parse_argument()?);
        }

        self.expect(TokenKind::RParen)?;

        Ok(arguments)
    }

    fn parse_argument(&mut self) -> Result<Argument> {
        let name = self.parse_name()?;
        self.expect(TokenKind::Colon)?;
        let value = self.parse_input_value()?;

        Ok(Argument { name, value })
    }

    fn parse_directives(&mut self) -> Result<Vec<Directive>> {
        let mut directives = Vec::new();
        while self.check(TokenKind::At) {
            directives.push(self.parse_directive()?);
        }
        Ok(directives)
    }

    fn parse_directive(&mut self) -> Result<Directive> {
        self.expect(TokenKind::At)?;
        let name = self.parse_name()?;

        let arguments = if self.check(TokenKind::LParen) {
            self.parse_arguments()?
        } else {
            Vec::new()
        };

        Ok(Directive { name, arguments })
    }

    fn parse_input_value(&mut self) -> Result<InputValue> {
        let token = self.advance_token()?;
        match token.kind {
            TokenKind::Dollar => {
                let name = self.parse_name()?;
                Ok(InputValue::Variable(name))
            }
            TokenKind::Int(n) => Ok(InputValue::Int(n)),
            TokenKind::Float(f) => Ok(InputValue::Float(f)),
            TokenKind::String(s) | TokenKind::BlockString(s) => Ok(InputValue::String(s)),
            TokenKind::True => Ok(InputValue::Boolean(true)),
            TokenKind::False => Ok(InputValue::Boolean(false)),
            TokenKind::Null => Ok(InputValue::Null),
            TokenKind::Name(s) => Ok(InputValue::Enum(s)),
            TokenKind::LBracket => {
                let mut items = Vec::new();
                while !self.check(TokenKind::RBracket) && !self.is_eof() {
                    items.push(self.parse_input_value()?);
                }
                self.expect(TokenKind::RBracket)?;
                Ok(InputValue::List(items))
            }
            TokenKind::LBrace => {
                let mut fields = Vec::new();
                while !self.check(TokenKind::RBrace) && !self.is_eof() {
                    let name = self.parse_name()?;
                    self.expect(TokenKind::Colon)?;
                    let value = self.parse_input_value()?;
                    fields.push((name, value));
                }
                self.expect(TokenKind::RBrace)?;
                Ok(InputValue::Object(fields))
            }
            _ => Err(self.error("Expected value")),
        }
    }

    fn parse_name(&mut self) -> Result<String> {
        let token = self.advance_token()?;
        match token.kind {
            TokenKind::Name(s) => Ok(s),
            // Keywords are also valid names
            TokenKind::Query => Ok("query".to_string()),
            TokenKind::Mutation => Ok("mutation".to_string()),
            TokenKind::Subscription => Ok("subscription".to_string()),
            TokenKind::Fragment => Ok("fragment".to_string()),
            TokenKind::On => Ok("on".to_string()),
            TokenKind::True => Ok("true".to_string()),
            TokenKind::False => Ok("false".to_string()),
            TokenKind::Null => Ok("null".to_string()),
            _ => Err(self.error("Expected name")),
        }
    }

    fn check(&self, kind: TokenKind) -> bool {
        self.current_kind() == Some(&kind)
    }

    fn current_kind(&self) -> Option<&TokenKind> {
        self.tokens.get(self.position).map(|t| &t.kind)
    }

    fn is_eof(&self) -> bool {
        matches!(self.current_kind(), Some(TokenKind::Eof) | None)
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
        Error::Query(grafeo_common::utils::error::QueryError::new(
            grafeo_common::utils::error::QueryErrorKind::Syntax,
            message,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_query() {
        let mut parser = Parser::new("{ user { name } }");
        let result = parser.parse();
        assert!(result.is_ok());
        let doc = result.unwrap();
        assert_eq!(doc.definitions.len(), 1);
    }

    #[test]
    fn test_parse_named_query() {
        let mut parser = Parser::new("query GetUser { user { name } }");
        let result = parser.parse();
        assert!(result.is_ok());
        let doc = result.unwrap();
        if let Definition::Operation(op) = &doc.definitions[0] {
            assert_eq!(op.operation, OperationType::Query);
            assert_eq!(op.name, Some("GetUser".to_string()));
        } else {
            panic!("Expected operation definition");
        }
    }

    #[test]
    fn test_parse_query_with_arguments() {
        let mut parser = Parser::new("{ user(id: 123) { name } }");
        let result = parser.parse();
        assert!(result.is_ok());
        let doc = result.unwrap();
        if let Definition::Operation(op) = &doc.definitions[0] {
            if let Selection::Field(field) = &op.selection_set.selections[0] {
                assert_eq!(field.name, "user");
                assert_eq!(field.arguments.len(), 1);
                assert_eq!(field.arguments[0].name, "id");
            } else {
                panic!("Expected field selection");
            }
        } else {
            panic!("Expected operation definition");
        }
    }

    #[test]
    fn test_parse_mutation() {
        let mut parser =
            Parser::new("mutation CreateUser($name: String!) { createUser(name: $name) { id } }");
        let result = parser.parse();
        assert!(result.is_ok());
        let doc = result.unwrap();
        if let Definition::Operation(op) = &doc.definitions[0] {
            assert_eq!(op.operation, OperationType::Mutation);
            assert_eq!(op.variables.len(), 1);
        } else {
            panic!("Expected operation definition");
        }
    }

    #[test]
    fn test_parse_fragment() {
        let mut parser = Parser::new("fragment UserFields on User { name email }");
        let result = parser.parse();
        assert!(result.is_ok());
        let doc = result.unwrap();
        if let Definition::Fragment(frag) = &doc.definitions[0] {
            assert_eq!(frag.name, "UserFields");
            assert_eq!(frag.type_condition, "User");
        } else {
            panic!("Expected fragment definition");
        }
    }

    #[test]
    fn test_parse_nested_fields() {
        let mut parser = Parser::new("{ user { friends { name } } }");
        let result = parser.parse();
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_alias() {
        let mut parser = Parser::new("{ myUser: user { name } }");
        let result = parser.parse();
        assert!(result.is_ok());
        let doc = result.unwrap();
        if let Definition::Operation(op) = &doc.definitions[0] {
            if let Selection::Field(field) = &op.selection_set.selections[0] {
                assert_eq!(field.alias, Some("myUser".to_string()));
                assert_eq!(field.name, "user");
            } else {
                panic!("Expected field selection");
            }
        }
    }

    #[test]
    fn test_parse_directive() {
        let mut parser = Parser::new("{ user @include(if: true) { name } }");
        let result = parser.parse();
        assert!(result.is_ok());
        let doc = result.unwrap();
        if let Definition::Operation(op) = &doc.definitions[0]
            && let Selection::Field(field) = &op.selection_set.selections[0]
        {
            assert_eq!(field.directives.len(), 1);
            assert_eq!(field.directives[0].name, "include");
        }
    }
}
