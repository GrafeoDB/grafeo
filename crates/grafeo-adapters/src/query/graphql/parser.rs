//! GraphQL Parser.
//!
//! Parses tokenized GraphQL queries into an AST.

#[allow(clippy::wildcard_imports)]
use super::ast::*;
use super::lexer::{Lexer, Token, TokenKind};
use crate::query::limits::{Nesting, nesting_error_message};
use grafeo_common::utils::error::{Error, Result};

/// GraphQL parser.
pub struct Parser<'a> {
    tokens: Vec<Token>,
    position: usize,
    source: &'a str,
    /// How deep the document parsed so far nests (see
    /// [`MAX_NESTING_DEPTH`](crate::query::limits::MAX_NESTING_DEPTH)).
    nesting: Nesting,
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
            nesting: Nesting::default(),
        }
    }

    /// Parses the document.
    ///
    /// # Errors
    ///
    /// Returns an error if the input contains invalid or unexpected GraphQL syntax.
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
        self.enter_nesting()?;
        self.expect(TokenKind::LBrace)?;

        let mut selections = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_eof() {
            selections.push(self.parse_selection()?);
        }

        self.expect(TokenKind::RBrace)?;
        self.exit_nesting();

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
            // A list or an object nests one level.
            TokenKind::LBracket => {
                self.enter_nesting()?;
                let mut items = Vec::new();
                while !self.check(TokenKind::RBracket) && !self.is_eof() {
                    items.push(self.parse_input_value()?);
                }
                self.exit_nesting();
                self.expect(TokenKind::RBracket)?;
                Ok(InputValue::List(items))
            }
            TokenKind::LBrace => {
                self.enter_nesting()?;
                let mut fields = Vec::new();
                while !self.check(TokenKind::RBrace) && !self.is_eof() {
                    let name = self.parse_name()?;
                    self.expect(TokenKind::Colon)?;
                    let value = self.parse_input_value()?;
                    fields.push((name, value));
                }
                self.exit_nesting();
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
        let token = self
            .advance()
            .cloned()
            .ok_or_else(|| self.error("Unexpected end of input"))?;
        if let TokenKind::Error(msg) = &token.kind {
            return Err(self.error(msg));
        }
        Ok(token)
    }

    fn expect(&mut self, kind: TokenKind) -> Result<Token> {
        let token = self.advance_token()?;
        if std::mem::discriminant(&token.kind) == std::mem::discriminant(&kind) {
            Ok(token)
        } else {
            Err(self.error(&format!("Expected {:?}, found {:?}", kind, token.kind)))
        }
    }

    fn current_span(&self) -> grafeo_common::utils::error::SourceSpan {
        self.tokens
            .get(self.position)
            .or_else(|| self.tokens.last())
            .map_or(
                grafeo_common::utils::error::SourceSpan::new(0, 0, 1, 1),
                |t| t.span,
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

    fn error(&self, message: &str) -> Error {
        Error::Query(
            grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Syntax,
                format!("[GraphQL] {message}"),
            )
            .with_span(self.current_span())
            .with_source(self.source.to_string()),
        )
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

    // ==================== Error/Negative Cases ====================

    #[test]
    fn test_parse_empty_input() {
        let mut parser = Parser::new("");
        let result = parser.parse();
        assert!(result.is_ok(), "Empty input produces empty document");
        assert!(result.unwrap().definitions.is_empty());
    }

    #[test]
    fn test_parse_unclosed_brace() {
        let mut parser = Parser::new("{ user { name }");
        let result = parser.parse();
        assert!(result.is_err(), "Unclosed brace should fail");
    }

    #[test]
    fn test_parse_unclosed_selection_set() {
        let mut parser = Parser::new("{ user { name");
        let result = parser.parse();
        assert!(result.is_err(), "Unclosed nested selection set should fail");
    }

    #[test]
    fn test_parse_missing_selection_set() {
        let mut parser = Parser::new("query GetUser");
        let result = parser.parse();
        assert!(result.is_err(), "Named query without body should fail");
    }

    #[test]
    fn test_parse_invalid_argument_syntax() {
        let mut parser = Parser::new("{ user(id: ) { name } }");
        let result = parser.parse();
        assert!(result.is_err(), "Missing argument value should fail");
    }

    #[test]
    fn test_parse_unclosed_argument_list() {
        let mut parser = Parser::new("{ user(id: 123 { name } }");
        let result = parser.parse();
        assert!(result.is_err(), "Unclosed argument list should fail");
    }

    #[test]
    fn test_parse_fragment_without_type_condition() {
        let mut parser = Parser::new("fragment UserFields { name }");
        let result = parser.parse();
        assert!(result.is_err(), "Fragment without 'on Type' should fail");
    }

    #[test]
    fn test_parse_variable_without_type() {
        let mut parser = Parser::new("query ($name) { user { name } }");
        let result = parser.parse();
        assert!(result.is_err(), "Variable without type should fail");
    }

    #[test]
    fn test_parse_multiple_operations() {
        let mut parser = Parser::new("query A { user { name } } query B { post { title } }");
        let result = parser.parse();
        assert!(result.is_ok());
        let doc = result.unwrap();
        assert_eq!(doc.definitions.len(), 2);
    }

    #[test]
    fn test_parse_inline_fragment() {
        let mut parser = Parser::new("{ user { ... on Admin { role } name } }");
        let result = parser.parse();
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_fragment_spread() {
        let mut parser = Parser::new("{ user { ...UserFields } }");
        let result = parser.parse();
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_list_argument() {
        let mut parser = Parser::new("{ users(ids: [1, 2, 3]) { name } }");
        let result = parser.parse();
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_object_argument_not_supported() {
        // Object literals in arguments are not yet supported
        let mut parser = Parser::new("{ createUser(input: {name: \"Alix\", age: 30}) { id } }");
        let result = parser.parse();
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_boolean_values() {
        let mut parser = Parser::new("{ user(active: true, archived: false) { name } }");
        let result = parser.parse();
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_null_value() {
        let mut parser = Parser::new("{ user(name: null) { id } }");
        let result = parser.parse();
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_variable_reference() {
        let mut parser = Parser::new("query ($id: Int!) { user(id: $id) { name } }");
        let result = parser.parse();
        assert!(result.is_ok());
        let doc = result.unwrap();
        if let Definition::Operation(op) = &doc.definitions[0] {
            assert_eq!(op.variables.len(), 1);
            assert!(matches!(
                op.variables[0].variable_type,
                crate::query::graphql::ast::Type::NonNull(_)
            ));
        }
    }

    #[test]
    fn test_parse_subscription() {
        let mut parser = Parser::new("subscription { messageAdded { text } }");
        let result = parser.parse();
        assert!(result.is_ok());
        let doc = result.unwrap();
        if let Definition::Operation(op) = &doc.definitions[0] {
            assert_eq!(op.operation, OperationType::Subscription);
        }
    }

    // ==================== Recursion depth limit tests ====================

    #[test]
    fn test_deeply_nested_selection_sets_errors_not_stack_overflow() {
        // { f { f { f { ... } } } }
        let open = "f { ".repeat(300);
        let close = " }".repeat(300);
        let query = format!("{{ {open}x{close} }}");
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
    fn test_moderate_selection_nesting_succeeds() {
        // 10 levels of nesting should succeed
        let open = "f { ".repeat(10);
        let close = " }".repeat(10);
        let query = format!("{{ {open}x{close} }}");
        let mut parser = Parser::new(&query);
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "10 levels of selection nesting should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_deeply_nested_inline_fragments() {
        let open = "... on Type { ".repeat(300);
        let close = " }".repeat(300);
        let query = format!("{{ {open}x{close} }}");
        let mut parser = Parser::new(&query);
        let result = parser.parse();
        assert!(
            result.is_err(),
            "300 levels of inline fragments should error, not stack overflow"
        );
    }
}
