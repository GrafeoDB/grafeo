//! GraphQL Abstract Syntax Tree.
//!
//! Represents the structure of GraphQL documents according to the specification.

use graphos_common::types::Value;

/// A complete GraphQL document.
#[derive(Debug, Clone)]
pub struct Document {
    /// Definitions in the document.
    pub definitions: Vec<Definition>,
}

/// A definition in a GraphQL document.
#[derive(Debug, Clone)]
pub enum Definition {
    /// An operation (query, mutation, subscription).
    Operation(OperationDefinition),
    /// A fragment definition.
    Fragment(FragmentDefinition),
}

/// An operation definition.
#[derive(Debug, Clone)]
pub struct OperationDefinition {
    /// Operation type.
    pub operation: OperationType,
    /// Optional operation name.
    pub name: Option<String>,
    /// Variable definitions.
    pub variables: Vec<VariableDefinition>,
    /// Directives applied to the operation.
    pub directives: Vec<Directive>,
    /// Selection set.
    pub selection_set: SelectionSet,
}

/// Operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    /// Query operation.
    Query,
    /// Mutation operation.
    Mutation,
    /// Subscription operation.
    Subscription,
}

/// A variable definition.
#[derive(Debug, Clone)]
pub struct VariableDefinition {
    /// Variable name (without $).
    pub name: String,
    /// Variable type.
    pub variable_type: Type,
    /// Default value.
    pub default_value: Option<Value>,
    /// Directives.
    pub directives: Vec<Directive>,
}

/// A GraphQL type.
#[derive(Debug, Clone)]
pub enum Type {
    /// Named type (e.g., String, Int, User).
    Named(String),
    /// List type (e.g., [String]).
    List(Box<Type>),
    /// Non-null type (e.g., String!).
    NonNull(Box<Type>),
}

/// A directive applied to a field or operation.
#[derive(Debug, Clone)]
pub struct Directive {
    /// Directive name (without @).
    pub name: String,
    /// Arguments.
    pub arguments: Vec<Argument>,
}

/// A selection set.
#[derive(Debug, Clone)]
pub struct SelectionSet {
    /// Selections in this set.
    pub selections: Vec<Selection>,
}

/// A selection within a selection set.
#[derive(Debug, Clone)]
pub enum Selection {
    /// A field selection.
    Field(Field),
    /// A fragment spread.
    FragmentSpread(FragmentSpread),
    /// An inline fragment.
    InlineFragment(InlineFragment),
}

/// A field selection.
#[derive(Debug, Clone)]
pub struct Field {
    /// Optional alias for the field.
    pub alias: Option<String>,
    /// Field name.
    pub name: String,
    /// Arguments.
    pub arguments: Vec<Argument>,
    /// Directives.
    pub directives: Vec<Directive>,
    /// Nested selection set (for object types).
    pub selection_set: Option<SelectionSet>,
}

/// An argument (name-value pair).
#[derive(Debug, Clone)]
pub struct Argument {
    /// Argument name.
    pub name: String,
    /// Argument value.
    pub value: InputValue,
}

/// An input value in GraphQL.
#[derive(Debug, Clone)]
pub enum InputValue {
    /// A variable reference.
    Variable(String),
    /// Integer value.
    Int(i64),
    /// Float value.
    Float(f64),
    /// String value.
    String(String),
    /// Boolean value.
    Boolean(bool),
    /// Null value.
    Null,
    /// Enum value.
    Enum(String),
    /// List value.
    List(Vec<InputValue>),
    /// Object (input object) value.
    Object(Vec<(String, InputValue)>),
}

impl InputValue {
    /// Converts to a graphos Value.
    pub fn to_value(&self) -> Value {
        match self {
            InputValue::Int(n) => Value::Int64(*n),
            InputValue::Float(f) => Value::Float64(*f),
            InputValue::String(s) => Value::String(s.clone().into()),
            InputValue::Boolean(b) => Value::Bool(*b),
            InputValue::Null => Value::Null,
            InputValue::Enum(s) => Value::String(s.clone().into()),
            InputValue::List(items) => {
                Value::List(items.iter().map(|i| i.to_value()).collect())
            }
            InputValue::Object(_) => Value::Null, // Objects not directly supported
            InputValue::Variable(_) => Value::Null, // Variables need runtime resolution
        }
    }
}

/// A fragment spread.
#[derive(Debug, Clone)]
pub struct FragmentSpread {
    /// Fragment name.
    pub name: String,
    /// Directives.
    pub directives: Vec<Directive>,
}

/// An inline fragment.
#[derive(Debug, Clone)]
pub struct InlineFragment {
    /// Type condition (optional).
    pub type_condition: Option<String>,
    /// Directives.
    pub directives: Vec<Directive>,
    /// Selection set.
    pub selection_set: SelectionSet,
}

/// A fragment definition.
#[derive(Debug, Clone)]
pub struct FragmentDefinition {
    /// Fragment name.
    pub name: String,
    /// Type condition.
    pub type_condition: String,
    /// Directives.
    pub directives: Vec<Directive>,
    /// Selection set.
    pub selection_set: SelectionSet,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_document() {
        let doc = Document {
            definitions: vec![Definition::Operation(OperationDefinition {
                operation: OperationType::Query,
                name: Some("GetUsers".to_string()),
                variables: Vec::new(),
                directives: Vec::new(),
                selection_set: SelectionSet {
                    selections: vec![Selection::Field(Field {
                        alias: None,
                        name: "users".to_string(),
                        arguments: Vec::new(),
                        directives: Vec::new(),
                        selection_set: Some(SelectionSet {
                            selections: vec![
                                Selection::Field(Field {
                                    alias: None,
                                    name: "id".to_string(),
                                    arguments: Vec::new(),
                                    directives: Vec::new(),
                                    selection_set: None,
                                }),
                                Selection::Field(Field {
                                    alias: None,
                                    name: "name".to_string(),
                                    arguments: Vec::new(),
                                    directives: Vec::new(),
                                    selection_set: None,
                                }),
                            ],
                        }),
                    })],
                },
            })],
        };

        assert_eq!(doc.definitions.len(), 1);
    }

    #[test]
    fn test_input_value_conversion() {
        let int_val = InputValue::Int(42);
        assert_eq!(int_val.to_value(), Value::Int64(42));

        let str_val = InputValue::String("hello".to_string());
        assert_eq!(str_val.to_value(), Value::String("hello".into()));

        let bool_val = InputValue::Boolean(true);
        assert_eq!(bool_val.to_value(), Value::Bool(true));
    }
}
