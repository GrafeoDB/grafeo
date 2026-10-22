//! The value types of the value type predicate `IS TYPED`.
//!
//! ISO/IEC 39075 19.7 `<value type predicate>`: `x IS TYPED t` is true when
//! the value of `x` is a value of type `t`. The types are the property types
//! type DDL declares ([`PropertyDataType`]), so `IS TYPED` and the catalog
//! read one set of names, plus `NULL`, `PATH`, `GRAPH` and `NOTHING`, which no
//! property is.

use std::fmt;

use super::{PropertyDataType, Value};

/// A type `IS TYPED` checks a value against.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ValueType {
    /// The null type: only the null value.
    Null,
    /// A path.
    Path,
    /// A graph (`GRAPH`, ISO's `ANY [PROPERTY] GRAPH`): Grafeo has no graph
    /// values, so no value is of it.
    Graph,
    /// `NOTHING`, the empty type: no value is of it.
    Nothing,
    /// A type a property can have: `INT64`, `ZONED DATETIME`,
    /// `LIST<LIST<INT64>>`, `NODE`, `ANY` and the rest.
    Property(PropertyDataType),
}

impl ValueType {
    /// The type `name` names, as [`fmt::Display`] spells it or type DDL
    /// takes it (case does not matter), or `None` for a name that is no type.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("NULL") {
            return Some(Self::Null);
        }
        if name.eq_ignore_ascii_case("PATH") {
            return Some(Self::Path);
        }
        if name.eq_ignore_ascii_case("GRAPH") {
            return Some(Self::Graph);
        }
        if name.eq_ignore_ascii_case("NOTHING") {
            return Some(Self::Nothing);
        }
        PropertyDataType::from_known_type_name(name).map(Self::Property)
    }

    /// Whether `value` is a value of this type. The null value is only of
    /// the null type, and a list's items may be null. A node or an edge is
    /// not a [`Value`]: the caller checks [`NODE`](PropertyDataType::Node)
    /// and [`EDGE`](PropertyDataType::Edge) against the column it reads,
    /// and every other value is neither.
    #[must_use]
    pub fn is_type_of(&self, value: &Value) -> bool {
        match self {
            Self::Null => value.is_null(),
            Self::Path => matches!(value, Value::Path { .. }),
            Self::Graph | Self::Nothing => false,
            Self::Property(data_type) => !value.is_null() && is_of(data_type, value),
        }
    }
}

/// Whether the value `value`, not null, is of `data_type`.
fn is_of(data_type: &PropertyDataType, value: &Value) -> bool {
    match (data_type, value) {
        (PropertyDataType::Any, _) => true,
        (PropertyDataType::Node | PropertyDataType::Edge, _) => false,
        (PropertyDataType::ListTyped(item_type), Value::List(items)) => items
            .iter()
            .all(|item| item.is_null() || is_of(item_type, item)),
        _ => data_type.matches(value),
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => write!(f, "NULL"),
            Self::Path => write!(f, "PATH"),
            Self::Graph => write!(f, "GRAPH"),
            Self::Nothing => write!(f, "NOTHING"),
            Self::Property(data_type) => write!(f, "{data_type}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Decimal, Timestamp};

    fn of(name: &str, value: &Value) -> bool {
        ValueType::from_name(name)
            .unwrap_or_else(|| panic!("{name} is a type"))
            .is_type_of(value)
    }

    #[test]
    fn scalars_are_of_their_types_only() {
        assert!(of("INT64", &Value::Int64(3)));
        assert!(of("integer", &Value::Int64(3)));
        assert!(!of("FLOAT64", &Value::Int64(3)), "an integer is no float");
        assert!(of("FLOAT64", &Value::Float64(1.5)));
        assert!(of("STRING", &Value::from("Alix")));
        assert!(!of("STRING", &Value::Int64(3)));
        assert!(of("DECIMAL", &Value::Decimal(Decimal::from_i64(3))));
        assert!(of(
            "LOCAL DATETIME",
            &Value::Timestamp(Timestamp::from_secs(0))
        ));
        assert!(of("DATETIME", &Value::Timestamp(Timestamp::from_secs(0))));
        assert!(!of(
            "ZONED DATETIME",
            &Value::Timestamp(Timestamp::from_secs(0))
        ));
    }

    #[test]
    fn lists_check_every_item() {
        let nested = Value::List(vec![Value::List(vec![Value::Int64(1)].into())].into());
        assert!(of("LIST<LIST<INT64>>", &nested));
        assert!(!of("LIST<LIST<STRING>>", &nested));
        assert!(of("LIST<LIST>", &nested));
        assert!(of("LIST", &nested));
        let mixed = Value::List(vec![Value::Int64(1), Value::from("x")].into());
        assert!(!of("LIST<INT64>", &mixed));
        let with_null = Value::List(vec![Value::Int64(1), Value::Null].into());
        assert!(of("LIST<INT64>", &with_null), "a list item may be null");
    }

    #[test]
    fn null_is_only_of_the_null_type() {
        assert!(of("NULL", &Value::Null));
        assert!(!of("INT64", &Value::Null));
        assert!(!of("ANY", &Value::Null));
        assert!(of("ANY", &Value::Int64(3)));
        assert!(!of("NULL", &Value::Int64(3)));
    }

    #[test]
    fn a_value_is_no_node_or_edge() {
        assert!(!of("NODE", &Value::Map(std::sync::Arc::default())));
        assert!(!of("EDGE", &Value::Int64(1)));
    }

    #[test]
    fn no_value_is_a_graph_or_of_nothing() {
        assert!(!of("GRAPH", &Value::from("Alix")));
        assert!(!of("NOTHING", &Value::Int64(3)));
        assert!(!of("NOTHING", &Value::Null));
    }

    #[test]
    fn unknown_names_are_no_type() {
        assert_eq!(ValueType::from_name("STIRNG"), None);
        assert_eq!(ValueType::from_name("LIST<STIRNG>"), None);
        assert_eq!(
            ValueType::from_name("list<list<int64>>").map(|t| t.to_string()),
            Some("LIST<LIST<INT64>>".to_string())
        );
    }
}
