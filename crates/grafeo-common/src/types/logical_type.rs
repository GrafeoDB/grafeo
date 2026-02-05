//! The type system for schemas and query type checking.
//!
//! [`LogicalType`] describes the types that can appear in schemas and query
//! results. Used by the query optimizer for type inference and coercion.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Describes the type of a value or column.
///
/// Follows the GQL type system. Used for schema definitions and query type
/// checking. Supports coercion rules (e.g., Int32 can coerce to Int64).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LogicalType {
    /// Unknown or any type (used during type inference)
    Any,

    /// Null type (only value is NULL)
    Null,

    /// Boolean type
    Bool,

    /// 8-bit signed integer
    Int8,

    /// 16-bit signed integer
    Int16,

    /// 32-bit signed integer
    Int32,

    /// 64-bit signed integer
    Int64,

    /// 32-bit floating point
    Float32,

    /// 64-bit floating point
    Float64,

    /// Variable-length UTF-8 string
    String,

    /// Binary data
    Bytes,

    /// Date (year, month, day)
    Date,

    /// Time (hour, minute, second, nanosecond)
    Time,

    /// Timestamp with timezone
    Timestamp,

    /// Duration/interval
    Duration,

    /// Homogeneous list of elements
    List(Box<LogicalType>),

    /// Key-value map
    Map {
        /// Type of map keys (usually String)
        key: Box<LogicalType>,
        /// Type of map values
        value: Box<LogicalType>,
    },

    /// Struct with named fields
    Struct(Vec<(String, LogicalType)>),

    /// Node reference
    Node,

    /// Edge reference
    Edge,

    /// Path (sequence of nodes and edges)
    Path,

    /// Fixed-dimension vector of floats (for embeddings).
    ///
    /// The usize parameter is the dimension (e.g., 384, 768, 1536).
    /// Common embedding sizes: 384 (MiniLM), 768 (BERT), 1536 (OpenAI).
    Vector(usize),
}

impl LogicalType {
    /// Returns true if this type is numeric (integer or floating point).
    #[must_use]
    pub const fn is_numeric(&self) -> bool {
        matches!(
            self,
            LogicalType::Int8
                | LogicalType::Int16
                | LogicalType::Int32
                | LogicalType::Int64
                | LogicalType::Float32
                | LogicalType::Float64
        )
    }

    /// Returns true if this type is an integer type.
    #[must_use]
    pub const fn is_integer(&self) -> bool {
        matches!(
            self,
            LogicalType::Int8 | LogicalType::Int16 | LogicalType::Int32 | LogicalType::Int64
        )
    }

    /// Returns true if this type is a floating point type.
    #[must_use]
    pub const fn is_float(&self) -> bool {
        matches!(self, LogicalType::Float32 | LogicalType::Float64)
    }

    /// Returns true if this type is a temporal type.
    #[must_use]
    pub const fn is_temporal(&self) -> bool {
        matches!(
            self,
            LogicalType::Date | LogicalType::Time | LogicalType::Timestamp | LogicalType::Duration
        )
    }

    /// Returns true if this type is a graph element type.
    #[must_use]
    pub const fn is_graph_element(&self) -> bool {
        matches!(
            self,
            LogicalType::Node | LogicalType::Edge | LogicalType::Path
        )
    }

    /// Returns true if this type is nullable (can hold NULL values).
    ///
    /// In Grafeo, all types except Null itself are nullable by default.
    #[must_use]
    pub const fn is_nullable(&self) -> bool {
        true
    }

    /// Returns the element type if this is a List, otherwise None.
    #[must_use]
    pub fn list_element_type(&self) -> Option<&LogicalType> {
        match self {
            LogicalType::List(elem) => Some(elem),
            _ => None,
        }
    }

    /// Returns true if this type is a vector type.
    #[must_use]
    pub const fn is_vector(&self) -> bool {
        matches!(self, LogicalType::Vector(_))
    }

    /// Returns the vector dimensions if this is a Vector type.
    #[must_use]
    pub const fn vector_dimensions(&self) -> Option<usize> {
        match self {
            LogicalType::Vector(dim) => Some(*dim),
            _ => None,
        }
    }

    /// Checks if a value of `other` type can be implicitly coerced to this type.
    #[must_use]
    pub fn can_coerce_from(&self, other: &LogicalType) -> bool {
        if self == other {
            return true;
        }

        // Any accepts everything
        if matches!(self, LogicalType::Any) {
            return true;
        }

        // Null coerces to any nullable type
        if matches!(other, LogicalType::Null) && self.is_nullable() {
            return true;
        }

        // Numeric coercion: smaller integers coerce to larger
        match (other, self) {
            (LogicalType::Int8, LogicalType::Int16 | LogicalType::Int32 | LogicalType::Int64) => {
                true
            }
            (LogicalType::Int16, LogicalType::Int32 | LogicalType::Int64) => true,
            (LogicalType::Int32, LogicalType::Int64) => true,
            (LogicalType::Float32, LogicalType::Float64) => true,
            // Integers coerce to floats
            (
                LogicalType::Int8 | LogicalType::Int16 | LogicalType::Int32,
                LogicalType::Float32 | LogicalType::Float64,
            ) => true,
            (LogicalType::Int64, LogicalType::Float64) => true,
            _ => false,
        }
    }

    /// Returns the common supertype of two types, if one exists.
    #[must_use]
    pub fn common_type(&self, other: &LogicalType) -> Option<LogicalType> {
        if self == other {
            return Some(self.clone());
        }

        // Handle Any
        if matches!(self, LogicalType::Any) {
            return Some(other.clone());
        }
        if matches!(other, LogicalType::Any) {
            return Some(self.clone());
        }

        // Handle Null
        if matches!(self, LogicalType::Null) {
            return Some(other.clone());
        }
        if matches!(other, LogicalType::Null) {
            return Some(self.clone());
        }

        // Numeric promotion
        if self.is_numeric() && other.is_numeric() {
            // Float64 is the ultimate numeric type
            if self.is_float() || other.is_float() {
                return Some(LogicalType::Float64);
            }
            // Otherwise promote to largest integer
            return Some(LogicalType::Int64);
        }

        None
    }
}

impl fmt::Display for LogicalType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogicalType::Any => write!(f, "ANY"),
            LogicalType::Null => write!(f, "NULL"),
            LogicalType::Bool => write!(f, "BOOL"),
            LogicalType::Int8 => write!(f, "INT8"),
            LogicalType::Int16 => write!(f, "INT16"),
            LogicalType::Int32 => write!(f, "INT32"),
            LogicalType::Int64 => write!(f, "INT64"),
            LogicalType::Float32 => write!(f, "FLOAT32"),
            LogicalType::Float64 => write!(f, "FLOAT64"),
            LogicalType::String => write!(f, "STRING"),
            LogicalType::Bytes => write!(f, "BYTES"),
            LogicalType::Date => write!(f, "DATE"),
            LogicalType::Time => write!(f, "TIME"),
            LogicalType::Timestamp => write!(f, "TIMESTAMP"),
            LogicalType::Duration => write!(f, "DURATION"),
            LogicalType::List(elem) => write!(f, "LIST<{elem}>"),
            LogicalType::Map { key, value } => write!(f, "MAP<{key}, {value}>"),
            LogicalType::Struct(fields) => {
                write!(f, "STRUCT<")?;
                for (i, (name, ty)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{name}: {ty}")?;
                }
                write!(f, ">")
            }
            LogicalType::Node => write!(f, "NODE"),
            LogicalType::Edge => write!(f, "EDGE"),
            LogicalType::Path => write!(f, "PATH"),
            LogicalType::Vector(dim) => write!(f, "VECTOR({dim})"),
        }
    }
}

impl Default for LogicalType {
    fn default() -> Self {
        LogicalType::Any
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numeric_checks() {
        assert!(LogicalType::Int64.is_numeric());
        assert!(LogicalType::Float64.is_numeric());
        assert!(!LogicalType::String.is_numeric());

        assert!(LogicalType::Int64.is_integer());
        assert!(!LogicalType::Float64.is_integer());

        assert!(LogicalType::Float64.is_float());
        assert!(!LogicalType::Int64.is_float());
    }

    #[test]
    fn test_coercion() {
        // Same type always coerces
        assert!(LogicalType::Int64.can_coerce_from(&LogicalType::Int64));

        // Null coerces to anything
        assert!(LogicalType::Int64.can_coerce_from(&LogicalType::Null));
        assert!(LogicalType::String.can_coerce_from(&LogicalType::Null));

        // Integer widening
        assert!(LogicalType::Int64.can_coerce_from(&LogicalType::Int32));
        assert!(LogicalType::Int32.can_coerce_from(&LogicalType::Int16));
        assert!(!LogicalType::Int32.can_coerce_from(&LogicalType::Int64));

        // Float widening
        assert!(LogicalType::Float64.can_coerce_from(&LogicalType::Float32));

        // Int to float
        assert!(LogicalType::Float64.can_coerce_from(&LogicalType::Int64));
        assert!(LogicalType::Float32.can_coerce_from(&LogicalType::Int32));
    }

    #[test]
    fn test_common_type() {
        // Same types
        assert_eq!(
            LogicalType::Int64.common_type(&LogicalType::Int64),
            Some(LogicalType::Int64)
        );

        // Numeric promotion
        assert_eq!(
            LogicalType::Int32.common_type(&LogicalType::Int64),
            Some(LogicalType::Int64)
        );
        assert_eq!(
            LogicalType::Int64.common_type(&LogicalType::Float64),
            Some(LogicalType::Float64)
        );

        // Null handling
        assert_eq!(
            LogicalType::Null.common_type(&LogicalType::String),
            Some(LogicalType::String)
        );

        // Incompatible types
        assert_eq!(LogicalType::String.common_type(&LogicalType::Int64), None);
    }

    #[test]
    fn test_display() {
        assert_eq!(LogicalType::Int64.to_string(), "INT64");
        assert_eq!(
            LogicalType::List(Box::new(LogicalType::String)).to_string(),
            "LIST<STRING>"
        );
        assert_eq!(
            LogicalType::Map {
                key: Box::new(LogicalType::String),
                value: Box::new(LogicalType::Int64)
            }
            .to_string(),
            "MAP<STRING, INT64>"
        );
    }

    #[test]
    fn test_vector_type() {
        let v384 = LogicalType::Vector(384);
        let v768 = LogicalType::Vector(768);
        let v1536 = LogicalType::Vector(1536);

        // Type checks
        assert!(v384.is_vector());
        assert!(v768.is_vector());
        assert!(!LogicalType::Float64.is_vector());
        assert!(!LogicalType::List(Box::new(LogicalType::Float32)).is_vector());

        // Dimensions
        assert_eq!(v384.vector_dimensions(), Some(384));
        assert_eq!(v768.vector_dimensions(), Some(768));
        assert_eq!(v1536.vector_dimensions(), Some(1536));
        assert_eq!(LogicalType::Float64.vector_dimensions(), None);

        // Display
        assert_eq!(v384.to_string(), "VECTOR(384)");
        assert_eq!(v768.to_string(), "VECTOR(768)");
        assert_eq!(v1536.to_string(), "VECTOR(1536)");

        // Equality
        assert_eq!(LogicalType::Vector(384), LogicalType::Vector(384));
        assert_ne!(LogicalType::Vector(384), LogicalType::Vector(768));

        // Not numeric
        assert!(!v384.is_numeric());
        assert!(!v384.is_integer());
        assert!(!v384.is_float());
    }
}
