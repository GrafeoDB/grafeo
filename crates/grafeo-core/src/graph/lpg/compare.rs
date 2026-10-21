//! The comparison of a zone map check.

/// Comparison operators used for zone map predicate checks.
///
/// These map directly to GQL comparison operators like `=`, `<`, `>=`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompareOp {
    /// Equal to value.
    Eq,
    /// Not equal to value.
    Ne,
    /// Less than value.
    Lt,
    /// Less than or equal to value.
    Le,
    /// Greater than value.
    Gt,
    /// Greater than or equal to value.
    Ge,
}
