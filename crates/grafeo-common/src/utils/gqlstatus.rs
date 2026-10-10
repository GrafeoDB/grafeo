//! GQLSTATUS diagnostic codes per ISO/IEC 39075:2024, sec 23.
//!
//! Every query result carries a [`GqlStatus`] code (5-character string like `"00000"`)
//! and, on errors, an optional [`DiagnosticRecord`] with operation context.

use std::fmt;

/// A GQLSTATUS code: 2-character class + 3-character subclass.
///
/// Standard classes:
/// - `00` successful completion
/// - `01` warning
/// - `02` no data
/// - `22` data exception
/// - `25` invalid transaction state
/// - `40` transaction rollback
/// - `42` syntax error or access rule violation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GqlStatus {
    code: [u8; 5],
}

impl GqlStatus {
    // ── Successful completion (class 00) ──────────────────────────────

    /// `00000` - successful completion, no subclass.
    pub const SUCCESS: Self = Self::from_bytes(*b"00000");

    /// `00001` - successful completion, omitted result.
    pub const SUCCESS_OMITTED_RESULT: Self = Self::from_bytes(*b"00001");

    // ── Warning (class 01) ────────────────────────────────────────────

    /// `01000` - warning, no subclass.
    pub const WARNING: Self = Self::from_bytes(*b"01000");

    /// `01004` - warning: string data, right truncation.
    pub const WARNING_STRING_TRUNCATION: Self = Self::from_bytes(*b"01004");

    /// `01G03` - warning: graph does not exist.
    pub const WARNING_GRAPH_NOT_EXIST: Self = Self::from_bytes(*b"01G03");

    /// `01G04` - warning: graph type does not exist.
    pub const WARNING_GRAPH_TYPE_NOT_EXIST: Self = Self::from_bytes(*b"01G04");

    /// `01G11` - warning: null value eliminated in set function.
    pub const WARNING_NULL_ELIMINATED: Self = Self::from_bytes(*b"01G11");

    // ── No data (class 02) ────────────────────────────────────────────

    /// `02000` - no data.
    pub const NO_DATA: Self = Self::from_bytes(*b"02000");

    // ── Data exception (class 22) ─────────────────────────────────────

    /// `22000` - data exception, no subclass.
    pub const DATA_EXCEPTION: Self = Self::from_bytes(*b"22000");

    /// `22001` - string data, right truncation.
    pub const DATA_STRING_TRUNCATION: Self = Self::from_bytes(*b"22001");

    /// `22003` - numeric value out of range.
    pub const DATA_NUMERIC_OUT_OF_RANGE: Self = Self::from_bytes(*b"22003");

    /// `22004` - null value not allowed.
    pub const DATA_NULL_NOT_ALLOWED: Self = Self::from_bytes(*b"22004");

    /// `22012` - division by zero.
    pub const DATA_DIVISION_BY_ZERO: Self = Self::from_bytes(*b"22012");

    /// `22G02` - negative limit value.
    pub const DATA_NEGATIVE_LIMIT: Self = Self::from_bytes(*b"22G02");

    /// `22G03` - invalid value type.
    pub const DATA_INVALID_VALUE_TYPE: Self = Self::from_bytes(*b"22G03");

    /// `22G04` - values not comparable.
    pub const DATA_VALUES_NOT_COMPARABLE: Self = Self::from_bytes(*b"22G04");

    // ── Invalid transaction state (class 25) ──────────────────────────

    /// `25000` - invalid transaction state, no subclass.
    pub const INVALID_TX_STATE: Self = Self::from_bytes(*b"25000");

    /// `25G01` - active GQL-transaction.
    pub const INVALID_TX_ACTIVE: Self = Self::from_bytes(*b"25G01");

    /// `25G03` - read-only GQL-transaction.
    pub const INVALID_TX_READ_ONLY: Self = Self::from_bytes(*b"25G03");

    // ── Invalid transaction termination (class 2D) ────────────────────

    /// `2D000` - invalid transaction termination.
    pub const INVALID_TX_TERMINATION: Self = Self::from_bytes(*b"2D000");

    // ── Transaction rollback (class 40) ───────────────────────────────

    /// `40000` - transaction rollback, no subclass.
    pub const TX_ROLLBACK: Self = Self::from_bytes(*b"40000");

    /// `40003` - statement completion unknown.
    pub const TX_COMPLETION_UNKNOWN: Self = Self::from_bytes(*b"40003");

    // ── Syntax error or access rule violation (class 42) ──────────────

    /// `42000` - syntax error or access rule violation, no subclass.
    pub const SYNTAX_ERROR: Self = Self::from_bytes(*b"42000");

    /// `42001` - invalid syntax.
    pub const SYNTAX_INVALID: Self = Self::from_bytes(*b"42001");

    /// `42002` - invalid reference.
    pub const SYNTAX_INVALID_REFERENCE: Self = Self::from_bytes(*b"42002");

    // ── Dependent object error (class G1) ─────────────────────────────

    /// `G1000` - dependent object error, no subclass.
    pub const DEPENDENT_OBJECT_ERROR: Self = Self::from_bytes(*b"G1000");

    // ── Graph type violation (class G2) ───────────────────────────────

    /// `G2000` - graph type violation.
    pub const GRAPH_TYPE_VIOLATION: Self = Self::from_bytes(*b"G2000");

    // ── Constructors ──────────────────────────────────────────────────

    /// Creates a `GqlStatus` from a 5-byte array. Panics if bytes are not ASCII.
    #[must_use]
    const fn from_bytes(bytes: [u8; 5]) -> Self {
        Self { code: bytes }
    }

    /// Creates a `GqlStatus` from a 5-character string slice.
    ///
    /// Returns `None` if the string is not exactly 5 ASCII characters.
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.len() != 5 {
            return None;
        }
        if !bytes.iter().all(|b| b.is_ascii_alphanumeric()) {
            return None;
        }
        Some(Self {
            code: [bytes[0], bytes[1], bytes[2], bytes[3], bytes[4]],
        })
    }

    /// Returns the 5-character GQLSTATUS code as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        // Safety: all codes are constructed from ASCII bytes
        std::str::from_utf8(&self.code).unwrap_or("?????")
    }

    /// Returns the 2-character class code (e.g., `"00"`, `"42"`).
    #[must_use]
    pub fn class_code(&self) -> &str {
        &self.as_str()[..2]
    }

    /// Returns the 3-character subclass code (e.g., `"000"`, `"001"`).
    #[must_use]
    pub fn subclass_code(&self) -> &str {
        &self.as_str()[2..]
    }

    /// True if this is a successful completion (class `00`).
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.code[0] == b'0' && self.code[1] == b'0'
    }

    /// True if this is a warning (class `01`).
    #[must_use]
    pub fn is_warning(&self) -> bool {
        self.code[0] == b'0' && self.code[1] == b'1'
    }

    /// True if this is a no-data condition (class `02`).
    #[must_use]
    pub fn is_no_data(&self) -> bool {
        self.code[0] == b'0' && self.code[1] == b'2'
    }

    /// True if this is an exception condition (not success, warning, or no-data).
    #[must_use]
    pub fn is_exception(&self) -> bool {
        !self.is_success() && !self.is_warning() && !self.is_no_data()
    }
}

impl fmt::Display for GqlStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Maps a Grafeo [`super::error::Error`] to a GQLSTATUS code.
impl From<&super::error::Error> for GqlStatus {
    fn from(err: &super::error::Error) -> Self {
        use super::error::{Error, QueryErrorKind, TransactionError};

        match err {
            Error::Query(q) => match q.kind {
                QueryErrorKind::Lexer | QueryErrorKind::Syntax => GqlStatus::SYNTAX_INVALID,
                QueryErrorKind::Semantic => GqlStatus::SYNTAX_INVALID_REFERENCE,
                QueryErrorKind::Optimization => GqlStatus::SYNTAX_ERROR,
                QueryErrorKind::Execution => GqlStatus::DATA_EXCEPTION,
                QueryErrorKind::Timeout => GqlStatus::DATA_EXCEPTION,
            },
            Error::Transaction(t) => match t {
                TransactionError::ReadOnly => GqlStatus::INVALID_TX_READ_ONLY,
                TransactionError::InvalidState(_) => GqlStatus::INVALID_TX_STATE,
                TransactionError::Aborted
                | TransactionError::Conflict
                | TransactionError::WriteConflict(_) => GqlStatus::TX_ROLLBACK,
                TransactionError::SerializationFailure(_) => GqlStatus::TX_ROLLBACK,
                TransactionError::Deadlock => GqlStatus::TX_ROLLBACK,
                TransactionError::Timeout | TransactionError::DatabaseClosed => {
                    GqlStatus::INVALID_TX_STATE
                }
                // The status flags the database, not the refused operation
                // (which did not happen): it holds an earlier commit whose
                // completion is unknown, and must be reopened.
                TransactionError::IncompleteCommit => GqlStatus::TX_COMPLETION_UNKNOWN,
            },
            Error::TypeMismatch { .. } => GqlStatus::DATA_INVALID_VALUE_TYPE,
            Error::InvalidValue(_) => GqlStatus::DATA_EXCEPTION,
            Error::NodeNotFound(_) | Error::EdgeNotFound(_) => GqlStatus::NO_DATA,
            Error::PropertyNotFound(_) | Error::LabelNotFound(_) => {
                GqlStatus::SYNTAX_INVALID_REFERENCE
            }
            Error::Storage(_) => GqlStatus::DATA_EXCEPTION,
            Error::Serialization(_) | Error::Corruption(_) => GqlStatus::DATA_EXCEPTION,
            Error::Io(_) => GqlStatus::DATA_EXCEPTION,
            Error::Internal(_) => GqlStatus::DATA_EXCEPTION,
        }
    }
}

/// Diagnostic record attached to error conditions (ISO sec 23.2).
///
/// Contains contextual information about the operation that raised the condition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticRecord {
    /// Identifier of the operation executed (e.g., `"MATCH STATEMENT"`).
    pub operation: String,
    /// Numeric operation code per Table 9 of the spec.
    pub operation_code: i32,
    /// Current working schema reference, if any.
    pub current_schema: Option<String>,
    /// Invalid reference identifier (only for GQLSTATUS `42002`).
    pub invalid_reference: Option<String>,
}

impl DiagnosticRecord {
    /// Creates a diagnostic record for a query operation.
    #[must_use]
    pub fn for_query(operation: impl Into<String>, operation_code: i32) -> Self {
        Self {
            operation: operation.into(),
            operation_code,
            current_schema: None,
            invalid_reference: None,
        }
    }
}

/// Operation codes from ISO Table 9.
pub mod operation_codes {
    /// SESSION SET SCHEMA.
    pub const SESSION_SET_SCHEMA: i32 = 1;
    /// SESSION SET GRAPH.
    pub const SESSION_SET_GRAPH: i32 = 2;
    /// SESSION SET TIME ZONE.
    pub const SESSION_SET_TIME_ZONE: i32 = 3;
    /// SESSION RESET.
    pub const SESSION_RESET: i32 = 7;
    /// SESSION CLOSE.
    pub const SESSION_CLOSE: i32 = 8;
    /// START TRANSACTION.
    pub const START_TRANSACTION: i32 = 50;
    /// ROLLBACK.
    pub const ROLLBACK: i32 = 51;
    /// COMMIT.
    pub const COMMIT: i32 = 52;
    /// CREATE SCHEMA.
    pub const CREATE_SCHEMA: i32 = 100;
    /// DROP SCHEMA.
    pub const DROP_SCHEMA: i32 = 101;
    /// CREATE GRAPH.
    pub const CREATE_GRAPH: i32 = 200;
    /// DROP GRAPH.
    pub const DROP_GRAPH: i32 = 201;
    /// CREATE GRAPH TYPE.
    pub const CREATE_GRAPH_TYPE: i32 = 300;
    /// DROP GRAPH TYPE.
    pub const DROP_GRAPH_TYPE: i32 = 301;
    /// INSERT.
    pub const INSERT: i32 = 500;
    /// SET.
    pub const SET: i32 = 501;
    /// REMOVE.
    pub const REMOVE: i32 = 502;
    /// DELETE.
    pub const DELETE: i32 = 503;
    /// MATCH.
    pub const MATCH: i32 = 600;
    /// FILTER.
    pub const FILTER: i32 = 601;
    /// LET.
    pub const LET: i32 = 602;
    /// FOR.
    pub const FOR: i32 = 603;
    /// ORDER BY / LIMIT / SKIP.
    pub const ORDER_BY_AND_PAGE: i32 = 604;
    /// RETURN.
    pub const RETURN: i32 = 605;
    /// SELECT.
    pub const SELECT: i32 = 606;
    /// CALL procedure.
    pub const CALL_PROCEDURE: i32 = 800;
    /// Unrecognized operation.
    pub const UNRECOGNIZED: i32 = 0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gqlstatus_constants() {
        assert_eq!(GqlStatus::SUCCESS.as_str(), "00000");
        assert_eq!(GqlStatus::NO_DATA.as_str(), "02000");
        assert_eq!(GqlStatus::SYNTAX_INVALID.as_str(), "42001");
        assert_eq!(GqlStatus::TX_ROLLBACK.as_str(), "40000");
    }

    #[test]
    fn test_gqlstatus_classification() {
        assert!(GqlStatus::SUCCESS.is_success());
        assert!(!GqlStatus::SUCCESS.is_warning());
        assert!(!GqlStatus::SUCCESS.is_exception());

        assert!(GqlStatus::WARNING.is_warning());
        assert!(!GqlStatus::WARNING.is_success());
        assert!(!GqlStatus::WARNING.is_exception());

        assert!(GqlStatus::NO_DATA.is_no_data());
        assert!(!GqlStatus::NO_DATA.is_exception());

        assert!(GqlStatus::SYNTAX_ERROR.is_exception());
        assert!(GqlStatus::DATA_EXCEPTION.is_exception());
        assert!(GqlStatus::TX_ROLLBACK.is_exception());
    }

    #[test]
    fn test_gqlstatus_class_subclass() {
        assert_eq!(GqlStatus::SUCCESS.class_code(), "00");
        assert_eq!(GqlStatus::SUCCESS.subclass_code(), "000");

        assert_eq!(GqlStatus::SYNTAX_INVALID.class_code(), "42");
        assert_eq!(GqlStatus::SYNTAX_INVALID.subclass_code(), "001");

        assert_eq!(GqlStatus::DATA_DIVISION_BY_ZERO.class_code(), "22");
        assert_eq!(GqlStatus::DATA_DIVISION_BY_ZERO.subclass_code(), "012");
    }

    #[test]
    fn test_gqlstatus_from_str() {
        assert_eq!(GqlStatus::from_str("00000"), Some(GqlStatus::SUCCESS));
        assert_eq!(GqlStatus::from_str("0000"), None); // too short
        assert_eq!(GqlStatus::from_str("000000"), None); // too long
        assert_eq!(GqlStatus::from_str("00 00"), None); // has space
    }

    #[test]
    fn test_gqlstatus_display() {
        assert_eq!(format!("{}", GqlStatus::SUCCESS), "00000");
        assert_eq!(format!("{}", GqlStatus::SYNTAX_INVALID), "42001");
    }

    #[test]
    fn test_error_to_gqlstatus() {
        use super::super::error::{Error, QueryError, QueryErrorKind, TransactionError};

        let syntax_err = Error::Query(QueryError::new(QueryErrorKind::Syntax, "bad syntax"));
        assert_eq!(GqlStatus::from(&syntax_err), GqlStatus::SYNTAX_INVALID);

        let semantic_err = Error::Query(QueryError::new(QueryErrorKind::Semantic, "unknown label"));
        assert_eq!(
            GqlStatus::from(&semantic_err),
            GqlStatus::SYNTAX_INVALID_REFERENCE
        );

        let tx_err = Error::Transaction(TransactionError::ReadOnly);
        assert_eq!(GqlStatus::from(&tx_err), GqlStatus::INVALID_TX_READ_ONLY);

        let conflict = Error::Transaction(TransactionError::Conflict);
        assert_eq!(GqlStatus::from(&conflict), GqlStatus::TX_ROLLBACK);

        let type_err = Error::TypeMismatch {
            expected: "INT64".into(),
            found: "STRING".into(),
        };
        assert_eq!(
            GqlStatus::from(&type_err),
            GqlStatus::DATA_INVALID_VALUE_TYPE
        );
    }

    /// After a commit that did not complete, the database holds a commit
    /// whose completion is unknown and must be reopened: every operation it
    /// refuses for that reason gets `40003` ("completion unknown"), so a
    /// client knows to reopen, not the generic invalid state of a closed
    /// database.
    #[test]
    fn an_incomplete_commit_is_completion_unknown() {
        use super::super::error::{Error, TransactionError};

        assert_eq!(
            GqlStatus::from(&Error::Transaction(TransactionError::IncompleteCommit)),
            GqlStatus::TX_COMPLETION_UNKNOWN
        );
        assert_eq!(GqlStatus::TX_COMPLETION_UNKNOWN.as_str(), "40003");
        assert_eq!(
            GqlStatus::from(&Error::Transaction(TransactionError::DatabaseClosed)),
            GqlStatus::INVALID_TX_STATE,
            "a closed database keeps its status"
        );
    }

    #[test]
    fn test_diagnostic_record() {
        let record = DiagnosticRecord::for_query("MATCH STATEMENT", operation_codes::MATCH);
        assert_eq!(record.operation, "MATCH STATEMENT");
        assert_eq!(record.operation_code, 600);
        assert!(record.current_schema.is_none());
        assert!(record.invalid_reference.is_none());
    }
}
