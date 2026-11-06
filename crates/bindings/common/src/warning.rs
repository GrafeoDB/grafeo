//! Query warnings as every binding shows them: code, kind and message.
//!
//! This is the one place a binding reads [`QueryWarning`]: each binding maps [`Warning`] to its
//! own record type field by field, with no `match` on a code or a kind, so a new warning in the
//! registry needs no binding change.

use grafeo_common::utils::warning::{QueryWarning, WarningSink};
use grafeo_engine::database::QueryResult;

/// A warning as a binding shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Warning {
    /// Its GQLSTATUS code, of class `01` (`01X01`).
    pub code: String,
    /// `deprecation` or `notice` (more kinds may come).
    pub kind: &'static str,
    /// What it says, naming the replacement of a deprecated form.
    pub message: String,
}

impl Warning {
    /// Whether it is a deprecation, which Python raises as a `DeprecationWarning`.
    #[must_use]
    pub fn is_deprecation(&self) -> bool {
        self.kind == grafeo_common::utils::warning::WarningKind::Deprecation.as_str()
    }

    /// The warning as a JSON object: `code`, `kind` and `message`.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "code": self.code,
            "kind": self.kind,
            "message": self.message,
        })
    }
}

impl From<&QueryWarning> for Warning {
    fn from(warning: &QueryWarning) -> Self {
        Self {
            code: warning.code.as_str().to_string(),
            kind: warning.kind.as_str(),
            message: warning.message.clone(),
        }
    }
}

/// The warnings of `result`, ordered by code; empty when it raised none.
#[must_use]
pub fn of_result(result: &QueryResult) -> Vec<Warning> {
    of_warnings(result.warnings())
}

/// The warnings raised into `sink` so far, ordered by code (a stream's, see
/// `OwnedResultStream::warning_sink`).
#[must_use]
pub fn of_sink(sink: &WarningSink) -> Vec<Warning> {
    of_warnings(&sink.raised().to_query_warnings())
}

/// `warnings` as a binding shows them, in their order.
#[must_use]
pub fn of_warnings(warnings: &[QueryWarning]) -> Vec<Warning> {
    warnings.iter().map(Warning::from).collect()
}
