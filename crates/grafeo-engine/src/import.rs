//! Import of CSV and JSON Lines files as nodes.
//!
//! [`import_nodes`] turns each row of a file into one node with a given
//! label, through a GQL `LOAD DATA` statement. It is the one implementation
//! behind the Python `import_csv` and `import_jsonl`, the Node.js `importCsv`
//! and `importJsonl`, and the CLI `grafeo import csv` and `grafeo import jsonl`.
//!
//! # Stability
//!
//! The `grafeo` crate re-exports this module. Its items are stable unless
//! their documentation has a `# Stability: Experimental` section: see
//! <https://grafeo.dev/stability/>.

use std::path::Path;

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, Result};

use crate::session::Session;

/// The format of a file [`import_nodes`] reads.
///
/// # Stability: Experimental
///
/// New in 0.6.0; formats and options may be added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImportFormat {
    /// Comma-separated values. With `headers`, the first record names the
    /// columns; without, the columns are `column0`, `column1`, and so on.
    /// `delimiter` is the field separator (a comma when `None`).
    Csv {
        /// Whether the first record holds the column names.
        headers: bool,
        /// The field separator, a comma when `None`.
        delimiter: Option<char>,
    },
    /// JSON Lines: one JSON object per line, whose keys name the properties.
    Jsonl,
}

/// Creates one node labeled `label` for each row of the file at `path` and
/// returns how many it created.
///
/// Each column or key becomes a property under its own name, as the file
/// spells it (`home city`, `geo.lat`); an empty CSV field or a JSON null
/// sets no property. CSV values are strings; JSON values keep their types.
/// The label is used as given, quoted, so it may hold spaces or punctuation.
/// A CSV without headers names its columns `column0`, `column1`, and so on,
/// as many as its widest record has.
///
/// The rows go in through the session, so an open transaction holds them
/// until it commits.
///
/// # Errors
///
/// Fails when the file cannot be read, when `label` is empty, or when the
/// statement fails (a JSON line that is no object, or a build without the
/// `jsonl-import` feature for [`ImportFormat::Jsonl`]).
///
/// # Example
///
/// ```no_run
/// use grafeo_engine::GrafeoDB;
/// use grafeo_engine::import::{import_nodes, ImportFormat};
///
/// let db = GrafeoDB::new_in_memory();
/// let session = db.session();
/// let format = ImportFormat::Csv { headers: true, delimiter: None };
/// let created = import_nodes(&session, "people.csv".as_ref(), "Person", format)?;
/// println!("imported {created} people");
/// # Ok::<(), grafeo_common::utils::error::Error>(())
/// ```
///
/// # Stability: Experimental
///
/// New in 0.6.0; the options may change.
pub fn import_nodes(
    session: &Session,
    path: &Path,
    label: &str,
    format: ImportFormat,
) -> Result<u64> {
    if label.is_empty() {
        return Err(Error::InvalidValue(
            "the label of imported nodes must not be empty".to_string(),
        ));
    }
    // Fail on a missing file before the statement does, with the I/O error.
    std::fs::metadata(path)?;
    let source = string_literal(&path.to_string_lossy());
    let label = quoted_name(label);
    let (load, properties) = match format {
        ImportFormat::Jsonl => (
            format!("LOAD DATA FROM {source} FORMAT JSONL AS row"),
            "row".to_string(),
        ),
        ImportFormat::Csv { headers, delimiter } => {
            let terminator = delimiter.map_or_else(String::new, |delimiter| {
                format!(
                    " FIELDTERMINATOR {}",
                    string_literal(&delimiter.to_string())
                )
            });
            if headers {
                (
                    format!("LOAD DATA FROM {source} FORMAT CSV WITH HEADERS AS row{terminator}"),
                    "row".to_string(),
                )
            } else {
                let load = format!("LOAD DATA FROM {source} FORMAT CSV AS row{terminator}");
                // The widest record decides the columns, as DuckDB sniffs
                // them; a shorter record leaves its last columns unset.
                let width = single_integer(session, &format!("{load} RETURN max(size(row))"))?;
                if width == 0 {
                    return Ok(0);
                }
                let columns = (0..width)
                    .map(|index| format!("column{index}: row[{index}]"))
                    .collect::<Vec<_>>()
                    .join(", ");
                (load, format!("{{{columns}}}"))
            }
        }
    };
    single_integer(
        session,
        &format!("{load} INSERT (n:{label}) SET n += {properties} RETURN count(n)"),
    )
}

/// The one integer `query` returns, 0 when it returns null or no row.
fn single_integer(session: &Session, query: &str) -> Result<u64> {
    let result = session.execute(query)?;
    match result.rows().first().and_then(|row| row.first()) {
        Some(Value::Int64(value)) => {
            u64::try_from(*value).map_err(|_| Error::Internal(format!("{query} returned {value}")))
        }
        Some(Value::Null) | None => Ok(0),
        Some(other) => Err(Error::Internal(format!("{query} returned {other:?}"))),
    }
}

/// `text` as a GQL string literal.
fn string_literal(text: &str) -> String {
    format!("'{}'", text.replace('\\', "\\\\").replace('\'', "\\'"))
}

/// `name` as a GQL name in backticks, which may hold any character.
fn quoted_name(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_string_literal_escapes_backslashes_and_quotes() {
        assert_eq!(
            string_literal(r"C:\data\O'Brien.csv"),
            r"'C:\\data\\O\'Brien.csv'"
        );
        assert_eq!(string_literal("plain"), "'plain'");
    }

    #[test]
    fn a_quoted_name_doubles_its_backticks() {
        assert_eq!(quoted_name("My Label"), "`My Label`");
        assert_eq!(quoted_name("a`b"), "`a``b`");
    }
}
