//! Database validation command.

use std::path::Path;

use anyhow::Result;
use comfy_table::{Cell, Color};
use serde::Serialize;

use crate::OutputFormat;
use crate::output::{self, Format};

/// Sentinel error indicating the database failed validation.
///
/// Main maps this to exit code 2.
#[derive(Debug)]
pub struct ValidationFailed;

impl std::fmt::Display for ValidationFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("validation failed")
    }
}

impl std::error::Error for ValidationFailed {}

/// Validation result output.
#[derive(Serialize)]
struct ValidationOutput {
    valid: bool,
    error_count: usize,
    warning_count: usize,
    errors: Vec<ErrorOutput>,
    warnings: Vec<WarningOutput>,
}

/// Error output.
#[derive(Serialize)]
struct ErrorOutput {
    code: String,
    message: String,
    context: Option<String>,
}

/// Warning output.
#[derive(Serialize)]
struct WarningOutput {
    code: String,
    message: String,
    context: Option<String>,
}

/// Run the validate command.
pub fn run(path: &Path, format: OutputFormat, quiet: bool) -> Result<()> {
    let db = super::open_existing(path)?;
    let result = db.validate();

    let output = ValidationOutput {
        valid: result.errors.is_empty(),
        error_count: result.errors.len(),
        warning_count: result.warnings.len(),
        errors: result
            .errors
            .iter()
            .map(|e| ErrorOutput {
                code: e.code.clone(),
                message: e.message.clone(),
                context: e.context.clone(),
            })
            .collect(),
        warnings: result
            .warnings
            .iter()
            .map(|w| WarningOutput {
                code: w.code.clone(),
                message: w.message.clone(),
                context: w.context.clone(),
            })
            .collect(),
    };

    let fmt: Format = format.into();
    match fmt {
        Format::Json => {
            if !quiet {
                println!("{}", serde_json::to_string_pretty(&output)?);
            }
        }
        Format::Csv => {
            if !quiet {
                println!("severity,code,message,context");
                for error in &output.errors {
                    println!(
                        "error,{},{},{}",
                        error.code,
                        error.message,
                        error.context.as_deref().unwrap_or("")
                    );
                }
                for warning in &output.warnings {
                    println!(
                        "warning,{},{},{}",
                        warning.code,
                        warning.message,
                        warning.context.as_deref().unwrap_or("")
                    );
                }
            }
        }
        Format::Table => {
            if !quiet {
                if output.valid {
                    println!("{}", output::colored("Database is valid", Color::Green));
                } else {
                    println!("{}", output::colored("Database has errors", Color::Red));
                }

                println!(
                    "\nErrors: {}, Warnings: {}\n",
                    output.error_count, output.warning_count
                );

                if !output.errors.is_empty() {
                    let mut table = output::create_table();
                    output::add_header(&mut table, &["Code", "Message", "Context"]);
                    for error in &output.errors {
                        table.add_row(vec![
                            Cell::new(&error.code).fg(Color::Red),
                            Cell::new(&error.message),
                            Cell::new(error.context.as_deref().unwrap_or("-")),
                        ]);
                    }
                    println!("Errors:\n{table}\n");
                }

                if !output.warnings.is_empty() {
                    let mut table = output::create_table();
                    output::add_header(&mut table, &["Code", "Message", "Context"]);
                    for warning in &output.warnings {
                        table.add_row(vec![
                            Cell::new(&warning.code).fg(Color::Yellow),
                            Cell::new(&warning.message),
                            Cell::new(warning.context.as_deref().unwrap_or("-")),
                        ]);
                    }
                    println!("Warnings:\n{table}");
                }
            }
        }
    }

    // Return error if validation failed (main maps this to exit code 2)
    if !output.valid {
        anyhow::bail!(ValidationFailed);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_failed_display() {
        let err = ValidationFailed;
        assert_eq!(err.to_string(), "validation failed");
    }

    #[test]
    fn test_validation_failed_is_error() {
        let err: Box<dyn std::error::Error> = Box::new(ValidationFailed);
        assert_eq!(err.to_string(), "validation failed");
    }

    #[test]
    fn test_validation_failed_downcast() {
        let err: anyhow::Error = ValidationFailed.into();
        assert!(err.downcast_ref::<ValidationFailed>().is_some());
    }
}
