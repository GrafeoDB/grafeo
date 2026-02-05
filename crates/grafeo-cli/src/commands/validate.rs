//! Database validation command.

use std::path::Path;

use anyhow::Result;
use comfy_table::{Cell, Color};
use grafeo_engine::GrafeoDB;
use serde::Serialize;

use crate::OutputFormat;
use crate::output::{self, Format};

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
    let db = GrafeoDB::open(path)?;
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
        Format::Table => {
            if !quiet {
                if output.valid {
                    println!("{}", console_format("✓ Database is valid", Color::Green));
                } else {
                    println!("{}", console_format("✗ Database has errors", Color::Red));
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

    // Return error exit code if validation failed
    if !output.valid {
        std::process::exit(1);
    }

    Ok(())
}

/// Format a string with ANSI color for console output.
fn console_format(text: &str, color: Color) -> String {
    match color {
        Color::Green => format!("\x1b[32m{}\x1b[0m", text),
        Color::Red => format!("\x1b[31m{}\x1b[0m", text),
        Color::Yellow => format!("\x1b[33m{}\x1b[0m", text),
        _ => text.to_string(),
    }
}
