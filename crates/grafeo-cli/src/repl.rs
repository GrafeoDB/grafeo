//! Interactive REPL (Read-Eval-Print Loop) for Grafeo.
//!
//! `grafeo shell <path>` launches a persistent session with history,
//! transaction tracking, and meta-commands.

use std::path::Path;

use anyhow::{Context, Result};
use grafeo_engine::GrafeoDB;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

use crate::commands::query::format_value;
use crate::output::{self, Format, format_duration_ms};
use crate::{OutputFormat, QueryLanguage};

/// REPL transaction state.
enum ReplState {
    /// No active transaction.
    Idle,
    /// Inside an explicit transaction.
    InTransaction,
}

/// Mutable display and state settings for the REPL session.
struct ReplSettings {
    state: ReplState,
    format: Format,
    show_timing: bool,
    quiet: bool,
}

/// Run the interactive REPL.
pub fn run(
    path: &Path,
    lang: QueryLanguage,
    format: OutputFormat,
    quiet: bool,
    timing: bool,
) -> Result<()> {
    let db = crate::commands::open_existing(path)?;

    // In REPL, always use table format unless explicitly set
    let fmt = if matches!(format, OutputFormat::Auto) {
        Format::Table
    } else {
        format.into()
    };

    if !quiet {
        let info = db.info();
        println!(
            "Grafeo {} - {:?} mode, {} nodes, {} edges",
            info.version, info.mode, info.node_count, info.edge_count
        );
        println!("Type :help for commands, :quit to exit.\n");
    }

    let history_path = dirs_history_path();
    let mut rl = DefaultEditor::new().context("Failed to initialize readline")?;
    if let Some(ref hp) = history_path {
        let _ = rl.load_history(hp);
    }

    let mut session = db.session();
    let mut settings = ReplSettings {
        state: ReplState::Idle,
        format: fmt,
        show_timing: timing,
        quiet,
    };

    loop {
        let prompt = match settings.state {
            ReplState::Idle => "grafeo> ",
            ReplState::InTransaction => "grafeo[tx]> ",
        };

        let line = match rl.readline(prompt) {
            Ok(line) => line,
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
            Err(e) => {
                output::error(&format!("Read error: {e}"));
                break;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let _ = rl.add_history_entry(trimmed);

        // Handle meta-commands
        if trimmed.starts_with(':') {
            match handle_meta_command(trimmed, &db, &mut session, &mut settings) {
                MetaResult::Continue => continue,
                MetaResult::Quit => break,
                MetaResult::Error(msg) => {
                    output::error(&msg);
                    continue;
                }
            }
        }

        // Handle transaction commands as plain text
        let upper = trimmed.to_uppercase();
        if upper == "BEGIN" || upper == "BEGIN TRANSACTION" {
            match session.begin_transaction() {
                Ok(()) => {
                    settings.state = ReplState::InTransaction;
                    if !settings.quiet {
                        println!("Transaction started.");
                    }
                }
                Err(e) => output::error(&e.to_string()),
            }
            continue;
        }
        if upper == "COMMIT" {
            match session.commit() {
                Ok(()) => {
                    settings.state = ReplState::Idle;
                    if !settings.quiet {
                        println!("Transaction committed.");
                    }
                }
                Err(e) => output::error(&e.to_string()),
            }
            continue;
        }
        if upper == "ROLLBACK" {
            match session.rollback() {
                Ok(()) => {
                    settings.state = ReplState::Idle;
                    if !settings.quiet {
                        println!("Transaction rolled back.");
                    }
                }
                Err(e) => output::error(&e.to_string()),
            }
            continue;
        }

        // Execute query
        let result = match lang {
            QueryLanguage::Gql => session.execute(trimmed),
            #[cfg(feature = "cypher")]
            QueryLanguage::Cypher => session.execute_cypher(trimmed),
            #[cfg(not(feature = "cypher"))]
            QueryLanguage::Cypher => {
                output::error("Cypher support not enabled");
                continue;
            }
            #[cfg(feature = "sparql")]
            QueryLanguage::Sparql => session.execute_sparql(trimmed),
            #[cfg(not(feature = "sparql"))]
            QueryLanguage::Sparql => {
                output::error("SPARQL support not enabled");
                continue;
            }
            #[cfg(feature = "sql-pgq")]
            QueryLanguage::Sql => session.execute_sql(trimmed),
            #[cfg(not(feature = "sql-pgq"))]
            QueryLanguage::Sql => {
                output::error("SQL/PGQ support not enabled");
                continue;
            }
        };

        match result {
            Ok(qr) => {
                let headers = qr.columns.clone();
                let rows: Vec<Vec<String>> = qr
                    .rows()
                    .iter()
                    .map(|row| row.iter().map(format_value).collect())
                    .collect();

                output::print_result_table(&headers, &rows, settings.format, settings.quiet);

                if !settings.quiet {
                    let timing_str = if settings.show_timing {
                        qr.execution_time_ms
                            .map_or_else(String::new, |ms| format!(" ({})", format_duration_ms(ms)))
                    } else {
                        String::new()
                    };
                    println!(
                        "{} row{}{}",
                        rows.len(),
                        if rows.len() == 1 { "" } else { "s" },
                        timing_str
                    );
                }
            }
            Err(e) => {
                output::error(&e.to_string());
            }
        }
    }

    // Save history
    if let Some(ref hp) = history_path {
        if let Some(parent) = hp.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = rl.save_history(hp);
    }

    if !quiet {
        println!("Goodbye.");
    }

    Ok(())
}

/// Result of processing a meta-command.
enum MetaResult {
    Continue,
    Quit,
    Error(String),
}

/// Process a `:` meta-command.
fn handle_meta_command(
    cmd: &str,
    db: &GrafeoDB,
    session: &mut grafeo_engine::Session,
    settings: &mut ReplSettings,
) -> MetaResult {
    let ReplSettings {
        state,
        format,
        show_timing,
        quiet,
    } = settings;
    let quiet = *quiet;
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    let command = parts[0].to_lowercase();

    match command.as_str() {
        ":quit" | ":q" | ":exit" => MetaResult::Quit,
        ":help" | ":h" | ":?" => {
            if !quiet {
                println!("Meta-commands:");
                println!("  :help         Show this help");
                println!("  :quit         Exit the shell");
                println!("  :schema       Show database schema");
                println!("  :info         Show database info");
                println!("  :stats        Show detailed statistics");
                println!("  :memory       Show hierarchical memory breakdown");
                println!("  :format <f>   Set output format (table, json, csv)");
                println!("  :timing       Toggle query timing display");
                println!("  :begin        Start a transaction");
                println!("  :commit       Commit the current transaction");
                println!("  :rollback     Roll back the current transaction");
                println!();
                println!("Enter GQL queries directly. Use BEGIN/COMMIT/ROLLBACK for transactions.");
            }
            MetaResult::Continue
        }
        ":schema" => {
            let schema = db.schema();
            match schema {
                grafeo_engine::SchemaInfo::Lpg(lpg) if !quiet => {
                    println!("Labels:");
                    for label in &lpg.labels {
                        println!("  :{} ({})", label.name, label.count);
                    }
                    println!("Edge Types:");
                    for et in &lpg.edge_types {
                        println!("  :{} ({})", et.name, et.count);
                    }
                    println!("Property Keys:");
                    for key in &lpg.property_keys {
                        println!("  .{key}");
                    }
                }
                grafeo_engine::SchemaInfo::Rdf(rdf) if !quiet => {
                    println!("Predicates:");
                    for pred in &rdf.predicates {
                        println!("  {} ({})", pred.iri, pred.count);
                    }
                }
                _ => {}
            }
            MetaResult::Continue
        }
        ":info" => {
            let info = db.info();
            if !quiet {
                println!(
                    "Mode: {:?}, Nodes: {}, Edges: {}, WAL: {}, Version: {}",
                    info.mode, info.node_count, info.edge_count, info.wal_enabled, info.version
                );
            }
            MetaResult::Continue
        }
        ":stats" => {
            let stats = db.detailed_stats();
            if !quiet {
                println!("Nodes: {}, Edges: {}", stats.node_count, stats.edge_count);
                println!(
                    "Labels: {}, Edge Types: {}",
                    stats.label_count, stats.edge_type_count
                );
                println!(
                    "Property Keys: {}, Indexes: {}",
                    stats.property_key_count, stats.index_count
                );
                println!(
                    "Memory: {}",
                    crate::output::format_bytes(stats.memory_bytes)
                );
            }
            MetaResult::Continue
        }
        ":memory" => {
            if !quiet {
                crate::output::print_memory(&db.memory_usage());
            }
            MetaResult::Continue
        }
        ":format" => {
            if parts.len() < 2 {
                if !quiet {
                    let current = match format {
                        Format::Table => "table",
                        Format::Json => "json",
                        Format::Csv => "csv",
                    };
                    println!("Current format: {current}");
                    println!("Usage: :format <table|json|csv>");
                }
                return MetaResult::Continue;
            }
            match parts[1].to_lowercase().as_str() {
                "table" => *format = Format::Table,
                "json" => *format = Format::Json,
                "csv" => *format = Format::Csv,
                other => return MetaResult::Error(format!("Unknown format: {other}")),
            }
            if !quiet {
                println!("Output format set to: {}", parts[1].to_lowercase());
            }
            MetaResult::Continue
        }
        ":timing" => {
            if parts.len() < 2 {
                // Toggle
                *show_timing = !*show_timing;
            } else {
                match parts[1].to_lowercase().as_str() {
                    "on" | "true" | "1" => *show_timing = true,
                    "off" | "false" | "0" => *show_timing = false,
                    other => {
                        return MetaResult::Error(format!("Unknown value: {other} (use on/off)"));
                    }
                }
            }
            if !quiet {
                println!("Timing: {}", if *show_timing { "on" } else { "off" });
            }
            MetaResult::Continue
        }
        ":begin" => {
            match session.begin_transaction() {
                Ok(()) => {
                    *state = ReplState::InTransaction;
                    if !quiet {
                        println!("Transaction started.");
                    }
                }
                Err(e) => return MetaResult::Error(e.to_string()),
            }
            MetaResult::Continue
        }
        ":commit" => {
            match session.commit() {
                Ok(()) => {
                    *state = ReplState::Idle;
                    if !quiet {
                        println!("Transaction committed.");
                    }
                }
                Err(e) => return MetaResult::Error(e.to_string()),
            }
            MetaResult::Continue
        }
        ":rollback" => {
            match session.rollback() {
                Ok(()) => {
                    *state = ReplState::Idle;
                    if !quiet {
                        println!("Transaction rolled back.");
                    }
                }
                Err(e) => return MetaResult::Error(e.to_string()),
            }
            MetaResult::Continue
        }
        _ => MetaResult::Error(format!("Unknown command: {command}. Type :help for help.")),
    }
}

/// Get the history file path.
fn dirs_history_path() -> Option<std::path::PathBuf> {
    // Use %APPDATA%/grafeo/history on Windows, ~/.config/grafeo/history on Unix
    let base = if cfg!(windows) {
        std::env::var_os("APPDATA").map(std::path::PathBuf::from)
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".config"))
            })
    };
    base.map(|b| b.join("grafeo").join("history"))
}
