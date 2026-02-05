//! Grafeo CLI - Admin tool for Grafeo graph databases.
//!
//! A focused admin CLI for operators and DevOps. The query API is for building
//! applications; the CLI is for inspection, backup, and maintenance.

mod commands;
mod output;

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// Grafeo database administration tool.
///
/// A command-line interface for inspecting, backing up, and maintaining
/// Grafeo graph databases.
#[derive(Parser)]
#[command(name = "grafeo")]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Output format
    #[arg(long, global = true, default_value = "table")]
    format: OutputFormat,

    /// Suppress progress and info messages
    #[arg(long, short, global = true)]
    quiet: bool,

    /// Enable verbose debug logging
    #[arg(long, short, global = true)]
    verbose: bool,
}

/// Output format options.
#[derive(Clone, Copy, ValueEnum, Default)]
enum OutputFormat {
    /// Human-readable table format (default for TTY)
    #[default]
    Table,
    /// Machine-readable JSON format
    Json,
}

/// Available commands.
#[derive(Subcommand)]
enum Commands {
    /// Display database information (counts, size, mode)
    Info {
        /// Path to the database
        path: PathBuf,
    },

    /// Show detailed statistics
    Stats {
        /// Path to the database
        path: PathBuf,
    },

    /// Display schema information (labels, edge types, property keys)
    Schema {
        /// Path to the database
        path: PathBuf,
    },

    /// Validate database integrity
    Validate {
        /// Path to the database
        path: PathBuf,
    },

    /// Manage indexes
    #[command(subcommand)]
    Index(IndexCommands),

    /// Manage backups
    #[command(subcommand)]
    Backup(BackupCommands),

    /// Export/import data
    #[command(subcommand)]
    Data(DataCommands),

    /// Manage Write-Ahead Log
    #[command(subcommand)]
    Wal(WalCommands),

    /// Compact the database
    Compact {
        /// Path to the database
        path: PathBuf,

        /// Perform a dry-run (show what would be done)
        #[arg(long)]
        dry_run: bool,
    },
}

/// Index management commands.
#[derive(Subcommand)]
enum IndexCommands {
    /// List all indexes
    List {
        /// Path to the database
        path: PathBuf,
    },

    /// Show index statistics
    Stats {
        /// Path to the database
        path: PathBuf,
    },
}

/// Backup commands.
#[derive(Subcommand)]
enum BackupCommands {
    /// Create a native backup
    Create {
        /// Path to the database
        path: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Restore from a native backup
    Restore {
        /// Path to the backup file
        backup: PathBuf,

        /// Target database path
        path: PathBuf,

        /// Overwrite if exists
        #[arg(long)]
        force: bool,
    },
}

/// Data export/import commands.
#[derive(Subcommand)]
enum DataCommands {
    /// Export data to a portable format
    Dump {
        /// Path to the database
        path: PathBuf,

        /// Output file or directory
        #[arg(short, long)]
        output: PathBuf,

        /// Export format (parquet, turtle, json)
        #[arg(long)]
        format: Option<String>,
    },

    /// Import data from a dump
    Load {
        /// Path to the dump file/directory
        input: PathBuf,

        /// Target database path
        path: PathBuf,
    },
}

/// WAL management commands.
#[derive(Subcommand)]
enum WalCommands {
    /// Show WAL status
    Status {
        /// Path to the database
        path: PathBuf,
    },

    /// Force a WAL checkpoint
    Checkpoint {
        /// Path to the database
        path: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    // Set up logging based on verbosity
    if cli.verbose {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .init();
    } else if !cli.quiet {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
    }

    let result = match cli.command {
        Commands::Info { path } => commands::info::run(&path, cli.format, cli.quiet),
        Commands::Stats { path } => commands::stats::run(&path, cli.format, cli.quiet),
        Commands::Schema { path } => commands::schema::run(&path, cli.format, cli.quiet),
        Commands::Validate { path } => commands::validate::run(&path, cli.format, cli.quiet),
        Commands::Index(cmd) => commands::index::run(cmd, cli.format, cli.quiet),
        Commands::Backup(cmd) => commands::backup::run(cmd, cli.format, cli.quiet),
        Commands::Data(cmd) => commands::data::run(cmd, cli.format, cli.quiet),
        Commands::Wal(cmd) => commands::wal::run(cmd, cli.format, cli.quiet),
        Commands::Compact { path, dry_run } => {
            commands::compact::run(&path, dry_run, cli.format, cli.quiet)
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
