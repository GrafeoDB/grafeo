//! Version information command.

/// Run the version command.
pub fn run(quiet: bool) {
    if quiet {
        return;
    }

    let version = env!("CARGO_PKG_VERSION");
    println!("grafeo {version}");
    println!();

    // Build info
    println!("Build:");
    println!("  rustc:    {}", rustc_version());
    println!("  target:   {}", std::env::consts::ARCH);
    println!("  os:       {}", std::env::consts::OS);

    // Features
    let mut features = Vec::new();
    if cfg!(feature = "gql") {
        features.push("gql");
    }
    if cfg!(feature = "cypher") {
        features.push("cypher");
    }
    if cfg!(feature = "sparql") {
        features.push("sparql");
    }
    if cfg!(feature = "sql-pgq") {
        features.push("sql-pgq");
    }
    if cfg!(feature = "triple-store") {
        features.push("rdf");
    }
    let feature_str = features.join(", ");
    println!(
        "  features: {}",
        if features.is_empty() {
            "none"
        } else {
            &feature_str
        }
    );

    // Paths
    println!();
    println!("Paths:");
    let config_path = config_dir().map_or_else(|| "-".to_string(), |p| p.display().to_string());
    let history_path = history_file().map_or_else(|| "-".to_string(), |p| p.display().to_string());
    println!("  config:   {config_path}");
    println!("  history:  {history_path}");
}

/// Get the rustc version used to compile.
fn rustc_version() -> &'static str {
    option_env!("RUSTC_VERSION").unwrap_or(env!("CARGO_PKG_RUST_VERSION"))
}

/// Get the config directory path.
fn config_dir() -> Option<std::path::PathBuf> {
    let base = if cfg!(windows) {
        std::env::var_os("APPDATA").map(std::path::PathBuf::from)
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".config"))
            })
    };
    base.map(|b| b.join("grafeo"))
}

/// Get the history file path.
fn history_file() -> Option<std::path::PathBuf> {
    config_dir().map(|d| d.join("history"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rustc_version_not_empty() {
        let v = rustc_version();
        assert!(!v.is_empty(), "v is empty");
    }

    #[test]
    fn test_config_dir_returns_some() {
        // On CI or dev machines, HOME/APPDATA is usually set
        let dir = config_dir();
        if let Some(d) = dir {
            assert!(d.ends_with("grafeo"));
        }
    }

    #[test]
    fn test_history_file_ends_with_history() {
        let path = history_file();
        if let Some(p) = path {
            assert!(p.ends_with("grafeo/history") || p.ends_with("grafeo\\history"));
        }
    }
}
