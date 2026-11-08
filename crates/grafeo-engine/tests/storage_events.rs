//! With the `tracing` feature, the events of every crate below the engine reach a subscriber: a
//! storage event (the warning of a checkpoint whose free-space probe fails) is captured. The macros check
//! `feature = "tracing"` in the crate that calls them, so a crate whose own `tracing` feature the
//! engine does not enable logged nothing, its WAL flush failures included. CI runs this binary
//! with the engine's tracing and storage features only (`--all-features` turns every crate's
//! feature on and would hide a missing link); the class test reads the manifests and needs no
//! build of its own.
//!
//! A test binary of its own: it installs the global subscriber.
//!
//! ```bash
//! cargo test -p grafeo-engine --no-default-features \
//!     --features gql,wal,grafeo-file,tracing,testing-crash-injection --test storage_events
//! ```

use std::path::Path;
use std::sync::Mutex;

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};

/// The targets and messages of every event, from any thread.
static EVENTS: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());

struct Capture;

struct Message(String);

impl Visit for Message {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.0 = format!("{value:?}");
        }
    }
}

impl Subscriber for Capture {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _span: &Id, _values: &Record<'_>) {}

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let mut message = Message(String::new());
        event.record(&mut message);
        EVENTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((event.metadata().target().to_string(), message.0));
    }

    fn enter(&self, _span: &Id) {}

    fn exit(&self, _span: &Id) {}
}

/// A checkpoint whose free-space probe fails goes ahead without the check, and grafeo-storage
/// warns about it: the warning reaches the subscriber. The probe's fake answers on the calling
/// thread, so the test checkpoints itself rather than leaving it to the checkpoint thread.
#[test]
fn a_storage_event_reaches_the_subscriber() {
    tracing::subscriber::set_global_default(Capture).expect("the only subscriber of this binary");
    let dir = tempfile::tempdir().unwrap();
    let db = grafeo_engine::GrafeoDB::open(dir.path().join("cities.grafeo")).unwrap();
    db.execute("INSERT (:City {name: 'Amsterdam'})").unwrap();
    grafeo_storage::file::space::with_free_space(None, || db.wal_checkpoint())
        .expect("a checkpoint goes ahead without the check");
    db.close().unwrap();
    let events = EVENTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert!(
        events
            .iter()
            .any(|(target, message)| target.starts_with("grafeo_storage")
                && message.contains("free disk space")),
        "no grafeo_storage warning about the free space among {} captured: {events:#?}",
        events.len()
    );
}

/// The `tracing` features a manifest's `tracing` feature turns on, as written.
fn tracing_feature(manifest: &str) -> Option<String> {
    manifest
        .lines()
        .find(|line| line.trim_start().starts_with("tracing = ["))
        .map(str::to_string)
}

/// The grafeo crates of a manifest's `[dependencies]`.
fn grafeo_dependencies(manifest: &str) -> Vec<String> {
    manifest
        .lines()
        .skip_while(|line| line.trim() != "[dependencies]")
        .skip(1)
        .take_while(|line| !line.starts_with('['))
        .filter_map(|line| {
            let name = line.split(['=', '.', ' ']).next()?;
            name.starts_with("grafeo-").then(|| name.to_string())
        })
        .collect()
}

/// Every grafeo crate the engine depends on that has a `tracing` feature is turned on by the
/// engine's: a crate added with one, and not linked, fails here before its events go missing.
#[test]
fn the_engines_tracing_turns_on_every_crates_tracing() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let manifest = std::fs::read_to_string(crates.join("grafeo-engine/Cargo.toml")).unwrap();
    let engine = tracing_feature(&manifest).expect("the engine has a tracing feature");
    let dependencies = grafeo_dependencies(&manifest);
    assert!(dependencies.len() >= 4, "{dependencies:?}");
    let mut checked = Vec::new();
    for name in &dependencies {
        let manifest = std::fs::read_to_string(crates.join(name).join("Cargo.toml")).unwrap();
        if tracing_feature(&manifest).is_none() {
            continue;
        }
        checked.push(name.as_str());
        assert!(
            engine.contains(&format!("\"{name}/tracing\""))
                || engine.contains(&format!("\"{name}?/tracing\"")),
            "{name} has a tracing feature the engine's does not turn on: {engine}"
        );
    }
    assert!(checked.contains(&"grafeo-storage"), "{checked:?}");
}
