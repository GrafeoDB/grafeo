//! A compacted 0.5.x database file in a build without the `compact-store`
//! feature (the `grafeo` facade's default `embedded` profile, the CLI).
//!
//! A 0.5.x file compacted by 0.5.44 or older holds the compacted base in its
//! `CompactStore` section and the deletes of base nodes and edges since
//! `compact()` in its `OverlayDeletions` section, next to the overlay, its LPG
//! section. A build without `compact-store` cannot read the first two. Served
//! alone, the overlay would miss every node and edge only the base holds, and
//! the migration would write it without the base, losing them for good. So
//! such a build refuses the file: a read-write open, a read-only open and
//! `open_in_memory` fail with an error that names the file and the feature,
//! and change nothing on disk; the file is neither read nor migrated.
//!
//! ```bash
//! cargo test -p grafeo-engine --no-default-features --features lpg,gql,grafeo-file \
//!     --test compacted_file_without_compact_store
//! cargo test -p grafeo-engine --no-default-features --features lpg,gql,wal,grafeo-file \
//!     --test compacted_file_without_compact_store
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "grafeo-file",
    not(feature = "compact-store"),
    not(miri)
))]

#[path = "common/legacy_file.rs"]
mod legacy_file;

use std::path::{Path, PathBuf};

use grafeo_common::storage::section::SectionType;
use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

/// The people and the city of each, sorted by name.
fn people(db: &GrafeoDB) -> Vec<(String, Option<String>)> {
    let text = |value: &Value| match value {
        Value::String(text) => Some(text.to_string()),
        _ => None,
    };
    db.execute("MATCH (p:Person) RETURN p.name AS name, p.city AS city ORDER BY name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| (text(&row[0]).expect("a name"), text(&row[1])))
        .collect()
}

/// An open of a database file, as a build without `compact-store` has it.
type Open = fn(&Path) -> grafeo_common::utils::error::Result<GrafeoDB>;

/// Every open of an existing database file: read-write, read-only and,
/// with the `wal` feature, `open_in_memory` (which reads the file as a
/// read-only open does).
fn opens() -> Vec<(&'static str, Open)> {
    #[cfg_attr(
        not(feature = "wal"),
        expect(unused_mut, reason = "only the `wal` feature adds `open_in_memory`")
    )]
    let mut opens: Vec<(&'static str, Open)> = vec![
        ("read-write", |path| {
            GrafeoDB::with_config(Config::persistent(path))
        }),
        ("read-only", |path| GrafeoDB::open_read_only(path)),
    ];
    #[cfg(feature = "wal")]
    opens.push(("in-memory", |path| GrafeoDB::open_in_memory(path)));
    opens
}

/// Every file and directory under `root`, with the bytes of each file.
fn files(root: &Path) -> std::collections::BTreeMap<PathBuf, Option<Vec<u8>>> {
    let mut found = std::collections::BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let relative = path.strip_prefix(root).unwrap().to_path_buf();
            if path.is_dir() {
                found.insert(relative, None);
                pending.push(path);
            } else {
                found.insert(relative, Some(std::fs::read(&path).unwrap()));
            }
        }
    }
    found
}

/// Checks that the `kind` open of the compacted database at `path` failed
/// with an error that names the file, the `compacted` sections it holds and
/// the feature, and that every file next to it is still what `before` holds.
fn assert_refused(
    kind: &str,
    outcome: grafeo_common::utils::error::Result<GrafeoDB>,
    path: &Path,
    compacted: &[SectionType],
    before: &std::collections::BTreeMap<PathBuf, Option<Vec<u8>>>,
) {
    let error = match outcome {
        Ok(db) => panic!(
            "the {kind} open of a compacted file succeeded without `compact-store`, and serves \
             only {:?}",
            people(&db)
        ),
        Err(error) => error.to_string(),
    };
    let sections: Vec<String> = compacted
        .iter()
        .map(|section_type| format!("{section_type:?}"))
        .collect();
    assert!(
        error.contains(&path.display().to_string())
            && error.contains(&format!("(sections {})", sections.join(", ")))
            && error.contains("only a build with the `compact-store` feature can read"),
        "the {kind} error names the file, the sections {sections:?} and the feature: {error}"
    );
    assert!(
        files(path.parent().unwrap()) == *before,
        "the refused {kind} open changes nothing on disk"
    );
}

/// Writes at `path` a 0.5.x file (container v2) as 0.5.x left a compacted
/// database: the sections of the released 0.5.43 fixture `closed.grafeo`
/// (its catalog, and its LPG section as the overlay; not its triples, which
/// this build refuses too), then the `compacted` sections (`CompactStore`,
/// the base, and `OverlayDeletions`, the deletion log). These hold no real
/// encoding: the refusal looks at which sections the file holds before it
/// reads any, so their bytes do not matter. Without the refusal, a build
/// without `compact-store` reads the file, or migrates it, without them.
fn write_compacted_0_5_file(path: &Path, compacted: &[SectionType]) {
    let added: Vec<(SectionType, Vec<u8>)> = compacted
        .iter()
        .map(|section_type| {
            (
                *section_type,
                format!("{section_type:?} of Vincent").into_bytes(),
            )
        })
        .collect();
    legacy_file::write_0_5_file(
        path,
        "0.5.43",
        |section_type| section_type != SectionType::RdfStore,
        &added,
    );
}

/// A 0.5.x file compacted by 0.5.44 or older holds the base and the
/// deletion log in its section directory. A build without `compact-store`
/// neither reads it (read-only, `open_in_memory`) nor migrates it
/// (read-write): every open fails with an error that names the file, the
/// sections and the feature, and every file stays as it was, with no
/// `.pre-0.6` copy and no image left behind. Either section alone is refused
/// as well.
#[test]
fn a_build_without_compact_store_refuses_a_compacted_0_5_file() {
    for compacted in [
        &[SectionType::CompactStore, SectionType::OverlayDeletions][..],
        &[SectionType::CompactStore],
        &[SectionType::OverlayDeletions],
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        write_compacted_0_5_file(&path, compacted);
        assert_eq!(
            grafeo_storage::file::detect::detect(&path).unwrap(),
            grafeo_storage::file::detect::OnDisk::LegacyFile,
            "the file is a 0.5.x file"
        );
        let before = files(dir.path());

        for (kind, open) in opens() {
            let kind = format!("{kind} ({compacted:?})");
            assert_refused(&kind, open(&path), &path, compacted, &before);
        }
    }
}
