//! A compacted database file in a build without the `compact-store` feature
//! (the `grafeo` facade's default `embedded` profile, the CLI).
//!
//! The file (`fixtures/compacted/`, see its README) holds the compacted base
//! in its `CompactStore` section, the deletes of base nodes and edges since
//! `compact()` in its `OverlayDeletions` section, and the writes since then
//! in the overlay, its LPG section. A build without `compact-store` cannot
//! read the first two. Served alone, the overlay would miss every node and
//! edge only the base holds, and the next checkpoint would write it without
//! the base, losing them for good. So such a build refuses the file: a
//! read-write open, a read-only open and `open_in_memory` fail with an error
//! that names the file and the feature, and change nothing on disk. A 0.5.x
//! file compacted by 0.5.44 or older is refused the same way: it is neither
//! read nor migrated.
//!
//! A build with `compact-store` folds the base into the store as it opens
//! the file: the database then holds one store, its writes go to the WAL as
//! any database's do, and its next checkpoint writes the file without the
//! base and the deletion log.
//!
//! ```bash
//! cargo test -p grafeo-engine --no-default-features --features lpg,gql,grafeo-file \
//!     --test compacted_file_without_compact_store
//! cargo test -p grafeo-engine --no-default-features --features lpg,gql,wal,grafeo-file \
//!     --test compacted_file_without_compact_store
//! cargo test -p grafeo-engine --all-features --test compacted_file_without_compact_store
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "grafeo-file", not(miri)))]

#[cfg(not(feature = "compact-store"))]
#[path = "common/legacy_file.rs"]
mod legacy_file;
#[cfg(all(feature = "compact-store", feature = "wal"))]
#[path = "common/replay.rs"]
mod replay;

use std::path::{Path, PathBuf};

use grafeo_common::storage::section::SectionType;
use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

/// The fixture, written by a 0.6 development build with `compact-store`
/// (see its README).
const FIXTURE: &str = "tests/fixtures/compacted/0.6.0-dev/people.grafeo";

/// A copy of the fixture, and the path of the copy.
fn fixture() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    copy_fixture(&path);
    (dir, path)
}

/// Copies the fixture to `path`.
fn copy_fixture(path: &Path) {
    std::fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE), path).unwrap();
}

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

/// What the compacted database holds: Alix (in Paris since `compact()`),
/// Mia (created after it) and Vincent (only in the base), and no edges (Gus
/// and both his edges were deleted after `compact()`).
#[cfg(feature = "compact-store")]
fn everyone() -> Vec<(String, Option<String>)> {
    vec![
        ("Alix".to_string(), Some("Paris".to_string())),
        ("Mia".to_string(), None),
        ("Vincent".to_string(), None),
    ]
}

/// The number of `KNOWS` edges.
#[cfg(feature = "compact-store")]
fn knows(db: &GrafeoDB) -> usize {
    db.execute("MATCH ()-[k:KNOWS]->() RETURN k")
        .unwrap()
        .row_count()
}

/// The chunks of the compacted base and of the deletion log in the file at
/// `path`, as stored.
fn compacted_sections(path: &Path) -> Vec<(SectionType, Vec<bytes::Bytes>)> {
    let manager = grafeo_storage::file::GrafeoFileManager::open_read_only(path, None).unwrap();
    let sections = manager
        .read_image(|image| {
            let mut sections = Vec::new();
            for section_type in [SectionType::CompactStore, SectionType::OverlayDeletions] {
                if let Some(source) = image.section_source(section_type) {
                    let chunks = (0..source.chunks().len())
                        .map(|index| source.fetch(index))
                        .collect::<grafeo_common::utils::error::Result<Vec<_>>>()?;
                    sections.push((section_type, chunks));
                }
            }
            Ok(sections)
        })
        .unwrap();
    manager.close().unwrap();
    sections
}

/// The section types of [`compacted_sections`].
fn section_types(sections: &[(SectionType, Vec<bytes::Bytes>)]) -> Vec<SectionType> {
    sections
        .iter()
        .map(|(section_type, _)| *section_type)
        .collect()
}

/// A build with `compact-store` folds the fixture's base into the store as
/// it opens the file: Vincent, whom only the base holds, is there with Alix
/// (the overlay's copy, in Paris) and Mia, and Gus and his edges stay
/// deleted. The checkpoint of `close()` writes the database without the base
/// and the deletion log, and it reopens the same.
#[cfg(feature = "compact-store")]
#[test]
fn a_build_with_compact_store_folds_a_compacted_file() {
    let (_dir, path) = fixture();
    assert_eq!(
        section_types(&compacted_sections(&path)),
        [SectionType::CompactStore, SectionType::OverlayDeletions],
        "the file holds the base and the deletion log"
    );
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(people(&db), everyone());
    assert_eq!(knows(&db), 0, "Gus's edges are deleted");
    db.close().unwrap();

    assert_eq!(
        section_types(&compacted_sections(&path)),
        [],
        "the checkpoint writes no compacted sections"
    );
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(people(&db), everyone(), "after a reopen");
    assert_eq!(knows(&db), 0);
    db.close().unwrap();
}

/// A read-only open folds the base in memory and leaves the file as it is.
#[cfg(feature = "compact-store")]
#[test]
fn a_read_only_open_folds_a_compacted_file_in_memory() {
    let (_dir, path) = fixture();
    let before = std::fs::read(&path).unwrap();
    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_eq!(people(&db), everyone());
    db.close().unwrap();
    assert!(
        std::fs::read(&path).unwrap() == before,
        "the file is unchanged"
    );
}

/// Writes after the open of a compacted file go to its WAL: a crash before
/// the first checkpoint leaves the file with its base, and the reopen folds
/// it again and replays the writes on top, the delete of Vincent, whom only
/// the base held, included.
#[cfg(all(feature = "compact-store", feature = "wal"))]
#[test]
fn writes_after_a_fold_replay_after_a_crash() {
    let open = |path: &Path| {
        if !path.exists() {
            copy_fixture(path);
        }
        GrafeoDB::with_config(Config::persistent(path)).unwrap()
    };
    let (_dir, db) =
        replay::reopened_after_crash("writes_after_a_fold_replay_after_a_crash", open, |db| {
            db.execute("MATCH (v:Person {name: 'Vincent'}) DELETE v")
                .unwrap();
            db.execute("MATCH (a:Person {name: 'Alix'}) SET a.city = 'Prague'")
                .unwrap();
            db.execute("INSERT (:Person {name: 'Jules'})").unwrap();
        });
    assert_eq!(
        people(&db),
        [
            ("Alix".to_string(), Some("Prague".to_string())),
            ("Jules".to_string(), None),
            ("Mia".to_string(), None),
        ]
    );
    db.close().unwrap();
}

/// An open of a database file, as a build without `compact-store` has it.
#[cfg(not(feature = "compact-store"))]
type Open = fn(&Path) -> grafeo_common::utils::error::Result<GrafeoDB>;

/// Every open of an existing database file: read-write, read-only and,
/// with the `wal` feature, `open_in_memory` (which reads the file as a
/// read-only open does).
#[cfg(not(feature = "compact-store"))]
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
#[cfg(not(feature = "compact-store"))]
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
#[cfg(not(feature = "compact-store"))]
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

/// A build without `compact-store` refuses the compacted fixture, with an
/// error that names the file, its compacted sections and the feature, on a
/// read-write open, a read-only open and `open_in_memory`, and every file
/// stays as it was, byte for byte: the base and the deletion log are still
/// there for a build with the feature.
#[cfg(not(feature = "compact-store"))]
#[test]
fn a_build_without_compact_store_refuses_a_compacted_file() {
    let (dir, path) = fixture();
    let compacted = [SectionType::CompactStore, SectionType::OverlayDeletions];
    assert_eq!(
        section_types(&compacted_sections(&path)),
        compacted,
        "the fixture holds the base and the deletion log"
    );
    let before = files(dir.path());

    for (kind, open) in opens() {
        assert_refused(kind, open(&path), &path, &compacted, &before);
    }
}

/// Writes at `path` a 0.5.x file (container v2) as 0.5.x left a compacted
/// database: the sections of the released 0.5.43 fixture `closed.grafeo`
/// (its catalog, and its LPG section as the overlay; not its triples, which
/// this build refuses too), then the `compacted` sections (`CompactStore`,
/// the base, and `OverlayDeletions`, the deletion log). These hold no real
/// encoding: the refusal looks at which sections the file holds before it
/// reads any, so their bytes do not matter. Without the refusal, a build
/// without `compact-store` reads the file, or migrates it, without them.
#[cfg(not(feature = "compact-store"))]
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
#[cfg(not(feature = "compact-store"))]
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
