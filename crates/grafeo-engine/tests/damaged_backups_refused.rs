//! Class test 1, backups (class-tests design, 2026-11-05): a damaged backup is refused with a
//! named error, never a panic and never a silent default.
//!
//! A chain of every kind of backup the catalogue names (`tests/catalogue/backup_kind.txt`): a
//! full backup and two incremental ones, with writes between them. Every file of the backup
//! directory (each segment and the manifest) is damaged in turn: a byte flipped at its first,
//! middle and last byte, cut to half, zeroed. Then the chain is restored to the end of each
//! segment. A restore must give exactly what the undamaged chain restores at that epoch, or fail
//! with an error naming the damaged file (`Error::Corruption` for a segment, whose checksum the
//! manifest holds) and leave no database at the target path.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main damaged_backups_refused::
//! ```

#![cfg(all(feature = "gql", feature = "wal", feature = "grafeo-file"))]

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use grafeo_common::types::EpochId;
use grafeo_common::utils::error::{Error, Result};
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::backup::BackupSegment;

const BACKUP_KINDS: &str = include_str!("../../../tests/catalogue/backup_kind.txt");

fn names(file: &'static str) -> Vec<&'static str> {
    file.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect()
}

/// A full backup and two incremental ones of a database written between them, in `dir`.
fn chain(dir: &Path) -> (PathBuf, Vec<BackupSegment>) {
    let backups = dir.join("backups");
    let db = GrafeoDB::open(dir.join("source.grafeo")).unwrap();
    let mut segments = Vec::new();
    db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();
    segments.push(db.backup_full(&backups).unwrap());
    db.execute("INSERT (:Person {name: 'Vincent', age: 40})")
        .unwrap();
    db.execute("MATCH (n:Person {name: 'Gus'}) SET n.age = 33")
        .unwrap();
    segments.push(db.backup_incremental(&backups).unwrap());
    db.execute("MATCH (n:Person {name: 'Alix'}) DETACH DELETE n")
        .unwrap();
    db.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    segments.push(db.backup_incremental(&backups).unwrap());
    db.close().unwrap();
    (backups, segments)
}

/// Every file of `dir`, sorted, with its bytes.
fn files_in(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut files: Vec<(PathBuf, Vec<u8>)> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| {
            let path = entry.unwrap().path();
            let bytes = std::fs::read(&path).unwrap();
            (path.file_name().unwrap().into(), bytes)
        })
        .collect();
    files.sort();
    files
}

fn restore_files(dir: &Path, files: &[(PathBuf, Vec<u8>)]) {
    if dir.exists() {
        std::fs::remove_dir_all(dir).unwrap();
    }
    std::fs::create_dir_all(dir).unwrap();
    for (name, bytes) in files {
        std::fs::write(dir.join(name), bytes).unwrap();
    }
}

/// What a restored database holds, written out.
fn read_whole(path: &Path) -> Result<Vec<String>> {
    let db = GrafeoDB::open_read_only(path)?;
    let mut read = Vec::new();
    for query in [
        "MATCH (n) RETURN labels(n), properties(n)",
        "MATCH (a)-[e]->(b) RETURN a.name, type(e), b.name",
    ] {
        let mut rows: Vec<String> = db
            .execute(query)?
            .rows()
            .iter()
            .map(|row| format!("{row:?}"))
            .collect();
        rows.sort();
        read.push(rows.join("; "));
    }
    Ok(read)
}

#[derive(Clone, Copy, Debug)]
enum Damage {
    FlipFirst,
    FlipMiddle,
    FlipLast,
    Half,
    Zeros,
}

impl Damage {
    fn apply(self, bytes: &mut Vec<u8>) {
        let last = bytes.len().saturating_sub(1);
        let middle = bytes.len() / 2;
        match self {
            Damage::FlipFirst => bytes[0] ^= 0x5A,
            Damage::FlipMiddle => bytes[middle] ^= 0x5A,
            Damage::FlipLast => bytes[last] ^= 0x5A,
            Damage::Half => bytes.truncate(middle),
            Damage::Zeros => bytes.iter_mut().for_each(|byte| *byte = 0),
        }
    }
}

/// How a restore went.
enum Restored {
    Read(Vec<String>),
    Failed(Error),
    Panicked(String),
}

fn restore(backups: &Path, epoch: EpochId, target: &Path) -> Restored {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        GrafeoDB::restore_to_epoch(backups, epoch, target)?;
        read_whole(target)
    }));
    match outcome {
        Ok(Ok(read)) => Restored::Read(read),
        Ok(Err(error)) => Restored::Failed(error),
        Err(panic) => Restored::Panicked(
            panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_default(),
        ),
    }
}

#[test]
fn the_chain_holds_every_kind_of_backup() {
    let dir = tempfile::tempdir().unwrap();
    let (_, segments) = chain(dir.path());
    let held: Vec<String> = segments
        .iter()
        .map(|segment| format!("{:?}", segment.kind))
        .collect();
    for kind in names(BACKUP_KINDS) {
        assert!(
            held.iter().any(|h| h == kind),
            "the chain holds no {kind} backup: {held:?}"
        );
    }
}

#[test]
fn a_damaged_backup_is_refused_or_restores_the_same() {
    let dir = tempfile::tempdir().unwrap();
    let (backups, segments) = chain(dir.path());
    let pristine = files_in(&backups);
    let epochs: Vec<EpochId> = segments.iter().map(|segment| segment.end_epoch).collect();
    let mut want = Vec::new();
    for (n, &epoch) in epochs.iter().enumerate() {
        let target = dir.path().join(format!("pristine-{n}.grafeo"));
        match restore(&backups, epoch, &target) {
            Restored::Read(read) => want.push(read),
            Restored::Failed(error) => panic!("the undamaged chain to {epoch:?}: {error}"),
            Restored::Panicked(message) => panic!("the undamaged chain to {epoch:?}: {message}"),
        }
    }
    let work = dir.path().join("work");
    let mut wrong = Vec::new();
    let mut checked = 0;
    let mut attempt = 0;
    for (name, bytes) in &pristine {
        // `backup.lock` is empty: a lock, which holds nothing to damage.
        if bytes.is_empty() {
            continue;
        }
        for damage in [
            Damage::FlipFirst,
            Damage::FlipMiddle,
            Damage::FlipLast,
            Damage::Half,
            Damage::Zeros,
        ] {
            for (n, &epoch) in epochs.iter().enumerate() {
                restore_files(&work, &pristine);
                let file = work.join(name);
                let mut bytes = std::fs::read(&file).unwrap();
                damage.apply(&mut bytes);
                std::fs::write(&file, bytes).unwrap();
                attempt += 1;
                let target = dir.path().join(format!("restored-{attempt}.grafeo"));
                checked += 1;
                let what = format!("{} {damage:?}, restored to {epoch:?}", name.display());
                match restore(&work, epoch, &target) {
                    Restored::Read(read) if read == want[n] => {}
                    Restored::Read(read) => wrong.push(format!(
                        "{what}: restored something else: {read:?}, want {:?}",
                        want[n]
                    )),
                    Restored::Failed(error) => {
                        let names_it = match &error {
                            // The segment in the corruption's file, or in its description.
                            Error::Corruption(corruption) => {
                                corruption.file.as_deref() == Some(file.as_path())
                                    || corruption.what.contains(&*name.to_string_lossy())
                            }
                            other => other.to_string().contains(&*name.to_string_lossy()),
                        };
                        if !names_it {
                            wrong.push(format!(
                                "{what}: an error that names no damaged file: {error:?}"
                            ));
                        }
                        if target.exists() {
                            wrong.push(format!(
                                "{what}: a failed restore left {}",
                                target.display()
                            ));
                        }
                    }
                    Restored::Panicked(message) => {
                        wrong.push(format!("{what}: panicked: {message}"));
                    }
                }
            }
        }
    }
    assert!(checked > 50, "{checked} checked");
    assert!(
        wrong.is_empty(),
        "{} of {checked} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
