//! #744: a restore checks the full backup's checksum as it checks an incremental one's. A full segment
//! with a byte flipped anywhere, or cut short, is refused as damaged before anything is written, with
//! the same kind of error a damaged incremental gives, and with an incremental after it too; an intact
//! chain restores as before. The CLI's `backup restore-to-epoch` path is in `grafeo-cli`'s `cli_runs`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main backup_full_segment_checked::
//! ```

#![cfg(all(feature = "wal", feature = "grafeo-file", feature = "gql"))]

use std::path::{Path, PathBuf};

use grafeo_common::types::{EpochId, Value};
use grafeo_common::utils::error::Error;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::backup::BackupKind;

struct Chain {
    _dir: tempfile::TempDir,
    root: PathBuf,
    backups: PathBuf,
    full: PathBuf,
    incremental: Option<PathBuf>,
    full_epoch: EpochId,
    last_epoch: EpochId,
}

/// Alix in a full backup and, with `incremental`, Gus in an incremental one after it.
fn chain(incremental: bool) -> Chain {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let backups = root.join("backups");
    let db = GrafeoDB::open(root.join("live.grafeo")).unwrap();
    db.session()
        .execute("INSERT (:Person {name: 'Alix'})")
        .unwrap();
    let full = db.backup_full(&backups).unwrap();
    assert_eq!(full.kind, BackupKind::Full);
    let full_epoch = full.end_epoch;
    let mut last_epoch = full_epoch;
    let mut incremental_path = None;
    if incremental {
        db.session()
            .execute("INSERT (:Person {name: 'Gus'})")
            .unwrap();
        let segment = db.backup_incremental(&backups).unwrap();
        assert_eq!(segment.kind, BackupKind::Incremental);
        last_epoch = segment.end_epoch;
        incremental_path = Some(backups.join(&segment.filename));
    }
    db.close().unwrap();
    Chain {
        full: backups.join(&full.filename),
        _dir: dir,
        root,
        backups,
        incremental: incremental_path,
        full_epoch,
        last_epoch,
    }
}

fn flip(path: &Path, at: impl Fn(usize) -> usize) {
    let mut bytes = std::fs::read(path).unwrap();
    let index = at(bytes.len());
    bytes[index] ^= 0x5A;
    std::fs::write(path, bytes).unwrap();
}

fn names(path: &Path) -> Vec<Value> {
    let db = GrafeoDB::open_read_only(path).unwrap();
    let names = db
        .execute("MATCH (p:Person) RETURN p.name AS name ORDER BY name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect();
    db.close().unwrap();
    names
}

/// The restore is refused as damage, names the segment, and leaves nothing at the output nor beside
/// it.
fn refused(chain: &Chain, epoch: EpochId, segment: &Path, case: &str) -> String {
    let output = chain
        .root
        .join(format!("restored-{}.grafeo", case.replace(' ', "-")));
    let before: Vec<_> = std::fs::read_dir(&chain.root)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    let error = GrafeoDB::restore_to_epoch(&chain.backups, epoch, &output).expect_err(case);
    assert!(matches!(error, Error::Corruption(_)), "{case}: {error:?}");
    let text = error.to_string();
    let name = segment.file_name().unwrap().to_str().unwrap();
    assert!(
        text.contains(name),
        "{case}: the error names {name}: {text}"
    );
    assert!(!output.exists(), "{case}: nothing published");
    let after: Vec<_> = std::fs::read_dir(&chain.root)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(before, after, "{case}: nothing left beside the output");
    text
}

#[test]
fn an_intact_chain_restores() {
    for incremental in [false, true] {
        let chain = chain(incremental);
        let output = chain.root.join("restored.grafeo");
        GrafeoDB::restore_to_epoch(&chain.backups, chain.last_epoch, &output).unwrap();
        let wanted: Vec<Value> = if incremental {
            vec![Value::from("Alix"), Value::from("Gus")]
        } else {
            vec![Value::from("Alix")]
        };
        assert_eq!(names(&output), wanted, "incremental {incremental}");
    }
}

#[test]
fn a_flipped_byte_in_a_full_backup_is_refused() {
    let cases: [(&str, fn(usize) -> usize); 4] = [
        ("first byte", |_| 0),
        ("middle", |len| len / 2),
        ("past the header", |len| len.min(4096) - 1),
        ("last byte", |len| len - 1),
    ];
    for incremental in [false, true] {
        for (place, at) in cases {
            let chain = chain(incremental);
            flip(&chain.full, at);
            let case = format!("{place}, incremental {incremental}");
            refused(&chain, chain.full_epoch, &chain.full, &case);
            refused(
                &chain,
                chain.last_epoch,
                &chain.full,
                &format!("{case}, to the last epoch"),
            );
        }
    }
}

#[test]
fn a_full_backup_cut_short_is_refused() {
    let chain = chain(false);
    let bytes = std::fs::read(&chain.full).unwrap();
    std::fs::write(&chain.full, &bytes[..bytes.len() / 2]).unwrap();
    refused(&chain, chain.full_epoch, &chain.full, "cut short");
}

/// A damaged full and a damaged incremental are the same kind of error, both named.
#[test]
fn a_damaged_full_and_incremental_give_the_same_error() {
    let damaged_incremental = chain(true);
    let incremental = damaged_incremental.incremental.clone().unwrap();
    flip(&incremental, |len| len / 2);
    let incremental_text = refused(
        &damaged_incremental,
        damaged_incremental.last_epoch,
        &incremental,
        "a damaged incremental",
    );
    let damaged_full = chain(true);
    flip(&damaged_full.full, |len| len / 2);
    let full_text = refused(
        &damaged_full,
        damaged_full.last_epoch,
        &damaged_full.full,
        "a damaged full",
    );
    for text in [&incremental_text, &full_text] {
        assert!(text.contains("CRC mismatch"), "{text}");
    }
}
