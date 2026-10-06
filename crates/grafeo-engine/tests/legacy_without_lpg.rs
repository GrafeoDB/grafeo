//! A 0.5.x file or WAL directory in a build without the `lpg` feature (such
//! as the `rdf` profile), which can neither read it nor migrate it to the 0.6
//! format: a read-write or read-only open fails with an error saying so and
//! naming the feature a build needs to migrate it, and leaves the database as
//! it is. Run it with:
//!
//! ```bash
//! cargo test -p grafeo-engine --no-default-features --features gql,grafeo-file,wal \
//!     --test legacy_without_lpg
//! ```

#![cfg(all(feature = "grafeo-file", feature = "wal", not(feature = "lpg")))]

use std::path::Path;

use grafeo_engine::GrafeoDB;

#[test]
fn a_build_without_lpg_says_it_cannot_migrate_a_0_5_file() {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/released/0.5.44/closed.grafeo");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    std::fs::copy(&fixture, &path).unwrap();
    let before = std::fs::read(&path).unwrap();

    for (open, outcome) in [
        ("read-only", GrafeoDB::open_read_only(&path)),
        ("read-write", GrafeoDB::open(&path)),
    ] {
        let error = match outcome {
            Ok(_) => panic!("a build without `lpg` opened a 0.5.x file ({open})"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains(&path.display().to_string())
                && error.contains("0.5.x")
                && error.contains("cannot read or migrate")
                && error.contains("`lpg` feature"),
            "{open}: the error names the file and says this build cannot read or migrate it, \
             and that a build with the `lpg` feature can: {error}"
        );
    }
    assert!(
        std::fs::read(&path).unwrap() == before,
        "the 0.5.x file is left as it is"
    );
    assert!(
        !Path::new(&format!("{}.pre-0.6", path.display())).exists(),
        "nothing is kept: nothing was migrated"
    );
}

#[test]
fn a_build_without_lpg_says_it_cannot_migrate_a_0_5_wal_directory() {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/released/0.5.44/directory");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam");
    std::fs::create_dir_all(path.join("wal")).unwrap();
    let wal = fixture.join("wal/wal_00000000.log");
    std::fs::copy(&wal, path.join("wal/wal_00000000.log")).unwrap();
    let before = std::fs::read(path.join("wal/wal_00000000.log")).unwrap();

    for (open, outcome) in [
        ("read-only", GrafeoDB::open_read_only(&path)),
        ("read-write", GrafeoDB::open(&path)),
    ] {
        let error = match outcome {
            Ok(_) => panic!("a build without `lpg` opened a 0.5.x WAL directory ({open})"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains(&path.display().to_string())
                && error.contains("0.5.x")
                && error.contains("cannot read or migrate")
                && error.contains("`lpg` feature"),
            "{open}: the error names the directory and says this build cannot read or migrate \
             it, and that a build with the `lpg` feature can: {error}"
        );
    }
    let entries: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(
        entries,
        ["amsterdam"],
        "nothing is created next to the directory"
    );
    assert!(
        std::fs::read(path.join("wal/wal_00000000.log")).unwrap() == before
            && std::fs::read_dir(&path).unwrap().count() == 1,
        "the 0.5.x directory is left as it is"
    );
}

/// `<path><suffix>`, next to the database file.
fn with_suffix(path: &Path, suffix: &str) -> std::path::PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    std::path::PathBuf::from(name)
}

/// The database id in the file header of the 0.6 file at `path`.
fn database_id(path: &Path) -> u128 {
    let bytes = std::fs::read(path).unwrap();
    grafeo_storage::file::v3::header::FileHeaderV3::decode(&bytes[..4096])
        .unwrap()
        .database_id
}

/// A build without `lpg` uses the migration lock protocol too. A migration a
/// crash cut off between its renames (`<p>.pre-0.6` and a complete image at
/// `<p>.migrating`, no `<p>`) only needs the image renamed into place, which
/// such a build does: the open finds the migrated database, never a new empty
/// one. A kept copy without `<p>` and without an image is refused, as in every
/// build, and nothing is created.
#[test]
fn a_build_without_lpg_finishes_an_interrupted_migration_and_never_creates_over_it() {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/released/0.5.44/closed.grafeo");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    let image = dir.path().join("image.grafeo");
    GrafeoDB::open(&image).unwrap().close().unwrap();
    let image_id = database_id(&image);
    std::fs::copy(&fixture, with_suffix(&path, ".pre-0.6")).unwrap();
    std::fs::rename(&image, with_suffix(&path, ".migrating")).unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    db.close().unwrap();
    drop(db);
    assert_eq!(
        database_id(&path),
        image_id,
        "the open installed the migrated image, not a new database"
    );
    assert!(
        !with_suffix(&path, ".migrating").exists(),
        "the image was moved into place"
    );
    assert!(
        std::fs::read(with_suffix(&path, ".pre-0.6")).unwrap() == std::fs::read(&fixture).unwrap(),
        "the kept copy is unchanged"
    );

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    let kept = with_suffix(&path, ".pre-0.6");
    std::fs::copy(&fixture, &kept).unwrap();
    for (open, outcome) in [
        ("read-write", GrafeoDB::open(&path)),
        ("read-only", GrafeoDB::open_read_only(&path)),
    ] {
        let error = match outcome {
            Ok(_) => panic!("{open}: a database was opened next to a kept copy without its file"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains(&kept.display().to_string()) && error.contains("missing"),
            "{open}: the error names the kept copy and says the database file is missing: {error}"
        );
        assert!(!path.exists(), "{open}: nothing was created at the path");
    }
}
