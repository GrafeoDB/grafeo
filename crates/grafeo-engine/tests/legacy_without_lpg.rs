//! A 0.5.x file in a build without the `lpg` feature (such as the `rdf`
//! profile), which can neither read it nor migrate it to the 0.6 format: a
//! read-write or read-only open fails with an error saying so and
//! naming the feature a build needs to migrate it, and leaves the file as it
//! is. Run it with:
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
