//! Restore publishes its image last and removes outputs after an I/O error.
//! Named failure injection tests cleanup and retry, not power-loss durability.

#![cfg(all(
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file",
    feature = "testing-crash-injection"
))]
#![allow(
    missing_docs,
    reason = "integration tests exercise public restore callers"
)]

#[cfg(feature = "encryption")]
use grafeo_common::encryption::KeyChain;
use grafeo_common::testing::crash::with_failure_at;
use grafeo_common::types::{EpochId, Value};
use grafeo_common::utils::error::{Error, Result};
#[cfg(feature = "encryption")]
use grafeo_engine::config::EncryptionConfig;
use grafeo_engine::{Config, GrafeoDB};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
#[cfg(feature = "encryption")]
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
enum Case {
    Full,
    Incremental,
    LegacyIncremental,
    OverlapAtFull,
    OverlapAtLast,
    #[cfg(feature = "encryption")]
    KeyedIncremental,
    #[cfg(feature = "encryption")]
    KeylessEncryptedFull,
}
impl Case {
    fn has_wal(self) -> bool {
        match self {
            Self::Full => false,
            #[cfg(feature = "encryption")]
            Self::KeylessEncryptedFull => false,
            _ => true,
        }
    }
}
type Snapshot = BTreeMap<PathBuf, Option<Vec<u8>>>;
fn snapshot(root: &Path) -> Snapshot {
    let mut result = Snapshot::new();
    if root.exists() {
        let mut pending = vec![root.to_path_buf()];
        while let Some(path) = pending.pop() {
            let relative = path.strip_prefix(root).unwrap().to_path_buf();
            if path.is_dir() {
                result.insert(relative, None);
                for entry in std::fs::read_dir(path).unwrap() {
                    pending.push(entry.unwrap().path());
                }
            } else {
                result.insert(relative, Some(std::fs::read(path).unwrap()));
            }
        }
    }
    result
}
fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}
fn person(name: &str, city: &str) -> Vec<Value> {
    vec![Value::from(name), Value::from(city)]
}

struct Fixture {
    root: tempfile::TempDir,
    backups: PathBuf,
    source: PathBuf,
    target: EpochId,
    people: Vec<Vec<Value>>,
    edges: Vec<Vec<Value>>,
    #[cfg(feature = "encryption")]
    key: Option<Arc<KeyChain>>,
    #[cfg(feature = "encryption")]
    keyed_restore: bool,
}
impl Fixture {
    fn new(case: Case) -> Self {
        let root = tempfile::tempdir().unwrap();
        let mut fixture = Self {
            backups: root.path().join("backups"),
            source: root.path().join("source.grafeo"),
            root,
            target: EpochId::new(0),
            people: Vec::new(),
            edges: Vec::new(),
            #[cfg(feature = "encryption")]
            key: matches!(case, Case::KeyedIncremental | Case::KeylessEncryptedFull)
                .then(|| Arc::new(KeyChain::new([3; 32]))),
            #[cfg(feature = "encryption")]
            keyed_restore: matches!(case, Case::KeyedIncremental),
        };
        if matches!(case, Case::LegacyIncremental) {
            fixture.copy_legacy_backup();
        } else {
            fixture.make_current_backup(case);
        }
        fixture
    }
    fn config(&self, path: &Path) -> Config {
        let config = Config::persistent(path);
        #[cfg(feature = "encryption")]
        let config = match &self.key {
            Some(key) => config.with_encryption(EncryptionConfig::new(Arc::clone(key))),
            None => config,
        };
        config
    }
    fn make_current_backup(&mut self, case: Case) {
        let db = GrafeoDB::with_config(self.config(&self.source)).expect("open fixture source");
        db.execute("INSERT (:Person {name: 'Alpha', city: 'North'})")
            .unwrap();
        let full = db.backup_full(&self.backups).expect("take full backup");
        if !case.has_wal() {
            self.target = full.end_epoch;
            self.people = vec![person("Alpha", "North")];
            db.close().unwrap();
            return;
        }
        db.execute("INSERT (:Person {name: 'Beta', city: 'South'})")
            .unwrap();
        if matches!(case, Case::OverlapAtFull | Case::OverlapAtLast) {
            // The existing cursor pins the older WAL while close checkpoints
            // the second commit. A read-only full backup cannot advance it.
            db.close().unwrap();
            let read_only = GrafeoDB::open_read_only(&self.source).unwrap();
            let newer_full = read_only.backup_full(&self.backups).unwrap();
            assert!(newer_full.end_epoch > full.end_epoch);
            read_only.close().unwrap();
            let writer = GrafeoDB::open(&self.source).unwrap();
            writer
                .execute("INSERT (:Person {name: 'Gamma', city: 'West'})")
                .unwrap();
            let incremental = writer.backup_incremental(&self.backups).unwrap();
            assert!(incremental.start_epoch <= newer_full.end_epoch);
            assert!(incremental.end_epoch > newer_full.end_epoch);
            self.people = vec![person("Alpha", "North"), person("Beta", "South")];
            self.target = match case {
                // No group after this full image's log position is at or below
                // its epoch: the returned current-format WAL has no copied group.
                Case::OverlapAtFull => newer_full.end_epoch,
                Case::OverlapAtLast => {
                    self.people.push(person("Gamma", "West"));
                    incremental.end_epoch
                }
                _ => unreachable!(),
            };
            writer.close().unwrap();
        } else {
            self.target = db.backup_incremental(&self.backups).unwrap().end_epoch;
            self.people = vec![person("Alpha", "North"), person("Beta", "South")];
            db.close().unwrap();
        }
    }

    fn copy_legacy_backup(&mut self) {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/backups/0.5.44");
        std::fs::create_dir(&self.backups).unwrap();
        for entry in std::fs::read_dir(source).unwrap() {
            let entry = entry.unwrap();
            let name = if entry.file_name() == "backup_manifest.bincode" {
                "backup_manifest.json".into()
            } else {
                entry.file_name()
            };
            std::fs::copy(entry.path(), self.backups.join(name)).unwrap();
        }
        self.target = EpochId::new(7);
        self.people = vec![
            person("Alix", "Amsterdam"),
            person("Gus", "Barcelona"),
            person("Mia", "Prague"),
            person("Vincent", "Paris"),
        ];
        self.edges = vec![
            vec![Value::from("Mia"), Value::from("Alix"), Value::Int64(1988)],
            vec![Value::from("Alix"), Value::from("Gus"), Value::Int64(2019)],
        ];
    }

    fn inputs(&self) -> [Snapshot; 3] {
        [
            snapshot(&self.backups),
            snapshot(&self.source),
            snapshot(&with_suffix(&self.source, ".wal")),
        ]
    }
    fn output(&self, name: &str) -> PathBuf {
        let directory = self.root.path().join("outputs").join(name);
        std::fs::create_dir_all(&directory).unwrap();
        directory.join("restored.grafeo")
    }
    fn restore(&self, output: &Path) -> Result<()> {
        #[cfg(feature = "encryption")]
        if self.keyed_restore {
            return GrafeoDB::restore_to_epoch_with(
                &self.backups,
                self.target,
                output,
                &EncryptionConfig::new(Arc::clone(self.key.as_ref().unwrap())),
            );
        }
        GrafeoDB::restore_to_epoch(&self.backups, self.target, output)
    }
    fn assert_two_reopens(&self, output: &Path, case: Case) {
        assert_eq!(with_suffix(output, ".wal").is_dir(), case.has_wal());
        let mut names: Vec<_> = std::fs::read_dir(output.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        names.sort();
        let mut expected = vec![output.file_name().unwrap().to_os_string()];
        if case.has_wal() {
            expected.push(
                with_suffix(output, ".wal")
                    .file_name()
                    .unwrap()
                    .to_os_string(),
            );
        }
        expected.sort();
        assert_eq!(names, expected, "successful restore left staging artifacts");
        for _ in 0..2 {
            let db = GrafeoDB::with_config(self.config(output))
                .expect("open writable restored database");
            assert_eq!(
                db.execute("MATCH (p:Person) RETURN p.name, p.city ORDER BY p.name")
                    .unwrap()
                    .rows(),
                &self.people
            );
            assert_eq!(db.execute("MATCH (a:Person)-[k:KNOWS]->(b:Person) RETURN a.name, b.name, k.since ORDER BY k.since").unwrap().rows(), &self.edges);
            db.close().unwrap();
        }
    }
}
fn assert_clean(output: &Path) {
    assert!(!output.exists(), "failed restore published its image");
    assert!(
        !with_suffix(output, ".wal").exists(),
        "failed restore published its sidecar"
    );
    assert_eq!(
        std::fs::read_dir(output.parent().unwrap()).unwrap().count(),
        0,
        "failed restore left staging artifacts"
    );
}
fn assert_failure(case: Case, phase: &str) {
    let fixture = Fixture::new(case);
    let before = fixture.inputs();
    let mut matched = 0;
    for count in 1..=64 {
        let output = fixture.output(&count.to_string());
        let result = with_failure_at(count, || fixture.restore(&output));
        assert_eq!(fixture.inputs(), before, "restore changed its inputs");
        match result {
            Err(error)
                if error
                    .to_string()
                    .contains(&format!("injected failure at: {phase}")) =>
            {
                matched += 1;
                assert_clean(&output);
                fixture
                    .restore(&output)
                    .expect("retry the same destination without injection");
                fixture.assert_two_reopens(&output, case);
                assert_eq!(fixture.inputs(), before, "retry or reopen changed inputs");
            }
            Err(error) => assert!(
                error.to_string().contains("injected failure at:"),
                "{error}"
            ),
            Ok(()) => {
                assert!(
                    matched > 0,
                    "{case:?}: restore skipped required barrier {phase}"
                );
                if phase == "restore:output_directory_sync" {
                    // Full restore publishes one name; an incremental also
                    // publishes its sidecar, whose rename needs its own sync.
                    assert_eq!(matched, if case.has_wal() { 2 } else { 1 });
                }
                fixture.assert_two_reopens(&output, case);
                assert_eq!(fixture.inputs(), before);
                return;
            }
        }
    }
    panic!("{case:?}: {phase} did not complete its bounded failure sweep");
}

#[test]
fn restores_full_current_legacy_and_overlapping_chains() {
    for case in [
        Case::Full,
        Case::Incremental,
        Case::LegacyIncremental,
        Case::OverlapAtFull,
        Case::OverlapAtLast,
        #[cfg(feature = "encryption")]
        Case::KeyedIncremental,
        #[cfg(feature = "encryption")]
        Case::KeylessEncryptedFull,
    ] {
        let fixture = Fixture::new(case);
        let before = fixture.inputs();
        let output = fixture.output("success");
        fixture.restore(&output).unwrap();
        fixture.assert_two_reopens(&output, case);
        assert_eq!(fixture.inputs(), before);
    }
}
#[test]
fn image_sync_failure_cleans_up_and_allows_same_path_retry() {
    for case in [Case::Full, Case::Incremental] {
        assert_failure(case, "restore:image_sync");
    }
}
#[test]
fn wal_sync_failure_cleans_up_current_and_legacy_outputs() {
    for case in [Case::Incremental, Case::LegacyIncremental] {
        assert_failure(case, "restore:wal_sync");
    }
}
#[cfg(unix)]
#[test]
fn wal_directory_sync_failure_cleans_up_and_allows_retry() {
    assert_failure(Case::Incremental, "restore:wal_directory_sync");
}
#[test]
fn sidecar_rename_failure_cleans_up_and_allows_retry() {
    assert_failure(Case::Incremental, "restore:sidecar_rename");
}
#[test]
fn image_rename_failure_removes_the_already_published_sidecar() {
    assert_failure(Case::Incremental, "restore:image_rename");
}
#[cfg(unix)]
#[test]
fn output_directory_sync_failure_removes_published_outputs() {
    for case in [Case::Full, Case::Incremental] {
        assert_failure(case, "restore:output_directory_sync");
    }
}
#[test]
fn a_damaged_incremental_leaves_no_output_and_can_be_retried_after_repair() {
    let fixture = Fixture::new(Case::Incremental);
    let original = fixture.inputs();
    let manifest = GrafeoDB::read_backup_manifest(&fixture.backups)
        .unwrap()
        .unwrap();
    let path = fixture
        .backups
        .join(&manifest.segments.last().unwrap().filename);
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&path, &bytes).unwrap();
    let damaged = fixture.inputs();
    let output = fixture.output("damaged");
    let error = fixture
        .restore(&output)
        .expect_err("damaged incremental must be refused");
    assert!(error.to_string().contains("CRC mismatch"), "{error}");
    assert_clean(&output);
    assert_eq!(fixture.inputs(), damaged);
    bytes[last] ^= 1;
    std::fs::write(path, bytes).unwrap();
    assert_eq!(fixture.inputs(), original);
    fixture.restore(&output).unwrap();
    fixture.assert_two_reopens(&output, Case::Incremental);
    assert_eq!(fixture.inputs(), original);
}
#[test]
fn existing_image_and_sidecar_are_never_overwritten() {
    let fixture = Fixture::new(Case::Full);
    let inputs = fixture.inputs();
    for image_exists in [true, false] {
        let output = fixture.output(if image_exists {
            "existing-image"
        } else {
            "existing-sidecar"
        });
        if image_exists {
            std::fs::write(&output, b"existing image").unwrap();
        } else {
            let wal = with_suffix(&output, ".wal");
            std::fs::create_dir(&wal).unwrap();
            std::fs::write(wal.join("sentinel"), b"existing WAL").unwrap();
        }
        let before = snapshot(output.parent().unwrap());
        let error = fixture
            .restore(&output)
            .expect_err("existing destination must be refused");
        assert!(matches!(error, Error::InvalidValue(_)), "{error}");
        assert_eq!(snapshot(output.parent().unwrap()), before);
        assert_eq!(fixture.inputs(), inputs);
    }
}
#[test]
fn a_readonly_backup_restores_to_a_writable_database_without_changing_inputs() {
    let fixture = Fixture::new(Case::Full);
    let manifest = GrafeoDB::read_backup_manifest(&fixture.backups)
        .unwrap()
        .unwrap();
    let image = fixture
        .backups
        .join(&manifest.latest_full().unwrap().filename);
    let original = std::fs::metadata(&image).unwrap().permissions();
    let output = fixture.output("readonly");
    struct RestorePermissions {
        paths: [PathBuf; 2],
        original: std::fs::Permissions,
    }
    impl Drop for RestorePermissions {
        fn drop(&mut self) {
            for path in &self.paths {
                if path.exists() {
                    let _ = std::fs::set_permissions(path, self.original.clone());
                }
            }
        }
    }
    let _cleanup = RestorePermissions {
        paths: [image.clone(), output.clone()],
        original: original.clone(),
    };
    let mut readonly = original;
    readonly.set_readonly(true);
    std::fs::set_permissions(&image, readonly.clone()).unwrap();
    let before = fixture.inputs();
    fixture.restore(&output).unwrap();
    assert_eq!(std::fs::metadata(&image).unwrap().permissions(), readonly);
    assert!(
        !std::fs::metadata(&output).unwrap().permissions().readonly(),
        "restore output must be writable"
    );
    fixture.assert_two_reopens(&output, Case::Full);
    assert_eq!(fixture.inputs(), before);
    assert_eq!(std::fs::metadata(&image).unwrap().permissions(), readonly);
}
#[cfg(feature = "encryption")]
#[test]
fn a_wrong_key_writes_nothing_and_a_correct_key_can_retry_the_same_path() {
    let fixture = Fixture::new(Case::KeyedIncremental);
    let before = fixture.inputs();
    let output = fixture.output("wrong-key");
    let wrong = EncryptionConfig::new(Arc::new(KeyChain::new([4; 32])));
    let error = GrafeoDB::restore_to_epoch_with(&fixture.backups, fixture.target, &output, &wrong)
        .expect_err("wrong key must be refused");
    assert!(matches!(error, Error::InvalidValue(_)), "{error}");
    assert!(error.to_string().contains("key"));
    assert_clean(&output);
    assert_eq!(fixture.inputs(), before);
    fixture.restore(&output).unwrap();
    fixture.assert_two_reopens(&output, Case::KeyedIncremental);
    assert_eq!(fixture.inputs(), before);
}
