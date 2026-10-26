//! Restore acknowledges success only after its returned files and namespace sync.
//!
//! Named I/O failures check error propagation, input preservation and retry.
//! They do not simulate losing the operating system's cache or prove power-loss
//! safety. Temporary restore staging is not a returned durable artifact.

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

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
#[cfg(feature = "encryption")]
use std::sync::Arc;

#[cfg(feature = "encryption")]
use grafeo_common::encryption::KeyChain;
use grafeo_common::testing::crash::with_failure_at;
use grafeo_common::types::{EpochId, Value};
use grafeo_common::utils::error::{Error, Result};
#[cfg(feature = "encryption")]
use grafeo_engine::config::EncryptionConfig;
use grafeo_engine::{Config, GrafeoDB};

#[derive(Clone, Copy, Debug)]
enum Case {
    Full,
    Incremental,
    ReadOnlyFull,
    ReadOnlyIncremental,
    LegacyIncremental,
    ReadOnlyOverlapAtFull,
    ReadOnlyOverlapAtLast,
    #[cfg(feature = "encryption")]
    KeyedIncremental,
    #[cfg(feature = "encryption")]
    KeylessEncryptedFull,
    #[cfg(feature = "encryption")]
    ReadOnlyKeyedIncremental,
    #[cfg(feature = "encryption")]
    ReadOnlyKeylessEncryptedFull,
}

impl Case {
    fn has_wal(self) -> bool {
        if matches!(self, Self::Full | Self::ReadOnlyFull) {
            return false;
        }
        #[cfg(feature = "encryption")]
        if matches!(
            self,
            Self::KeylessEncryptedFull | Self::ReadOnlyKeylessEncryptedFull
        ) {
            return false;
        }
        true
    }

    fn read_only_image(self) -> bool {
        if matches!(self, Self::ReadOnlyFull | Self::ReadOnlyIncremental) {
            return true;
        }
        #[cfg(feature = "encryption")]
        if matches!(
            self,
            Self::ReadOnlyKeyedIncremental | Self::ReadOnlyKeylessEncryptedFull
        ) {
            return true;
        }
        false
    }
}

fn cases(need_wal: bool) -> Vec<Case> {
    let cases = vec![
        Case::Full,
        Case::Incremental,
        Case::ReadOnlyFull,
        Case::ReadOnlyIncremental,
        Case::LegacyIncremental,
        Case::ReadOnlyOverlapAtFull,
        Case::ReadOnlyOverlapAtLast,
        #[cfg(feature = "encryption")]
        Case::KeyedIncremental,
        #[cfg(feature = "encryption")]
        Case::KeylessEncryptedFull,
        #[cfg(feature = "encryption")]
        Case::ReadOnlyKeyedIncremental,
        #[cfg(feature = "encryption")]
        Case::ReadOnlyKeylessEncryptedFull,
    ];
    cases
        .into_iter()
        .filter(|case| !need_wal || case.has_wal())
        .collect()
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// Relative names and bytes, including empty directories, so a refused retry
/// cannot silently replace a partial file or sidecar.
type Snapshot = BTreeMap<PathBuf, Option<Vec<u8>>>;

fn snapshot(root: &Path) -> Snapshot {
    let mut result = BTreeMap::new();
    if !root.exists() {
        return result;
    }
    let mut pending = vec![root.to_path_buf()];
    while let Some(path) = pending.pop() {
        let relative = path.strip_prefix(root).unwrap().to_path_buf();
        if path.is_dir() {
            result.insert(relative, None);
            for entry in std::fs::read_dir(&path).expect("read snapshot directory") {
                pending.push(entry.expect("read snapshot entry").path());
            }
        } else {
            result.insert(
                relative,
                Some(std::fs::read(path).expect("read snapshot file")),
            );
        }
    }
    result
}

fn person(name: &str, city: &str) -> Vec<Value> {
    vec![Value::from(name), Value::from(city)]
}

/// Restores temporary permissions before TempDir removes files on Windows.
#[derive(Default)]
struct PermissionCleanup(RefCell<Vec<(PathBuf, std::fs::Permissions)>>);

impl PermissionCleanup {
    fn remember(&self, path: &Path, permissions: &std::fs::Permissions) {
        self.0
            .borrow_mut()
            .push((path.to_path_buf(), permissions.clone()));
    }
}

impl Drop for PermissionCleanup {
    fn drop(&mut self) {
        for (path, permissions) in self.0.get_mut().drain(..).rev() {
            if path.exists() {
                let restored = std::fs::set_permissions(path, permissions);
                if !std::thread::panicking() {
                    restored.expect("restore temporary file permissions before cleanup");
                }
            }
        }
    }
}

fn assert_permissions(path: &Path, expected: &std::fs::Permissions) {
    let actual = std::fs::metadata(path)
        .expect("read permissions")
        .permissions();
    assert_eq!(actual.readonly(), expected.readonly(), "{}", path.display());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(actual.mode(), expected.mode(), "{}", path.display());
    }
}

#[cfg(windows)]
const RESTORE_METADATA: &[u8] = b"restore metadata sentinel";

#[cfg(windows)]
fn add_restore_metadata_stream(path: &Path) -> bool {
    match std::fs::write(with_suffix(path, ":restore_metadata"), RESTORE_METADATA) {
        Ok(()) => true,
        // Windows filesystems without named-stream support can reject the
        // syntax (INVALID_FUNCTION, NOT_SUPPORTED or INVALID_NAME).
        Err(error) if matches!(error.raw_os_error(), Some(1 | 50 | 123)) => false,
        Err(error) => panic!("cannot write fixture metadata stream: {error}"),
    }
}

struct ReadOnlyImage {
    path: PathBuf,
    original_permissions: std::fs::Permissions,
    read_only_permissions: std::fs::Permissions,
    source_permissions: std::fs::Permissions,
    #[cfg(windows)]
    has_metadata_stream: bool,
}

struct Fixture {
    // Fields drop in order: permissions are restored before TempDir cleanup.
    permission_cleanup: PermissionCleanup,
    root: tempfile::TempDir,
    backups: PathBuf,
    source: PathBuf,
    target: EpochId,
    people: Vec<Vec<Value>>,
    edges: Vec<Vec<Value>>,
    read_only_image: Option<ReadOnlyImage>,
    #[cfg(feature = "encryption")]
    key: Option<Arc<KeyChain>>,
    #[cfg(feature = "encryption")]
    keyed_restore: bool,
}

impl Fixture {
    fn new(case: Case) -> Self {
        let root = tempfile::tempdir().expect("create fixture directory");
        let mut fixture = Self {
            permission_cleanup: PermissionCleanup::default(),
            backups: root.path().join("backups"),
            source: root.path().join("source.grafeo"),
            root,
            target: EpochId::new(0),
            people: Vec::new(),
            edges: Vec::new(),
            read_only_image: None,
            #[cfg(feature = "encryption")]
            key: match case {
                Case::KeyedIncremental
                | Case::KeylessEncryptedFull
                | Case::ReadOnlyKeyedIncremental
                | Case::ReadOnlyKeylessEncryptedFull => Some(Arc::new(KeyChain::new([3; 32]))),
                _ => None,
            },
            #[cfg(feature = "encryption")]
            keyed_restore: matches!(
                case,
                Case::KeyedIncremental | Case::ReadOnlyKeyedIncremental
            ),
        };
        if matches!(case, Case::LegacyIncremental) {
            fixture.copy_legacy_backup();
        } else {
            fixture.make_current_backup(case);
        }
        if case.read_only_image() {
            fixture.make_image_read_only();
        }
        fixture
    }

    fn config(&self, path: &Path, read_only: bool) -> Config {
        let config = if read_only {
            Config::read_only(path)
        } else {
            Config::persistent(path)
        };
        #[cfg(feature = "encryption")]
        let config = match &self.key {
            Some(key) => config.with_encryption(EncryptionConfig::new(Arc::clone(key))),
            None => config,
        };
        config
    }

    fn make_current_backup(&mut self, case: Case) {
        let db =
            GrafeoDB::with_config(self.config(&self.source, false)).expect("open fixture source");
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
        if matches!(
            case,
            Case::ReadOnlyOverlapAtFull | Case::ReadOnlyOverlapAtLast
        ) {
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
                Case::ReadOnlyOverlapAtFull => newer_full.end_epoch,
                Case::ReadOnlyOverlapAtLast => {
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

    fn make_image_read_only(&mut self) {
        let manifest = GrafeoDB::read_backup_manifest(&self.backups)
            .unwrap()
            .unwrap();
        let path = self.backups.join(&manifest.latest_full().unwrap().filename);
        let original = std::fs::metadata(&path).unwrap().permissions();
        #[cfg(windows)]
        let has_metadata_stream = add_restore_metadata_stream(&path);
        #[cfg(unix)]
        let read_only = {
            use std::os::unix::fs::PermissionsExt;
            std::fs::Permissions::from_mode(0o444)
        };
        #[cfg(not(unix))]
        let read_only = {
            let mut permissions = original.clone();
            permissions.set_readonly(true);
            permissions
        };
        self.permission_cleanup.remember(&path, &original);
        std::fs::set_permissions(&path, read_only).expect("make only the fixture image read-only");
        let read_only_permissions = std::fs::metadata(&path).unwrap().permissions();
        assert!(read_only_permissions.readonly());
        self.read_only_image = Some(ReadOnlyImage {
            path,
            original_permissions: original,
            read_only_permissions,
            source_permissions: std::fs::metadata(&self.source).unwrap().permissions(),
            #[cfg(windows)]
            has_metadata_stream,
        });
    }

    fn assert_read_only_copy(&self, output: &Path) {
        if let Some(image) = &self.read_only_image {
            assert_permissions(&image.path, &image.read_only_permissions);
            assert_permissions(&self.source, &image.source_permissions);
            #[cfg(windows)]
            if image.has_metadata_stream {
                assert_eq!(
                    std::fs::read(with_suffix(&image.path, ":restore_metadata")).unwrap(),
                    RESTORE_METADATA,
                    "restore must preserve the source metadata stream"
                );
            }
            if output.exists() {
                assert_eq!(
                    std::fs::read(output).unwrap(),
                    std::fs::read(&image.path).unwrap(),
                    "a restored read-only image must remain an exact copy"
                );
                assert_permissions(output, &image.read_only_permissions);
                #[cfg(windows)]
                if image.has_metadata_stream {
                    assert_eq!(
                        std::fs::read(with_suffix(output, ":restore_metadata")).unwrap(),
                        RESTORE_METADATA,
                        "restore must preserve the copied metadata stream, including on failure"
                    );
                }
            }
        }
    }

    fn inputs(&self) -> [Snapshot; 3] {
        [
            snapshot(&self.backups),
            snapshot(&self.source),
            snapshot(&with_suffix(&self.source, ".wal")),
        ]
    }

    fn restore(&self, output: &Path) -> Result<()> {
        if let Some(image) = &self.read_only_image {
            // Restore may leave a read-only partial image. Do not change its
            // permissions for the oracle; only restore them during cleanup.
            self.permission_cleanup
                .remember(output, &image.original_permissions);
        }
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
        self.assert_read_only_copy(output);
        for _ in 0..2 {
            let db = GrafeoDB::with_config(self.config(output, self.read_only_image.is_some()))
                .expect("open restored database");
            let people = db
                .execute("MATCH (p:Person) RETURN p.name, p.city ORDER BY p.name")
                .unwrap()
                .rows()
                .to_vec();
            let edges = db.execute(
                "MATCH (a:Person)-[k:KNOWS]->(b:Person) RETURN a.name, b.name, k.since ORDER BY k.since",
            ).unwrap().rows().to_vec();
            assert_eq!(people, self.people, "{case:?}: exact restored people");
            assert_eq!(edges, self.edges, "{case:?}: exact restored edges");
            db.close().expect("close restored database");
            self.assert_read_only_copy(output);
        }
    }
}

fn assert_barrier(case: Case, phase: &str) {
    let fixture = Fixture::new(case);
    let before = fixture.inputs();
    let injected_phase = format!("injected failure at: {phase}");
    let mut matched = 0;
    for count in 1..=64 {
        let output_dir = fixture
            .root
            .path()
            .join("outputs")
            .join(count.to_string())
            .join("nested");
        std::fs::create_dir_all(&output_dir).unwrap();
        let output = output_dir.join("restored.grafeo");
        let result = with_failure_at(count, || fixture.restore(&output));
        fixture.assert_read_only_copy(&output);
        assert_eq!(
            fixture.inputs(),
            before,
            "{case:?}: restore changed its inputs"
        );
        match result {
            Err(error) if error.to_string().contains(&injected_phase) => {
                matched += 1;
                assert!(
                    output.is_file(),
                    "{case:?}: failed restore left its partial image"
                );
                let partial = snapshot(&output_dir);
                let refused = fixture
                    .restore(&output)
                    .expect_err("partial target is not fresh");
                assert!(matches!(refused, Error::InvalidValue(_)), "{refused}");
                assert!(refused.to_string().contains("already exists"), "{refused}");
                assert_eq!(
                    snapshot(&output_dir),
                    partial,
                    "refused retry changed partial output"
                );
                assert_eq!(fixture.inputs(), before, "refused retry changed inputs");
            }
            Err(error) => assert!(
                error.to_string().contains("injected failure at:"),
                "{case:?}: unexpected failure selecting {phase} at {count}: {error}",
            ),
            Ok(()) => {
                assert!(
                    matched > 0,
                    "{case:?}: restore acknowledged success without the required {phase} barrier",
                );
                // Advancing the count reaches every occurrence of the named
                // barrier (each returned WAL file or directory ancestor).
                // The first success is a fresh destination after those failures.
                fixture.assert_two_reopens(&output, case);
                assert_eq!(
                    fixture.inputs(),
                    before,
                    "opening restored output changed inputs"
                );
                return;
            }
        }
    }
    panic!("{case:?}: {phase} never completed its bounded fault sweep");
}

#[test]
fn returned_image_sync_failures_are_reported_before_restore_success() {
    for case in cases(false) {
        assert_barrier(case, "restore:image_sync");
    }
}

#[test]
fn returned_wal_file_sync_failures_are_reported_before_restore_success() {
    for case in cases(true) {
        assert_barrier(case, "restore:wal_sync");
    }
}

#[cfg(unix)]
#[test]
fn returned_wal_directory_sync_failures_are_reported_before_restore_success() {
    for case in cases(true) {
        assert_barrier(case, "restore:wal_directory_sync");
    }
}

#[cfg(unix)]
#[test]
fn returned_output_directory_sync_failures_are_reported_before_restore_success() {
    for case in cases(false) {
        assert_barrier(case, "restore:output_directory_sync");
    }
}

#[test]
fn read_only_backup_images_restore_with_original_permissions() {
    for case in cases(false)
        .into_iter()
        .filter(|case| case.read_only_image())
    {
        let fixture = Fixture::new(case);
        let before = fixture.inputs();
        let output_dir = fixture.root.path().join("outputs").join("positive");
        std::fs::create_dir_all(&output_dir).unwrap();
        let output = output_dir.join("restored.grafeo");
        fixture
            .restore(&output)
            .expect("restore from an actually read-only backup image");
        fixture.assert_read_only_copy(&output);
        assert_eq!(
            fixture.inputs(),
            before,
            "restore changed its read-only source"
        );
        fixture.assert_two_reopens(&output, case);
        assert_eq!(
            fixture.inputs(),
            before,
            "read-only reopens changed source bytes"
        );
    }
}

#[cfg(windows)]
#[test]
fn readonly_open_failure_restores_image_permissions() {
    for case in cases(false)
        .into_iter()
        .filter(|case| case.read_only_image())
    {
        assert_barrier(case, "restore:readonly_file_open");
    }
}

#[cfg(windows)]
#[test]
fn readonly_permission_restore_failure_cleans_up() {
    for case in cases(false)
        .into_iter()
        .filter(|case| case.read_only_image())
    {
        assert_barrier(case, "restore:readonly_permissions_restore");
    }
}
