//! Backup management commands.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use grafeo_engine::GrafeoDB;

use crate::output;
use crate::{BackupCommands, OutputFormat};

/// Run backup commands.
pub fn run(cmd: BackupCommands, _format: OutputFormat, quiet: bool) -> Result<()> {
    match cmd {
        BackupCommands::Create { path, output: out } => {
            output::status(&format!("Creating backup of {}...", path.display()), quiet);

            let db = super::open_existing(&path)?;
            db.save(&out)
                .with_context(|| format!("Failed to create backup at {}", out.display()))?;

            output::success(&format!("Backup created at {}", out.display()), quiet);
        }
        BackupCommands::Restore {
            backup,
            path,
            force,
        } => {
            restore(&backup, &path, force, quiet)?;
            output::success(&format!("Database restored to {}", path.display()), quiet);
        }
        BackupCommands::Full { path, output: out } => {
            output::status(
                &format!("Creating full backup of {}...", path.display()),
                quiet,
            );

            let db = super::open_existing(&path)?;
            let segment = db
                .backup_full(&out)
                .with_context(|| format!("Failed to create full backup at {}", out.display()))?;

            output::success(
                &format!(
                    "Full backup created: {} ({} bytes, epoch 0-{})",
                    segment.filename,
                    segment.size_bytes,
                    segment.end_epoch.as_u64()
                ),
                quiet,
            );
        }
        BackupCommands::Incremental { path, output: out } => {
            output::status(
                &format!("Creating incremental backup of {}...", path.display()),
                quiet,
            );

            let db = super::open_existing(&path)?;
            let segment = db.backup_incremental(&out).with_context(|| {
                format!("Failed to create incremental backup at {}", out.display())
            })?;

            output::success(
                &format!(
                    "Incremental backup created: {} ({} bytes, epoch {}-{})",
                    segment.filename,
                    segment.size_bytes,
                    segment.start_epoch.as_u64(),
                    segment.end_epoch.as_u64()
                ),
                quiet,
            );
        }
        BackupCommands::Status { path } => {
            let manifest = GrafeoDB::read_backup_manifest(&path)
                .with_context(|| format!("Failed to read manifest at {}", path.display()))?;

            match manifest {
                Some(m) => {
                    println!("Backup manifest (version {})", m.version);
                    println!("Segments: {}", m.segments.len());
                    if let Some((start, end)) = m.epoch_range() {
                        println!("Epoch range: {} - {}", start.as_u64(), end.as_u64());
                    }
                    println!();
                    for (i, seg) in m.segments.iter().enumerate() {
                        println!(
                            "  [{i}] {:?} {} (epoch {}-{}, {} bytes)",
                            seg.kind,
                            seg.filename,
                            seg.start_epoch.as_u64(),
                            seg.end_epoch.as_u64(),
                            seg.size_bytes
                        );
                    }
                }
                None => {
                    println!("No backup manifest found at {}", path.display());
                }
            }
        }
        BackupCommands::RestoreToEpoch {
            backup_dir,
            epoch,
            output: out,
        } => {
            output::status(
                &format!(
                    "Restoring to epoch {epoch} from {}...",
                    backup_dir.display()
                ),
                quiet,
            );

            let target = grafeo_common::types::EpochId::new(epoch);
            GrafeoDB::restore_to_epoch(&backup_dir, target, &out)
                .with_context(|| format!("Failed to restore to epoch {epoch}"))?;

            output::success(
                &format!("Database restored to epoch {epoch} at {}", out.display()),
                quiet,
            );
        }
    }

    Ok(())
}

/// Restores the backup at `backup` to a database at `path`, replacing the
/// database there when `force` is set.
///
/// Nothing at `path` changes until the restored database is complete: the
/// backup is opened read-only (so it is never changed, and a 0.5.x backup is
/// read in place instead of migrated, its WAL replayed as a read-write open
/// would) and written to `<path>.restoring`; only then is the old database
/// removed, with its sidecar WAL, and the image moved into its place without
/// replacing anything: a database another process creates there meanwhile
/// stays as it is, and the restore fails. A missing or unreadable backup, a
/// backup inside the target, or a target that is not a database, is in use,
/// or is (or holds) the current directory, leaves everything as it was. The
/// target stays locked from its check until right before its removal.
fn restore(backup: &Path, path: &Path, force: bool, quiet: bool) -> Result<()> {
    restore_with(backup, path, force, quiet, || {})
}

/// [`restore`], calling `before_install` right before the restored image is
/// moved to the target path: the moment another process could create a
/// database there.
fn restore_with(
    backup: &Path,
    path: &Path,
    force: bool,
    quiet: bool,
    before_install: impl FnOnce(),
) -> Result<()> {
    // The spelling the engine uses: side-file names are appended to the path,
    // so `db/` is `db` and `.` is the directory it names.
    let path = normalize_database_path(path)?;
    let backup = normalize_database_path(backup)?;
    let backup = backup.as_path();
    refuse_current_directory(&path)?;
    let sidecar = with_suffix(&path, ".wal");
    let taken = path.exists() || sidecar.exists();
    if taken && !force {
        anyhow::bail!(
            "Target path {} already exists. Use --force to overwrite.",
            path.display()
        );
    }

    output::status(&format!("Restoring from {}...", backup.display()), quiet);
    if !backup.exists() {
        anyhow::bail!("Backup not found: {}", backup.display());
    }
    refuse_backup_inside(backup, &path)?;
    let db = GrafeoDB::open_read_only(backup)
        .with_context(|| format!("Failed to open backup at {}", backup.display()))?;
    // Held until right before the removal: a process that opens the target
    // meanwhile fails, instead of writing to a database that is about to go.
    let target_lock = if path.exists() {
        Some(check_replaceable(&path)?)
    } else {
        None
    };

    let restoring = with_suffix(&path, ".restoring");
    if restoring.exists() {
        // Left by a restore that failed: it is never the database.
        fs::remove_file(&restoring)
            .with_context(|| format!("Failed to remove {}", restoring.display()))?;
    }
    let saved = db
        .save(&restoring)
        .with_context(|| format!("Failed to restore to {}", path.display()));
    db.close()
        .with_context(|| format!("Failed to close backup at {}", backup.display()))?;
    saved?;

    if taken {
        output::status(
            &format!("Removing existing database at {}...", path.display()),
            quiet,
        );
        // Released first: Windows removes no file or directory that has a
        // handle open.
        drop(target_lock);
        if let Err(error) = remove_database(&path, &sidecar) {
            // Best effort: the error that matters is the removal's.
            let _ = fs::remove_file(&restoring);
            return Err(error);
        }
    }
    before_install();
    install(&restoring, &path, &sidecar)
}

/// Moves the restored image `restoring` to `path`, where no database is (it
/// was absent at the check, or removed), without replacing anything: a
/// database another process created at `path` meanwhile, or its sidecar WAL
/// `sidecar` (which would be replayed into the restored database), is left as
/// it is, and the restore fails without leaving the image behind.
fn install(restoring: &Path, path: &Path, sidecar: &Path) -> Result<()> {
    let appeared = |what: &Path| {
        // Best effort: the error that matters is that the target appeared.
        let _ = fs::remove_file(restoring);
        anyhow::anyhow!(
            "{} appeared while the backup was restored: it is left as it is, and nothing was \
             restored",
            what.display()
        )
    };
    if sidecar.exists() {
        return Err(appeared(sidecar));
    }
    // A hard link fails when `path` exists, where a rename would replace it
    // (on Windows as well as on Unix).
    match fs::hard_link(restoring, path) {
        Ok(()) => {
            if fs::remove_file(restoring).is_err() {
                // The restore is complete; the next one removes the leftover.
                output::error(&format!(
                    "The database is restored, but {} could not be removed",
                    restoring.display()
                ));
            }
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Err(appeared(path)),
        // A file system without hard links (FAT): a rename after a last
        // check, which only a database created within that moment escapes.
        Err(_) if path.exists() => Err(appeared(path)),
        Err(_) => fs::rename(restoring, path).with_context(|| {
            format!(
                "Failed to rename the restored database {} to {}; it is complete, rename it \
                 yourself",
                restoring.display(),
                path.display()
            )
        }),
    }
}

/// `path` as the engine spells a database path: without trailing separators
/// and inner `.` components, and a path that ends in `.` or `..` resolved to
/// the directory it names (through the file system when it exists, lexically
/// from the current directory when it does not).
fn normalize_database_path(path: &Path) -> Result<PathBuf> {
    use std::path::Component;

    if matches!(path.components().next_back(), Some(Component::Normal(_))) {
        return Ok(path.components().collect());
    }
    if let Ok(resolved) = fs::canonicalize(path) {
        return Ok(resolved);
    }
    let absolute = std::path::absolute(path)
        .with_context(|| format!("Cannot resolve the path {}", path.display()))?;
    let mut resolved = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            other => resolved.push(other),
        }
    }
    Ok(resolved)
}

/// Refuses a target that is, or holds, the current directory of this
/// process: it cannot be removed while it is the working directory (on
/// Windows not at all, elsewhere only emptied), so the restore would end
/// with neither database.
fn refuse_current_directory(target: &Path) -> Result<()> {
    let (Ok(current), Ok(target)) = (
        std::env::current_dir().and_then(fs::canonicalize),
        fs::canonicalize(target),
    ) else {
        // Without a current directory, or without the target, there is
        // nothing it could hold.
        return Ok(());
    };
    if current.starts_with(&target) {
        anyhow::bail!(
            "The target {} is the current directory of this process, or holds it, and cannot be \
             replaced: run the restore from outside it",
            target.display()
        );
    }
    Ok(())
}

/// `path` with `suffix` appended to its last component.
fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// Refuses a backup that is the target or lies inside it: removing the
/// target would remove the backup.
fn refuse_backup_inside(backup: &Path, target: &Path) -> Result<()> {
    let (Ok(backup), Ok(target)) = (fs::canonicalize(backup), fs::canonicalize(target)) else {
        // A target that does not exist holds nothing.
        return Ok(());
    };
    if backup.starts_with(&target) {
        anyhow::bail!(
            "The backup {} is inside the target {}, which a restore removes: copy the backup \
             elsewhere first",
            backup.display(),
            target.display()
        );
    }
    Ok(())
}

/// The lock that keeps other processes away from a target the restore will
/// replace: the database file itself, or the `LOCK` file of a 0.5.x WAL
/// directory (none for a 0.5.43 directory, which has no `LOCK`). Released
/// when dropped.
struct TargetLock {
    _file: Option<fs::File>,
}

/// Checks that the existing `path` is a database no process has open, and
/// keeps it locked: a database file (0.6 or 0.5.x), or a 0.5.x WAL directory
/// (a directory holding `wal/`). Since 0.6 a database is a file, so any
/// other directory is refused, as is anything that does not open as a
/// database.
fn check_replaceable(path: &Path) -> Result<TargetLock> {
    if path.is_dir() {
        if !path.join("wal").is_dir() {
            anyhow::bail!(
                "{} is a directory and not a database (0.6 databases are files): it is not \
                 removed",
                path.display()
            );
        }
        // A 0.5.44 process holds the directory's `LOCK` while it has it open.
        let lock = path.join("LOCK");
        if !lock.exists() {
            return Ok(TargetLock { _file: None });
        }
        return Ok(TargetLock {
            _file: Some(lock_or_refuse(path, &lock)?),
        });
    }
    // Fails for a file that is not a database, and while a writer has it open.
    let db = GrafeoDB::open_read_only(path).with_context(|| {
        format!(
            "{} cannot be replaced: it does not open as a database",
            path.display()
        )
    })?;
    db.close()
        .with_context(|| format!("Failed to close {}", path.display()))?;
    // Readers hold a shared lock, which an exclusive one cannot join.
    Ok(TargetLock {
        _file: Some(lock_or_refuse(path, path)?),
    })
}

/// Locks `lock`, the lock file of the database at `path`, exclusively and
/// returns it; fails while another process (or another handle of this one)
/// holds a lock on it.
fn lock_or_refuse(path: &Path, lock: &Path) -> Result<fs::File> {
    let file =
        fs::File::open(lock).with_context(|| format!("Failed to open {}", lock.display()))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(fs::TryLockError::WouldBlock) => anyhow::bail!(
            "{} is in use by another process (database file is locked): close it first",
            path.display()
        ),
        Err(fs::TryLockError::Error(error)) => {
            Err(error).with_context(|| format!("Failed to lock {}", lock.display()))
        }
    }
}

/// Removes the database at `path`, which [`check_replaceable`] accepted (or
/// that no longer exists), and its sidecar WAL `sidecar`, whose records would
/// otherwise be replayed into the database restored at that path.
fn remove_database(path: &Path, sidecar: &Path) -> Result<()> {
    if path.is_dir() {
        fs::remove_dir_all(path).with_context(|| format!("Failed to remove {}", path.display()))?;
    } else if path.exists() {
        fs::remove_file(path).with_context(|| format!("Failed to remove {}", path.display()))?;
    }
    if sidecar.exists() {
        fs::remove_dir_all(sidecar)
            .with_context(|| format!("Failed to remove {}", sidecar.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use grafeo_engine::GrafeoDB;

    use super::run;
    use crate::{BackupCommands, OutputFormat};

    fn restore(backup: &Path, path: &Path, force: bool) -> anyhow::Result<()> {
        run(
            BackupCommands::Restore {
                backup: backup.to_path_buf(),
                path: path.to_path_buf(),
                force,
            },
            OutputFormat::Auto,
            true,
        )
    }

    /// A closed database at `path` holding one person.
    fn database_with(path: &Path, name: &str) {
        let db = GrafeoDB::open(path).unwrap();
        db.execute(&format!("INSERT (:Person {{name: '{name}'}})"))
            .unwrap();
        db.close().unwrap();
    }

    /// The names of the people in the database at `path`, read without
    /// changing it.
    fn names(path: &Path) -> Vec<String> {
        let db = GrafeoDB::open_read_only(path).unwrap();
        let names = db
            .execute("MATCH (p:Person) RETURN p.name AS name ORDER BY name")
            .unwrap()
            .rows()
            .iter()
            .map(|row| row[0].as_str().expect("a name").to_owned())
            .collect();
        db.close().unwrap();
        names
    }

    fn sidecar(path: &Path) -> PathBuf {
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        PathBuf::from(sidecar)
    }

    /// `--force` replaces an existing single-file database, and removes its
    /// sidecar WAL, which would otherwise replay the old database's last
    /// commits into the restored one.
    #[test]
    fn restore_with_force_replaces_a_database_file_and_its_wal() {
        let dir = tempfile::tempdir().unwrap();
        let backup = dir.path().join("backup.grafeo");
        let live = dir.path().join("live.grafeo");
        database_with(&backup, "Alix");
        database_with(&live, "Gus");
        std::fs::create_dir_all(sidecar(&live)).unwrap();

        restore(&backup, &live, false).expect_err("an existing target needs --force");
        restore(&backup, &live, true).unwrap_or_else(|error| panic!("{error:#}"));

        assert!(live.is_file(), "the restored database is a file");
        assert!(!sidecar(&live).exists(), "the old sidecar WAL is gone");
        assert_eq!(names(&live), vec!["Alix".to_string()]);
    }

    /// Restoring reads the backup and changes nothing in it: a backup written
    /// by 0.5.x is read in place, never migrated.
    #[test]
    fn restore_leaves_a_0_5_backup_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let backup = dir.path().join("backup.grafeo");
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../grafeo-engine/tests/fixtures/released/0.5.44/closed.grafeo");
        std::fs::copy(&fixture, &backup).unwrap();
        let before = files_under(dir.path());

        let restored = dir.path().join("restored.grafeo");
        restore(&backup, &restored, false).unwrap_or_else(|error| panic!("{error:#}"));

        let restored_names = names(&restored);
        assert_eq!(
            restored_names,
            vec!["Alix".to_string(), "Gus".to_string(), "Mia".to_string()],
            "the restored database holds the backup's people"
        );
        assert_eq!(
            restored_names,
            names(&backup),
            "the restored database holds what a read-only open of the backup shows"
        );
        let mut after = files_under(dir.path());
        after.retain(|(name, _)| name != "restored.grafeo");
        assert!(
            after == before,
            "the backup is unchanged, and was not migrated: {:?}",
            after.iter().map(|(name, _)| name).collect::<Vec<_>>()
        );
    }

    /// Every file under `dir`, by its path relative to `dir` (with `/`), with
    /// its bytes, sorted. Spill directories (`<path>.spill/`, which every open
    /// may create for memory pressure) are left out.
    fn files_under(dir: &Path) -> Vec<(String, Vec<u8>)> {
        fn walk(root: &Path, dir: &Path, files: &mut Vec<(String, Vec<u8>)>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path
                    .extension()
                    .is_some_and(|extension| extension == "spill")
                {
                    continue;
                }
                if path.is_dir() {
                    files.push((relative(root, &path) + "/", Vec::new()));
                    walk(root, &path, files);
                } else {
                    files.push((relative(root, &path), std::fs::read(&path).unwrap()));
                }
            }
        }
        fn relative(root: &Path, path: &Path) -> String {
            path.strip_prefix(root)
                .unwrap()
                .components()
                .map(|part| part.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/")
        }
        let mut files = Vec::new();
        walk(dir, dir, &mut files);
        files.sort();
        files
    }

    /// A misspelled backup with `--force` fails before the target is
    /// touched: the database and its sidecar WAL stay as they were.
    #[test]
    fn a_missing_backup_leaves_the_target_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let live = dir.path().join("live.grafeo");
        database_with(&live, "Gus");
        std::fs::create_dir_all(sidecar(&live)).unwrap();
        std::fs::write(sidecar(&live).join("wal_00000000.log"), b"Vincent").unwrap();
        let before = files_under(dir.path());

        let error = restore(&dir.path().join("backup.grafeo"), &live, true)
            .expect_err("a missing backup fails");
        assert!(
            format!("{error:#}").contains("Backup not found"),
            "{error:#}"
        );
        assert!(
            files_under(dir.path()) == before,
            "the target and its sidecar WAL are unchanged"
        );
    }

    /// A backup that cannot be opened fails before the target is touched.
    #[test]
    fn an_unreadable_backup_leaves_the_target_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let live = dir.path().join("live.grafeo");
        database_with(&live, "Gus");
        let backup = dir.path().join("backup.grafeo");
        std::fs::write(&backup, b"not a database, just Amsterdam").unwrap();
        let before = files_under(dir.path());

        let error = restore(&backup, &live, true).expect_err("an unreadable backup fails");
        assert!(
            format!("{error:#}").contains("Failed to open backup"),
            "{error:#}"
        );
        assert!(
            files_under(dir.path()) == before,
            "the target and the backup are unchanged"
        );
        assert_eq!(names(&live), vec!["Gus".to_string()]);
    }

    /// A backup inside the target (a 0.5.x WAL directory holding it) would be
    /// removed with the target: the restore refuses and changes nothing.
    #[test]
    fn a_backup_inside_the_target_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        copy_dir(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../grafeo-engine/tests/fixtures/released/0.5.44/directory"),
            &data,
        );
        let backup = data.join("backup.grafeo");
        database_with(&backup, "Alix");
        let before = files_under(dir.path());

        let error = restore(&backup, &data, true).expect_err("a backup inside the target fails");
        assert!(format!("{error:#}").contains("inside"), "{error:#}");
        assert!(
            files_under(dir.path()) == before,
            "the target and the backup are unchanged"
        );
    }

    fn copy_dir(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let path = entry.unwrap().path();
            let target = to.join(path.file_name().unwrap());
            if path.is_dir() {
                copy_dir(&path, &target);
            } else {
                std::fs::copy(&path, &target).unwrap();
            }
        }
    }

    /// `--force` removes a 0.6 database file with its sidecar WAL: the old
    /// WAL's commits are not replayed into the restored database.
    #[test]
    fn restore_with_force_over_a_file_with_a_sidecar_wal_replays_none_of_it() {
        let dir = tempfile::tempdir().unwrap();
        let backup = dir.path().join("backup.grafeo");
        database_with(&backup, "Alix");
        let live = dir.path().join("live.grafeo");
        // The live database, left as a crash leaves it: Gus is only in its
        // sidecar WAL.
        crashed_database(
            "commands::backup::tests::restore_with_force_over_a_file_with_a_sidecar_wal_replays_none_of_it",
            &live,
        );
        assert!(
            std::fs::read_dir(sidecar(&live)).unwrap().count() > 0,
            "the live database has a sidecar WAL"
        );

        restore(&backup, &live, true).unwrap_or_else(|error| panic!("{error:#}"));

        assert!(!sidecar(&live).exists(), "the old sidecar WAL is gone");
        assert_eq!(names(&live), vec!["Alix".to_string()]);
        assert!(
            !dir.path().join("live.grafeo.restoring").exists(),
            "no image is left behind"
        );
    }

    /// A backup copied from a database that was not closed (a file-system
    /// snapshot, a crashed server's files) has commits in its sidecar WAL:
    /// the restore holds every one, and changes nothing in the backup.
    #[test]
    fn a_backup_with_a_sidecar_wal_restores_every_commit() {
        let dir = tempfile::tempdir().unwrap();
        let backup = dir.path().join("backup.grafeo");
        crashed_database(
            "commands::backup::tests::a_backup_with_a_sidecar_wal_restores_every_commit",
            &backup,
        );
        let before = files_under(dir.path());

        let restored = dir.path().join("restored.grafeo");
        restore(&backup, &restored, false).unwrap_or_else(|error| panic!("{error:#}"));

        assert_eq!(
            names(&restored),
            vec!["Alix".to_string(), "Gus".to_string()],
            "the restore holds the commit only the backup's WAL has"
        );
        let mut after = files_under(dir.path());
        after.retain(|(name, _)| name != "restored.grafeo");
        assert!(after == before, "the backup and its WAL are unchanged");
    }

    /// The path where [`crashed_database`] runs in a child process.
    const CHILD_PATH: &str = "GRAFEO_CLI_CRASHED_DATABASE";

    /// A database at `path` as a process leaves it that exits without
    /// `close()`: the file holds Alix (checkpointed), and only its sidecar
    /// WAL holds Gus. `test` is the calling test's path: the child runs it.
    fn crashed_database(test: &str, path: &Path) {
        if let Some(child_path) = std::env::var_os(CHILD_PATH) {
            let db = GrafeoDB::open(Path::new(&child_path)).unwrap();
            db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
            db.wal_checkpoint().unwrap();
            db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
            // A crash: no close(), no checkpoint, no destructors.
            std::process::exit(0);
        }
        let status = grafeo_common::testing::child_process::run(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", test, "--include-ignored", "--nocapture"])
                .env(CHILD_PATH, path),
        )
        .unwrap();
        assert!(status.success(), "the child process failed");
        assert!(
            path.is_file(),
            "the child process created {}",
            path.display()
        );
    }

    /// A database another process creates at an absent target while the
    /// backup is restored is never replaced: the restore fails, the new
    /// database stays as it is, and no image is left behind. The same holds
    /// with `--force`, for a database created after the old one was removed.
    #[test]
    fn a_target_that_appears_during_the_restore_is_never_replaced() {
        for force in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let backup = dir.path().join("backup.grafeo");
            database_with(&backup, "Alix");
            let live = dir.path().join("live.grafeo");
            if force {
                database_with(&live, "Vincent");
            }

            let error = super::restore_with(&backup, &live, force, true, || {
                assert!(!live.exists(), "force {force}: the target is absent");
                database_with(&live, "Gus");
            })
            .expect_err("a target that appeared is not replaced");

            // The target itself appeared (not its sidecar WAL): the
            // no-replace install refused it.
            assert!(
                format!("{error:#}").contains(&format!("{} appeared", live.display())),
                "force {force}: the error names the target: {error:#}"
            );
            assert_eq!(
                names(&live),
                vec!["Gus".to_string()],
                "force {force}: the database that appeared is untouched"
            );
            assert!(
                !dir.path().join("live.grafeo.restoring").exists(),
                "force {force}: no image is left behind"
            );
        }
    }

    /// A sidecar WAL that appears at an absent target during the restore
    /// would be replayed into the restored database: the restore fails and
    /// leaves it as it is.
    #[test]
    fn a_sidecar_wal_that_appears_during_the_restore_is_never_adopted() {
        let dir = tempfile::tempdir().unwrap();
        let backup = dir.path().join("backup.grafeo");
        database_with(&backup, "Alix");
        let live = dir.path().join("live.grafeo");

        let error = super::restore_with(&backup, &live, false, true, || {
            std::fs::create_dir_all(sidecar(&live)).unwrap();
            std::fs::write(sidecar(&live).join("wal_00000000.log"), b"Vincent").unwrap();
        })
        .expect_err("a sidecar WAL that appeared is not adopted");

        assert!(
            format!("{error:#}").contains(&format!("{} appeared", sidecar(&live).display())),
            "the error names the sidecar WAL: {error:#}"
        );
        assert!(!live.exists(), "nothing was restored");
        assert_eq!(
            std::fs::read(sidecar(&live).join("wal_00000000.log")).unwrap(),
            b"Vincent",
            "the sidecar WAL is untouched"
        );
        assert!(
            !dir.path().join("live.grafeo.restoring").exists(),
            "no image is left behind"
        );
    }

    /// A backup path spelled with a trailing separator (`backup.grafeo/`)
    /// names the backup file, as the target path and the engine read it.
    #[test]
    fn a_backup_path_with_a_trailing_separator_is_restored() {
        let dir = tempfile::tempdir().unwrap();
        let backup = dir.path().join("backup.grafeo");
        database_with(&backup, "Alix");
        let restored = dir.path().join("restored.grafeo");

        restore(&dir.path().join("backup.grafeo/"), &restored, false)
            .unwrap_or_else(|error| panic!("{error:#}"));

        assert_eq!(names(&restored), vec!["Alix".to_string()]);
    }

    /// A target directory that is not a database (0.6 databases are files)
    /// is never removed.
    #[test]
    fn a_directory_that_is_not_a_database_is_never_removed() {
        let dir = tempfile::tempdir().unwrap();
        let backup = dir.path().join("backup.grafeo");
        database_with(&backup, "Alix");
        let target = dir.path().join("exports");
        std::fs::create_dir(&target).unwrap();
        std::fs::write(target.join("people.csv"), b"Alix,Amsterdam").unwrap();
        let before = files_under(dir.path());

        let error = restore(&backup, &target, true).expect_err("a plain directory is refused");
        assert!(format!("{error:#}").contains("not a database"), "{error:#}");
        assert!(files_under(dir.path()) == before, "nothing was changed");
    }

    /// A database another process (or this one) has open is not removed.
    #[test]
    fn a_target_in_use_is_not_removed() {
        let dir = tempfile::tempdir().unwrap();
        let backup = dir.path().join("backup.grafeo");
        database_with(&backup, "Alix");
        let live = dir.path().join("live.grafeo");
        database_with(&live, "Gus");

        for read_only in [false, true] {
            let in_use = if read_only {
                GrafeoDB::open_read_only(&live).unwrap()
            } else {
                GrafeoDB::open(&live).unwrap()
            };
            let error =
                restore(&backup, &live, true).expect_err("a database in use is not replaced");
            assert!(
                format!("{error:#}").contains("locked") || format!("{error:#}").contains("in use"),
                "read_only {read_only}: {error:#}"
            );
            in_use.close().unwrap();
            drop(in_use);
            assert!(
                !dir.path().join("live.grafeo.restoring").exists(),
                "read_only {read_only}: no image is left behind"
            );
            assert_eq!(
                names(&live),
                vec!["Gus".to_string()],
                "read_only {read_only}"
            );
        }
    }

    /// The in-use check keeps the target locked until the restore removes
    /// it: a process that opens it meanwhile (while the image is written)
    /// fails, instead of writing to a database that is about to go.
    #[test]
    fn the_target_stays_locked_from_the_check_until_its_removal() {
        let dir = tempfile::tempdir().unwrap();
        let live = dir.path().join("live.grafeo");
        database_with(&live, "Gus");

        let lock = super::check_replaceable(&live).unwrap();
        for read_only in [false, true] {
            let opened = if read_only {
                GrafeoDB::open_read_only(&live)
            } else {
                GrafeoDB::open(&live)
            };
            match opened {
                Ok(_) => panic!("read_only {read_only}: the target opened while checked"),
                Err(error) => assert!(
                    error.to_string().contains("locked"),
                    "read_only {read_only}: {error}"
                ),
            }
        }
        drop(lock);
        let db = GrafeoDB::open(&live).expect("released, the target opens again");
        db.close().unwrap();

        // A 0.5.44 WAL directory: its `LOCK` stays held.
        let data = dir.path().join("data");
        copy_dir(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../grafeo-engine/tests/fixtures/released/0.5.44/directory"),
            &data,
        );
        let lock = super::check_replaceable(&data).unwrap();
        let file = std::fs::File::open(data.join("LOCK")).unwrap();
        assert!(
            matches!(file.try_lock(), Err(std::fs::TryLockError::WouldBlock)),
            "the directory's LOCK is held while checked"
        );
        drop(lock);
        file.try_lock().expect("released, the LOCK is free");
    }

    /// The backup [`restore_into_the_current_directory_child`] restores.
    const CURRENT_DIRECTORY_CHILD_VAR: &str = "GRAFEO_CLI_RESTORE_INTO_CURRENT";
    /// The target it restores to (`.` or `..`).
    const CURRENT_DIRECTORY_TARGET_VAR: &str = "GRAFEO_CLI_RESTORE_TARGET";

    /// Child-process entry for
    /// [`a_target_that_is_or_holds_the_current_directory_is_refused`]; a no-op
    /// when run directly. Restores with `--force` to the target it is given,
    /// from inside it, and prints the outcome.
    #[test]
    fn restore_into_the_current_directory_child() {
        let (Some(backup), Some(target)) = (
            std::env::var_os(CURRENT_DIRECTORY_CHILD_VAR),
            std::env::var_os(CURRENT_DIRECTORY_TARGET_VAR),
        ) else {
            return;
        };
        match restore(Path::new(&backup), Path::new(&target), true) {
            Ok(()) => println!("RESTORED"),
            Err(error) => println!("REFUSED: {error:#}"),
        }
        std::process::exit(0);
    }

    /// A restore into the current directory (`.`), or into a directory that
    /// holds it (`..` from inside), is refused: the target cannot be removed
    /// while it is a process's working directory, and `.restoring` would be
    /// written inside it. Before, such a restore emptied the 0.5.x directory
    /// and lost both databases. The paths are resolved as the engine resolves
    /// them.
    #[test]
    fn a_target_that_is_or_holds_the_current_directory_is_refused() {
        for (inside, target) in [("", "."), ("wal", "..")] {
            let dir = tempfile::tempdir().unwrap();
            let data = dir.path().join("data");
            copy_dir(
                &Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../grafeo-engine/tests/fixtures/released/0.5.44/directory"),
                &data,
            );
            let backup = dir.path().join("backup.grafeo");
            database_with(&backup, "Alix");
            let before = files_under(dir.path());

            let output = grafeo_common::testing::child_process::output(
                std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "commands::backup::tests::restore_into_the_current_directory_child",
                        "--nocapture",
                    ])
                    .env(CURRENT_DIRECTORY_CHILD_VAR, &backup)
                    .env(CURRENT_DIRECTORY_TARGET_VAR, target)
                    .current_dir(data.join(inside)),
            )
            .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success()
                    && stdout.contains("REFUSED")
                    && stdout.contains("current directory"),
                "{target} from data/{inside}: the restore is refused: {stdout}\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                files_under(dir.path()) == before,
                "{target} from data/{inside}: nothing changed"
            );
        }
    }
}
