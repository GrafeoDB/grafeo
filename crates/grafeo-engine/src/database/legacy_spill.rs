//! Embeddings an older build left in `<file>.spill` (#594).
//!
//! Before 0.6, spilling a vector index moved the embeddings of its column into
//! `vectors_<label>%3A<property>.bin` in the spill directory (`<file>.spill`,
//! or the configured spill path) and out of the database: a database closed
//! while spilled holds them only there. An open of a database file folds them
//! back into their columns as a load step, like WAL replay (no WAL record, no
//! CDC event, no new epoch): a value fills a node that exists and has none,
//! so one written later, which the file holds, wins, and a second fold
//! changes nothing. A read-write open then checkpoints and deletes the files
//! it folded from `<file>.spill` (a crash in between folds them again); a
//! read-only open keeps them. Each file is read in batches of at most
//! [`BATCH_BYTES`], so the fold needs memory for the values it fills and one
//! batch, not for the whole file. The vector index of every file found, read
//! or kept, is rebuilt from the stored values: its topology was saved while
//! the values could not be read, so it can miss an embedding set while
//! spilled or name a node that has none (a read-write open checkpoints the
//! rebuilt index, a read-only one keeps it in memory).
//!
//! Old files in a configured spill path are read only while a 0.5.x database
//! is read in place: by the migration (its read of the old database puts them
//! into the new 0.6 file) and by a read-only open of a 0.5.x database or of a
//! kept `.pre-0.6` copy. They are never deleted, as other databases may share
//! the path and the kept copy needs them to go back; a 0.6 database never
//! reads them again, so an embedding removed after the upgrade stays removed.
//! A database written by a 0.6 development build that spilled to a configured
//! spill path does not get those embeddings back.
//!
//! A file stays where it is, and is not read, when this database has no
//! vector index for its `label:property` (a spill path may be shared, and
//! another database may own it; one in `<file>.spill` is logged, as a crash
//! can have lost the index, #401) or when its vectors have another number of
//! dimensions than the index (a foreign or damaged file, warned). One that
//! cannot be read stays too (warned).
//!
//! A value fills only a node that carries the label of the file's vector
//! index. Each file is folded once: `<file>.spill` at the first open (which
//! deletes it), a configured spill path by the migration. Limits, as the old
//! files record no removals and no epochs: a property removed while spilled
//! before 0.6 comes back (its reload brought it back too); 0.5.x databases
//! sharing one spill path wrote over each other's files, so the files cannot
//! say which database they came from; and a node of the index's label that
//! 0.5.x created after a node was deleted while spilled can have taken its id
//! (ids are not kept across opens), and with it the deleted node's
//! embedding.
//!
//! File layout (`MmapStorage` before 0.6, little-endian): a 64-byte header
//! (magic `GRAFVEC1`, dimensions `u64` at 8, count `u64` at 16), then `count`
//! records of a node id (`u64`) and `dimensions` `f32` values.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_core::graph::lpg::LpgStore;

const MAGIC: [u8; 8] = *b"GRAFVEC1";
const HEADER_BYTES: usize = 64;

/// The most record bytes one batch of a fold reads (at least one record).
pub(crate) const BATCH_BYTES: usize = 8 << 20;

/// A spill file an older build left: the embeddings of one vector index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LegacyVectorFile {
    pub(crate) path: PathBuf,
    pub(crate) label: String,
    pub(crate) property: PropertyKey,
    /// Found in the database's own `<file>.spill`, not in a configured spill
    /// path.
    pub(crate) derived: bool,
}

/// The old vector spill files at the top level of `spill_dir`, by name
/// (`cache/` and query directories are not old files), marked `derived` when
/// `spill_dir` is the database's own `<file>.spill`. A name without a
/// `label:property` key is an older encoding nothing reads: left alone.
pub(crate) fn find(spill_dir: &Path, derived: bool) -> Vec<LegacyVectorFile> {
    let Ok(entries) = std::fs::read_dir(spill_dir) else {
        return Vec::new();
    };
    let mut files: Vec<LegacyVectorFile> = entries
        .flatten()
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let key = name.strip_prefix("vectors_")?.strip_suffix(".bin")?;
            // "Label%3Aproperty": `:` was written as %3A and `%` as %25.
            let key = key.replace("%3A", ":").replace("%25", "%");
            let (label, property) = key.split_once(':')?;
            Some(LegacyVectorFile {
                path: entry.path(),
                label: label.to_string(),
                property: PropertyKey::new(property),
                derived,
            })
        })
        .collect();
    files.sort_by(|a, b| a.path.cmp(&b.path));
    files
}

/// An old spill file whose header was read and checked against its length,
/// read record by record from there.
pub(crate) struct OldFile {
    reader: BufReader<File>,
    path: PathBuf,
    /// The number of `f32` values of each vector.
    pub(crate) dimensions: usize,
    /// The records not read yet.
    remaining: usize,
}

impl OldFile {
    /// Opens the old spill file at `path` and checks its header.
    ///
    /// # Errors
    ///
    /// Returns the error of opening or reading it (naming the file), or
    /// `InvalidData` for a file that is not an old spill file or is shorter
    /// than its header says.
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        let named = |error: io::Error| {
            io::Error::new(
                error.kind(),
                format!("old spill file {}: {error}", path.display()),
            )
        };
        let invalid = |what: &str| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("old spill file {}: {what}", path.display()),
            )
        };
        let file = File::open(path).map_err(named)?;
        let length = file.metadata().map_err(named)?.len();
        let body = length
            .checked_sub(HEADER_BYTES as u64)
            .ok_or_else(|| invalid("no spill file header"))?;
        let mut reader = BufReader::new(file);
        let mut header = [0u8; HEADER_BYTES];
        reader.read_exact(&mut header).map_err(named)?;
        if header[..8] != MAGIC {
            return Err(invalid("no spill file header"));
        }
        let number = |at: usize| -> io::Result<usize> {
            let raw: [u8; 8] = header[at..at + 8]
                .try_into()
                .map_err(|_| invalid("header"))?;
            usize::try_from(u64::from_le_bytes(raw))
                .map_err(|_| invalid("header number out of range"))
        };
        let (dimensions, count) = (number(8)?, number(16)?);
        let record =
            Self::record_bytes(dimensions).ok_or_else(|| invalid("dimensions out of range"))?;
        count
            .checked_mul(record)
            .and_then(|size| u64::try_from(size).ok())
            .filter(|size| *size <= body)
            .ok_or_else(|| invalid("shorter than its header says"))?;
        Ok(Self {
            reader,
            path: path.to_path_buf(),
            dimensions,
            remaining: count,
        })
    }

    /// The bytes of one record: the id, then the values.
    fn record_bytes(dimensions: usize) -> Option<usize> {
        dimensions
            .checked_mul(4)
            .and_then(|values| values.checked_add(8))
    }

    /// The next records, at most `max_bytes` of them (at least one record);
    /// empty once every record was read.
    ///
    /// # Errors
    ///
    /// Returns the error of reading the file, naming it.
    pub(crate) fn next_batch(&mut self, max_bytes: usize) -> io::Result<Vec<(NodeId, Vec<f32>)>> {
        // Checked in `open`.
        let record = Self::record_bytes(self.dimensions).unwrap_or(usize::MAX);
        let take = (max_bytes / record)
            .clamp(1, usize::MAX)
            .min(self.remaining);
        let mut bytes = vec![0u8; take * record];
        self.reader.read_exact(&mut bytes).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("old spill file {}: {error}", self.path.display()),
            )
        })?;
        self.remaining -= take;
        Ok(bytes
            .chunks_exact(record)
            .map(|chunk| {
                let (id, values) = chunk.split_at(8);
                let id = u64::from_le_bytes(id.try_into().unwrap_or_default());
                let vector = values
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|value| f32::from_le_bytes(*value))
                    .collect();
                (NodeId::new(id), vector)
            })
            .collect())
    }
}

/// What a fold did.
#[derive(Debug, Default)]
pub(crate) struct Folded {
    /// The files it read through: safe to delete once the database holds
    /// their values.
    pub(crate) files: Vec<LegacyVectorFile>,
    /// How many values it filled, from every file (a file whose read stopped
    /// partway included).
    pub(crate) filled: usize,
    /// The vector indexes, as `(label, property)`, that had an old file,
    /// folded or kept: their topology was saved while their values could not
    /// be read, so they are rebuilt from the stored values.
    pub(crate) indexes: BTreeSet<(String, String)>,
    /// How many batches it read, every file together.
    pub(crate) batches: usize,
}

/// Folds the `files` whose index `store` has into it, each named in a
/// warning. A file stays out of the fold, and of [`Folded::files`] (so it is
/// never deleted), when `store` has no index for it (logged for
/// `<file>.spill`), when its dimensions differ from the index's, or when it
/// cannot be read (both warned, naming the file; values read before a read
/// that stops partway stay, counted in the warning).
pub(crate) fn fold_in(store: &LpgStore, files: &[LegacyVectorFile]) -> Folded {
    fold_in_batches(store, files, BATCH_BYTES)
}

/// [`fold_in`], reading at most `batch_bytes` of records at a time.
pub(crate) fn fold_in_batches(
    store: &LpgStore,
    files: &[LegacyVectorFile],
    batch_bytes: usize,
) -> Folded {
    let mut folded = Folded::default();
    for file in files {
        let Some(index) = store.get_vector_index(&file.label, file.property.as_str()) else {
            if file.derived {
                grafeo_common::grafeo_info!(
                    "{} stays: this database has no vector index on :{}({}) (a crash may \
                     have lost one created since the last checkpoint); create it and open the \
                     database again to read the embeddings it holds",
                    file.path.display(),
                    file.label,
                    file.property.as_str()
                );
            }
            continue;
        };
        folded
            .indexes
            .insert((file.label.clone(), file.property.as_str().to_string()));
        let dimensions = index.config().dimensions;
        let mut old = match OldFile::open(&file.path) {
            Ok(old) => old,
            Err(error) => {
                grafeo_common::grafeo_warn!(
                    "the embeddings of :{}({}) in {} stay out of the database, and the file \
                     stays: {error}",
                    file.label,
                    file.property.as_str(),
                    file.path.display()
                );
                continue;
            }
        };
        if old.dimensions != dimensions {
            grafeo_common::grafeo_warn!(
                "{} stays and none of its embeddings are read: they have {} dimensions, the \
                 vector index on :{}({}) has {}",
                file.path.display(),
                old.dimensions,
                file.label,
                file.property.as_str(),
                dimensions
            );
            continue;
        }
        let (filled, batches, stopped) = fill_from(store, file, &mut old, batch_bytes);
        folded.filled += filled;
        folded.batches += batches;
        match stopped {
            None => {
                grafeo_common::grafeo_warn!(
                    "folded {filled} embeddings of :{}({}) back into the database from {}, \
                     which an older version spilled them to",
                    file.label,
                    file.property.as_str(),
                    file.path.display()
                );
                folded.files.push(file.clone());
            }
            Some(error) => grafeo_common::grafeo_warn!(
                "reading {} stopped: {error}; the {filled} embeddings of :{}({}) read before \
                 stay in the database, the rest stay out, and the file stays",
                file.path.display(),
                file.label,
                file.property.as_str()
            ),
        }
    }
    folded
}

/// Fills the values of `old` into `store`, at most `batch_bytes` of records
/// at a time. Returns how many it filled, how many batches it read, and the
/// error that stopped the read, if one did (what was filled before stays).
fn fill_from(
    store: &LpgStore,
    file: &LegacyVectorFile,
    old: &mut OldFile,
    batch_bytes: usize,
) -> (usize, usize, Option<io::Error>) {
    let (mut filled, mut batches) = (0, 0);
    loop {
        let batch = match old.next_batch(batch_bytes) {
            Ok(batch) => batch,
            Err(error) => return (filled, batches, Some(error)),
        };
        if batch.is_empty() {
            return (filled, batches, None);
        }
        batches += 1;
        filled += store.fill_missing_node_values(
            &file.label,
            &file.property,
            batch
                .into_iter()
                .map(|(id, vector)| (id, Value::Vector(vector.into()))),
        );
    }
}

/// Rebuilds the vector index on `label:property` from the stored values, with
/// its configuration: a topology saved while the column was spilled was kept
/// with vectors it could not read, so it can miss a node whose embedding
/// changed then, or name one that has none.
fn rebuild_index(store: &LpgStore, label: &str, property: &str) {
    use grafeo_core::index::vector::{
        PropertyVectorAccessor, QuantizationType, VectorAccessor, VectorIndexKind,
    };

    let Some(old) = store.get_vector_index(label, property) else {
        return;
    };
    let config = old.config().clone();
    let accessor = PropertyVectorAccessor::new(store, property);
    let vectors: Vec<(NodeId, std::sync::Arc<[f32]>)> = store
        .nodes_by_label(label)
        .into_iter()
        .filter_map(|id| {
            accessor
                .get_vector(id)
                .filter(|vector| vector.len() == config.dimensions)
                .map(|vector| (id, vector))
        })
        .collect();
    let index = super::GrafeoDB::build_vector_index(
        config.dimensions,
        config.metric,
        Some(config.m),
        Some(config.ef_construction),
        old.quantization_type().unwrap_or(QuantizationType::None),
        vectors.len(),
    );
    match &index {
        VectorIndexKind::Hnsw(_) => {
            for (id, vector) in &vectors {
                index.insert(*id, vector, &accessor);
            }
        }
        VectorIndexKind::Quantized(quantized) => {
            for (id, vector) in &vectors {
                quantized.insert(*id, vector);
            }
        }
    }
    store.add_vector_index(label, property, std::sync::Arc::new(index));
}

/// Every record of the old spill file at `path`.
#[cfg(test)]
pub(crate) fn read(path: &Path) -> io::Result<Vec<(NodeId, Vec<f32>)>> {
    let mut old = OldFile::open(path)?;
    let mut records = Vec::new();
    loop {
        let batch = old.next_batch(BATCH_BYTES)?;
        if batch.is_empty() {
            return Ok(records);
        }
        records.extend(batch);
    }
}

/// A directory an older build may have spilled embeddings to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SpillDirectory {
    pub(crate) path: PathBuf,
    /// The derived `<file>.spill`, removed once a fold leaves it empty; a
    /// configured spill path is the user's and stays.
    pub(crate) derived: bool,
}

/// The directories an older build of the database file at `database` may
/// have spilled embeddings to: `<file>.spill`, and the configured spill path.
/// None for an in-memory database, which has nothing spilled by an earlier
/// open.
pub(crate) fn directories(
    database: Option<&Path>,
    spill_path: Option<&Path>,
) -> Vec<SpillDirectory> {
    let Some(database) = database else {
        return Vec::new();
    };
    let mut directories = Vec::new();
    if let (Some(parent), Some(name)) = (database.parent(), database.file_name()) {
        let mut name = name.to_os_string();
        name.push(".spill");
        directories.push(SpillDirectory {
            path: parent.join(name),
            derived: true,
        });
    }
    if let Some(spill_path) = spill_path
        && directories.iter().all(|known| known.path != spill_path)
    {
        directories.push(SpillDirectory {
            path: spill_path.to_path_buf(),
            derived: false,
        });
    }
    directories
}

impl super::GrafeoDB {
    /// Folds the embeddings an older build spilled to `directories` back into
    /// their columns (see the module docs; the caller passes a configured
    /// spill path only when it reads a 0.5.x database). A read-write open then
    /// makes them durable with a checkpoint and deletes the files it folded
    /// from `<file>.spill`; a crash before the delete folds them again at the
    /// next open, which changes nothing and then deletes them. A file in a
    /// configured spill path stays. A read-only open keeps them in memory.
    ///
    /// # Errors
    ///
    /// Returns the error of the checkpoint; the files then stay.
    pub(super) fn fold_in_legacy_spill(
        &self,
        directories: &[SpillDirectory],
    ) -> grafeo_common::utils::error::Result<()> {
        let files: Vec<LegacyVectorFile> = directories
            .iter()
            .flat_map(|directory| find(&directory.path, directory.derived))
            .collect();
        if files.is_empty() {
            return Ok(());
        }
        let folded = fold_in(self.lpg_store(), &files);
        for (label, property) in &folded.indexes {
            rebuild_index(self.lpg_store(), label, property);
        }
        let shared: Vec<String> = folded
            .files
            .iter()
            .filter(|file| !file.derived)
            .map(|file| file.path.display().to_string())
            .collect();
        if !shared.is_empty() {
            grafeo_common::grafeo_info!(
                "read the old spill files of this 0.5.x database in the configured spill path: \
                 {}; they stay there (other databases may share the path, and the kept 0.5.x \
                 copy needs them), and a 0.6 database does not read them again",
                shared.join(", ")
            );
        }
        if self.read_only || (folded.files.is_empty() && folded.indexes.is_empty()) {
            return Ok(());
        }
        // The filled values and the rebuilt indexes are durable before a file
        // goes.
        if folded.filled > 0 || !folded.indexes.is_empty() {
            grafeo_common::testing::crash::maybe_crash("legacy_spill:after_fill");
            self.wal_checkpoint()?;
        }
        grafeo_common::testing::crash::maybe_crash("legacy_spill:before_delete");
        // Only `<file>.spill` is this database's own: in a configured spill
        // path, which databases may share, another one may still need a file.
        for file in folded.files.iter().filter(|file| file.derived) {
            if let Err(error) = std::fs::remove_file(&file.path) {
                grafeo_common::grafeo_warn!(
                    "could not remove the folded spill file {}: {error}",
                    file.path.display()
                );
            }
        }
        for directory in directories.iter().filter(|directory| directory.derived) {
            let _ = std::fs::remove_dir(&directory.path);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes an old spill file with `records`, as `MmapStorage` did.
    fn write_old(path: &Path, dimensions: usize, records: &[(u64, Vec<f32>)]) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&(dimensions as u64).to_le_bytes());
        bytes.extend_from_slice(&(records.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.resize(HEADER_BYTES, 0);
        for (id, vector) in records {
            bytes.extend_from_slice(&id.to_le_bytes());
            for value in vector {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn the_directories_are_the_derived_one_and_the_spill_path() {
        let database = Path::new("db").join("paris.grafeo");
        let derived = Path::new("db").join("paris.grafeo.spill");
        assert_eq!(
            directories(Some(&database), None),
            vec![SpillDirectory {
                path: derived.clone(),
                derived: true
            }]
        );
        let shared = Path::new("shared");
        assert_eq!(
            directories(Some(&database), Some(shared)),
            vec![
                SpillDirectory {
                    path: derived.clone(),
                    derived: true
                },
                SpillDirectory {
                    path: shared.to_path_buf(),
                    derived: false
                }
            ]
        );
        assert_eq!(directories(Some(&database), Some(&derived)).len(), 1);
        assert_eq!(directories(None, Some(shared)), Vec::new(), "in memory");
    }

    #[test]
    fn finds_old_files_by_name_and_decodes_the_property() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "vectors_Item%3Aembedding.bin",
            "vectors_Doc%3Aemb%25b.bin",
            "vectors_Itemembedding.bin",
            "other.bin",
        ] {
            std::fs::write(dir.path().join(name), b"").unwrap();
        }
        std::fs::create_dir(dir.path().join("cache")).unwrap();
        std::fs::write(
            dir.path()
                .join("cache")
                .join("vectors_Item%3Aembedding.bin"),
            b"",
        )
        .unwrap();

        let found = find(dir.path(), true);
        let properties: Vec<&str> = found.iter().map(|file| file.property.as_str()).collect();
        assert_eq!(properties, ["emb%b", "embedding"]);
        assert!(
            found
                .iter()
                .all(|file| file.path.parent() == Some(dir.path()))
        );
        assert!(found.iter().all(|file| file.derived));
        assert_eq!(find(&dir.path().join("missing"), false), Vec::new());
    }

    #[test]
    fn reads_the_records_and_refuses_a_short_or_foreign_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors_Item%3Aembedding.bin");
        write_old(&path, 2, &[(3, vec![3.0, 19.0]), (88, vec![88.0, 3.19])]);
        assert_eq!(
            read(&path).unwrap(),
            vec![
                (NodeId::new(3), vec![3.0, 19.0]),
                (NodeId::new(88), vec![88.0, 3.19])
            ]
        );

        let bytes = std::fs::read(&path).unwrap();
        let mut other_magic = bytes.clone();
        other_magic[..8].copy_from_slice(b"GRAFVEC0");
        std::fs::write(&path, &other_magic).unwrap();
        assert!(read(&path).is_err(), "a full header with another magic");
        std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
        assert!(read(&path).is_err(), "shorter than its header says");
        std::fs::write(&path, b"GRAFVEC0").unwrap();
        assert!(read(&path).is_err(), "no header");

        // A count whose records fill nearly the whole address space: the
        // length check must refuse it, not overflow.
        let mut huge = bytes[..HEADER_BYTES].to_vec();
        huge[16..24].copy_from_slice(&(u64::MAX / 16).to_le_bytes());
        std::fs::write(&path, &huge).unwrap();
        assert!(read(&path).is_err(), "a count no file can hold");
    }

    /// A file is read in batches of at most the given bytes (at least one
    /// record each), in order.
    #[test]
    fn a_file_is_read_in_batches_of_bounded_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors_Item%3Aembedding.bin");
        let records: Vec<(u64, Vec<f32>)> = [3, 19, 88, 319, 1988]
            .into_iter()
            .map(|id| (id, vec![id as f32, 3.19]))
            .collect();
        write_old(&path, 2, &records);

        // A record is 8 + 2 * 4 = 16 bytes.
        for (max_bytes, sizes) in [(40, vec![2, 2, 1]), (1, vec![1, 1, 1, 1, 1])] {
            let mut old = OldFile::open(&path).unwrap();
            assert_eq!(old.dimensions, 2);
            let mut read_back = Vec::new();
            let mut batch_sizes = Vec::new();
            loop {
                let batch = old.next_batch(max_bytes).unwrap();
                if batch.is_empty() {
                    break;
                }
                batch_sizes.push(batch.len());
                read_back.extend(batch);
            }
            assert_eq!(batch_sizes, sizes, "{max_bytes} bytes a batch");
            let expected: Vec<(NodeId, Vec<f32>)> = records
                .iter()
                .map(|(id, vector)| (NodeId::new(*id), vector.clone()))
                .collect();
            assert_eq!(read_back, expected);
        }
    }

    /// A file whose vectors have another number of dimensions than the index
    /// (0 and 3 against 2) is not read: it fills nothing and stays out of the
    /// fold, so it is never deleted.
    #[test]
    fn a_file_of_another_dimension_is_left_out() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors_Item%3Aembedding.bin");
        let db = crate::GrafeoDB::new_in_memory();
        let alix = db
            .create_node_with_props(&["Item"], [("name", Value::from("Alix"))])
            .unwrap();
        db.create_vector_index("Item", "embedding", Some(2), None, None, None, None)
            .unwrap();
        let store = db.lpg_store();

        for dimensions in [0, 3] {
            let vector: Vec<f32> = [3.0, 19.0, 88.0].into_iter().take(dimensions).collect();
            write_old(&path, dimensions, &[(alix.as_u64(), vector)]);
            let folded = fold_in(store, &find(dir.path(), true));
            assert!(folded.files.is_empty(), "{dimensions} dimensions");
            assert_eq!(folded.filled, 0);
            assert_eq!(
                store.get_node_property(alix, &PropertyKey::new("embedding")),
                None
            );
        }
    }

    /// Items with a 2-dimension index: an in-memory database and the ids of
    /// `count` items.
    fn items(count: usize) -> (crate::GrafeoDB, Vec<NodeId>) {
        let db = crate::GrafeoDB::new_in_memory();
        let ids = (0..count)
            .map(|_| db.create_node(&["Item"]).unwrap())
            .collect::<Vec<_>>();
        db.create_vector_index("Item", "embedding", Some(2), None, None, None, None)
            .unwrap();
        (db, ids)
    }

    /// The fold reads a file in batches of bounded size: five records of 16
    /// bytes in batches of at most 40 bytes are three batches, and every
    /// value is filled.
    #[test]
    fn the_fold_reads_in_bounded_batches() {
        let dir = tempfile::tempdir().unwrap();
        let (db, ids) = items(5);
        let records: Vec<(u64, Vec<f32>)> = ids
            .iter()
            .map(|id| (id.as_u64(), vec![3.0, 19.0]))
            .collect();
        write_old(
            &dir.path().join("vectors_Item%3Aembedding.bin"),
            2,
            &records,
        );

        let folded = fold_in_batches(db.lpg_store(), &find(dir.path(), true), 40);
        assert_eq!((folded.batches, folded.filled), (3, 5));
        assert_eq!(folded.files.len(), 1);
    }

    /// A read that stops partway keeps what it filled: here the file is cut
    /// short under an open reader, past what its buffer holds.
    #[test]
    fn a_read_that_stops_partway_keeps_what_it_filled() {
        let dir = tempfile::tempdir().unwrap();
        let (db, ids) = items(1988);
        let records: Vec<(u64, Vec<f32>)> = ids
            .iter()
            .map(|id| (id.as_u64(), vec![3.0, 19.0]))
            .collect();
        let path = dir.path().join("vectors_Item%3Aembedding.bin");
        write_old(&path, 2, &records);
        let file = find(dir.path(), true).remove(0);

        let mut old = OldFile::open(&path).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(HEADER_BYTES as u64 + 16 * 319)
            .unwrap();
        let (filled, batches, stopped) = fill_from(db.lpg_store(), &file, &mut old, 16);
        assert!(stopped.is_some(), "the read stops");
        assert!(filled > 0 && filled < 1988, "filled {filled}");
        assert_eq!(filled, batches, "one value a batch");
    }

    /// Only the files of this database's vector indexes are folded; a file
    /// that cannot be read is neither folded nor reported as folded, so it
    /// is never deleted.
    #[test]
    fn folds_the_files_of_its_indexes_and_leaves_the_others() {
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("vectors_Item%3Aembedding.bin");
        let foreign = dir.path().join("vectors_Doc%3Aembedding.bin");
        let bad = dir.path().join("vectors_Item%3Abroken.bin");
        write_old(&good, 2, &[(0, vec![3.0, 19.0])]);
        write_old(&foreign, 2, &[(0, vec![88.0, 3.19])]);
        std::fs::write(&bad, b"not a spill file").unwrap();

        let db = crate::GrafeoDB::new_in_memory();
        let alix = db
            .create_node_with_props(&["Item"], [("name", Value::from("Alix"))])
            .unwrap();
        for property in ["embedding", "broken"] {
            db.create_vector_index("Item", property, Some(2), None, None, None, None)
                .unwrap();
        }
        let store = db.lpg_store();

        let folded = fold_in(store, &find(dir.path(), true));
        let paths: Vec<&PathBuf> = folded.files.iter().map(|file| &file.path).collect();
        assert_eq!(paths, vec![&good]);
        assert_eq!(folded.filled, 1);
        let key = PropertyKey::new("embedding");
        assert_eq!(
            store.get_node_property(alix, &key),
            Some(Value::Vector(vec![3.0, 19.0].into()))
        );
        let hits = db
            .vector_search("Item", "embedding", &[3.0, 19.0], 1, None, None)
            .unwrap();
        assert_eq!(hits.first().map(|hit| hit.0), Some(alix), "indexed too");
    }
}
