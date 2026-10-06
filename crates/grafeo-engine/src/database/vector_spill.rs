//! The cache file of a spilled vector column (#594).
//!
//! Spilling a vector property column writes its vectors into a cache file
//! and hands the column a [`VectorSpillFile`] to read through (see
//! [`ColumnBacking`]). The file is a cache, never data: the values stay part
//! of the column, which checkpoints, copies and queries read through it. A
//! file lives as long as its backing (until the column is reloaded or the
//! database is dropped), and only the process that wrote it reads it, so it
//! holds its numbers in native byte order and search reads the vectors in
//! place, without copying them.
//!
//! Layout, every section 8-byte aligned:
//!
//! | Bytes | Content |
//! | --- | --- |
//! | 8 | magic `GRFVCAC1` |
//! | 8 | `count`: the number of vectors |
//! | 8 | `floats`: the number of `f32` values in all vectors |
//! | 8 | reserved |
//! | `8 * count` | the node ids, ascending |
//! | `8 * (count + 1)` | where each vector starts in the values (and where the last ends) |
//! | `4 * floats` | the values |

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use grafeo_common::types::{NodeId, Value};
use grafeo_core::graph::lpg::ColumnBacking;
use memmap2::Mmap;

use super::spill_directory::SpillDirectory;

const MAGIC: [u8; 8] = *b"GRFVCAC1";
const HEADER_BYTES: usize = 32;

/// A spilled vector column: its vectors in a cache file, mapped in place.
pub(crate) struct VectorSpillFile {
    path: PathBuf,
    /// `None` only while dropping (the mapping goes before the file).
    map: Option<Mmap>,
    count: usize,
    floats: usize,
    /// Keeps the directory until its last file is gone.
    _directory: Arc<SpillDirectory>,
}

impl VectorSpillFile {
    /// Writes `vectors` (in ascending id order, each id once) into a new file
    /// in `directory`, creating the directory if needed, and maps it.
    ///
    /// # Errors
    ///
    /// Returns the error of creating, writing or mapping the file; nothing is
    /// left behind then.
    pub(crate) fn write(
        directory: &Arc<SpillDirectory>,
        vectors: &[(NodeId, Arc<[f32]>)],
    ) -> io::Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        directory.create()?;
        // A name no other file ever had: the process id keeps processes
        // apart, the counter the spills of this one, and the random part the
        // rest. So a search still reading a reloaded column's file never
        // meets a new spill, and no open deletes another one's file.
        let path = directory.path().join(format!(
            "vectors_{}_{}_{:016x}.bin",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
            super::spill_directory::random_u64()
        ));
        let map = write_new(&path, vectors)?;
        let floats = vectors.iter().map(|(_, vector)| vector.len()).sum();
        let spill = Self {
            path,
            map: Some(map),
            count: vectors.len(),
            floats,
            _directory: Arc::clone(directory),
        };
        spill.check()?;
        Ok(spill)
    }

    /// Checks that the mapping holds what [`write`](Self::write) wrote, so
    /// the readers below can index it without checks.
    fn check(&self) -> io::Result<()> {
        let invalid = |what: &str| {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("vector spill file {}: {what}", self.path.display()),
            ))
        };
        let bytes = self.bytes();
        if bytes.len() != HEADER_BYTES + 8 * self.count + 8 * (self.count + 1) + 4 * self.floats
            || bytes[..8] != MAGIC
        {
            return invalid("unexpected length or magic");
        }
        let (Some(ids), Some(offsets), Some(_)) = (
            view::<u64>(self.id_bytes()),
            view::<u64>(self.offset_bytes()),
            view::<f32>(self.value_bytes()),
        ) else {
            return invalid("misaligned");
        };
        if !ids.windows(2).all(|pair| pair[0] < pair[1])
            || offsets.first() != Some(&0)
            || !offsets.windows(2).all(|pair| pair[0] <= pair[1])
            || usize::try_from(offsets[self.count]).ok() != Some(self.floats)
        {
            return invalid("ids or offsets out of order");
        }
        Ok(())
    }

    fn bytes(&self) -> &[u8] {
        self.map.as_deref().unwrap_or_default()
    }

    fn id_bytes(&self) -> &[u8] {
        &self.bytes()[HEADER_BYTES..HEADER_BYTES + 8 * self.count]
    }

    fn offset_bytes(&self) -> &[u8] {
        let start = HEADER_BYTES + 8 * self.count;
        &self.bytes()[start..start + 8 * (self.count + 1)]
    }

    fn value_bytes(&self) -> &[u8] {
        &self.bytes()[HEADER_BYTES + 8 * self.count + 8 * (self.count + 1)..]
    }

    fn ids(&self) -> &[u64] {
        view(self.id_bytes()).unwrap_or_default()
    }

    /// The vector of `id`, read in place.
    fn vector(&self, id: NodeId) -> Option<&[f32]> {
        let index = self.ids().binary_search(&id.as_u64()).ok()?;
        let offsets: &[u64] = view(self.offset_bytes()).unwrap_or_default();
        let values: &[f32] = view(self.value_bytes()).unwrap_or_default();
        let start = usize::try_from(offsets[index]).ok()?;
        let end = usize::try_from(offsets[index + 1]).ok()?;
        values.get(start..end)
    }
}

// Reads cannot fail: they come from a mapping checked when it was written
// (`check`), so there is no read error to report.
impl ColumnBacking<NodeId> for VectorSpillFile {
    fn get(&self, id: NodeId) -> io::Result<Option<Value>> {
        Ok(self.vector(id).map(|vector| Value::Vector(vector.into())))
    }

    fn contains(&self, id: NodeId) -> bool {
        self.ids().binary_search(&id.as_u64()).is_ok()
    }

    fn ids(&self) -> Vec<NodeId> {
        self.ids().iter().map(|&id| NodeId::new(id)).collect()
    }

    fn len(&self) -> usize {
        self.count
    }

    fn heap_bytes(&self) -> usize {
        // The vectors are in the page cache, not on the heap.
        std::mem::size_of::<Self>()
    }

    fn with_vector(&self, id: NodeId, f: &mut dyn FnMut(&[f32])) -> io::Result<bool> {
        Ok(match self.vector(id) {
            Some(vector) => {
                f(vector);
                true
            }
            None => false,
        })
    }
}

impl Drop for VectorSpillFile {
    fn drop(&mut self) {
        // Unmap first: a mapped file cannot be deleted everywhere.
        self.map = None;
        if let Err(error) = std::fs::remove_file(&self.path) {
            grafeo_common::grafeo_warn!(
                "could not remove the vector spill file {}: {error}",
                self.path.display()
            );
        }
    }
}

/// Writes the layout of the module docs to `path`.
/// The file is created new (`create_new`): an existing file, which someone
/// may have mapped, is never truncated.
/// Writes `vectors` into a new file at `path` and maps it. Nothing is left
/// behind on an error, and a file already at `path` (whose name a clash
/// gave twice) is neither written over nor removed: it is not this call's.
fn write_new(path: &std::path::Path, vectors: &[(NodeId, Arc<[f32]>)]) -> io::Result<Mmap> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let written = write_file(file, vectors).and_then(|()| {
        let file = File::open(path)?;
        // SAFETY: the file is this process's own (it is in the open's
        // cache directory, named uniquely, written above and never
        // written again), so nothing changes it while it is mapped; an
        // outside process truncating it is the caveat of every mapping.
        #[allow(
            unsafe_code,
            reason = "mapping a file only this process writes, written before it is mapped"
        )]
        let map = unsafe { Mmap::map(&file) }?;
        Ok(map)
    });
    match written {
        Ok(map) => Ok(map),
        Err(error) => {
            let _ = std::fs::remove_file(path);
            Err(error)
        }
    }
}

/// Writes the layout (see the module docs) into `file`.
fn write_file(file: File, vectors: &[(NodeId, Arc<[f32]>)]) -> io::Result<()> {
    let mut out = BufWriter::new(file);
    let floats: usize = vectors.iter().map(|(_, vector)| vector.len()).sum();
    out.write_all(&MAGIC)?;
    out.write_all(&(vectors.len() as u64).to_ne_bytes())?;
    out.write_all(&(floats as u64).to_ne_bytes())?;
    out.write_all(&0u64.to_ne_bytes())?;
    for (id, _) in vectors {
        out.write_all(&id.as_u64().to_ne_bytes())?;
    }
    let mut offset = 0u64;
    out.write_all(&offset.to_ne_bytes())?;
    for (_, vector) in vectors {
        offset += vector.len() as u64;
        out.write_all(&offset.to_ne_bytes())?;
    }
    for (_, vector) in vectors {
        for value in vector.iter() {
            out.write_all(&value.to_ne_bytes())?;
        }
    }
    out.flush()
}

/// Numbers every bit pattern of which is a valid value.
trait Plain: Copy {}
impl Plain for u64 {}
impl Plain for f32 {}

/// Views `bytes` as a slice of `T`, or `None` when they are not aligned for
/// `T` or not a whole number of them.
fn view<T: Plain>(bytes: &[u8]) -> Option<&[T]> {
    // SAFETY: `T` is `u64` or `f32` (the only `Plain` types), for which
    // every bit pattern is a valid value, and `align_to` hands out only the
    // aligned middle part; a view with anything before or after it is
    // refused.
    #[allow(
        unsafe_code,
        reason = "every bit pattern of the `Plain` types is valid, and only the aligned middle is used"
    )]
    let (before, middle, after) = unsafe { bytes.align_to::<T>() };
    (before.is_empty() && after.is_empty()).then_some(middle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_core::graph::lpg::ColumnBacking;

    fn directory(root: &std::path::Path) -> Arc<SpillDirectory> {
        super::super::spill_directory::SpillLayout::for_open(Some(root), None, false)
            .vector_cache
            .unwrap()
    }

    fn vectors() -> Vec<(NodeId, Arc<[f32]>)> {
        vec![
            (NodeId::new(3), vec![3.0, 19.0].into()),
            (NodeId::new(19), vec![88.0, 3.19, 319.0].into()),
            (NodeId::new(88), Vec::new().into()),
        ]
    }

    /// A cache file is always new: a file already at the name (which another
    /// open may map) is neither truncated nor removed; the write fails and
    /// leaves it as it was (#594).
    #[test]
    fn a_taken_name_is_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors_taken.bin");
        std::fs::write(&path, b"Vincent's").unwrap();
        let error = write_new(&path, &vectors()).map(|_| ()).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&path).unwrap(), b"Vincent's");
    }

    #[test]
    fn reads_back_what_it_wrote() {
        let dir = tempfile::tempdir().unwrap();
        let spill = VectorSpillFile::write(&directory(dir.path()), &vectors()).unwrap();
        assert_eq!(spill.len(), 3);
        assert_eq!(
            ColumnBacking::ids(&spill),
            vec![NodeId::new(3), NodeId::new(19), NodeId::new(88)]
        );
        assert_eq!(
            spill.get(NodeId::new(19)).unwrap(),
            Some(Value::Vector(vec![88.0, 3.19, 319.0].into()))
        );
        assert_eq!(
            spill.get(NodeId::new(88)).unwrap(),
            Some(Value::Vector(Vec::new().into()))
        );
        assert_eq!(spill.get(NodeId::new(319)).unwrap(), None);
        assert!(spill.contains(NodeId::new(3)) && !spill.contains(NodeId::new(1988)));

        let mut seen = Vec::new();
        assert!(
            spill
                .with_vector(NodeId::new(3), &mut |v| seen.extend_from_slice(v))
                .unwrap()
        );
        assert_eq!(seen, vec![3.0, 19.0]);
        assert!(
            !spill
                .with_vector(NodeId::new(319), &mut |_| panic!("no vector"))
                .unwrap()
        );
    }

    /// Once the database closed, a spill writes no file.
    #[test]
    fn a_closed_cache_takes_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let cache = directory(dir.path());
        cache.close_for_writes();
        assert!(VectorSpillFile::write(&cache, &vectors()).is_err());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    /// The file and then its directory go with the last user.
    #[test]
    fn dropping_it_removes_the_file_and_then_the_directory() {
        let dir = tempfile::tempdir().unwrap();
        let cache = directory(dir.path());
        let spill = VectorSpillFile::write(&cache, &vectors()).unwrap();
        let path = spill.path.clone();
        assert!(path.exists());
        drop(cache);
        drop(spill);
        assert!(!path.exists());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    /// Each spill gets its own file, also for the same property.
    #[test]
    fn each_spill_gets_its_own_file() {
        let dir = tempfile::tempdir().unwrap();
        let cache = directory(dir.path());
        let first = VectorSpillFile::write(&cache, &vectors()).unwrap();
        let second = VectorSpillFile::write(&cache, &vectors()).unwrap();
        assert_ne!(first.path, second.path);
        assert_eq!(first.path.parent(), Some(cache.path()));
    }
}
