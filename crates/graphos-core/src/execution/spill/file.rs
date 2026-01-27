//! Spill file read/write abstraction.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Buffer size for spill file I/O (64 KB).
const BUFFER_SIZE: usize = 64 * 1024;

/// Handle for a single spill file.
///
/// SpillFile manages a temporary file used for spilling operator state to disk.
/// It supports:
/// - Buffered writing for efficiency
/// - Multiple readers for concurrent access
/// - Automatic byte counting
pub struct SpillFile {
    /// Path to the spill file.
    path: PathBuf,
    /// Buffered writer (Some during write phase, None after finish).
    writer: Option<BufWriter<File>>,
    /// Total bytes written to this file.
    bytes_written: u64,
}

impl SpillFile {
    /// Creates a new spill file at the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created.
    pub fn new(path: PathBuf) -> std::io::Result<Self> {
        let file = File::create(&path)?;
        let writer = BufWriter::with_capacity(BUFFER_SIZE, file);

        Ok(Self {
            path,
            writer: Some(writer),
            bytes_written: 0,
        })
    }

    /// Returns the path to this spill file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the number of bytes written to this file.
    #[must_use]
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Writes raw bytes to the file.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    pub fn write_all(&mut self, data: &[u8]) -> std::io::Result<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "Write phase ended"))?;

        writer.write_all(data)?;
        self.bytes_written += data.len() as u64;
        Ok(())
    }

    /// Writes a u64 in little-endian format.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    pub fn write_u64_le(&mut self, value: u64) -> std::io::Result<()> {
        self.write_all(&value.to_le_bytes())
    }

    /// Writes an i64 in little-endian format.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    pub fn write_i64_le(&mut self, value: i64) -> std::io::Result<()> {
        self.write_all(&value.to_le_bytes())
    }

    /// Writes a length-prefixed byte slice.
    ///
    /// Format: [length: u64][data: bytes]
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    pub fn write_bytes(&mut self, data: &[u8]) -> std::io::Result<()> {
        self.write_u64_le(data.len() as u64)?;
        self.write_all(data)
    }

    /// Finishes writing and flushes buffers.
    ///
    /// After this call, the file is ready for reading.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush fails.
    pub fn finish_write(&mut self) -> std::io::Result<()> {
        if let Some(mut writer) = self.writer.take() {
            writer.flush()?;
        }
        Ok(())
    }

    /// Returns whether this file is still in write mode.
    #[must_use]
    pub fn is_writable(&self) -> bool {
        self.writer.is_some()
    }

    /// Creates a reader for this file.
    ///
    /// Can be called multiple times to create multiple readers.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened for reading.
    pub fn reader(&self) -> std::io::Result<SpillFileReader> {
        let file = File::open(&self.path)?;
        let reader = BufReader::with_capacity(BUFFER_SIZE, file);
        Ok(SpillFileReader { reader })
    }

    /// Deletes this spill file.
    ///
    /// Consumes the SpillFile handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be deleted.
    pub fn delete(mut self) -> std::io::Result<()> {
        // Close the writer first
        self.writer = None;
        std::fs::remove_file(&self.path)
    }
}

impl std::fmt::Debug for SpillFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpillFile")
            .field("path", &self.path)
            .field("bytes_written", &self.bytes_written)
            .field("is_writable", &self.is_writable())
            .finish()
    }
}

/// Reader for a spill file.
///
/// Provides buffered reading of spill file contents.
pub struct SpillFileReader {
    /// Buffered reader.
    reader: BufReader<File>,
}

impl SpillFileReader {
    /// Reads exactly `buf.len()` bytes from the file.
    ///
    /// # Errors
    ///
    /// Returns an error if not enough bytes are available.
    pub fn read_exact(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        self.reader.read_exact(buf)
    }

    /// Reads a u64 in little-endian format.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    pub fn read_u64_le(&mut self) -> std::io::Result<u64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    /// Reads an i64 in little-endian format.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    pub fn read_i64_le(&mut self) -> std::io::Result<i64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }

    /// Reads a f64 in little-endian format.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    pub fn read_f64_le(&mut self) -> std::io::Result<f64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }

    /// Reads a u8 byte.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    pub fn read_u8(&mut self) -> std::io::Result<u8> {
        let mut buf = [0u8; 1];
        self.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    /// Reads a length-prefixed byte slice.
    ///
    /// Format: [length: u64][data: bytes]
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    pub fn read_bytes(&mut self) -> std::io::Result<Vec<u8>> {
        let len = self.read_u64_le()? as usize;
        let mut buf = vec![0u8; len];
        self.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Seeks to a position in the file.
    ///
    /// # Errors
    ///
    /// Returns an error if the seek fails.
    pub fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.reader.seek(pos)
    }

    /// Seeks to the beginning of the file.
    ///
    /// # Errors
    ///
    /// Returns an error if the seek fails.
    pub fn rewind(&mut self) -> std::io::Result<()> {
        self.reader.seek(SeekFrom::Start(0))?;
        Ok(())
    }

    /// Returns the current position in the file.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn position(&mut self) -> std::io::Result<u64> {
        self.reader.stream_position()
    }
}

impl std::fmt::Debug for SpillFileReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpillFileReader").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_spill_file_write_read() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.spill");

        // Write phase
        let mut file = SpillFile::new(file_path).unwrap();
        file.write_all(b"hello ").unwrap();
        file.write_all(b"world").unwrap();
        assert_eq!(file.bytes_written(), 11);
        file.finish_write().unwrap();

        // Read phase
        let mut reader = file.reader().unwrap();
        let mut buf = [0u8; 11];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"hello world");
    }

    #[test]
    fn test_spill_file_integers() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.spill");

        let mut file = SpillFile::new(file_path).unwrap();
        file.write_u64_le(u64::MAX).unwrap();
        file.write_i64_le(i64::MIN).unwrap();
        file.finish_write().unwrap();

        let mut reader = file.reader().unwrap();
        assert_eq!(reader.read_u64_le().unwrap(), u64::MAX);
        assert_eq!(reader.read_i64_le().unwrap(), i64::MIN);
    }

    #[test]
    fn test_spill_file_bytes_prefixed() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.spill");

        let mut file = SpillFile::new(file_path).unwrap();
        file.write_bytes(b"short").unwrap();
        file.write_bytes(b"longer string here").unwrap();
        file.finish_write().unwrap();

        let mut reader = file.reader().unwrap();
        assert_eq!(reader.read_bytes().unwrap(), b"short");
        assert_eq!(reader.read_bytes().unwrap(), b"longer string here");
    }

    #[test]
    fn test_spill_file_multiple_readers() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.spill");

        let mut file = SpillFile::new(file_path).unwrap();
        file.write_u64_le(42).unwrap();
        file.write_u64_le(100).unwrap();
        file.finish_write().unwrap();

        // Create multiple readers
        let mut reader1 = file.reader().unwrap();
        let mut reader2 = file.reader().unwrap();

        // Read from reader1
        assert_eq!(reader1.read_u64_le().unwrap(), 42);

        // reader2 still at beginning
        assert_eq!(reader2.read_u64_le().unwrap(), 42);
        assert_eq!(reader2.read_u64_le().unwrap(), 100);

        // reader1 continues
        assert_eq!(reader1.read_u64_le().unwrap(), 100);
    }

    #[test]
    fn test_spill_file_delete() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.spill");

        let mut file = SpillFile::new(file_path.clone()).unwrap();
        file.write_all(b"data").unwrap();
        file.finish_write().unwrap();

        assert!(file_path.exists());
        file.delete().unwrap();
        assert!(!file_path.exists());
    }

    #[test]
    fn test_reader_seek() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.spill");

        let mut file = SpillFile::new(file_path).unwrap();
        file.write_u64_le(1).unwrap();
        file.write_u64_le(2).unwrap();
        file.write_u64_le(3).unwrap();
        file.finish_write().unwrap();

        let mut reader = file.reader().unwrap();

        // Read second value directly
        reader.seek(SeekFrom::Start(8)).unwrap();
        assert_eq!(reader.read_u64_le().unwrap(), 2);

        // Rewind and read from beginning
        reader.rewind().unwrap();
        assert_eq!(reader.read_u64_le().unwrap(), 1);
    }
}
