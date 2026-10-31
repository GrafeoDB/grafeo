//! Async spill file read/write abstraction using tokio.

use std::path::{Path, PathBuf};
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufReader, BufWriter, SeekFrom};

/// Buffer size for async spill file I/O (64 KB).
const BUFFER_SIZE: usize = 64 * 1024;

/// Async handle for a single spill file.
///
/// AsyncSpillFile manages a temporary file used for spilling operator state to disk
/// using tokio's async I/O primitives for non-blocking operations.
#[deprecated(
    since = "0.6.1",
    note = "the engine spills through `grafeo_core::execution::spill::SpillFile`, use it instead; removed in 0.7.0 with the `async-storage` feature"
)]
pub struct AsyncSpillFile {
    /// Path to the spill file.
    path: PathBuf,
    /// Buffered writer (Some during write phase, None after finish).
    writer: Option<BufWriter<File>>,
    /// Total bytes written to this file.
    bytes_written: u64,
}

impl AsyncSpillFile {
    /// Creates a new async spill file at the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created.
    pub async fn new(path: PathBuf) -> std::io::Result<Self> {
        let file = File::create(&path).await?;
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

    /// Writes raw bytes to the file asynchronously.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    pub async fn write_all(&mut self, data: &[u8]) -> std::io::Result<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| std::io::Error::other("Write phase ended"))?;

        writer.write_all(data).await?;
        self.bytes_written += data.len() as u64;
        Ok(())
    }

    /// Writes a u64 in little-endian format.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    pub async fn write_u64_le(&mut self, value: u64) -> std::io::Result<()> {
        self.write_all(&value.to_le_bytes()).await
    }

    /// Writes an i64 in little-endian format.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    pub async fn write_i64_le(&mut self, value: i64) -> std::io::Result<()> {
        self.write_all(&value.to_le_bytes()).await
    }

    /// Writes a length-prefixed byte slice.
    ///
    /// Format: [length: u64][data: bytes]
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    pub async fn write_bytes(&mut self, data: &[u8]) -> std::io::Result<()> {
        self.write_u64_le(data.len() as u64).await?;
        self.write_all(data).await
    }

    /// Finishes writing and flushes buffers.
    ///
    /// After this call, the file is ready for reading.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush fails.
    pub async fn finish_write(&mut self) -> std::io::Result<()> {
        if let Some(mut writer) = self.writer.take() {
            writer.flush().await?;
        }
        Ok(())
    }

    /// Returns whether this file is still in write mode.
    #[must_use]
    pub fn is_writable(&self) -> bool {
        self.writer.is_some()
    }

    /// Creates an async reader for this file.
    ///
    /// Can be called multiple times to create multiple readers.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened for reading.
    pub async fn reader(&self) -> std::io::Result<AsyncSpillFileReader> {
        let file = File::open(&self.path).await?;
        let reader = BufReader::with_capacity(BUFFER_SIZE, file);
        Ok(AsyncSpillFileReader { reader })
    }

    /// Deletes this spill file.
    ///
    /// Consumes the AsyncSpillFile handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be deleted.
    pub async fn delete(mut self) -> std::io::Result<()> {
        // Close the writer first
        self.writer = None;
        tokio::fs::remove_file(&self.path).await
    }
}

impl std::fmt::Debug for AsyncSpillFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncSpillFile")
            .field("path", &self.path)
            .field("bytes_written", &self.bytes_written)
            .field("is_writable", &self.is_writable())
            .finish()
    }
}

/// Async reader for a spill file.
///
/// Provides buffered async reading of spill file contents.
#[deprecated(
    since = "0.6.1",
    note = "the engine spills through `grafeo_core::execution::spill::SpillFileReader`, use it instead; removed in 0.7.0 with the `async-storage` feature"
)]
pub struct AsyncSpillFileReader {
    /// Buffered reader.
    reader: BufReader<File>,
}

impl AsyncSpillFileReader {
    /// Reads exactly `buf.len()` bytes from the file.
    ///
    /// # Errors
    ///
    /// Returns an error if not enough bytes are available.
    pub async fn read_exact(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        self.reader.read_exact(buf).await?;
        Ok(())
    }

    /// Reads a u64 in little-endian format.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    pub async fn read_u64_le(&mut self) -> std::io::Result<u64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf).await?;
        Ok(u64::from_le_bytes(buf))
    }

    /// Reads an i64 in little-endian format.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    pub async fn read_i64_le(&mut self) -> std::io::Result<i64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf).await?;
        Ok(i64::from_le_bytes(buf))
    }

    /// Reads a f64 in little-endian format.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    pub async fn read_f64_le(&mut self) -> std::io::Result<f64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf).await?;
        Ok(f64::from_le_bytes(buf))
    }

    /// Reads a u8 byte.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    pub async fn read_u8(&mut self) -> std::io::Result<u8> {
        let mut buf = [0u8; 1];
        self.read_exact(&mut buf).await?;
        Ok(buf[0])
    }

    /// Reads a length-prefixed byte slice.
    ///
    /// Format: [length: u64][data: bytes]
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    pub async fn read_bytes(&mut self) -> std::io::Result<Vec<u8>> {
        // reason: spill buffer lengths fit usize on 64-bit targets
        #[allow(clippy::cast_possible_truncation)]
        let len = self.read_u64_le().await? as usize;
        let mut buf = vec![0u8; len];
        self.read_exact(&mut buf).await?;
        Ok(buf)
    }

    /// Seeks to a position in the file.
    ///
    /// # Errors
    ///
    /// Returns an error if the seek fails.
    pub async fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.reader.seek(pos).await
    }

    /// Seeks to the beginning of the file.
    ///
    /// # Errors
    ///
    /// Returns an error if the seek fails.
    pub async fn rewind(&mut self) -> std::io::Result<()> {
        self.reader.seek(SeekFrom::Start(0)).await?;
        Ok(())
    }

    /// Returns the current position in the file.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub async fn position(&mut self) -> std::io::Result<u64> {
        self.reader.stream_position().await
    }
}

impl std::fmt::Debug for AsyncSpillFileReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncSpillFileReader").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_async_spill_file_write_read() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.spill");

        // Write phase
        let mut file = AsyncSpillFile::new(file_path).await.unwrap();
        file.write_all(b"hello ").await.unwrap();
        file.write_all(b"world").await.unwrap();
        assert_eq!(file.bytes_written(), 11);
        file.finish_write().await.unwrap();

        // Read phase
        let mut reader = file.reader().await.unwrap();
        let mut buf = [0u8; 11];
        reader.read_exact(&mut buf).await.unwrap();
        assert_eq!(&buf, b"hello world");
    }

    #[tokio::test]
    async fn test_async_spill_file_integers() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.spill");

        let mut file = AsyncSpillFile::new(file_path).await.unwrap();
        file.write_u64_le(u64::MAX).await.unwrap();
        file.write_i64_le(i64::MIN).await.unwrap();
        file.finish_write().await.unwrap();

        let mut reader = file.reader().await.unwrap();
        assert_eq!(reader.read_u64_le().await.unwrap(), u64::MAX);
        assert_eq!(reader.read_i64_le().await.unwrap(), i64::MIN);
    }

    #[tokio::test]
    async fn test_async_spill_file_bytes_prefixed() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.spill");

        let mut file = AsyncSpillFile::new(file_path).await.unwrap();
        file.write_bytes(b"short").await.unwrap();
        file.write_bytes(b"longer string here").await.unwrap();
        file.finish_write().await.unwrap();

        let mut reader = file.reader().await.unwrap();
        assert_eq!(reader.read_bytes().await.unwrap(), b"short");
        assert_eq!(reader.read_bytes().await.unwrap(), b"longer string here");
    }

    #[tokio::test]
    async fn test_async_spill_file_multiple_readers() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.spill");

        let mut file = AsyncSpillFile::new(file_path).await.unwrap();
        file.write_u64_le(42).await.unwrap();
        file.write_u64_le(100).await.unwrap();
        file.finish_write().await.unwrap();

        // Create multiple readers
        let mut reader1 = file.reader().await.unwrap();
        let mut reader2 = file.reader().await.unwrap();

        // Read from reader1
        assert_eq!(reader1.read_u64_le().await.unwrap(), 42);

        // reader2 still at beginning
        assert_eq!(reader2.read_u64_le().await.unwrap(), 42);
        assert_eq!(reader2.read_u64_le().await.unwrap(), 100);

        // reader1 continues
        assert_eq!(reader1.read_u64_le().await.unwrap(), 100);
    }

    #[tokio::test]
    async fn test_async_spill_file_delete() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.spill");
        let file_path_clone = file_path.clone();

        let mut file = AsyncSpillFile::new(file_path).await.unwrap();
        file.write_all(b"data").await.unwrap();
        file.finish_write().await.unwrap();

        assert!(file_path_clone.exists());
        file.delete().await.unwrap();
        assert!(!file_path_clone.exists());
    }

    #[tokio::test]
    async fn test_async_reader_seek() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.spill");

        let mut file = AsyncSpillFile::new(file_path).await.unwrap();
        file.write_u64_le(1).await.unwrap();
        file.write_u64_le(2).await.unwrap();
        file.write_u64_le(3).await.unwrap();
        file.finish_write().await.unwrap();

        let mut reader = file.reader().await.unwrap();

        // Read second value directly
        reader.seek(SeekFrom::Start(8)).await.unwrap();
        assert_eq!(reader.read_u64_le().await.unwrap(), 2);

        // Rewind and read from beginning
        reader.rewind().await.unwrap();
        assert_eq!(reader.read_u64_le().await.unwrap(), 1);
    }
}
