//! Run-length encoding for highly repetitive data.
//!
//! Run-length encoding (RLE) compresses sequences with consecutive repeated
//! values by storing each unique value once along with its run length.
//!
//! | Data pattern | Compression ratio |
//! | ------------ | ----------------- |
//! | Constant value | ~100x |
//! | Few distinct values, long runs | 10-50x |
//! | Many short runs | 2-5x |
//! | Random data | < 1x (expansion) |
//!
//! # Example
//!
//! ```ignore
//! use grafeo_core::storage::RunLengthEncoding;
//!
//! // Compress data with many repeated values
//! let values = vec![1, 1, 1, 2, 2, 3, 3, 3, 3, 3];
//! let encoded = RunLengthEncoding::encode(&values);
//!
//! println!("Runs: {}", encoded.run_count()); // 3 runs
//! println!("Compression: {:.1}x", encoded.compression_ratio()); // ~3.3x
//!
//! // Decode back to original
//! let decoded = encoded.decode();
//! assert_eq!(values, decoded);
//! ```

use std::io::{self, Read};

/// A run in run-length encoding: a value and how many times it repeats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Run<T> {
    /// The value for this run.
    pub value: T,
    /// Number of consecutive occurrences.
    pub length: u64,
}

impl<T> Run<T> {
    /// Creates a new run.
    #[must_use]
    pub fn new(value: T, length: u64) -> Self {
        Self { value, length }
    }
}

/// Run-length encoded data for u64 values.
///
/// Stores sequences of (value, count) pairs. Achieves excellent compression
/// when data has long runs of repeated values.
#[derive(Debug, Clone)]
pub struct RunLengthEncoding {
    /// The runs: each is (value, count).
    runs: Vec<Run<u64>>,
    /// Total number of values (sum of all run lengths).
    total_count: usize,
}

impl<'a> IntoIterator for &'a RunLengthEncoding {
    type Item = u64;
    type IntoIter = RunLengthIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl RunLengthEncoding {
    /// Encodes a slice of u64 values using run-length encoding.
    ///
    /// # Example
    /// ```ignore
    /// let values = vec![1, 1, 1, 2, 2, 3];
    /// let encoded = RunLengthEncoding::encode(&values);
    /// // Results in 3 runs: (1, 3), (2, 2), (3, 1)
    /// ```
    #[must_use]
    pub fn encode(values: &[u64]) -> Self {
        if values.is_empty() {
            return Self {
                runs: Vec::new(),
                total_count: 0,
            };
        }

        let mut runs = Vec::new();
        let mut current_value = values[0];
        let mut current_length = 1u64;

        for &value in &values[1..] {
            if value == current_value {
                current_length += 1;
            } else {
                runs.push(Run::new(current_value, current_length));
                current_value = value;
                current_length = 1;
            }
        }

        // Don't forget the last run
        runs.push(Run::new(current_value, current_length));

        Self {
            runs,
            total_count: values.len(),
        }
    }

    /// Creates a run-length encoding from pre-built runs.
    #[must_use]
    pub fn from_runs(runs: Vec<Run<u64>>) -> Self {
        let total_count = runs.iter().map(|r| r.length as usize).sum();
        Self { runs, total_count }
    }

    /// Decodes back to the original values.
    #[must_use]
    pub fn decode(&self) -> Vec<u64> {
        let mut values = Vec::with_capacity(self.total_count);

        for run in &self.runs {
            for _ in 0..run.length {
                values.push(run.value);
            }
        }

        values
    }

    /// Returns the number of runs.
    #[must_use]
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    /// Returns the total number of values represented.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.total_count
    }

    /// Returns the runs.
    #[must_use]
    pub fn runs(&self) -> &[Run<u64>] {
        &self.runs
    }

    /// Returns true if there are no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total_count == 0
    }

    /// Returns the compression ratio (original size / encoded size).
    ///
    /// Values > 1.0 indicate compression, < 1.0 indicate expansion.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.runs.is_empty() {
            return 1.0;
        }

        // Original: total_count * 8 bytes
        // Encoded: runs.len() * 16 bytes (8 for value, 8 for length)
        let original_size = self.total_count * 8;
        let encoded_size = self.runs.len() * 16;

        if encoded_size == 0 {
            return 1.0;
        }

        original_size as f64 / encoded_size as f64
    }

    /// Returns true if run-length encoding is beneficial for this data.
    ///
    /// Returns true when compression ratio > 1.0 (actual compression achieved).
    #[must_use]
    pub fn is_beneficial(&self) -> bool {
        self.compression_ratio() > 1.0
    }

    /// Returns the memory size in bytes of the encoded representation.
    #[must_use]
    pub fn encoded_size(&self) -> usize {
        // Each run: 8 bytes value + 8 bytes length
        self.runs.len() * 16
    }

    /// Serializes the run-length encoding to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 + self.runs.len() * 16);

        // Write run count
        bytes.extend_from_slice(&(self.runs.len() as u64).to_le_bytes());

        // Write each run
        for run in &self.runs {
            bytes.extend_from_slice(&run.value.to_le_bytes());
            bytes.extend_from_slice(&run.length.to_le_bytes());
        }

        bytes
    }

    /// Deserializes run-length encoding from bytes.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let mut cursor = io::Cursor::new(bytes);

        // Read run count
        let mut buf = [0u8; 8];
        cursor.read_exact(&mut buf)?;
        let run_count = u64::from_le_bytes(buf) as usize;

        // Read runs
        let mut runs = Vec::with_capacity(run_count);
        for _ in 0..run_count {
            cursor.read_exact(&mut buf)?;
            let value = u64::from_le_bytes(buf);

            cursor.read_exact(&mut buf)?;
            let length = u64::from_le_bytes(buf);

            runs.push(Run::new(value, length));
        }

        Ok(Self::from_runs(runs))
    }

    /// Gets the value at a specific index without full decompression.
    ///
    /// Returns None if index is out of bounds.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.total_count {
            return None;
        }

        let mut offset = 0usize;
        for run in &self.runs {
            let run_end = offset + run.length as usize;
            if index < run_end {
                return Some(run.value);
            }
            offset = run_end;
        }

        None
    }

    /// Returns an iterator over the decoded values.
    pub fn iter(&self) -> RunLengthIterator<'_> {
        RunLengthIterator {
            runs: &self.runs,
            run_index: 0,
            within_run: 0,
        }
    }
}

/// Iterator over run-length encoded values.
pub struct RunLengthIterator<'a> {
    runs: &'a [Run<u64>],
    run_index: usize,
    within_run: u64,
}

impl Iterator for RunLengthIterator<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        while self.run_index < self.runs.len() {
            let run = &self.runs[self.run_index];
            if self.within_run < run.length {
                self.within_run += 1;
                return Some(run.value);
            }
            self.run_index += 1;
            self.within_run = 0;
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining: u64 = self.runs[self.run_index..]
            .iter()
            .map(|r| r.length)
            .sum::<u64>()
            - self.within_run;
        (remaining as usize, Some(remaining as usize))
    }
}

impl ExactSizeIterator for RunLengthIterator<'_> {}

/// Run-length encoding for signed integers.
///
/// Uses zigzag encoding internally for efficient storage.
#[derive(Debug, Clone)]
pub struct SignedRunLengthEncoding {
    inner: RunLengthEncoding,
}

impl SignedRunLengthEncoding {
    /// Encodes signed integers using run-length encoding.
    #[must_use]
    pub fn encode(values: &[i64]) -> Self {
        let unsigned: Vec<u64> = values.iter().map(|&v| zigzag_encode(v)).collect();
        Self {
            inner: RunLengthEncoding::encode(&unsigned),
        }
    }

    /// Decodes back to signed integers.
    #[must_use]
    pub fn decode(&self) -> Vec<i64> {
        self.inner.decode().into_iter().map(zigzag_decode).collect()
    }

    /// Returns the compression ratio.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        self.inner.compression_ratio()
    }

    /// Returns the number of runs.
    #[must_use]
    pub fn run_count(&self) -> usize {
        self.inner.run_count()
    }

    /// Serializes to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.inner.to_bytes()
    }

    /// Deserializes from bytes.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        Ok(Self {
            inner: RunLengthEncoding::from_bytes(bytes)?,
        })
    }
}

/// Zigzag encodes a signed integer to unsigned.
#[inline]
#[must_use]
pub fn zigzag_encode(n: i64) -> u64 {
    ((n << 1) ^ (n >> 63)) as u64
}

/// Zigzag decodes an unsigned integer to signed.
#[inline]
#[must_use]
pub fn zigzag_decode(n: u64) -> i64 {
    ((n >> 1) as i64) ^ -((n & 1) as i64)
}

/// Analyzes data to determine if run-length encoding is beneficial.
pub struct RunLengthAnalyzer;

impl RunLengthAnalyzer {
    /// Estimates the compression ratio without actually encoding.
    ///
    /// This is faster than encoding for decision-making.
    #[must_use]
    pub fn estimate_ratio(values: &[u64]) -> f64 {
        if values.is_empty() {
            return 1.0;
        }

        // Count runs
        let mut run_count = 1usize;
        for i in 1..values.len() {
            if values[i] != values[i - 1] {
                run_count += 1;
            }
        }

        // Original: values.len() * 8 bytes
        // Encoded: run_count * 16 bytes
        let original = values.len() * 8;
        let encoded = run_count * 16;

        if encoded == 0 {
            return 1.0;
        }

        original as f64 / encoded as f64
    }

    /// Returns true if run-length encoding would be beneficial.
    #[must_use]
    pub fn is_beneficial(values: &[u64]) -> bool {
        Self::estimate_ratio(values) > 1.0
    }

    /// Returns the average run length in the data.
    #[must_use]
    pub fn average_run_length(values: &[u64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut run_count = 1usize;
        for i in 1..values.len() {
            if values[i] != values[i - 1] {
                run_count += 1;
            }
        }

        values.len() as f64 / run_count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_basic() {
        let values = vec![1, 1, 1, 2, 2, 3, 3, 3, 3, 3];
        let encoded = RunLengthEncoding::encode(&values);

        assert_eq!(encoded.run_count(), 3);
        assert_eq!(encoded.total_count(), 10);

        let decoded = encoded.decode();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_encode_empty() {
        let values: Vec<u64> = vec![];
        let encoded = RunLengthEncoding::encode(&values);

        assert_eq!(encoded.run_count(), 0);
        assert_eq!(encoded.total_count(), 0);
        assert!(encoded.is_empty());

        let decoded = encoded.decode();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_encode_single() {
        let values = vec![42];
        let encoded = RunLengthEncoding::encode(&values);

        assert_eq!(encoded.run_count(), 1);
        assert_eq!(encoded.total_count(), 1);

        let decoded = encoded.decode();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_encode_all_same() {
        let values = vec![7u64; 1000];
        let encoded = RunLengthEncoding::encode(&values);

        assert_eq!(encoded.run_count(), 1);
        assert_eq!(encoded.total_count(), 1000);

        // Should compress very well
        let ratio = encoded.compression_ratio();
        assert!(ratio > 50.0, "Expected ratio > 50, got {}", ratio);

        let decoded = encoded.decode();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_encode_all_different() {
        let values: Vec<u64> = (0..100).collect();
        let encoded = RunLengthEncoding::encode(&values);

        assert_eq!(encoded.run_count(), 100);
        assert_eq!(encoded.total_count(), 100);

        // Should not compress (expand by 2x)
        let ratio = encoded.compression_ratio();
        assert!(ratio < 1.0, "Expected ratio < 1, got {}", ratio);

        let decoded = encoded.decode();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_compression_ratio() {
        // Perfect case: all same values
        let all_same = vec![1u64; 100];
        let encoded = RunLengthEncoding::encode(&all_same);
        assert!(encoded.compression_ratio() > 1.0);
        assert!(encoded.is_beneficial());

        // Bad case: all different values
        let all_diff: Vec<u64> = (0..100).collect();
        let encoded = RunLengthEncoding::encode(&all_diff);
        assert!(encoded.compression_ratio() < 1.0);
        assert!(!encoded.is_beneficial());
    }

    #[test]
    fn test_serialization() {
        let values = vec![1, 1, 1, 2, 2, 3, 3, 3, 3, 3];
        let encoded = RunLengthEncoding::encode(&values);

        let bytes = encoded.to_bytes();
        let decoded_encoding = RunLengthEncoding::from_bytes(&bytes).unwrap();

        assert_eq!(encoded.run_count(), decoded_encoding.run_count());
        assert_eq!(encoded.decode(), decoded_encoding.decode());
    }

    #[test]
    fn test_get_index() {
        let values = vec![1, 1, 1, 2, 2, 3, 3, 3, 3, 3];
        let encoded = RunLengthEncoding::encode(&values);

        assert_eq!(encoded.get(0), Some(1));
        assert_eq!(encoded.get(2), Some(1));
        assert_eq!(encoded.get(3), Some(2));
        assert_eq!(encoded.get(4), Some(2));
        assert_eq!(encoded.get(5), Some(3));
        assert_eq!(encoded.get(9), Some(3));
        assert_eq!(encoded.get(10), None);
    }

    #[test]
    fn test_iterator() {
        let values = vec![1, 1, 1, 2, 2, 3];
        let encoded = RunLengthEncoding::encode(&values);

        let iterated: Vec<u64> = encoded.iter().collect();
        assert_eq!(values, iterated);
    }

    #[test]
    fn test_signed_integers() {
        let values = vec![-5, -5, -5, 0, 0, 10, 10, 10, 10];
        let encoded = SignedRunLengthEncoding::encode(&values);

        assert_eq!(encoded.run_count(), 3);

        let decoded = encoded.decode();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_signed_serialization() {
        let values = vec![-100, -100, 0, 0, 0, 100];
        let encoded = SignedRunLengthEncoding::encode(&values);

        let bytes = encoded.to_bytes();
        let decoded_encoding = SignedRunLengthEncoding::from_bytes(&bytes).unwrap();

        assert_eq!(encoded.decode(), decoded_encoding.decode());
    }

    #[test]
    fn test_zigzag() {
        assert_eq!(zigzag_encode(0), 0);
        assert_eq!(zigzag_encode(-1), 1);
        assert_eq!(zigzag_encode(1), 2);
        assert_eq!(zigzag_encode(-2), 3);
        assert_eq!(zigzag_encode(2), 4);

        for i in -1000i64..1000 {
            assert_eq!(zigzag_decode(zigzag_encode(i)), i);
        }
    }

    #[test]
    fn test_analyzer_estimate() {
        let all_same = vec![1u64; 100];
        let ratio = RunLengthAnalyzer::estimate_ratio(&all_same);
        assert!(ratio > 1.0);
        assert!(RunLengthAnalyzer::is_beneficial(&all_same));

        let all_diff: Vec<u64> = (0..100).collect();
        let ratio = RunLengthAnalyzer::estimate_ratio(&all_diff);
        assert!(ratio < 1.0);
        assert!(!RunLengthAnalyzer::is_beneficial(&all_diff));
    }

    #[test]
    fn test_analyzer_average_run_length() {
        let values = vec![1, 1, 1, 2, 2, 3, 3, 3, 3, 3]; // 3 runs, 10 values
        let avg = RunLengthAnalyzer::average_run_length(&values);
        assert!((avg - 3.33).abs() < 0.1);

        let all_same = vec![1u64; 100];
        let avg = RunLengthAnalyzer::average_run_length(&all_same);
        assert!((avg - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_from_runs() {
        let runs = vec![Run::new(1, 3), Run::new(2, 2), Run::new(3, 5)];
        let encoded = RunLengthEncoding::from_runs(runs);

        assert_eq!(encoded.run_count(), 3);
        assert_eq!(encoded.total_count(), 10);
        assert_eq!(encoded.decode(), vec![1, 1, 1, 2, 2, 3, 3, 3, 3, 3]);
    }
}
