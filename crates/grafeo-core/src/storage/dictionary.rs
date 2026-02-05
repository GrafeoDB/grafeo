//! Dictionary encoding for repeated strings.
//!
//! If your data has lots of repeated strings (like node labels or edge types),
//! dictionary encoding stores each unique string once and references it by a
//! small integer code. A million "Person" labels becomes one string + a million
//! 4-byte codes instead of a million strings.
//!
//! # Example
//!
//! ```ignore
//! let mut builder = DictionaryBuilder::new();
//! builder.add("Person");
//! builder.add("Company");
//! builder.add("Person");  // same as first - reuses code 0
//! builder.add("Person");  // reuses code 0 again
//!
//! let dict = builder.build();
//! // Dictionary: ["Person", "Company"]
//! // Codes:      [0, 1, 0, 0]
//! assert_eq!(dict.dictionary_size(), 2);  // Only 2 unique strings stored
//! ```

use std::collections::HashMap;
use std::sync::Arc;

/// Stores repeated strings efficiently by referencing them with integer codes.
///
/// Each unique string appears once in the dictionary. Values are stored as u32
/// indices pointing into that dictionary. Great for labels, categories, and
/// other low-cardinality string columns.
#[derive(Debug, Clone)]
pub struct DictionaryEncoding {
    /// The dictionary of unique strings.
    dictionary: Arc<[Arc<str>]>,
    /// Encoded values as indices into the dictionary.
    codes: Vec<u32>,
    /// Null bitmap (bit set = null).
    null_bitmap: Option<Vec<u64>>,
}

impl DictionaryEncoding {
    /// Creates a new dictionary encoding from a dictionary and codes.
    pub fn new(dictionary: Arc<[Arc<str>]>, codes: Vec<u32>) -> Self {
        Self {
            dictionary,
            codes,
            null_bitmap: None,
        }
    }

    /// Creates a dictionary encoding with a null bitmap.
    pub fn with_nulls(mut self, null_bitmap: Vec<u64>) -> Self {
        self.null_bitmap = Some(null_bitmap);
        self
    }

    /// Returns the number of values.
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Returns whether the encoding is empty.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Returns the number of unique strings in the dictionary.
    pub fn dictionary_size(&self) -> usize {
        self.dictionary.len()
    }

    /// Returns the dictionary.
    pub fn dictionary(&self) -> &Arc<[Arc<str>]> {
        &self.dictionary
    }

    /// Returns the encoded values.
    pub fn codes(&self) -> &[u32] {
        &self.codes
    }

    /// Returns whether the value at index is null.
    pub fn is_null(&self, index: usize) -> bool {
        if let Some(bitmap) = &self.null_bitmap {
            let word_idx = index / 64;
            let bit_idx = index % 64;
            if word_idx < bitmap.len() {
                return (bitmap[word_idx] & (1 << bit_idx)) != 0;
            }
        }
        false
    }

    /// Returns the string value at the given index.
    ///
    /// Returns `None` if the value is null.
    pub fn get(&self, index: usize) -> Option<&str> {
        if self.is_null(index) {
            return None;
        }
        let code = self.codes.get(index)? as &u32;
        self.dictionary.get(*code as usize).map(|s| s.as_ref())
    }

    /// Returns the code at the given index.
    pub fn get_code(&self, index: usize) -> Option<u32> {
        if self.is_null(index) {
            return None;
        }
        self.codes.get(index).copied()
    }

    /// Iterates over all values, yielding `Option<&str>`.
    pub fn iter(&self) -> impl Iterator<Item = Option<&str>> {
        (0..self.len()).map(move |i| self.get(i))
    }

    /// Returns the compression ratio (original size / compressed size).
    ///
    /// A ratio > 1.0 means compression is effective.
    pub fn compression_ratio(&self) -> f64 {
        if self.codes.is_empty() {
            return 1.0;
        }

        // Estimate original size: sum of string lengths
        let original_size: usize = self
            .codes
            .iter()
            .map(|&code| {
                if (code as usize) < self.dictionary.len() {
                    self.dictionary[code as usize].len()
                } else {
                    0
                }
            })
            .sum();

        // Compressed size: dictionary + codes
        let dict_size: usize = self.dictionary.iter().map(|s| s.len()).sum();
        let codes_size = self.codes.len() * std::mem::size_of::<u32>();
        let compressed_size = dict_size + codes_size;

        if compressed_size == 0 {
            return 1.0;
        }

        original_size as f64 / compressed_size as f64
    }

    /// Encodes a lookup value into a code, if it exists in the dictionary.
    pub fn encode(&self, value: &str) -> Option<u32> {
        self.dictionary
            .iter()
            .position(|s| s.as_ref() == value)
            .map(|i| i as u32)
    }

    /// Filters the encoding to only include rows matching a predicate code.
    pub fn filter_by_code(&self, predicate: impl Fn(u32) -> bool) -> Vec<usize> {
        self.codes
            .iter()
            .enumerate()
            .filter_map(|(i, &code)| {
                if !self.is_null(i) && predicate(code) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Builds a dictionary encoding by streaming values through.
///
/// Call [`add()`](Self::add) for each value - we'll automatically assign codes
/// and build the dictionary. Then [`build()`](Self::build) to get the final encoding.
#[derive(Debug)]
pub struct DictionaryBuilder {
    /// Map from string to code.
    string_to_code: HashMap<Arc<str>, u32>,
    /// Dictionary (code -> string).
    dictionary: Vec<Arc<str>>,
    /// Encoded values.
    codes: Vec<u32>,
    /// Null positions (for marking nulls).
    null_positions: Vec<usize>,
}

impl DictionaryBuilder {
    /// Creates a new dictionary builder.
    pub fn new() -> Self {
        Self {
            string_to_code: HashMap::new(),
            dictionary: Vec::new(),
            codes: Vec::new(),
            null_positions: Vec::new(),
        }
    }

    /// Creates a new dictionary builder with estimated capacity.
    pub fn with_capacity(value_capacity: usize, dictionary_capacity: usize) -> Self {
        Self {
            string_to_code: HashMap::with_capacity(dictionary_capacity),
            dictionary: Vec::with_capacity(dictionary_capacity),
            codes: Vec::with_capacity(value_capacity),
            null_positions: Vec::new(),
        }
    }

    /// Adds a string value to the encoding.
    ///
    /// Returns the code assigned to this value.
    pub fn add(&mut self, value: &str) -> u32 {
        if let Some(&code) = self.string_to_code.get(value) {
            self.codes.push(code);
            code
        } else {
            let code = self.dictionary.len() as u32;
            let arc_value: Arc<str> = value.into();
            self.string_to_code.insert(arc_value.clone(), code);
            self.dictionary.push(arc_value);
            self.codes.push(code);
            code
        }
    }

    /// Adds a null value.
    pub fn add_null(&mut self) {
        let idx = self.codes.len();
        self.null_positions.push(idx);
        self.codes.push(0); // Placeholder code
    }

    /// Adds an optional value.
    pub fn add_optional(&mut self, value: Option<&str>) -> Option<u32> {
        match value {
            Some(v) => Some(self.add(v)),
            None => {
                self.add_null();
                None
            }
        }
    }

    /// Returns the current number of values.
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Returns whether the builder is empty.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Returns the current dictionary size.
    pub fn dictionary_size(&self) -> usize {
        self.dictionary.len()
    }

    /// Builds the dictionary encoding.
    pub fn build(self) -> DictionaryEncoding {
        let null_bitmap = if self.null_positions.is_empty() {
            None
        } else {
            let num_words = (self.codes.len() + 63) / 64;
            let mut bitmap = vec![0u64; num_words];
            for &pos in &self.null_positions {
                let word_idx = pos / 64;
                let bit_idx = pos % 64;
                bitmap[word_idx] |= 1 << bit_idx;
            }
            Some(bitmap)
        };

        let dict: Arc<[Arc<str>]> = self.dictionary.into();

        let mut encoding = DictionaryEncoding::new(dict, self.codes);
        if let Some(bitmap) = null_bitmap {
            encoding = encoding.with_nulls(bitmap);
        }
        encoding
    }

    /// Clears the builder for reuse.
    pub fn clear(&mut self) {
        self.string_to_code.clear();
        self.dictionary.clear();
        self.codes.clear();
        self.null_positions.clear();
    }
}

impl Default for DictionaryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for building dictionary encodings from iterators.
pub trait IntoDictionaryEncoding {
    /// Creates a dictionary encoding from an iterator of strings.
    fn into_dictionary_encoding(self) -> DictionaryEncoding;
}

impl<'a, I> IntoDictionaryEncoding for I
where
    I: IntoIterator<Item = &'a str>,
{
    fn into_dictionary_encoding(self) -> DictionaryEncoding {
        let mut builder = DictionaryBuilder::new();
        for s in self {
            builder.add(s);
        }
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_builder_basic() {
        let mut builder = DictionaryBuilder::new();
        builder.add("apple");
        builder.add("banana");
        builder.add("apple");
        builder.add("cherry");
        builder.add("apple");

        let dict = builder.build();

        assert_eq!(dict.len(), 5);
        assert_eq!(dict.dictionary_size(), 3);

        assert_eq!(dict.get(0), Some("apple"));
        assert_eq!(dict.get(1), Some("banana"));
        assert_eq!(dict.get(2), Some("apple"));
        assert_eq!(dict.get(3), Some("cherry"));
        assert_eq!(dict.get(4), Some("apple"));
    }

    #[test]
    fn test_dictionary_codes() {
        let mut builder = DictionaryBuilder::new();
        let code_apple = builder.add("apple");
        let code_banana = builder.add("banana");
        let code_apple2 = builder.add("apple");

        assert_eq!(code_apple, code_apple2);
        assert_ne!(code_apple, code_banana);

        let dict = builder.build();
        assert_eq!(dict.codes(), &[0, 1, 0]);
    }

    #[test]
    fn test_dictionary_with_nulls() {
        let mut builder = DictionaryBuilder::new();
        builder.add("apple");
        builder.add_null();
        builder.add("banana");
        builder.add_null();

        let dict = builder.build();

        assert_eq!(dict.len(), 4);
        assert_eq!(dict.get(0), Some("apple"));
        assert_eq!(dict.get(1), None);
        assert!(dict.is_null(1));
        assert_eq!(dict.get(2), Some("banana"));
        assert_eq!(dict.get(3), None);
        assert!(dict.is_null(3));
    }

    #[test]
    fn test_dictionary_encode_lookup() {
        let mut builder = DictionaryBuilder::new();
        builder.add("apple");
        builder.add("banana");
        builder.add("cherry");

        let dict = builder.build();

        assert_eq!(dict.encode("apple"), Some(0));
        assert_eq!(dict.encode("banana"), Some(1));
        assert_eq!(dict.encode("cherry"), Some(2));
        assert_eq!(dict.encode("date"), None);
    }

    #[test]
    fn test_dictionary_filter_by_code() {
        let mut builder = DictionaryBuilder::new();
        builder.add("apple");
        builder.add("banana");
        builder.add("apple");
        builder.add("cherry");
        builder.add("apple");

        let dict = builder.build();
        let apple_code = dict.encode("apple").unwrap();

        let indices = dict.filter_by_code(|code| code == apple_code);
        assert_eq!(indices, vec![0, 2, 4]);
    }

    #[test]
    fn test_compression_ratio() {
        let mut builder = DictionaryBuilder::new();

        // Add many repeated long strings
        for _ in 0..100 {
            builder.add("this_is_a_very_long_string_that_repeats_many_times");
        }

        let dict = builder.build();

        // Compression ratio should be > 1 for highly repetitive data
        let ratio = dict.compression_ratio();
        assert!(ratio > 1.0, "Expected compression ratio > 1, got {}", ratio);
    }

    #[test]
    fn test_into_dictionary_encoding() {
        let strings = vec!["apple", "banana", "apple", "cherry"];
        let dict: DictionaryEncoding = strings.into_iter().into_dictionary_encoding();

        assert_eq!(dict.len(), 4);
        assert_eq!(dict.dictionary_size(), 3);
    }

    #[test]
    fn test_empty_dictionary() {
        let builder = DictionaryBuilder::new();
        let dict = builder.build();

        assert!(dict.is_empty());
        assert_eq!(dict.dictionary_size(), 0);
        assert_eq!(dict.get(0), None);
    }

    #[test]
    fn test_single_value() {
        let mut builder = DictionaryBuilder::new();
        builder.add("only_value");

        let dict = builder.build();

        assert_eq!(dict.len(), 1);
        assert_eq!(dict.dictionary_size(), 1);
        assert_eq!(dict.get(0), Some("only_value"));
    }

    #[test]
    fn test_all_unique() {
        let mut builder = DictionaryBuilder::new();
        builder.add("a");
        builder.add("b");
        builder.add("c");
        builder.add("d");

        let dict = builder.build();

        assert_eq!(dict.len(), 4);
        assert_eq!(dict.dictionary_size(), 4);
        assert_eq!(dict.codes(), &[0, 1, 2, 3]);
    }

    #[test]
    fn test_all_same() {
        let mut builder = DictionaryBuilder::new();
        for _ in 0..10 {
            builder.add("same");
        }

        let dict = builder.build();

        assert_eq!(dict.len(), 10);
        assert_eq!(dict.dictionary_size(), 1);
        assert!(dict.codes().iter().all(|&c| c == 0));
    }
}
