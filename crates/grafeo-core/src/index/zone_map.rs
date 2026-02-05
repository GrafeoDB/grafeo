//! Zone maps for intelligent data skipping.
//!
//! Each chunk of property data tracks its min, max, and null count. When filtering
//! with a predicate like `age < 30`, we check the zone map first - if the minimum
//! age in a chunk is 50, we skip the entire chunk without reading a single value.
//!
//! This is huge for large scans. Combined with columnar storage, you often skip
//! 90%+ of the data for selective predicates.

use grafeo_common::types::Value;
use std::cmp::Ordering;
use std::collections::HashMap;

/// Statistics for a single chunk of property data.
///
/// The query optimizer uses these to skip chunks that can't match a predicate.
/// For example, if `max < 30` and the predicate is `value > 50`, skip the chunk.
#[derive(Debug, Clone)]
pub struct ZoneMapEntry {
    /// Minimum value in the chunk (None if all nulls).
    pub min: Option<Value>,
    /// Maximum value in the chunk (None if all nulls).
    pub max: Option<Value>,
    /// Number of null values in the chunk.
    pub null_count: u64,
    /// Total number of values in the chunk.
    pub row_count: u64,
    /// Optional Bloom filter for equality checks.
    pub bloom_filter: Option<BloomFilter>,
}

impl ZoneMapEntry {
    /// Creates a new empty zone map entry.
    pub fn new() -> Self {
        Self {
            min: None,
            max: None,
            null_count: 0,
            row_count: 0,
            bloom_filter: None,
        }
    }

    /// Creates a zone map entry with min/max.
    pub fn with_min_max(min: Value, max: Value, null_count: u64, row_count: u64) -> Self {
        Self {
            min: Some(min),
            max: Some(max),
            null_count,
            row_count,
            bloom_filter: None,
        }
    }

    /// Sets the Bloom filter.
    pub fn with_bloom_filter(mut self, filter: BloomFilter) -> Self {
        self.bloom_filter = Some(filter);
        self
    }

    /// Checks if this chunk might contain values matching an equality predicate.
    ///
    /// Returns `true` if the chunk might contain matches, `false` if it definitely doesn't.
    pub fn might_contain_equal(&self, value: &Value) -> bool {
        // If value is null, check if there are nulls
        if matches!(value, Value::Null) {
            return self.null_count > 0;
        }

        // If the chunk is all nulls, it can't contain non-null values
        if self.is_all_null() {
            return false;
        }

        // Check Bloom filter first if available
        if let Some(ref bloom) = self.bloom_filter {
            if !bloom.might_contain(value) {
                return false;
            }
        }

        // Check min/max bounds
        match (&self.min, &self.max) {
            (Some(min), Some(max)) => {
                let cmp_min = compare_values(value, min);
                let cmp_max = compare_values(value, max);

                // Value must be >= min and <= max
                match (cmp_min, cmp_max) {
                    (Some(Ordering::Less), _) => false,    // value < min
                    (_, Some(Ordering::Greater)) => false, // value > max
                    _ => true,
                }
            }
            // No bounds but might have non-null values - can't skip
            _ => self.might_contain_non_null(),
        }
    }

    /// Checks if this chunk might contain values matching a less-than predicate.
    ///
    /// Returns `true` if the chunk might contain matches, `false` if it definitely doesn't.
    pub fn might_contain_less_than(&self, value: &Value, inclusive: bool) -> bool {
        match &self.min {
            Some(min) => {
                let cmp = compare_values(min, value);
                match cmp {
                    Some(Ordering::Less) => true,
                    Some(Ordering::Equal) => inclusive,
                    Some(Ordering::Greater) => false,
                    None => true,
                }
            }
            None => self.null_count > 0, // Only nulls, which don't satisfy < predicate
        }
    }

    /// Checks if this chunk might contain values matching a greater-than predicate.
    ///
    /// Returns `true` if the chunk might contain matches, `false` if it definitely doesn't.
    pub fn might_contain_greater_than(&self, value: &Value, inclusive: bool) -> bool {
        match &self.max {
            Some(max) => {
                let cmp = compare_values(max, value);
                match cmp {
                    Some(Ordering::Greater) => true,
                    Some(Ordering::Equal) => inclusive,
                    Some(Ordering::Less) => false,
                    None => true,
                }
            }
            None => self.null_count > 0,
        }
    }

    /// Checks if this chunk might contain values in a range.
    pub fn might_contain_range(
        &self,
        lower: Option<&Value>,
        upper: Option<&Value>,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> bool {
        // Check lower bound
        if let Some(lower_val) = lower {
            if !self.might_contain_greater_than(lower_val, lower_inclusive) {
                return false;
            }
        }

        // Check upper bound
        if let Some(upper_val) = upper {
            if !self.might_contain_less_than(upper_val, upper_inclusive) {
                return false;
            }
        }

        true
    }

    /// Checks if this chunk might contain non-null values.
    pub fn might_contain_non_null(&self) -> bool {
        self.row_count > self.null_count
    }

    /// Checks if this chunk contains only null values.
    pub fn is_all_null(&self) -> bool {
        self.row_count > 0 && self.null_count == self.row_count
    }

    /// Returns the null fraction.
    pub fn null_fraction(&self) -> f64 {
        if self.row_count == 0 {
            0.0
        } else {
            self.null_count as f64 / self.row_count as f64
        }
    }
}

impl Default for ZoneMapEntry {
    fn default() -> Self {
        Self::new()
    }
}

/// Incrementally builds a zone map entry as you add values.
///
/// Feed it values one at a time; it tracks min/max/nulls automatically.
///
/// By default, Bloom filters are enabled for efficient equality checks.
/// Use [`without_bloom_filter`](Self::without_bloom_filter) if you don't need them.
pub struct ZoneMapBuilder {
    min: Option<Value>,
    max: Option<Value>,
    null_count: u64,
    row_count: u64,
    bloom_builder: Option<BloomFilterBuilder>,
}

/// Default expected items for Bloom filter in a chunk.
const DEFAULT_BLOOM_EXPECTED_ITEMS: usize = 2048;

/// Default false positive rate for Bloom filter (1%).
const DEFAULT_BLOOM_FALSE_POSITIVE_RATE: f64 = 0.01;

impl ZoneMapBuilder {
    /// Creates a new zone map builder with Bloom filter enabled by default.
    ///
    /// Uses reasonable defaults:
    /// - Expected items: 2048 (typical chunk size)
    /// - False positive rate: 1%
    pub fn new() -> Self {
        Self {
            min: None,
            max: None,
            null_count: 0,
            row_count: 0,
            bloom_builder: Some(BloomFilterBuilder::new(
                DEFAULT_BLOOM_EXPECTED_ITEMS,
                DEFAULT_BLOOM_FALSE_POSITIVE_RATE,
            )),
        }
    }

    /// Creates a builder without Bloom filter support.
    ///
    /// Use this when you know you won't need equality checks, or when
    /// memory is constrained and the Bloom filter overhead isn't worth it.
    pub fn without_bloom_filter() -> Self {
        Self {
            min: None,
            max: None,
            null_count: 0,
            row_count: 0,
            bloom_builder: None,
        }
    }

    /// Creates a builder with custom Bloom filter settings.
    ///
    /// # Arguments
    ///
    /// * `expected_items` - Expected number of items in the chunk
    /// * `false_positive_rate` - Desired false positive rate (e.g., 0.01 for 1%)
    pub fn with_bloom_filter(expected_items: usize, false_positive_rate: f64) -> Self {
        Self {
            min: None,
            max: None,
            null_count: 0,
            row_count: 0,
            bloom_builder: Some(BloomFilterBuilder::new(expected_items, false_positive_rate)),
        }
    }

    /// Adds a value to the zone map.
    pub fn add(&mut self, value: &Value) {
        self.row_count += 1;

        if matches!(value, Value::Null) {
            self.null_count += 1;
            return;
        }

        // Update min
        self.min = match &self.min {
            None => Some(value.clone()),
            Some(current) => {
                if compare_values(value, current) == Some(Ordering::Less) {
                    Some(value.clone())
                } else {
                    self.min.clone()
                }
            }
        };

        // Update max
        self.max = match &self.max {
            None => Some(value.clone()),
            Some(current) => {
                if compare_values(value, current) == Some(Ordering::Greater) {
                    Some(value.clone())
                } else {
                    self.max.clone()
                }
            }
        };

        // Add to Bloom filter
        if let Some(ref mut bloom) = self.bloom_builder {
            bloom.add(value);
        }
    }

    /// Builds the zone map entry.
    pub fn build(self) -> ZoneMapEntry {
        let bloom_filter = self.bloom_builder.map(|b| b.build());

        ZoneMapEntry {
            min: self.min,
            max: self.max,
            null_count: self.null_count,
            row_count: self.row_count,
            bloom_filter,
        }
    }
}

impl Default for ZoneMapBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Collection of zone maps for all chunks of a property column.
///
/// Use [`filter_equal`](Self::filter_equal) or [`filter_range`](Self::filter_range)
/// to get chunk IDs that might contain matching values.
pub struct ZoneMapIndex {
    /// Zone map entries per chunk.
    entries: HashMap<u64, ZoneMapEntry>,
    /// Property name.
    property: String,
}

impl ZoneMapIndex {
    /// Creates a new zone map index for a property.
    pub fn new(property: impl Into<String>) -> Self {
        Self {
            entries: HashMap::new(),
            property: property.into(),
        }
    }

    /// Returns the property name.
    pub fn property(&self) -> &str {
        &self.property
    }

    /// Adds or updates a zone map entry for a chunk.
    pub fn insert(&mut self, chunk_id: u64, entry: ZoneMapEntry) {
        self.entries.insert(chunk_id, entry);
    }

    /// Gets the zone map entry for a chunk.
    pub fn get(&self, chunk_id: u64) -> Option<&ZoneMapEntry> {
        self.entries.get(&chunk_id)
    }

    /// Removes the zone map entry for a chunk.
    pub fn remove(&mut self, chunk_id: u64) -> Option<ZoneMapEntry> {
        self.entries.remove(&chunk_id)
    }

    /// Returns the number of chunks with zone maps.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if there are no zone maps.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Filters chunk IDs that might match an equality predicate.
    pub fn filter_equal<'a>(
        &'a self,
        value: &'a Value,
        chunk_ids: impl Iterator<Item = u64> + 'a,
    ) -> impl Iterator<Item = u64> + 'a {
        chunk_ids.filter(move |&id| {
            self.entries
                .get(&id)
                .map(|e| e.might_contain_equal(value))
                .unwrap_or(true) // No zone map = assume might contain
        })
    }

    /// Filters chunk IDs that might match a range predicate.
    pub fn filter_range<'a>(
        &'a self,
        lower: Option<&'a Value>,
        upper: Option<&'a Value>,
        lower_inclusive: bool,
        upper_inclusive: bool,
        chunk_ids: impl Iterator<Item = u64> + 'a,
    ) -> impl Iterator<Item = u64> + 'a {
        chunk_ids.filter(move |&id| {
            self.entries
                .get(&id)
                .map(|e| e.might_contain_range(lower, upper, lower_inclusive, upper_inclusive))
                .unwrap_or(true)
        })
    }

    /// Returns chunk IDs sorted by their minimum value.
    pub fn chunks_ordered_by_min(&self) -> Vec<u64> {
        let mut chunks: Vec<_> = self.entries.iter().collect();
        chunks.sort_by(|(_, a), (_, b)| match (&a.min, &b.min) {
            (Some(a_min), Some(b_min)) => compare_values(a_min, b_min).unwrap_or(Ordering::Equal),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        });
        chunks.into_iter().map(|(&id, _)| id).collect()
    }

    /// Returns overall statistics across all chunks.
    pub fn overall_stats(&self) -> (Option<Value>, Option<Value>, u64, u64) {
        let mut min: Option<Value> = None;
        let mut max: Option<Value> = None;
        let mut null_count = 0u64;
        let mut row_count = 0u64;

        for entry in self.entries.values() {
            null_count += entry.null_count;
            row_count += entry.row_count;

            if let Some(ref entry_min) = entry.min {
                min = match min {
                    None => Some(entry_min.clone()),
                    Some(ref current) => {
                        if compare_values(entry_min, current) == Some(Ordering::Less) {
                            Some(entry_min.clone())
                        } else {
                            min
                        }
                    }
                };
            }

            if let Some(ref entry_max) = entry.max {
                max = match max {
                    None => Some(entry_max.clone()),
                    Some(ref current) => {
                        if compare_values(entry_max, current) == Some(Ordering::Greater) {
                            Some(entry_max.clone())
                        } else {
                            max
                        }
                    }
                };
            }
        }

        (min, max, null_count, row_count)
    }
}

/// A probabilistic data structure for fast "definitely not in set" checks.
///
/// Bloom filters can have false positives (says "maybe" when absent) but never
/// false negatives (if it says "no", the value is definitely absent). Use this
/// to quickly rule out chunks that can't contain a value.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    /// Bit array.
    bits: Vec<u64>,
    /// Number of hash functions.
    num_hashes: usize,
    /// Number of bits.
    num_bits: usize,
}

impl BloomFilter {
    /// Creates a new Bloom filter.
    pub fn new(num_bits: usize, num_hashes: usize) -> Self {
        let num_words = (num_bits + 63) / 64;
        Self {
            bits: vec![0; num_words],
            num_hashes,
            num_bits,
        }
    }

    /// Adds a value to the filter.
    pub fn add(&mut self, value: &Value) {
        let hashes = self.compute_hashes(value);
        for h in hashes {
            let bit_idx = h % self.num_bits;
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;
            self.bits[word_idx] |= 1 << bit_pos;
        }
    }

    /// Checks if the filter might contain the value.
    pub fn might_contain(&self, value: &Value) -> bool {
        let hashes = self.compute_hashes(value);
        for h in hashes {
            let bit_idx = h % self.num_bits;
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;
            if (self.bits[word_idx] & (1 << bit_pos)) == 0 {
                return false;
            }
        }
        true
    }

    fn compute_hashes(&self, value: &Value) -> Vec<usize> {
        // Use a simple hash combination scheme
        let base_hash = value_hash(value);
        let mut hashes = Vec::with_capacity(self.num_hashes);

        for i in 0..self.num_hashes {
            // Double hashing: h(i) = h1 + i * h2
            let h1 = base_hash;
            let h2 = base_hash.rotate_left(17);
            hashes.push((h1.wrapping_add((i as u64).wrapping_mul(h2))) as usize);
        }

        hashes
    }
}

/// Builder for Bloom filters.
pub struct BloomFilterBuilder {
    filter: BloomFilter,
}

impl BloomFilterBuilder {
    /// Creates a new Bloom filter builder.
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        // Calculate optimal number of bits and hashes
        let num_bits = optimal_num_bits(expected_items, false_positive_rate);
        let num_hashes = optimal_num_hashes(num_bits, expected_items);

        Self {
            filter: BloomFilter::new(num_bits, num_hashes),
        }
    }

    /// Adds a value to the filter.
    pub fn add(&mut self, value: &Value) {
        self.filter.add(value);
    }

    /// Builds the Bloom filter.
    pub fn build(self) -> BloomFilter {
        self.filter
    }
}

/// Calculates optimal number of bits for a Bloom filter.
fn optimal_num_bits(n: usize, p: f64) -> usize {
    let ln2_squared = std::f64::consts::LN_2 * std::f64::consts::LN_2;
    ((-(n as f64) * p.ln()) / ln2_squared).ceil() as usize
}

/// Calculates optimal number of hash functions for a Bloom filter.
fn optimal_num_hashes(m: usize, n: usize) -> usize {
    ((m as f64 / n as f64) * std::f64::consts::LN_2).ceil() as usize
}

/// Computes a hash for a value.
fn value_hash(value: &Value) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    match value {
        Value::Null => 0u64.hash(&mut hasher),
        Value::Bool(b) => b.hash(&mut hasher),
        Value::Int64(i) => i.hash(&mut hasher),
        Value::Float64(f) => f.to_bits().hash(&mut hasher),
        Value::String(s) => s.hash(&mut hasher),
        Value::Bytes(b) => b.hash(&mut hasher),
        _ => format!("{value:?}").hash(&mut hasher),
    }

    hasher.finish()
}

/// Compares two values.
fn compare_values(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
        (Value::Int64(a), Value::Float64(b)) => (*a as f64).partial_cmp(b),
        (Value::Float64(a), Value::Int64(b)) => a.partial_cmp(&(*b as f64)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zone_map_entry_equal() {
        let entry = ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(100), 0, 100);

        assert!(entry.might_contain_equal(&Value::Int64(50)));
        assert!(entry.might_contain_equal(&Value::Int64(10)));
        assert!(entry.might_contain_equal(&Value::Int64(100)));
        assert!(!entry.might_contain_equal(&Value::Int64(5)));
        assert!(!entry.might_contain_equal(&Value::Int64(105)));
    }

    #[test]
    fn test_zone_map_entry_range() {
        let entry = ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(100), 0, 100);

        // Range fully contains chunk
        assert!(entry.might_contain_range(
            Some(&Value::Int64(0)),
            Some(&Value::Int64(200)),
            true,
            true
        ));

        // Range overlaps
        assert!(entry.might_contain_range(
            Some(&Value::Int64(50)),
            Some(&Value::Int64(150)),
            true,
            true
        ));

        // Range doesn't overlap
        assert!(!entry.might_contain_range(
            Some(&Value::Int64(101)),
            Some(&Value::Int64(200)),
            true,
            true
        ));
    }

    #[test]
    fn test_zone_map_builder() {
        let mut builder = ZoneMapBuilder::new();

        for i in 1..=100 {
            builder.add(&Value::Int64(i));
        }
        builder.add(&Value::Null);
        builder.add(&Value::Null);

        let entry = builder.build();

        assert_eq!(entry.min, Some(Value::Int64(1)));
        assert_eq!(entry.max, Some(Value::Int64(100)));
        assert_eq!(entry.null_count, 2);
        assert_eq!(entry.row_count, 102);
    }

    #[test]
    fn test_zone_map_with_bloom() {
        let mut builder = ZoneMapBuilder::with_bloom_filter(100, 0.01);

        for i in 1..=100 {
            builder.add(&Value::Int64(i));
        }

        let entry = builder.build();

        assert!(entry.bloom_filter.is_some());
        assert!(entry.might_contain_equal(&Value::Int64(50)));
        // Note: Bloom filters may have false positives but not false negatives
    }

    #[test]
    fn test_zone_map_index() {
        let mut index = ZoneMapIndex::new("age");

        index.insert(
            0,
            ZoneMapEntry::with_min_max(Value::Int64(0), Value::Int64(30), 0, 100),
        );
        index.insert(
            1,
            ZoneMapEntry::with_min_max(Value::Int64(31), Value::Int64(60), 0, 100),
        );
        index.insert(
            2,
            ZoneMapEntry::with_min_max(Value::Int64(61), Value::Int64(100), 0, 100),
        );

        // Filter for age = 25 - should only return chunk 0
        let matching: Vec<_> = index.filter_equal(&Value::Int64(25), 0..3).collect();
        assert_eq!(matching, vec![0]);

        // Filter for age = 75 - should only return chunk 2
        let matching: Vec<_> = index.filter_equal(&Value::Int64(75), 0..3).collect();
        assert_eq!(matching, vec![2]);

        // Filter for age between 25 and 65 - should return chunks 0, 1, 2
        let matching: Vec<_> = index
            .filter_range(
                Some(&Value::Int64(25)),
                Some(&Value::Int64(65)),
                true,
                true,
                0..3,
            )
            .collect();
        assert_eq!(matching, vec![0, 1, 2]);
    }

    #[test]
    fn test_bloom_filter() {
        let mut filter = BloomFilter::new(1000, 7);

        for i in 0..100 {
            filter.add(&Value::Int64(i));
        }

        // Should definitely contain values we added
        for i in 0..100 {
            assert!(filter.might_contain(&Value::Int64(i)));
        }

        // May have false positives for values not added, but let's test a few
        // (we can't assert false because of false positive rate)
        let _ = filter.might_contain(&Value::Int64(1000));
    }

    #[test]
    fn test_zone_map_nulls() {
        let entry = ZoneMapEntry {
            min: None,
            max: None,
            null_count: 10,
            row_count: 10,
            bloom_filter: None,
        };

        assert!(entry.is_all_null());
        assert!(!entry.might_contain_non_null());
        assert!(entry.might_contain_equal(&Value::Null));
        assert!(!entry.might_contain_equal(&Value::Int64(5)));
    }
}
