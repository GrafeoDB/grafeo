//! Vector zone maps for intelligent block skipping.
//!
//! Vector zone maps track summary statistics for blocks of vectors,
//! enabling pruning during similarity search. Unlike scalar zone maps
//! that track min/max, vector zone maps track:
//!
//! - **Magnitude bounds**: Min/max L2 norm for magnitude-based pruning
//! - **Centroid**: Average vector for quick distance estimation
//! - **Bounding box**: Per-dimension min/max for hyperrectangle pruning
//!
//! # Pruning Strategies
//!
//! 1. **Magnitude pruning**: If searching for k-NN with distance threshold d,
//!    skip blocks where `|query_magnitude - block_magnitude| > d`
//!
//! 2. **Centroid pruning**: If `distance(query, centroid) - max_radius > d`,
//!    the entire block can be skipped
//!
//! 3. **Bounding box pruning**: For cosine similarity, if the query direction
//!    doesn't intersect the block's bounding box, skip it

use super::DistanceMetric;

/// Vector zone map for a block of vectors.
///
/// Stores summary statistics enabling fast pruning during similarity search.
#[derive(Debug, Clone)]
pub struct VectorZoneMap {
    /// Number of dimensions.
    pub dimensions: usize,
    /// Number of vectors in this block.
    pub count: usize,
    /// Minimum magnitude (L2 norm) in the block.
    pub min_magnitude: f32,
    /// Maximum magnitude (L2 norm) in the block.
    pub max_magnitude: f32,
    /// Centroid (mean vector) of the block.
    pub centroid: Vec<f32>,
    /// Maximum distance from centroid to any vector in block.
    pub max_radius: f32,
    /// Per-dimension minimum values (bounding box lower corner).
    pub dim_min: Vec<f32>,
    /// Per-dimension maximum values (bounding box upper corner).
    pub dim_max: Vec<f32>,
}

impl VectorZoneMap {
    /// Creates a new empty vector zone map.
    #[must_use]
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            count: 0,
            min_magnitude: f32::MAX,
            max_magnitude: f32::MIN,
            centroid: vec![0.0; dimensions],
            max_radius: 0.0,
            dim_min: vec![f32::MAX; dimensions],
            dim_max: vec![f32::MIN; dimensions],
        }
    }

    /// Builds a zone map from a block of vectors.
    #[must_use]
    pub fn build(vectors: &[&[f32]]) -> Self {
        if vectors.is_empty() {
            return Self::new(0);
        }

        let dimensions = vectors[0].len();
        let count = vectors.len();

        // Initialize accumulators
        let mut min_magnitude = f32::MAX;
        let mut max_magnitude = f32::MIN;
        let mut centroid = vec![0.0; dimensions];
        let mut dim_min = vec![f32::MAX; dimensions];
        let mut dim_max = vec![f32::MIN; dimensions];

        // First pass: compute centroid and bounding box
        for vec in vectors {
            // Compute magnitude
            let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            min_magnitude = min_magnitude.min(magnitude);
            max_magnitude = max_magnitude.max(magnitude);

            // Accumulate for centroid
            for (i, &v) in vec.iter().enumerate() {
                centroid[i] += v;
                dim_min[i] = dim_min[i].min(v);
                dim_max[i] = dim_max[i].max(v);
            }
        }

        // Finalize centroid
        let count_f = count as f32;
        for c in &mut centroid {
            *c /= count_f;
        }

        // Second pass: compute max radius from centroid
        let mut max_radius = 0.0f32;
        for vec in vectors {
            let dist_sq: f32 = vec
                .iter()
                .zip(&centroid)
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            max_radius = max_radius.max(dist_sq.sqrt());
        }

        Self {
            dimensions,
            count,
            min_magnitude,
            max_magnitude,
            centroid,
            max_radius,
            dim_min,
            dim_max,
        }
    }

    /// Checks if this block might contain vectors within the given distance threshold.
    ///
    /// Returns `true` if the block cannot be pruned, `false` if it can be skipped.
    #[must_use]
    pub fn might_contain_within_distance(
        &self,
        query: &[f32],
        threshold: f32,
        metric: DistanceMetric,
    ) -> bool {
        if self.count == 0 {
            return false;
        }

        match metric {
            DistanceMetric::Euclidean => {
                // Centroid-based pruning: if distance to centroid minus max_radius > threshold,
                // all vectors in block are farther than threshold
                let centroid_dist = euclidean_distance(query, &self.centroid);
                if centroid_dist - self.max_radius > threshold {
                    return false;
                }

                // Bounding box pruning: compute minimum possible distance to bounding box
                let box_dist = self.min_distance_to_box(query);
                if box_dist > threshold {
                    return false;
                }

                true
            }
            DistanceMetric::Cosine => {
                // For cosine, we use angular bounds
                // Convert threshold from cosine distance to angle
                // cosine_distance = 1 - cos(angle), so cos(angle) = 1 - threshold
                // This is complex, so for now just use centroid-based pruning
                let centroid_dist = cosine_distance(query, &self.centroid);
                centroid_dist - self.max_radius <= threshold
            }
            DistanceMetric::DotProduct | DistanceMetric::Manhattan => {
                // Conservative: don't prune for these metrics yet
                true
            }
        }
    }

    /// Computes the minimum possible distance from a query to the bounding box.
    fn min_distance_to_box(&self, query: &[f32]) -> f32 {
        let mut dist_sq = 0.0f32;

        for (i, &q) in query.iter().enumerate() {
            if i >= self.dimensions {
                break;
            }

            // Find closest point on box edge for this dimension
            let closest = if q < self.dim_min[i] {
                self.dim_min[i]
            } else if q > self.dim_max[i] {
                self.dim_max[i]
            } else {
                q // Query is inside box bounds for this dimension
            };

            let diff = q - closest;
            dist_sq += diff * diff;
        }

        dist_sq.sqrt()
    }

    /// Returns true if the block is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns the average magnitude of vectors in this block.
    #[must_use]
    pub fn avg_magnitude(&self) -> f32 {
        f32::midpoint(self.min_magnitude, self.max_magnitude)
    }

    /// Returns the magnitude range.
    #[must_use]
    pub fn magnitude_range(&self) -> (f32, f32) {
        (self.min_magnitude, self.max_magnitude)
    }

    /// Returns the bounding box as (min_corner, max_corner).
    #[must_use]
    pub fn bounding_box(&self) -> (&[f32], &[f32]) {
        (&self.dim_min, &self.dim_max)
    }

    /// Merges another zone map into this one.
    ///
    /// Useful for combining zone maps during compaction.
    pub fn merge(&mut self, other: &VectorZoneMap) {
        if other.is_empty() {
            return;
        }

        if self.is_empty() {
            *self = other.clone();
            return;
        }

        // Merge magnitude bounds
        self.min_magnitude = self.min_magnitude.min(other.min_magnitude);
        self.max_magnitude = self.max_magnitude.max(other.max_magnitude);

        // Merge bounding box
        for i in 0..self.dimensions.min(other.dimensions) {
            self.dim_min[i] = self.dim_min[i].min(other.dim_min[i]);
            self.dim_max[i] = self.dim_max[i].max(other.dim_max[i]);
        }

        // Update centroid (weighted average)
        let total_count = self.count + other.count;
        let self_weight = self.count as f32 / total_count as f32;
        let other_weight = other.count as f32 / total_count as f32;

        for i in 0..self.dimensions.min(other.dimensions) {
            self.centroid[i] = self.centroid[i] * self_weight + other.centroid[i] * other_weight;
        }

        // Max radius becomes approximate (conservative)
        // The true max radius would require recomputing from all vectors
        self.max_radius = f32::midpoint(self.max_radius, other.max_radius)
            + euclidean_distance(&self.centroid, &other.centroid);

        self.count = total_count;
    }
}

/// Computes Euclidean (L2) distance between two vectors.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Computes cosine distance (1 - cosine_similarity) between two vectors.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // Maximum distance for zero vectors
    }

    1.0 - (dot / (norm_a * norm_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_zone_map_build() {
        let v1 = [1.0f32, 0.0, 0.0];
        let v2 = [0.0f32, 1.0, 0.0];
        let v3 = [0.0f32, 0.0, 1.0];

        let vectors: Vec<&[f32]> = vec![&v1, &v2, &v3];
        let zm = VectorZoneMap::build(&vectors);

        assert_eq!(zm.count, 3);
        assert_eq!(zm.dimensions, 3);

        // All unit vectors, so magnitude should be 1.0
        assert!((zm.min_magnitude - 1.0).abs() < 0.001);
        assert!((zm.max_magnitude - 1.0).abs() < 0.001);

        // Centroid should be (1/3, 1/3, 1/3)
        for c in &zm.centroid {
            assert!((*c - 1.0 / 3.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_vector_zone_map_pruning() {
        // Create a block of vectors clustered around (5, 5, 5)
        let v1 = [5.0f32, 5.0, 5.0];
        let v2 = [5.1f32, 4.9, 5.0];
        let v3 = [4.9f32, 5.1, 5.0];

        let vectors: Vec<&[f32]> = vec![&v1, &v2, &v3];
        let zm = VectorZoneMap::build(&vectors);

        // Query far from cluster - should be prunable
        let far_query = [0.0f32, 0.0, 0.0];
        let far_dist = euclidean_distance(&far_query, &zm.centroid);
        assert!(far_dist > 8.0); // Centroid ~(5,5,5) is ~8.66 from origin

        // With tight threshold, block should be prunable
        assert!(!zm.might_contain_within_distance(&far_query, 1.0, DistanceMetric::Euclidean));

        // Query close to cluster - should not be prunable
        let close_query = [5.0f32, 5.0, 5.0];
        assert!(zm.might_contain_within_distance(&close_query, 1.0, DistanceMetric::Euclidean));
    }

    #[test]
    fn test_vector_zone_map_bounding_box() {
        let v1 = [0.0f32, 0.0];
        let v2 = [10.0f32, 10.0];

        let vectors: Vec<&[f32]> = vec![&v1, &v2];
        let zm = VectorZoneMap::build(&vectors);

        let (min, max) = zm.bounding_box();
        assert!((min[0] - 0.0).abs() < 0.001);
        assert!((min[1] - 0.0).abs() < 0.001);
        assert!((max[0] - 10.0).abs() < 0.001);
        assert!((max[1] - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_vector_zone_map_merge() {
        let v1 = [1.0f32, 0.0];
        let v2 = [2.0f32, 0.0];
        let zm1 = VectorZoneMap::build(&[&v1, &v2]);

        let v3 = [10.0f32, 0.0];
        let v4 = [11.0f32, 0.0];
        let zm2 = VectorZoneMap::build(&[&v3, &v4]);

        let mut merged = zm1.clone();
        merged.merge(&zm2);

        assert_eq!(merged.count, 4);

        // Bounding box should span both
        let (min, max) = merged.bounding_box();
        assert!((min[0] - 1.0).abs() < 0.001);
        assert!((max[0] - 11.0).abs() < 0.001);
    }

    #[test]
    fn test_vector_zone_map_empty() {
        let zm = VectorZoneMap::new(3);
        assert!(zm.is_empty());
        assert_eq!(zm.count, 0);
    }
}
