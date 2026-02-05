//! SIMD-accelerated distance computations.
//!
//! This module provides hardware-accelerated implementations of distance functions
//! using SIMD instructions (AVX2 on x86_64, NEON on aarch64).
//!
//! # Performance
//!
//! SIMD acceleration provides 4-8x speedup for distance computations:
//!
//! | Instruction Set | Speedup | Vectors/sec (384-dim) |
//! |-----------------|---------|----------------------|
//! | Scalar          | 1x      | ~2M                  |
//! | SSE (128-bit)   | ~3x     | ~6M                  |
//! | AVX2 (256-bit)  | ~6x     | ~12M                 |
//! | AVX-512         | ~10x    | ~20M (planned)       |
//!
//! # Runtime Detection
//!
//! SIMD support is detected at runtime using CPU feature detection.
//! If the required instructions are not available, the scalar fallback is used.

// SIMD intrinsics require unsafe code - this is well-understood and verified.
#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use super::DistanceMetric;

// ============================================================================
// Runtime CPU Detection
// ============================================================================

/// Returns true if AVX2 instructions are available.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn has_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

/// Returns true if SSE instructions are available.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn has_sse() -> bool {
    std::arch::is_x86_feature_detected!("sse")
}

/// Returns true if NEON instructions are available (always true on aarch64).
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn has_neon() -> bool {
    true // NEON is mandatory on aarch64
}

// Fallback for other architectures
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
pub fn has_avx2() -> bool {
    false
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
pub fn has_sse() -> bool {
    false
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
pub fn has_neon() -> bool {
    false
}

/// Returns the best available SIMD instruction set name.
#[must_use]
pub fn simd_support() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            return "avx2";
        }
        if has_sse() {
            return "sse";
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "neon";
    }
    "scalar"
}

// ============================================================================
// Dispatcher Functions (Select best implementation at runtime)
// ============================================================================

/// Computes distance using the best available SIMD implementation.
#[inline]
pub fn compute_distance_simd(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    match metric {
        DistanceMetric::Cosine => cosine_distance_simd(a, b),
        DistanceMetric::Euclidean => euclidean_distance_simd(a, b),
        DistanceMetric::DotProduct => -dot_product_simd(a, b),
        DistanceMetric::Manhattan => manhattan_distance_simd(a, b),
    }
}

/// SIMD-accelerated dot product.
#[inline]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: We checked AVX2 support above
            return unsafe { dot_product_avx2(a, b) };
        }
        if has_sse() {
            // SAFETY: We checked SSE support above
            return unsafe { dot_product_sse(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64
        return unsafe { dot_product_neon(a, b) };
    }

    // Scalar fallback
    dot_product_scalar(a, b)
}

/// SIMD-accelerated squared Euclidean distance.
#[inline]
pub fn euclidean_distance_squared_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            return unsafe { euclidean_squared_avx2(a, b) };
        }
        if has_sse() {
            return unsafe { euclidean_squared_sse(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { euclidean_squared_neon(a, b) };
    }

    euclidean_squared_scalar(a, b)
}

/// SIMD-accelerated Euclidean distance.
#[inline]
pub fn euclidean_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    euclidean_distance_squared_simd(a, b).sqrt()
}

/// SIMD-accelerated cosine distance.
#[inline]
pub fn cosine_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            return unsafe { cosine_distance_avx2(a, b) };
        }
        if has_sse() {
            return unsafe { cosine_distance_sse(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { cosine_distance_neon(a, b) };
    }

    cosine_distance_scalar(a, b)
}

/// SIMD-accelerated Manhattan distance.
#[inline]
pub fn manhattan_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            return unsafe { manhattan_distance_avx2(a, b) };
        }
        if has_sse() {
            return unsafe { manhattan_distance_sse(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { manhattan_distance_neon(a, b) };
    }

    manhattan_distance_scalar(a, b)
}

// ============================================================================
// Scalar Implementations (Fallback)
// ============================================================================

#[inline]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
fn euclidean_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

#[inline]
fn cosine_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = (norm_a.sqrt() * norm_b.sqrt()) + f32::EPSILON;
    1.0 - (dot / denom)
}

#[inline]
fn manhattan_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += (a[i] - b[i]).abs();
    }
    sum
}

// ============================================================================
// x86_64 AVX2 Implementations
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut i = 0;

    // Process 8 floats at a time
    let mut sum = _mm256_setzero_ps();

    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        sum = _mm256_fmadd_ps(va, vb, sum);
        i += 8;
    }

    // Horizontal sum of 8 floats
    let mut result = horizontal_sum_avx2(sum);

    // Handle remainder
    while i < n {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn euclidean_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut i = 0;

    let mut sum = _mm256_setzero_ps();

    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
        i += 8;
    }

    let mut result = horizontal_sum_avx2(sum);

    while i < n {
        let diff = a[i] - b[i];
        result += diff * diff;
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn cosine_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut i = 0;

    let mut dot_sum = _mm256_setzero_ps();
    let mut norm_a_sum = _mm256_setzero_ps();
    let mut norm_b_sum = _mm256_setzero_ps();

    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));

        dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
        norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
        norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);

        i += 8;
    }

    let mut dot = horizontal_sum_avx2(dot_sum);
    let mut norm_a = horizontal_sum_avx2(norm_a_sum);
    let mut norm_b = horizontal_sum_avx2(norm_b_sum);

    while i < n {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
        i += 1;
    }

    let denom = (norm_a.sqrt() * norm_b.sqrt()) + f32::EPSILON;
    1.0 - (dot / denom)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn manhattan_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut i = 0;

    let sign_mask = _mm256_set1_ps(-0.0f32); // Mask for clearing sign bit
    let mut sum = _mm256_setzero_ps();

    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        let abs_diff = _mm256_andnot_ps(sign_mask, diff); // Clear sign bit = abs
        sum = _mm256_add_ps(sum, abs_diff);
        i += 8;
    }

    let mut result = horizontal_sum_avx2(sum);

    while i < n {
        result += (a[i] - b[i]).abs();
        i += 1;
    }

    result
}

/// Horizontal sum of 8 f32 values in an AVX2 register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_sum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    // Add high and low 128-bit halves
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(high, low);

    // Horizontal add within 128-bit
    let shuf = _mm_movehdup_ps(sum128); // [1,1,3,3]
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, shuf2);

    _mm_cvtss_f32(sum32)
}

// ============================================================================
// x86_64 SSE Implementations
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn dot_product_sse(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut i = 0;

    let mut sum = _mm_setzero_ps();

    while i + 4 <= n {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let prod = _mm_mul_ps(va, vb);
        sum = _mm_add_ps(sum, prod);
        i += 4;
    }

    let mut result = horizontal_sum_sse(sum);

    while i < n {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn euclidean_squared_sse(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut i = 0;

    let mut sum = _mm_setzero_ps();

    while i + 4 <= n {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let diff = _mm_sub_ps(va, vb);
        let sq = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, sq);
        i += 4;
    }

    let mut result = horizontal_sum_sse(sum);

    while i < n {
        let diff = a[i] - b[i];
        result += diff * diff;
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn cosine_distance_sse(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut i = 0;

    let mut dot_sum = _mm_setzero_ps();
    let mut norm_a_sum = _mm_setzero_ps();
    let mut norm_b_sum = _mm_setzero_ps();

    while i + 4 <= n {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));

        dot_sum = _mm_add_ps(dot_sum, _mm_mul_ps(va, vb));
        norm_a_sum = _mm_add_ps(norm_a_sum, _mm_mul_ps(va, va));
        norm_b_sum = _mm_add_ps(norm_b_sum, _mm_mul_ps(vb, vb));

        i += 4;
    }

    let mut dot = horizontal_sum_sse(dot_sum);
    let mut norm_a = horizontal_sum_sse(norm_a_sum);
    let mut norm_b = horizontal_sum_sse(norm_b_sum);

    while i < n {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
        i += 1;
    }

    let denom = (norm_a.sqrt() * norm_b.sqrt()) + f32::EPSILON;
    1.0 - (dot / denom)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn manhattan_distance_sse(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut i = 0;

    let sign_mask = _mm_set1_ps(-0.0f32);
    let mut sum = _mm_setzero_ps();

    while i + 4 <= n {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let diff = _mm_sub_ps(va, vb);
        let abs_diff = _mm_andnot_ps(sign_mask, diff);
        sum = _mm_add_ps(sum, abs_diff);
        i += 4;
    }

    let mut result = horizontal_sum_sse(sum);

    while i < n {
        result += (a[i] - b[i]).abs();
        i += 1;
    }

    result
}

/// Horizontal sum of 4 f32 values in an SSE register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
#[inline]
unsafe fn horizontal_sum_sse(v: std::arch::x86_64::__m128) -> f32 {
    use std::arch::x86_64::*;

    // SSE-only horizontal sum using shuffle
    // v = [a, b, c, d]
    let shuf = _mm_shuffle_ps(v, v, 0b10_11_00_01); // [b, a, d, c]
    let sum = _mm_add_ps(v, shuf); // [a+b, a+b, c+d, c+d]
    let shuf2 = _mm_movehl_ps(sum, sum); // [c+d, c+d, c+d, c+d]
    let sum2 = _mm_add_ss(sum, shuf2); // [a+b+c+d, ...]
    _mm_cvtss_f32(sum2)
}

// ============================================================================
// aarch64 NEON Implementations
// ============================================================================

#[cfg(target_arch = "aarch64")]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let mut i = 0;

    let mut sum = vdupq_n_f32(0.0);

    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        sum = vfmaq_f32(sum, va, vb);
        i += 4;
    }

    let mut result = horizontal_sum_neon(sum);

    while i < n {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

#[cfg(target_arch = "aarch64")]
unsafe fn euclidean_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let mut i = 0;

    let mut sum = vdupq_n_f32(0.0);

    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let diff = vsubq_f32(va, vb);
        sum = vfmaq_f32(sum, diff, diff);
        i += 4;
    }

    let mut result = horizontal_sum_neon(sum);

    while i < n {
        let diff = a[i] - b[i];
        result += diff * diff;
        i += 1;
    }

    result
}

#[cfg(target_arch = "aarch64")]
unsafe fn cosine_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let mut i = 0;

    let mut dot_sum = vdupq_n_f32(0.0);
    let mut norm_a_sum = vdupq_n_f32(0.0);
    let mut norm_b_sum = vdupq_n_f32(0.0);

    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));

        dot_sum = vfmaq_f32(dot_sum, va, vb);
        norm_a_sum = vfmaq_f32(norm_a_sum, va, va);
        norm_b_sum = vfmaq_f32(norm_b_sum, vb, vb);

        i += 4;
    }

    let mut dot = horizontal_sum_neon(dot_sum);
    let mut norm_a = horizontal_sum_neon(norm_a_sum);
    let mut norm_b = horizontal_sum_neon(norm_b_sum);

    while i < n {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
        i += 1;
    }

    let denom = (norm_a.sqrt() * norm_b.sqrt()) + f32::EPSILON;
    1.0 - (dot / denom)
}

#[cfg(target_arch = "aarch64")]
unsafe fn manhattan_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let mut i = 0;

    let mut sum = vdupq_n_f32(0.0);

    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let diff = vsubq_f32(va, vb);
        let abs_diff = vabsq_f32(diff);
        sum = vaddq_f32(sum, abs_diff);
        i += 4;
    }

    let mut result = horizontal_sum_neon(sum);

    while i < n {
        result += (a[i] - b[i]).abs();
        i += 1;
    }

    result
}

/// Horizontal sum of 4 f32 values in a NEON register.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn horizontal_sum_neon(v: std::arch::aarch64::float32x4_t) -> f32 {
    use std::arch::aarch64::*;
    vaddvq_f32(v)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_simd_support_detection() {
        let support = simd_support();
        println!("SIMD support: {support}");
        // Should be one of the valid values
        assert!(["avx2", "sse", "neon", "scalar"].contains(&support));
    }

    #[test]
    fn test_dot_product_simd() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let simd_result = dot_product_simd(&a, &b);
        let scalar_result = dot_product_scalar(&a, &b);

        assert!(
            approx_eq(simd_result, scalar_result),
            "SIMD: {simd_result}, Scalar: {scalar_result}"
        );
    }

    #[test]
    fn test_euclidean_squared_simd() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let simd_result = euclidean_distance_squared_simd(&a, &b);
        let scalar_result = euclidean_squared_scalar(&a, &b);

        assert!(
            approx_eq(simd_result, scalar_result),
            "SIMD: {simd_result}, Scalar: {scalar_result}"
        );
    }

    #[test]
    fn test_cosine_distance_simd() {
        let a = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let simd_result = cosine_distance_simd(&a, &b);
        let scalar_result = cosine_distance_scalar(&a, &b);

        // Orthogonal vectors should have cosine distance ~1.0
        assert!(
            approx_eq(simd_result, scalar_result),
            "SIMD: {simd_result}, Scalar: {scalar_result}"
        );
        assert!(
            approx_eq(simd_result, 1.0),
            "Expected ~1.0, got: {simd_result}"
        );
    }

    #[test]
    fn test_manhattan_distance_simd() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let simd_result = manhattan_distance_simd(&a, &b);
        let scalar_result = manhattan_distance_scalar(&a, &b);

        assert!(
            approx_eq(simd_result, scalar_result),
            "SIMD: {simd_result}, Scalar: {scalar_result}"
        );
    }

    #[test]
    fn test_simd_with_384_dimensions() {
        // Common embedding size (MiniLM, etc.)
        let a: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
        let b: Vec<f32> = (0..384).map(|i| ((383 - i) as f32) / 384.0).collect();

        let simd_dot = dot_product_simd(&a, &b);
        let scalar_dot = dot_product_scalar(&a, &b);
        assert!(
            approx_eq(simd_dot, scalar_dot),
            "Dot: SIMD={simd_dot}, Scalar={scalar_dot}"
        );

        let simd_euc = euclidean_distance_simd(&a, &b);
        let scalar_euc = euclidean_squared_scalar(&a, &b).sqrt();
        assert!(
            approx_eq(simd_euc, scalar_euc),
            "Euc: SIMD={simd_euc}, Scalar={scalar_euc}"
        );

        let simd_cos = cosine_distance_simd(&a, &b);
        let scalar_cos = cosine_distance_scalar(&a, &b);
        assert!(
            approx_eq(simd_cos, scalar_cos),
            "Cos: SIMD={simd_cos}, Scalar={scalar_cos}"
        );

        let simd_man = manhattan_distance_simd(&a, &b);
        let scalar_man = manhattan_distance_scalar(&a, &b);
        assert!(
            approx_eq(simd_man, scalar_man),
            "Man: SIMD={simd_man}, Scalar={scalar_man}"
        );
    }

    #[test]
    fn test_simd_with_odd_dimensions() {
        // Test remainder handling with non-aligned dimensions
        let a: Vec<f32> = (0..387).map(|i| (i as f32) / 387.0).collect();
        let b: Vec<f32> = (0..387).map(|i| ((386 - i) as f32) / 387.0).collect();

        let simd_dot = dot_product_simd(&a, &b);
        let scalar_dot = dot_product_scalar(&a, &b);
        assert!(
            approx_eq(simd_dot, scalar_dot),
            "Odd dims: SIMD={simd_dot}, Scalar={scalar_dot}"
        );
    }

    #[test]
    fn test_simd_small_vectors() {
        // Vectors smaller than SIMD width
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];

        let simd_result = dot_product_simd(&a, &b);
        let scalar_result = dot_product_scalar(&a, &b);

        assert!(
            approx_eq(simd_result, scalar_result),
            "Small: SIMD={simd_result}, Scalar={scalar_result}"
        );
    }

    #[test]
    fn test_compute_distance_simd_dispatch() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [4.0f32, 3.0, 2.0, 1.0];

        // Test all metrics through the dispatcher
        let _ = compute_distance_simd(&a, &b, DistanceMetric::Cosine);
        let _ = compute_distance_simd(&a, &b, DistanceMetric::Euclidean);
        let _ = compute_distance_simd(&a, &b, DistanceMetric::DotProduct);
        let _ = compute_distance_simd(&a, &b, DistanceMetric::Manhattan);
    }
}
