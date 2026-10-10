//! In-process embedding generation for zero-API-dependency vector workflows.
//!
//! When the `embed` feature is enabled, you can load ONNX models and generate
//! embeddings directly in your application without calling external APIs.
//!
//! # Example
//!
//! ```no_run
//! use grafeo_engine::embedding::{EmbeddingModel, OnnxEmbeddingModel};
//!
//! # fn main() -> grafeo_common::utils::error::Result<()> {
//! let model = OnnxEmbeddingModel::from_files(
//!     "all-MiniLM-L6-v2",
//!     "model.onnx",
//!     "tokenizer.json",
//! )?;
//! let vectors = model.embed(&["Hello world", "Graph databases"])?;
//! assert_eq!(vectors.len(), 2);
//! assert_eq!(vectors[0].len(), model.dimensions());
//! # Ok(())
//! # }
//! ```

mod config;
#[cfg(feature = "embed")]
mod download;
mod onnx;

pub use config::{EmbeddingModelConfig, EmbeddingOptions};
pub use onnx::OnnxEmbeddingModel;

use grafeo_common::utils::error::Result;

/// Trait for embedding models that convert text to vectors.
///
/// Implement this trait for custom embedding backends (API-based, local
/// inference, etc.). The default implementation provided is [`OnnxEmbeddingModel`]
/// which runs ONNX models locally via the ONNX Runtime.
pub trait EmbeddingModel: Send + Sync {
    /// Generates embeddings for a batch of texts.
    ///
    /// Returns one vector per input text, each with [`EmbeddingModel::dimensions()`] elements.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or inference fails.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Returns the dimensionality of the embedding vectors.
    fn dimensions(&self) -> usize;

    /// Returns the model name.
    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// A deterministic mock embedding model for testing.
    /// Generates vectors based on text length and character sum.
    struct MockEmbeddingModel {
        dims: usize,
    }

    impl MockEmbeddingModel {
        fn new(dims: usize) -> Self {
            Self { dims }
        }
    }

    impl EmbeddingModel for MockEmbeddingModel {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    let seed = t.bytes().map(|b| b as f32).sum::<f32>();
                    (0..self.dims)
                        .map(|i| ((seed + i as f32) * 0.01).sin())
                        .collect()
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            self.dims
        }

        fn name(&self) -> &str {
            "mock-embedding"
        }
    }

    #[test]
    fn trait_returns_correct_dimensions() {
        let model = MockEmbeddingModel::new(384);
        let vecs = model.embed(&["hello"]).unwrap();
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0].len(), 384);
        assert_eq!(model.dimensions(), 384);
    }

    #[test]
    fn trait_batch_returns_one_vec_per_input() {
        let model = MockEmbeddingModel::new(128);
        let texts = &["one", "two", "three", "four", "five"];
        let vecs = model.embed(texts).unwrap();
        assert_eq!(vecs.len(), 5);
        for v in &vecs {
            assert_eq!(v.len(), 128);
        }
    }

    #[test]
    fn trait_empty_input_returns_empty() {
        let model = MockEmbeddingModel::new(64);
        let vecs = model.embed(&[]).unwrap();
        assert!(vecs.is_empty(), "{vecs:?}");
    }

    #[test]
    fn trait_different_texts_produce_different_vectors() {
        let model = MockEmbeddingModel::new(32);
        let vecs = model.embed(&["hello", "world"]).unwrap();
        assert_ne!(vecs[0], vecs[1]);
    }

    #[test]
    fn trait_same_text_produces_same_vector() {
        let model = MockEmbeddingModel::new(32);
        let v1 = model.embed(&["hello"]).unwrap();
        let v2 = model.embed(&["hello"]).unwrap();
        assert_eq!(v1[0], v2[0]);
    }

    #[test]
    fn trait_is_send_sync() {
        // Verify the trait object can be shared across threads
        let model: Arc<dyn EmbeddingModel> = Arc::new(MockEmbeddingModel::new(64));
        let model_clone = Arc::clone(&model);
        let handle =
            std::thread::spawn(move || model_clone.embed(&["from another thread"]).unwrap());
        let result = handle.join().unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), model.dimensions());
    }

    #[test]
    fn database_register_and_embed() {
        use crate::{Config, GrafeoDB};

        let db = GrafeoDB::with_config(Config::in_memory()).unwrap();
        let model: Arc<dyn EmbeddingModel> = Arc::new(MockEmbeddingModel::new(128));
        db.register_embedding_model("test-model", model);

        let vecs = db.embed_text("test-model", &["hello", "world"]).unwrap();
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0].len(), 128);
        assert_eq!(vecs[1].len(), 128);
    }

    #[test]
    fn database_embed_model_not_found() {
        use crate::{Config, GrafeoDB};

        let db = GrafeoDB::with_config(Config::in_memory()).unwrap();
        let result = db.embed_text("nonexistent", &["hello"]);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("no embedding model is registered as 'nonexistent'"),
            "Error should mention model not registered: {err}"
        );
    }
}
