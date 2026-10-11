//! Embedding model management for GrafeoDB.

use std::sync::Arc;

use grafeo_common::utils::error::Result;

impl super::GrafeoDB {
    // ── Embedding ────────────────────────────────────────────────────────

    /// Loads a pre-configured embedding model, downloading from HuggingFace Hub if needed.
    ///
    /// For preset models, the ONNX model and tokenizer are automatically
    /// downloaded on first use and cached locally (in `~/.cache/huggingface/`).
    /// The model is registered under its display name (e.g., `"all-MiniLM-L6-v2"`).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use grafeo_engine::{GrafeoDB, Config, embedding::EmbeddingModelConfig};
    ///
    /// # fn main() -> grafeo_common::utils::error::Result<()> {
    /// let db = GrafeoDB::with_config(Config::in_memory())?;
    /// db.load_embedding_model(EmbeddingModelConfig::MiniLmL6v2)?;
    /// let vecs = db.embed_text("all-MiniLM-L6-v2", &["hello"])?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the model fails to download or load.
    #[cfg(feature = "embed")]
    pub fn load_embedding_model(
        &self,
        config: crate::embedding::EmbeddingModelConfig,
    ) -> Result<()> {
        self.load_embedding_model_with_options(
            config,
            crate::embedding::EmbeddingOptions::default(),
        )
    }

    /// Loads a pre-configured embedding model with custom options.
    ///
    /// See [`EmbeddingOptions`](crate::embedding::EmbeddingOptions) for
    /// batch size and thread configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the model fails to download, load, or initialize.
    #[cfg(feature = "embed")]
    pub fn load_embedding_model_with_options(
        &self,
        config: crate::embedding::EmbeddingModelConfig,
        options: crate::embedding::EmbeddingOptions,
    ) -> Result<()> {
        let name = config.display_name();
        let model =
            crate::embedding::OnnxEmbeddingModel::from_config_with_options(config, options)?;
        self.register_embedding_model(&name, Arc::new(model));
        Ok(())
    }

    /// Registers an embedding model for text-to-vector conversion.
    ///
    /// Once registered, you can use [`embed_text()`](Self::embed_text) and
    /// [`vector_search_text()`](Self::vector_search_text) with the model name.
    #[cfg(feature = "embed")]
    pub fn register_embedding_model(
        &self,
        name: &str,
        model: Arc<dyn crate::embedding::EmbeddingModel>,
    ) {
        self.embedding_models
            .write()
            .insert(name.to_string(), model);
    }

    /// Generates embeddings for a batch of texts using a registered model.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not registered or embedding fails.
    #[cfg(feature = "embed")]
    pub fn embed_text(&self, model_name: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let models = self.embedding_models.read();
        let model = models.get(model_name).ok_or_else(|| {
            grafeo_common::utils::error::Error::InvalidValue(format!(
                "no embedding model is registered as '{model_name}'"
            ))
        })?;
        model.embed(texts)
    }

    /// Searches a vector index using a text query, generating the embedding on-the-fly.
    ///
    /// This combines [`embed_text()`](Self::embed_text) with
    /// [`vector_search()`](Self::vector_search) in a single call.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not registered, embedding fails,
    /// or the vector index doesn't exist.
    #[cfg(all(feature = "embed", feature = "vector-index"))]
    pub fn vector_search_text(
        &self,
        label: &str,
        property: &str,
        model_name: &str,
        query_text: &str,
        k: usize,
        ef: Option<usize>,
    ) -> Result<Vec<(grafeo_common::types::NodeId, f32)>> {
        let vectors = self.embed_text(model_name, &[query_text])?;
        let query_vec = vectors.into_iter().next().ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "Embedding model returned no vectors".to_string(),
            )
        })?;
        self.vector_search(label, property, &query_vec, k, ef, None)
    }
}
