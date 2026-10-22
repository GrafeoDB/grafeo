//! ONNX Runtime embedding model implementation.

use grafeo_common::utils::error::{Error, Result};
use ort::session::Session;
use parking_lot::Mutex;
use std::path::Path;
use tokenizers::Tokenizer;

use super::EmbeddingModel;

/// An embedding model backed by an ONNX session and a Hugging Face tokenizer.
///
/// Load from local files using [`from_files()`](Self::from_files). The model
/// must accept `input_ids` and `attention_mask` as i64 tensors and produce a
/// pooled embedding output.
pub struct OnnxEmbeddingModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    dimensions: usize,
    name: String,
    batch_size: usize,
}

// SAFETY: Tokenizer is !Send+!Sync by default, but our usage is safe because
// we protect the session with a Mutex and never share mutable tokenizer state
// after construction.
#[allow(unsafe_code)]
unsafe impl Send for OnnxEmbeddingModel {}
#[allow(unsafe_code)]
unsafe impl Sync for OnnxEmbeddingModel {}

impl std::fmt::Debug for OnnxEmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxEmbeddingModel")
            .field("name", &self.name)
            .field("dimensions", &self.dimensions)
            .field("batch_size", &self.batch_size)
            .finish()
    }
}

impl OnnxEmbeddingModel {
    /// Loads a pre-configured embedding model, downloading from HuggingFace Hub if needed.
    ///
    /// For preset models, the ONNX model and tokenizer are automatically downloaded
    /// on first use and cached locally. See [`EmbeddingModelConfig`](super::EmbeddingModelConfig)
    /// for available presets.
    ///
    /// # Errors
    ///
    /// Returns an error if the model fails to download, load, or initialize.
    #[cfg(feature = "embed")]
    pub fn from_config(config: super::EmbeddingModelConfig) -> Result<Self> {
        Self::from_config_with_options(config, super::EmbeddingOptions::default())
    }

    /// Loads a pre-configured embedding model with custom options.
    ///
    /// See [`EmbeddingOptions`](super::EmbeddingOptions) for batch size and thread configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the ONNX session or tokenizer fails to initialize.
    #[cfg(feature = "embed")]
    pub fn from_config_with_options(
        config: super::EmbeddingModelConfig,
        options: super::EmbeddingOptions,
    ) -> Result<Self> {
        let resolved = super::download::resolve(&config)?;

        let mut session = Session::builder()
            .map_err(|e| Error::Internal(format!("Failed to create ONNX session builder: {e}")))?
            .with_inter_threads(options.inter_threads)
            .map_err(|e| Error::Internal(format!("Failed to set inter threads: {e}")))?
            .with_intra_threads(options.intra_threads)
            .map_err(|e| Error::Internal(format!("Failed to set intra threads: {e}")))?
            .commit_from_file(&resolved.model_path)
            .map_err(|e| Error::Internal(format!("Failed to load ONNX model: {e}")))?;

        let tokenizer = Tokenizer::from_file(&resolved.tokenizer_path)
            .map_err(|e| Error::Internal(format!("Failed to load tokenizer: {e}")))?;

        let dimensions = Self::probe_dimensions(&mut session, &tokenizer)?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            dimensions,
            name: resolved.name,
            batch_size: options.batch_size,
        })
    }

    /// Loads an embedding model from local ONNX model and tokenizer files.
    ///
    /// # Arguments
    ///
    /// * `name` - Human-readable model name (e.g., "all-MiniLM-L6-v2")
    /// * `model_path` - Path to the `.onnx` model file
    /// * `tokenizer_path` - Path to the `tokenizer.json` file
    ///
    /// # Errors
    ///
    /// Returns an error if the files cannot be loaded or the model is invalid.
    pub fn from_files(
        name: impl Into<String>,
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let mut session = Session::builder()
            .map_err(|e| Error::Internal(format!("Failed to create ONNX session builder: {e}")))?
            .with_inter_threads(1)
            .map_err(|e| Error::Internal(format!("Failed to set inter threads: {e}")))?
            .with_intra_threads(1)
            .map_err(|e| Error::Internal(format!("Failed to set intra threads: {e}")))?
            .commit_from_file(model_path.as_ref())
            .map_err(|e| Error::Internal(format!("Failed to load ONNX model: {e}")))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| Error::Internal(format!("Failed to load tokenizer: {e}")))?;

        // Probe dimensions by running a dummy input
        let dimensions = Self::probe_dimensions(&mut session, &tokenizer)?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            dimensions,
            name: name.into(),
            batch_size: 32,
        })
    }

    /// Sets the maximum batch size for embedding generation.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Creates an i64 tensor from shape and flat data.
    fn make_i64_tensor(
        shape: Vec<usize>,
        data: Vec<i64>,
    ) -> Result<ort::value::Value<ort::value::TensorValueType<i64>>> {
        ort::value::Tensor::from_array((shape, data.into_boxed_slice()))
            .map_err(|e| Error::Internal(format!("Failed to create tensor: {e}")))
    }

    /// Probes model output dimensions with a dummy input.
    fn probe_dimensions(session: &mut Session, tokenizer: &Tokenizer) -> Result<usize> {
        let encoding = tokenizer
            .encode("hello", false)
            .map_err(|e| Error::Internal(format!("Tokenizer encode error: {e}")))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let len = ids.len();

        let input_ids = Self::make_i64_tensor(vec![1, len], ids)?;
        let attention_mask = Self::make_i64_tensor(vec![1, len], mask)?;

        let outputs = session
            .run(ort::inputs![input_ids, attention_mask])
            .map_err(|e| Error::Internal(format!("Model inference failed: {e}")))?;

        let output = &outputs[0];
        let (shape, _data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Internal(format!("Failed to extract output tensor: {e}")))?;

        // Shape is typically [batch, dims] or [batch, seq_len, dims]
        let dims = *shape
            .last()
            .ok_or_else(|| Error::Internal("Model output has no dimensions".to_string()))?;

        // reason: ONNX tensor dimensions are always non-negative and fit usize
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let dim = dims as usize;
        Ok(dim)
    }

    /// Embeds a single batch of texts.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let text_strings: Vec<String> = texts.iter().map(|t| (*t).to_string()).collect();
        let encodings = self
            .tokenizer
            .encode_batch(text_strings, false)
            .map_err(|e| Error::Internal(format!("Tokenizer batch encode error: {e}")))?;

        // Pad to uniform length
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        let batch = texts.len();
        let mut all_ids = vec![0i64; batch * max_len];
        let mut all_mask = vec![0i64; batch * max_len];

        for (i, enc) in encodings.iter().enumerate() {
            for (j, &id) in enc.get_ids().iter().enumerate() {
                all_ids[i * max_len + j] = id as i64;
            }
            for (j, &m) in enc.get_attention_mask().iter().enumerate() {
                all_mask[i * max_len + j] = m as i64;
            }
        }

        let input_ids = Self::make_i64_tensor(vec![batch, max_len], all_ids)?;
        let attention_mask = Self::make_i64_tensor(vec![batch, max_len], all_mask)?;

        let mut session = self.session.lock();
        let outputs = session
            .run(ort::inputs![input_ids, attention_mask])
            .map_err(|e| Error::Internal(format!("Model inference failed: {e}")))?;

        let output = &outputs[0];
        let (shape, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Internal(format!("Failed to extract output tensor: {e}")))?;

        // Handle different output shapes
        // reason: ONNX tensor dimensions are always non-negative and fit usize
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();

        let result = if shape_usize.len() == 3 {
            // [batch, seq_len, dims] → take [CLS] token embedding (index 0)
            let seq_len = shape_usize[1];
            let dims = shape_usize[2];
            (0..batch)
                .map(|b| {
                    let offset = b * seq_len * dims; // CLS = seq index 0
                    data[offset..offset + dims].to_vec()
                })
                .collect()
        } else if shape_usize.len() == 2 {
            // [batch, dims] → already pooled
            let dims = shape_usize[1];
            (0..batch)
                .map(|b| {
                    let offset = b * dims;
                    data[offset..offset + dims].to_vec()
                })
                .collect()
        } else {
            return Err(Error::Internal(format!(
                "Unexpected output shape: {shape_usize:?}"
            )));
        };

        Ok(result)
    }
}

impl EmbeddingModel for OnnxEmbeddingModel {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size) {
            results.extend(self.embed_batch(chunk)?);
        }
        Ok(results)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        &self.name
    }
}
