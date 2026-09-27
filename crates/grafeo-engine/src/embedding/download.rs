//! Model cache management and download from HuggingFace Hub.
//!
//! Downloads ONNX model and tokenizer files on first use and caches them
//! locally. The cache is managed by `hf-hub` and defaults to
//! `~/.cache/huggingface/hub/` (compatible with the Python `huggingface_hub`
//! package). Override with the `HF_HOME` or `HF_HUB_CACHE` environment
//! variables.

use std::path::PathBuf;

use grafeo_common::utils::error::{Error, Result};
use hf_hub::HFClientSync;

use super::config::{EmbeddingModelConfig, ResolveInfo};

/// Resolved local paths to a downloaded (or already local) model.
pub(crate) struct ResolvedModel {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub name: String,
}

/// Resolves model files to local paths, downloading from HuggingFace Hub if necessary.
///
/// For [`Local`](EmbeddingModelConfig::Local) configs, returns the paths directly.
/// For preset and [`HuggingFace`](EmbeddingModelConfig::HuggingFace) configs,
/// downloads the model and tokenizer files from HuggingFace Hub on first use.
/// Subsequent calls return cached paths instantly.
pub(crate) fn resolve(config: &EmbeddingModelConfig) -> Result<ResolvedModel> {
    let name = config.display_name();

    match config.resolve_info() {
        ResolveInfo::Local {
            model_path,
            tokenizer_path,
        } => Ok(ResolvedModel {
            model_path: model_path.clone(),
            tokenizer_path: tokenizer_path.clone(),
            name,
        }),
        ResolveInfo::Hub {
            repo_id,
            model_file,
            tokenizer_file,
        } => {
            let (owner, repo_name) = repo_id.split_once('/').ok_or_else(|| {
                Error::Internal(format!(
                    "Invalid HuggingFace repo id '{repo_id}': expected 'owner/name'"
                ))
            })?;

            let client = HFClientSync::new().map_err(|e| {
                Error::Internal(format!("Failed to initialize HuggingFace Hub client: {e}"))
            })?;

            let repo = client.model(owner, repo_name);

            let model_path = repo
                .download_file()
                .filename(model_file.to_string())
                .send()
                .map_err(|e| {
                    Error::Internal(format!(
                        "Failed to download model file '{model_file}' from '{repo_id}': {e}"
                    ))
                })?;

            let tokenizer_path = repo
                .download_file()
                .filename(tokenizer_file.to_string())
                .send()
                .map_err(|e| {
                    Error::Internal(format!(
                        "Failed to download tokenizer file '{tokenizer_file}' from '{repo_id}': {e}"
                    ))
                })?;

            Ok(ResolvedModel {
                model_path,
                tokenizer_path,
                name,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_config_resolves_directly() {
        let config = EmbeddingModelConfig::Local {
            model_path: "/some/path/model.onnx".into(),
            tokenizer_path: "/some/path/tokenizer.json".into(),
        };
        let resolved = resolve(&config).unwrap();
        assert_eq!(resolved.model_path, PathBuf::from("/some/path/model.onnx"));
        assert_eq!(
            resolved.tokenizer_path,
            PathBuf::from("/some/path/tokenizer.json")
        );
        assert_eq!(resolved.name, "model");
    }

    #[test]
    fn hub_repo_id_without_owner_is_rejected_before_network() {
        let config = EmbeddingModelConfig::HuggingFace {
            repo_id: "all-MiniLM-L6-v2".to_string(),
            model_file: "onnx/model.onnx".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
        };
        let Err(err) = resolve(&config) else {
            panic!("repo id without an owner must be rejected");
        };
        let message = err.to_string();
        assert!(
            message.contains("all-MiniLM-L6-v2") && message.contains("owner/name"),
            "error should name the repo id and the expected form, got: {message}"
        );
    }

    #[test]
    #[ignore = "requires network access (~23MB download on first run)"]
    fn preset_downloads_minilm_l6() {
        let config = EmbeddingModelConfig::MiniLmL6v2;
        let resolved = resolve(&config).unwrap();
        assert!(resolved.model_path.exists());
        assert!(resolved.tokenizer_path.exists());
        assert_eq!(resolved.name, "all-MiniLM-L6-v2");
    }
}
