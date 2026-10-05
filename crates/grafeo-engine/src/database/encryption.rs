//! The keys of an encrypted database, derived from `Config::encryption`.
//!
//! The key chain derives one key per database and component, with the
//! database id of the file header as the component id (its little-endian
//! bytes):
//!
//! - `"grafeo-container"` encrypts the chunks and directory blocks of the
//!   `.grafeo` file;
//! - `"grafeo-wal"` encrypts the records of its sidecar WAL (and of the WAL a
//!   restore writes next to a restored file).
//!
//! Two databases configured with one key chain therefore have different keys,
//! and so does a copy written by `save` or a migration, which gets a new
//! database id.

use grafeo_storage::file::v3::ChunkCipher;
#[cfg(feature = "wal")]
use grafeo_storage::wal::WalCipher;

use crate::config::Config;

/// Context of the key that encrypts the `.grafeo` file.
const CONTAINER_CONTEXT: &str = "grafeo-container";

/// Context of the key that encrypts the sidecar WAL.
#[cfg(feature = "wal")]
const WAL_CONTEXT: &str = "grafeo-wal";

/// The key chain of an encrypted database.
#[cfg(feature = "encryption")]
type KeyChainHandle = std::sync::Arc<grafeo_common::encryption::KeyChain>;

/// Without the `encryption` feature there is no key chain: the type has no
/// values, so a database is never encrypted.
#[cfg(not(feature = "encryption"))]
type KeyChainHandle = std::convert::Infallible;

/// Derives the ciphers of a database from the key chain of
/// `Config::encryption`. Without one the database is not encrypted.
pub(crate) struct DatabaseKeys {
    key_chain: Option<KeyChainHandle>,
}

impl DatabaseKeys {
    /// The keys `config` asks for.
    pub(crate) fn from_config(config: &Config) -> Self {
        #[cfg(feature = "encryption")]
        if let Some(encryption) = &config.encryption {
            return Self::from_encryption(encryption);
        }
        #[cfg(not(feature = "encryption"))]
        let _ = config;
        Self::none()
    }

    /// The keys of `encryption`'s key chain.
    #[cfg(feature = "encryption")]
    pub(crate) fn from_encryption(encryption: &crate::config::EncryptionConfig) -> Self {
        Self {
            key_chain: Some(std::sync::Arc::clone(&encryption.key_chain)),
        }
    }

    /// No keys: the database is not encrypted.
    pub(crate) fn none() -> Self {
        Self { key_chain: None }
    }

    /// Whether the database is to be encrypted.
    pub(crate) fn is_encrypted(&self) -> bool {
        self.key_chain.is_some()
    }

    /// The cipher of the `.grafeo` file of the database `database_id`, or
    /// `None` for a database without a key.
    pub(crate) fn container_cipher(&self, database_id: u128) -> Option<ChunkCipher> {
        self.key_chain
            .as_ref()
            .map(|chain| derive(chain, CONTAINER_CONTEXT, database_id))
    }

    /// The cipher of the sidecar WAL of the database `database_id`, or
    /// `None` for a database without a key.
    #[cfg(feature = "wal")]
    pub(crate) fn wal_cipher(&self, database_id: u128) -> Option<WalCipher> {
        self.key_chain
            .as_ref()
            .map(|chain| derive(chain, WAL_CONTEXT, database_id))
    }
}

/// The key `chain` derives for `context` and the database `database_id`.
#[cfg(feature = "encryption")]
fn derive(
    chain: &KeyChainHandle,
    context: &str,
    database_id: u128,
) -> grafeo_common::encryption::PageEncryptor {
    chain.encryptor_for(context, &database_id.to_le_bytes())
}

/// Without the `encryption` feature there is no key chain to derive from.
#[cfg(not(feature = "encryption"))]
fn derive<Cipher>(chain: &KeyChainHandle, _context: &str, _database_id: u128) -> Cipher {
    match *chain {}
}
