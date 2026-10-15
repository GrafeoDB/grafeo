//! The keys of an encrypted database, derived from `Config::encryption`.
//!
//! The key chain derives one key per database and component, with the
//! database id of the file header as the component id (its little-endian
//! bytes):
//!
//! - `"grafeo-container"` encrypts the chunks and directory blocks of the
//!   `.grafeo` file;
//! - `"grafeo-wal"` encrypts the records of its sidecar WAL as 0.5.x and
//!   0.6.0-beta.1 wrote it (and of the WAL a restore of their backups writes);
//! - `"grafeo-wal-segment"` encrypts the frames of a WAL segment, one key per
//!   segment: the component id is the database id followed by the segment's
//!   salt (32 bytes, from its header).
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

/// Context of the key that encrypts the sidecar WAL of the first log format.
#[cfg(feature = "wal")]
const WAL_CONTEXT: &str = "grafeo-wal";
/// Context of the keys that encrypt WAL segments, one per segment salt.
#[cfg(all(feature = "wal", feature = "encryption"))]
const WAL_SEGMENT_CONTEXT: &str = "grafeo-wal-segment";

/// The key chain of an encrypted database.
#[cfg(feature = "encryption")]
type KeyChainHandle = std::sync::Arc<grafeo_common::encryption::KeyChain>;

/// Without the `encryption` feature there is no key chain: the type has no
/// values, so a database is never encrypted. (Not `Infallible`, which is
/// `Copy`: the handle is passed by reference as with the feature.)
#[cfg(not(feature = "encryption"))]
enum KeyChainHandle {}

/// Derives the ciphers of a database from the key chain of
/// `Config::encryption`. Without one the database is not encrypted.
pub(crate) struct DatabaseKeys {
    key_chain: Option<KeyChainHandle>,
}

impl DatabaseKeys {
    /// The keys `config` asks for.
    pub(crate) fn from_config(config: &Config) -> Self {
        #[cfg(feature = "encryption")]
        if let Some(encryption) = &config.encryption() {
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

    /// The ciphers of the WAL segments of the database `database_id`, one per
    /// segment salt, or `None` for a database without a key.
    #[cfg(feature = "wal")]
    pub(crate) fn wal_segment_ciphers(
        &self,
        database_id: u128,
    ) -> Option<grafeo_storage::wal::CipherForSalt> {
        #[cfg(feature = "encryption")]
        {
            let chain = std::sync::Arc::clone(self.key_chain.as_ref()?);
            Some(std::sync::Arc::new(
                move |salt: &[u8; grafeo_storage::wal::SALT_BYTES]| {
                    let mut component = database_id.to_le_bytes().to_vec();
                    component.extend_from_slice(salt);
                    chain.encryptor_for(WAL_SEGMENT_CONTEXT, &component)
                },
            ))
        }
        #[cfg(not(feature = "encryption"))]
        {
            let _ = database_id;
            self.key_chain.as_ref().map(|chain| match *chain {})
        }
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
