//! Encryption of WAL v2 segments.
//!
//! - **Segment key:** every segment has a random 32-byte salt in its header;
//!   its key comes from a [`CipherForSalt`] closure the engine supplies
//!   (the database's master key, its id and the salt). A fresh key per
//!   segment keeps the random 96-bit nonces far below GCM's limit of about
//!   2^32 messages per key.
//! - **Key check:** the header stores the nonce and tag of an empty message
//!   under the segment key, with header bytes 0..80 as associated data, so a
//!   wrong key is told apart from damage before any frame is read.
//! - **Frames:** a random 12-byte nonce per frame, and the associated data
//!   `"grafeo-wal-v2" || database id || frame header bytes 0..4 and 8..25`.
//!   It binds the length, LSN, transaction id and flags, so a frame cannot be
//!   moved, duplicated, swapped, cut at a flag or carried into another
//!   database's WAL. Nonces are random, not the frame LSN: a failed write
//!   cuts the log back and writes the same LSN again under the same key.
//! - **Checksum:** the frame CRC covers the ciphertext, so damage is found
//!   without the key, and a frame whose checksum is right but whose tag is
//!   not is reported as an authentication failure.

#![deny(clippy::let_underscore_must_use)]

use std::path::Path;
use std::sync::Arc;

use super::WalCipher;
use super::error::WalError;
use super::frame::FrameHeader;
#[cfg(feature = "encryption")]
use super::segment::KEY_CHECK_BYTES;
use super::segment::{KEY_CHECK_AAD_BYTES, SALT_BYTES, SegmentHeader};

/// Gives the cipher of a segment from the salt in its header. The engine
/// derives the segment key from the database's master key, its id and the
/// salt; the WAL never sees a key.
pub type CipherForSalt = Arc<dyn Fn(&[u8; SALT_BYTES]) -> WalCipher + Send + Sync>;

/// The first bytes of every frame's associated data.
const FRAME_AAD_PREFIX: &[u8; 13] = b"grafeo-wal-v2";

/// Size of a frame's associated data: the prefix, the database id and the
/// bound header bytes.
pub const FRAME_AAD_BYTES: usize = 13 + 16 + 21;

/// The associated data a frame is encrypted with:
/// `"grafeo-wal-v2" || database id (u128 LE) || header bytes 0..4 and 8..25`.
#[must_use]
pub fn frame_aad(database_id: u128, header: &FrameHeader) -> [u8; FRAME_AAD_BYTES] {
    let mut aad = [0u8; FRAME_AAD_BYTES];
    aad[..13].copy_from_slice(FRAME_AAD_PREFIX);
    aad[13..29].copy_from_slice(&database_id.to_le_bytes());
    aad[29..].copy_from_slice(&header.bound_bytes());
    aad
}

/// A random salt for a new segment.
#[cfg(feature = "encryption")]
pub(crate) fn new_salt() -> [u8; SALT_BYTES] {
    use grafeo_common::encryption::{NONCE_SIZE, random_nonce};
    let mut salt = [0u8; SALT_BYTES];
    for chunk in salt.chunks_mut(NONCE_SIZE) {
        chunk.copy_from_slice(&random_nonce()[..chunk.len()]);
    }
    salt
}

/// The key check of a segment header whose bytes 0..80 are `aad`: the nonce
/// and tag of an empty message under the segment key.
#[cfg(feature = "encryption")]
pub(crate) fn key_check(
    cipher: &WalCipher,
    aad: &[u8; KEY_CHECK_AAD_BYTES],
) -> Result<[u8; KEY_CHECK_BYTES], WalError> {
    key_check_with_nonce(cipher, aad, &grafeo_common::encryption::random_nonce())
}

/// [`key_check`] with a given nonce (for pinned test vectors).
#[cfg(feature = "encryption")]
fn key_check_with_nonce(
    cipher: &WalCipher,
    aad: &[u8; KEY_CHECK_AAD_BYTES],
    nonce: &[u8; grafeo_common::encryption::NONCE_SIZE],
) -> Result<[u8; KEY_CHECK_BYTES], WalError> {
    let sealed = cipher
        .encrypt(&[], nonce, aad)
        .map_err(|error| WalError::Encryption {
            reason: error.to_string(),
        })?;
    <[u8; KEY_CHECK_BYTES]>::try_from(sealed.as_slice()).map_err(|_| WalError::Encryption {
        reason: format!(
            "the key check is {} bytes, expected {KEY_CHECK_BYTES}",
            sealed.len()
        ),
    })
}

/// Whether `check` is the key check of the header bytes `aad` under
/// `cipher`'s key.
#[cfg(feature = "encryption")]
pub(crate) fn key_check_matches(
    cipher: &WalCipher,
    aad: &[u8; KEY_CHECK_AAD_BYTES],
    check: &[u8; KEY_CHECK_BYTES],
) -> bool {
    cipher
        .decrypt(check, aad)
        .is_ok_and(|plaintext| plaintext.is_empty())
}

/// The cipher of the segment at `path` whose header is `header`, checked
/// against the header's key check over `stored_aad`: the header bytes 0..80
/// as the file holds them ([`stored_key_check_aad`](super::segment::stored_key_check_aad)),
/// never rebuilt from `header`, which leaves out a compatible flag of a later
/// release that the key check authenticates.
///
/// # Errors
///
/// [`WalError::MissingKey`] for an encrypted segment without a key (or
/// without the `encryption` feature), [`WalError::NotEncrypted`] for a
/// plaintext segment of an encrypted database, and [`WalError::WrongKey`]
/// when the key check fails.
pub(crate) fn segment_cipher(
    cipher_for_salt: Option<&CipherForSalt>,
    header: &SegmentHeader,
    stored_aad: &[u8; KEY_CHECK_AAD_BYTES],
    path: &Path,
) -> Result<Option<Arc<WalCipher>>, WalError> {
    match (cipher_for_salt, header.encrypted, stored_aad) {
        (None, false, _) => Ok(None),
        (None, true, _) => Err(WalError::MissingKey {
            path: path.to_path_buf(),
        }),
        (Some(_), false, _) => Err(WalError::NotEncrypted {
            path: path.to_path_buf(),
        }),
        #[cfg(feature = "encryption")]
        (Some(cipher_for_salt), true, stored_aad) => {
            let cipher = cipher_for_salt(&header.salt);
            if key_check_matches(&cipher, stored_aad, &header.key_check) {
                Ok(Some(Arc::new(cipher)))
            } else {
                Err(WalError::WrongKey {
                    path: path.to_path_buf(),
                    database_id: header.database_id,
                })
            }
        }
        #[cfg(not(feature = "encryption"))]
        (Some(_), true, _) => Err(WalError::MissingKey {
            path: path.to_path_buf(),
        }),
    }
}

/// Encrypts a frame's plaintext payload with a random nonce:
/// `nonce || ciphertext || tag`.
#[cfg(feature = "encryption")]
pub(crate) fn seal_payload(
    cipher: &WalCipher,
    aad: &[u8; FRAME_AAD_BYTES],
    plaintext: &[u8],
) -> Result<Vec<u8>, WalError> {
    cipher
        .encrypt(plaintext, &grafeo_common::encryption::random_nonce(), aad)
        .map_err(|error| WalError::Encryption {
            reason: error.to_string(),
        })
}

/// Without the `encryption` feature no cipher exists, so this is never
/// called.
#[cfg(not(feature = "encryption"))]
pub(crate) fn seal_payload(
    cipher: &WalCipher,
    _aad: &[u8; FRAME_AAD_BYTES],
    _plaintext: &[u8],
) -> Result<Vec<u8>, WalError> {
    match *cipher {}
}

/// Decrypts a frame's payload, or `None` when it does not authenticate
/// under `cipher`'s key with `aad`.
#[cfg(feature = "encryption")]
pub(crate) fn open_payload(
    cipher: &WalCipher,
    aad: &[u8; FRAME_AAD_BYTES],
    sealed: &[u8],
) -> Option<Vec<u8>> {
    cipher.decrypt(sealed, aad).ok()
}

/// Without the `encryption` feature no cipher exists, so this is never
/// called.
#[cfg(not(feature = "encryption"))]
pub(crate) fn open_payload(
    cipher: &WalCipher,
    _aad: &[u8; FRAME_AAD_BYTES],
    _sealed: &[u8],
) -> Option<Vec<u8>> {
    match *cipher {}
}

#[cfg(test)]
mod tests {
    use super::super::frame::FrameFlags;
    use super::*;

    fn header() -> FrameHeader {
        FrameHeader::new(
            0x0102_0304,
            0x1112_1314_1516_1718,
            0x2122_2324_2526_2728,
            FrameFlags::LAST,
        )
    }

    /// The exact associated data of a known frame, as every encrypted frame
    /// written so far was sealed with: a change here makes those frames
    /// undecryptable.
    #[test]
    fn the_frame_associated_data_is_pinned() {
        let aad = frame_aad(0x3132_3334_3536_3738_393A_3B3C_3D3E_3F40, &header());
        let mut expected = b"grafeo-wal-v2".to_vec();
        expected.extend_from_slice(&[
            0x40, 0x3F, 0x3E, 0x3D, 0x3C, 0x3B, 0x3A, 0x39, 0x38, 0x37, 0x36, 0x35, 0x34, 0x33,
            0x32, 0x31,
        ]);
        expected.extend_from_slice(&[0x04, 0x03, 0x02, 0x01]); // length
        expected.extend_from_slice(&[0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11]); // LSN
        expected.extend_from_slice(&[0x28, 0x27, 0x26, 0x25, 0x24, 0x23, 0x22, 0x21]); // transaction
        expected.push(0x02); // flags: LAST
        assert_eq!(aad.to_vec(), expected);
    }

    #[cfg(feature = "encryption")]
    mod encrypted {
        use grafeo_common::encryption::PageEncryptor;

        use super::super::super::frame::FrameFlags;
        use super::super::*;
        use super::header;

        fn cipher(byte: u8) -> WalCipher {
            PageEncryptor::new(&[byte; 32])
        }

        /// The key check of a known key, header and nonce: a change here
        /// makes every encrypted segment written so far report a wrong key.
        #[test]
        fn the_key_check_bytes_are_pinned() {
            let aad = [0x19u8; KEY_CHECK_AAD_BYTES];
            let check = key_check_with_nonce(&cipher(3), &aad, &[0x88; 12]).unwrap();
            assert_eq!(&check[..12], &[0x88; 12], "the nonce comes first");
            assert_eq!(
                check[12..].to_vec(),
                [
                    0xE3, 0xCE, 0xF0, 0xFF, 0xF8, 0x0A, 0x5D, 0x15, 0xDF, 0x49, 0x9D, 0x27, 0xA3,
                    0xAD, 0xF1, 0xA9
                ],
                "the tag of the empty message: {:02X?}",
                &check[12..]
            );
            assert!(key_check_matches(&cipher(3), &aad, &check));
        }

        #[test]
        fn the_key_check_tells_a_wrong_key_and_another_header_apart() {
            let aad = [0x03u8; KEY_CHECK_AAD_BYTES];
            let check = key_check(&cipher(3), &aad).unwrap();
            assert!(key_check_matches(&cipher(3), &aad, &check));
            assert!(!key_check_matches(&cipher(19), &aad, &check), "a wrong key");
            let mut other = aad;
            other[40] ^= 1;
            assert!(
                !key_check_matches(&cipher(3), &other, &check),
                "the header bytes are bound"
            );
        }

        #[test]
        fn a_sealed_payload_opens_only_with_its_own_header_and_database() {
            let header = header();
            let aad = frame_aad(88, &header);
            let sealed = seal_payload(&cipher(3), &aad, b"Vincent in Paris").unwrap();
            assert_eq!(
                open_payload(&cipher(3), &aad, &sealed).as_deref(),
                Some(&b"Vincent in Paris"[..])
            );
            let moved = FrameHeader {
                lsn: header.lsn + 25,
                ..header
            };
            let other_transaction = FrameHeader {
                transaction_id: 19,
                ..header
            };
            let other_flags = FrameHeader {
                flags: FrameFlags::FIRST_AND_LAST,
                ..header
            };
            let other_length = FrameHeader {
                length: header.length + 1,
                ..header
            };
            for (what, other) in [
                ("another LSN", frame_aad(88, &moved)),
                ("another transaction", frame_aad(88, &other_transaction)),
                ("other flags", frame_aad(88, &other_flags)),
                ("another length", frame_aad(88, &other_length)),
                ("another database", frame_aad(319, &header)),
            ] {
                assert!(
                    open_payload(&cipher(3), &other, &sealed).is_none(),
                    "{what} must not open the frame"
                );
            }
            assert!(
                open_payload(&cipher(19), &aad, &sealed).is_none(),
                "a wrong key"
            );
        }

        #[test]
        fn every_seal_and_salt_is_fresh() {
            let aad = frame_aad(3, &header());
            let first = seal_payload(&cipher(3), &aad, b"Mia").unwrap();
            let second = seal_payload(&cipher(3), &aad, b"Mia").unwrap();
            assert_ne!(
                first, second,
                "random nonces: the same frame seals differently"
            );
            assert_ne!(new_salt(), new_salt());
            assert!(new_salt().iter().any(|&byte| byte != 0));
        }
    }
}
