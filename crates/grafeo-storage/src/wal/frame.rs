//! WAL v2 frames: the 25-byte header in front of every payload.
//!
//! | Offset | Field |
//! | --- | --- |
//! | 0 | length u32: payload bytes, after encryption |
//! | 4 | crc32 u32 over bytes 0..4 and 8..25 and the payload (ciphertext when encrypted) |
//! | 8 | lsn u64: the frame's own log position |
//! | 16 | transaction id u64: the same in every frame of a group |
//! | 24 | flags u8: bit 0 FIRST, bit 1 LAST; any other bit is refused |
//! | 25 | payload |
//!
//! A transaction is one group of frames: FIRST, then middle frames, then
//! LAST (a one-frame group is FIRST|LAST). The plaintext payload of a frame
//! is a run of whole records; the FIRST frame's payload starts with the
//! 8-byte prologue `synced_lsn u64`, the WAL's durable LSN when the group
//! began. An encrypted payload is `nonce(12) || ciphertext || tag(16)`.

#![deny(clippy::let_underscore_must_use)]

/// Size of a frame header in bytes.
pub const FRAME_HEADER_BYTES: usize = 25;

/// The plaintext payload size the writer packs records up to.
pub const FRAME_TARGET: usize = 64 * 1024;

/// The largest payload a frame may declare: a reader refuses a longer one
/// before it allocates anything.
pub const MAX_FRAME_PAYLOAD: u32 = 1 << 30;

/// Size of the prologue at the start of a FIRST frame's plaintext payload.
pub const FRAME_PROLOGUE_BYTES: usize = 8;

/// Bytes an encrypted payload adds to its plaintext: the nonce and the tag.
pub const SEALED_OVERHEAD: usize = 12 + 16;

/// The largest record the writer accepts: one that fits in a FIRST frame of
/// its own, encrypted or not.
pub const MAX_FRAME_RECORD_BYTES: usize =
    MAX_FRAME_PAYLOAD as usize - FRAME_PROLOGUE_BYTES - SEALED_OVERHEAD;

/// The flags of a frame: where it sits in its group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FrameFlags(u8);

impl FrameFlags {
    /// A middle frame: neither the first nor the last of its group.
    pub const MIDDLE: Self = Self(0);
    /// The first frame of a group.
    pub const FIRST: Self = Self(1);
    /// The last frame of a group: the group's commit marker.
    pub const LAST: Self = Self(2);
    /// Both: a group of one frame.
    pub const FIRST_AND_LAST: Self = Self(3);

    /// The flags byte as stored.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// The flags of a stored byte, or `None` when a bit other than FIRST and
    /// LAST is set.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Option<Self> {
        if bits & !Self::FIRST_AND_LAST.0 == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    /// Whether the frame starts its group.
    #[must_use]
    pub const fn is_first(self) -> bool {
        self.0 & Self::FIRST.0 != 0
    }

    /// Whether the frame ends its group.
    #[must_use]
    pub const fn is_last(self) -> bool {
        self.0 & Self::LAST.0 != 0
    }

    /// The flags of both.
    #[must_use]
    pub const fn with(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// The header in front of a frame's payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    /// Bytes of payload after the header (after encryption).
    pub length: u32,
    /// CRC-32 over the header bytes 0..4 and 8..25 and the payload.
    pub crc: u32,
    /// The log position of the frame: its byte position in the stream of
    /// frames, segment headers not counted.
    pub lsn: u64,
    /// The transaction whose group the frame belongs to.
    pub transaction_id: u64,
    /// Where the frame sits in its group.
    pub flags: FrameFlags,
}

impl FrameHeader {
    /// A header for a payload of `length` bytes, without its checksum yet
    /// (see [`with_checksum`](Self::with_checksum)).
    #[must_use]
    pub const fn new(length: u32, lsn: u64, transaction_id: u64, flags: FrameFlags) -> Self {
        Self {
            length,
            crc: 0,
            lsn,
            transaction_id,
            flags,
        }
    }

    /// The header bytes the checksum and the encryption bind: bytes 0..4
    /// (length) and 8..25 (LSN, transaction id, flags).
    #[must_use]
    pub fn bound_bytes(&self) -> [u8; FRAME_HEADER_BYTES - 4] {
        let mut bytes = [0u8; FRAME_HEADER_BYTES - 4];
        bytes[0..4].copy_from_slice(&self.length.to_le_bytes());
        bytes[4..12].copy_from_slice(&self.lsn.to_le_bytes());
        bytes[12..20].copy_from_slice(&self.transaction_id.to_le_bytes());
        bytes[20] = self.flags.bits();
        bytes
    }

    /// The checksum of this header with `payload` (ciphertext when
    /// encrypted): CRC-32 over [`bound_bytes`](Self::bound_bytes) and the
    /// payload, so damage is found without a key.
    #[must_use]
    pub fn checksum(&self, payload: &[u8]) -> u32 {
        let bound = self.bound_bytes();
        bound_checksum(&bound[..4], &bound[4..], payload)
    }

    /// Whether the checksum stored in the 25 header bytes `bytes` matches
    /// them and `payload`, whatever their flags byte holds. A reader checks
    /// a frame this way before it decodes the flags, so a whole frame with a
    /// flag of a later release is told apart from a damaged one.
    #[must_use]
    pub fn stored_checksum_matches(bytes: &[u8; FRAME_HEADER_BYTES], payload: &[u8]) -> bool {
        let stored = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        stored == bound_checksum(&bytes[..4], &bytes[8..], payload)
    }

    /// This header with the checksum of `payload` filled in.
    #[must_use]
    pub fn with_checksum(mut self, payload: &[u8]) -> Self {
        self.crc = self.checksum(payload);
        self
    }

    /// Whether the stored checksum matches `payload`.
    #[must_use]
    pub fn checksum_matches(&self, payload: &[u8]) -> bool {
        self.crc == self.checksum(payload)
    }

    /// Encodes the header into its 25 bytes.
    #[must_use]
    pub fn encode(&self) -> [u8; FRAME_HEADER_BYTES] {
        let mut bytes = [0u8; FRAME_HEADER_BYTES];
        bytes[0..4].copy_from_slice(&self.length.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.crc.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.lsn.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.transaction_id.to_le_bytes());
        bytes[24] = self.flags.bits();
        bytes
    }

    /// Decodes a frame header from its 25 bytes, refusing unknown flags and
    /// a length over [`MAX_FRAME_PAYLOAD`], so a reader never allocates for
    /// a declared length it would refuse. The checksum is not checked here:
    /// it covers the payload too.
    ///
    /// # Errors
    ///
    /// Returns what is wrong with the header.
    pub fn decode(bytes: &[u8; FRAME_HEADER_BYTES]) -> Result<Self, String> {
        let length = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if length > MAX_FRAME_PAYLOAD {
            return Err(format!(
                "the frame declares a payload of {length} bytes, over the limit of \
                 {MAX_FRAME_PAYLOAD}"
            ));
        }
        let flags = FrameFlags::from_bits(bytes[24])
            .ok_or_else(|| format!("the frame sets unknown flags {:#04x}", bytes[24]))?;
        let mut crc = [0u8; 4];
        crc.copy_from_slice(&bytes[4..8]);
        let mut lsn = [0u8; 8];
        lsn.copy_from_slice(&bytes[8..16]);
        let mut transaction_id = [0u8; 8];
        transaction_id.copy_from_slice(&bytes[16..24]);
        Ok(Self {
            length,
            crc: u32::from_le_bytes(crc),
            lsn: u64::from_le_bytes(lsn),
            transaction_id: u64::from_le_bytes(transaction_id),
            flags,
        })
    }

    /// Bytes the frame takes in the log: header and payload.
    #[must_use]
    pub fn frame_bytes(&self) -> u64 {
        FRAME_HEADER_BYTES as u64 + u64::from(self.length)
    }
}

/// CRC-32 over a frame's length bytes, its LSN, transaction and flags bytes,
/// and its payload.
fn bound_checksum(length: &[u8], rest: &[u8], payload: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(length);
    hasher.update(rest);
    hasher.update(payload);
    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known_header() -> FrameHeader {
        FrameHeader::new(
            5,
            0x0102_0304_0506_0708,
            0x1112_1314_1516_1718,
            FrameFlags::FIRST_AND_LAST,
        )
        .with_checksum(b"Alix!")
    }

    /// The exact bytes of a known frame, as every frame written so far
    /// carries them: a change here makes those frames unreadable.
    #[test]
    fn the_frame_header_bytes_are_pinned() {
        let header = known_header();
        let mut bound = Vec::new();
        bound.extend_from_slice(&[5, 0, 0, 0]);
        bound.extend_from_slice(&[0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]);
        bound.extend_from_slice(&[0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11]);
        bound.push(0x03);
        let mut covered = bound.clone();
        covered.extend_from_slice(b"Alix!");
        let crc = crc32fast::hash(&covered);
        assert_eq!(
            header.crc, crc,
            "the checksum covers bytes 0..4, 8..25 and the payload"
        );
        assert_eq!(
            crc, 0xF45B_48E1,
            "the checksum of the known frame: {crc:#010x}"
        );
        let mut expected = vec![5, 0, 0, 0];
        expected.extend_from_slice(&crc.to_le_bytes());
        expected.extend_from_slice(&bound[4..]);
        assert_eq!(header.encode().to_vec(), expected);
        assert_eq!(header.bound_bytes().to_vec(), bound);
    }

    #[test]
    fn a_header_round_trips_and_its_checksum_covers_every_byte() {
        let header = known_header();
        let bytes = header.encode();
        assert_eq!(FrameHeader::decode(&bytes).unwrap(), header);
        assert!(header.checksum_matches(b"Alix!"));
        assert!(!header.checksum_matches(b"Alix?"), "the payload is covered");
        for index in (0..4).chain(8..FRAME_HEADER_BYTES) {
            let mut damaged = bytes;
            damaged[index] ^= 0x01;
            if let Ok(decoded) = FrameHeader::decode(&damaged) {
                assert!(
                    !decoded.checksum_matches(b"Alix!"),
                    "a change of header byte {index} is found by the checksum"
                );
            }
        }
    }

    /// The stored checksum is checked over the bytes as stored, an unknown
    /// flags byte included, and agrees with the decoded header's checksum.
    #[test]
    fn the_stored_checksum_covers_the_flags_byte_as_stored() {
        let header = known_header();
        let bytes = header.encode();
        assert!(FrameHeader::stored_checksum_matches(&bytes, b"Alix!"));
        assert!(!FrameHeader::stored_checksum_matches(&bytes, b"Alix?"));
        for index in 0..FRAME_HEADER_BYTES {
            let mut damaged = bytes;
            damaged[index] ^= 0x10;
            assert!(
                !FrameHeader::stored_checksum_matches(&damaged, b"Alix!"),
                "header byte {index} is covered"
            );
        }
        // A flag of a later release, with the checksum over it.
        let mut later = bytes;
        later[24] |= 0x04;
        let mut covered = later[..4].to_vec();
        covered.extend_from_slice(&later[8..]);
        covered.extend_from_slice(b"Alix!");
        later[4..8].copy_from_slice(&crc32fast::hash(&covered).to_le_bytes());
        assert!(FrameHeader::stored_checksum_matches(&later, b"Alix!"));
        assert!(FrameHeader::decode(&later).is_err(), "the flag is unknown");
    }

    #[test]
    fn a_frame_length_over_the_limit_is_refused_before_allocation() {
        let mut bytes = known_header().encode();
        bytes[0..4].copy_from_slice(&(MAX_FRAME_PAYLOAD + 1).to_le_bytes());
        let error = FrameHeader::decode(&bytes).unwrap_err();
        assert!(error.contains("over the limit"), "{error}");
        bytes[0..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(FrameHeader::decode(&bytes).is_err());
        bytes[0..4].copy_from_slice(&MAX_FRAME_PAYLOAD.to_le_bytes());
        assert_eq!(
            FrameHeader::decode(&bytes).unwrap().length,
            MAX_FRAME_PAYLOAD,
            "the limit itself is allowed"
        );
    }

    #[test]
    fn unknown_frame_flags_are_refused() {
        for bits in 4..=u8::MAX {
            let mut bytes = known_header().encode();
            bytes[24] = bits;
            let error = FrameHeader::decode(&bytes).unwrap_err();
            assert!(error.contains("unknown flags"), "{bits:#04x}: {error}");
        }
        for (bits, first, last) in [
            (0, false, false),
            (1, true, false),
            (2, false, true),
            (3, true, true),
        ] {
            let flags = FrameFlags::from_bits(bits).unwrap();
            assert_eq!((flags.is_first(), flags.is_last()), (first, last), "{bits}");
        }
        assert_eq!(
            FrameFlags::FIRST.with(FrameFlags::LAST),
            FrameFlags::FIRST_AND_LAST
        );
    }

    #[test]
    fn a_record_at_the_limit_fits_an_encrypted_first_frame() {
        let payload = MAX_FRAME_RECORD_BYTES + FRAME_PROLOGUE_BYTES + SEALED_OVERHEAD;
        assert_eq!(u32::try_from(payload).unwrap(), MAX_FRAME_PAYLOAD);
        #[cfg(feature = "encryption")]
        assert_eq!(
            SEALED_OVERHEAD,
            grafeo_common::encryption::ENCRYPTION_OVERHEAD,
            "the overhead is that of the cipher"
        );
    }
}
