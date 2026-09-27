//! Checked size conversions for on-disk formats.
//!
//! Section serializers store sizes, offsets and counts in fixed-width fields
//! (`u32`, `u16`). A plain `as` cast silently wraps a value that does not fit,
//! which writes a file that cannot be read back (for example an LPG section
//! over 4 GiB, issue #392). These helpers turn that into a serialization error
//! before anything is written.

use grafeo_common::utils::error::{Error, Result};

/// Converts `value` to `u32`, or fails naming `what` and the limit.
///
/// # Errors
///
/// Returns [`Error::Serialization`] if `value` exceeds `u32::MAX`.
pub fn checked_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| too_large(value, what, u64::from(u32::MAX)))
}

/// Converts `value` to `u16`, or fails naming `what` and the limit.
///
/// # Errors
///
/// Returns [`Error::Serialization`] if `value` exceeds `u16::MAX`.
pub fn checked_u16(value: usize, what: &str) -> Result<u16> {
    u16::try_from(value).map_err(|_| too_large(value, what, u64::from(u16::MAX)))
}

fn too_large(value: usize, what: &str, max: u64) -> Error {
    Error::Serialization(format!(
        "cannot write {what}: {value} exceeds the storage format limit of {max}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_at_the_limit_convert() {
        assert_eq!(
            checked_u16(usize::from(u16::MAX), "count").unwrap(),
            u16::MAX
        );
        assert_eq!(checked_u32(0, "offset").unwrap(), 0);
        #[cfg(target_pointer_width = "64")]
        assert_eq!(checked_u32(u32::MAX as usize, "offset").unwrap(), u32::MAX);
    }

    #[test]
    fn values_over_the_limit_fail_with_a_named_error() {
        let err = checked_u16(usize::from(u16::MAX) + 1, "LPG block count").unwrap_err();
        assert!(matches!(err, Error::Serialization(_)));
        assert!(
            err.to_string().contains(
                "cannot write LPG block count: 65536 exceeds the storage format limit of 65535"
            ),
            "{err}"
        );

        #[cfg(target_pointer_width = "64")]
        {
            let err = checked_u32(u32::MAX as usize + 1, "LPG block offset").unwrap_err();
            assert!(
                err.to_string()
                    .contains("LPG block offset: 4294967296 exceeds the storage format limit"),
                "{err}"
            );
        }
    }
}
