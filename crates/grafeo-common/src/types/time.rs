//! Time of day type with optional UTC offset.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

/// Maximum nanoseconds in a day (exclusive).
const NANOS_PER_DAY: u64 = 86_400_000_000_000;
const NANOS_PER_HOUR: u64 = 3_600_000_000_000;
const NANOS_PER_MINUTE: u64 = 60_000_000_000;
const NANOS_PER_SECOND: u64 = 1_000_000_000;

/// A time of day with optional UTC offset.
///
/// Stored as nanoseconds since midnight plus an optional UTC offset in seconds.
/// Without an offset, this is a "local time."
///
/// # Examples
///
/// ```
/// use grafeo_common::types::Time;
///
/// let t = Time::from_hms(14, 30, 0).unwrap();
/// assert_eq!(t.hour(), 14);
/// assert_eq!(t.minute(), 30);
/// assert_eq!(t.to_string(), "14:30:00");
///
/// let tz = t.with_offset(3600); // +01:00
/// assert_eq!(tz.to_string(), "14:30:00+01:00");
/// ```
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Time {
    /// Nanoseconds since midnight (0..86_400_000_000_000).
    nanos: u64,
    /// UTC offset in seconds, or None for local time.
    offset: Option<i32>,
}

impl Time {
    /// Creates a time from hours (0-23), minutes (0-59), and seconds (0-59).
    #[must_use]
    pub fn from_hms(hour: u32, min: u32, sec: u32) -> Option<Self> {
        Self::from_hms_nano(hour, min, sec, 0)
    }

    /// Creates a time from hours, minutes, seconds, and nanoseconds.
    #[must_use]
    pub fn from_hms_nano(hour: u32, min: u32, sec: u32, nano: u32) -> Option<Self> {
        if hour >= 24 || min >= 60 || sec >= 60 || nano >= 1_000_000_000 {
            return None;
        }
        let nanos = hour as u64 * NANOS_PER_HOUR
            + min as u64 * NANOS_PER_MINUTE
            + sec as u64 * NANOS_PER_SECOND
            + nano as u64;
        Some(Self {
            nanos,
            offset: None,
        })
    }

    /// Creates a time from nanoseconds since midnight.
    #[must_use]
    pub fn from_nanos(nanos: u64) -> Option<Self> {
        if nanos >= NANOS_PER_DAY {
            return None;
        }
        Some(Self {
            nanos,
            offset: None,
        })
    }

    /// Returns a new Time with the given UTC offset in seconds.
    #[must_use]
    pub fn with_offset(self, offset_secs: i32) -> Self {
        Self {
            nanos: self.nanos,
            offset: Some(offset_secs),
        }
    }

    /// Returns the hour component (0-23).
    #[must_use]
    pub fn hour(&self) -> u32 {
        (self.nanos / NANOS_PER_HOUR) as u32
    }

    /// Returns the minute component (0-59).
    #[must_use]
    pub fn minute(&self) -> u32 {
        ((self.nanos % NANOS_PER_HOUR) / NANOS_PER_MINUTE) as u32
    }

    /// Returns the second component (0-59).
    #[must_use]
    pub fn second(&self) -> u32 {
        ((self.nanos % NANOS_PER_MINUTE) / NANOS_PER_SECOND) as u32
    }

    /// Returns the nanosecond component (0-999_999_999).
    #[must_use]
    pub fn nanosecond(&self) -> u32 {
        (self.nanos % NANOS_PER_SECOND) as u32
    }

    /// Returns the total nanoseconds since midnight.
    #[must_use]
    pub fn as_nanos(&self) -> u64 {
        self.nanos
    }

    /// Returns the UTC offset in seconds, if present.
    #[must_use]
    pub fn offset_seconds(&self) -> Option<i32> {
        self.offset
    }

    /// Parses a time from ISO 8601 format `HH:MM:SS[.nnn][+HH:MM|Z]`.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        // Split off timezone suffix
        let (time_part, offset) = parse_offset_suffix(s);

        let parts: Vec<&str> = time_part.splitn(2, '.').collect();
        let hms: Vec<&str> = parts[0].splitn(3, ':').collect();
        if hms.len() < 2 {
            return None;
        }

        let hour: u32 = hms[0].parse().ok()?;
        let min: u32 = hms[1].parse().ok()?;
        let sec: u32 = if hms.len() == 3 {
            hms[2].parse().ok()?
        } else {
            0
        };

        // Parse fractional seconds
        let nano: u32 = if parts.len() == 2 {
            let frac = parts[1];
            // Pad or truncate to 9 digits
            let padded = if frac.len() >= 9 {
                &frac[..9]
            } else {
                // We need to pad, but can't modify the slice, so parse differently
                return {
                    let n: u32 = frac.parse().ok()?;
                    // reason: frac.len() is at most 8 here (< 9 branch), so 9 - len fits u32
                    #[allow(clippy::cast_possible_truncation)]
                    let scale = 10u32.pow(9 - frac.len() as u32);
                    let mut t = Self::from_hms_nano(hour, min, sec, n * scale)?;
                    if let Some(off) = offset {
                        t = t.with_offset(off);
                    }
                    Some(t)
                };
            };
            padded.parse().ok()?
        } else {
            0
        };

        let mut t = Self::from_hms_nano(hour, min, sec, nano)?;
        if let Some(off) = offset {
            t = t.with_offset(off);
        }
        Some(t)
    }

    /// Returns the current local time (UTC).
    #[must_use]
    pub fn now() -> Self {
        let ts = super::Timestamp::now();
        ts.to_time()
    }

    /// Adds a duration's time component to this time.
    ///
    /// Only the nanosecond component of the duration is used (months and days
    /// are not meaningful for time-of-day). The result wraps around at midnight.
    #[must_use]
    pub fn add_duration(self, dur: &super::Duration) -> Self {
        // reason: nanos < NANOS_PER_DAY (86.4e12), well within i64 range
        #[allow(clippy::cast_possible_wrap)]
        let total = self.nanos as i64 + dur.nanos();
        // reason: NANOS_PER_DAY (86.4e12) is well within i64::MAX
        #[allow(clippy::cast_possible_wrap)]
        let wrapped = total.rem_euclid(NANOS_PER_DAY as i64) as u64;
        Self {
            nanos: wrapped,
            offset: self.offset,
        }
    }

    /// Truncates this time to the given unit, preserving the offset.
    ///
    /// - `"hour"`: zeros minutes, seconds, and nanoseconds
    /// - `"minute"`: zeros seconds and nanoseconds
    /// - `"second"`: zeros nanoseconds
    #[must_use]
    pub fn truncate(&self, unit: &str) -> Option<Self> {
        let truncated_nanos = match unit {
            "hour" => (self.nanos / NANOS_PER_HOUR) * NANOS_PER_HOUR,
            "minute" => (self.nanos / NANOS_PER_MINUTE) * NANOS_PER_MINUTE,
            "second" => (self.nanos / NANOS_PER_SECOND) * NANOS_PER_SECOND,
            _ => return None,
        };
        Some(Self {
            nanos: truncated_nanos,
            offset: self.offset,
        })
    }

    /// Returns UTC-normalized nanoseconds (for comparison).
    fn utc_nanos(&self) -> u64 {
        match self.offset {
            Some(off) => {
                // reason: nanos < NANOS_PER_DAY and constants are well within i64 range
                #[allow(clippy::cast_possible_wrap)]
                let adjusted = self.nanos as i64 - off as i64 * NANOS_PER_SECOND as i64;
                // reason: rem_euclid result is non-negative and < NANOS_PER_DAY, fits u64
                #[allow(clippy::cast_possible_wrap)]
                let result = adjusted.rem_euclid(NANOS_PER_DAY as i64) as u64;
                result
            }
            None => self.nanos,
        }
    }
}

impl Default for Time {
    fn default() -> Self {
        Self {
            nanos: 0,
            offset: None,
        }
    }
}

// Equality, ordering and hashing agree: two times with offsets compare by
// their UTC instant (`14:00+01:00` equals `13:00Z`, like `ZonedDatetime`);
// otherwise by wall-clock time, with the offset breaking ties so a local time
// never equals an offset time.

impl PartialEq for Time {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Time {}

impl Hash for Time {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.offset {
            Some(_) => (true, self.utc_nanos()).hash(state),
            None => (false, self.nanos).hash(state),
        }
    }
}

impl Ord for Time {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.offset, other.offset) {
            (Some(_), Some(_)) => self.utc_nanos().cmp(&other.utc_nanos()),
            _ => self
                .nanos
                .cmp(&other.nanos)
                .then_with(|| self.offset.is_some().cmp(&other.offset.is_some())),
        }
    }
}

impl PartialOrd for Time {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Parses an optional timezone offset suffix from a time string.
/// Returns (time_part, offset_in_seconds).
fn parse_offset_suffix(s: &str) -> (&str, Option<i32>) {
    if let Some(rest) = s.strip_suffix('Z') {
        return (rest, Some(0));
    }
    // Look for +HH:MM or -HH:MM at the end
    if s.len() >= 6 {
        let sign_pos = s.len() - 6;
        let candidate = &s[sign_pos..];
        if (candidate.starts_with('+') || candidate.starts_with('-'))
            && candidate.as_bytes()[3] == b':'
        {
            let sign: i32 = if candidate.starts_with('+') { 1 } else { -1 };
            if let (Ok(h), Ok(m)) = (
                candidate[1..3].parse::<i32>(),
                candidate[4..6].parse::<i32>(),
            ) {
                return (&s[..sign_pos], Some(sign * (h * 3600 + m * 60)));
            }
        }
    }
    (s, None)
}

impl fmt::Debug for Time {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Time({})", self)
    }
}

impl fmt::Display for Time {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let h = self.hour();
        let m = self.minute();
        let s = self.second();
        let ns = self.nanosecond();

        if ns > 0 {
            // Trim trailing zeros from fractional part
            let frac = format!("{:09}", ns);
            let trimmed = frac.trim_end_matches('0');
            write!(f, "{h:02}:{m:02}:{s:02}.{trimmed}")?;
        } else {
            write!(f, "{h:02}:{m:02}:{s:02}")?;
        }

        match self.offset {
            Some(0) => write!(f, "Z"),
            Some(off) => {
                let sign = if off >= 0 { '+' } else { '-' };
                let abs = off.unsigned_abs();
                let oh = abs / 3600;
                let om = (abs % 3600) / 60;
                write!(f, "{sign}{oh:02}:{om:02}")
            }
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn equal_instants_with_different_offsets_are_equal() {
        use std::collections::hash_map::DefaultHasher;
        let hash = |t: &Time| {
            let mut h = DefaultHasher::new();
            t.hash(&mut h);
            h.finish()
        };
        let amsterdam = Time::from_hms(14, 0, 0).unwrap().with_offset(3600);
        let utc = Time::from_hms(13, 0, 0).unwrap().with_offset(0);
        assert_eq!(amsterdam.cmp(&utc), Ordering::Equal);
        assert_eq!(amsterdam, utc);
        assert_eq!(hash(&amsterdam), hash(&utc));

        // A local time never equals an offset time, even at the same clock time.
        let local = Time::from_hms(14, 0, 0).unwrap();
        assert_ne!(local, amsterdam);
        assert_ne!(local.cmp(&amsterdam), Ordering::Equal);
        assert_eq!(local, Time::from_hms(14, 0, 0).unwrap());
    }
    use super::*;

    #[test]
    fn test_basic() {
        let t = Time::from_hms(14, 30, 45).unwrap();
        assert_eq!(t.hour(), 14);
        assert_eq!(t.minute(), 30);
        assert_eq!(t.second(), 45);
        assert_eq!(t.nanosecond(), 0);
    }

    #[test]
    fn test_with_nanos() {
        let t = Time::from_hms_nano(0, 0, 0, 123_456_789).unwrap();
        assert_eq!(t.nanosecond(), 123_456_789);
        assert_eq!(t.to_string(), "00:00:00.123456789");
    }

    #[test]
    fn test_validation() {
        assert!(Time::from_hms(24, 0, 0).is_none());
        assert!(Time::from_hms(0, 60, 0).is_none());
        assert!(Time::from_hms(0, 0, 60).is_none());
        assert!(Time::from_hms_nano(0, 0, 0, 1_000_000_000).is_none());
    }

    #[test]
    fn test_parse_basic() {
        let t = Time::parse("14:30:00").unwrap();
        assert_eq!(t.hour(), 14);
        assert_eq!(t.minute(), 30);
        assert_eq!(t.second(), 0);
        assert!(t.offset_seconds().is_none());
    }

    #[test]
    fn test_parse_with_offset() {
        let t = Time::parse("14:30:00+02:00").unwrap();
        assert_eq!(t.hour(), 14);
        assert_eq!(t.offset_seconds(), Some(7200));

        let t = Time::parse("14:30:00Z").unwrap();
        assert_eq!(t.offset_seconds(), Some(0));

        let t = Time::parse("14:30:00-05:30").unwrap();
        assert_eq!(t.offset_seconds(), Some(-19800));
    }

    #[test]
    fn test_parse_fractional() {
        let t = Time::parse("14:30:00.5").unwrap();
        assert_eq!(t.nanosecond(), 500_000_000);

        let t = Time::parse("14:30:00.123").unwrap();
        assert_eq!(t.nanosecond(), 123_000_000);
    }

    #[test]
    fn test_display() {
        assert_eq!(Time::from_hms(9, 5, 3).unwrap().to_string(), "09:05:03");
        assert_eq!(
            Time::from_hms(14, 30, 0)
                .unwrap()
                .with_offset(0)
                .to_string(),
            "14:30:00Z"
        );
        assert_eq!(
            Time::from_hms(14, 30, 0)
                .unwrap()
                .with_offset(5 * 3600 + 30 * 60)
                .to_string(),
            "14:30:00+05:30"
        );
    }

    #[test]
    fn test_ordering() {
        let t1 = Time::from_hms(10, 0, 0).unwrap();
        let t2 = Time::from_hms(14, 0, 0).unwrap();
        assert!(t1 < t2);
    }

    #[test]
    fn test_ordering_with_offsets() {
        // 14:00 UTC+2 = 12:00 UTC
        // 13:00 UTC+0 = 13:00 UTC
        // So the UTC+2 time is earlier in UTC
        let t1 = Time::from_hms(14, 0, 0).unwrap().with_offset(7200);
        let t2 = Time::from_hms(13, 0, 0).unwrap().with_offset(0);
        assert!(t1 < t2);
    }

    #[test]
    fn test_truncate() {
        let t = Time::from_hms_nano(14, 30, 45, 123_456_789).unwrap();

        let hour = t.truncate("hour").unwrap();
        assert_eq!(hour.hour(), 14);
        assert_eq!(hour.minute(), 0);
        assert_eq!(hour.second(), 0);
        assert_eq!(hour.nanosecond(), 0);

        let minute = t.truncate("minute").unwrap();
        assert_eq!(minute.hour(), 14);
        assert_eq!(minute.minute(), 30);
        assert_eq!(minute.second(), 0);

        let second = t.truncate("second").unwrap();
        assert_eq!(second.hour(), 14);
        assert_eq!(second.minute(), 30);
        assert_eq!(second.second(), 45);
        assert_eq!(second.nanosecond(), 0);

        assert!(t.truncate("day").is_none());
    }

    #[test]
    fn test_truncate_preserves_offset() {
        let t = Time::from_hms_nano(14, 30, 45, 0)
            .unwrap()
            .with_offset(19800); // +05:30
        let truncated = t.truncate("hour").unwrap();
        assert_eq!(truncated.offset_seconds(), Some(19800));
        assert_eq!(truncated.hour(), 14);
        assert_eq!(truncated.minute(), 0);
    }

    #[test]
    fn test_default() {
        let t = Time::default();
        assert_eq!(t.hour(), 0);
        assert_eq!(t.minute(), 0);
        assert_eq!(t.second(), 0);
    }

    #[test]
    fn test_add_duration_wraps_at_midnight() {
        use crate::types::Duration;
        // 23:00:00 + 2 hours should wrap to 01:00:00
        let t = Time::from_hms(23, 0, 0).unwrap();
        let dur = Duration::from_nanos(2 * 3_600_000_000_000); // 2 hours
        let result = t.add_duration(&dur);
        assert_eq!(result.hour(), 1);
        assert_eq!(result.minute(), 0);
    }

    #[test]
    fn test_add_duration_within_day() {
        use crate::types::Duration;
        let t = Time::from_hms(10, 30, 0).unwrap();
        let dur = Duration::from_nanos(90 * 60 * 1_000_000_000); // 90 minutes
        let result = t.add_duration(&dur);
        assert_eq!(result.hour(), 12);
        assert_eq!(result.minute(), 0);
    }
}
