//! ISO 8601 duration type with separate month, day, and nanosecond components.

use serde::{Deserialize, Serialize};
use std::fmt;

use super::decimal::{Decimal, DecimalError};

const NANOS_PER_SECOND: i64 = 1_000_000_000;
const NANOS_PER_DAY: i64 = 86_400 * NANOS_PER_SECOND;
const NANOS_PER_MINUTE: i64 = 60 * NANOS_PER_SECOND;
const NANOS_PER_HOUR: i64 = 60 * NANOS_PER_MINUTE;

/// An ISO 8601 duration with separate month, day, and nanosecond components.
///
/// The three components are kept separate because months have variable length
/// and cannot be converted to days without a reference date. This means
/// durations are only partially ordered.
///
/// # Examples
///
/// ```
/// use grafeo_common::types::Duration;
///
/// let d = Duration::parse("P1Y2M3DT4H5M6S").unwrap();
/// assert_eq!(d.months(), 14); // 1 year + 2 months
/// assert_eq!(d.days(), 3);
/// assert_eq!(d.to_string(), "P1Y2M3DT4H5M6S");
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Duration {
    /// Total months (years * 12 + months).
    months: i64,
    /// Days component.
    days: i64,
    /// Sub-day component in nanoseconds.
    nanos: i64,
}

impl Duration {
    /// Creates a duration from months, days, and nanoseconds.
    #[must_use]
    pub const fn new(months: i64, days: i64, nanos: i64) -> Self {
        Self {
            months,
            days,
            nanos,
        }
    }

    /// Creates a duration from a number of months.
    #[must_use]
    pub const fn from_months(months: i64) -> Self {
        Self {
            months,
            days: 0,
            nanos: 0,
        }
    }

    /// Creates a duration from a number of days.
    #[must_use]
    pub const fn from_days(days: i64) -> Self {
        Self {
            months: 0,
            days,
            nanos: 0,
        }
    }

    /// Creates a duration from nanoseconds.
    #[must_use]
    pub const fn from_nanos(nanos: i64) -> Self {
        Self {
            months: 0,
            days: 0,
            nanos,
        }
    }

    /// Creates a duration from seconds.
    #[must_use]
    pub const fn from_seconds(secs: i64) -> Self {
        Self {
            months: 0,
            days: 0,
            nanos: secs * NANOS_PER_SECOND,
        }
    }

    /// Returns the months component.
    #[must_use]
    pub const fn months(&self) -> i64 {
        self.months
    }

    /// Returns the days component.
    #[must_use]
    pub const fn days(&self) -> i64 {
        self.days
    }

    /// Returns the nanoseconds component.
    #[must_use]
    pub const fn nanos(&self) -> i64 {
        self.nanos
    }

    /// Returns true if all components are zero.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.months == 0 && self.days == 0 && self.nanos == 0
    }

    /// Negates all components.
    #[must_use]
    pub const fn neg(self) -> Self {
        Self {
            months: -self.months,
            days: -self.days,
            nanos: -self.nanos,
        }
    }

    /// Adds two durations component-wise.
    #[must_use]
    pub const fn add(self, other: Self) -> Self {
        Self {
            months: self.months + other.months,
            days: self.days + other.days,
            nanos: self.nanos + other.nanos,
        }
    }

    /// Subtracts another duration component-wise.
    #[must_use]
    pub const fn sub(self, other: Self) -> Self {
        Self {
            months: self.months - other.months,
            days: self.days - other.days,
            nanos: self.nanos - other.nanos,
        }
    }

    /// Multiplies all components by a factor.
    #[must_use]
    pub const fn mul(self, factor: i64) -> Self {
        Self {
            months: self.months * factor,
            days: self.days * factor,
            nanos: self.nanos * factor,
        }
    }

    /// Divides all components by a divisor (truncating).
    #[must_use]
    pub const fn div(self, divisor: i64) -> Self {
        Self {
            months: self.months / divisor,
            days: self.days / divisor,
            nanos: self.nanos / divisor,
        }
    }

    /// The duration times a number, each component scaled: a fraction of a
    /// month carries into days (a month is 30.436875 days, the average of the
    /// Gregorian year) and a fraction of a day into nanoseconds, rounded half
    /// away from zero, as openCypher multiplies durations. `P1D * 1.5` is
    /// `P1DT12H`.
    ///
    /// # Errors
    ///
    /// [`DecimalError::Overflow`] when a component does not fit.
    pub fn checked_mul_decimal(self, factor: &Decimal) -> Result<Self, DecimalError> {
        self.scaled(|component| component.checked_mul(factor))
    }

    /// The duration divided by a number, each component scaled as
    /// [`checked_mul_decimal`](Self::checked_mul_decimal) scales it:
    /// `P1D / 2` is `PT12H`.
    ///
    /// # Errors
    ///
    /// [`DecimalError::DivisionByZero`] for a divisor of zero,
    /// [`DecimalError::Overflow`] when a component does not fit.
    pub fn checked_div_decimal(self, divisor: &Decimal) -> Result<Self, DecimalError> {
        self.scaled(|component| component.checked_div(divisor))
    }

    /// Applies `step` to each component, carrying the fractions down.
    fn scaled(
        self,
        step: impl Fn(&Decimal) -> Result<Decimal, DecimalError>,
    ) -> Result<Self, DecimalError> {
        // The integer part toward zero, and what is left of the value
        fn split(value: &Decimal) -> Result<(i64, Decimal), DecimalError> {
            let whole = value.trunc_to_i64().ok_or(DecimalError::Overflow)?;
            Ok((whole, value.checked_sub(&Decimal::from_i64(whole))?))
        }
        let days_per_month = Decimal::new(30_436_875, 6)?;
        let (months, month_fraction) = split(&step(&Decimal::from_i64(self.months))?)?;
        let days = step(&Decimal::from_i64(self.days))?
            .checked_add(&month_fraction.checked_mul(&days_per_month)?)?;
        let (days, day_fraction) = split(&days)?;
        let nanos = step(&Decimal::from_i64(self.nanos))?
            .checked_add(&day_fraction.checked_mul(&Decimal::from_i64(NANOS_PER_DAY))?)?;
        let nanos = nanos.round().trunc_to_i64().ok_or(DecimalError::Overflow)?;
        Ok(Self::new(months, days, nanos))
    }

    /// Parses a duration from ISO 8601 format "OnYnMnDTnHnMnS".
    ///
    /// Examples: "P1Y", "P2M3D", "PT4H5M6S", "P1Y2M3DT4H5M6S", "PT0.5S"
    ///
    /// A fraction of a second takes the sign of its seconds: `PT-1.5S` is
    /// minus one and a half seconds. `None` when the text is not a duration
    /// or a component does not fit 64 bits.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let (negative, s) = if let Some(rest) = s.strip_prefix('-') {
            (true, rest)
        } else {
            (false, s)
        };

        let s = s.strip_prefix('P')?;
        let mut months: i64 = 0;
        let mut days: i64 = 0;
        let mut nanos: i64 = 0;
        let mut in_time = false;
        let mut num_start = 0;
        let mut has_content = false;

        let bytes = s.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            // The component a designator adds to, and its unit there
            let (total, unit) = match bytes[i] {
                b'T' => {
                    in_time = true;
                    i += 1;
                    num_start = i;
                    continue;
                }
                b'Y' if !in_time => (&mut months, 12),
                b'M' if !in_time => (&mut months, 1),
                b'W' if !in_time => (&mut days, 7),
                b'D' if !in_time => (&mut days, 1),
                b'H' if in_time => (&mut nanos, NANOS_PER_HOUR),
                b'M' if in_time => (&mut nanos, NANOS_PER_MINUTE),
                b'S' if in_time => (&mut nanos, 0),
                _ => {
                    i += 1;
                    continue;
                }
            };
            let text = &s[num_start..i];
            let part = if unit == 0 {
                seconds_nanos(text)?
            } else {
                text.parse::<i64>().ok()?.checked_mul(unit)?
            };
            *total = total.checked_add(part)?;
            has_content = true;
            i += 1;
            num_start = i;
        }

        if !has_content {
            return None;
        }

        if negative {
            Some(Self {
                months: months.checked_neg()?,
                days: days.checked_neg()?,
                nanos: nanos.checked_neg()?,
            })
        } else {
            Some(Self {
                months,
                days,
                nanos,
            })
        }
    }
}

/// The nanoseconds of a duration's seconds, `n` or `n.fraction`, the
/// fraction's first nine digits counted: the fraction takes the sign of the
/// whole seconds, `-0` included, so `-1.5` and `-0.5` are negative. `None`
/// when a part is not a number or the sum does not fit 64 bits.
///
/// It also reads what 0.6.0 wrote for a negative fraction in a duration of
/// mixed signs, the sign inside the fraction (`-1.-5` for -1.5 s): nine
/// characters with that sign, zero padded, before the trailing zeros were
/// cut. Digits that start with a zero are that eight digit field (`0.-05` is
/// -0.005 s), other digits the plain fraction: 0.6.0 wrote a fraction from
/// -0.01 to -0.1 s as it wrote ten times it (`-5` for both -0.05 and -0.5),
/// and this reads the larger.
fn seconds_nanos(text: &str) -> Option<i64> {
    let (whole_text, fraction) = match text.split_once('.') {
        Some((whole, fraction)) => (whole, Some(fraction)),
        None => (text, None),
    };
    let whole: i64 = whole_text.parse().ok()?;
    let nanos = whole.checked_mul(NANOS_PER_SECOND)?;
    let Some(fraction) = fraction else {
        return Some(nanos);
    };
    let (negative, digits, width) = match fraction.strip_prefix('-') {
        // 0.6.0 wrote it after seconds of 0 or less only
        Some(_) if whole > 0 => return None,
        Some(digits) if digits.starts_with('0') && digits.len() <= 8 => (true, digits, 8),
        Some(digits) => (true, digits, 9),
        None => (whole_text.starts_with('-'), fraction, 9),
    };
    let digits = digits.get(..digits.len().min(width))?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let scale = 10i64.pow(u32::try_from(width - digits.len()).ok()?);
    let fraction_nanos = digits.parse::<i64>().ok()? * scale;
    if negative {
        nanos.checked_sub(fraction_nanos)
    } else {
        nanos.checked_add(fraction_nanos)
    }
}

// Duration is deliberately NOT Ord (only PartialOrd).
// Comparing "P1M" vs "P30D" is ambiguous without a reference date.
impl PartialOrd for Duration {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Only comparable when all components are individually comparable
        // and the relationship is consistent across components.
        let m = self.months.cmp(&other.months);
        let d = self.days.cmp(&other.days);
        let n = self.nanos.cmp(&other.nanos);

        if m == d && d == n {
            Some(m)
        } else if (m == std::cmp::Ordering::Equal || m == d)
            && (d == std::cmp::Ordering::Equal || d == n)
            && (m == std::cmp::Ordering::Equal || m == n)
        {
            // All non-equal components agree on direction
            if m != std::cmp::Ordering::Equal {
                Some(m)
            } else if d != std::cmp::Ordering::Equal {
                Some(d)
            } else {
                Some(n)
            }
        } else {
            None // Incomparable
        }
    }
}

impl fmt::Debug for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Duration({})", self)
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (neg, months, days, nanos) = if self.months < 0 || self.days < 0 || self.nanos < 0 {
            // If any component is negative, display as negative duration
            if self.months <= 0 && self.days <= 0 && self.nanos <= 0 {
                (true, -self.months, -self.days, -self.nanos)
            } else {
                // Mixed signs: display as-is (unusual but valid)
                (false, self.months, self.days, self.nanos)
            }
        } else {
            (false, self.months, self.days, self.nanos)
        };

        if neg {
            write!(f, "-")?;
        }
        write!(f, "P")?;

        let years = months / 12;
        let m = months % 12;

        if years != 0 {
            write!(f, "{years}Y")?;
        }
        if m != 0 {
            write!(f, "{m}M")?;
        }
        if days != 0 {
            write!(f, "{days}D")?;
        }

        // Time part
        let hours = nanos / NANOS_PER_HOUR;
        let remaining = nanos % NANOS_PER_HOUR;
        let minutes = remaining / NANOS_PER_MINUTE;
        let remaining = remaining % NANOS_PER_MINUTE;
        let secs = remaining / NANOS_PER_SECOND;
        let sub_nanos = remaining % NANOS_PER_SECOND;

        if hours != 0 || minutes != 0 || secs != 0 || sub_nanos != 0 {
            write!(f, "T")?;
            if hours != 0 {
                write!(f, "{hours}H")?;
            }
            if minutes != 0 {
                write!(f, "{minutes}M")?;
            }
            if secs != 0 || sub_nanos != 0 {
                // The sign once, before the whole seconds (negative only in
                // a duration of mixed signs), the fraction as a magnitude:
                // `T-1.5S`, `T-0.5S`
                let sign = if remaining < 0 { "-" } else { "" };
                write!(f, "{sign}{}", secs.unsigned_abs())?;
                if sub_nanos != 0 {
                    let frac = format!("{:09}", sub_nanos.unsigned_abs());
                    write!(f, ".{}", frac.trim_end_matches('0'))?;
                }
                write!(f, "S")?;
            }
        }

        // Handle zero duration
        if !neg && years == 0 && m == 0 && days == 0 && nanos == 0 {
            write!(f, "T0S")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scaled_by(text: &str, factor: &str) -> String {
        let factor = Decimal::parse(factor).unwrap();
        Duration::parse(text)
            .unwrap()
            .checked_mul_decimal(&factor)
            .unwrap()
            .to_string()
    }

    fn divided_by(text: &str, divisor: i64) -> Result<String, DecimalError> {
        Duration::parse(text)
            .unwrap()
            .checked_div_decimal(&Decimal::from_i64(divisor))
            .map(|d| d.to_string())
    }

    #[test]
    fn multiplying_carries_fractions_down() {
        assert_eq!(scaled_by("P1D", "1.5"), "P1DT12H");
        assert_eq!(scaled_by("P1D", "2"), "P2D");
        assert_eq!(scaled_by("PT1H", "0.5"), "PT30M");
        // Half a month is 15.2184375 days: 15 days, 5 h 14 min 33 s
        assert_eq!(scaled_by("P1M", "1.5"), "P1M15DT5H14M33S");
        assert_eq!(scaled_by("P1Y", "0.5"), "P6M");
        assert_eq!(scaled_by("P1D", "-1.5"), "-P1DT12H");
        assert_eq!(scaled_by("P1M2DT3H", "0"), "PT0S");
    }

    #[test]
    fn dividing_carries_fractions_down() {
        assert_eq!(divided_by("P1D", 2).unwrap(), "PT12H");
        assert_eq!(divided_by("P1M", 2).unwrap(), "P15DT5H14M33S");
        assert_eq!(divided_by("PT1S", 3).unwrap(), "PT0.333333333S");
        assert_eq!(divided_by("P1D", 0), Err(DecimalError::DivisionByZero));
    }

    #[test]
    fn scaling_past_the_range_is_an_overflow() {
        let factor = Decimal::from_i64(i64::MAX);
        assert_eq!(
            Duration::new(0, 2, 0).checked_mul_decimal(&factor),
            Err(DecimalError::Overflow)
        );
    }

    #[test]
    fn test_constructors() {
        let d = Duration::new(14, 3, 0);
        assert_eq!(d.months(), 14);
        assert_eq!(d.days(), 3);
        assert_eq!(d.nanos(), 0);

        assert_eq!(Duration::from_months(6).months(), 6);
        assert_eq!(Duration::from_days(10).days(), 10);
        assert_eq!(Duration::from_nanos(1_000_000_000).nanos(), 1_000_000_000);
    }

    #[test]
    fn test_parse_full() {
        let d = Duration::parse("P1Y2M3DT4H5M6S").unwrap();
        assert_eq!(d.months(), 14); // 12 + 2
        assert_eq!(d.days(), 3);
        assert_eq!(
            d.nanos(),
            4 * NANOS_PER_HOUR + 5 * NANOS_PER_MINUTE + 6 * NANOS_PER_SECOND
        );
    }

    #[test]
    fn test_parse_partial() {
        let d = Duration::parse("P1Y").unwrap();
        assert_eq!(d.months(), 12);
        assert_eq!(d.days(), 0);

        let d = Duration::parse("PT30S").unwrap();
        assert_eq!(d.months(), 0);
        assert_eq!(d.nanos(), 30 * NANOS_PER_SECOND);

        let d = Duration::parse("P2W").unwrap();
        assert_eq!(d.days(), 14);
    }

    #[test]
    fn test_parse_fractional_seconds() {
        let d = Duration::parse("PT0.5S").unwrap();
        assert_eq!(d.nanos(), 500_000_000);

        let d = Duration::parse("PT1.123S").unwrap();
        assert_eq!(d.nanos(), 1_123_000_000);
    }

    #[test]
    fn test_parse_negative() {
        let d = Duration::parse("-P1Y").unwrap();
        assert_eq!(d.months(), -12);
    }

    #[test]
    fn test_parse_invalid() {
        assert!(Duration::parse("").is_none());
        assert!(Duration::parse("P").is_none());
        assert!(Duration::parse("not-a-duration").is_none());
    }

    #[test]
    fn test_display_roundtrip() {
        let cases = ["P1Y2M3DT4H5M6S", "P1Y", "P3D", "PT30S", "PT0.5S", "P2W"];
        for case in cases {
            let d = Duration::parse(case).unwrap();
            let reparsed = Duration::parse(&d.to_string()).unwrap();
            assert_eq!(d, reparsed, "roundtrip failed for {case}");
        }
    }

    #[test]
    fn test_arithmetic() {
        let a = Duration::new(1, 2, 3);
        let b = Duration::new(4, 5, 6);
        assert_eq!(a.add(b), Duration::new(5, 7, 9));
        assert_eq!(a.sub(b), Duration::new(-3, -3, -3));
        assert_eq!(a.mul(3), Duration::new(3, 6, 9));
        assert_eq!(a.neg(), Duration::new(-1, -2, -3));
    }

    #[test]
    fn test_partial_ord() {
        let a = Duration::new(1, 2, 3);
        let b = Duration::new(2, 3, 4);
        assert!(a < b);

        // Incomparable: more months but fewer days
        let c = Duration::new(2, 1, 0);
        let d = Duration::new(1, 100, 0);
        assert_eq!(c.partial_cmp(&d), None);
    }

    #[test]
    fn test_zero() {
        let z = Duration::default();
        assert!(z.is_zero());
        assert_eq!(z.to_string(), "PT0S");
    }

    #[test]
    fn negative_seconds_carry_their_sign_into_the_fraction() {
        assert_eq!(
            Duration::parse("PT-1.5S"),
            Some(Duration::from_nanos(-1_500_000_000))
        );
        assert_eq!(
            Duration::parse("PT-0.5S"),
            Some(Duration::from_nanos(-500_000_000))
        );
        assert_eq!(
            Duration::parse("PT-0.000000001S"),
            Some(Duration::from_nanos(-1))
        );
        assert_eq!(
            Duration::parse("-PT-1.5S"),
            Some(Duration::from_nanos(1_500_000_000))
        );
        assert_eq!(Duration::parse("PT-0S"), Some(Duration::from_nanos(0)));
        assert_eq!(
            Duration::parse("PT1.5S"),
            Some(Duration::from_nanos(1_500_000_000))
        );
    }

    #[test]
    fn mixed_signs_write_the_sign_once_on_the_seconds() {
        assert_eq!(Duration::new(1, 0, -1_500_000_000).to_string(), "P1MT-1.5S");
        assert_eq!(Duration::new(1, 0, -500_000_000).to_string(), "P1MT-0.5S");
        assert_eq!(Duration::new(1, 0, -1_000_000_000).to_string(), "P1MT-1S");
        assert_eq!(Duration::new(0, 2, -5_000_000).to_string(), "P2DT-0.005S");
        assert_eq!(Duration::new(0, 2, -1).to_string(), "P2DT-0.000000001S");
        assert_eq!(
            Duration::new(
                1,
                0,
                -(NANOS_PER_HOUR + 30 * NANOS_PER_MINUTE + 1_250_000_000)
            )
            .to_string(),
            "P1MT-1H-30M-1.25S"
        );
        // Whole negative and positive durations are written as before
        assert_eq!(Duration::new(0, -1, -500_000_000).to_string(), "-P1DT0.5S");
        assert_eq!(Duration::from_nanos(-1_500_000_000).to_string(), "-PT1.5S");
        assert_eq!(Duration::new(-1, 0, 1_500_000_000).to_string(), "P-1MT1.5S");
        for (text, nanos) in [
            ("P1MT-1.5S", -1_500_000_000),
            ("P1MT-0.5S", -500_000_000),
            ("P1MT-1S", -1_000_000_000),
        ] {
            assert_eq!(
                Duration::parse(text),
                Some(Duration::new(1, 0, nanos)),
                "{text}"
            );
        }
        assert_eq!(
            Duration::parse("-P1DT0.5S"),
            Some(Duration::new(0, -1, -500_000_000))
        );
    }

    #[test]
    fn the_mixed_sign_text_of_0_6_0_reads_as_it_was_meant() {
        // 0.6.0 wrote a negative fraction with its own sign, nine digits wide
        // with that sign: `-0.05` as `-05`, `-0.5` as `-5`
        for (text, nanos) in [
            ("P1MT-1.-5S", -1_500_000_000),
            ("P1MT0.-5S", -500_000_000),
            ("P1MT0.-05S", -5_000_000),
            ("P1MT-2.-25S", -2_250_000_000),
            ("P1MT0.-0000001S", -10),
            ("P1MT0.-00000005S", -5),
            ("P1MT0.-123456789S", -123_456_789),
        ] {
            assert_eq!(
                Duration::parse(text),
                Some(Duration::new(1, 0, nanos)),
                "{text}"
            );
        }
        // 0.6.0 never wrote a negative fraction after positive seconds
        assert_eq!(Duration::parse("P1MT1.-5S"), None);
    }

    #[test]
    fn components_past_64_bits_are_refused() {
        for text in [
            "P768614336404564651Y",
            "P1317624576693539402W",
            "P9223372036854775808D",
            "PT2562048H",
            "PT153722867M281S",
            "PT9223372037S",
            "PT-9223372037S",
            "PT9223372036.854775808S",
            "P9223372036854775807M1Y",
            "-P-9223372036854775808D",
        ] {
            assert_eq!(Duration::parse(text), None, "{text}");
        }
        assert_eq!(
            Duration::parse("P9223372036854775807M"),
            Some(Duration::from_months(i64::MAX))
        );
        assert_eq!(
            Duration::parse("P-9223372036854775808D"),
            Some(Duration::from_days(i64::MIN))
        );
        assert_eq!(
            Duration::parse("PT9223372036.854775807S"),
            Some(Duration::from_nanos(i64::MAX))
        );
        assert_eq!(
            Duration::parse("PT-9223372036.854775808S"),
            Some(Duration::from_nanos(i64::MIN))
        );
    }

    mod round_trip {
        use super::*;
        use proptest::prelude::*;
        use proptest::test_runner::{Config, RngSeed};

        // Each component of either sign, and nanoseconds both under a
        // second and over many days
        fn component(limit: i64) -> impl Strategy<Value = i64> {
            prop_oneof![Just(0), -limit..=limit]
        }

        proptest! {
            #![proptest_config(Config {
                cases: if cfg!(miri) { 8 } else { 1024 },
                failure_persistence: None,
                rng_seed: RngSeed::Fixed(0x6610_0000_0000_0001),
                ..Config::default()
            })]

            #[test]
            fn a_written_duration_reads_back_as_itself(
                months in component(1_000_000),
                days in component(1_000_000),
                nanos in prop_oneof![component(NANOS_PER_SECOND), component(100_000 * NANOS_PER_DAY)],
            ) {
                let duration = Duration::new(months, days, nanos);
                let text = duration.to_string();
                prop_assert_eq!(Duration::parse(&text), Some(duration), "{}", text);
                prop_assert_eq!(text.contains(".-"), false, "{}", text);
            }
        }
    }
}
