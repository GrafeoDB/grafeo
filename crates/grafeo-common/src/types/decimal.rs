//! Exact decimal numbers: a signed mantissa of at most 38 digits and a scale,
//! the number of those digits after the decimal point (as SQL `DECIMAL(p,s)`,
//! Arrow `Decimal128` and `xsd:decimal`).

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

/// The most digits a decimal holds, and the largest scale.
pub const MAX_DECIMAL_DIGITS: u8 = 38;

/// Why a decimal cannot be built or computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DecimalError {
    /// The value needs more than 38 digits.
    #[error("decimal value out of range: more than 38 digits")]
    Overflow,
    /// A division by zero.
    #[error("division by zero")]
    DivisionByZero,
    /// The text is not a decimal number.
    #[error("not a decimal number")]
    Invalid,
    /// A decimal type whose precision is not from 1 to 38, or whose scale is
    /// larger than its precision.
    #[error("a decimal type takes a precision from 1 to 38 and a scale from 0 to its precision")]
    InvalidPrecision,
}

/// An exact decimal number: `mantissa / 10^scale`, with at most 38 digits in
/// the mantissa and a scale from 0 to 38.
///
/// The scale is kept as written (`1.50` has scale 2 and prints as `1.50`),
/// while equality, hashing and order compare values: `1.50` equals `1.5`. The
/// mantissa is stored as two 64-bit words so the type is 8-byte aligned.
#[derive(Clone, Copy)]
pub struct Decimal {
    scale: u8,
    mantissa: [u64; 2],
}

/// `10^n` for `n` from 0 to 38.
const POW10: [u128; 39] = {
    let mut table = [1u128; 39];
    let mut n = 1;
    while n < 39 {
        table[n] = table[n - 1] * 10;
        n += 1;
    }
    table
};

/// The first magnitude a mantissa may not reach: `10^38`.
const LIMIT: u128 = POW10[38];

/// `10^n`, for a scale or a difference of scales (at most 38).
fn pow10(n: u8) -> u128 {
    POW10[usize::from(n)]
}

/// The number of decimal digits of `value` (0 for zero), for values below
/// `10^38`.
fn digit_count(value: u128) -> u8 {
    let mut count = 0;
    while count < MAX_DECIMAL_DIGITS && pow10(count) <= value {
        count += 1;
    }
    count
}

/// `magnitude / 10^digits`, rounded half away from zero.
fn round_off(magnitude: u128, digits: u8) -> u128 {
    if digits == 0 {
        return magnitude;
    }
    if digits > MAX_DECIMAL_DIGITS {
        return 0;
    }
    let divisor = pow10(digits);
    let (quotient, remainder) = (magnitude / divisor, magnitude % divisor);
    if remainder >= divisor - remainder {
        quotient + 1
    } else {
        quotient
    }
}

/// `10^n` as an `i128`: every power up to `10^38` fits.
fn pow10_signed(n: u8) -> i128 {
    pow10(n).cast_signed()
}

/// The low 64 bits of `value`.
fn low_word(value: u128) -> u64 {
    u64::try_from(value & u128::from(u64::MAX)).unwrap_or_default()
}

/// The two's complement words of `mantissa`, low word first.
fn words(mantissa: i128) -> [u64; 2] {
    let bits = mantissa.cast_unsigned();
    [low_word(bits), low_word(bits >> 64)]
}

impl Decimal {
    /// The decimal `mantissa / 10^scale`.
    ///
    /// # Errors
    ///
    /// [`DecimalError::Overflow`] when the mantissa has more than 38 digits or
    /// the scale is above 38.
    pub fn new(mantissa: i128, scale: u8) -> Result<Self, DecimalError> {
        if scale > MAX_DECIMAL_DIGITS || mantissa.unsigned_abs() >= LIMIT {
            return Err(DecimalError::Overflow);
        }
        Ok(Self {
            scale,
            mantissa: words(mantissa),
        })
    }

    /// The decimal of a magnitude, a sign and a scale.
    fn from_magnitude(magnitude: u128, negative: bool, scale: u8) -> Result<Self, DecimalError> {
        let signed = i128::try_from(magnitude).map_err(|_| DecimalError::Overflow)?;
        Self::new(if negative { -signed } else { signed }, scale)
    }

    /// The mantissa: the value times `10^scale`.
    #[must_use]
    pub fn mantissa(&self) -> i128 {
        ((u128::from(self.mantissa[1]) << 64) | u128::from(self.mantissa[0])).cast_signed()
    }

    /// The number of digits after the decimal point.
    #[must_use]
    pub fn scale(&self) -> u8 {
        self.scale
    }

    /// The same value without trailing zeros after the point (`1.50` is
    /// `1.5`, `3.00` is `3`): one decimal for every scale of a value, as `==`
    /// and `Hash` see it. For identities such as grouping keys; a value shown
    /// to the user keeps its own scale.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let (mantissa, scale) = self.normalized();
        Self {
            scale,
            mantissa: words(mantissa),
        }
    }

    /// The mantissa and scale without trailing zeros after the point: the
    /// same pair for every scale of a value.
    fn normalized(&self) -> (i128, u8) {
        let (mut mantissa, mut scale) = (self.mantissa(), self.scale);
        while scale > 0 && mantissa % 10 == 0 {
            mantissa /= 10;
            scale -= 1;
        }
        (mantissa, scale)
    }

    /// Parses `[+-]digits[.digits]` (`1.50`, `-0.5`, `.5`, `5.`).
    ///
    /// # Errors
    ///
    /// [`DecimalError::Invalid`] for other text, [`DecimalError::Overflow`] for
    /// more than 38 digits.
    pub fn parse(text: &str) -> Result<Self, DecimalError> {
        Self::parse_digits(text, false)
    }

    /// [`parse`](Self::parse); with `round`, more than 38 digits after the
    /// point round half away from zero to 38 instead of failing.
    fn parse_digits(text: &str, round: bool) -> Result<Self, DecimalError> {
        let (negative, unsigned) = match text.as_bytes().first() {
            Some(b'-') => (true, &text[1..]),
            Some(b'+') => (false, &text[1..]),
            _ => (false, text),
        };
        let (integer, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));
        let digits = |part: &str| part.bytes().all(|byte| byte.is_ascii_digit());
        if (integer.is_empty() && fraction.is_empty()) || !digits(integer) || !digits(fraction) {
            return Err(DecimalError::Invalid);
        }
        let most = usize::from(MAX_DECIMAL_DIGITS);
        let (kept, dropped) = if fraction.len() > most {
            if !round {
                return Err(DecimalError::Overflow);
            }
            fraction.split_at(most)
        } else {
            (fraction, "")
        };
        let mut magnitude: u128 = 0;
        for byte in integer.bytes().chain(kept.bytes()) {
            magnitude = magnitude
                .checked_mul(10)
                .map(|value| value + u128::from(byte - b'0'))
                .filter(|&value| value < LIMIT)
                .ok_or(DecimalError::Overflow)?;
        }
        if dropped
            .as_bytes()
            .first()
            .is_some_and(|&digit| digit >= b'5')
        {
            magnitude += 1;
        }
        let scale = u8::try_from(kept.len()).unwrap_or(MAX_DECIMAL_DIGITS);
        Self::from_magnitude(magnitude, negative, scale)
    }

    /// The same value with `scale` digits after the point, rounded half away
    /// from zero when digits go.
    ///
    /// # Errors
    ///
    /// [`DecimalError::Overflow`] when the value needs more than 38 digits at
    /// that scale.
    pub fn rescale(&self, scale: u8) -> Result<Self, DecimalError> {
        if scale > MAX_DECIMAL_DIGITS {
            return Err(DecimalError::Overflow);
        }
        let (magnitude, negative) = (self.mantissa().unsigned_abs(), self.mantissa() < 0);
        let rescaled = if scale >= self.scale {
            magnitude
                .checked_mul(pow10(scale - self.scale))
                .ok_or(DecimalError::Overflow)?
        } else {
            round_off(magnitude, self.scale - scale)
        };
        Self::from_magnitude(rescaled, negative, scale)
    }

    /// The value as a `DECIMAL(precision, scale)` holds it: with `scale`
    /// digits after the point, rounded half away from zero when digits go
    /// ([`rescale`](Self::rescale)), and at most `precision - scale` before it.
    ///
    /// # Errors
    ///
    /// [`DecimalError::InvalidPrecision`] for a precision outside 1 to 38 or a
    /// scale above it; [`DecimalError::Overflow`] when the value has more
    /// integer digits than the type holds, also once its fraction rounds up.
    pub fn with_precision(&self, precision: u8, scale: u8) -> Result<Self, DecimalError> {
        if !(1..=MAX_DECIMAL_DIGITS).contains(&precision) || scale > precision {
            return Err(DecimalError::InvalidPrecision);
        }
        let rescaled = self.rescale(scale)?;
        if rescaled.mantissa().unsigned_abs() >= pow10(precision) {
            return Err(DecimalError::Overflow);
        }
        Ok(rescaled)
    }

    /// The mantissas of `self` and `other` at the larger of their scales.
    fn aligned(&self, other: &Self) -> Result<(i128, i128, u8), DecimalError> {
        let scale = self.scale.max(other.scale);
        Ok((
            self.rescale(scale)?.mantissa(),
            other.rescale(scale)?.mantissa(),
            scale,
        ))
    }

    /// The sum, at the larger scale of the two.
    ///
    /// # Errors
    ///
    /// [`DecimalError::Overflow`] past 38 digits.
    pub fn checked_add(&self, other: &Self) -> Result<Self, DecimalError> {
        let (left, right, scale) = self.aligned(other)?;
        Self::new(
            left.checked_add(right).ok_or(DecimalError::Overflow)?,
            scale,
        )
    }

    /// The difference, at the larger scale of the two.
    ///
    /// # Errors
    ///
    /// [`DecimalError::Overflow`] past 38 digits.
    pub fn checked_sub(&self, other: &Self) -> Result<Self, DecimalError> {
        let (left, right, scale) = self.aligned(other)?;
        Self::new(
            left.checked_sub(right).ok_or(DecimalError::Overflow)?,
            scale,
        )
    }

    /// The product, at the sum of the scales (at most 38, rounded half away
    /// from zero beyond that).
    ///
    /// # Errors
    ///
    /// [`DecimalError::Overflow`] when the product needs more than 38 digits.
    pub fn checked_mul(&self, other: &Self) -> Result<Self, DecimalError> {
        let negative = (self.mantissa() < 0) != (other.mantissa() < 0);
        let product = self
            .mantissa()
            .unsigned_abs()
            .checked_mul(other.mantissa().unsigned_abs())
            .ok_or(DecimalError::Overflow)?;
        let scale = self.scale + other.scale;
        let (magnitude, scale) = if scale > MAX_DECIMAL_DIGITS {
            (
                round_off(product, scale - MAX_DECIMAL_DIGITS),
                MAX_DECIMAL_DIGITS,
            )
        } else {
            (product, scale)
        };
        Self::from_magnitude(magnitude, negative, scale)
    }

    /// The quotient with at least 16 significant digits and at least the
    /// larger scale of the two, at most 38 digits in all, rounded half away
    /// from zero: `1 / 3` is `0.3333333333333333`.
    ///
    /// # Errors
    ///
    /// [`DecimalError::DivisionByZero`], or [`DecimalError::Overflow`] when
    /// the integer part needs more than 38 digits.
    pub fn checked_div(&self, other: &Self) -> Result<Self, DecimalError> {
        const SIGNIFICANT: u8 = 16;
        if other.mantissa() == 0 {
            return Err(DecimalError::DivisionByZero);
        }
        let negative = (self.mantissa() < 0) != (other.mantissa() < 0);
        // (a / 10^as) / (b / 10^bs) = (a * 10^bs) / (b * 10^as)
        let numerator = U256::from(self.mantissa().unsigned_abs()).times_pow10(other.scale);
        let denominator = U256::from(other.mantissa().unsigned_abs()).times_pow10(self.scale);
        let (integer, mut remainder) = numerator.div_rem(&denominator);
        let integer = integer
            .to_u128()
            .filter(|&value| value < LIMIT)
            .ok_or(DecimalError::Overflow)?;
        let integer_digits = digit_count(integer);
        // The fraction's digits, one more than the most a result keeps.
        let mut fraction = [0u8; 39];
        for digit in &mut fraction {
            remainder = remainder.times_pow10(1);
            while remainder >= denominator {
                remainder = remainder.minus(&denominator);
                *digit += 1;
            }
        }
        let wanted = if integer > 0 {
            SIGNIFICANT.saturating_sub(integer_digits)
        } else {
            match fraction.iter().position(|&digit| digit != 0) {
                Some(zeros) => u8::try_from(zeros)
                    .unwrap_or(MAX_DECIMAL_DIGITS)
                    .saturating_add(SIGNIFICANT),
                None => 0,
            }
        };
        let scale = wanted
            .max(self.scale)
            .max(other.scale)
            .min(MAX_DECIMAL_DIGITS - integer_digits);
        let kept = fraction[..usize::from(scale)]
            .iter()
            .fold(0u128, |value, &digit| value * 10 + u128::from(digit));
        let mut magnitude = integer * pow10(scale) + kept;
        if fraction[usize::from(scale)] >= 5 {
            magnitude += 1;
        }
        if magnitude >= LIMIT {
            return Err(DecimalError::Overflow);
        }
        Self::from_magnitude(magnitude, negative, scale)
    }

    /// The remainder of the division toward zero, with the sign of `self`, at
    /// the larger scale of the two: `7.5 % 2` is `1.5`.
    ///
    /// # Errors
    ///
    /// [`DecimalError::DivisionByZero`], or [`DecimalError::Overflow`] when
    /// aligning the scales needs more than 38 digits.
    pub fn checked_rem(&self, other: &Self) -> Result<Self, DecimalError> {
        if other.mantissa() == 0 {
            return Err(DecimalError::DivisionByZero);
        }
        let (left, right, scale) = self.aligned(other)?;
        Self::new(left % right, scale)
    }

    /// The value with the opposite sign.
    #[must_use]
    pub fn neg(&self) -> Self {
        // A mantissa below 10^38 in magnitude negates within range.
        Self {
            scale: self.scale,
            mantissa: words(-self.mantissa()),
        }
    }

    /// The absolute value.
    #[must_use]
    pub fn abs(&self) -> Self {
        if self.mantissa() < 0 {
            self.neg()
        } else {
            *self
        }
    }

    /// -1, 0 or 1, as the sign of the value.
    #[must_use]
    pub fn signum(&self) -> i64 {
        match self.mantissa().cmp(&0) {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        }
    }

    /// The integer part toward zero and whether a fraction was cut off, with
    /// its sign: -1, 0 or 1.
    fn truncated(&self) -> (i128, i128) {
        let unit = pow10_signed(self.scale);
        let mantissa = self.mantissa();
        (mantissa / unit, (mantissa % unit).signum())
    }

    /// An integer at scale 0. With a fraction the integer part has at most
    /// 37 digits, so one more stays within range.
    fn whole(integer: i128) -> Self {
        Self {
            scale: 0,
            mantissa: words(integer),
        }
    }

    /// The greatest integer not above the value, at scale 0.
    #[must_use]
    pub fn floor(&self) -> Self {
        let (integer, fraction) = self.truncated();
        Self::whole(if fraction < 0 { integer - 1 } else { integer })
    }

    /// The least integer not below the value, at scale 0.
    #[must_use]
    pub fn ceil(&self) -> Self {
        let (integer, fraction) = self.truncated();
        Self::whole(if fraction > 0 { integer + 1 } else { integer })
    }

    /// The nearest integer, half away from zero, at scale 0.
    #[must_use]
    pub fn round(&self) -> Self {
        let magnitude = round_off(self.mantissa().unsigned_abs(), self.scale);
        let integer = magnitude.cast_signed();
        Self::whole(if self.mantissa() < 0 {
            -integer
        } else {
            integer
        })
    }

    /// Compares with a float by the float's shortest round-trip text (the
    /// float `1.5` equals `1.5`). NaN is greater than every decimal; the
    /// infinities and floats past 38 digits order by their sign. A float
    /// whose text has more than 38 digits after the point (`1e-300`) equals
    /// no decimal: it orders next to the decimal it rounds to, on its own
    /// side of it.
    #[must_use]
    pub fn cmp_f64(&self, value: f64) -> Ordering {
        if value.is_nan() {
            return Ordering::Less;
        }
        match Self::from_f64(value) {
            Ok(other) => match self.cmp(&other) {
                Ordering::Equal if other.to_f64() < value => Ordering::Less,
                Ordering::Equal if other.to_f64() > value => Ordering::Greater,
                order => order,
            },
            Err(_) if value > 0.0 => Ordering::Less,
            Err(_) => Ordering::Greater,
        }
    }

    /// The decimal of an integer, at scale 0.
    #[must_use]
    pub fn from_i64(value: i64) -> Self {
        // An i64 has at most 19 digits.
        Self {
            scale: 0,
            mantissa: words(i128::from(value)),
        }
    }

    /// The decimal a float prints as (its shortest round-trip form), with at
    /// most 38 digits after the point (rounded half away from zero).
    ///
    /// # Errors
    ///
    /// [`DecimalError::Invalid`] for NaN and the infinities,
    /// [`DecimalError::Overflow`] past 38 digits before the point.
    pub fn from_f64(value: f64) -> Result<Self, DecimalError> {
        if !value.is_finite() {
            return Err(DecimalError::Invalid);
        }
        Self::parse_digits(&value.to_string(), true)
    }

    /// The nearest float.
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        // A decimal prints as a float literal, so the parse succeeds.
        self.to_string().parse().unwrap_or(f64::NAN)
    }

    /// The integer part, toward zero, when it fits an `i64`.
    #[must_use]
    pub fn trunc_to_i64(&self) -> Option<i64> {
        i64::try_from(self.mantissa() / pow10_signed(self.scale)).ok()
    }

    /// The canonical `xsd:decimal` form: no trailing zeros after the point,
    /// and at least one digit on each side (`1.0`, `-0.5`, `12.25`).
    #[must_use]
    pub fn to_xsd_string(&self) -> String {
        let (mantissa, scale) = self.normalized();
        let text = Self {
            scale,
            mantissa: words(mantissa),
        }
        .to_string();
        if scale == 0 {
            format!("{text}.0")
        } else {
            text
        }
    }
}

impl fmt::Display for Decimal {
    /// The value with its scale: `1.50`, `-0.5`, `12`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mantissa = self.mantissa();
        let digits = mantissa.unsigned_abs().to_string();
        let scale = usize::from(self.scale);
        if mantissa < 0 {
            f.write_str("-")?;
        }
        if scale == 0 {
            f.write_str(&digits)
        } else if digits.len() > scale {
            let (integer, fraction) = digits.split_at(digits.len() - scale);
            write!(f, "{integer}.{fraction}")
        } else {
            write!(f, "0.{}{digits}", "0".repeat(scale - digits.len()))
        }
    }
}

impl fmt::Debug for Decimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Decimal({self})")
    }
}

impl PartialEq for Decimal {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Decimal {}

impl PartialOrd for Decimal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Decimal {
    /// Compares the integer parts, then the fractions widened to 38 digits:
    /// both fit an `i128` whatever the scales, and share the value's sign.
    fn cmp(&self, other: &Self) -> Ordering {
        let parts = |value: &Self| {
            let unit = pow10_signed(value.scale);
            let widen = pow10_signed(MAX_DECIMAL_DIGITS - value.scale);
            let mantissa = value.mantissa();
            (mantissa / unit, (mantissa % unit) * widen)
        };
        parts(self).cmp(&parts(other))
    }
}

impl serde::Serialize for Decimal {
    /// The text with its scale (`"1.50"`), in every serde format.
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> serde::Deserialize<'de> for Decimal {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = <std::borrow::Cow<'de, str>>::deserialize(deserializer)?;
        Self::parse(&text).map_err(serde::de::Error::custom)
    }
}

impl Hash for Decimal {
    /// Hashes the value without trailing zeros, so equal values hash alike.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.normalized().hash(state);
    }
}

/// The precision and scale of a `DECIMAL(precision, scale)` type: a
/// precision from 1 to 38, the digits a value has in all, and a scale from 0
/// to the precision, the digits after the point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecimalType {
    precision: u8,
    scale: u8,
}

impl DecimalType {
    /// `DECIMAL(precision, scale)`.
    ///
    /// # Errors
    ///
    /// [`DecimalError::InvalidPrecision`] for a precision outside 1 to 38 or a
    /// scale above it.
    pub fn new(precision: u8, scale: u8) -> Result<Self, DecimalError> {
        if (1..=MAX_DECIMAL_DIGITS).contains(&precision) && scale <= precision {
            Ok(Self { precision, scale })
        } else {
            Err(DecimalError::InvalidPrecision)
        }
    }

    /// The digits a value of the type has in all.
    #[must_use]
    pub const fn precision(self) -> u8 {
        self.precision
    }

    /// The digits after the point.
    #[must_use]
    pub const fn scale(self) -> u8 {
        self.scale
    }

    /// `value` as the type holds it: see [`Decimal::with_precision`].
    ///
    /// # Errors
    ///
    /// [`DecimalError::Overflow`] when `value` has more integer digits than
    /// the type holds.
    pub fn fit(self, value: &Decimal) -> Result<Decimal, DecimalError> {
        value.with_precision(self.precision, self.scale)
    }

    /// Whether `value` is a value of the type as it is: at its scale, with no
    /// more integer digits than the type holds.
    #[must_use]
    pub fn holds(self, value: &Decimal) -> bool {
        value.scale == self.scale && value.mantissa().unsigned_abs() < pow10(self.precision)
    }

    /// Reads a decimal type name, in any case: `DECIMAL` or `DEC` alone is
    /// `Some(None)`, any decimal; with `(precision)` or `(precision, scale)`
    /// it is that type (a left-out scale is 0). `None` when `name` names no
    /// decimal type, or one whose precision or scale is out of range.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Option<Self>> {
        let name = name.trim();
        let (base, arguments) = match name.split_once('(') {
            Some((base, rest)) => (base.trim_end(), Some(rest.strip_suffix(')')?)),
            None => (name, None),
        };
        if !base.eq_ignore_ascii_case("DECIMAL") && !base.eq_ignore_ascii_case("DEC") {
            return None;
        }
        let Some(arguments) = arguments else {
            return Some(None);
        };
        let number = |text: &str| text.trim().parse::<u8>().ok();
        let (precision, scale) = match arguments.split_once(',') {
            Some((precision, scale)) => (number(precision)?, number(scale)?),
            None => (number(arguments)?, 0),
        };
        Self::new(precision, scale).ok().map(Some)
    }
}

impl fmt::Display for DecimalType {
    /// `DECIMAL(10,2)`: the spelling [`from_name`](Self::from_name) reads.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DECIMAL({},{})", self.precision, self.scale)
    }
}

impl serde::Serialize for DecimalType {
    /// The precision and the scale.
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (self.precision, self.scale).serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for DecimalType {
    /// Refuses a precision or scale out of range.
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (precision, scale) = <(u8, u8)>::deserialize(deserializer)?;
        Self::new(precision, scale).map_err(serde::de::Error::custom)
    }
}

/// An unsigned 256-bit integer, enough for a 38-digit mantissa times `10^38`
/// (the operands of a division at a common scale) and ten times its
/// remainder.
#[derive(Clone, Copy, PartialEq, Eq)]
struct U256 {
    /// Little-endian 64-bit limbs.
    limbs: [u64; 4],
}

impl From<u128> for U256 {
    fn from(value: u128) -> Self {
        Self {
            limbs: [low_word(value), low_word(value >> 64), 0, 0],
        }
    }
}

impl U256 {
    /// `self * 10^n`; every caller stays below `10^77 < 2^256`.
    fn times_pow10(mut self, n: u8) -> Self {
        for _ in 0..n {
            let mut carry = 0u128;
            for limb in &mut self.limbs {
                let product = u128::from(*limb) * 10 + carry;
                *limb = low_word(product);
                carry = product >> 64;
            }
            debug_assert_eq!(carry, 0, "a decimal division stays below 2^256");
        }
        self
    }

    fn minus(&self, other: &Self) -> Self {
        let mut limbs = [0u64; 4];
        let mut borrow = false;
        for (index, limb) in limbs.iter_mut().enumerate() {
            let (difference, under) = self.limbs[index].overflowing_sub(other.limbs[index]);
            let (difference, under_again) = difference.overflowing_sub(u64::from(borrow));
            *limb = difference;
            borrow = under || under_again;
        }
        Self { limbs }
    }

    fn bit(&self, index: usize) -> bool {
        (self.limbs[index / 64] >> (index % 64)) & 1 == 1
    }

    fn shifted_left_with(&self, low_bit: bool) -> Self {
        let mut limbs = [0u64; 4];
        let mut carry = u64::from(low_bit);
        for (index, limb) in limbs.iter_mut().enumerate() {
            *limb = (self.limbs[index] << 1) | carry;
            carry = self.limbs[index] >> 63;
        }
        Self { limbs }
    }

    /// The quotient and remainder, by binary long division.
    fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        let mut quotient = Self::from(0u128);
        let mut remainder = Self::from(0u128);
        for index in (0..256).rev() {
            remainder = remainder.shifted_left_with(self.bit(index));
            if remainder >= *divisor {
                remainder = remainder.minus(divisor);
                quotient.limbs[index / 64] |= 1 << (index % 64);
            }
        }
        (quotient, remainder)
    }

    fn to_u128(self) -> Option<u128> {
        (self.limbs[2] == 0 && self.limbs[3] == 0)
            .then(|| (u128::from(self.limbs[1]) << 64) | u128::from(self.limbs[0]))
    }
}

impl PartialOrd for U256 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for U256 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.limbs.iter().rev().cmp(other.limbs.iter().rev())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn d(text: &str) -> Decimal {
        Decimal::parse(text).unwrap_or_else(|e| panic!("{text}: {e}"))
    }

    fn hash_of(value: &Decimal) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn parses_and_prints_with_its_scale() {
        for (text, printed, scale) in [
            ("1.50", "1.50", 2),
            ("-0.5", "-0.5", 1),
            (".5", "0.5", 1),
            ("5.", "5", 0),
            ("+12", "12", 0),
            ("0.000", "0.000", 3),
            ("-0", "0", 0),
            (
                "12345678901234567890.123456789012345678",
                "12345678901234567890.123456789012345678",
                18,
            ),
        ] {
            let value = d(text);
            assert_eq!(value.to_string(), printed, "{text}");
            assert_eq!(value.scale(), scale, "{text}");
        }
        for invalid in ["", "-", ".", "1.2.3", "1e5", "abc", "1,5", " 1", "1 "] {
            assert_eq!(
                Decimal::parse(invalid).unwrap_err(),
                DecimalError::Invalid,
                "{invalid:?}"
            );
        }
        // 39 digits, before or after the point.
        assert_eq!(
            Decimal::parse("123456789012345678901234567890123456789").unwrap_err(),
            DecimalError::Overflow
        );
        assert_eq!(
            Decimal::parse("0.123456789012345678901234567890123456789").unwrap_err(),
            DecimalError::Overflow
        );
        assert_eq!(d("99999999999999999999999999999999999999").scale(), 0);
    }

    #[test]
    fn a_precision_and_scale_round_the_fraction_and_bound_the_integer_digits() {
        for (text, precision, scale, expected) in [
            ("1.5", 5, 2, "1.50"),
            ("1.005", 5, 2, "1.01"),
            ("-1.005", 5, 2, "-1.01"),
            ("999.994", 5, 2, "999.99"),
            ("12", 2, 0, "12"),
            ("0.5", 1, 0, "1"),
            ("-0.4", 1, 0, "0"),
            (
                "0.12345",
                38,
                38,
                "0.12345000000000000000000000000000000000",
            ),
        ] {
            assert_eq!(
                d(text)
                    .with_precision(precision, scale)
                    .unwrap()
                    .to_string(),
                expected,
                "{text} as DECIMAL({precision},{scale})"
            );
        }
        // Too many integer digits, also once the fraction rounds up.
        for (text, precision, scale) in [
            ("1000", 5, 2),
            ("999.995", 5, 2),
            ("10", 1, 0),
            ("1", 38, 38),
        ] {
            assert_eq!(
                d(text).with_precision(precision, scale).unwrap_err(),
                DecimalError::Overflow,
                "{text} as DECIMAL({precision},{scale})"
            );
        }
        // A precision from 1 to 38 and a scale up to the precision.
        for (precision, scale) in [(0, 0), (39, 0), (5, 6)] {
            assert_eq!(
                d("1").with_precision(precision, scale).unwrap_err(),
                DecimalError::InvalidPrecision,
                "DECIMAL({precision},{scale})"
            );
        }
    }

    #[test]
    fn decimal_type_names_read_back_as_they_print() {
        let of = |precision, scale| Some(Some(DecimalType::new(precision, scale).unwrap()));
        for (name, expected) in [
            ("DECIMAL", Some(None)),
            ("dec", Some(None)),
            ("DECIMAL(10,2)", of(10, 2)),
            ("decimal( 10 , 2 )", of(10, 2)),
            ("DEC(3)", of(3, 0)),
            ("DECIMAL (38, 38)", of(38, 38)),
            ("DECIMAL(0)", None),
            ("DECIMAL(39,0)", None),
            ("DECIMAL(5,6)", None),
            ("DECIMAL(5,", None),
            ("DECIMAL(x)", None),
            ("DECIMALS", None),
            ("FLOAT64", None),
        ] {
            assert_eq!(DecimalType::from_name(name), expected, "{name}");
        }
        let money = DecimalType::new(6, 2).unwrap();
        assert_eq!(money.to_string(), "DECIMAL(6,2)");
        assert_eq!(
            DecimalType::from_name(&money.to_string()),
            Some(Some(money))
        );
        assert!(money.holds(&d("1234.50")));
        assert!(!money.holds(&d("1234.5")), "another scale");
        assert!(!money.holds(&d("12345.00")), "five integer digits");
        assert_eq!(money.fit(&d("1234.5")).unwrap().to_string(), "1234.50");
    }

    #[test]
    fn the_canonical_xsd_form_drops_trailing_zeros() {
        for (text, xsd) in [
            ("1.50", "1.5"),
            ("2", "2.0"),
            ("-0.050", "-0.05"),
            ("0", "0.0"),
            ("10.0", "10.0"),
        ] {
            assert_eq!(d(text).to_xsd_string(), xsd, "{text}");
        }
    }

    #[test]
    fn equal_values_at_other_scales_are_equal_and_hash_alike() {
        assert_eq!(d("1.50"), d("1.5"));
        assert_eq!(hash_of(&d("1.50")), hash_of(&d("1.5")));
        assert_eq!(d("0.000"), d("0"));
        assert_eq!(hash_of(&d("-0.00")), hash_of(&d("0")));
        assert_ne!(d("1.5"), d("1.05"));
    }

    #[test]
    fn normalize_drops_trailing_zeros_after_the_point_and_keeps_the_value() {
        for (text, normalized, scale) in [
            ("1.50", "1.5", 1),
            ("3.000", "3", 0),
            ("-0.120", "-0.12", 2),
            ("0.000", "0", 0),
            ("-0.00", "0", 0),
            ("100", "100", 0),
            ("1.05", "1.05", 2),
            (
                "0.12345678901234567890123456780",
                "0.1234567890123456789012345678",
                28,
            ),
        ] {
            let value = d(text);
            let once = value.normalize();
            assert_eq!(once.to_string(), normalized, "{text}");
            assert_eq!(once.scale(), scale, "{text}");
            assert_eq!(once, value, "{text}: the same value");
            assert_eq!(hash_of(&once), hash_of(&value), "{text}");
            assert_eq!(once.normalize().scale(), scale, "{text}: idempotent");
        }
    }

    #[test]
    fn values_order_across_scales_and_signs() {
        let mut values = [
            d("1.5"),
            d("-2"),
            d("1.49"),
            d("0"),
            d("-2.01"),
            d("100"),
            d("0.001"),
        ];
        values.sort();
        let printed: Vec<String> = values.iter().map(ToString::to_string).collect();
        assert_eq!(printed, ["-2.01", "-2", "0", "0.001", "1.49", "1.5", "100"]);
        assert!(
            d("99999999999999999999999999999999999999")
                > d("9999999999999999999.9999999999999999999")
        );
    }

    #[test]
    fn addition_and_subtraction_are_exact() {
        assert_eq!(d("0.1").checked_add(&d("0.2")).unwrap().to_string(), "0.3");
        assert_eq!(
            d("1.50").checked_add(&d("2.125")).unwrap().to_string(),
            "3.625"
        );
        assert_eq!(d("1").checked_sub(&d("1.01")).unwrap().to_string(), "-0.01");
        let max = d("99999999999999999999999999999999999999");
        assert_eq!(
            max.checked_add(&d("1")).unwrap_err(),
            DecimalError::Overflow
        );
        // Aligning the scales needs more than 38 digits.
        assert_eq!(
            max.checked_add(&d("0.1")).unwrap_err(),
            DecimalError::Overflow
        );
    }

    #[test]
    fn multiplication_adds_the_scales() {
        assert_eq!(
            d("1.5").checked_mul(&d("2.25")).unwrap().to_string(),
            "3.375"
        );
        assert_eq!(
            d("-0.1").checked_mul(&d("0.1")).unwrap().to_string(),
            "-0.01"
        );
        assert_eq!(
            d("10000000000000000000")
                .checked_mul(&d("10000000000000000000"))
                .unwrap_err(),
            DecimalError::Overflow
        );
        // A scale beyond 38 rounds half away from zero.
        let tiny = Decimal::new(5, 37).unwrap();
        let tenth = tiny.checked_mul(&d("0.1")).unwrap();
        assert_eq!((tenth.mantissa(), tenth.scale()), (5, 38));
        let hundredth = tiny.checked_mul(&d("0.01")).unwrap();
        assert_eq!((hundredth.mantissa(), hundredth.scale()), (1, 38));
        let negative = Decimal::new(-5, 37)
            .unwrap()
            .checked_mul(&d("0.01"))
            .unwrap();
        assert_eq!((negative.mantissa(), negative.scale()), (-1, 38));
    }

    #[test]
    fn division_keeps_sixteen_significant_digits() {
        for (a, b, quotient) in [
            ("1", "3", "0.3333333333333333"),
            ("2", "3", "0.6666666666666667"),
            ("-2", "3", "-0.6666666666666667"),
            ("10", "4", "2.500000000000000"),
            ("1.50", "0.5", "3.000000000000000"),
            ("0.0001", "3", "0.00003333333333333333"),
            ("12345", "7", "1763.571428571429"),
            ("1", "0.00000001", "100000000.00000000"),
        ] {
            assert_eq!(
                d(a).checked_div(&d(b)).unwrap().to_string(),
                quotient,
                "{a} / {b}"
            );
        }
        assert_eq!(
            d("1").checked_div(&d("0.00")).unwrap_err(),
            DecimalError::DivisionByZero
        );
        assert_eq!(
            d("99999999999999999999999999999999999999")
                .checked_div(&d("0.1"))
                .unwrap_err(),
            DecimalError::Overflow
        );
    }

    #[test]
    fn rescaling_rounds_half_away_from_zero() {
        for (text, scale, rescaled) in [
            ("2.345", 2, "2.35"),
            ("-2.345", 2, "-2.35"),
            ("2.344", 2, "2.34"),
            ("9.99", 1, "10.0"),
            ("1.5", 0, "2"),
            ("-1.5", 0, "-2"),
            ("1.5", 3, "1.500"),
        ] {
            assert_eq!(
                d(text).rescale(scale).unwrap().to_string(),
                rescaled,
                "{text} at {scale}"
            );
        }
        assert_eq!(
            d("99999999999999999999999999999999999999")
                .rescale(1)
                .unwrap_err(),
            DecimalError::Overflow
        );
    }

    #[test]
    fn conversions_to_and_from_integers_and_floats() {
        assert_eq!(Decimal::from_i64(-42).to_string(), "-42");
        assert_eq!(
            Decimal::from_i64(i64::MIN).to_string(),
            i64::MIN.to_string()
        );
        assert_eq!(Decimal::from_f64(0.1).unwrap().to_string(), "0.1");
        assert_eq!(Decimal::from_f64(-2.5).unwrap().to_string(), "-2.5");
        assert_eq!(
            Decimal::from_f64(1e20).unwrap().to_string(),
            "100000000000000000000"
        );
        assert_eq!(
            Decimal::from_f64(f64::NAN).unwrap_err(),
            DecimalError::Invalid
        );
        assert_eq!(
            Decimal::from_f64(f64::INFINITY).unwrap_err(),
            DecimalError::Invalid
        );
        assert_eq!(Decimal::from_f64(1e40).unwrap_err(), DecimalError::Overflow);
        assert!((d("0.1").to_f64() - 0.1).abs() < f64::EPSILON);
        assert_eq!(d("-2.75").to_f64(), -2.75);
        assert_eq!(d("-2.75").trunc_to_i64(), Some(-2));
        assert_eq!(d("2.75").trunc_to_i64(), Some(2));
        assert_eq!(d("99999999999999999999").trunc_to_i64(), None);
    }

    #[test]
    fn signs_remainders_and_whole_numbers() {
        assert_eq!(d("1.50").neg().to_string(), "-1.50");
        assert_eq!(d("-2.345").abs().to_string(), "2.345");
        assert_eq!(d("-2.345").signum(), -1);
        assert_eq!(d("0.00").signum(), 0);
        assert_eq!(d("7.5").checked_rem(&d("2")).unwrap().to_string(), "1.5");
        assert_eq!(d("-7.5").checked_rem(&d("2")).unwrap().to_string(), "-1.5");
        assert_eq!(d("1.50").checked_rem(&d("1")).unwrap().to_string(), "0.50");
        assert_eq!(
            d("1").checked_rem(&d("0")).unwrap_err(),
            DecimalError::DivisionByZero
        );
        for (text, floor, ceil, round) in [
            ("2.345", "2", "3", "2"),
            ("-2.345", "-3", "-2", "-2"),
            ("2.5", "2", "3", "3"),
            ("-2.5", "-3", "-2", "-3"),
            ("4", "4", "4", "4"),
            ("4.00", "4", "4", "4"),
        ] {
            assert_eq!(d(text).floor().to_string(), floor, "floor {text}");
            assert_eq!(d(text).ceil().to_string(), ceil, "ceil {text}");
            assert_eq!(d(text).round().to_string(), round, "round {text}");
        }
    }

    /// A float compares by its shortest round-trip text, so `1.5` equals the
    /// float `1.5`; NaN comes after every decimal, and the infinities and
    /// floats past 38 digits order by their sign.
    #[test]
    fn decimals_compare_with_floats() {
        assert_eq!(d("1.5").cmp_f64(1.5), Ordering::Equal);
        assert_eq!(d("0.1").cmp_f64(0.1), Ordering::Equal);
        assert_eq!(d("1.49").cmp_f64(1.5), Ordering::Less);
        assert_eq!(d("-1").cmp_f64(-1.5), Ordering::Greater);
        assert_eq!(
            d("99999999999999999999999999999999999999").cmp_f64(1e40),
            Ordering::Less
        );
        assert_eq!(d("0").cmp_f64(-1e40), Ordering::Greater);
        assert_eq!(d("0").cmp_f64(f64::NAN), Ordering::Less);
        assert_eq!(d("0").cmp_f64(f64::INFINITY), Ordering::Less);
        assert_eq!(d("0").cmp_f64(f64::NEG_INFINITY), Ordering::Greater);
        // Below 38 digits after the point a float equals no decimal, but
        // orders on its own side of the decimal it rounds to.
        assert_eq!(d("0").cmp_f64(1e-300), Ordering::Less);
        assert_eq!(d("0").cmp_f64(-1e-300), Ordering::Greater);
        assert_eq!(d("0").cmp_f64(-0.0), Ordering::Equal);
        let smallest = d("0.00000000000000000000000000000000000001");
        assert_eq!(smallest.cmp_f64(0.6e-38), Ordering::Greater);
        assert_eq!(smallest.cmp_f64(1.4e-38), Ordering::Less);
        assert_eq!(smallest.cmp_f64(1e-38), Ordering::Equal);
    }

    #[test]
    fn the_type_is_eight_byte_aligned_and_small() {
        assert_eq!(std::mem::align_of::<Decimal>(), 8);
        assert_eq!(std::mem::size_of::<Decimal>(), 24);
    }
}
