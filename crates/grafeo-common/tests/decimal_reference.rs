//! Decimal division and decimal sums against an exact reference (#669),
//! `num-bigint`, over random operands of every scale and sign.
//!
//! - `Decimal::checked_div`: the exact quotient put through the documented
//!   rule: at least 16 significant digits, at least the larger operand
//!   scale, at most 38 digits, rounded half away from zero, an error when the
//!   integer part needs more than 38 digits.
//! - `DecimalSum`, in every order of the rows: `total` is the exact sum at
//!   the largest row scale, its fraction rounded half away from zero to fit
//!   next to the integer part, an error only when the integer part needs more
//!   than 38 digits; `average` is the exact sum divided by the count by the
//!   division rule, with the sum's scale as the least scale, and never fails.

use grafeo_common::types::{Decimal, DecimalError, DecimalSum};
use num_bigint::BigInt;
use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

const COEFFICIENT_LIMIT: i128 = 10_i128.pow(38);
const MAX_DIGITS: u32 = 38;
const SIGNIFICANT: u32 = 16;

fn pow10(exponent: u32) -> BigInt {
    BigInt::from(10).pow(exponent)
}

fn digits(value: &BigInt) -> u32 {
    if *value == BigInt::from(0) {
        0
    } else {
        u32::try_from(value.to_string().trim_start_matches('-').len()).unwrap()
    }
}

/// `numerator / denominator` (both positive) times `10^scale`, rounded half
/// away from zero.
fn rounded_quotient(numerator: &BigInt, denominator: &BigInt, scale: u32) -> BigInt {
    let scaled = numerator * pow10(scale);
    let quotient = &scaled / denominator;
    let remainder = &scaled % denominator;
    if remainder * 2 >= *denominator {
        quotient + 1
    } else {
        quotient
    }
}

/// `numerator / denominator` (magnitudes, the result negative when
/// `negative`) by the division rule, with `least_scale` as the least scale.
fn by_division_rule(
    negative: bool,
    numerator: &BigInt,
    denominator: &BigInt,
    least_scale: u32,
) -> Result<(i128, u8), DecimalError> {
    let zero = BigInt::from(0);
    let integer = numerator / denominator;
    if integer >= pow10(MAX_DIGITS) {
        return Err(DecimalError::Overflow);
    }
    let integer_digits = digits(&integer);
    let wanted = if integer > zero {
        SIGNIFICANT.saturating_sub(integer_digits)
    } else {
        // The zeros after the point before the first other digit, among
        // the first 39 fraction digits; none at all keeps scale 0.
        let fraction = (numerator % denominator) * pow10(39) / denominator;
        if fraction == zero {
            0
        } else {
            39 - digits(&fraction) + SIGNIFICANT
        }
    };
    let scale = wanted.max(least_scale).min(MAX_DIGITS - integer_digits);
    let magnitude = rounded_quotient(numerator, denominator, scale);
    if magnitude >= pow10(MAX_DIGITS) {
        return Err(DecimalError::Overflow);
    }
    let magnitude = i128::try_from(&magnitude).unwrap();
    let mantissa = if negative { -magnitude } else { magnitude };
    Ok((mantissa, u8::try_from(scale).unwrap()))
}

fn reference_quotient(left: (i128, u8), right: (i128, u8)) -> Result<(i128, u8), DecimalError> {
    if right.0 == 0 {
        return Err(DecimalError::DivisionByZero);
    }
    // (a / 10^as) / (b / 10^bs) = (a * 10^bs) / (b * 10^as), as magnitudes.
    by_division_rule(
        (left.0 < 0) != (right.0 < 0),
        &(BigInt::from(left.0.unsigned_abs()) * pow10(u32::from(right.1))),
        &(BigInt::from(right.0.unsigned_abs()) * pow10(u32::from(left.1))),
        u32::from(left.1.max(right.1)),
    )
}

/// The exact sum of `values` as a mantissa at the largest scale, and that
/// scale (0 for no values).
fn exact_sum(values: &[(i128, u8)]) -> (BigInt, u32) {
    let scale = values
        .iter()
        .map(|value| u32::from(value.1))
        .max()
        .unwrap_or(0);
    let sum = values
        .iter()
        .map(|(mantissa, value_scale)| {
            BigInt::from(*mantissa) * pow10(scale - u32::from(*value_scale))
        })
        .sum();
    (sum, scale)
}

fn reference_total(values: &[(i128, u8)]) -> Result<(i128, u8), DecimalError> {
    let (sum, scale) = exact_sum(values);
    let negative = sum < BigInt::from(0);
    let magnitude = if negative { -sum } else { sum };
    // The largest scale at which the rounded value fits, as multiplication.
    for kept in (0..=scale).rev() {
        let rounded = rounded_quotient(&magnitude, &pow10(scale), kept);
        if rounded < pow10(MAX_DIGITS) {
            let rounded = i128::try_from(&rounded).unwrap();
            return Ok((
                if negative { -rounded } else { rounded },
                u8::try_from(kept).unwrap(),
            ));
        }
    }
    Err(DecimalError::Overflow)
}

fn reference_average(values: &[(i128, u8)]) -> Result<(i128, u8), DecimalError> {
    let (sum, scale) = exact_sum(values);
    let negative = sum < BigInt::from(0);
    let magnitude = if negative { -sum } else { sum };
    let count = BigInt::from(values.len());
    by_division_rule(negative, &magnitude, &(count * pow10(scale)), scale)
}

fn decimal_input() -> impl Strategy<Value = (i128, u8)> {
    let coefficient = prop_oneof![
        4 => prop::sample::select(vec![
            0, 1, -1, 3, -3, 7, 5, -5,
            COEFFICIENT_LIMIT - 1, -(COEFFICIENT_LIMIT - 1),
            COEFFICIENT_LIMIT / 2, -COEFFICIENT_LIMIT / 2,
            10_i128.pow(19) - 1, 10_i128.pow(19) + 1,
            10_i128.pow(37) - 1, 10_i128.pow(37) + 1,
        ]),
        6 => (-COEFFICIENT_LIMIT + 1)..COEFFICIENT_LIMIT,
    ];
    let scale = prop_oneof![
        4 => prop::sample::select(vec![0_u8, 1, 16, 18, 19, 20, 37, 38]),
        6 => 0_u8..=38,
    ];
    (coefficient, scale)
}

fn decimal(value: (i128, u8)) -> Decimal {
    Decimal::new(value.0, value.1).unwrap()
}

fn as_pair(value: Result<Decimal, DecimalError>) -> Result<(i128, u8), DecimalError> {
    value.map(|value| (value.mantissa(), value.scale()))
}

fn quotient(left: (i128, u8), right: (i128, u8)) -> Result<(i128, u8), DecimalError> {
    as_pair(decimal(left).checked_div(&decimal(right)))
}

fn summed(values: &[(i128, u8)]) -> DecimalSum {
    let mut sum = DecimalSum::new();
    for value in values {
        sum.add(&decimal(*value));
    }
    sum
}

#[test]
fn the_reference_agrees_with_the_documented_examples() {
    for (left, right, expected) in [
        ((1, 0), (3, 0), Ok((3_333_333_333_333_333, 16))),
        ((150, 2), (2, 0), Ok((7_500_000_000_000_000, 16))),
        ((1, 0), (0, 0), Err(DecimalError::DivisionByZero)),
        (
            (COEFFICIENT_LIMIT - 1, 0),
            (1, 1),
            Err(DecimalError::Overflow),
        ),
    ] {
        assert_eq!(
            reference_quotient(left, right),
            expected,
            "{left:?} / {right:?}"
        );
        assert_eq!(quotient(left, right), expected, "{left:?} / {right:?}");
    }
    // The issue's sum, in its three orders, and the average of two of the
    // largest integers.
    let a = (COEFFICIENT_LIMIT / 10 * 10 - 10, 1); // 37 nines and a zero decimal
    let minus_a = (-a.0, 1);
    for values in [[a, a, minus_a], [a, minus_a, a], [minus_a, a, a]] {
        assert_eq!(reference_total(&values), Ok(a));
        assert_eq!(as_pair(summed(&values).total()), Ok(a), "{values:?}");
    }
    let m = (COEFFICIENT_LIMIT - 1, 0);
    assert_eq!(
        as_pair(summed(&[m, m]).total()),
        Err(DecimalError::Overflow)
    );
    assert_eq!(
        as_pair(summed(&[m, m]).average(2)),
        reference_average(&[m, m])
    );
    assert_eq!(
        reference_average(&[m, m]).map(|(mantissa, _)| mantissa),
        Ok(m.0)
    );
}

/// An integer sum continues as a decimal sum exactly, also past 38 digits:
/// `i128::MAX` has 39, and the next rows bring it back.
#[test]
fn an_integer_sum_continues_exactly() {
    let mut sum = DecimalSum::from_i128(i128::MAX);
    assert_eq!(sum.total(), Err(DecimalError::Overflow));
    sum.add(&decimal((-(COEFFICIENT_LIMIT - 1), 0)));
    sum.add(&decimal((5, 1)));
    // 38 integer digits and .5: no room for the fraction, so it rounds
    // half away from zero to scale 0.
    let expected = BigInt::from(i128::MAX) - BigInt::from(COEFFICIENT_LIMIT - 1) + 1;
    assert_eq!(
        as_pair(sum.total()),
        Ok((i128::try_from(expected).unwrap(), 0))
    );
    let mut negative = DecimalSum::from_i128(i128::MIN);
    negative.add(&decimal((COEFFICIENT_LIMIT - 1, 0)));
    negative.add(&decimal((COEFFICIENT_LIMIT - 1, 0)));
    let expected = BigInt::from(i128::MIN) + BigInt::from(COEFFICIENT_LIMIT - 1) * 2;
    assert_eq!(
        as_pair(negative.total()),
        Ok((i128::try_from(expected).unwrap(), 0))
    );
}

/// No rows: a total of zero, and no average.
#[test]
fn an_empty_sum() {
    let sum = DecimalSum::new();
    assert_eq!(sum.total().map(|value| value.mantissa()), Ok(0));
    assert_eq!(sum.average(0), Err(DecimalError::DivisionByZero));
}

proptest! {
    #![proptest_config(Config {
        cases: 1024,
        failure_persistence: None,
        rng_seed: RngSeed::Fixed(0x6690_0000_0000_0002),
        ..Config::default()
    })]

    #[test]
    fn division_matches_exact_arithmetic(left in decimal_input(), right in decimal_input()) {
        prop_assert_eq!(
            quotient(left, right),
            reference_quotient(left, right),
            "left={:?}, right={:?}", left, right
        );
    }

    #[test]
    fn a_sum_and_its_average_match_exact_arithmetic_in_any_order(
        (values, shuffled) in prop::collection::vec(decimal_input(), 1..=12)
            .prop_flat_map(|values| (Just(values.clone()), Just(values).prop_shuffle()))
    ) {
        let total = reference_total(&values);
        let average = reference_average(&values);
        prop_assert!(average.is_ok(), "an average never fails: {:?}", values);
        for order in [&values, &shuffled] {
            let sum = summed(order);
            prop_assert_eq!(as_pair(sum.total()), total, "values={:?}", order);
            prop_assert_eq!(
                as_pair(sum.average(u64::try_from(order.len()).unwrap())),
                average,
                "values={:?}", order
            );
        }
    }

    /// Many rows that cancel, so that the total fits while running totals
    /// do not: the case the order used to decide.
    #[test]
    fn cancelling_rows_sum_exactly_in_any_order(
        (values, shuffled) in prop::collection::vec(decimal_input(), 1..=6)
            .prop_flat_map(|values| {
                let mut both = values.clone();
                both.extend(values.iter().map(|(mantissa, scale)| (-mantissa, *scale)));
                both.extend(values.iter().take(1).copied());
                (Just(both.clone()), Just(both).prop_shuffle())
            })
    ) {
        let total = reference_total(&values);
        prop_assert!(total.is_ok(), "the rows cancel to one row: {:?}", values);
        for order in [&values, &shuffled] {
            prop_assert_eq!(as_pair(summed(order).total()), total, "values={:?}", order);
        }
    }
}
