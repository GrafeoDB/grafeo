//! How a comparison reads a zoned datetime or time against a local one.

use std::cmp::Ordering;

use super::{Timestamp, Value};

/// How `=`, `<` and the other comparisons read a zoned value (a ZONED
/// DATETIME, or a time with an offset) against a local one of the same kind
/// (a LOCAL DATETIME, or a time without one). Two zoned or two local values
/// compare the same way in every mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalComparison {
    /// The local value is at this offset from UTC, in seconds, as SQL and ISO
    /// GQL read it in the session time zone (UTC, an offset of 0, without one).
    LocalAt(i32),
    /// A zoned and a local value are of different types, as openCypher
    /// compares them: never equal, and not ordered.
    TypesApart,
}

impl Default for TemporalComparison {
    fn default() -> Self {
        Self::LocalAt(0)
    }
}

/// The largest offset ISO GQL and SQL allow: 14 hours.
const MAX_DISPLACEMENT_SECONDS: i32 = 14 * 3600;

impl TemporalComparison {
    /// The offset from UTC, in seconds, that a time zone displacement names:
    /// `+01:00`, `-05:30`, `Z` or `UTC`, at most 14 hours either way (ISO
    /// GQL's time zone displacement). `None` for anything else, a named zone
    /// such as `Europe/Amsterdam` included.
    #[must_use]
    pub fn parse_displacement(text: &str) -> Option<i32> {
        let text = text.trim();
        if text == "Z" || text.eq_ignore_ascii_case("UTC") {
            return Some(0);
        }
        let (sign, rest) = match text.as_bytes().first()? {
            b'+' => (1, &text[1..]),
            b'-' => (-1, &text[1..]),
            _ => return None,
        };
        let (hours, minutes) = rest.split_once(':')?;
        let two_digits = |part: &str| part.len() == 2 && part.bytes().all(|b| b.is_ascii_digit());
        if !two_digits(hours) || !two_digits(minutes) {
            return None;
        }
        let minutes: i32 = minutes.parse().ok()?;
        let seconds = hours.parse::<i32>().ok()? * 3600 + minutes * 60;
        (minutes < 60 && seconds <= MAX_DISPLACEMENT_SECONDS).then_some(sign * seconds)
    }

    /// Whether `a` and `b` are a zoned and a local value of one kind: a
    /// zoned and a local datetime, or a time with an offset and one without.
    #[must_use]
    pub fn is_mixed_pair(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Timestamp(_), Value::ZonedDatetime(_))
            | (Value::ZonedDatetime(_), Value::Timestamp(_)) => true,
            (Value::Time(a), Value::Time(b)) => {
                a.offset_seconds().is_some() != b.offset_seconds().is_some()
            }
            _ => false,
        }
    }

    /// `value` as this mode reads it against zoned values: under
    /// [`LocalAt`](Self::LocalAt) a local datetime is the zoned datetime of
    /// its instant at the mode's offset, and a time without an offset is that
    /// time at the offset, inside lists, maps and paths too; under
    /// [`TypesApart`](Self::TypesApart) every value is itself. Grouping,
    /// DISTINCT and ORDER BY read a value so, which makes them agree with
    /// `=` and `<` (see [`compare_mixed`](Self::compare_mixed)).
    #[must_use]
    pub fn normalize(self, value: &Value) -> std::borrow::Cow<'_, Value> {
        use std::borrow::Cow;
        let Self::LocalAt(offset) = self else {
            return Cow::Borrowed(value);
        };
        match value {
            Value::Timestamp(local) => Cow::Owned(Value::ZonedDatetime(
                super::ZonedDatetime::from_timestamp_offset(
                    Timestamp::from_micros(local.as_micros() - i64::from(offset) * 1_000_000),
                    0,
                ),
            )),
            Value::Time(time) if time.offset_seconds().is_none() => {
                Cow::Owned(Value::Time(time.with_offset(offset)))
            }
            Value::List(items) => match self.normalize_all(items) {
                Some(items) => Cow::Owned(Value::List(items.into())),
                None => Cow::Borrowed(value),
            },
            Value::Map(map) => {
                if map
                    .values()
                    .all(|item| matches!(self.normalize(item), Cow::Borrowed(_)))
                {
                    return Cow::Borrowed(value);
                }
                Cow::Owned(Value::Map(std::sync::Arc::new(
                    map.iter()
                        .map(|(key, item)| (key.clone(), self.normalize(item).into_owned()))
                        .collect(),
                )))
            }
            Value::Path { nodes, edges } => {
                match (self.normalize_all(nodes), self.normalize_all(edges)) {
                    (None, None) => Cow::Borrowed(value),
                    (new_nodes, new_edges) => Cow::Owned(Value::Path {
                        nodes: new_nodes.map_or_else(|| nodes.clone(), Into::into),
                        edges: new_edges.map_or_else(|| edges.clone(), Into::into),
                    }),
                }
            }
            _ => Cow::Borrowed(value),
        }
    }

    /// `items` normalized, or `None` when none of them changes.
    fn normalize_all(self, items: &[Value]) -> Option<Vec<Value>> {
        use std::borrow::Cow;
        let first = items
            .iter()
            .position(|item| matches!(self.normalize(item), Cow::Owned(_)))?;
        let mut normalized = items[..first].to_vec();
        normalized.extend(
            items[first..]
                .iter()
                .map(|item| self.normalize(item).into_owned()),
        );
        Some(normalized)
    }

    /// Orders a zoned and a local value of one kind (see
    /// [`is_mixed_pair`](Self::is_mixed_pair)) as this mode reads them: the
    /// local one at the mode's offset, or `None` when the types stay apart.
    /// `None` for any other pair.
    #[must_use]
    pub fn compare_mixed(self, a: &Value, b: &Value) -> Option<Ordering> {
        let Self::LocalAt(offset) = self else {
            return None;
        };
        let at_offset = |local: Timestamp| {
            Timestamp::from_micros(local.as_micros() - i64::from(offset) * 1_000_000)
        };
        match (a, b) {
            (Value::Timestamp(local), Value::ZonedDatetime(zoned)) => {
                Some(at_offset(*local).cmp(&zoned.as_timestamp()))
            }
            (Value::ZonedDatetime(zoned), Value::Timestamp(local)) => {
                Some(zoned.as_timestamp().cmp(&at_offset(*local)))
            }
            (Value::Time(a), Value::Time(b))
                if Self::is_mixed_pair(&Value::Time(*a), &Value::Time(*b)) =>
            {
                let zoned = |time: super::Time| match time.offset_seconds() {
                    Some(_) => time,
                    None => time.with_offset(offset),
                };
                Some(zoned(*a).cmp(&zoned(*b)))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Time, ZonedDatetime};

    fn local(text: &str) -> Value {
        let (date, time) = text.split_once('T').unwrap();
        Value::Timestamp(Timestamp::from_date_time(
            crate::types::Date::parse(date).unwrap(),
            Time::parse(time).unwrap(),
        ))
    }

    fn zoned(text: &str) -> Value {
        Value::ZonedDatetime(ZonedDatetime::parse(text).unwrap())
    }

    fn time(text: &str) -> Value {
        Value::Time(Time::parse(text).unwrap())
    }

    #[test]
    fn displacements_are_offsets_only() {
        assert_eq!(TemporalComparison::parse_displacement("+01:00"), Some(3600));
        assert_eq!(
            TemporalComparison::parse_displacement("-05:30"),
            Some(-19_800)
        );
        assert_eq!(TemporalComparison::parse_displacement("Z"), Some(0));
        assert_eq!(TemporalComparison::parse_displacement("utc"), Some(0));
        assert_eq!(
            TemporalComparison::parse_displacement(" +14:00 "),
            Some(50_400)
        );
        for text in [
            "Europe/Amsterdam",
            "+1:00",
            "+01",
            "+0100",
            "+14:01",
            "+01:60",
            "01:00",
            "",
        ] {
            assert_eq!(TemporalComparison::parse_displacement(text), None, "{text}");
        }
    }

    #[test]
    fn a_local_datetime_reads_at_the_mode_offset() {
        let at_utc = TemporalComparison::default();
        let at_one = TemporalComparison::LocalAt(3600);
        let ten_at_one = zoned("2024-03-15T10:00:00+01:00");
        assert_eq!(
            at_utc.compare_mixed(&ten_at_one, &local("2024-03-15T09:00:00")),
            Some(Ordering::Equal)
        );
        assert_eq!(
            at_one.compare_mixed(&ten_at_one, &local("2024-03-15T10:00:00")),
            Some(Ordering::Equal)
        );
        assert_eq!(
            at_one.compare_mixed(&local("2024-03-15T09:00:00"), &ten_at_one),
            Some(Ordering::Less)
        );
    }

    #[test]
    fn a_local_time_reads_at_the_mode_offset() {
        let at_one = TemporalComparison::LocalAt(3600);
        assert_eq!(
            at_one.compare_mixed(&time("10:00:00+01:00"), &time("10:00:00")),
            Some(Ordering::Equal)
        );
        assert_eq!(
            TemporalComparison::default().compare_mixed(&time("10:00:00"), &time("10:00:00+01:00")),
            Some(Ordering::Greater)
        );
    }

    #[test]
    fn types_apart_order_nothing() {
        let apart = TemporalComparison::TypesApart;
        assert_eq!(
            apart.compare_mixed(
                &zoned("2024-03-15T09:00:00Z"),
                &local("2024-03-15T09:00:00")
            ),
            None
        );
        assert_eq!(
            apart.compare_mixed(&time("10:00:00Z"), &time("10:00:00")),
            None
        );
    }

    #[test]
    fn only_zoned_and_local_pairs_are_mixed() {
        assert!(TemporalComparison::is_mixed_pair(
            &zoned("2024-03-15T09:00:00Z"),
            &local("2024-03-15T09:00:00")
        ));
        assert!(TemporalComparison::is_mixed_pair(
            &time("10:00:00Z"),
            &time("10:00:00")
        ));
        assert!(!TemporalComparison::is_mixed_pair(
            &time("10:00:00Z"),
            &time("11:00:00+01:00")
        ));
        assert!(!TemporalComparison::is_mixed_pair(
            &local("2024-03-15T09:00:00"),
            &local("2024-03-15T09:00:00")
        ));
        assert!(!TemporalComparison::is_mixed_pair(
            &Value::Int64(1),
            &time("10:00:00")
        ));
        assert_eq!(
            TemporalComparison::default().compare_mixed(&Value::Int64(1), &Value::Int64(2)),
            None
        );
    }
}
