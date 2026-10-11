//! The components of temporal values, read like properties: `d.year`,
//! `t.hour`, `dt.epochMillis`, `dur.minutesOfHour`, ... (the temporal
//! accessors of Cypher). The GQL component functions (`year(d)`, ...) read
//! the same values.

use super::{Date, Duration, Time, Timestamp, Value};

const NANOS_PER_SECOND: i64 = 1_000_000_000;

impl Value {
    /// The component `name` of a date, time, datetime or duration value, or
    /// `None` when this is no temporal value or has no such component. Names
    /// are compared without case.
    ///
    /// - Date (and the date of a datetime): `year`, `quarter`, `month`, `week`
    ///   and `weekYear` (ISO 8601), `day`, `ordinalDay`, `dayOfWeek` (Monday
    ///   is 1), `dayOfQuarter`.
    /// - Time (and the time of a datetime): `hour`, `minute`, `second`,
    ///   `millisecond`, `microsecond` and `nanosecond` (each of the second);
    ///   with an offset also `offset` and `timezone` (`+02:00`, `Z`),
    ///   `offsetMinutes` and `offsetSeconds`.
    /// - Datetime: `epochSeconds` and `epochMillis` (after
    ///   1970-01-01T00:00:00Z); a zoned datetime reads its date and time
    ///   at its own offset.
    /// - Duration: `years`, `quarters`, `months`, `weeks`, `days`, `hours`,
    ///   `minutes`, `seconds`, `milliseconds`, `microseconds`, `nanoseconds`
    ///   (each the whole duration in that unit, of its month, day or time
    ///   part), and `quartersOfYear`, `monthsOfQuarter`, `monthsOfYear`,
    ///   `daysOfWeek`, `minutesOfHour`, `secondsOfMinute`,
    ///   `millisecondsOfSecond`, `microsecondsOfSecond`, `nanosecondsOfSecond`.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_common::types::{Date, Value};
    ///
    /// let date = Value::Date(Date::from_ymd(2020, 5, 25).unwrap());
    /// assert_eq!(date.temporal_component("month"), Some(Value::Int64(5)));
    /// assert_eq!(date.temporal_component("dayOfWeek"), Some(Value::Int64(1)));
    /// assert_eq!(date.temporal_component("hour"), None);
    /// ```
    #[must_use]
    pub fn temporal_component(&self, name: &str) -> Option<Value> {
        let name = name.to_ascii_lowercase();
        let name = name.as_str();
        match self {
            Value::Date(date) => date_component(*date, name),
            Value::Time(time) => time_component(*time, name),
            Value::Timestamp(timestamp) => instant_component(*timestamp, name)
                .or_else(|| date_component(timestamp.to_date(), name))
                .or_else(|| clock_component(timestamp.to_time(), name)),
            Value::ZonedDatetime(zoned) => instant_component(zoned.as_timestamp(), name)
                .or_else(|| date_component(zoned.to_local_date(), name))
                .or_else(|| time_component(zoned.to_local_time(), name)),
            Value::Duration(duration) => duration_component(*duration, name),
            _ => None,
        }
    }
}

fn int(value: impl Into<i64>) -> Option<Value> {
    Some(Value::Int64(value.into()))
}

/// The ISO 8601 day of the week of the date `days` after 1970-01-01 (a
/// Thursday): Monday is 1, Sunday 7.
fn iso_day_of_week(days: i64) -> i64 {
    (days + 3).rem_euclid(7) + 1
}

/// The ISO 8601 week-based year and week of `date`: the week (Monday to
/// Sunday) belongs to the year of its Thursday.
fn iso_week(date: Date) -> Option<(i32, i64)> {
    let days = i64::from(date.as_days());
    let thursday = Date::from_days(i32::try_from(days - iso_day_of_week(days) + 4).ok()?);
    let week_year = thursday.year();
    let first_day = i64::from(Date::from_ymd(week_year, 1, 1)?.as_days());
    Some((
        week_year,
        (i64::from(thursday.as_days()) - first_day) / 7 + 1,
    ))
}

fn date_component(date: Date, name: &str) -> Option<Value> {
    let (year, month, day) = date.to_ymd();
    let days = i64::from(date.as_days());
    match name {
        "year" => int(year),
        "quarter" => int((month - 1) / 3 + 1),
        "month" => int(month),
        "day" => int(day),
        "ordinalday" => int(days - i64::from(Date::from_ymd(year, 1, 1)?.as_days()) + 1),
        "dayofquarter" => {
            let first_month = (month - 1) / 3 * 3 + 1;
            int(days - i64::from(Date::from_ymd(year, first_month, 1)?.as_days()) + 1)
        }
        "dayofweek" => int(iso_day_of_week(days)),
        "week" => int(iso_week(date)?.1),
        "weekyear" => int(iso_week(date)?.0),
        _ => None,
    }
}

/// The components of a time of day, without its offset.
fn clock_component(time: Time, name: &str) -> Option<Value> {
    let nanosecond = time.nanosecond();
    match name {
        "hour" => int(time.hour()),
        "minute" => int(time.minute()),
        "second" => int(time.second()),
        "millisecond" => int(nanosecond / 1_000_000),
        "microsecond" => int(nanosecond / 1_000),
        "nanosecond" => int(nanosecond),
        _ => None,
    }
}

/// The components of a time of day, with its offset when it has one.
fn time_component(time: Time, name: &str) -> Option<Value> {
    clock_component(time, name).or_else(|| {
        let offset = time.offset_seconds()?;
        match name {
            "offset" | "timezone" => Some(Value::String(offset_text(offset).into())),
            "offsetminutes" => int(offset / 60),
            "offsetseconds" => int(offset),
            _ => None,
        }
    })
}

/// An offset from UTC as text: `Z`, `+02:00`, or `-03:30:19` with seconds.
fn offset_text(offset: i32) -> String {
    if offset == 0 {
        return "Z".to_string();
    }
    let sign = if offset > 0 { '+' } else { '-' };
    let seconds = offset.unsigned_abs();
    let (hours, minutes, seconds) = (seconds / 3600, seconds % 3600 / 60, seconds % 60);
    if seconds == 0 {
        format!("{sign}{hours:02}:{minutes:02}")
    } else {
        format!("{sign}{hours:02}:{minutes:02}:{seconds:02}")
    }
}

/// The components of an instant: the time since the Unix epoch.
fn instant_component(instant: Timestamp, name: &str) -> Option<Value> {
    match name {
        "epochmillis" => int(instant.as_micros().div_euclid(1_000)),
        "epochseconds" => int(instant.as_micros().div_euclid(1_000_000)),
        _ => None,
    }
}

fn duration_component(duration: Duration, name: &str) -> Option<Value> {
    let months = duration.months();
    let days = duration.days();
    let nanos = duration.nanos();
    let seconds = nanos.div_euclid(NANOS_PER_SECOND);
    let nanos_of_second = nanos.rem_euclid(NANOS_PER_SECOND);
    match name {
        "years" => int(months / 12),
        "quarters" => int(months / 3),
        "months" => int(months),
        "weeks" => int(days / 7),
        "days" => int(days),
        "hours" => int(seconds / 3600),
        "minutes" => int(seconds / 60),
        "seconds" => int(seconds),
        "milliseconds" => int(nanos.div_euclid(1_000_000)),
        "microseconds" => int(nanos.div_euclid(1_000)),
        "nanoseconds" => int(nanos),
        "quartersofyear" => int(months / 3 % 4),
        "monthsofquarter" => int(months % 3),
        "monthsofyear" => int(months % 12),
        "daysofweek" => int(days % 7),
        "minutesofhour" => int(seconds / 60 % 60),
        "secondsofminute" => int(seconds % 60),
        "millisecondsofsecond" => int(nanos_of_second / 1_000_000),
        "microsecondsofsecond" => int(nanos_of_second / 1_000),
        "nanosecondsofsecond" => int(nanos_of_second),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ZonedDatetime;

    fn date(year: i32, month: u32, day: u32) -> Value {
        Value::Date(Date::from_ymd(year, month, day).unwrap())
    }

    #[test]
    fn iso_weeks_at_the_turn_of_the_year() {
        // 2020-12-31 (Thursday) and 2021-01-03 (Sunday) are in week 53 of
        // 2020; 2021-01-04 (Monday) starts week 1 of 2021; 2019-12-30
        // (Monday) is in week 1 of 2020.
        for (value, week_year, week, day_of_week) in [
            (date(2020, 12, 31), 2020, 53, 4),
            (date(2021, 1, 3), 2020, 53, 7),
            (date(2021, 1, 4), 2021, 1, 1),
            (date(2019, 12, 30), 2020, 1, 1),
            (date(1970, 1, 1), 1970, 1, 4),
            (date(1969, 12, 28), 1969, 52, 7),
        ] {
            assert_eq!(
                value.temporal_component("weekYear"),
                Some(Value::Int64(week_year)),
                "{value:?}"
            );
            assert_eq!(
                value.temporal_component("week"),
                Some(Value::Int64(week)),
                "{value:?}"
            );
            assert_eq!(
                value.temporal_component("dayOfWeek"),
                Some(Value::Int64(day_of_week)),
                "{value:?}"
            );
        }
    }

    #[test]
    fn day_counts_within_the_year_and_quarter() {
        // 2020 is a leap year: 2020-12-31 is its 366th day.
        let last = date(2020, 12, 31);
        assert_eq!(
            last.temporal_component("ordinalDay"),
            Some(Value::Int64(366))
        );
        assert_eq!(last.temporal_component("quarter"), Some(Value::Int64(4)));
        assert_eq!(
            last.temporal_component("dayOfQuarter"),
            Some(Value::Int64(92))
        );
        let first = date(2019, 1, 1);
        assert_eq!(
            first.temporal_component("ordinalDay"),
            Some(Value::Int64(1))
        );
        assert_eq!(
            first.temporal_component("dayOfQuarter"),
            Some(Value::Int64(1))
        );
    }

    #[test]
    fn epoch_components_before_the_epoch_round_down() {
        // 1969-12-31T23:59:59.912Z
        let instant = Value::Timestamp(Timestamp::from_micros(-88_000));
        assert_eq!(
            instant.temporal_component("epochMillis"),
            Some(Value::Int64(-88))
        );
        assert_eq!(
            instant.temporal_component("epochSeconds"),
            Some(Value::Int64(-1))
        );
        assert_eq!(instant.temporal_component("second"), Some(Value::Int64(59)));
        assert_eq!(
            instant.temporal_component("millisecond"),
            Some(Value::Int64(912))
        );
    }

    #[test]
    fn a_zoned_datetime_reads_its_local_date_and_time() {
        // 2020-05-24T23:03:08Z at -03:30 is 2020-05-24T19:33:08-03:30.
        let zoned = Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
            Timestamp::from_secs(1_590_361_388),
            -12_600,
        ));
        for (name, expected) in [
            ("day", Value::Int64(24)),
            ("hour", Value::Int64(19)),
            ("minute", Value::Int64(33)),
            ("offset", Value::from("-03:30")),
            ("offsetMinutes", Value::Int64(-210)),
            ("epochSeconds", Value::Int64(1_590_361_388)),
        ] {
            assert_eq!(zoned.temporal_component(name), Some(expected), "{name}");
        }
    }

    #[test]
    fn names_are_compared_without_case_and_unknown_ones_are_none() {
        let value = date(2020, 5, 25);
        assert_eq!(value.temporal_component("MONTH"), Some(Value::Int64(5)));
        assert_eq!(value.temporal_component("fortnight"), None);
        assert_eq!(Value::Int64(3).temporal_component("year"), None);
        // A time without an offset has no offset component.
        let time = Value::Time(Time::from_hms(19, 3, 8).unwrap());
        assert_eq!(time.temporal_component("offset"), None);
    }

    #[test]
    fn durations_split_by_unit() {
        // -15 months: Java-like truncation toward zero, as in Cypher.
        let negative = Value::Duration(Duration::new(-15, 0, 0));
        assert_eq!(negative.temporal_component("years"), Some(Value::Int64(-1)));
        assert_eq!(
            negative.temporal_component("monthsOfYear"),
            Some(Value::Int64(-3))
        );
        let seconds = Value::Duration(Duration::new(0, 0, 3_723 * NANOS_PER_SECOND + 19));
        assert_eq!(
            seconds.temporal_component("minutesOfHour"),
            Some(Value::Int64(2))
        );
        assert_eq!(
            seconds.temporal_component("secondsOfMinute"),
            Some(Value::Int64(3))
        );
        assert_eq!(
            seconds.temporal_component("nanosecondsOfSecond"),
            Some(Value::Int64(19))
        );
    }
}
