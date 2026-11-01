/// GQL's temporal values that no Dart type holds without loss: a `DATE`, a
/// `TIME` (with or without a UTC offset), a `DURATION` and a `ZONED
/// DATETIME`. A `TIMESTAMP` is a UTC [DateTime]. Each type comes back from a
/// query and goes in as a parameter as the same GQL type again.
library;

const _nanosPerSecond = 1000000000;
const _nanosPerMinute = 60 * _nanosPerSecond;
const _nanosPerHour = 60 * _nanosPerMinute;
const _secondsPerDay = 86400;

/// The smallest and the largest 64-bit integers.
const _minInt = -1 << 63;
const _maxInt = ~_minInt;

/// Two digits of [n], or more.
String _two(int n) => n.toString().padLeft(2, '0');

/// A whole number of ASCII digits, as a number; null for any other text.
int? _digits(String text) =>
    RegExp(r'^\d+$').hasMatch(text) ? int.parse(text) : null;

/// An offset as GQL writes it: `Z`, `+02:00`, `-05:30`.
String _offsetText(int seconds) {
  if (seconds == 0) return 'Z';
  final sign = seconds < 0 ? '-' : '+';
  final abs = seconds.abs();
  return '$sign${_two(abs ~/ 3600)}:${_two(abs % 3600 ~/ 60)}';
}

/// The offset at the end of [text] (`Z`, `+HH:MM`, `-HH:MM`) in seconds and
/// the text before it, or the whole text and null when it has none; null
/// when the offset does not read.
(String, int?)? _splitOffset(String text) {
  if (text.endsWith('Z')) return (text.substring(0, text.length - 1), 0);
  final n = text.length;
  if (n >= 6 &&
      (text[n - 6] == '+' || text[n - 6] == '-') &&
      text[n - 3] == ':') {
    final hours = _digits(text.substring(n - 5, n - 3));
    final minutes = _digits(text.substring(n - 2));
    if (hours == null || minutes == null || minutes > 59) return null;
    final seconds = hours * 3600 + minutes * 60;
    return (text.substring(0, n - 6), text[n - 6] == '-' ? -seconds : seconds);
  }
  return (text, null);
}

/// A GQL `DATE`: a day of the calendar, without a time or zone.
final class GrafeoDate {
  /// The date of [year], [month] (1 to 12) and [day].
  ///
  /// Throws an [ArgumentError] for a day the calendar does not have.
  GrafeoDate(this.year, this.month, this.day) {
    final utc = DateTime.utc(year, month, day);
    if (month < 1 ||
        month > 12 ||
        utc.year != year ||
        utc.month != month ||
        utc.day != day) {
      throw ArgumentError('$year-$month-$day is not a date');
    }
  }

  /// The date of [dateTime] in its own zone (local or UTC).
  factory GrafeoDate.fromDateTime(DateTime dateTime) =>
      GrafeoDate(dateTime.year, dateTime.month, dateTime.day);

  /// A date from its text as GQL writes it: `YYYY-MM-DD`, with a leading
  /// minus for a year before year 0.
  ///
  /// Throws a [FormatException] when [text] is not a date.
  factory GrafeoDate.parse(String text) =>
      tryParse(text) ?? (throw FormatException('not a date', text));

  /// The date [text] names, or null when it is not one.
  static GrafeoDate? tryParse(String text) {
    final negative = text.startsWith('-');
    final parts = (negative ? text.substring(1) : text).split('-');
    if (parts.length != 3) return null;
    final numbers = parts.map(_digits).toList();
    if (numbers.contains(null)) return null;
    try {
      return GrafeoDate(
          negative ? -numbers[0]! : numbers[0]!, numbers[1]!, numbers[2]!);
    } on ArgumentError {
      return null;
    }
  }

  /// The year.
  final int year;

  /// The month, 1 to 12.
  final int month;

  /// The day of the month.
  final int day;

  /// Midnight at the start of the date, in UTC.
  DateTime toDateTime() => DateTime.utc(year, month, day);

  /// The text as GQL writes it: `2024-01-15`.
  String get text {
    final y = year.abs().toString().padLeft(4, '0');
    return '${year < 0 ? '-' : ''}$y-${_two(month)}-${_two(day)}';
  }

  @override
  String toString() => text;

  @override
  bool operator ==(Object other) =>
      other is GrafeoDate &&
      other.year == year &&
      other.month == month &&
      other.day == day;

  @override
  int get hashCode => Object.hash(year, month, day);
}

/// A GQL `TIME`: a time of day to the nanosecond, with a UTC offset
/// (`12:30:00+02:00`) or without one (a local time, `12:30:00`). Two times
/// are equal when they have the same clock reading and offset.
final class GrafeoTime {
  /// A time of day from its parts, with an [offset] of whole minutes less
  /// than a day, or none for a local time.
  ///
  /// Throws an [ArgumentError] when a part is out of its range.
  GrafeoTime(this.hour, this.minute, this.second,
      {this.nanosecond = 0, this.offset}) {
    if (hour < 0 ||
        hour > 23 ||
        minute < 0 ||
        minute > 59 ||
        second < 0 ||
        second > 59 ||
        nanosecond < 0 ||
        nanosecond >= _nanosPerSecond) {
      throw ArgumentError(
          '$hour:$minute:$second.$nanosecond is not a time of day');
    }
    final o = offset;
    if (o != null &&
        (o.inMicroseconds % Duration.microsecondsPerMinute != 0 ||
            o.inSeconds.abs() >= _secondsPerDay)) {
      throw ArgumentError('an offset of $o is not whole minutes within a day');
    }
  }

  /// A time from its text as GQL writes it: `HH:MM[:SS[.fraction]]`, then
  /// `Z`, `+HH:MM` or `-HH:MM` for an offset.
  ///
  /// Throws a [FormatException] when [text] is not a time.
  factory GrafeoTime.parse(String text) =>
      tryParse(text) ?? (throw FormatException('not a time', text));

  /// The time [text] names, or null when it is not one.
  static GrafeoTime? tryParse(String text) {
    final split = _splitOffset(text);
    if (split == null) return null;
    final (body, offsetSeconds) = split;
    final point = body.indexOf('.');
    final fields = (point < 0 ? body : body.substring(0, point)).split(':');
    if (fields.length < 2 || fields.length > 3) return null;
    final hms = fields.map(_digits).toList();
    if (hms.contains(null)) return null;
    var nanos = 0;
    if (point >= 0) {
      final fraction = body.substring(point + 1);
      if (_digits(fraction) == null) return null;
      nanos = int.parse('${fraction}000000000'.substring(0, 9));
    }
    try {
      return GrafeoTime(hms[0]!, hms[1]!, hms.length == 3 ? hms[2]! : 0,
          nanosecond: nanos,
          offset:
              offsetSeconds == null ? null : Duration(seconds: offsetSeconds));
    } on ArgumentError {
      return null;
    }
  }

  /// The hour, 0 to 23.
  final int hour;

  /// The minute, 0 to 59.
  final int minute;

  /// The second, 0 to 59.
  final int second;

  /// The fraction of the second in nanoseconds.
  final int nanosecond;

  /// The UTC offset, or null for a local time.
  final Duration? offset;

  /// The text as GQL writes it: `14:30:00`, `14:30:00.5+02:00`, `14:30:00Z`.
  String get text {
    final out = StringBuffer('${_two(hour)}:${_two(minute)}:${_two(second)}');
    if (nanosecond != 0) {
      out.write(
          '.${nanosecond.toString().padLeft(9, '0').replaceFirst(RegExp(r'0+$'), '')}');
    }
    final o = offset;
    if (o != null) out.write(_offsetText(o.inSeconds));
    return out.toString();
  }

  @override
  String toString() => text;

  @override
  bool operator ==(Object other) =>
      other is GrafeoTime &&
      other.hour == hour &&
      other.minute == minute &&
      other.second == second &&
      other.nanosecond == nanosecond &&
      other.offset == offset;

  @override
  int get hashCode => Object.hash(hour, minute, second, nanosecond, offset);
}

/// A GQL `DURATION`: months, days and nanoseconds, kept apart because a
/// month and a day have no fixed length (`P1M` is not `P30D`, and `P1D` is
/// not `PT24H`). A Dart [Duration] parameter is a duration of its
/// microseconds alone. Two durations are equal when their three parts are.
final class GrafeoDuration {
  /// A duration of [months], [days] and [nanoseconds].
  const GrafeoDuration({this.months = 0, this.days = 0, this.nanoseconds = 0});

  /// The duration of [duration], as nanoseconds.
  factory GrafeoDuration.fromDuration(Duration duration) =>
      GrafeoDuration(nanoseconds: duration.inMicroseconds * 1000);

  /// A duration from its ISO 8601 text as GQL writes it:
  /// `P[nY][nM][nW][nD][T[nH][nM][n[.fraction]S]]`, with a leading minus for
  /// a negative one.
  ///
  /// Throws a [FormatException] when [text] is not a duration.
  factory GrafeoDuration.parse(String text) =>
      tryParse(text) ?? (throw FormatException('not a duration', text));

  /// The duration [text] names, or null when it is not one.
  static GrafeoDuration? tryParse(String text) {
    final negative = text.startsWith('-');
    var body = negative ? text.substring(1) : text;
    if (!body.startsWith('P')) return null;
    body = body.substring(1);
    final parts = [0, 0, 0];
    var inTime = false, start = 0, content = false;
    for (var i = 0; i < body.length; i++) {
      final c = body[i];
      if (c == 'T' && !inTime && start == i) {
        inTime = true;
        start = i + 1;
        continue;
      }
      if (!(inTime ? 'HMS' : 'YMWD').contains(c)) continue;
      final number = body.substring(start, i);
      // The part a designator adds to (months, days, nanoseconds) and the
      // number of it in one unit
      final (part, unit) = switch (c) {
        'Y' => (0, 12),
        'M' when !inTime => (0, 1),
        'W' => (1, 7),
        'D' => (1, 1),
        'H' => (2, _nanosPerHour),
        'M' => (2, _nanosPerMinute),
        _ => (2, 1),
      };
      final n = c == 'S' ? _secondsNanos(number) : _integer(number);
      if (n == null) return null;
      // A part past 64 bits: refused, not wrapped around
      final sum = _addProduct(parts[part], n, unit);
      if (sum == null) return null;
      parts[part] = sum;
      start = i + 1;
      content = true;
    }
    if (!content || start != body.length) return null;
    if (negative && parts.contains(_minInt)) return null;
    final [months, days, nanos] =
        negative ? [for (final p in parts) -p] : parts;
    return GrafeoDuration(months: months, days: days, nanoseconds: nanos);
  }

  /// The months (a year is 12).
  final int months;

  /// The days.
  final int days;

  /// The part shorter than a day, in nanoseconds.
  final int nanoseconds;

  /// The duration as a Dart [Duration] (a day as 24 hours), or null when it
  /// has months or a part finer than a microsecond.
  Duration? toDuration() => months != 0 || nanoseconds % 1000 != 0
      ? null
      : Duration(days: days, microseconds: nanoseconds ~/ 1000);

  /// The text as GQL writes it: `P1Y2M3DT4H5M6.5S`, `PT0S`.
  String get text {
    var m = months, d = days, n = nanoseconds;
    final negative = (m < 0 || d < 0 || n < 0) && m <= 0 && d <= 0 && n <= 0;
    if (negative) {
      m = -m;
      d = -d;
      n = -n;
    }
    final out = StringBuffer(negative ? '-P' : 'P');
    // `~/` and `remainder` truncate toward zero, as the engine's arithmetic
    if (m ~/ 12 != 0) out.write('${m ~/ 12}Y');
    if (m.remainder(12) != 0) out.write('${m.remainder(12)}M');
    if (d != 0) out.write('${d}D');
    if (n != 0) {
      out.write('T');
      final hours = n ~/ _nanosPerHour;
      final minutes = n.remainder(_nanosPerHour) ~/ _nanosPerMinute;
      final seconds = n.remainder(_nanosPerMinute) ~/ _nanosPerSecond;
      final fraction = n.remainder(_nanosPerSecond);
      if (hours != 0) out.write('${hours}H');
      if (minutes != 0) out.write('${minutes}M');
      if (fraction != 0 || seconds != 0) {
        // The sign once, before the whole seconds (negative only in a
        // duration of mixed signs), the fraction as a magnitude: `T-1.5S`
        if (n.remainder(_nanosPerMinute) < 0) out.write('-');
        out.write(seconds.abs());
        if (fraction != 0) {
          final digits = fraction.abs().toString().padLeft(9, '0');
          out.write('.${digits.replaceFirst(RegExp(r'0+$'), '')}');
        }
        out.write('S');
      }
    }
    if (!negative && m == 0 && d == 0 && n == 0) out.write('T0S');
    return out.toString();
  }

  @override
  String toString() => text;

  @override
  bool operator ==(Object other) =>
      other is GrafeoDuration &&
      other.months == months &&
      other.days == days &&
      other.nanoseconds == nanoseconds;

  @override
  int get hashCode => Object.hash(months, days, nanoseconds);

  /// An integer with an optional minus, as the engine reads one.
  static int? _integer(String text) =>
      RegExp(r'^-?\d+$').hasMatch(text) ? int.tryParse(text) : null;

  /// The seconds of a duration, `n` or `n.fraction`, as the engine reads
  /// them: the fraction's first nine digits, with the sign of the whole
  /// seconds (`-0.5` is negative). It also reads what 0.6.0 wrote for a
  /// negative fraction in a duration of mixed signs, the sign inside the
  /// fraction (`-1.-5` for -1.5 s, nine characters wide with that sign
  /// before its trailing zeros were cut): digits that start with a zero are
  /// that eight digit field (`0.-05` is -0.005 s), others the plain fraction.
  static int? _secondsNanos(String text) {
    final point = text.indexOf('.');
    final wholeText = point < 0 ? text : text.substring(0, point);
    final whole = _integer(wholeText);
    if (whole == null) return null;
    final nanos = _addProduct(0, whole, _nanosPerSecond);
    if (nanos == null || point < 0) return nanos;
    var fraction = text.substring(point + 1);
    var negative = wholeText.startsWith('-'), width = 9;
    if (fraction.startsWith('-')) {
      // 0.6.0 wrote it after seconds of 0 or less only
      if (whole > 0) return null;
      negative = true;
      fraction = fraction.substring(1);
      if (fraction.startsWith('0') && fraction.length <= 8) width = 8;
    }
    if (fraction.length > width) fraction = fraction.substring(0, width);
    final digits = _digits(fraction);
    if (digits == null) return null;
    var scale = 1;
    for (var i = fraction.length; i < width; i++) {
      scale *= 10;
    }
    return _checkedAdd(nanos, (negative ? -digits : digits) * scale);
  }

  /// [a] + [b], or null when the sum does not fit 64 bits.
  static int? _checkedAdd(int a, int b) {
    final sum = a + b;
    return (b >= 0) == (sum >= a) ? sum : null;
  }

  /// [total] + [n] * [unit] for a positive [unit], or null when the product
  /// or the sum does not fit 64 bits.
  static int? _addProduct(int total, int n, int unit) {
    // `~/` truncates toward zero: n * unit fits exactly when n lies between
    // these bounds
    if (n > _maxInt ~/ unit || n < _minInt ~/ unit) return null;
    return _checkedAdd(total, n * unit);
  }
}

/// A GQL `ZONED DATETIME`: an instant, to the microsecond, with the UTC
/// offset it was written at. A Dart [DateTime] is UTC or local and keeps no
/// other offset, so a zoned datetime comes back as this type. Two zoned
/// datetimes are equal when they have the same instant and offset.
final class GrafeoZonedDateTime {
  /// The zoned datetime of [instant] at [offset], whole minutes less than a
  /// day.
  ///
  /// Throws an [ArgumentError] for another offset.
  GrafeoZonedDateTime(DateTime instant, this.offset)
      : instant = instant.toUtc() {
    if (offset.inMicroseconds % Duration.microsecondsPerMinute != 0 ||
        offset.inSeconds.abs() >= _secondsPerDay) {
      throw ArgumentError(
          'an offset of $offset is not whole minutes within a day');
    }
  }

  /// A zoned datetime from its text as GQL writes it:
  /// `YYYY-MM-DDTHH:MM:SS[.fraction]` then `Z`, `+HH:MM` or `-HH:MM`.
  ///
  /// Throws a [FormatException] when [text] is not a zoned datetime.
  factory GrafeoZonedDateTime.parse(String text) =>
      tryParse(text) ?? (throw FormatException('not a zoned datetime', text));

  /// The zoned datetime [text] names, or null when it is not one.
  static GrafeoZonedDateTime? tryParse(String text) {
    final t = text.indexOf('T');
    if (t < 0) return null;
    final date = GrafeoDate.tryParse(text.substring(0, t));
    final time = GrafeoTime.tryParse(text.substring(t + 1));
    final offset = time?.offset;
    if (date == null || time == null || offset == null) return null;
    final wallClock = DateTime.utc(date.year, date.month, date.day, time.hour,
        time.minute, time.second, 0, time.nanosecond ~/ 1000);
    return GrafeoZonedDateTime(wallClock.subtract(offset), offset);
  }

  /// The instant, as a UTC [DateTime].
  final DateTime instant;

  /// The UTC offset.
  final Duration offset;

  /// The text as GQL writes it: `2024-03-05T10:30:00.5+01:00`.
  String get text {
    final wall = instant.add(offset);
    final date = GrafeoDate(wall.year, wall.month, wall.day);
    final micros = wall.millisecond * 1000 + wall.microsecond;
    final fraction = micros == 0
        ? ''
        : '.${micros.toString().padLeft(6, '0').replaceFirst(RegExp(r'0+$'), '')}';
    return '${date}T${_two(wall.hour)}:${_two(wall.minute)}:${_two(wall.second)}'
        '$fraction${_offsetText(offset.inSeconds)}';
  }

  @override
  String toString() => text;

  @override
  bool operator ==(Object other) =>
      other is GrafeoZonedDateTime &&
      other.instant == instant &&
      other.offset == offset;

  @override
  int get hashCode => Object.hash(instant, offset);
}
