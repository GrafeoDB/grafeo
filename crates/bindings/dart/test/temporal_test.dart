import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

/// Temporal values cross the Dart binding as their GQL types, both ways: a
/// value read back goes in again as the type it came as.
void main() {
  late GrafeoDB db;
  setUp(() => db = GrafeoDB.memory());
  tearDown(() => db.close());

  /// Whether `$v` arrived as [gqlType], and what came back.
  (bool, dynamic) typedBack(String gqlType, Object value) {
    final row = db
        .executeWithParams(
            'RETURN \$v IS TYPED $gqlType AS typed, \$v AS back', {'v': value})
        .rows
        .single;
    return (row['typed'] as bool, row['back']);
  }

  for (final (gqlType, value) in <(String, Object)>[
    ('LOCAL DATETIME', DateTime.utc(2024, 3, 5, 10, 30, 0, 123, 456)),
    ('TIMESTAMP', DateTime.utc(1969, 7, 20, 20, 17, 40, 654, 321)),
    (
      'ZONED DATETIME',
      GrafeoZonedDateTime(
          DateTime.utc(2024, 3, 5, 9, 30, 0, 250), const Duration(hours: 1))
    ),
    (
      'ZONED DATETIME',
      GrafeoZonedDateTime(DateTime.utc(2024, 3, 5), Duration.zero)
    ),
    (
      'ZONED DATETIME',
      GrafeoZonedDateTime(DateTime.utc(1969, 7, 21, 1, 47, 40, 0, 1),
          const Duration(hours: -5, minutes: -30))
    ),
    ('DATE', GrafeoDate(2024, 2, 29)),
    ('DATE', GrafeoDate(-44, 3, 15)),
    ('TIME', GrafeoTime(14, 30, 15, nanosecond: 500000000)),
    ('TIME', GrafeoTime(12, 30, 0, offset: const Duration(hours: 2))),
    (
      'TIME',
      GrafeoTime(23, 59, 59, nanosecond: 999999999, offset: Duration.zero)
    ),
    (
      'DURATION',
      const GrafeoDuration(months: 14, days: 3, nanoseconds: 14706500000000)
    ),
    ('DURATION', const GrafeoDuration(months: -1, days: -2, nanoseconds: -3)),
    ('DURATION', const GrafeoDuration(days: 1)),
    ('DURATION', const GrafeoDuration()),
  ]) {
    test('$value goes in and comes back as a $gqlType', () {
      final (typed, back) = typedBack(gqlType, value);
      expect(typed, isTrue, reason: 'arrived as $gqlType');
      expect(back, equals(value));
      expect(back.runtimeType, equals(value.runtimeType));
      if (value is DateTime) expect((back as DateTime).isUtc, isTrue);
      expect(typedBack(gqlType, back as Object).$1, isTrue,
          reason: 'read back, it goes in as $gqlType again');
    });
  }

  test('a local DateTime is the same instant as a TIMESTAMP, back in UTC', () {
    final local = DateTime(2024, 3, 5, 10, 30, 0, 0, 7);
    final (typed, back) = typedBack('TIMESTAMP', local);
    expect(typed, isTrue);
    expect(back, equals(local.toUtc()));
    expect((back as DateTime).isUtc, isTrue);
  });

  test('a Dart Duration is a DURATION of its microseconds, signed', () {
    for (final duration in [
      const Duration(hours: 1, minutes: 30, microseconds: 5),
      const Duration(seconds: -90, milliseconds: -250),
      const Duration(days: 2),
    ]) {
      final (typed, back) = typedBack('DURATION', duration);
      expect(typed, isTrue);
      expect(back, equals(GrafeoDuration.fromDuration(duration)));
      expect((back as GrafeoDuration).toDuration(), equals(duration));
    }
    expect(const GrafeoDuration(months: 1).toDuration(), isNull);
    expect(const GrafeoDuration(nanoseconds: 1).toDuration(), isNull);
  });

  test(
      'values a query makes come back typed, in lists and maps, and go in again',
      () {
    final row = db
        .execute("RETURN DATE '2024-03-05' AS d, "
            "TIME '12:30:00+02:00' AS zt, TIME '08:15:30.25' AS lt, "
            "DURATION 'P1Y2M3DT4H5M6.5S' AS dur, "
            "ZONED DATETIME '2024-03-05T10:30:00.5+05:30' AS zdt, "
            "local_datetime('2024-03-05T10:30:00.000001') AS ldt, "
            "[DATE '2024-03-05', DATE '2024-03-06'] AS dates, "
            "{at: TIME '12:30:00Z', took: DURATION 'PT1M'} AS m")
        .rows
        .single;
    expect(row['d'], equals(GrafeoDate(2024, 3, 5)));
    expect(row['zt'],
        equals(GrafeoTime(12, 30, 0, offset: const Duration(hours: 2))));
    expect(row['zt'].toString(), equals('12:30:00+02:00'));
    expect(row['lt'], equals(GrafeoTime(8, 15, 30, nanosecond: 250000000)));
    expect(
        row['dur'],
        equals(const GrafeoDuration(
            months: 14, days: 3, nanoseconds: 14706500000000)));
    final zdt = row['zdt'] as GrafeoZonedDateTime;
    expect(zdt.instant, equals(DateTime.utc(2024, 3, 5, 5, 0, 0, 500)));
    expect(zdt.offset, equals(const Duration(hours: 5, minutes: 30)));
    expect(zdt.text, equals('2024-03-05T10:30:00.5+05:30'));
    expect(row['ldt'], equals(DateTime.utc(2024, 3, 5, 10, 30, 0, 0, 1)));
    expect(
        row['dates'], equals([GrafeoDate(2024, 3, 5), GrafeoDate(2024, 3, 6)]));
    expect(
        row['m'],
        equals({
          'at': GrafeoTime(12, 30, 0, offset: Duration.zero),
          'took': const GrafeoDuration(nanoseconds: 60000000000),
        }));

    final again = db
        .executeWithParams(
            r'RETURN $d IS TYPED DATE AND $zt IS TYPED TIME AND $lt IS TYPED TIME '
            r'AND $dur IS TYPED DURATION AND $zdt IS TYPED ZONED DATETIME '
            r'AND $ldt IS TYPED LOCAL DATETIME AND $dates IS TYPED LIST<DATE> '
            r'AND $m.at IS TYPED TIME AND $m.took IS TYPED DURATION AS typed, '
            r"$zt = TIME '12:30:00+02:00' AND $lt = TIME '08:15:30.25' "
            r"AND $zdt = ZONED DATETIME '2024-03-05T10:30:00.5+05:30' AS equal",
            row)
        .rows
        .single;
    expect(again, equals({'typed': true, 'equal': true}));
  });

  test('stored temporal properties come back typed everywhere', () {
    final history = GrafeoDB.memory(historyRetention: HistoryRetention.forever);
    addTearDown(history.close);
    final born = GrafeoDate(1994, 5, 12);
    final wakes = GrafeoTime(7, 30, 0, offset: const Duration(hours: 1));
    final seen = DateTime.utc(2024, 3, 5, 10, 30, 0, 0, 1);
    final id = history
        .executeWithParams(
            r'INSERT (p:Person {name: $name, born: $born, wakes: $wakes, seen: $seen}) '
            'RETURN id(p) AS id',
            {
              'name': 'Shosanna',
              'born': born,
              'wakes': wakes,
              'seen': [seen],
            })
        .rows
        .single['id'] as int;
    void check(String where, Map<String, dynamic> properties) {
      expect(properties['born'], equals(born), reason: where);
      expect(properties['wakes'], equals(wakes), reason: where);
      expect(properties['seen'], equals([seen]), reason: where);
    }

    check('getNode', history.getNode(id)!.properties);
    final stream = history.executeStream(
        'MATCH (p:Person) RETURN p.born AS born, p.wakes AS wakes, p.seen AS seen');
    try {
      check('stream', stream.toList().single);
    } finally {
      stream.close();
    }
    final view = history.asOf(TimePoint.epoch(history.currentEpoch));
    try {
      check('asOf', {
        for (final key in ['born', 'wakes', 'seen'])
          key: view.getNodeProperty(id, key),
      });
    } finally {
      view.close();
    }
  });

  group('texts agree with the engine', () {
    for (final text in ['2024-01-15', '-0044-03-15', '0001-01-01']) {
      test('date $text',
          () => expect(GrafeoDate.parse(text).text, equals(text)));
    }
    for (final text in [
      '2024-02-30',
      '2024-13-01',
      '2024-1-1x',
      '',
      '+2024-01-01'
    ]) {
      test('not a date: "$text"',
          () => expect(GrafeoDate.tryParse(text), isNull));
    }
    for (final text in [
      '14:30:00',
      '14:30:00.5',
      '00:00:00.000000001',
      '12:30:00Z',
      '12:30:00+02:00',
      '12:30:00-05:30',
    ]) {
      test('time $text',
          () => expect(GrafeoTime.parse(text).text, equals(text)));
    }
    test('time 14:30',
        () => expect(GrafeoTime.parse('14:30').text, equals('14:30:00')));
    for (final text in [
      '24:00:00',
      '12:60:00',
      '12:30:00.',
      '12',
      '12:30:00+02:60',
      'noon'
    ]) {
      test('not a time: "$text"', () {
        expect(GrafeoTime.tryParse(text), isNull);
        expect(() => GrafeoTime.parse(text), throwsFormatException);
      });
    }
    for (final text in [
      'P1Y2M3DT4H5M6.5S',
      'PT0S',
      '-P1M2DT0.000000003S',
      'P1D',
      'PT24H',
      'P1M-2D',
    ]) {
      test('duration $text',
          () => expect(GrafeoDuration.parse(text).text, equals(text)));
    }
    test(
        'duration P2W',
        () => expect(GrafeoDuration.parse('P2W'),
            equals(const GrafeoDuration(days: 14))));
    for (final text in ['P', '1D', 'P1X', 'PT1D', 'P+1D']) {
      test('not a duration: "$text"',
          () => expect(GrafeoDuration.tryParse(text), isNull));
    }
    for (final text in [
      '2024-03-05T10:30:00.5+05:30',
      '2024-03-05T10:30:00Z',
      '1969-07-20T20:17:40.000001-05:30'
    ]) {
      test('zoned datetime $text',
          () => expect(GrafeoZonedDateTime.parse(text).text, equals(text)));
    }
    test('a zoned datetime needs an offset', () {
      expect(GrafeoZonedDateTime.tryParse('2024-03-05T10:30:00'), isNull);
    });
    test('parts are checked', () {
      expect(() => GrafeoTime(24, 0, 0), throwsArgumentError);
      expect(() => GrafeoTime(12, 0, 0, offset: const Duration(seconds: 90)),
          throwsArgumentError);
      expect(() => GrafeoDate(2023, 2, 29), throwsArgumentError);
      expect(
          () => GrafeoZonedDateTime(
              DateTime.utc(2024), const Duration(hours: 24)),
          throwsArgumentError);
    });
  });
}
