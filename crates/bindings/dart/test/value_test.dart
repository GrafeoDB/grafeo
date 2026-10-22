import 'dart:convert';
import 'package:grafeo/src/value.dart';
import 'package:test/test.dart';

void main() {
  group('Duration decoding', () {
    // Durations are returned as ISO 8601 strings because Dart's Duration type
    // cannot represent calendar components (years, months, days).
    test('decodes PT1H30M10S to ISO string', () {
      final json = jsonEncode([
        {r'$duration': 'PT1H30M10S'}
      ]);
      final rows = parseRows(json);
      expect(rows, hasLength(1));
      final val = rows[0].values.first;
      expect(val, isA<String>());
      expect(val, equals('PT1H30M10S'));
    });

    test('decodes PT0S to string', () {
      final json = jsonEncode([
        {r'$duration': 'PT0S'}
      ]);
      final rows = parseRows(json);
      final val = rows[0].values.first;
      expect(val, isA<String>());
      expect(val, equals('PT0S'));
    });

    test('decodes hours-only duration PT2H', () {
      final json = jsonEncode([
        {r'$duration': 'PT2H'}
      ]);
      final rows = parseRows(json);
      final val = rows[0].values.first;
      expect(val, isA<String>());
      expect(val, equals('PT2H'));
    });

    test('decodes calendar duration P1Y2M3D', () {
      final json = jsonEncode([
        {r'$duration': 'P1Y2M3D'}
      ]);
      final rows = parseRows(json);
      final val = rows[0].values.first;
      expect(val, isA<String>());
      expect(val, equals('P1Y2M3D'));
    });

    test('decodes seconds-only PT45S', () {
      final json = jsonEncode([
        {r'$duration': 'PT45S'}
      ]);
      final rows = parseRows(json);
      final val = rows[0].values.first;
      expect(val, isA<String>());
      expect(val, equals('PT45S'));
    });
  });

  group('Duration encoding', () {
    test('encodes Duration to ISO format', () {
      final encoded = encodeParams({'dur': const Duration(hours: 1, minutes: 30)});
      final decoded = jsonDecode(encoded) as Map<String, dynamic>;
      expect(decoded['dur'], isA<Map>());
      expect((decoded['dur'] as Map)[r'$duration'], equals('PT1H30M'));
    });

    test('encodes zero Duration', () {
      final encoded = encodeParams({'dur': Duration.zero});
      final decoded = jsonDecode(encoded) as Map<String, dynamic>;
      expect((decoded['dur'] as Map)[r'$duration'], equals('PT0S'));
    });
  });

  group('Timestamp decoding', () {
    test('decodes timestamp_us marker to DateTime', () {
      final us = DateTime(2024, 1, 15, 12, 0, 0).toUtc().microsecondsSinceEpoch;
      final json = jsonEncode([
        {r'$timestamp_us': us}
      ]);
      final rows = parseRows(json);
      final val = rows[0].values.first;
      expect(val, isA<DateTime>());
    });
  });

  group('Nested maps', () {
    test('decodes nested regular maps', () {
      final json = jsonEncode([
        {
          'outer': {'inner': 'value', 'num': 42}
        }
      ]);
      final rows = parseRows(json);
      final val = rows[0]['outer'] as Map<String, dynamic>;
      expect(val['inner'], equals('value'));
      expect(val['num'], equals(42));
    });
  });
}
