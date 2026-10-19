import 'dart:convert';
import 'package:grafeo/src/types.dart';
import 'package:grafeo/src/value.dart';
import 'package:test/test.dart';

void main() {
  group('Decimal', () {
    test('encodes as the tagged text with its scale', () {
      expect(
          encodeValue(GrafeoDecimal('1.50')), equals(r'{"$decimal":"1.50"}'));
    });

    test('decodes the tag alone as a GrafeoDecimal', () {
      final rows = parseRows(jsonEncode([
        {
          'total': {r'$decimal': '0.30'}
        }
      ]));
      final total = rows[0]['total'];
      expect(total, isA<GrafeoDecimal>());
      expect(total.toString(), equals('0.30'));
      expect((total as GrafeoDecimal).toDouble(), closeTo(0.3, 1e-12));
    });

    test('keeps a map with other keys a map', () {
      final rows = parseRows(jsonEncode([
        {
          'm': {r'$decimal': '1', 'other': 2}
        }
      ]));
      expect(rows[0]['m'], equals({r'$decimal': '1', 'other': 2}));
    });

    test('compares by value whatever the scale', () {
      expect(GrafeoDecimal('1.50'), equals(GrafeoDecimal('1.5')));
      expect(GrafeoDecimal('1.50').hashCode,
          equals(GrafeoDecimal('1.5').hashCode));
      expect(GrafeoDecimal('-0.00'), equals(GrafeoDecimal('0')));
      expect(GrafeoDecimal('1.5'), isNot(equals(GrafeoDecimal('1.05'))));
    });

    test('refuses text that is not a decimal', () {
      expect(() => GrafeoDecimal('1.2.3'), throwsFormatException);
      expect(() => GrafeoDecimal('1e5'), throwsFormatException);
    });
  });

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
