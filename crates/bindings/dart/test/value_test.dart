import 'dart:convert';
import 'package:grafeo/src/temporal.dart';
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
    // A Dart Duration cannot hold months, and P1D is not PT24H: a duration
    // comes back as a GrafeoDuration of its three parts.
    for (final (text, months, days, nanoseconds) in [
      ('PT1H30M10S', 0, 0, 5410000000000),
      ('PT0S', 0, 0, 0),
      ('PT2H', 0, 0, 7200000000000),
      ('P1Y2M3D', 14, 3, 0),
      ('PT45S', 0, 0, 45000000000),
    ]) {
      test('decodes $text to a GrafeoDuration', () {
        final rows = parseRows(jsonEncode([
          {
            'd': {r'$duration': text}
          }
        ]));
        final val = rows[0]['d'];
        expect(
            val,
            equals(GrafeoDuration(
                months: months, days: days, nanoseconds: nanoseconds)));
        expect(val.toString(), equals(text));
      });
    }
  });

  group('Duration encoding', () {
    test('encodes Duration to ISO format', () {
      final encoded =
          encodeParams({'dur': const Duration(hours: 1, minutes: 30)});
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
        {
          'ts': {r'$timestamp_us': us}
        }
      ]);
      final rows = parseRows(json);
      final val = rows[0]['ts'];
      expect(val, isA<DateTime>());
      expect(val, equals(DateTime.fromMicrosecondsSinceEpoch(us, isUtc: true)));
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
