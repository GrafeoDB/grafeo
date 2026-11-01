import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

/// NaN and the infinities, which JSON has no number for (jsonEncode refuses
/// them), travel as {"$float": "NaN"} and come back as double NaN and the
/// infinities; the string "NaN" stays a string; a whole double stays FLOAT64.
void main() {
  late GrafeoDB db;
  setUp(() => db = GrafeoDB.memory());
  tearDown(() => db.close());

  Map<String, dynamic> typed(Object? value) => db
      .executeWithParams(
        r'RETURN $x AS x, $x IS TYPED FLOAT64 AS float, $x IS TYPED STRING AS str',
        {'x': value},
      )
      .rows
      .first;

  for (final value in [double.nan, double.infinity, double.negativeInfinity]) {
    test('$value round-trips as a FLOAT64', () {
      final row = typed(value);
      expect(row['float'], isTrue);
      final back = row['x'] as double;
      if (value.isNaN) {
        expect(back.isNaN, isTrue);
      } else {
        expect(back, equals(value));
      }
    });
  }

  test('special floats inside a list', () {
    final row = db
        .executeWithParams(
            r'RETURN $xs[0] IS TYPED FLOAT64 AS first, $xs[1] AS second', {
          'xs': [double.nan, double.infinity],
        })
        .rows
        .first;
    expect(row['first'], isTrue);
    expect(row['second'], equals(double.infinity));
  });

  for (final text in ['NaN', 'Infinity', '-Infinity']) {
    test('the string $text stays a string', () {
      final row = typed(text);
      expect(row['x'], equals(text));
      expect([row['float'], row['str']], equals([false, true]));
    });
  }

  test('a whole double is FLOAT64', () {
    for (final value in [3.0, 0.0, -20.0, 1e6]) {
      expect(typed(value)['float'], isTrue, reason: '$value');
    }
  });
}
