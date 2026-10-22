import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

/// A BigInt parameter or property past 64 bits is an exact decimal, up to 38
/// digits: one that fits 64 bits stays INT64, one past it but within 38
/// digits arrives as a DECIMAL with scale 0 (a BigInt used to arrive as a
/// string), and one past 38 digits is an error naming the parameter and the
/// value.
void main() {
  late GrafeoDB db;
  setUp(() => db = GrafeoDB.memory());
  tearDown(() => db.close());

  Map<String, dynamic> typed(Object value) => db
      .executeWithParams(
        r'RETURN $x AS x, $x IS TYPED INT64 AS int, $x IS TYPED DECIMAL AS decimal',
        {'x': value},
      )
      .rows
      .first;

  final wide = [
    BigInt.two.pow(63),
    BigInt.two.pow(64),
    -BigInt.two.pow(63) - BigInt.one,
    BigInt.from(10).pow(38) - BigInt.one,
  ];

  for (final value in wide) {
    test('a BigInt past 64 bits is an exact DECIMAL: $value', () {
      final row = typed(value);
      expect((row['x'] as GrafeoDecimal).text, equals('$value'));
      expect([row['int'], row['decimal']], equals([false, true]));
    });
  }

  test('a BigInt that fits 64 bits is INT64', () {
    for (final value in [BigInt.from(5), BigInt.two.pow(63) - BigInt.one]) {
      expect(typed(value)['int'], isTrue, reason: '$value');
    }
  });

  test('a BigInt past 38 digits is an error naming the parameter', () {
    final tooWide = BigInt.from(10).pow(38);
    expect(
      () => typed(tooWide),
      throwsA(isA<GrafeoException>().having(
        (error) => error.toString(),
        'message',
        allOf(contains("parameter 'x'"), contains('$tooWide'),
            contains('38 digits')),
      )),
    );
  });

  test('inside a list a BigInt follows the rule, and strings stay strings', () {
    final rows = db.executeWithParams(
      r'UNWIND $xs AS v RETURN v IS TYPED DECIMAL AS decimal',
      {
        'xs': [3, BigInt.two.pow(64)],
      },
    ).rows;
    expect(rows.map((row) => row['decimal']), equals([false, true]));
    final row = db
        .executeWithParams(r'RETURN $s AS s', {
          's': 'quote " and \\ backslash',
        })
        .rows
        .first;
    expect(row['s'], equals('quote " and \\ backslash'));
  });

  test('arithmetic over INT64 and DECIMAL properties stays exact', () {
    db.createNode(['N'], {'v': 5});
    db.createNode(['N'], {'v': BigInt.two.pow(64)});
    final total =
        db.execute('MATCH (n:N) RETURN sum(n.v) AS total').rows.first['total'];
    expect((total as GrafeoDecimal).text, equals('18446744073709551621'));
  });
}
