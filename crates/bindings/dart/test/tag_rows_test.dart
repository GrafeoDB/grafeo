import 'dart:typed_data';

import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

/// A row is a map of columns, never a tagged value: a column named like a
/// tag (`$bytes`, `$decimal`, `$date`) is a column, and only each value is
/// decoded. Streamed rows decode their values as `execute` does.
void main() {
  late GrafeoDB db;
  setUp(() => db = GrafeoDB.memory());
  tearDown(() => db.close());

  /// The rows of [query] through `execute` and through `executeStream`.
  Map<String, List<Map<String, dynamic>>> rowsBothWays(String query) {
    final stream = db.executeStream(query);
    try {
      return {'execute': db.execute(query).rows, 'stream': stream.toList()};
    } finally {
      stream.close();
    }
  }

  for (final (column, literal, value) in [
    (r'$bytes', "'AQID'", 'AQID'),
    (r'$decimal', "'1.50'", '1.50'),
    (r'$date', "'2024-01-02'", '2024-01-02'),
    (r'$float', "'NaN'", 'NaN'),
    (r'$timestamp_us', '5', 5),
    (r'$duration', "'P1D'", 'P1D'),
  ]) {
    test('a one-column row named $column stays a row', () {
      for (final MapEntry(key: how, value: rows)
          in rowsBothWays('RETURN $literal AS `$column`').entries) {
        expect(rows, hasLength(1), reason: how);
        expect(rows.first, equals({column: value}), reason: how);
      }
    });
  }

  test('a row with a column named \$date keeps its other columns', () {
    for (final MapEntry(key: how, value: rows) in rowsBothWays(
      r"RETURN '2024-01-02' AS `$date`, 'Gus' AS name",
    ).entries) {
      expect(
        rows.first,
        equals({r'$date': '2024-01-02', 'name': 'Gus'}),
        reason: how,
      );
    }
  });

  // A tag is the map's only key: a map value with a tag key and another key
  // is a map, for the temporal tags as for $bytes and $decimal.
  for (final key in [
    r'$date',
    r'$time',
    r'$duration',
    r'$zoned_datetime',
    r'$timestamp_us',
    r'$bytes',
    r'$decimal',
  ]) {
    test('a map with $key and a second key stays a map', () {
      final map = {
        key: key == r'$timestamp_us' ? 5 : '2024-01-02',
        'name': 'Gus'
      };
      final row =
          db.executeWithParams(r'RETURN $m AS m', {'m': map}).rows.single;
      expect(row['m'], equals(map));
    });
  }

  test('streamed rows decode bytes, decimals and NaN', () {
    db.executeWithParams(
      r"INSERT (:Blob {name: 'Mia', data: $data, amount: $amount, ratio: $ratio})",
      {
        'data': Uint8List.fromList([0, 1, 2, 255]),
        'amount': GrafeoDecimal('12.50'),
        'ratio': double.nan,
      },
    );
    for (final MapEntry(key: how, value: rows) in rowsBothWays(
      'MATCH (n:Blob) RETURN n.data AS data, n.amount AS amount, '
      'n.ratio AS ratio, [n.data] AS listed',
    ).entries) {
      final row = rows.single;
      expect(row['data'], isA<Uint8List>(), reason: how);
      expect(row['data'], equals([0, 1, 2, 255]), reason: how);
      expect(row['amount'], equals(GrafeoDecimal('12.5')), reason: how);
      expect((row['ratio'] as double).isNaN, isTrue, reason: how);
      final listed = row['listed'] as List;
      expect(listed.single, isA<Uint8List>(), reason: how);
    }
  });
}
