import 'dart:typed_data';

import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

/// A Uint8List travels in the {"$bytes": "..."} tag (0.6.1): it arrives as
/// BYTES (a base64 string before, which arrived as STRING), and BYTES come
/// back as a Uint8List with every byte equal: a few bytes, none, all 256
/// values. A string that looks like base64 stays a string, and a $bytes map
/// that is not base64, or has a second key, stays a map.
void main() {
  late GrafeoDB db;
  setUp(() => db = GrafeoDB.memory());
  tearDown(() => db.close());

  final all = Uint8List.fromList(List.generate(256, (i) => i));

  for (final value in [
    Uint8List.fromList([0, 1, 2, 255]),
    Uint8List(0),
    all,
  ]) {
    test('${value.length} bytes round-trip as BYTES', () {
      final row = db
          .executeWithParams(
            r'RETURN $b AS b, $b IS TYPED BYTES AS bytes, $b IS TYPED STRING AS str',
            {'b': value},
          )
          .rows
          .first;
      expect(row['bytes'], isTrue);
      expect(row['str'], isFalse);
      expect(row['b'], isA<Uint8List>());
      expect(row['b'], equals(value));
    });
  }

  test('stored bytes come back as a Uint8List, inside a list too', () {
    db.executeWithParams(
      r"INSERT (:Blob {name: 'Mia', data: $b, parts: $l})",
      {
        'b': all,
        'l': [
          Uint8List.fromList([7]),
          Uint8List(0)
        ],
      },
    );
    final row = db
        .execute('MATCH (n:Blob) RETURN n.data AS data, n.parts AS parts')
        .rows
        .first;
    expect(row['data'], isA<Uint8List>());
    expect(row['data'], equals(all));
    final parts = row['parts'] as List;
    expect(parts[0], isA<Uint8List>());
    expect(parts[0], equals([7]));
    expect(parts[1], isA<Uint8List>());
    expect(parts[1], isEmpty);

    final inner = db
        .executeWithParams(r'RETURN $m.k IS TYPED BYTES AS inner', {
          'm': {
            'k': Uint8List.fromList([1, 2])
          },
        })
        .rows
        .first;
    expect(inner['inner'], isTrue);
  });

  test('a string that looks like base64 stays a string', () {
    final row = db
        .executeWithParams(
          r'RETURN $s AS s, $s IS TYPED STRING AS str',
          {'s': 'AAEC/w=='},
        )
        .rows
        .first;
    expect(row['s'], equals('AAEC/w=='));
    expect(row['str'], isTrue);
  });

  for (final map in [
    {r'$bytes': 'not base64!'},
    {r'$bytes': 'AAEC/w'},
    {r'$bytes': 'AAEC_w=='},
    {r'$bytes': 'AAEC/x=='},
    {r'$bytes': 'AAEC /w=='},
    {r'$bytes': 'AAEC\n/w=='},
    {r'$bytes': 'AAEC/w==', 'name': 'Gus'},
  ]) {
    test('$map stays a map', () {
      final row = db
          .executeWithParams(
            r'RETURN $m AS m, $m IS TYPED BYTES AS bytes',
            {'m': map},
          )
          .rows
          .first;
      expect(row['bytes'], isFalse);
      expect(row['m'], equals(map));
    });
  }
}
