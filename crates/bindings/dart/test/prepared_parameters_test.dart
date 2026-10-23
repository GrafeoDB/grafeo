// A statement called again with other values answers for those values: the
// engine optimizes it once with its parameters in and fills a copy of that
// plan with each call's values, here through the Dart binding.

import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

void main() {
  late GrafeoDB db;

  setUp(() {
    db = GrafeoDB.memory();
    for (final (name, age) in [
      ('Alix', 30),
      ('Gus', 25),
      ('Vincent', 41),
      ('Mia', 25),
    ]) {
      db.executeWithParams(
        'INSERT (:Person {name: \$name, age: \$age})',
        {'name': name, 'age': age},
      );
    }
  });

  tearDown(() {
    db.close();
  });

  List<String> column(QueryResult result, String name) =>
      result.rows.map((row) => '${row[name]}').toList();

  test('a lookup takes each call\'s value', () {
    const query = 'MATCH (p:Person) WHERE p.name = \$name RETURN p.age AS age';
    for (final (name, ages) in [
      ('Gus', ['25']),
      ('Alix', ['30']),
      ('Nobody', <String>[]),
      ('Gus', ['25']),
    ]) {
      expect(column(db.executeWithParams(query, {'name': name}), 'age'),
          equals(ages));
    }
  });

  test('SKIP and LIMIT take each call\'s values', () {
    const query =
        'MATCH (p:Person) RETURN p.name AS name ORDER BY p.name SKIP \$skip LIMIT \$limit';
    final names = ['Alix', 'Gus', 'Mia', 'Vincent'];
    for (final (skip, limit) in [(0, 1), (1, 2), (3, 5), (0, 4)]) {
      final end = skip + limit < names.length ? skip + limit : names.length;
      expect(
        column(db.executeWithParams(query, {'skip': skip, 'limit': limit}),
            'name'),
        equals(names.sublist(skip, end)),
      );
    }
  });

  test('a value of another type answers as its literal', () {
    const query = 'MATCH (p:Person) WHERE p.age = \$age RETURN p.name AS name';
    for (final (value, literal) in <(Object, String)>[
      (25, '25'),
      ('25', "'25'"),
      (25, '25'),
      (true, 'true'),
    ]) {
      final got = column(db.executeWithParams(query, {'age': value}), 'name')
        ..sort();
      final expected =
          column(db.execute(query.replaceAll('\$age', literal)), 'name')
            ..sort();
      expect(got, equals(expected));
    }
  });
}
