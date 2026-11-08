import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

void main() {
  late GrafeoDB db;

  setUp(() {
    db = GrafeoDB.memory();
  });

  tearDown(() {
    db.close();
  });

  group('lifecycle', () {
    test('version returns a non-empty string', () {
      final v = GrafeoDB.version();
      expect(v, isNotEmpty);
      expect(v, matches(RegExp(r'^\d+\.\d+\.\d+')));
    });

    test('info returns a map with version', () {
      final dbInfo = db.info();
      expect(dbInfo, isA<Map<String, dynamic>>());
      expect(dbInfo, contains('version'));
    });

    test('empty database has zero counts', () {
      expect(db.nodeCount, equals(0));
      expect(db.edgeCount, equals(0));
    });

    test('double close is safe', () {
      db.close();
      db.close(); // should not throw
    });

    test('methods throw after close', () {
      db.close();
      expect(() => db.execute('RETURN 1'), throwsA(isA<DatabaseException>()));
      expect(() => db.nodeCount, throwsA(isA<DatabaseException>()));
    });
  });

  group('query execution', () {
    test('creates and retrieves a node', () {
      db.execute("INSERT (:Person {name: 'Alix', age: 30})");
      final result = db.execute(
        "MATCH (p:Person) WHERE p.name = 'Alix' RETURN p.name, p.age",
      );

      expect(result.rows, hasLength(1));
      expect(result.rows.first['p.name'], equals('Alix'));
      expect(result.rows.first['p.age'], equals(30));
    });

    test('handles parameterized queries', () {
      db.execute("INSERT (:City {name: 'Amsterdam'})");
      final result = db.executeWithParams(
        r'MATCH (c:City) WHERE c.name = $name RETURN c.name',
        {'name': 'Amsterdam'},
      );

      expect(result.rows, hasLength(1));
      expect(result.rows.first['c.name'], equals('Amsterdam'));
    });

    test('empty result has zero rows', () {
      final result = db.execute('MATCH (n:Nothing) RETURN n');
      expect(result.rows, isEmpty);
      expect(result.columns, isEmpty);
    });

    test('invalid query throws QueryException', () {
      expect(
        () => db.execute('THIS IS NOT VALID GQL'),
        throwsA(isA<GrafeoException>()),
      );
    });

    test('counts update after inserts', () {
      expect(db.nodeCount, equals(0));
      db.execute("INSERT (:Person {name: 'Alix'})");
      expect(db.nodeCount, equals(1));
      db.execute("INSERT (:Person {name: 'Gus'})");
      expect(db.nodeCount, equals(2));
    });
  });
}
