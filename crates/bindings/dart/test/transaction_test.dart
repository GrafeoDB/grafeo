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

  group('transactions', () {
    test('commit makes changes visible', () {
      final tx = db.beginTransaction();
      tx.execute("INSERT (:Person {name: 'Alix'})");
      tx.execute("INSERT (:Person {name: 'Gus'})");
      tx.commit();

      final result = db.execute('MATCH (p:Person) RETURN p.name');
      expect(result.rows, hasLength(2));
    });

    test('rollback discards changes', () {
      final tx = db.beginTransaction();
      tx.execute("INSERT (:Person {name: 'Jules'})");
      tx.rollback();

      final result = db.execute('MATCH (p:Person) RETURN p.name');
      expect(result.rows, isEmpty);
    });

    test('query within transaction sees own writes', () {
      final tx = db.beginTransaction();
      tx.execute("INSERT (:Person {name: 'Mia'})");
      final result = tx.execute(
        "MATCH (p:Person) WHERE p.name = 'Mia' RETURN p.name",
      );
      expect(result.rows, hasLength(1));
      tx.rollback();
    });

    test('parameterized query in transaction', () {
      final tx = db.beginTransaction();
      tx.execute("INSERT (:City {name: 'Berlin'})");
      final result = tx.executeWithParams(
        r'MATCH (c:City) WHERE c.name = $name RETURN c.name',
        {'name': 'Berlin'},
      );
      expect(result.rows.first['c.name'], equals('Berlin'));
      tx.commit();
    });

    test('throws after commit', () {
      final tx = db.beginTransaction();
      tx.commit();
      expect(
        () => tx.execute('MATCH (n) RETURN n'),
        throwsA(isA<TransactionException>()),
      );
    });

    // A commit that succeeded leaves nothing to roll back: as in every
    // binding, a rollback after it is an error, and the commit stands.
    test('rollback after commit throws', () {
      final tx = db.beginTransaction();
      tx.execute("INSERT (:Person {name: 'Vincent'})");
      tx.commit();
      expect(
        () => tx.rollback(),
        throwsA(isA<TransactionException>()),
      );
      expect(db.execute('MATCH (p:Person) RETURN p.name').rows, hasLength(1));
    });

    test('throws after rollback', () {
      final tx = db.beginTransaction();
      tx.rollback();
      expect(
        () => tx.commit(),
        throwsA(isA<TransactionException>()),
      );
    });
  });
}
