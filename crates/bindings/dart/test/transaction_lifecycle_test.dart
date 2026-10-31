import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

void main() {
  late GrafeoDB db;

  setUp(() {
    db = GrafeoDB.memory();
    db.execute("INSERT (:Account {name: 'Vincent', balance: 1})");
  });

  tearDown(() {
    db.close();
  });

  group('transaction lifecycle', () {
    // A commit that fails (a write conflict) ends the transaction, so a
    // rollback after it (a catch block's) does nothing, as in C, Go, C#,
    // Python and Node.js, and does not hide the commit's error.
    test('rollback after a failed commit does nothing', () {
      final first = db.beginTransaction();
      final second = db.beginTransaction();
      first.execute('MATCH (a:Account) SET a.balance = 2');
      first.commit();
      second.execute('MATCH (a:Account) SET a.balance = 3');

      expect(
        () => second.commit(),
        throwsA(
          isA<GrafeoException>().having(
            (error) => error.message,
            'message',
            contains('Write-write conflict'),
          ),
        ),
      );
      second.rollback();
      second.rollback();

      final result = db.execute('MATCH (a:Account) RETURN a.balance');
      expect(result.rows.single['a.balance'], equals(2));
    });

    // The failed commit ended the transaction: it runs nothing more.
    test('a transaction whose commit failed runs nothing', () {
      final first = db.beginTransaction();
      final second = db.beginTransaction();
      first.execute('MATCH (a:Account) SET a.balance = 2');
      first.commit();
      second.execute('MATCH (a:Account) SET a.balance = 3');
      expect(() => second.commit(), throwsA(isA<GrafeoException>()));

      expect(
        () => second.execute('MATCH (a:Account) RETURN a.balance'),
        throwsA(isA<TransactionException>()),
      );
      expect(() => second.commit(), throwsA(isA<TransactionException>()));
    });

    // A rollback after a commit that succeeded is still an error.
    test('rollback after a successful commit throws', () {
      final tx = db.beginTransaction();
      tx.execute('MATCH (a:Account) SET a.balance = 2');
      tx.commit();
      expect(() => tx.rollback(), throwsA(isA<TransactionException>()));
      final result = db.execute('MATCH (a:Account) RETURN a.balance');
      expect(result.rows.single['a.balance'], equals(2));
    });
  });
}
