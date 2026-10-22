import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

void main() {
  group('status code mapping', () {
    test('ok maps to 0', () {
      expect(GrafeoStatus.ok.code, equals(0));
    });

    test('query maps to 2', () {
      expect(GrafeoStatus.query.code, equals(2));
    });

    test('transaction maps to 3', () {
      expect(GrafeoStatus.transaction.code, equals(3));
    });

    test('fromCode with unknown value falls back to internal', () {
      expect(GrafeoStatus.fromCode(99), equals(GrafeoStatus.internal));
    });

    test('all status codes are unique', () {
      final codes = GrafeoStatus.values.map((s) => s.code).toSet();
      expect(codes.length, equals(GrafeoStatus.values.length));
    });
  });

  group('exception hierarchy', () {
    test('classifyError returns QueryException for status 2', () {
      final ex = classifyError(2, 'bad query');
      expect(ex, isA<QueryException>());
      expect(ex.message, equals('bad query'));
      expect(ex.status, equals(GrafeoStatus.query));
    });

    test('classifyError returns TransactionException for status 3', () {
      final ex = classifyError(3, 'conflict');
      expect(ex, isA<TransactionException>());
    });

    test('classifyError returns StorageException for status 4', () {
      final ex = classifyError(4, 'disk full');
      expect(ex, isA<StorageException>());
    });

    test('classifyError returns StorageException for IO status 5', () {
      final ex = classifyError(5, 'io error');
      expect(ex, isA<StorageException>());
    });

    test('classifyError returns SerializationException for status 6', () {
      final ex = classifyError(6, 'bad data');
      expect(ex, isA<SerializationException>());
    });

    test('classifyError returns DatabaseException for status 1', () {
      final ex = classifyError(1, 'db error');
      expect(ex, isA<DatabaseException>());
    });

    test('classifyError returns DatabaseException for unknown status', () {
      final ex = classifyError(99, 'unknown');
      expect(ex, isA<DatabaseException>());
    });

    test('GrafeoException toString includes type and message', () {
      final ex = QueryException('syntax error near X', GrafeoStatus.query);
      expect(ex.toString(), contains('QueryException'));
      expect(ex.toString(), contains('syntax error near X'));
    });
  });

  group('runtime error handling', () {
    late GrafeoDB db;

    setUp(() {
      db = GrafeoDB.memory();
    });

    tearDown(() {
      db.close();
    });

    test('invalid GQL throws GrafeoException', () {
      expect(
        () => db.execute('NOT VALID GQL AT ALL'),
        throwsA(isA<GrafeoException>()),
      );
    });

    test('exception message is descriptive', () {
      try {
        db.execute('COMPLETELY INVALID');
        fail('Should have thrown');
      } on GrafeoException catch (e) {
        expect(e.message, isNotEmpty);
      }
    });
  });
}
