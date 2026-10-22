import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

/// setSchema makes the check a query makes (#245): a schema that does not
/// exist is refused, and the current schema stays as it was.
void main() {
  test('setSchema of a missing schema throws and keeps the schema', () {
    final db = GrafeoDB.memory();
    addTearDown(db.close);
    expect(
      () => db.setSchema('missing'),
      throwsA(isA<GrafeoException>().having(
        (error) => error.toString(),
        'message',
        contains("Schema 'missing' does not exist"),
      )),
    );
    expect(db.currentSchema(), isNull);
  });

  test('setSchema of an existing schema scopes the queries', () {
    final db = GrafeoDB.memory();
    addTearDown(db.close);
    db.execute('CREATE SCHEMA reporting');
    db.setSchema('reporting');
    expect(db.currentSchema(), equals('reporting'));
    db.resetSchema();
    expect(db.currentSchema(), isNull);
  });
}
