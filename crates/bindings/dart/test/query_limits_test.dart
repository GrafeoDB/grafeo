/// The path search budget through Dart (#492): `GrafeoOptions(pathSearchBudget: ...)`. Zero
/// fails at the open; the budget binds, with its error code and the setting named in the
/// message. The snapshot retention limit, `GrafeoOptions(snapshotRetentionLimit: ...)`: below
/// 1 MiB fails at the open; past it an open transaction throws `GRAFEO-T012`.
library;

import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

const _star =
    "INSERT (c:Star {n: 0}), (c)-[:E]->(:Leaf {n: 1}), (c)-[:E]->(:Leaf {n: 2}), "
    "(c)-[:E]->(:Leaf {n: 3}), (c)-[:E]->(:Leaf {n: 4}), (c)-[:E]->(:Leaf {n: 5})";
const _paths = 'MATCH (s:Star)-[:E*1..2]->(t) RETURN t.n AS n';

GrafeoDB _opened(GrafeoOptions options) {
  final db = GrafeoDB.openWithOptions(null, options);
  db.execute(_star);
  return db;
}

void _refused(GrafeoDB db, String query, String code, String setting) {
  expect(
    () => db.execute(query),
    throwsA(
      isA<GrafeoException>()
          .having((e) => e.message, 'message', contains(code))
          .having((e) => e.message, 'message', contains(setting)),
    ),
  );
}

void main() {
  group('query limits (#492)', () {
    test('zero is refused at the open', () {
      expect(
        () => GrafeoDB.openWithOptions(null, GrafeoOptions(pathSearchBudget: 0)),
        throwsA(isA<GrafeoException>()),
      );
    });

    test('the path search budget binds', () {
      final db = _opened(GrafeoOptions(pathSearchBudget: 1));
      _refused(db, _paths, 'GRAFEO-S001', 'path_search_budget');
      db.close();
    });
  });

  group('snapshot retention limit', () {
    test('below 1 MiB is refused at the open', () {
      expect(
        () => GrafeoDB.openWithOptions(
            null, GrafeoOptions(snapshotRetentionLimit: 512 * 1024)),
        throwsA(isA<GrafeoException>()
            .having((e) => e.message, 'message', contains('at least 1 MiB'))),
      );
    });

    // Past the limit (20 commits replace 64 KiB each, over 1 MiB) the open
    // transaction that read Alix throws GRAFEO-T012, and rolls back.
    test('binds an open transaction', () {
      final db = GrafeoDB.openWithOptions(
          null, GrafeoOptions(snapshotRetentionLimit: 1 << 20));
      db.execute("INSERT (:Person {name: 'Alix', bio: 'Amsterdam'})");
      const readBio = "MATCH (p:Person {name: 'Alix'}) RETURN p.bio AS bio";
      final tx = db.beginTransaction();
      tx.execute(readBio);
      final text = 'Vincent ' * (8 * 1024);
      for (var round = 0; round < 20; round++) {
        db.execute(
            "MATCH (p:Person {name: 'Alix'}) SET p.bio = '$round$text'");
      }
      expect(
        () => tx.execute(readBio),
        throwsA(isA<GrafeoException>()
            .having((e) => e.message, 'message', contains('GRAFEO-T012'))),
      );
      tx.rollback();
      db.close();
    });
  });
}
