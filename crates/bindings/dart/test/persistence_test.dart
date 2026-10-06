/// End-to-end tests for persistent storage through the Dart FFI binding.
///
/// Covers the exact scenario from GitHub issue #185 (StorageFormat not exposed
/// via C FFI, .grafeo producing a directory instead of a single file) plus
/// additional edge cases: open/close/reopen, save from memory, multi-cycle
/// accumulation, edge property persistence, and WAL checkpoint.
library;

import 'dart:io';

import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

/// Create a temp directory and return a path for a .grafeo file inside it.
(String dir, String dbPath) _tempDbPath(String name) {
  final dir = Directory.systemTemp.createTempSync('grafeo-dart-');
  return (dir.path, '${dir.path}${Platform.pathSeparator}$name');
}

/// Best-effort cleanup of the temp directory.
void _cleanup(String dirPath) {
  try {
    Directory(dirPath).deleteSync(recursive: true);
  } catch (_) {
    // Windows may hold file locks briefly after close.
  }
}

void main() {
  group('persistence', () {
    // -- Issue #185: .grafeo path must be a file, not a directory ------------

    test('open() with .grafeo extension produces a single file (issue #185)',
        () {
      final (dir, dbPath) = _tempDbPath('single.grafeo');
      try {
        final db = GrafeoDB.open(dbPath);
        db.execute("INSERT (:Node {x: 1})");
        db.close();

        expect(FileSystemEntity.isFileSync(dbPath), isTrue,
            reason: '.grafeo path should be a file, not a directory');
        expect(FileSystemEntity.isDirectorySync(dbPath), isFalse,
            reason: '.grafeo path should not be a directory');
      } finally {
        _cleanup(dir);
      }
    });

    test('openSingleFile() produces a single file', () {
      final (dir, dbPath) = _tempDbPath('explicit.grafeo');
      try {
        // Deprecated, and still works until 0.7.0 removes it.
        // ignore: deprecated_member_use_from_same_package
        final db = GrafeoDB.openSingleFile(dbPath);
        db.execute("INSERT (:Node {x: 1})");
        db.close();

        expect(FileSystemEntity.isFileSync(dbPath), isTrue,
            reason: 'openSingleFile should produce a file');
      } finally {
        _cleanup(dir);
      }
    });

    test('sidecar WAL is cleaned up after close', () {
      final (dir, dbPath) = _tempDbPath('wal_cleanup.grafeo');
      try {
        final db = GrafeoDB.open(dbPath);
        db.execute("INSERT (:Node {x: 1})");
        db.close();

        final walPath = '$dbPath.wal';
        expect(File(walPath).existsSync(), isFalse,
            reason: 'sidecar WAL should be removed after close');
      } finally {
        _cleanup(dir);
      }
    });

    // -- Create, close, reopen ----------------------------------------------

    test('data persists after close and reopen', () {
      final (dir, dbPath) = _tempDbPath('reopen.grafeo');
      try {
        // Create and populate
        var db = GrafeoDB.open(dbPath);
        db.execute("INSERT (:Person {name: 'Alix', age: 30})");
        db.execute("INSERT (:Person {name: 'Gus', age: 25})");
        db.execute(
          "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) "
          "INSERT (a)-[:KNOWS]->(b)",
        );
        expect(db.nodeCount, equals(2));
        expect(db.edgeCount, equals(1));
        db.close();

        // Reopen and verify
        db = GrafeoDB.open(dbPath);
        expect(db.nodeCount, equals(2));
        expect(db.edgeCount, equals(1));

        final result =
            db.execute('MATCH (p:Person) RETURN p.name ORDER BY p.name');
        final names =
            result.rows.map((r) => r['p.name'] as String).toList()..sort();
        expect(names, equals(['Alix', 'Gus']));
        db.close();
      } finally {
        _cleanup(dir);
      }
    });

    // -- Save in-memory to file ---------------------------------------------

    test('save in-memory database to .grafeo file', () {
      final (dir, dbPath) = _tempDbPath('saved.grafeo');
      try {
        var db = GrafeoDB.memory();
        db.execute("INSERT (:City {name: 'Amsterdam'})");
        db.execute("INSERT (:City {name: 'Berlin'})");
        db.save(dbPath);
        db.close();

        db = GrafeoDB.open(dbPath);
        expect(db.nodeCount, equals(2));

        final result =
            db.execute('MATCH (c:City) RETURN c.name ORDER BY c.name');
        final names =
            result.rows.map((r) => r['c.name'] as String).toList()..sort();
        expect(names, equals(['Amsterdam', 'Berlin']));
        db.close();
      } finally {
        _cleanup(dir);
      }
    });

    // -- Multiple reopen cycles ---------------------------------------------

    test('data accumulates across multiple open/close cycles', () {
      final (dir, dbPath) = _tempDbPath('cycles.grafeo');
      try {
        // Cycle 1
        var db = GrafeoDB.open(dbPath);
        db.execute("INSERT (:Person {name: 'Alix'})");
        db.close();

        // Cycle 2
        db = GrafeoDB.open(dbPath);
        expect(db.nodeCount, equals(1));
        db.execute("INSERT (:Person {name: 'Gus'})");
        db.close();

        // Cycle 3
        db = GrafeoDB.open(dbPath);
        expect(db.nodeCount, equals(2));
        db.execute("INSERT (:Person {name: 'Vincent'})");
        db.close();

        // Final check
        db = GrafeoDB.open(dbPath);
        expect(db.nodeCount, equals(3));
        final result = db.execute('MATCH (p:Person) RETURN p.name');
        final names =
            result.rows.map((r) => r['p.name'] as String).toList()..sort();
        expect(names, equals(['Alix', 'Gus', 'Vincent']));
        db.close();
      } finally {
        _cleanup(dir);
      }
    });

    // -- WAL checkpoint -----------------------------------------------------

    test('checkpoint followed by more writes all persist', () {
      final (dir, dbPath) = _tempDbPath('checkpoint.grafeo');
      try {
        var db = GrafeoDB.open(dbPath);
        db.execute("INSERT (:Person {name: 'Alix'})");
        db.walCheckpoint();
        db.execute("INSERT (:Person {name: 'Gus'})");
        db.close();

        db = GrafeoDB.open(dbPath);
        expect(db.nodeCount, equals(2));
        db.close();
      } finally {
        _cleanup(dir);
      }
    });

    // -- Edge properties persist --------------------------------------------

    test('edge properties survive close/reopen', () {
      final (dir, dbPath) = _tempDbPath('edgeprops.grafeo');
      try {
        var db = GrafeoDB.open(dbPath);
        db.execute("INSERT (:Person {name: 'Alix'})");
        db.execute("INSERT (:Person {name: 'Gus'})");
        db.execute(
          "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) "
          "INSERT (a)-[:KNOWS {since: 2020}]->(b)",
        );
        db.close();

        db = GrafeoDB.open(dbPath);
        final result = db.execute('MATCH ()-[e:KNOWS]->() RETURN e.since');
        expect(result.rows, hasLength(1));
        expect(result.rows.first['e.since'], equals(2020));
        db.close();
      } finally {
        _cleanup(dir);
      }
    });

    // -- Read-only mode -----------------------------------------------------

    test('openReadOnly can read persisted data', () {
      final (dir, dbPath) = _tempDbPath('readonly.grafeo');
      try {
        // Write data first
        var db = GrafeoDB.open(dbPath);
        db.execute("INSERT (:Person {name: 'Alix'})");
        db.close();

        // Open read-only and verify
        final ro = GrafeoDB.openReadOnly(dbPath);
        expect(ro.nodeCount, equals(1));
        final result = ro.execute("MATCH (p:Person) RETURN p.name");
        expect(result.rows.first['p.name'], equals('Alix'));
        ro.close();
      } finally {
        _cleanup(dir);
      }
    });

    test('openReadOnly rejects writes', () {
      final (dir, dbPath) = _tempDbPath('readonly_reject.grafeo');
      try {
        var db = GrafeoDB.open(dbPath);
        db.execute("INSERT (:Node {x: 1})");
        db.close();

        final ro = GrafeoDB.openReadOnly(dbPath);
        expect(
          () => ro.execute("INSERT (:Node {x: 2})"),
          throwsA(isA<GrafeoException>()),
        );
        ro.close();
      } finally {
        _cleanup(dir);
      }
    });
  });
}
