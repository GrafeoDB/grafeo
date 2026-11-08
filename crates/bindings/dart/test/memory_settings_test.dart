/// The memory settings through Dart (#751): `GrafeoOptions` with the memory limit, the query
/// memory limit and the spill limit (null is the default; a zero spill limit means nothing
/// spills) and `unlimitedSpill`, opened with `GrafeoDB.openWithOptions(path or null, options)`.
/// Default options are the defaults, a zero memory limit fails at the open, the settings read
/// back through `memoryUsage()`, and a file opened over its limit opens, reads and warns.
library;

import 'dart:io';

import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

const mib = 1024 * 1024;

Map<String, dynamic> _spill(Map<String, dynamic> usage) =>
    usage['spill'] as Map<String, dynamic>;

bool _warns(GrafeoDB db, String code) =>
    db.openWarnings().any((warning) => warning['code'] == code);

void main() {
  group('memory settings (#751)', () {
    test('each setting reads back through the report', () {
      final db = GrafeoDB.openWithOptions(
        null,
        GrafeoOptions(
          memoryLimit: 64 * mib,
          queryMemoryLimit: 8 * mib,
          spillLimit: 0,
        ),
      );
      try {
        final usage = db.memoryUsage();
        expect(usage['limit_bytes'], equals(64 * mib));
        expect(usage['query_limit_bytes'], equals(8 * mib));
        expect(_spill(usage)['limit_bytes'], equals(0));
      } finally {
        db.close();
      }
    });

    test('unlimited spill reads back as no limit', () {
      // A database file has a spill directory.
      final dir = Directory.systemTemp.createTempSync('grafeo-dart-');
      final db = GrafeoDB.openWithOptions(
        '${dir.path}${Platform.pathSeparator}unlimited.grafeo',
        GrafeoOptions(spillLimit: 16 * mib, unlimitedSpill: true),
      );
      try {
        expect(_spill(db.memoryUsage())['limit_bytes'], isNull);
      } finally {
        db.close();
        dir.deleteSync(recursive: true);
      }
    });

    test('an in-memory open spills nothing whatever the settings', () {
      final db = GrafeoDB.openWithOptions(
        null,
        GrafeoOptions(unlimitedSpill: true),
      );
      try {
        expect(_spill(db.memoryUsage())['limit_bytes'], equals(0));
      } finally {
        db.close();
      }
    });

    test('default options are the defaults', () {
      final defaults = GrafeoDB.memory();
      final db = GrafeoDB.openWithOptions(null, GrafeoOptions());
      try {
        expect(
          db.memoryUsage()['limit_bytes'],
          equals(defaults.memoryUsage()['limit_bytes']),
        );
        expect(
          db.memoryUsage()['query_limit_bytes'],
          equals(defaults.memoryUsage()['query_limit_bytes']),
        );
      } finally {
        db.close();
        defaults.close();
      }
    });

    test('a zero memory limit fails at the open', () {
      expect(
        () => GrafeoDB.openWithOptions(null, GrafeoOptions(memoryLimit: 0)),
        throwsA(isA<GrafeoException>()),
      );
    });

    test('an open over the limit warns and reads', () {
      final dir = Directory.systemTemp.createTempSync('grafeo-dart-');
      final path = '${dir.path}${Platform.pathSeparator}over.grafeo';
      try {
        final db = GrafeoDB.open(path);
        // Distinct 1 KiB strings: data that stays past the limit once reloaded.
        db.execute(
          "UNWIND range(1, 20000) AS i INSERT (:Fill {i: i, text: CAST(i AS STRING) || '${'x' * 1000}'})",
        );
        db.close();

        final tight = GrafeoDB.openWithOptions(
          path,
          GrafeoOptions(memoryLimit: mib),
        );
        try {
          expect(_warns(tight, 'MEMORY_LIMIT_PASSED'), isTrue);
          expect(tight.nodeCount, equals(20000));
        } finally {
          tight.close();
        }

        final roomy = GrafeoDB.open(path);
        try {
          expect(_warns(roomy, 'MEMORY_LIMIT_PASSED'), isFalse);
        } finally {
          roomy.close();
        }
      } finally {
        try {
          dir.deleteSync(recursive: true);
        } catch (_) {
          // Windows may hold file locks briefly after close.
        }
      }
    });
  });
}
