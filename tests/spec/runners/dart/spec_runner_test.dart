/// Dart spec runner for .gtest files.
///
/// Discovers all .gtest files under tests/spec/, parses them with a line-based
/// parser (no YAML dependency), and creates `package:test` groups and tests
/// that execute queries through the Dart GrafeoDB bindings.
///
/// Run with:
///   dart test spec_runner_test.dart
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:crypto/crypto.dart' as crypto;
import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

// =============================================================================
// Paths
// =============================================================================

/// Resolve spec directory relative to this test file.
final _specDir = _resolveSpecDir();
final _datasetsDir = Directory('${_specDir.path}/datasets');

Directory _resolveSpecDir() {
  // Walk up from the working directory until we find Cargo.toml (repo root).
  var dir = Directory.current;
  while (dir.path != dir.parent.path) {
    if (File('${dir.path}/Cargo.toml').existsSync()) {
      return Directory('${dir.path}/tests/spec');
    }
    dir = dir.parent;
  }
  // Fallback: assume cwd is repo root
  return Directory('${Directory.current.path}/tests/spec');
}

// =============================================================================
// Language dispatch
// =============================================================================

/// Normalise a language key to the string accepted by `executeLanguage`.
String _normaliseLanguage(String lang) {
  return switch (lang) {
    'sql_pgq' => 'sql-pgq',
    _ => lang,
  };
}

/// Cached set of compiled feature flags from db.info().
Set<String>? _compiledFeatures;

Set<String> _getCompiledFeatures() {
  if (_compiledFeatures == null) {
    final db = GrafeoDB.memory();
    try {
      final info = db.info();
      final features = info['features'] as List<dynamic>?;
      _compiledFeatures = features?.map((e) => e.toString()).toSet() ?? {};
    } finally {
      db.close();
    }
  }
  return _compiledFeatures!;
}

/// Capabilities provided by this runner itself (not compiled into the native
/// library). For example, Dart natively supports 64-bit integers, so
/// "int64-safe" is always available.
const _runnerCapabilities = {'int64-safe'};

/// Check whether a language or feature requirement is available.
bool _isAvailable(String requirement) {
  final key = _normaliseLanguage(requirement).replaceAll('_', '-');
  if (key == 'gql' || key.isEmpty) return true;
  return _getCompiledFeatures().contains(key) || _runnerCapabilities.contains(key);
}

// =============================================================================
// Query execution helpers
// =============================================================================

/// Execute a query in the given [language] via `executeLanguage`.
QueryResult _executeQuery(
  GrafeoDB db,
  String language,
  String query, {
  Map<String, String>? params,
}) {
  final lang = _normaliseLanguage(language);
  final coerced = _coerceParams(params);
  if (lang == 'gql' || lang.isEmpty) {
    return coerced != null
        ? db.executeWithParams(query, coerced)
        : db.execute(query);
  }
  return db.executeLanguage(lang, query, params: coerced);
}

// The number syntax of the Rust reference runner (crates/grafeo-spec-tests/
// build.rs): decimal only. int.tryParse and double.tryParse also read `0x1F`,
// `Infinity` and `NaN`, which the reference runner keeps as strings.
final _decimalInteger = RegExp(r'^[+-]?[0-9]+$');
final _decimalNumber =
    RegExp(r'^[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$');

/// The value of a parameter written as [text] in a .gtest file, by the rule of
/// the Rust reference runner that every runner follows: a quoted value is a
/// string, whatever it reads like; a bare value starting with `[` or `{` is
/// JSON, whose numbers follow the rule of bare ones; a bare decimal integer
/// that fits 64 bits is an int and any other bare decimal number a double, an
/// error past the f64 range; a bare `true` or `false` is a bool; any other bare
/// value is a string.
dynamic _paramValue(String text) {
  text = text.trim();
  if (_isQuoted(text)) return _unquote(text);
  if (text.startsWith('[') || text.startsWith('{')) {
    // jsonDecode makes an integer past 64 bits a double already, and a
    // number past the f64 range (1e400) infinity, which is an error.
    final value = jsonDecode(text);
    if (_hasInfinity(value)) {
      throw FormatException('a number in parameter $text is out of the f64 range');
    }
    return value;
  }
  if (_decimalInteger.hasMatch(text)) {
    // null past the 64-bit range: then it is a double, as in Rust.
    final integer = int.tryParse(text);
    if (integer != null) return integer;
  }
  if (_decimalNumber.hasMatch(text)) {
    final number = double.parse(text);
    if (!number.isFinite) {
      throw FormatException('parameter $text is out of the f64 range');
    }
    return number;
  }
  if (text == 'true' || text == 'false') return text == 'true';
  return text;
}

/// Whether a decoded JSON [value] holds a number past the f64 range.
bool _hasInfinity(dynamic value) {
  if (value is double) return value.isInfinite;
  if (value is List) return value.any(_hasInfinity);
  if (value is Map) return value.values.any(_hasInfinity);
  return false;
}

/// Type each parameter by [_paramValue]; null when there are none.
Map<String, dynamic>? _coerceParams(Map<String, String>? raw) {
  if (raw == null || raw.isEmpty) return null;
  return raw.map((key, text) => MapEntry(key, _paramValue(text)));
}

/// Load a .setup dataset file into [db] using GQL.
void _loadDataset(GrafeoDB db, String datasetName) {
  final setupFile = File('${_datasetsDir.path}/$datasetName.setup');
  if (!setupFile.existsSync()) {
    fail('Dataset file not found: ${setupFile.path}');
  }
  final content = setupFile.readAsStringSync();
  for (final line in content.split(RegExp(r'\r?\n'))) {
    final trimmed = line.trim();
    if (trimmed.isEmpty || trimmed.startsWith('#')) continue;
    db.execute(trimmed);
  }
}

// =============================================================================
// Value serialization (canonical format matching Rust runner)
// =============================================================================

/// Convert a Dart value to the canonical string representation used by
/// the Rust spec runner's `value_to_string`.
String _valueToString(dynamic val) {
  if (val == null) return 'null';
  if (val is bool) return val ? 'true' : 'false';
  if (val is int) return val.toString();
  if (val is double) {
    if (val.isNaN) return 'NaN';
    if (val.isInfinite) return val > 0 ? 'Infinity' : '-Infinity';
    // Rust's Display for f64 drops ".0" for whole numbers.
    if (val == val.truncateToDouble() && val.abs() < (1 << 53)) {
      return val.toInt().toString();
    }
    return val.toString();
  }
  if (val is List) {
    final inner = val.map(_valueToString).join(', ');
    return '[$inner]';
  }
  if (val is Map) {
    // Temporal type-tagged objects from C FFI: {"$date": "2024-06-15"}
    if (val.length == 1) {
      final key = val.keys.first.toString();
      switch (key) {
        case r'$date':
        case r'$time':
        case r'$datetime':
        case r'$zoned_datetime':
        case r'$duration':
          return val.values.first.toString();
        case r'$timestamp_us':
          return (val.values.first as num).toInt().toString();
      }
    }
    // Duration: {months, days, nanos} -> ISO 8601
    if (val.length == 3 &&
        val.containsKey('months') &&
        val.containsKey('days') &&
        val.containsKey('nanos')) {
      return _durationToIso(
        (val['months'] as num).toInt(),
        (val['days'] as num).toInt(),
        (val['nanos'] as num).toInt(),
      );
    }
    final entries = val.entries
        .map((e) => '${e.key}: ${_valueToString(e.value)}')
        .toList()
      ..sort();
    return '{${entries.join(', ')}}';
  }
  if (val is DateTime) return val.toIso8601String();
  if (val is Duration) {
    // Format as ISO duration
    final hours = val.inHours;
    final minutes = val.inMinutes.remainder(60);
    final seconds = val.inSeconds.remainder(60);
    final buf = StringBuffer('PT');
    if (hours > 0) buf.write('${hours}H');
    if (minutes > 0) buf.write('${minutes}M');
    if (seconds > 0 || (hours == 0 && minutes == 0)) buf.write('${seconds}S');
    return buf.toString();
  }
  return val.toString();
}

/// Convert {months, days, nanos} to ISO 8601 duration string.
String _durationToIso(int totalMonths, int days, int nanos) {
  final years = totalMonths ~/ 12;
  final months = totalMonths % 12;
  final hours = nanos ~/ 3600000000000;
  var rem = nanos % 3600000000000;
  final minutes = rem ~/ 60000000000;
  rem = rem % 60000000000;
  final seconds = rem ~/ 1000000000;
  final subNanos = rem % 1000000000;

  final buf = StringBuffer('P');
  if (years != 0) buf.write('${years}Y');
  if (months != 0) buf.write('${months}M');
  if (days != 0) buf.write('${days}D');

  final timeBuf = StringBuffer();
  if (hours != 0) timeBuf.write('${hours}H');
  if (minutes != 0) timeBuf.write('${minutes}M');
  if (seconds != 0 || subNanos != 0) {
    if (subNanos != 0) {
      final frac = subNanos.toString().padLeft(9, '0').replaceAll(RegExp(r'0+$'), '');
      timeBuf.write('$seconds.${frac}S');
    } else {
      timeBuf.write('${seconds}S');
    }
  }
  if (timeBuf.isNotEmpty) {
    buf.write('T');
    buf.write(timeBuf);
  }

  final result = buf.toString();
  return result == 'P' ? 'P0D' : result;
}

/// Convert a QueryResult to rows of canonical strings.
List<List<String>> _resultToRows(QueryResult result) {
  final columns = result.columns;
  final rows = <List<String>>[];
  for (final row in result.rows) {
    final r = <String>[];
    for (final col in columns) {
      r.add(_valueToString(row[col]));
    }
    rows.add(r);
  }
  return rows;
}

// =============================================================================
// Assertions
// =============================================================================

/// Assert rows match after sorting both sides.
void _assertRowsSorted(QueryResult result, List<List<String>> expected) {
  final actual = _resultToRows(result);
  final sortedActual = List<List<String>>.from(actual)
    ..sort((a, b) => a.join('|').compareTo(b.join('|')));
  final sortedExpected = List<List<String>>.from(expected)
    ..sort((a, b) => a.join('|').compareTo(b.join('|')));

  expect(sortedActual.length, equals(sortedExpected.length),
      reason:
          'Row count mismatch: got ${sortedActual.length}, '
          'expected ${sortedExpected.length}\n'
          'Actual: $sortedActual\nExpected: $sortedExpected');

  for (var i = 0; i < sortedActual.length; i++) {
    for (var j = 0; j < sortedActual[i].length; j++) {
      expect(sortedActual[i][j], equals(sortedExpected[i][j]),
          reason:
              'Mismatch at sorted row $i, col $j: '
              "got '${sortedActual[i][j]}', expected '${sortedExpected[i][j]}'\n"
              'Actual row: ${sortedActual[i]}\n'
              'Expected row: ${sortedExpected[i]}');
    }
  }
}

/// Assert rows match in exact order.
void _assertRowsOrdered(QueryResult result, List<List<String>> expected) {
  final actual = _resultToRows(result);

  expect(actual.length, equals(expected.length),
      reason:
          'Row count mismatch: got ${actual.length}, '
          'expected ${expected.length}\n'
          'Actual: $actual\nExpected: $expected');

  for (var i = 0; i < actual.length; i++) {
    for (var j = 0; j < actual[i].length; j++) {
      expect(actual[i][j], equals(expected[i][j]),
          reason:
              'Mismatch at row $i, col $j: '
              "got '${actual[i][j]}', expected '${expected[i][j]}'\n"
              'Actual row: ${actual[i]}\n'
              'Expected row: ${expected[i]}');
    }
  }
}

/// Assert rows match with floating-point tolerance.
void _assertRowsWithPrecision(
  QueryResult result,
  List<List<String>> expected,
  int precision,
) {
  final actual = _resultToRows(result);
  final tolerance = math.pow(10, -precision).toDouble();

  expect(actual.length, equals(expected.length),
      reason:
          'Row count mismatch: got ${actual.length}, '
          'expected ${expected.length}');

  for (var i = 0; i < actual.length; i++) {
    for (var j = 0; j < actual[i].length; j++) {
      final a = actual[i][j];
      final e = expected[i][j];
      final af = double.tryParse(a);
      final ef = double.tryParse(e);
      if (af != null && ef != null) {
        expect((af - ef).abs() < tolerance, isTrue,
            reason:
                'Float mismatch at row $i, col $j: '
                'got $af, expected $ef (tolerance $tolerance)');
      } else {
        expect(a, equals(e),
            reason: "Mismatch at row $i, col $j: got '$a', expected '$e'");
      }
    }
  }
}

/// Assert that columns match.
void _assertColumns(QueryResult result, List<String> expected) {
  expect(result.columns, equals(expected),
      reason:
          'Column mismatch: got ${result.columns}, expected $expected');
}

/// Assert that the MD5 hash of sorted, pipe-delimited rows matches.
/// Mirrors assert_hash in the Rust runner.
void _assertHash(QueryResult result, String expectedHash) {
  final rows = _resultToRows(result);
  rows.sort((a, b) => a.join('|').compareTo(b.join('|')));

  final sink = AccumulatorSink<crypto.Digest>();
  final output = crypto.md5.startChunkedConversion(sink);
  for (final row in rows) {
    output.add(utf8.encode('${row.join('|')}\n'));
  }
  output.close();
  final actualHash = sink.events.single.toString();

  expect(actualHash, equals(expectedHash),
      reason: 'Hash mismatch: got $actualHash, expected $expectedHash\n'
          'Rows: $rows');
}

/// Accumulator sink for chunked hash computation.
class AccumulatorSink<T> implements Sink<T> {
  final events = <T>[];

  @override
  void add(T event) => events.add(event);

  @override
  void close() {}
}

// =============================================================================
// .gtest line-based parsing (no YAML dependency)
// =============================================================================

/// Parsed metadata from a .gtest file.
class _Meta {
  String language;
  String dataset;
  List<String> requires;
  List<String> tags;

  _Meta({
    this.language = 'gql',
    this.dataset = 'empty',
    List<String>? requires,
    List<String>? tags,
  })  : requires = requires ?? [],
        tags = tags ?? [];
}

/// Parsed expect block from a test case.
class _Expect {
  List<List<String>> rows;
  bool ordered;
  int? count;
  bool empty;
  String? error;
  String? hash;
  int? precision;
  List<String> columns;

  _Expect({
    List<List<String>>? rows,
    this.ordered = false,
    this.count,
    this.empty = false,
    this.error,
    this.hash,
    this.precision,
    List<String>? columns,
  })  : rows = rows ?? [],
        columns = columns ?? [];
}

/// Parsed test case.
class _TestCase {
  String name;
  String? query;
  List<String> statements;
  List<String> setup;
  String? skip;
  String? language;
  String? dataset;
  _Expect expect;
  Map<String, String> variants;
  List<String> tags;
  List<String> requires;

  /// Each parameter value as the file writes it, quotes included: whether it
  /// was quoted decides its type (see [_paramValue]).
  Map<String, String> params;

  _TestCase({
    this.name = '',
    this.query,
    List<String>? statements,
    List<String>? setup,
    this.skip,
    this.language,
    this.dataset,
    _Expect? expect,
    Map<String, String>? variants,
    List<String>? tags,
    List<String>? requires,
    Map<String, String>? params,
  })  : statements = statements ?? [],
        setup = setup ?? [],
        expect = expect ?? _Expect(),
        variants = variants ?? {},
        tags = tags ?? [],
        requires = requires ?? [],
        params = params ?? {};
}

/// Parsed .gtest file.
class _GtestFile {
  final _Meta meta;
  final List<_TestCase> tests;

  _GtestFile(this.meta, this.tests);
}

/// Mutable parse context that tracks the current line position.
class _ParseContext {
  final List<String> lines;
  int idx;

  _ParseContext(this.lines, [this.idx = 0]);
}

// ---------------------------------------------------------------------------
// Top-level parser
// ---------------------------------------------------------------------------

/// Parse a .gtest file using the line-based parser.
_GtestFile _parseGtestFile(File file) {
  final content = file.readAsStringSync();
  final lines = content.split(RegExp(r'\r?\n'));
  final ctx = _ParseContext(lines);

  _skipBlankAndComments(ctx);
  final meta = _parseMeta(ctx);
  _skipBlankAndComments(ctx);
  final tests = _parseTests(ctx);
  return _GtestFile(meta, tests);
}

// ---------------------------------------------------------------------------
// Meta block
// ---------------------------------------------------------------------------

_Meta _parseMeta(_ParseContext ctx) {
  final meta = _Meta();
  _expectLine(ctx, 'meta:');
  while (ctx.idx < ctx.lines.length) {
    _skipBlankAndComments(ctx);
    if (ctx.idx >= ctx.lines.length) break;
    final line = ctx.lines[ctx.idx];
    if (!line.startsWith(' ') && !line.startsWith('\t')) break;
    final kv = _parseKV(line.trim());
    if (kv == null) {
      ctx.idx++;
      continue;
    }
    final (key, value) = kv;
    switch (key) {
      case 'language':
        meta.language = value;
      case 'dataset':
        meta.dataset = value;
      case 'requires':
        meta.requires = _parseYamlList(value);
      case 'tags':
        meta.tags = _parseYamlList(value);
    }
    ctx.idx++;
  }
  return meta;
}

// ---------------------------------------------------------------------------
// Tests list
// ---------------------------------------------------------------------------

List<_TestCase> _parseTests(_ParseContext ctx) {
  _skipBlankAndComments(ctx);
  _expectLine(ctx, 'tests:');
  final tests = <_TestCase>[];
  while (ctx.idx < ctx.lines.length) {
    _skipBlankAndComments(ctx);
    if (ctx.idx >= ctx.lines.length) break;
    final trimmed = ctx.lines[ctx.idx].trim();
    if (trimmed.startsWith('- name:')) {
      tests.add(_parseSingleTest(ctx));
    } else {
      break;
    }
  }
  return tests;
}

_TestCase _parseSingleTest(_ParseContext ctx) {
  final tc = _TestCase();

  // First line: "- name: xxx"
  final first = ctx.lines[ctx.idx].trim();
  final kv = _parseKV(first.substring(2)); // strip "- "
  if (kv != null) tc.name = _unquote(kv.$2);
  ctx.idx++;

  while (ctx.idx < ctx.lines.length) {
    final line = ctx.lines[ctx.idx];
    final trimmed = line.trim();
    if (trimmed.startsWith('#')) {
      ctx.idx++;
      continue;
    }
    if (trimmed.startsWith('- name:')) break;
    if (trimmed.isEmpty) {
      ctx.idx++;
      continue;
    }

    final kv2 = _parseKV(trimmed);
    if (kv2 == null) {
      ctx.idx++;
      continue;
    }
    final (key, value) = kv2;
    switch (key) {
      case 'query':
        if (value == '|') {
          tc.query = _parseBlockScalar(ctx);
        } else {
          tc.query = _unquote(value);
          ctx.idx++;
        }
      case 'skip':
        tc.skip = _unquote(value);
        ctx.idx++;
      case 'language':
        tc.language = _unquote(value);
        ctx.idx++;
      case 'dataset':
        tc.dataset = _unquote(value);
        ctx.idx++;
      case 'setup':
        ctx.idx++;
        tc.setup = _parseStringList(ctx);
      case 'statements':
        ctx.idx++;
        tc.statements = _parseStringList(ctx);
      case 'tags':
        tc.tags = _parseYamlList(value);
        ctx.idx++;
      case 'requires':
        tc.requires = _parseYamlList(value);
        ctx.idx++;
      case 'params':
        ctx.idx++;
        tc.params = _parseParams(ctx);
      case 'expect':
        ctx.idx++;
        tc.expect = _parseExpectBlock(ctx);
      case 'variants':
        ctx.idx++;
        tc.variants = _parseMap(ctx, 6);
      default:
        ctx.idx++;
    }
  }
  return tc;
}

// ---------------------------------------------------------------------------
// Expect block
// ---------------------------------------------------------------------------

_Expect _parseExpectBlock(_ParseContext ctx) {
  final e = _Expect();
  while (ctx.idx < ctx.lines.length) {
    final line = ctx.lines[ctx.idx];
    final trimmed = line.trim();
    if (trimmed.startsWith('#')) {
      ctx.idx++;
      continue;
    }
    if (trimmed.startsWith('- name:')) break;
    if (!line.startsWith(' ') && !line.startsWith('\t') && trimmed.isNotEmpty) {
      break;
    }
    if (trimmed.isEmpty) {
      ctx.idx++;
      continue;
    }

    final kv = _parseKV(trimmed);
    if (kv == null) break;
    final (key, value) = kv;
    switch (key) {
      case 'ordered':
        e.ordered = value == 'true';
        ctx.idx++;
      case 'count':
        e.count = int.parse(value);
        ctx.idx++;
      case 'empty':
        e.empty = value == 'true';
        ctx.idx++;
      case 'error':
        e.error = _unquote(value);
        ctx.idx++;
      case 'hash':
        e.hash = _unquote(value);
        ctx.idx++;
      case 'precision':
        e.precision = int.parse(value);
        ctx.idx++;
      case 'columns':
        e.columns = _parseYamlList(value);
        ctx.idx++;
      case 'rows':
        ctx.idx++;
        e.rows = _parseRows(ctx);
      default:
        ctx.idx++;
    }
  }
  return e;
}

List<List<String>> _parseRows(_ParseContext ctx) {
  final rows = <List<String>>[];
  while (ctx.idx < ctx.lines.length) {
    final trimmed = ctx.lines[ctx.idx].trim();
    if (trimmed.startsWith('#')) {
      ctx.idx++;
      continue;
    }
    if (trimmed.isEmpty) {
      ctx.idx++;
      continue;
    }
    if (trimmed.startsWith('- [')) {
      rows.add(_parseInlineList(trimmed.substring(2)));
      ctx.idx++;
    } else {
      break;
    }
  }
  return rows;
}

// ---------------------------------------------------------------------------
// Line-based primitives
// ---------------------------------------------------------------------------

/// Split a string on the first unquoted colon. Returns (key, value) or null.
(String, String)? _parseKV(String s) {
  var inSingle = false;
  var inDouble = false;
  for (var i = 0; i < s.length; i++) {
    final c = s[i];
    if (c == "'" && !inDouble) {
      inSingle = !inSingle;
    } else if (c == '"' && !inSingle) {
      inDouble = !inDouble;
    } else if (c == ':' && !inSingle && !inDouble) {
      final key = s.substring(0, i).trim();
      final value = s.substring(i + 1).trim();
      if (key.isNotEmpty) return (key, value);
    }
  }
  return null;
}

/// Whether [s] is written in single or double quotes.
bool _isQuoted(String s) {
  s = s.trim();
  return s.length >= 2 &&
      ((s.startsWith('"') && s.endsWith('"')) ||
          (s.startsWith("'") && s.endsWith("'")));
}

/// Strip surrounding quotes and unescape YAML-level escapes only.
/// Do NOT process `\n` or `\t`: those are GQL string escapes handled
/// by the engine's parser.
String _unquote(String s) {
  s = s.trim();
  if (_isQuoted(s)) {
    return s
        .substring(1, s.length - 1)
        .replaceAll(r'\\', '\x00')
        .replaceAll(r'\"', '"')
        .replaceAll(r"\'", "'")
        .replaceAll('\x00', r'\');
  }
  return s;
}

/// Parse a YAML-style inline list like `[a, b, c]` into strings.
List<String> _parseYamlList(String s) {
  s = s.trim();
  if (s == '[]' || s.isEmpty) return [];
  if (s.startsWith('[') && s.endsWith(']')) {
    return s
        .substring(1, s.length - 1)
        .split(',')
        .map((v) => _unquote(v.trim()))
        .where((v) => v.isNotEmpty)
        .toList();
  }
  return [_unquote(s)];
}

/// Parse an inline list with nested bracket awareness, e.g. `[val1, [a, b]]`.
List<String> _parseInlineList(String s) {
  s = s.trim();
  if (!s.startsWith('[') || !s.endsWith(']')) return [_unquote(s)];
  final inner = s.substring(1, s.length - 1);
  final items = <String>[];
  var current = StringBuffer();
  var depth = 0;
  var inSingle = false;
  var inDouble = false;
  for (final c in inner.split('')) {
    if (c == "'" && !inDouble && depth == 0) {
      inSingle = !inSingle;
      current.write(c);
    } else if (c == '"' && !inSingle && depth == 0) {
      inDouble = !inDouble;
      current.write(c);
    } else if ((c == '[' || c == '{') && !inSingle && !inDouble) {
      depth++;
      current.write(c);
    } else if ((c == ']' || c == '}') && !inSingle && !inDouble) {
      depth--;
      current.write(c);
    } else if (c == ',' && depth == 0 && !inSingle && !inDouble) {
      items.add(_unquote(current.toString().trim()));
      current = StringBuffer();
    } else {
      current.write(c);
    }
  }
  final last = current.toString().trim();
  if (last.isNotEmpty) items.add(_unquote(last));
  return items;
}

/// Parse a YAML-style dash list (lines starting with `- `).
List<String> _parseStringList(_ParseContext ctx) {
  final items = <String>[];
  while (ctx.idx < ctx.lines.length) {
    final trimmed = ctx.lines[ctx.idx].trim();
    if (trimmed.startsWith('#')) {
      ctx.idx++;
      continue;
    }
    if (trimmed.isEmpty) {
      ctx.idx++;
      continue;
    }
    if (trimmed.startsWith('- ')) {
      final value = trimmed.substring(2);
      if (value == '|') {
        items.add(_parseBlockScalar(ctx));
      } else {
        items.add(_unquote(value));
        ctx.idx++;
      }
    } else {
      break;
    }
  }
  return items;
}

/// Parse a key-value map at the given minimum indentation level.
Map<String, String> _parseMap(_ParseContext ctx, int minIndent) {
  final map = <String, String>{};
  while (ctx.idx < ctx.lines.length) {
    final line = ctx.lines[ctx.idx];
    final trimmed = line.trim();
    if (trimmed.startsWith('#') || trimmed.isEmpty) {
      ctx.idx++;
      continue;
    }
    if (trimmed.startsWith('- name:')) break;
    final indent = line.length - line.trimLeft().length;
    if (indent < minIndent) break;
    final kv = _parseKV(trimmed);
    if (kv != null) {
      if (kv.$2 == '|') {
        map[kv.$1] = _parseBlockScalar(ctx);
      } else {
        map[kv.$1] = _unquote(kv.$2);
        ctx.idx++;
      }
    } else {
      break;
    }
  }
  return map;
}

/// Parse the params entries (indent 6 or deeper). Each value keeps its quotes,
/// so [_paramValue] can tell a quoted string from a bare value.
Map<String, String> _parseParams(_ParseContext ctx) {
  final params = <String, String>{};
  while (ctx.idx < ctx.lines.length) {
    final line = ctx.lines[ctx.idx];
    final trimmed = line.trim();
    if (trimmed.startsWith('#') || trimmed.isEmpty) {
      ctx.idx++;
      continue;
    }
    final indent = line.length - line.trimLeft().length;
    if (indent < 6) break;
    final kv = _parseKV(trimmed);
    if (kv == null) break;
    params[kv.$1] = kv.$2;
    ctx.idx++;
  }
  return params;
}

/// Parse a YAML block scalar (lines after `|`).
String _parseBlockScalar(_ParseContext ctx) {
  ctx.idx++; // skip the "|" line
  if (ctx.idx >= ctx.lines.length) return '';
  final blockIndent =
      ctx.lines[ctx.idx].length - ctx.lines[ctx.idx].trimLeft().length;
  final parts = <String>[];
  while (ctx.idx < ctx.lines.length) {
    final line = ctx.lines[ctx.idx];
    final trimmed = line.trim();
    if (trimmed.isEmpty) {
      parts.add('');
      ctx.idx++;
      continue;
    }
    final indent = line.length - line.trimLeft().length;
    if (indent < blockIndent) break;
    parts.add(line.substring(blockIndent));
    ctx.idx++;
  }
  // Trim trailing empty lines, then join
  while (parts.isNotEmpty && parts.last.isEmpty) {
    parts.removeLast();
  }
  return parts.join('\n');
}

// ---------------------------------------------------------------------------
// Navigation helpers
// ---------------------------------------------------------------------------

/// Skip blank lines and comment lines (starting with #).
void _skipBlankAndComments(_ParseContext ctx) {
  while (ctx.idx < ctx.lines.length) {
    final trimmed = ctx.lines[ctx.idx].trim();
    if (trimmed.isEmpty || trimmed.startsWith('#')) {
      ctx.idx++;
    } else {
      break;
    }
  }
}

/// Assert the current line matches [expected], then advance.
void _expectLine(_ParseContext ctx, String expected) {
  _skipBlankAndComments(ctx);
  if (ctx.idx >= ctx.lines.length ||
      ctx.lines[ctx.idx].trim() != expected) {
    final got = ctx.idx < ctx.lines.length
        ? ctx.lines[ctx.idx].trim()
        : '<EOF>';
    throw FormatException(
      "Expected '$expected' at line ${ctx.idx + 1}, got '$got'",
    );
  }
  ctx.idx++;
}

// =============================================================================
// File discovery
// =============================================================================

/// Recursively find all .gtest files under [dir].
List<File> _findGtestFiles(Directory dir) {
  final results = <File>[];
  if (!dir.existsSync()) return results;
  for (final entity in dir.listSync(recursive: true, followLinks: false)) {
    if (entity is File && entity.path.endsWith('.gtest')) {
      // Skip anything inside the runners directory.
      final normalized = entity.path.replaceAll('\\', '/');
      if (normalized.contains('/runners/')) continue;
      results.add(entity);
    }
  }
  results.sort((a, b) => a.path.compareTo(b.path));
  return results;
}

// =============================================================================
// Test runner core
// =============================================================================

/// Run a single test case against [db].
void _runTestCase(GrafeoDB db, _TestCase tc, String language,
    [String? setupLanguage]) {
  // Run setup queries in the file's declared language (not the variant language)
  final setupLang = setupLanguage ?? language;
  for (final setupQ in tc.setup) {
    _executeQuery(db, setupLang, setupQ);
  }

  final exp = tc.expect;

  // Determine queries
  final queries = tc.statements.isNotEmpty
      ? tc.statements
      : tc.query != null
          ? [tc.query!]
          : <String>[];
  if (queries.isEmpty) {
    fail("No query or statements in test '${tc.name}'");
  }

  // Error case
  if (exp.error != null) {
    _runErrorTest(db, language, queries, exp.error!,
        params: tc.params.isNotEmpty ? tc.params : null);
    return;
  }

  // Execute all queries, capture last result
  late QueryResult result;
  for (final q in queries) {
    result = _executeQuery(db, language, q,
        params: tc.params.isNotEmpty ? tc.params : null);
  }

  // Column assertion (checked before value assertions)
  if (exp.columns.isNotEmpty) {
    _assertColumns(result, exp.columns);
  }

  // Empty check
  if (exp.empty) {
    expect(result.rows.length, equals(0),
        reason: 'Expected empty result, got ${result.rows.length} row(s)');
    return;
  }

  // Count check
  if (exp.count != null) {
    expect(result.rows.length, equals(exp.count),
        reason:
            'Row count mismatch: got ${result.rows.length}, '
            'expected ${exp.count}');
    return;
  }

  // Hash check
  if (exp.hash != null) {
    _assertHash(result, exp.hash!);
    return;
  }

  // Rows check
  if (exp.rows.isNotEmpty) {
    if (exp.precision != null) {
      _assertRowsWithPrecision(result, exp.rows, exp.precision!);
    } else if (exp.ordered) {
      _assertRowsOrdered(result, exp.rows);
    } else {
      _assertRowsSorted(result, exp.rows);
    }
  }
}

/// Run the error test path: execute queries expecting the last to throw.
void _runErrorTest(
  GrafeoDB db,
  String language,
  List<String> queries,
  String expectedSubstring, {
  Map<String, String>? params,
}) {
  // Execute all-but-last normally
  for (var i = 0; i < queries.length - 1; i++) {
    _executeQuery(db, language, queries[i], params: params);
  }

  // Last query should fail
  try {
    _executeQuery(db, language, queries.last, params: params);
    fail("Expected error containing '$expectedSubstring' but query succeeded");
  } on GrafeoException catch (e) {
    expect(e.message, contains(expectedSubstring),
        reason:
            "Error '${e.message}' does not contain '$expectedSubstring'");
  }
}

// =============================================================================
// Main: discover files and register tests
// =============================================================================

void main() {
  // The typing of parameter values that no .gtest file can hold: a number
  // past the f64 range fails the Rust runner's build, so it is tested here.
  group('parameter values', () {
    test('a JSON integer that fits 64 bits is an int', () {
      expect(_paramValue('[3, -5000000000, 9223372036854775807]'),
          [3, -5000000000, 9223372036854775807]);
    });
    test('a JSON integer past 64 bits is a double', () {
      expect(_paramValue('[9223372036854775808]').single, isA<double>());
    });
    test('a number past the f64 range is an error', () {
      for (final text in ['[1e400]', '{"tiny": [-1e400]}', '1e999']) {
        expect(
            () => _paramValue(text),
            throwsA(isA<FormatException>().having(
                (error) => error.message, 'message', contains('out of the f64 range'))),
            reason: text);
      }
    });
  });

  final gtestFiles = _findGtestFiles(_specDir);

  if (gtestFiles.isEmpty) {
    test('no .gtest files found', () {
      // Not a failure: the spec files may not be present in all environments
      markTestSkipped('No .gtest files found under ${_specDir.path}');
    });
    return;
  }

  for (final file in gtestFiles) {
    final relPath = file.path
        .replaceAll('\\', '/')
        .replaceFirst('${_specDir.path.replaceAll('\\', '/')}/', '');

    late _GtestFile parsed;
    try {
      parsed = _parseGtestFile(file);
    } catch (e) {
      group(relPath, () {
        test('should parse without errors', () {
          fail('Parse error: $e');
        });
      });
      continue;
    }

    final meta = parsed.meta;

    group(relPath, () {
      for (final tc in parsed.tests) {
        // Handle rosetta variants
        if (tc.variants.isNotEmpty) {
          for (final entry in tc.variants.entries) {
            final lang = entry.key;
            final variantQuery = entry.value;
            test('${tc.name}_$lang', () {
              if (tc.skip != null) {
                markTestSkipped('skipped in .gtest: ${tc.skip}');
                return;
              }
              if (!_isAvailable(lang)) {
                markTestSkipped('Language "$lang" not available');
                return;
              }
              final db = GrafeoDB.memory();
              try {
                final effectiveDataset = tc.dataset ?? meta.dataset;
                if (effectiveDataset.isNotEmpty && effectiveDataset != 'empty') {
                  _loadDataset(db, effectiveDataset);
                }
                // Create a copy of the test case with the variant query
                final variantTc = _TestCase(
                  name: '${tc.name}_$lang',
                  query: variantQuery,
                  setup: tc.setup,
                  expect: tc.expect,
                  tags: tc.tags,
                  params: tc.params,
                );
                _runTestCase(db, variantTc, lang, meta.language);
              } finally {
                db.close();
              }
            });
          }
          continue;
        }

        test(tc.name, () {
          // Skip if test has a skip field
          if (tc.skip != null) {
            markTestSkipped('skipped in .gtest: ${tc.skip}');
            return;
          }

          // Check language availability (file-level and per-test)
          if (!_isAvailable(meta.language)) {
            markTestSkipped(
              'Language "${meta.language}" not available',
            );
            return;
          }
          if (tc.language != null && !_isAvailable(tc.language!)) {
            markTestSkipped(
              'Language "${tc.language}" not available',
            );
            return;
          }

          // Check requires
          for (final req in meta.requires) {
            if (!_isAvailable(req)) {
              markTestSkipped(
                'Required capability "$req" not available',
              );
              return;
            }
          }
          for (final req in tc.requires) {
            if (!_isAvailable(req)) {
              markTestSkipped(
                'Required capability "$req" not available',
              );
              return;
            }
          }

          final db = GrafeoDB.memory();
          try {
            // Load dataset (per-test override takes priority)
            final effectiveDataset = tc.dataset ?? meta.dataset;
            if (effectiveDataset.isNotEmpty && effectiveDataset != 'empty') {
              _loadDataset(db, effectiveDataset);
            }

            _runTestCase(
                db, tc, tc.language ?? meta.language, meta.language);
          } finally {
            db.close();
          }
        });
      }
    });
  }
}
