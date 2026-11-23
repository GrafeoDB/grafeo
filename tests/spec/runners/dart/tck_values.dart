// The openCypher TCK's value notation, for `values:`: parse a cell and compare it to a result
// value, type-exact (an int never matches a double). Mirrors the Rust runner's src/tck.rs; every
// runner checks itself against tests/spec/runners/values_cases.json.

import 'dart:convert';

import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

enum TckKind { nil, boolean, integer, float, string, list, map, node, relationship, path }

/// One cell of the TCK's value notation.
class TckCell {
  final TckKind kind;
  final bool boolean;
  final int integer;
  final double float;

  /// A string, or a relationship's type.
  final String text;
  final List<TckCell> items;

  /// A map's entries, or an entity's properties.
  final Map<String, TckCell> entries;
  final Set<String> labels;
  final List<TckCell> nodes;
  final List<TckCell> relationships;

  /// Whether each path relationship starts at the node on its left.
  final List<bool> forward;

  TckCell(this.kind,
      {this.boolean = false,
      this.integer = 0,
      this.float = 0,
      this.text = '',
      List<TckCell>? items,
      Map<String, TckCell>? entries,
      Set<String>? labels,
      List<TckCell>? nodes,
      List<TckCell>? relationships,
      List<bool>? forward})
      : items = items ?? [],
        entries = entries ?? {},
        labels = labels ?? {},
        nodes = nodes ?? [],
        relationships = relationships ?? [],
        forward = forward ?? [];
}

class _Parser {
  final String text;
  int at = 0;
  _Parser(this.text);

  FormatException failure(String what) => FormatException("$what at $at in '$text'");

  void skip() {
    while (at < text.length && text[at].trim().isEmpty) {
      at++;
    }
  }

  bool peek(String token) {
    skip();
    return text.startsWith(token, at);
  }

  void take(String token) {
    if (!peek(token)) throw failure("expected '$token'");
    at += token.length;
  }

  bool word(String word) {
    if (!text.startsWith(word, at)) return false;
    final after = at + word.length;
    if (RegExp(r'[\p{L}\p{N}]', unicode: true).matchAsPrefix(text, after) != null) {
      return false;
    }
    at = after;
    return true;
  }

  String name() {
    skip();
    if (peek('`')) {
      final end = text.indexOf('`', at + 1);
      if (end < 0) throw failure('unterminated name');
      final quoted = text.substring(at + 1, end);
      at = end + 1;
      return quoted;
    }
    // By code point: a letter outside the BMP is two code units
    final found = RegExp(r'[\p{L}\p{N}_]*', unicode: true).matchAsPrefix(text, at)![0]!;
    if (found.isEmpty) throw failure('expected a name');
    at += found.length;
    return found;
  }

  TckCell value() {
    skip();
    if (word('null')) return TckCell(TckKind.nil);
    if (word('true')) return TckCell(TckKind.boolean, boolean: true);
    if (word('false')) return TckCell(TckKind.boolean);
    for (final (w, v) in [
      ('NaN', double.nan),
      ('-Inf', double.negativeInfinity),
      ('Inf', double.infinity)
    ]) {
      if (text.startsWith(w, at)) {
        at += w.length;
        return TckCell(TckKind.float, float: v);
      }
    }
    if (at >= text.length) throw failure('expected a value');
    final c = text[at];
    if (c == "'") return TckCell(TckKind.string, text: string());
    if (c == '{') return TckCell(TckKind.map, entries: mapping());
    if (c == '(') return node();
    if (c == '<') return path();
    if (text.startsWith('[:', at)) return relationship();
    if (c == '[') {
      take('[');
      return TckCell(TckKind.list, items: itemsUntil(']', value));
    }
    return number();
  }

  String string() {
    take("'");
    final out = StringBuffer();
    const escapes = {
      "'": "'",
      '"': '"',
      r'\': r'\',
      'n': '\n',
      't': '\t',
      'r': '\r',
      'b': '\b',
      'f': '\f'
    };
    while (at < text.length) {
      final c = text[at];
      if (c == "'") {
        at++;
        return out.toString();
      }
      if (c == r'\' && at + 1 < text.length) {
        final code = text[at + 1];
        if (code == 'u' && at + 6 <= text.length) {
          out.writeCharCode(int.parse(text.substring(at + 2, at + 6), radix: 16));
          at += 6;
          continue;
        }
        out.write(escapes[code] ?? '\\$code');
        at += 2;
        continue;
      }
      out.write(c);
      at++;
    }
    throw failure('unterminated string');
  }

  TckCell number() {
    final match = RegExp(r'[+\-0-9.eE]+').matchAsPrefix(text, at);
    if (match == null) throw failure('expected a value');
    final number = match.group(0)!;
    at += number.length;
    if (number.contains(RegExp('[.eE]'))) {
      final f = double.tryParse(number);
      if (f == null) throw failure('bad float');
      return TckCell(TckKind.float, float: f);
    }
    final i = int.tryParse(number);
    if (i == null) throw failure('bad integer');
    return TckCell(TckKind.integer, integer: i);
  }

  List<T> itemsUntil<T>(String close, T Function() item) {
    final out = <T>[];
    if (peek(close)) {
      take(close);
      return out;
    }
    while (true) {
      out.add(item());
      if (peek(',')) {
        take(',');
        continue;
      }
      take(close);
      return out;
    }
  }

  Map<String, TckCell> mapping() {
    take('{');
    final entries = itemsUntil('}', () {
      final key = name();
      take(':');
      return MapEntry(key, value());
    });
    return Map.fromEntries(entries);
  }

  Map<String, TckCell> properties() => peek('{') ? mapping() : {};

  TckCell node() {
    take('(');
    final labels = <String>{};
    while (peek(':')) {
      take(':');
      labels.add(name());
    }
    final props = properties();
    take(')');
    return TckCell(TckKind.node, labels: labels, entries: props);
  }

  TckCell relationship() {
    take('[');
    take(':');
    final type = name();
    final props = properties();
    take(']');
    return TckCell(TckKind.relationship, text: type, entries: props);
  }

  TckCell path() {
    take('<');
    final cell = TckCell(TckKind.path);
    cell.nodes.add(node());
    while (!peek('>')) {
      final forward = !peek('<-');
      take(forward ? '-' : '<-');
      cell.relationships.add(relationship());
      take(forward ? '->' : '-');
      cell.forward.add(forward);
      cell.nodes.add(node());
    }
    take('>');
    return cell;
  }
}

/// Parses one cell of the TCK's value notation.
TckCell parseTck(String text) {
  final parser = _Parser(text);
  final cell = parser.value();
  parser.skip();
  if (parser.at != text.length) throw parser.failure('unexpected text');
  return cell;
}

/// A node or relationship as a result gives it.
class TckEntity {
  final bool isNode;
  final int id;
  final Set<String> labels;
  final String type;
  final int source;
  final int target;
  final Map<String, dynamic> properties;
  TckEntity(
      this.isNode, this.id, this.labels, this.type, this.source, this.target, this.properties);
}

/// The one place that recognises entities, by the shape the binding gives them: a map with an
/// int `_id` and a `_labels` list is a node; one with `_id`, a string `_type`, `_source` and
/// `_target` a relationship. A map the query builds with those keys comes back as an entity too.
TckEntity? entityOf(dynamic value) {
  if (value is! Map || value['_id'] is! int) return null;
  final id = value['_id'] as int;
  Map<String, dynamic> props(List<String> reserved) => {
        for (final e in value.entries)
          if (!reserved.contains(e.key)) e.key as String: e.value,
      };
  final labels = value['_labels'];
  if (labels is List) {
    return TckEntity(
        true, id, labels.whereType<String>().toSet(), '', 0, 0, props(['_id', '_labels']));
  }
  final type = value['_type'], source = value['_source'], target = value['_target'];
  if (type is String && source is int && target is int) {
    return TckEntity(
        false, id, {}, type, source, target, props(['_id', '_type', '_source', '_target']));
  }
  return null;
}

/// The nodes and relationships of a path, which the binding gives as `{"$path": {...}}`.
(List<dynamic>, List<dynamic>)? pathOf(dynamic value) {
  if (value is! Map || value.length != 1) return null;
  final inner = value[r'$path'];
  if (inner is! Map || inner['nodes'] is! List || inner['edges'] is! List) return null;
  return (inner['nodes'] as List, inner['edges'] as List);
}

bool _propertiesMatch(Map<String, TckCell> expected, Map actual, bool unordered) =>
    expected.length == actual.length &&
    expected.entries
        .every((e) => actual.containsKey(e.key) && matchTck(e.value, actual[e.key], unordered));

bool _bagMatches(List<TckCell> expected, List actual, bool unordered) {
  if (expected.length != actual.length) return false;
  final left = [...actual];
  for (final cell in expected) {
    final hit = left.indexWhere((v) => matchTck(cell, v, unordered));
    if (hit < 0) return false;
    left.removeAt(hit);
  }
  return true;
}

/// Whether [value] is the value [cell] writes, type-exact.
bool matchTck(TckCell cell, dynamic value, bool unordered) {
  switch (cell.kind) {
    case TckKind.nil:
      return value == null;
    case TckKind.boolean:
      return value is bool && value == cell.boolean;
    case TckKind.integer:
      return value is int && value == cell.integer;
    case TckKind.float:
      return value is double && (value == cell.float || (value.isNaN && cell.float.isNaN));
    case TckKind.string:
      return value is String && value == cell.text;
    case TckKind.list:
      if (value is! List) return false;
      if (unordered) return _bagMatches(cell.items, value, unordered);
      if (value.length != cell.items.length) return false;
      for (var i = 0; i < value.length; i++) {
        if (!matchTck(cell.items[i], value[i], unordered)) return false;
      }
      return true;
    case TckKind.map:
      return value is Map &&
          entityOf(value) == null &&
          pathOf(value) == null &&
          _propertiesMatch(cell.entries, value, unordered);
    case TckKind.node:
      final e = entityOf(value);
      return e != null &&
          e.isNode &&
          e.labels.length == cell.labels.length &&
          e.labels.containsAll(cell.labels) &&
          _propertiesMatch(cell.entries, e.properties, unordered);
    case TckKind.relationship:
      final e = entityOf(value);
      return e != null &&
          !e.isNode &&
          e.type == cell.text &&
          _propertiesMatch(cell.entries, e.properties, unordered);
    case TckKind.path:
      final path = pathOf(value);
      if (path == null) return false;
      final (nodes, edges) = path;
      if (nodes.length != cell.nodes.length || edges.length != cell.relationships.length)
        return false;
      for (var i = 0; i < edges.length; i++) {
        final edge = entityOf(edges[i]);
        var start = entityOf(nodes[i]), end = entityOf(nodes[i + 1]);
        if (!cell.forward[i]) (start, end) = (end, start);
        if (edge == null ||
            start == null ||
            end == null ||
            edge.isNode ||
            edge.source != start.id ||
            edge.target != end.id ||
            !matchTck(cell.relationships[i], edges[i], unordered)) {
          return false;
        }
      }
      for (var i = 0; i < nodes.length; i++) {
        if (!matchTck(cell.nodes[i], nodes[i], unordered)) return false;
      }
      return true;
  }
}

/// The rows of a `values:` table, each a list of cell texts (`\|` is a pipe in a cell).
List<List<String>> tableRows(String table) {
  final rows = <List<String>>[];
  for (final raw in table.split('\n')) {
    final line = raw.trim();
    if (line.isEmpty) continue;
    final inner = line.startsWith('|') && line.endsWith('|') && line.length >= 2
        ? line.substring(1, line.length - 1)
        : line;
    final cells = <String>[];
    final cell = StringBuffer();
    for (var i = 0; i < inner.length; i++) {
      if (inner[i] == r'\' && i + 1 < inner.length && inner[i + 1] == '|') {
        cell.write('|');
        i++;
      } else if (inner[i] == '|') {
        cells.add(cell.toString().trim());
        cell.clear();
      } else {
        cell.write(inner[i]);
      }
    }
    cells.add(cell.toString().trim());
    rows.add(cells);
  }
  return rows;
}

bool _isTemporal(dynamic value) =>
    value is GrafeoDate ||
    value is GrafeoTime ||
    value is GrafeoDuration ||
    value is GrafeoZonedDateTime ||
    value is DateTime;

/// [value] with each temporal in it written as its Cypher toString(), the engine's own text: how
/// a `values_dialect: tck` file compares temporals.
dynamic temporalsAsText(GrafeoDB db, dynamic value) {
  if (_isTemporal(value)) {
    final result = db.executeLanguage('cypher', r'RETURN toString($v) AS s', params: {'v': value});
    return result.rows.first['s'];
  }
  if (value is List) return [for (final v in value) temporalsAsText(db, v)];
  if (value is Map) return {for (final e in value.entries) e.key: temporalsAsText(db, e.value)};
  return value;
}

/// Asserts the result holds the rows of [table], a `values:` table whose first row names the
/// columns: in order when [ordered], otherwise as a bag.
void assertValues(
    GrafeoDB db, QueryResult result, String table, bool ordered, bool unordered, bool tckDialect) {
  final rows = tableRows(table);
  final header = rows.first.where((c) => c.isNotEmpty).toList();
  final columns = result.columns.toList();
  if (jsonEncode(columns) != jsonEncode(header)) fail('Columns differ: $columns, expected $header');
  final expected = [
    for (final row in rows.skip(1)) [for (final c in row) parseTck(c)]
  ];
  final actual = [
    for (final row in result.rows)
      [for (final c in columns) tckDialect ? temporalsAsText(db, row[c]) : row[c]],
  ];
  bool rowMatches(List<TckCell> cells, List values) {
    if (cells.length != values.length) return false;
    for (var i = 0; i < cells.length; i++) {
      if (!matchTck(cells[i], values[i], unordered)) return false;
    }
    return true;
  }

  var holds = expected.length == actual.length;
  if (holds && ordered) {
    for (var i = 0; i < expected.length && holds; i++) {
      holds = rowMatches(expected[i], actual[i]);
    }
  } else if (holds) {
    final left = [...actual];
    for (final cells in expected) {
      final hit = left.indexWhere((values) => rowMatches(cells, values));
      if (hit < 0) {
        holds = false;
        break;
      }
      left.removeAt(hit);
    }
  }
  if (!holds)
    fail('Rows differ${ordered ? ' (in order)' : ''}:\nactual:   $actual\nexpected: $table');
}

/// What a graph holds, as the openCypher TCK observes side effects.
Map<String, Set<String>> observeGraph(GrafeoDB db) {
  final seen = <String, Set<String>>{
    'nodes': {},
    'relationships': {},
    'labels': {},
    'properties': {}
  };
  for (final query in ['MATCH (n) RETURN n', 'MATCH ()-[r]->() RETURN r']) {
    for (final row in db.execute(query).rows) {
      for (final value in row.values) {
        final e = entityOf(value);
        if (e == null) continue;
        final kind = e.isNode ? 'nodes' : 'relationships';
        seen[kind]!.add('${e.id}');
        seen['labels']!.addAll(e.labels);
        for (final p in e.properties.entries) {
          seen['properties']!.add('$kind|${e.id}|${p.key}|${p.value.runtimeType}|${p.value}');
        }
      }
    }
  }
  return seen;
}

/// Asserts the side effects from [before] to [after] are [expected] (a key not named is 0).
void assertSideEffects(
    Map<String, Set<String>> before, Map<String, Set<String>> after, Map<String, int> expected) {
  final differ = <String>[];
  for (final key in ['nodes', 'relationships', 'labels', 'properties']) {
    final added = after[key]!.where((x) => !before[key]!.contains(x)).length;
    final removed = before[key]!.where((x) => !after[key]!.contains(x)).length;
    for (final (name, count) in [('+$key', added), ('-$key', removed)]) {
      final want = expected[name] ?? 0;
      if (count != want) differ.add('$name: $count, expected $want');
    }
  }
  if (differ.isNotEmpty) fail('Side effects differ: ${differ.join('; ')}');
}
