/// Bidirectional conversion between Dart types and Grafeo's JSON wire format.
///
/// Follows the `grafeo-bindings-common` markers: `$timestamp_us` (a UTC
/// [DateTime]), `$zoned_datetime` ([GrafeoZonedDateTime]), `$date`
/// ([GrafeoDate]), `$time` ([GrafeoTime]), `$duration` ([GrafeoDuration]),
/// `$decimal` ([GrafeoDecimal]) and `$term` ([RdfTerm]).
library;

import 'dart:convert';
import 'dart:typed_data';

import 'temporal.dart';
import 'types.dart';

// =============================================================================
// Encoding (Dart -> JSON for grafeo-c parameters)
// =============================================================================

/// Encode a parameter map as a JSON string for grafeo_execute_with_params.
String encodeParams(Map<String, dynamic> params) => _json({
      for (final entry in params.entries) entry.key: _encodeValue(entry.value),
    });

/// Encode a single value for the grafeo-c JSON wire format.
String encodeValue(dynamic value) => _json(_encodeValue(value));

/// The decimal digits of a [BigInt], written as a JSON number as they are:
/// an integer past 64 bits arrives as an exact decimal up to 38 digits, and
/// one past that is an error naming the parameter.
final class _IntegerDigits {
  const _IntegerDigits(this.digits);

  final String digits;
}

/// The JSON text of an encoded value: [jsonEncode], except that a [BigInt]'s
/// digits are a number (`jsonEncode` has no way to write one).
String _json(dynamic encoded) {
  final out = StringBuffer();
  void write(dynamic value) {
    switch (value) {
      case _IntegerDigits(:final digits):
        out.write(digits);
      case List list:
        out.write('[');
        for (var i = 0; i < list.length; i++) {
          if (i > 0) out.write(',');
          write(list[i]);
        }
        out.write(']');
      case Map map:
        out.write('{');
        var first = true;
        for (final entry in map.entries) {
          if (!first) out.write(',');
          first = false;
          out.write(jsonEncode(entry.key));
          out.write(':');
          write(entry.value);
        }
        out.write('}');
      default:
        out.write(jsonEncode(value));
    }
  }

  write(encoded);
  return out.toString();
}

dynamic _encodeValue(dynamic value) {
  return switch (value) {
    null => null,
    bool b => b,
    int i => i,
    BigInt b => _IntegerDigits(b.toString()),
    // JSON has no NaN or infinities: they go as the `$float` tag
    double d when d.isNaN => {r'$float': 'NaN'},
    double d when d.isInfinite => {r'$float': d > 0 ? 'Infinity' : '-Infinity'},
    double d => d,
    String s => s,
    GrafeoDecimal d => {r'$decimal': d.text},
    RdfTerm t => t.toJson(),
    DateTime dt => {r'$timestamp_us': dt.toUtc().microsecondsSinceEpoch},
    GrafeoZonedDateTime z => {r'$zoned_datetime': z.text},
    GrafeoDate d => {r'$date': d.text},
    GrafeoTime t => {r'$time': t.text},
    GrafeoDuration d => {r'$duration': d.text},
    Duration dur => {r'$duration': GrafeoDuration.fromDuration(dur).text},
    Float32List vec => vec.toList(),
    Float64List vec => vec.toList(),
    Uint8List bytes => {r'$bytes': base64Encode(bytes)},
    List list => list.map(_encodeValue).toList(),
    Map map => {
        for (final e in map.entries) e.key.toString(): _encodeValue(e.value),
      },
    _ => value.toString(),
  };
}

// =============================================================================
// Decoding (JSON from grafeo-c results -> Dart types)
// =============================================================================

/// Parse the JSON array string returned by `grafeo_result_json` into rows.
///
/// Each row is a `Map<String, dynamic>` where temporal markers are
/// automatically converted back to Dart types.
List<Map<String, dynamic>> parseRows(String json) {
  final decoded = jsonDecode(json);
  if (decoded is! List) return [];
  return decoded.map<Map<String, dynamic>>((row) {
    if (row is! Map) return <String, dynamic>{};
    // A row is a map of columns, never a tagged value: a one-column row
    // named `$bytes` or `$date` stays a row, and each value is decoded
    return decodeObject(row);
  }).toList();
}

/// The column names of a result from `grafeo_result_columns_json`, also for
/// a result without rows.
List<String> parseColumns(String json) =>
    (jsonDecode(json) as List).cast<String>();

/// The nodes and edges of a result, from `grafeo_result_nodes_json` and
/// `grafeo_result_edges_json`: the engine says by type which values are nodes
/// and edges, so a map with the keys of a node (`{_id: 1, _labels: []}`) is a
/// map.
(List<Node>, List<Edge>) parseEntities(String nodesJson, String edgesJson) {
  final nodes = [
    for (final node in jsonDecode(nodesJson) as List)
      Node(
        node['id'] as int,
        (node['labels'] as List).whereType<String>().toList(),
        decodeObject(node['properties']),
      ),
  ];
  final edges = [
    for (final edge in jsonDecode(edgesJson) as List)
      Edge(
        edge['id'] as int,
        edge['type'] as String,
        edge['source_id'] as int,
        edge['target_id'] as int,
        decodeObject(edge['properties']),
      ),
  ];
  return (nodes, edges);
}

/// Parse a JSON object string into a map with temporal decoding.
Map<String, dynamic> parseObject(String json) => decodeObject(jsonDecode(json));

/// Decode an already parsed JSON object into a map with temporal decoding.
Map<String, dynamic> decodeObject(Object? decoded) {
  if (decoded is! Map) return {};
  return {
    for (final entry in decoded.entries)
      entry.key.toString(): _decodeValue(entry.value),
  };
}

/// Parse a JSON array of strings.
List<String> parseStringArray(String json) {
  final decoded = jsonDecode(json);
  if (decoded is! List) return [];
  return decoded.whereType<String>().toList();
}

/// Decode one parsed JSON value with temporal decoding.
dynamic decodeValue(Object? decoded) => _decodeValue(decoded);

dynamic _decodeValue(dynamic value) {
  return switch (value) {
    null => null,
    bool b => b,
    int i => i,
    double d => d,
    String s => s,
    Map m => _decodeMap(m),
    List l => l.map(_decodeValue).toList(),
    _ => value,
  };
}

dynamic _decodeMap(Map m) {
  // An RDF term: the `$term` tag as the map's only key
  final term = RdfTerm.fromJson(m);
  if (term != null) return term;
  // Bytes: the `$bytes` tag, standard base64, as the map's only key
  if (m.length == 1 && m[r'$bytes'] is String) {
    // Canonical standard base64 only, as the engine reads it: a text it kept
    // as a map (the URL-safe alphabet, unused bits set) stays a map
    final text = m[r'$bytes'] as String;
    try {
      final bytes = base64Decode(text);
      if (base64Encode(bytes) == text) return bytes;
    } on FormatException {
      // Not base64: the map stays a map
    }
  }
  // A float JSON cannot write: the `$float` tag as the map's only key
  if (m.length == 1 && m[r'$float'] is String) {
    switch (m[r'$float'] as String) {
      case 'NaN':
        return double.nan;
      case 'Infinity':
        return double.infinity;
      case '-Infinity':
        return double.negativeInfinity;
    }
  }
  // An exact decimal: the `$decimal` tag as the map's only key
  if (m.length == 1 && m[r'$decimal'] is String) {
    return GrafeoDecimal(m[r'$decimal'] as String);
  }
  // Temporal tags from grafeo-bindings-common, each as the map's only key and
  // with a value of its type: a map with another key, a value of another
  // type, or a text that is no value of the type, stays a map
  if (m.length == 1) {
    final us = m[r'$timestamp_us'];
    if (us is int) {
      return DateTime.fromMicrosecondsSinceEpoch(us, isUtc: true);
    }
    final MapEntry(:key, :value) = m.entries.single;
    if (value is String) {
      final temporal = switch (key) {
        r'$zoned_datetime' => GrafeoZonedDateTime.tryParse(value),
        r'$date' => GrafeoDate.tryParse(value),
        r'$time' => GrafeoTime.tryParse(value),
        // A Dart Duration cannot hold months, and P1D is not PT24H
        r'$duration' => GrafeoDuration.tryParse(value),
        _ => null,
      };
      if (temporal != null) return temporal;
    }
  }

  // Regular map
  return <String, dynamic>{
    for (final entry in m.entries)
      entry.key.toString(): _decodeValue(entry.value),
  };
}
