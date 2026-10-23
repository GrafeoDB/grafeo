/// Bidirectional conversion between Dart types and Grafeo's JSON wire format.
///
/// Follows the `grafeo-bindings-common` markers: `$timestamp_us`, `$date`,
/// `$time`, `$duration`, `$decimal` ([GrafeoDecimal]) and `$term` ([RdfTerm]).
library;

import 'dart:convert';
import 'dart:typed_data';

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
    double d => d,
    String s => s,
    GrafeoDecimal d => {r'$decimal': d.text},
    RdfTerm t => t.toJson(),
    DateTime dt => {r'$timestamp_us': dt.toUtc().microsecondsSinceEpoch},
    Duration dur => {r'$duration': _formatIsoDuration(dur)},
    Float32List vec => vec.toList(),
    Float64List vec => vec.toList(),
    Uint8List bytes => base64Encode(bytes),
    List list => list.map(_encodeValue).toList(),
    Map map => {
        for (final e in map.entries) e.key.toString(): _encodeValue(e.value),
      },
    _ => value.toString(),
  };
}

String _formatIsoDuration(Duration d) {
  final hours = d.inHours;
  final minutes = d.inMinutes.remainder(60);
  final seconds = d.inSeconds.remainder(60);
  final buf = StringBuffer('PT');
  if (hours > 0) buf.write('${hours}H');
  if (minutes > 0) buf.write('${minutes}M');
  if (seconds > 0 || (hours == 0 && minutes == 0)) buf.write('${seconds}S');
  return buf.toString();
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
    // First, check if the row itself is a temporal marker map.
    // _decodeMap returns a scalar (Duration, DateTime, etc.) for markers.
    final asDecoded = _decodeMap(row);
    if (asDecoded is! Map) {
      // Wrap the decoded scalar so the row stays Map<String, dynamic>.
      return <String, dynamic>{row.keys.first.toString(): asDecoded};
    }
    return asDecoded.map<String, dynamic>(
      (key, value) => MapEntry(key.toString(), value),
    );
  }).toList();
}

/// Extract column names from the first row.
List<String> extractColumns(List<Map<String, dynamic>> rows) {
  if (rows.isEmpty) return [];
  return rows.first.keys.toList();
}

/// Extract Node and Edge entities from query result rows.
///
/// Mirrors `grafeo-bindings-common::entity::extract_entities`.
(List<Node>, List<Edge>) extractEntities(List<Map<String, dynamic>> rows) {
  final nodeIds = <int>{};
  final edgeIds = <int>{};
  final nodes = <Node>[];
  final edges = <Edge>[];

  for (final row in rows) {
    for (final value in row.values) {
      if (value is! Map<String, dynamic>) continue;
      final id = value['_id'];
      if (id is! int) continue;

      if (value.containsKey('_labels')) {
        if (!nodeIds.add(id)) continue;
        final labels =
            (value['_labels'] as List?)?.whereType<String>().toList() ?? [];
        nodes.add(Node(id, labels, _extractProperties(value, _nodeKeys)));
      } else if (value.containsKey('_type')) {
        if (!edgeIds.add(id)) continue;
        final edgeType = value['_type'] as String? ?? '';
        final sourceId = value['_source'] as int? ?? 0;
        final targetId = value['_target'] as int? ?? 0;
        edges.add(
          Edge(
            id,
            edgeType,
            sourceId,
            targetId,
            _extractProperties(value, _edgeKeys),
          ),
        );
      }
    }
  }

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
  // An exact decimal: the `$decimal` tag as the map's only key
  if (m.length == 1 && m[r'$decimal'] is String) {
    return GrafeoDecimal(m[r'$decimal'] as String);
  }
  // Check for temporal markers from grafeo-bindings-common
  if (m.containsKey(r'$timestamp_us')) {
    final us = m[r'$timestamp_us'];
    if (us is int) {
      return DateTime.fromMicrosecondsSinceEpoch(us, isUtc: true);
    }
  }
  if (m.containsKey(r'$date')) {
    return m[r'$date'] as String? ?? '';
  }
  if (m.containsKey(r'$time')) {
    return m[r'$time'] as String? ?? '';
  }
  if (m.containsKey(r'$duration')) {
    // Return the ISO string directly. Dart's Duration type cannot represent
    // calendar components (years, months, days), only time-based durations.
    return m[r'$duration'] as String? ?? 'PT0S';
  }
  if (m.containsKey(r'$zoned_datetime')) {
    return m[r'$zoned_datetime'] as String? ?? '';
  }

  // Regular map
  return <String, dynamic>{
    for (final entry in m.entries)
      entry.key.toString(): _decodeValue(entry.value),
  };
}

/// The structural keys of a node map: its id and labels.
const _nodeKeys = {'_id', '_labels'};

/// The structural keys of an edge map: its id, type and ends.
const _edgeKeys = {'_id', '_type', '_source', '_target'};

/// The user properties of a node or edge map: every key but its [structural]
/// ones, also a property whose name starts with `_` (#254).
Map<String, dynamic> _extractProperties(
  Map<String, dynamic> map,
  Set<String> structural,
) {
  return {
    for (final entry in map.entries)
      if (!structural.contains(entry.key)) entry.key: entry.value,
  };
}
