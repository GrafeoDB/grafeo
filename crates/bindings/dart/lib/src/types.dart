/// Data types for the Grafeo Dart binding.
library;

import 'dart:ffi';

/// Opaque pointer to a Grafeo database.
typedef GrafeoDatabasePtr = Pointer<Void>;

/// Opaque pointer to a Grafeo query result.
typedef GrafeoResultPtr = Pointer<Void>;

/// Opaque pointer to a Grafeo transaction.
typedef GrafeoTransactionPtr = Pointer<Void>;

/// Transaction isolation levels. [serializable] is not available yet: beginning
/// a transaction at that level fails with an error naming snapshot isolation,
/// the strongest level, until queries record their reads.
enum IsolationLevel {
  readCommitted(0),
  snapshotIsolation(1),
  serializable(2);

  final int code;
  const IsolationLevel(this.code);
}

/// A graph node with an ID, labels, and properties.
class Node {
  final int id;
  final List<String> labels;
  final Map<String, dynamic> properties;

  const Node(this.id, this.labels, this.properties);

  @override
  String toString() => 'Node($id, $labels, $properties)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) || (other is Node && other.id == id);

  @override
  int get hashCode => id.hashCode;
}

/// A graph edge with an ID, type, source/target, and properties.
class Edge {
  /// Unique numeric identifier for this edge.
  final int id;

  /// The relationship type label (e.g. `"KNOWS"`, `"WORKS_AT"`).
  final String type;

  /// The ID of the source (start) node.
  final int sourceId;

  /// The ID of the target (end) node.
  final int targetId;

  /// Key/value properties stored on this edge.
  final Map<String, dynamic> properties;

  /// Creates an edge with the given [id], [type], [sourceId], [targetId], and [properties].
  const Edge(this.id, this.type, this.sourceId, this.targetId, this.properties);

  @override
  String toString() => 'Edge($id, $type, $sourceId->$targetId, $properties)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) || (other is Edge && other.id == id);

  @override
  int get hashCode => id.hashCode;
}

/// The result of a query execution.
class QueryResult {
  final List<String> columns;
  final List<Map<String, dynamic>> rows;
  final List<Node> nodes;
  final List<Edge> edges;
  final double executionTimeMs;
  final int rowsScanned;

  /// What a write statement changed (all zero for a read).
  final WriteCounters counters;

  /// What the result holds: rows, one boolean (SPARQL ASK) or an RDF graph
  /// (SPARQL CONSTRUCT and DESCRIBE).
  final ResultKind kind;

  const QueryResult({
    required this.columns,
    required this.rows,
    required this.nodes,
    required this.edges,
    required this.executionTimeMs,
    required this.rowsScanned,
    this.counters = WriteCounters.none,
    this.kind = ResultKind.rows,
  });

  /// The answer of a SPARQL ASK; `null` for every other result.
  bool? get boolean => kind == ResultKind.boolean && rows.length == 1
      ? rows.first['boolean'] as bool?
      : null;

  @override
  String toString() =>
      'QueryResult(${rows.length} rows, ${nodes.length} nodes, '
      '${edges.length} edges, ${executionTimeMs.toStringAsFixed(2)}ms)';
}

/// What a write statement changed.
class WriteCounters {
  final int nodesCreated;
  final int nodesDeleted;
  final int edgesCreated;
  final int edgesDeleted;
  final int propertiesSet;
  final int labelsAdded;
  final int labelsRemoved;

  const WriteCounters({
    this.nodesCreated = 0,
    this.nodesDeleted = 0,
    this.edgesCreated = 0,
    this.edgesDeleted = 0,
    this.propertiesSet = 0,
    this.labelsAdded = 0,
    this.labelsRemoved = 0,
  });

  /// No changes.
  static const none = WriteCounters();

  @override
  bool operator ==(Object other) =>
      other is WriteCounters &&
      other.nodesCreated == nodesCreated &&
      other.nodesDeleted == nodesDeleted &&
      other.edgesCreated == edgesCreated &&
      other.edgesDeleted == edgesDeleted &&
      other.propertiesSet == propertiesSet &&
      other.labelsAdded == labelsAdded &&
      other.labelsRemoved == labelsRemoved;

  @override
  int get hashCode => Object.hash(nodesCreated, nodesDeleted, edgesCreated,
      edgesDeleted, propertiesSet, labelsAdded, labelsRemoved);

  @override
  String toString() => 'WriteCounters(nodes +$nodesCreated -$nodesDeleted, '
      'edges +$edgesCreated -$edgesDeleted, properties $propertiesSet, '
      'labels +$labelsAdded -$labelsRemoved)';
}

/// A single vector search result with a node ID and distance score.
class VectorResult {
  final int nodeId;
  final double distance;

  const VectorResult(this.nodeId, this.distance);

  @override
  String toString() => 'VectorResult(nodeId: $nodeId, distance: $distance)';
}

/// An exact decimal from Grafeo: up to 38 digits, kept as its text with the
/// scale it was written with (`1.50`).
///
/// Equal values are equal whatever their scale (`1.50` equals `1.5`). As a
/// query parameter it is passed exactly; Dart has no decimal type of its own,
/// so [toDouble] is the nearest double.
final class GrafeoDecimal {
  /// A decimal from its text: `[+-]digits[.digits]`.
  ///
  /// Throws a [FormatException] when [text] is not a decimal number.
  GrafeoDecimal(this.text) {
    if (!_pattern.hasMatch(text)) {
      throw FormatException('not a decimal number', text);
    }
  }

  /// The text with its scale, as Grafeo writes it.
  final String text;

  static final _pattern = RegExp(r'^[+-]?(\d+(\.\d*)?|\.\d+)$');

  /// The nearest double.
  double toDouble() => double.parse(text);

  @override
  String toString() => text;

  @override
  bool operator ==(Object other) =>
      other is GrafeoDecimal && _normalized(other.text) == _normalized(text);

  @override
  int get hashCode => _normalized(text).hashCode;

  /// The value without leading or trailing zeros: one text per value.
  static String _normalized(String text) {
    final negative = text.startsWith('-');
    final body =
        text.startsWith('-') || text.startsWith('+') ? text.substring(1) : text;
    final point = body.indexOf('.');
    final integer = (point < 0 ? body : body.substring(0, point))
        .replaceFirst(RegExp('^0+'), '');
    final fraction = (point < 0 ? '' : body.substring(point + 1))
        .replaceFirst(RegExp(r'0+$'), '');
    if (integer.isEmpty && fraction.isEmpty) return '0';
    return '${negative ? '-' : ''}$integer.$fraction';
  }
}

/// What a [QueryResult] holds.
enum ResultKind {
  /// Rows: SPARQL SELECT and updates, and every other language.
  rows,

  /// One boolean: SPARQL ASK, one row with a `boolean` column.
  boolean,

  /// An RDF graph: SPARQL CONSTRUCT and DESCRIBE, rows of `subject`,
  /// `predicate` and `object` terms.
  graph;

  /// The kind the C library reports (`grafeo_result_kind`).
  static ResultKind of(int code) => switch (code) {
        1 => ResultKind.boolean,
        2 => ResultKind.graph,
        _ => ResultKind.rows,
      };
}

/// The kind of an RDF term.
enum RdfTermType {
  /// An IRI.
  iri,

  /// A literal: its value with a language tag or a datatype.
  literal,

  /// A blank node: its value is the label, without `_:`.
  blankNode,
}

/// An RDF term as a SPARQL result holds it: an IRI, a blank node, or a
/// literal with its language tag or datatype. As a query parameter it is
/// that exact term.
///
/// Language tags compare without case, so a term keeps its tag in lowercase
/// (`'Pomme'@FR` is `'Pomme'@fr`), and a literal of datatype `xsd:string` is
/// the simple literal.
final class RdfTerm {
  /// A term of [type] with [value]; a literal may have a [language] tag or a
  /// [datatype] (with a tag, the datatype is `rdf:langString`).
  RdfTerm(this.type, this.value, {String? language, String? datatype})
      : language = type == RdfTermType.literal && language != null && language.isNotEmpty
            ? language.toLowerCase()
            : null,
        datatype = type != RdfTermType.literal ||
                (language != null && language.isNotEmpty) ||
                datatype == xsdString
            ? null
            : datatype;

  /// An IRI.
  RdfTerm.iri(String value) : this(RdfTermType.iri, value);

  /// A blank node with its label.
  RdfTerm.blankNode(String label) : this(RdfTermType.blankNode, label);

  /// A literal, simple, tagged or typed.
  RdfTerm.literal(String value, {String? language, String? datatype})
      : this(RdfTermType.literal, value,
            language: language, datatype: datatype);

  /// The datatype of a simple literal.
  static const xsdString = 'http://www.w3.org/2001/XMLSchema#string';

  /// The datatype of a language-tagged literal.
  static const rdfLangString =
      'http://www.w3.org/1999/02/22-rdf-syntax-ns#langString';

  /// The kind of term.
  final RdfTermType type;

  /// The IRI, the blank node's label or the literal's lexical form.
  final String value;

  /// The tag of a language-tagged literal in lowercase, `null` otherwise.
  final String? language;

  /// The datatype IRI of a typed literal other than `xsd:string`, `null`
  /// otherwise.
  final String? datatype;

  /// A literal's datatype as RDF reads it: `rdf:langString` for a tagged
  /// literal, `xsd:string` for a simple one; `null` for an IRI or a blank
  /// node.
  String? get literalDatatype => switch (type) {
        RdfTermType.literal when language != null => rdfLangString,
        RdfTermType.literal => datatype ?? xsdString,
        _ => null,
      };

  /// The term in the JSON form the C library reads and writes: the `$term`
  /// tag as the map's only key, holding `type` (`uri`, `bnode` or
  /// `literal`) and `value`, and for a literal `xml:lang` or `datatype`.
  Map<String, Map<String, String>> toJson() => {
        r'$term': {
          'type': switch (type) {
            RdfTermType.iri => 'uri',
            RdfTermType.blankNode => 'bnode',
            RdfTermType.literal => 'literal',
          },
          'value': value,
          if (language != null) 'xml:lang': language!,
          if (datatype != null) 'datatype': datatype!,
        },
      };

  /// The term a map holds in the form [toJson] writes; `null` for any other
  /// map, also one with `type` and `value` keys but no `$term` tag.
  static RdfTerm? fromJson(Map m) {
    final term = m[r'$term'];
    if (m.length != 1 || term is! Map) return null;
    final type = term['type'];
    final value = term['value'];
    if (type is! String || value is! String) return null;
    final language = term['xml:lang'];
    final datatype = term['datatype'];
    return switch (type) {
      'uri' when term.length == 2 => RdfTerm.iri(value),
      'bnode' when term.length == 2 => RdfTerm.blankNode(value),
      'literal' when language is String && datatype == null && term.length == 3 =>
        RdfTerm.literal(value, language: language),
      'literal' when datatype is String && language == null && term.length == 3 =>
        RdfTerm.literal(value, datatype: datatype),
      'literal' when term.length == 2 => RdfTerm.literal(value),
      _ => null,
    };
  }

  /// The term's text: the IRI, the blank node's label or the literal's
  /// lexical form.
  @override
  String toString() => value;

  @override
  bool operator ==(Object other) =>
      other is RdfTerm &&
      other.type == type &&
      other.value == value &&
      other.language == language &&
      other.literalDatatype == literalDatatype;

  @override
  int get hashCode => Object.hash(type, value, language, literalDatatype);
}
