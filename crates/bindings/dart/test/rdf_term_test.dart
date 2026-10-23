import 'dart:convert';
import 'package:grafeo/grafeo.dart';
import 'package:grafeo/src/value.dart';
import 'package:test/test.dart';

const _xsd = 'http://www.w3.org/2001/XMLSchema#';

void main() {
  group('RdfTerm', () {
    final terms = [
      RdfTerm.iri('http://example.org/Kraków'),
      RdfTerm.blankNode('b0'),
      RdfTerm.literal('Apple'),
      RdfTerm.literal('Pomme', language: 'fr'),
      RdfTerm.literal('2026-01-15', datatype: '${_xsd}date'),
    ];

    test('round-trips through the \$term tag', () {
      for (final term in terms) {
        final rows = parseRows(jsonEncode([
          {'t': term.toJson()}
        ]));
        expect(rows[0]['t'], equals(term), reason: '${term.toJson()}');
        expect(encodeValue(term), equals(jsonEncode(term.toJson())));
      }
    });

    test('writes the \$term tag as the map\'s only key', () {
      expect(
          encodeValue(RdfTerm.iri('http://example.org/Amsterdam')),
          equals('{"\$term":{"type":"uri",'
              '"value":"http://example.org/Amsterdam"}}'));
      expect(
          encodeValue(RdfTerm.literal('Pomme', language: 'fr')),
          equals(
              '{"\$term":{"type":"literal","value":"Pomme","xml:lang":"fr"}}'));
      expect(
          encodeValue(RdfTerm.literal('2026-01-15', datatype: '${_xsd}date')),
          equals('{"\$term":{"type":"literal","value":"2026-01-15",'
              '"datatype":"${_xsd}date"}}'));
    });

    test('reads its literal datatype as RDF does', () {
      expect(terms[2].literalDatatype, equals(RdfTerm.xsdString));
      expect(terms[3].literalDatatype, equals(RdfTerm.rdfLangString));
      expect(terms[4].literalDatatype, equals('${_xsd}date'));
      expect(terms[0].literalDatatype, isNull);
      final simple = RdfTerm.literal('x', datatype: RdfTerm.xsdString);
      expect(simple, equals(RdfTerm.literal('x')));
      expect(simple.datatype, isNull);
    });

    test('keeps a language tag in lowercase, so tags compare without case',
        () {
      final upper = RdfTerm.literal('Pomme', language: 'FR');
      expect(upper.language, equals('fr'));
      expect(upper, equals(RdfTerm.literal('Pomme', language: 'fr')));
      expect(upper.hashCode,
          equals(RdfTerm.literal('Pomme', language: 'fr').hashCode));
      expect(RdfTerm.literal('Colour', language: 'en-GB').language,
          equals('en-gb'));
      expect(encodeValue(upper), contains('"xml:lang":"fr"'));
      final decoded = parseRows(jsonEncode([
        {
          't': {
            r'$term': {'type': 'literal', 'value': 'Pomme', 'xml:lang': 'FR'}
          }
        }
      ]));
      expect(decoded[0]['t'], equals(upper));
      expect(RdfTerm.literal('Pomme', language: 'fr'),
          isNot(equals(RdfTerm.literal('Pomme', language: 'nl'))));
    });

    test('keeps a map with SPARQL JSON keys a map', () {
      for (final map in [
        {'type': 'uri', 'value': 'http://example.org/Paris'},
        {'type': 'literal', 'value': 'Pomme', 'xml:lang': 'fr'},
        {'type': 'bnode', 'value': 'b0'},
        {'type': 'uri', 'value': 'x', 'extra': 1},
      ]) {
        final rows = parseRows(jsonEncode([
          {'m': map}
        ]));
        expect(rows[0]['m'], isA<Map>(), reason: '$map');
        expect(rows[0]['m'], equals(map));
        expect(encodeValue(map), equals(jsonEncode(map)));
      }
    });

    test('keeps a \$term tag with other keys, or an unknown type, a map', () {
      for (final map in [
        {
          r'$term': {'type': 'uri', 'value': 'x'},
          'other': 1
        },
        {
          r'$term': {'type': 'page', 'value': 'x'}
        },
        {
          r'$term': {'type': 'uri', 'value': 'x', 'xml:lang': 'en'}
        },
      ]) {
        final rows = parseRows(jsonEncode([
          {'m': map}
        ]));
        expect(rows[0]['m'], isA<Map>(), reason: '$map');
      }
    });
  });

  group('RdfTerm through the database', () {
    late GrafeoDB db;
    setUp(() => db = GrafeoDB.memory());
    tearDown(() => db.close());

    test('a map with SPARQL JSON keys comes back a map', () {
      for (final map in [
        {'type': 'uri', 'value': 'http://example.org/Paris'},
        {'type': 'literal', 'value': 'Pomme', 'xml:lang': 'fr'},
        {'type': 'bnode', 'value': 'b0'},
      ]) {
        final row = db.executeWithParams(
            r'RETURN $m AS m, $m.type AS kind', {'m': map}).rows.first;
        expect(row['m'], equals(map), reason: '$map');
        expect(row['kind'], equals(map['type']), reason: '$map');
      }
    });

    test('a term comes back the term, its tag in lowercase', () {
      final term = RdfTerm.literal('Pomme', language: 'FR');
      final back = db.executeWithParams(
          r'RETURN $t AS t', {'t': term}).rows.first['t'] as RdfTerm;
      expect(back, equals(RdfTerm.literal('Pomme', language: 'fr')));
      expect(back.language, equals('fr'));
    });

    test('a SPARQL SELECT gives every kind of term exactly', () {
      db.executeSparql('''INSERT DATA {
        <http://ex.org/apple> <http://ex.org/label> "Pomme"@FR ;
          <http://ex.org/price> "1.50"^^<${_xsd}decimal> ;
          <http://ex.org/stock> "025"^^<${_xsd}integer> ;
          <http://ex.org/note> "fresh" ;
          <http://ex.org/kind> "x"^^<http://ex.org/unknown> ;
          <http://ex.org/grower> _:farm .
      }''');
      final result = db.executeSparql(
        'SELECT ?p ?o WHERE { <http://ex.org/apple> ?p ?o }',
      );
      expect(result.kind, equals(ResultKind.rows));
      expect(result.boolean, isNull);
      final objects = {
        for (final row in result.rows)
          (row['p'] as RdfTerm).value: row['o'] as RdfTerm,
      };
      for (final row in result.rows) {
        final predicate = row['p'] as RdfTerm;
        expect(predicate, equals(RdfTerm.iri(predicate.value)));
      }
      expect(
        objects['http://ex.org/label'],
        equals(RdfTerm.literal('Pomme', language: 'fr')),
      );
      expect(
        objects['http://ex.org/label']!.literalDatatype,
        equals(RdfTerm.rdfLangString),
      );
      expect(
        objects['http://ex.org/price'],
        equals(RdfTerm.literal('1.50', datatype: '${_xsd}decimal')),
      );
      // The lexical form is kept, not the value.
      expect(
        objects['http://ex.org/stock'],
        equals(RdfTerm.literal('025', datatype: '${_xsd}integer')),
      );
      expect(objects['http://ex.org/note'], equals(RdfTerm.literal('fresh')));
      expect(
        objects['http://ex.org/note']!.literalDatatype,
        equals(RdfTerm.xsdString),
      );
      expect(
        objects['http://ex.org/kind'],
        equals(RdfTerm.literal('x', datatype: 'http://ex.org/unknown')),
      );
      final grower = objects['http://ex.org/grower']!;
      expect(grower.type, equals(RdfTermType.blankNode));
      expect(grower.value, isNot(startsWith('_:')));
    });

    test('a SPARQL ASK is one boolean and CONSTRUCT a graph', () {
      db.executeSparql(
        'INSERT DATA { <http://ex.org/apple> '
        '<http://ex.org/label> "Pomme"@fr }',
      );
      final yes = db.executeSparql(
        'ASK { ?s <http://ex.org/label> "Pomme"@FR }',
      );
      expect(yes.kind, equals(ResultKind.boolean));
      expect(yes.boolean, isTrue);
      expect(
        db.executeSparql('ASK { ?s <http://ex.org/label> "Pomme"@en }').boolean,
        isFalse,
      );
      final graph = db.executeSparql(
        'CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }',
      );
      expect(graph.kind, equals(ResultKind.graph));
      expect(graph.boolean, isNull);
      expect(graph.columns, equals(['subject', 'predicate', 'object']));
      expect(
        graph.rows.single['object'],
        equals(RdfTerm.literal('Pomme', language: 'fr')),
      );
    });

    test('a SPARQL parameter is an initial binding of a term', () {
      db.executeSparql(
        'INSERT DATA { <http://ex.org/apple> '
        '<http://ex.org/label> "Pomme"@fr . <http://ex.org/pear> '
        '<http://ex.org/label> "Poire"@fr }',
      );
      final rows = db.executeSparqlWithParams(
        'SELECT ?s WHERE { ?s <http://ex.org/label> ?label }',
        {'label': RdfTerm.literal('Poire', language: 'FR')},
      ).rows;
      expect(rows, hasLength(1));
      expect(rows.single['s'], equals(RdfTerm.iri('http://ex.org/pear')));
    });
  });

  group('ResultKind', () {
    test('reads the C library codes', () {
      expect(ResultKind.of(0), equals(ResultKind.rows));
      expect(ResultKind.of(1), equals(ResultKind.boolean));
      expect(ResultKind.of(2), equals(ResultKind.graph));
    });
  });
}
