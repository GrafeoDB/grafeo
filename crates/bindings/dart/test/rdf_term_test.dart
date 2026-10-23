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
  });

  group('ResultKind', () {
    test('reads the C library codes', () {
      expect(ResultKind.of(0), equals(ResultKind.rows));
      expect(ResultKind.of(1), equals(ResultKind.boolean));
      expect(ResultKind.of(2), equals(ResultKind.graph));
    });
  });
}
