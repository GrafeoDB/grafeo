import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

void main() {
  late GrafeoDB db;

  setUp(() {
    db = GrafeoDB.memory();
  });

  tearDown(() {
    db.close();
  });

  test('batch creates nodes and edges', () {
    final ids = db.batchCreateNodesWithProps([
      'Person',
      'Employee'
    ], [
      {'name': 'Alix'},
      {'name': 'Gus'},
    ]);
    expect(ids, hasLength(2));
    expect(db.getNode(ids[1])!.properties['name'], equals('Gus'));
    expect(db.getNode(ids[0])!.labels.toSet(), equals({'Person', 'Employee'}));

    final edgeIds = db.batchCreateEdges([
      BatchEdge(ids[0], ids[1], 'KNOWS', {'since': 2020}),
      BatchEdge(ids[1], ids[0], 'KNOWS'),
    ]);
    final edge = db.getEdge(edgeIds[0])!;
    expect([edge.sourceId, edge.targetId, edge.properties['since']],
        equals([ids[0], ids[1], 2020]));

    expect(() => db.batchCreateEdges([BatchEdge(ids[0], 88888, 'KNOWS')]),
        throwsA(isA<GrafeoException>()));
    expect(db.edgeCount, equals(2));
    expect(db.batchCreateNodesWithProps(['Person'], []), isEmpty);
  });

  test('results count writes', () {
    final write = db.execute(
        "INSERT (:Person {name: 'Vincent'})-[:KNOWS]->(:Person {name: 'Mia'})");
    expect(
        write.counters,
        equals(const WriteCounters(
            nodesCreated: 2,
            edgesCreated: 1,
            propertiesSet: 2,
            labelsAdded: 2)));
    expect(db.execute('MATCH (p:Person) RETURN p.name').counters,
        equals(WriteCounters.none));

    final tx = db.beginTransaction();
    final deleted =
        tx.execute("MATCH (p:Person {name: 'Vincent'}) DETACH DELETE p");
    expect([deleted.counters.nodesDeleted, deleted.counters.edgesDeleted],
        equals([1, 1]));
    tx.rollback();
  });

  test('batch vector search answers each query', () {
    db.execute(
        "INSERT (:Doc {topic: 'a', e: vector([1.0, 0.0])}), (:Doc {topic: 'b', e: vector([0.0, 1.0])})");
    db.createVectorIndex('Doc', 'e', 2, 'cosine');

    String topic(VectorResult hit) =>
        db.getNode(hit.nodeId)!.properties['topic'] as String;
    final results = db.batchVectorSearch(
        'Doc',
        'e',
        [
          [1.0, 0.0],
          [0.0, 1.0],
        ],
        k: 1);
    expect(results.map((hits) => topic(hits.single)), equals(['a', 'b']));

    final filtered = db.batchVectorSearch(
        'Doc',
        'e',
        [
          [1.0, 0.0]
        ],
        k: 2,
        filters: {'topic': 'b'});
    expect(topic(filtered.single.single), equals('b'));

    expect(
        () => db.batchVectorSearch(
            'Doc',
            'e',
            [
              [1.0, 0.0],
              [1.0]
            ],
            k: 1),
        throwsArgumentError);
  });

  test('change history records a detach delete', () {
    db.cdcEnabled = true;
    expect(db.cdcEnabled, isTrue);
    db.execute(
        "INSERT (:Person {name: 'Butch'})-[:KNOWS]->(:Person {name: 'Hans'})");
    db.execute("MATCH (p:Person {name: 'Butch'}) DETACH DELETE p");

    final node = db.nodeHistory(0);
    expect(node.map((e) => e.kind), equals(['create', 'delete']));
    expect(node[0].entityType, equals('node'));
    expect(node[0].after!['name'], equals('Butch'));
    expect(node[0].before, isNull);
    expect(node[0].graph, isNull);

    final edgeDelete = db.edgeHistory(0).last;
    expect([
      edgeDelete.kind,
      edgeDelete.edgeType,
      edgeDelete.deletedWith,
      edgeDelete.sourceId,
      edgeDelete.targetId
    ], equals(['delete', 'KNOWS', 0, 0, 1]));

    final since = db.nodeHistorySince(0, node[1].epoch);
    expect(since.single.kind, equals('delete'));

    final all = db.changesBetween(0, 0x7FFFFFFFFFFFFFFF);
    expect(all, hasLength(5));
  });

  test('upserts create, update and skip', () {
    final created = db.upsertNodes([
      'File'
    ], [
      {'path': 'a.rs', 'size': 3},
      {'path': 'b.rs'},
      {'size': 1},
    ], key: 'path');
    expect(
        [created.created, created.updated, created.skipped], equals([2, 0, 1]));
    expect(created.skippedRows, equals([2]));

    final updated = db.upsertNodes([
      'File'
    ], [
      {'path': 'a.rs', 'size': 19}
    ], key: 'path');
    expect(updated.updated, equals(1));
    expect(db.nodeCount, equals(2));

    final edges = db.upsertEdges(
        'IMPORTS',
        [
          {'id': 'a-b', 'from': 'a.rs', 'to': 'b.rs'},
          {'id': 'a-x', 'from': 'a.rs', 'to': 'x.rs'},
        ],
        endpointKey: 'path',
        endpointLabels: ['File'],
        sourceField: 'from',
        targetField: 'to');
    expect([edges.created, edges.skipped], equals([1, 1]));
    expect(db.edgeCount, equals(1));
  });

  test('graph handle works in its graph', () {
    db.execute('CREATE GRAPH trips');
    expect(() => db.graph('nowhere'), throwsA(isA<GrafeoException>()));

    final trips = db.graph('trips');
    addTearDown(trips.close);
    final prague = trips.createNode(['City'], {'name': 'Prague'});
    final berlin = trips.createNode(['City'], {'name': 'Berlin'});
    final route = trips.createEdge(prague, berlin, 'ROUTE', {'km': 350});
    trips.setNodeProperty(prague, 'country', 'CZ');
    expect(trips.getNode(prague)!.properties['country'], equals('CZ'));
    expect(trips.getEdge(route)!.properties['km'], equals(350));
    expect(trips.addNodeLabel(prague, 'Capital'), isTrue);
    expect(trips.addNodeLabel(prague, 'Capital'), isFalse);

    final result = trips.executeWithParams(
        r'MATCH (c:City {name: $name})-[:ROUTE]->(d) RETURN d.name AS d',
        {'name': 'Prague'});
    expect(result.rows.single['d'], equals('Berlin'));
    expect(db.nodeCount, equals(0));

    final upsert = trips.upsertNodes([
      'City'
    ], [
      {'name': 'Prague', 'country': 'CZ'}
    ], key: 'name');
    expect(upsert.updated, equals(1));
    expect(
        trips.batchCreateNodesWithProps([
          'City'
        ], [
          {'name': 'Paris'},
          {'name': 'Amsterdam'},
        ]),
        hasLength(2));

    expect(() => trips.deleteNode(prague), throwsA(isA<GrafeoException>()));
    expect(trips.detachDeleteNode(prague), isTrue);
    expect(trips.detachDeleteNode(prague), isFalse);

    expect(trips.getNode(424242), isNull, reason: 'absent, not a failure');
    db.execute('DROP GRAPH trips');
    // A read of a dropped graph throws: a failure, not a null.
    expect(() => trips.getNode(berlin), throwsA(isA<GrafeoException>()));
    expect(() => trips.execute('RETURN 1'), throwsA(isA<GrafeoException>()));
    expect(
        () => trips.createNode(['City'], {}), throwsA(isA<GrafeoException>()));

    trips.close();
    expect(() => trips.execute('RETURN 1'), throwsA(isA<GrafeoException>()));
  });
}
