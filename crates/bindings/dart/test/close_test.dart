// After close() every call on the database throws the GRAFEO-T007 error, and
// so does every call on what it handed out before: graphs, transactions,
// streams and views. Only close() answers, as a no-op.

import 'dart:io';

import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

bool _isClosedError(Object? error) =>
    error is DatabaseException && error.message.contains('GRAFEO-T007');

/// The rows whose call did not throw GRAFEO-T007, with what they did.
List<String> _notClosed(Map<String, Object? Function()> calls) {
  final missed = <String>[];
  for (final MapEntry(key: name, value: call) in calls.entries) {
    try {
      call();
      missed.add('$name: answered');
    } catch (error) {
      if (!_isClosedError(error)) missed.add('$name: $error');
    }
  }
  return missed..sort();
}

/// The public members `lib/src/database.dart` declares on GrafeoDB.
Set<String> _publicMembers() {
  final source = File('lib/src/database.dart').readAsStringSync();
  final member = RegExp(
    r'^  (?:[A-Z][\w<>?, ]*|void|int|bool|double|String\??|List<[^\n]*?>\??|Map<[^\n]*?>\??) '
    r'(?:get |set )?([a-z]\w*)\s*(?:\(|=>|\{)',
    multiLine: true,
  );
  return member.allMatches(source).map((m) => m.group(1)!).toSet();
}

void main() {
  late Directory dir;
  late String path;

  setUp(() {
    dir = Directory.systemTemp.createTempSync('grafeo-dart-close-');
    path = '${dir.path}/amsterdam.grafeo';
  });

  tearDown(() {
    try {
      dir.deleteSync(recursive: true);
    } on FileSystemException {
      // a handle may still hold the file on Windows
    }
  });

  GrafeoDB openPersistent() {
    final db = GrafeoDB.open(path);
    db.execute(
      "INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})",
    );
    return db;
  }

  test('every member of a closed database throws GRAFEO-T007', () {
    final db = openPersistent();
    db.close();
    db.close(); // a second close does nothing

    const query = 'MATCH (p:Person) RETURN p.name';
    final vector = [0.1, 0.2, 0.3, 0.4];
    final calls = <String, Object? Function()>{
      'execute': () => db.execute(query),
      'executeStream': () => db.executeStream(query),
      'executeWithParams': () => db.executeWithParams(query, {}),
      'executeCypher': () => db.executeCypher(query),
      'executeGremlin': () => db.executeGremlin('g.V()'),
      'executeGraphql': () => db.executeGraphql('{ Person { name } }'),
      'executeSparql': () => db.executeSparql('SELECT ?s WHERE { ?s ?p ?o }'),
      'executeCypherWithParams': () => db.executeCypherWithParams(query, {}),
      'executeGremlinWithParams': () =>
          db.executeGremlinWithParams('g.V()', {}),
      'executeGraphqlWithParams': () =>
          db.executeGraphqlWithParams('{ Person { name } }', {}),
      'executeSparqlWithParams': () =>
          db.executeSparqlWithParams('SELECT ?s WHERE { ?s ?p ?o }', {}),
      'executeLanguage': () => db.executeLanguage('gql', query),
      'nodeCount': () => db.nodeCount,
      'edgeCount': () => db.edgeCount,
      'memoryUsage': () => db.memoryUsage(),
      'openWarnings': () => db.openWarnings(),
      'info': () => db.info(),
      'setSchema': () => db.setSchema('berlin'),
      'resetSchema': () => db.resetSchema(),
      'currentSchema': () => db.currentSchema(),
      'beginTransaction': () => db.beginTransaction(),
      'beginTransactionWithIsolation': () =>
          db.beginTransactionWithIsolation(IsolationLevel.readCommitted),
      'createNode': () => db.createNode(['Person'], {'name': 'Vincent'}),
      'getNode': () => db.getNode(0),
      'getNodeLabels': () => db.getNodeLabels(0),
      'deleteNode': () => db.deleteNode(0),
      'detachDeleteNode': () => db.detachDeleteNode(0),
      'setNodeProperty': () => db.setNodeProperty(0, 'city', 'Amsterdam'),
      'removeNodeProperty': () => db.removeNodeProperty(0, 'name'),
      'addNodeLabel': () => db.addNodeLabel(0, 'Director'),
      'removeNodeLabel': () => db.removeNodeLabel(0, 'Person'),
      'createEdge': () => db.createEdge(0, 1, 'KNOWS', {}),
      'getEdge': () => db.getEdge(0),
      'getNodeEdges': () => db.getNodeEdges(0),
      'deleteEdge': () => db.deleteEdge(0),
      'setEdgeProperty': () => db.setEdgeProperty(0, 'since', 2020),
      'removeEdgeProperty': () => db.removeEdgeProperty(0, 'since'),
      'batchCreateNodesWithProps': () => db.batchCreateNodesWithProps(
        ['Person'],
        [
          {'name': 'Mia'},
        ],
      ),
      'batchCreateEdges': () => db.batchCreateEdges([]),
      'upsertNodes': () => db.upsertNodes(
        ['Person'],
        [
          {'id': 1},
        ],
      ),
      'upsertEdges': () => db.upsertEdges('KNOWS', [
        {'id': 1, 'src': 1, 'dst': 2},
      ]),
      'currentEpoch': () => db.currentEpoch,
      'epochAsOf': () => db.epochAsOf(TimePoint.epoch(0)),
      'asOf': () => db.asOf(TimePoint.epoch(0)),
      'executeAsOf': () => db.executeAsOf(query, TimePoint.epoch(0)),
      'graph': () => db.graph('berlin'),
      'cdcEnabled': () => db.cdcEnabled,
      'nodeHistory': () => db.nodeHistory(0),
      'edgeHistory': () => db.edgeHistory(0),
      'nodeHistorySince': () => db.nodeHistorySince(0, 0),
      'changesBetween': () => db.changesBetween(0, 1),
      'createPropertyIndex': () => db.createPropertyIndex('name'),
      'dropPropertyIndex': () => db.dropPropertyIndex('name'),
      'hasPropertyIndex': () => db.hasPropertyIndex('name'),
      'findNodesByProperty': () => db.findNodesByProperty('name', 'Alix'),
      'createVectorIndex': () =>
          db.createVectorIndex('Item', 'embedding', 4, 'cosine'),
      'vectorSearch': () => db.vectorSearch('Item', 'embedding', vector, k: 1),
      'batchCreateNodes': () =>
          db.batchCreateNodes('Item', 'embedding', [vector]),
      'dropVectorIndex': () => db.dropVectorIndex('Item', 'embedding'),
      'rebuildVectorIndex': () => db.rebuildVectorIndex('Item', 'embedding'),
      'batchVectorSearch': () =>
          db.batchVectorSearch('Item', 'embedding', [vector], k: 1),
      'mmrSearch': () => db.mmrSearch(
        'Item',
        'embedding',
        vector,
        k: 1,
        fetchK: 4,
        lambda: 0.5,
        ef: 16,
      ),
      'save': () => db.save('${dir.path}/copy.grafeo'),
      'walCheckpoint': () => db.walCheckpoint(),
      'setCheckpointTriggers': () =>
          db.setCheckpointTriggers(interval: const Duration(minutes: 1)),
      'checkpointStatus': () => db.checkpointStatus(),
    };

    // The setter shares its name with the getter row above.
    final setter = _notClosed({'cdcEnabled=': () => db.cdcEnabled = true});
    final unlisted =
        _publicMembers()
            .where((name) => name != 'close' && !calls.containsKey(name))
            .toList()
          ..sort();
    expect(unlisted, isEmpty, reason: 'public members without a row');
    expect([..._notClosed(calls), ...setter], isEmpty);
  });

  test('the closed error names its code and what to do', () {
    final db = openPersistent();
    db.close();
    expect(
      () => db.execute('MATCH (p:Person) RETURN p.name'),
      throwsA(
        isA<DatabaseException>()
            .having(
              (e) => e.message,
              'message',
              contains('GRAFEO-T007: the database is closed: open it again'),
            )
            .having((e) => e.gqlStatus, 'gqlStatus', '25000'),
      ),
    );
  });

  test('what a closed database handed out throws GRAFEO-T007', () {
    final db = openPersistent();
    db.execute('CREATE GRAPH berlin');
    final graph = db.graph('berlin');
    final transaction = db.beginTransaction();
    transaction.execute("INSERT (:Person {name: 'Mia'})");
    final stream = db.executeStream('MATCH (p:Person) RETURN p.name');
    final view = db.asOf(TimePoint.epoch(db.currentEpoch));
    db.close();

    const query = 'MATCH (p:Person) RETURN p.name';
    expect(
      _notClosed({
        'graph.execute': () => graph.execute(query),
        'transaction.execute': () => transaction.execute(query),
        'transaction.commit': () => transaction.commit(),
        'stream.next': () => stream.next(),
        'stream.columns': () => stream.columns,
        'stream.warnings': () => stream.warnings,
        'view.epoch': () => view.epoch,
        'view.execute': () => view.execute(query),
      }),
      isEmpty,
    );
    view.close();
    stream.close();
    graph.close();

    final reopened = GrafeoDB.open(path);
    addTearDown(reopened.close);
    expect(
      reopened.nodeCount,
      2,
      reason: 'the transaction open across close() left nothing',
    );
  });
}
