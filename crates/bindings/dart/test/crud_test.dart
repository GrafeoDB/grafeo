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

  group('node CRUD', () {
    test('create and get node', () {
      final id = db.createNode(['Person'], {'name': 'Alix', 'age': 30});
      expect(id, greaterThanOrEqualTo(0));

      final node = db.getNode(id);
      expect(node.id, equals(id));
      expect(node.labels, contains('Person'));
      expect(node.properties['name'], equals('Alix'));
      expect(node.properties['age'], equals(30));
    });

    test('delete node', () {
      final id = db.createNode(['Person'], {'name': 'Gus'});
      expect(db.nodeCount, equals(1));

      final deleted = db.deleteNode(id);
      expect(deleted, isTrue);
      expect(db.nodeCount, equals(0));
    });

    test('set and remove node property', () {
      final id = db.createNode(['Person'], {'name': 'Vincent'});

      db.setNodeProperty(id, 'role', 'hitman');
      var node = db.getNode(id);
      expect(node.properties['role'], equals('hitman'));

      db.removeNodeProperty(id, 'role');
      node = db.getNode(id);
      expect(node.properties.containsKey('role'), isFalse);
    });

    test('add and remove node label', () {
      final id = db.createNode(['Person'], {'name': 'Butch'});

      db.addNodeLabel(id, 'Boxer');
      var node = db.getNode(id);
      expect(node.labels, containsAll(['Person', 'Boxer']));

      db.removeNodeLabel(id, 'Boxer');
      node = db.getNode(id);
      expect(node.labels, contains('Person'));
      expect(node.labels, isNot(contains('Boxer')));
    });
  });

  group('edge CRUD', () {
    test('create and get edge', () {
      final alixId = db.createNode(['Person'], {'name': 'Alix'});
      final gusId = db.createNode(['Person'], {'name': 'Gus'});

      final edgeId = db.createEdge(alixId, gusId, 'KNOWS', {'since': 2020});
      expect(edgeId, greaterThanOrEqualTo(0));
      expect(db.edgeCount, equals(1));

      final edge = db.getEdge(edgeId);
      expect(edge.id, equals(edgeId));
      expect(edge.type, equals('KNOWS'));
      expect(edge.sourceId, equals(alixId));
      expect(edge.targetId, equals(gusId));
      expect(edge.properties['since'], equals(2020));
    });

    test('delete edge', () {
      final a = db.createNode(['Person'], {'name': 'Django'});
      final b = db.createNode(['Person'], {'name': 'Shosanna'});
      final edgeId = db.createEdge(a, b, 'MEETS', {});

      expect(db.edgeCount, equals(1));
      final deleted = db.deleteEdge(edgeId);
      expect(deleted, isTrue);
      expect(db.edgeCount, equals(0));
    });

    test('set and remove edge property', () {
      final a = db.createNode(['Person'], {'name': 'Hans'});
      final b = db.createNode(['Person'], {'name': 'Beatrix'});
      final edgeId = db.createEdge(a, b, 'CONFRONTS', {});

      db.setEdgeProperty(edgeId, 'intensity', 'high');
      var edge = db.getEdge(edgeId);
      expect(edge.properties['intensity'], equals('high'));

      db.removeEdgeProperty(edgeId, 'intensity');
      edge = db.getEdge(edgeId);
      expect(edge.properties.containsKey('intensity'), isFalse);
    });
  });

  group('entity extraction', () {
    test('query result contains extracted nodes', () {
      db.execute("INSERT (:Person {name: 'Alix', age: 30})");
      final result = db.execute('MATCH (p:Person) RETURN p');
      expect(result.nodes, hasLength(1));
      expect(result.nodes.first.labels, contains('Person'));
    });

    test('query result contains extracted edges', () {
      db.execute(
          "INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})");
      final result = db.execute('MATCH ()-[e:KNOWS]->() RETURN e');
      expect(result.edges, hasLength(1));
      expect(result.edges.first.type, equals('KNOWS'));
    });
  });
}
