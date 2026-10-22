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

  List<NodeEdge> sorted(List<NodeEdge> edges) =>
      [...edges]..sort((a, b) => a.edgeId.compareTo(b.edgeId));

  test('lists edges by direction and type', () {
    final alix = db.createNode(['Person'], {'name': 'Alix'});
    final gus = db.createNode(['Person'], {'name': 'Gus'});
    final vincent = db.createNode(['Person'], {'name': 'Vincent'});
    final knowsGus = db.createEdge(alix, gus, 'KNOWS', {});
    final knowsVincent = db.createEdge(alix, vincent, 'KNOWS', {});
    final worksWith = db.createEdge(gus, alix, 'WORKS_WITH', {});
    final note = db.createEdge(alix, alix, 'NOTE', {});

    expect(
        sorted(db.getNodeEdges(alix)),
        equals([
          NodeEdge(gus, knowsGus),
          NodeEdge(vincent, knowsVincent),
          NodeEdge(alix, note),
        ]));
    expect(sorted(db.getNodeEdges(alix, direction: Direction.incoming)),
        equals([NodeEdge(gus, worksWith), NodeEdge(alix, note)]));
    expect(
        sorted(db.getNodeEdges(alix, direction: Direction.both)),
        equals([
          NodeEdge(gus, knowsGus),
          NodeEdge(vincent, knowsVincent),
          NodeEdge(gus, worksWith),
          NodeEdge(alix, note),
        ]));
    expect(
        sorted(db.getNodeEdges(alix,
            direction: Direction.both, edgeTypes: ['KNOWS'])),
        equals([NodeEdge(gus, knowsGus), NodeEdge(vincent, knowsVincent)]));

    db.deleteEdge(knowsGus);
    expect(db.getNodeEdges(gus, direction: Direction.incoming), isEmpty);
    expect(db.getNodeEdges(88888, direction: Direction.both), isEmpty);
  });

  test('a graph handle lists the edges of its graph', () {
    db.execute('CREATE GRAPH trips');
    final trips = db.graph('trips');
    addTearDown(trips.close);
    final prague = trips.createNode(['City'], {'name': 'Prague'});
    final berlin = trips.createNode(['City'], {'name': 'Berlin'});
    final route = trips.createEdge(prague, berlin, 'ROUTE', {});
    expect(
        trips.getNodeEdges(berlin,
            direction: Direction.incoming, edgeTypes: ['ROUTE']),
        equals([NodeEdge(prague, route)]));
    expect(db.getNodeEdges(prague, direction: Direction.both), isEmpty);

    db.execute('DROP GRAPH trips');
    expect(() => trips.getNodeEdges(berlin), throwsA(isA<GrafeoException>()));
  });
}
