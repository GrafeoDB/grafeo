// Time travel: reads of the database as it was at an earlier epoch or time.

import 'dart:io';

import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

const peopleQuery = 'MATCH (p:Person) RETURN p.name AS name ORDER BY name';

List<Object?> people(QueryResult result) =>
    [for (final row in result.rows) row['name']];

void main() {
  late GrafeoDB db;

  setUp(() {
    db = GrafeoDB.memory(historyRetention: HistoryRetention.forever);
    db.execute("INSERT (:Person {name: 'Alix'})");
  });

  tearDown(() {
    db.close();
  });

  test('asOf an epoch reads the state then', () {
    final then = db.currentEpoch;
    db.execute("INSERT (:Person {name: 'Gus'})");
    final view = db.asOf(TimePoint.epoch(then));
    addTearDown(view.close);
    expect(view.epoch, equals(then));
    expect(people(view.execute(peopleQuery)), equals(['Alix']));
    expect(people(db.execute(peopleQuery)), equals(['Alix', 'Gus']));
    expect(() => view.execute("INSERT (:Person {name: 'Mia'})"),
        throwsA(isA<GrafeoException>()));
  });

  test('a view reads nodes, edges and properties as they were', () {
    final ids = db.execute("MATCH (a:Person {name: 'Alix'}) "
        "INSERT (a)-[k:KNOWS {since: 2020}]->(:Person {name: 'Gus'}) "
        'RETURN id(a) AS a, id(k) AS k');
    final alix = ids.rows[0]['a'] as int;
    final knows = ids.rows[0]['k'] as int;
    final then = TimePoint.epoch(db.currentEpoch);
    db.execute("MATCH (p:Person {name: 'Alix'}) SET p.name = 'Alix B.'");
    db.execute('MATCH ()-[k:KNOWS]->() DELETE k');

    final view = db.asOf(then);
    addTearDown(view.close);
    expect(view.getNode(alix)!.properties['name'], equals('Alix'));
    expect(view.getNodeProperty(alix, 'name'), equals('Alix'));
    expect(view.getNodeProperty(alix, 'city'), isNull);
    expect(view.getNode(alix + 1000), isNull);
    expect(view.getEdge(knows)!.properties['since'], equals(2020));
    final now = db.asOf(TimePoint.ago(Duration.zero));
    addTearDown(now.close);
    expect(now.getEdge(knows), isNull);
    expect(now.getNodeProperty(alix, 'name'), equals('Alix B.'));
  });

  test('executeAsOf and a view take params', () {
    final then = TimePoint.epoch(db.currentEpoch);
    db.execute("INSERT (:Person {name: 'Gus'})");
    const query =
        r'MATCH (p:Person) WHERE p.name <> $not RETURN p.name AS name';
    final params = {'not': 'Vincent'};
    expect(
        people(db.executeAsOf(query, then, params: params)), equals(['Alix']));
    final view = db.asOf(then);
    addTearDown(view.close);
    expect(people(view.execute(query, params: params)), equals(['Alix']));
  });

  test('a time reads the last commit at or before it', () {
    sleep(const Duration(milliseconds: 10));
    final beforeGus = DateTime.now();
    // Commit times have the resolution of a millisecond: Gus commits after it.
    sleep(const Duration(milliseconds: 10));
    db.execute("INSERT (:Person {name: 'Gus'})");
    expect(people(db.executeAsOf(peopleQuery, TimePoint.at(beforeGus))),
        equals(['Alix']));
    expect(db.epochAsOf(TimePoint.at(beforeGus.toUtc())),
        equals(db.epochAsOf(TimePoint.at(beforeGus))));
    expect(db.epochAsOf(TimePoint.ago(Duration.zero)), equals(db.currentEpoch));
    expect(db.epochAsOf(TimePoint.epoch(db.currentEpoch + 1000)),
        equals(db.currentEpoch));
  });

  test('negative points and retentions throw', () {
    expect(() => TimePoint.epoch(-1), throwsRangeError);
    expect(() => TimePoint.ago(const Duration(seconds: -1)), throwsRangeError);
    expect(() => HistoryRetention.duration(const Duration(hours: -1)),
        throwsRangeError);
  });

  test('a time without history or before it throws', () {
    final long = TimePoint.at(DateTime.utc(2000));
    final none = GrafeoDB.memory();
    addTearDown(none.close);
    expect(
        () => none.asOf(long),
        throwsA(isA<GrafeoException>()
            .having((e) => e.message, 'message', contains('needs history'))));
    final hour = GrafeoDB.memory(
        historyRetention: HistoryRetention.duration(const Duration(hours: 1)));
    addTearDown(hour.close);
    hour.execute("INSERT (:Person {name: 'Alix'})");
    expect(
        () => hour.epochAsOf(long),
        throwsA(isA<GrafeoException>()
            .having((e) => e.message, 'message', contains('not retained'))));
    GrafeoDB.memory(historyRetention: HistoryRetention.off).close();
  });

  test('a closed view throws', () {
    final view = db.asOf(TimePoint.epoch(db.currentEpoch));
    view.close();
    view.close();
    expect(() => view.execute('MATCH (n) RETURN n'),
        throwsA(isA<DatabaseException>()));
  });

  test('the file stores the retention', () {
    final dir = Directory.systemTemp.createTempSync('grafeo-dart-');
    addTearDown(() => dir.deleteSync(recursive: true));
    final path = '${dir.path}${Platform.pathSeparator}history.grafeo';
    final first =
        GrafeoDB.open(path, historyRetention: HistoryRetention.forever);
    first.execute("INSERT (:Person {name: 'Alix'})");
    final then = first.currentEpoch;
    first.execute("INSERT (:Person {name: 'Gus'})");
    first.close();
    final again = GrafeoDB.open(path);
    addTearDown(again.close);
    expect(people(again.executeAsOf(peopleQuery, TimePoint.epoch(then))),
        equals(['Alix']));
  });
}
