package grafeo

import (
	"errors"
	"testing"
)

func mustOpen(t *testing.T) *Database {
	t.Helper()
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

func TestBatchCreateNodesAndEdges(t *testing.T) {
	db := mustOpen(t)
	ids, err := db.BatchCreateNodesWithProps([]string{"Person", "Employee"}, []map[string]any{
		{"name": "Alix"}, {"name": "Gus"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 2 {
		t.Fatalf("expected 2 ids, got %v", ids)
	}
	edgeIDs, err := db.BatchCreateEdges([]BatchEdge{
		{Src: ids[0], Dst: ids[1], Type: "KNOWS", Properties: map[string]any{"since": 2020}},
		{Src: ids[1], Dst: ids[0], Type: "KNOWS"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(edgeIDs) != 2 || db.EdgeCount() != 2 {
		t.Fatalf("expected 2 edges, got %v and %d", edgeIDs, db.EdgeCount())
	}
	if _, err := db.BatchCreateEdges([]BatchEdge{{Src: ids[0], Dst: 88888, Type: "KNOWS"}}); err == nil {
		t.Error("expected an edge to a missing node to fail the batch")
	}
	if db.EdgeCount() != 2 {
		t.Errorf("the failed batch wrote %d edges", db.EdgeCount()-2)
	}
}

func TestResultCounters(t *testing.T) {
	db := mustOpen(t)
	result, err := db.Execute("INSERT (:Person {name: 'Vincent'})-[:KNOWS]->(:Person {name: 'Mia'})")
	if err != nil {
		t.Fatal(err)
	}
	if result.Counters.NodesCreated != 2 || result.Counters.EdgesCreated != 1 || result.Counters.PropertiesSet != 2 {
		t.Errorf("unexpected counters %+v", result.Counters)
	}
	read, err := db.Execute("MATCH (p:Person) RETURN p.name")
	if err != nil {
		t.Fatal(err)
	}
	if read.Counters != (WriteCounters{}) {
		t.Errorf("a read counted %+v", read.Counters)
	}
}

func TestBatchVectorSearch(t *testing.T) {
	db := mustOpen(t)
	if _, err := db.Execute("INSERT (:Doc {topic: 'a', e: vector([1.0, 0.0])}), (:Doc {topic: 'b', e: vector([0.0, 1.0])})"); err != nil {
		t.Fatal(err)
	}
	if err := db.CreateVectorIndex("Doc", "e", WithMetric("cosine")); err != nil {
		t.Fatal(err)
	}
	results, err := db.BatchVectorSearch("Doc", "e", [][]float32{{1, 0}, {0, 1}}, 1, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 || len(results[0]) != 1 || results[0][0].NodeID == results[1][0].NodeID {
		t.Errorf("unexpected results %+v", results)
	}
}

func TestChangeHistory(t *testing.T) {
	db := mustOpen(t)
	db.SetCDCEnabled(true)
	if !db.IsCDCEnabled() {
		t.Fatal("expected change data capture on")
	}
	if _, err := db.Execute("INSERT (:Person {name: 'Butch'})-[:KNOWS]->(:Person {name: 'Hans'})"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Execute("MATCH (p:Person {name: 'Butch'}) DETACH DELETE p"); err != nil {
		t.Fatal(err)
	}
	node, err := db.NodeHistory(0)
	if err != nil {
		t.Fatal(err)
	}
	if len(node) != 2 || node[0].Kind != "create" || node[1].Kind != "delete" || node[0].EntityType != "node" {
		t.Fatalf("unexpected node history %+v", node)
	}
	edge, err := db.EdgeHistory(0)
	if err != nil {
		t.Fatal(err)
	}
	last := edge[len(edge)-1]
	if last.DeletedWith == nil || *last.DeletedWith != 0 || last.EdgeType == nil || *last.EdgeType != "KNOWS" {
		t.Errorf("unexpected edge delete %+v", last)
	}
	all, err := db.ChangesBetween(0, ^uint64(0))
	if err != nil {
		t.Fatal(err)
	}
	if len(all) < 5 {
		t.Errorf("expected every change, got %d", len(all))
	}
	since, err := db.NodeHistorySince(0, node[1].Epoch)
	if err != nil || len(since) != 1 {
		t.Errorf("expected the delete only, got %+v, %v", since, err)
	}
}

func TestUpserts(t *testing.T) {
	db := mustOpen(t)
	summary, err := db.UpsertNodes([]string{"File"}, []map[string]any{
		{"path": "a.rs", "size": 3}, {"path": "b.rs"}, {"size": 1},
	}, &UpsertNodesOptions{Key: "path"})
	if err != nil {
		t.Fatal(err)
	}
	if summary.Created != 2 || summary.Skipped != 1 || len(summary.SkippedRows) != 1 || summary.SkippedRows[0] != 2 {
		t.Errorf("unexpected summary %+v", summary)
	}
	edges, err := db.UpsertEdges("IMPORTS", []map[string]any{
		{"id": "a-b", "from": "a.rs", "to": "b.rs"},
	}, &UpsertEdgesOptions{EndpointKey: "path", EndpointLabels: []string{"File"}, SrcField: "from", DstField: "to"})
	if err != nil {
		t.Fatal(err)
	}
	if edges.Created != 1 || db.EdgeCount() != 1 {
		t.Errorf("unexpected edge summary %+v", edges)
	}
}

func TestGraphHandle(t *testing.T) {
	db := mustOpen(t)
	if _, err := db.Execute("CREATE GRAPH trips"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Graph("nowhere"); err == nil || !errors.Is(err, ErrDatabase) {
		t.Errorf("expected an error for a missing graph, got %v", err)
	}
	g, err := db.Graph("trips")
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	prague, err := g.CreateNode([]string{"City"}, map[string]any{"name": "Prague"})
	if err != nil {
		t.Fatal(err)
	}
	berlin, err := g.CreateNode([]string{"City"}, map[string]any{"name": "Berlin"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := g.CreateEdge(prague.ID, berlin.ID, "ROUTE", nil); err != nil {
		t.Fatal(err)
	}
	if err := g.SetNodeProperty(prague.ID, "country", "CZ"); err != nil {
		t.Fatal(err)
	}
	node, err := g.GetNode(prague.ID)
	if err != nil || node.Properties["country"] != "CZ" {
		t.Errorf("unexpected node %+v, %v", node, err)
	}
	result, err := g.ExecuteParams("MATCH (c:City {name: $name})-[:ROUTE]->(d) RETURN d.name AS d", map[string]any{"name": "Prague"})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Rows) != 1 || result.Rows[0]["d"] != "Berlin" {
		t.Errorf("unexpected rows %+v", result.Rows)
	}
	if db.NodeCount() != 0 {
		t.Errorf("the default graph has %d nodes", db.NodeCount())
	}
	summary, err := g.UpsertNodes([]string{"City"}, []map[string]any{{"name": "Prague", "country": "CZ"}}, &UpsertNodesOptions{Key: "name"})
	if err != nil || summary.Updated != 1 {
		t.Errorf("unexpected upsert %+v, %v", summary, err)
	}
	deleted, err := g.DetachDeleteNode(prague.ID)
	if err != nil || !deleted {
		t.Errorf("expected the detach delete, got %v, %v", deleted, err)
	}
	if _, err := db.Execute("DROP GRAPH trips"); err != nil {
		t.Fatal(err)
	}
	if _, err := g.Execute("RETURN 1"); err == nil {
		t.Error("expected a dropped graph to fail")
	}
}
