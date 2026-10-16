package grafeo

import (
	"sort"
	"testing"
)

func sortedEdges(edges []NodeEdge) []NodeEdge {
	sort.Slice(edges, func(i, j int) bool { return edges[i].EdgeID < edges[j].EdgeID })
	return edges
}

func TestGetNodeEdges(t *testing.T) {
	db := mustOpen(t)
	node := func() uint64 {
		n, err := db.CreateNode([]string{"Person"}, nil)
		if err != nil {
			t.Fatal(err)
		}
		return n.ID
	}
	edge := func(src, dst uint64, edgeType string) uint64 {
		e, err := db.CreateEdge(src, dst, edgeType, nil)
		if err != nil {
			t.Fatal(err)
		}
		return e.ID
	}
	alix, gus, vincent := node(), node(), node()
	knowsGus := edge(alix, gus, "KNOWS")
	knowsVincent := edge(alix, vincent, "KNOWS")
	worksWith := edge(gus, alix, "WORKS_WITH")
	note := edge(alix, alix, "NOTE")

	check := func(got []NodeEdge, err error, want ...NodeEdge) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		got = sortedEdges(got)
		if len(got) != len(want) {
			t.Fatalf("got %v, want %v", got, want)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("got %v, want %v", got, want)
			}
		}
	}
	got, err := db.GetNodeEdges(alix, DirectionOut)
	check(got, err, NodeEdge{gus, knowsGus}, NodeEdge{vincent, knowsVincent}, NodeEdge{alix, note})
	got, err = db.GetNodeEdges(alix, DirectionIn)
	check(got, err, NodeEdge{gus, worksWith}, NodeEdge{alix, note})
	got, err = db.GetNodeEdges(alix, DirectionBoth)
	check(got, err, NodeEdge{gus, knowsGus}, NodeEdge{vincent, knowsVincent}, NodeEdge{gus, worksWith}, NodeEdge{alix, note})
	got, err = db.GetNodeEdges(alix, DirectionBoth, "KNOWS")
	check(got, err, NodeEdge{gus, knowsGus}, NodeEdge{vincent, knowsVincent})
	got, err = db.GetNodeEdges(88888, DirectionBoth)
	check(got, err)
	if _, err := db.GetNodeEdges(alix, Direction(7)); err == nil {
		t.Error("expected an unknown direction to fail")
	}
}

func TestGraphGetNodeEdges(t *testing.T) {
	db := mustOpen(t)
	if _, err := db.Execute("CREATE GRAPH trips"); err != nil {
		t.Fatal(err)
	}
	g, err := db.Graph("trips")
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	prague, _ := g.CreateNode([]string{"City"}, nil)
	berlin, _ := g.CreateNode([]string{"City"}, nil)
	route, err := g.CreateEdge(prague.ID, berlin.ID, "ROUTE", nil)
	if err != nil {
		t.Fatal(err)
	}
	edges, err := g.GetNodeEdges(berlin.ID, DirectionIn, "ROUTE")
	if err != nil || len(edges) != 1 || edges[0] != (NodeEdge{prague.ID, route.ID}) {
		t.Errorf("unexpected edges %v, %v", edges, err)
	}
	if _, err := db.Execute("DROP GRAPH trips"); err != nil {
		t.Fatal(err)
	}
	if _, err := g.GetNodeEdges(berlin.ID, DirectionIn); err == nil {
		t.Error("expected a dropped graph to fail")
	}
}
