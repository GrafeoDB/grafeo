package grafeo

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

const peopleQuery = "MATCH (p:Person) RETURN p.name AS name ORDER BY name"

func mustOpenWithHistory(t *testing.T) *Database {
	t.Helper()
	db, err := OpenInMemoryWithHistory(HistoryForever)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	if _, err := db.Execute("INSERT (:Person {name: 'Alix'})"); err != nil {
		t.Fatal(err)
	}
	return db
}

// people reads the names of a query's result, as people(t)(db.Execute(query)).
func people(t *testing.T) func(*QueryResult, error) string {
	return func(result *QueryResult, err error) string {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		names := make([]string, len(result.Rows))
		for i, row := range result.Rows {
			names[i] = fmt.Sprint(row["name"])
		}
		return strings.Join(names, ",")
	}
}

func TestAsOfAnEpochReadsTheStateThen(t *testing.T) {
	db := mustOpenWithHistory(t)
	then := db.CurrentEpoch()
	if _, err := db.Execute("INSERT (:Person {name: 'Gus'})"); err != nil {
		t.Fatal(err)
	}
	view, err := db.AsOf(AtEpoch(then))
	if err != nil {
		t.Fatal(err)
	}
	defer view.Close()
	if view.Epoch() != then {
		t.Errorf("view epoch %d, want %d", view.Epoch(), then)
	}
	if got := people(t)(view.Execute(peopleQuery)); got != "Alix" {
		t.Errorf("then: %q", got)
	}
	if got := people(t)(db.Execute(peopleQuery)); got != "Alix,Gus" {
		t.Errorf("now: %q", got)
	}
	if _, err := view.Execute("INSERT (:Person {name: 'Mia'})"); err == nil {
		t.Error("a write through the view succeeded")
	}
}

func TestTheViewReadsNodesEdgesAndPropertiesAsTheyWere(t *testing.T) {
	db := mustOpenWithHistory(t)
	ids, err := db.Execute("MATCH (a:Person {name: 'Alix'}) INSERT (a)-[k:KNOWS {since: 2020}]->(:Person {name: 'Gus'}) RETURN id(a) AS a, id(k) AS k")
	if err != nil {
		t.Fatal(err)
	}
	alix, _ := strconv.ParseUint(fmt.Sprint(ids.Rows[0]["a"]), 10, 64)
	knows, _ := strconv.ParseUint(fmt.Sprint(ids.Rows[0]["k"]), 10, 64)
	then := AtEpoch(db.CurrentEpoch())
	for _, query := range []string{"MATCH (p:Person {name: 'Alix'}) SET p.name = 'Alix B.'", "MATCH ()-[k:KNOWS]->() DELETE k"} {
		if _, err := db.Execute(query); err != nil {
			t.Fatal(err)
		}
	}
	view, err := db.AsOf(then)
	if err != nil {
		t.Fatal(err)
	}
	defer view.Close()
	if node, err := view.GetNode(alix); err != nil || node.Properties["name"] != "Alix" {
		t.Errorf("GetNode: %v %v", node, err)
	}
	if name, err := view.GetNodeProperty(alix, "name"); err != nil || name != "Alix" {
		t.Errorf("GetNodeProperty: %v %v", name, err)
	}
	if city, err := view.GetNodeProperty(alix, "city"); err != nil || city != nil {
		t.Errorf("a missing property: %v %v", city, err)
	}
	if _, err := view.GetNode(alix + 1000); err == nil {
		t.Error("a node that did not exist was found")
	}
	if edge, err := view.GetEdge(knows); err != nil || edge.Properties["since"] != float64(2020) {
		t.Errorf("GetEdge: %v %v", edge, err)
	}
	now, err := db.AsOf(Ago(0))
	if err != nil {
		t.Fatal(err)
	}
	defer now.Close()
	if _, err := now.GetEdge(knows); err == nil {
		t.Error("a deleted edge was found now")
	}
}

func TestExecuteAsOfTakesParams(t *testing.T) {
	db := mustOpenWithHistory(t)
	then := AtEpoch(db.CurrentEpoch())
	if _, err := db.Execute("INSERT (:Person {name: 'Gus'})"); err != nil {
		t.Fatal(err)
	}
	params := map[string]any{"not": "Vincent"}
	query := "MATCH (p:Person) WHERE p.name <> $not RETURN p.name AS name"
	if got := people(t)(db.ExecuteAsOf(query, then, params)); got != "Alix" {
		t.Errorf("ExecuteAsOf: %q", got)
	}
	if got := people(t)(db.ExecuteAsOf(peopleQuery, then, nil)); got != "Alix" {
		t.Errorf("ExecuteAsOf without params: %q", got)
	}
	view, err := db.AsOf(then)
	if err != nil {
		t.Fatal(err)
	}
	defer view.Close()
	if got := people(t)(view.ExecuteParams(query, params)); got != "Alix" {
		t.Errorf("ExecuteParams: %q", got)
	}
}

func TestATimeReadsTheLastCommitAtOrBeforeIt(t *testing.T) {
	db := mustOpenWithHistory(t)
	time.Sleep(10 * time.Millisecond)
	beforeGus := time.Now()
	// Commit times have the resolution of a millisecond: Gus commits after it.
	time.Sleep(10 * time.Millisecond)
	if _, err := db.Execute("INSERT (:Person {name: 'Gus'})"); err != nil {
		t.Fatal(err)
	}
	if got := people(t)(db.ExecuteAsOf(peopleQuery, AtTime(beforeGus), nil)); got != "Alix" {
		t.Errorf("before Gus: %q", got)
	}
	amsterdam := beforeGus.In(time.FixedZone("CEST", 2*60*60))
	a, errA := db.EpochAsOf(AtTime(beforeGus))
	b, errB := db.EpochAsOf(AtTime(amsterdam))
	if errA != nil || errB != nil || a != b {
		t.Errorf("zones: %d %v, %d %v", a, errA, b, errB)
	}
	if epoch, err := db.EpochAsOf(Ago(0)); err != nil || epoch != db.CurrentEpoch() {
		t.Errorf("Ago(0): %d %v", epoch, err)
	}
	if epoch, err := db.EpochAsOf(AtEpoch(^uint64(0))); err != nil || epoch != db.CurrentEpoch() {
		t.Errorf("a later epoch: %d %v", epoch, err)
	}
	if _, err := db.EpochAsOf(Ago(-time.Second)); err == nil {
		t.Error("a negative Ago succeeded")
	}
}

func TestHistoryLimits(t *testing.T) {
	long := AtTime(time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC))
	db := mustOpen(t)
	if _, err := db.AsOf(long); err == nil || !strings.Contains(err.Error(), "needs history") {
		t.Errorf("without history: %v", err)
	}
	hour, err := OpenInMemoryWithHistory(HistoryFor(time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	defer hour.Close()
	if _, err := hour.Execute("INSERT (:Person {name: 'Alix'})"); err != nil {
		t.Fatal(err)
	}
	if _, err := hour.EpochAsOf(long); err == nil || !strings.Contains(err.Error(), "not retained") {
		t.Errorf("before the history: %v", err)
	}
	off, err := OpenInMemoryWithHistory(HistoryOff)
	if err != nil {
		t.Fatal(err)
	}
	off.Close()
}

func TestAClosedViewFails(t *testing.T) {
	db := mustOpenWithHistory(t)
	view, err := db.AsOf(AtEpoch(db.CurrentEpoch()))
	if err != nil {
		t.Fatal(err)
	}
	view.Close()
	view.Close()
	if _, err := view.Execute("MATCH (n) RETURN n"); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Errorf("closed view: %v", err)
	}
}

func TestTheFileStoresTheRetention(t *testing.T) {
	path := filepath.Join(t.TempDir(), "history.grafeo")
	db, err := OpenWithHistory(path, HistoryForever)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Execute("INSERT (:Person {name: 'Alix'})"); err != nil {
		t.Fatal(err)
	}
	then := db.CurrentEpoch()
	if _, err := db.Execute("INSERT (:Person {name: 'Gus'})"); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	db, err = Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if got := people(t)(db.ExecuteAsOf(peopleQuery, AtEpoch(then), nil)); got != "Alix" {
		t.Errorf("after reopen: %q", got)
	}
}
