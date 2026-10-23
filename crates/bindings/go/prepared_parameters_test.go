package grafeo

import (
	"fmt"
	"reflect"
	"sort"
	"testing"
)

// A statement called again with other values answers for those values: the
// engine optimizes it once with its parameters in and fills a copy of that
// plan with each call's values, here through the C API's parameter path.

func preparedPeople(t *testing.T) *Database {
	t.Helper()
	db := mustOpen(t)
	for _, person := range []struct {
		name string
		age  int
	}{{"Alix", 30}, {"Gus", 25}, {"Vincent", 41}, {"Mia", 25}} {
		if _, err := db.ExecuteParams(
			"INSERT (:Person {name: $name, age: $age})",
			map[string]any{"name": person.name, "age": person.age},
		); err != nil {
			t.Fatal(err)
		}
	}
	return db
}

// column returns one column of a result as text, for comparisons that do
// not depend on how JSON numbers decode.
func column(t *testing.T, result *QueryResult, err error, name string) []string {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	values := []string{}
	for _, row := range result.Rows {
		values = append(values, fmt.Sprint(row[name]))
	}
	return values
}

func TestPreparedLookupTakesEachCallsValue(t *testing.T) {
	db := preparedPeople(t)
	query := "MATCH (p:Person) WHERE p.name = $name RETURN p.age AS age"
	for _, call := range []struct {
		name string
		ages []string
	}{{"Gus", []string{"25"}}, {"Alix", []string{"30"}}, {"Nobody", []string{}}, {"Gus", []string{"25"}}} {
		result, err := db.ExecuteParams(query, map[string]any{"name": call.name})
		if got := column(t, result, err, "age"); !reflect.DeepEqual(got, call.ages) {
			t.Errorf("%s: got %v, want %v", call.name, got, call.ages)
		}
	}
}

func TestPreparedSkipAndLimitTakeEachCallsValues(t *testing.T) {
	db := preparedPeople(t)
	query := "MATCH (p:Person) RETURN p.name AS name ORDER BY p.name SKIP $skip LIMIT $limit"
	names := []string{"Alix", "Gus", "Mia", "Vincent"}
	for _, call := range [][2]int{{0, 1}, {1, 2}, {3, 5}, {0, 4}} {
		skip, limit := call[0], call[1]
		result, err := db.ExecuteParams(query, map[string]any{"skip": skip, "limit": limit})
		want := names[skip:min(skip+limit, len(names))]
		if got := column(t, result, err, "name"); !reflect.DeepEqual(got, want) {
			t.Errorf("skip %d limit %d: got %v, want %v", skip, limit, got, want)
		}
	}
}

func TestPreparedValueOfAnotherTypeAnswersAsItsLiteral(t *testing.T) {
	db := preparedPeople(t)
	query := "MATCH (p:Person) WHERE p.age = $age RETURN p.name AS name"
	for _, call := range []struct {
		value   any
		literal string
	}{{25, "25"}, {"25", "'25'"}, {25, "25"}, {true, "true"}} {
		result, err := db.ExecuteParams(query, map[string]any{"age": call.value})
		got := column(t, result, err, "name")
		literalQuery := "MATCH (p:Person) WHERE p.age = " + call.literal + " RETURN p.name AS name"
		expected, err := db.Execute(literalQuery)
		want := column(t, expected, err, "name")
		sort.Strings(got)
		sort.Strings(want)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%v: got %v, want %v", call.value, got, want)
		}
	}
}
