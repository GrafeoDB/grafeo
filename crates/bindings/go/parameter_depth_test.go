package grafeo

import (
	"strings"
	"testing"
)

// A parameter nested deeper than a property value may be (128 levels) is
// refused with an error through the C API's JSON parameters, and the
// database answers afterwards: no stack overflow, also far past the limit.

// nestedList is the JSON of a list nested depth levels deep around 1.
func nestedList(depth int) string {
	return strings.Repeat("[", depth) + "1" + strings.Repeat("]", depth)
}

// nestedMap is the JSON of a map nested depth levels deep ({"a": {"a": ... 1}}).
func nestedMap(depth int) string {
	return strings.Repeat(`{"a":`, depth) + "1" + strings.Repeat("}", depth)
}

func TestDeeplyNestedParametersAreRefused(t *testing.T) {
	db := mustOpen(t)
	for _, depth := range []int{129, 100_000} {
		for name, value := range map[string]string{
			"list": nestedList(depth),
			"map":  nestedMap(depth),
		} {
			params := `{"p": ` + value + `}`
			_, err := db.ExecuteWithParams("RETURN $p AS p", params)
			if err == nil {
				t.Errorf("a %s %d deep: accepted, want an error", name, depth)
				continue
			}
			if _, err := db.Execute("RETURN 1 AS one"); err != nil {
				t.Fatalf("a %s %d deep: the database no longer answers: %v", name, depth, err)
			}
		}
	}
}

func TestParametersWithinTheLimitWork(t *testing.T) {
	db := mustOpen(t)
	// The parameter object is one level, so its value may nest 127 more.
	for name, value := range map[string]string{
		"list": nestedList(100),
		"map":  nestedMap(100),
	} {
		result, err := db.ExecuteWithParams("RETURN $p = $p AS same", `{"p": `+value+`}`)
		if err != nil {
			t.Fatalf("a %s 100 deep: %v", name, err)
		}
		if len(result.Rows) != 1 {
			t.Fatalf("a %s 100 deep: %d rows", name, len(result.Rows))
		}
	}
}
