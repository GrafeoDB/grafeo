package grafeo

import (
	"math"
	"testing"
)

// NaN and the infinities, which JSON has no number for (json.Marshal refuses
// them), travel as {"$float": "NaN"} and come back as float64 NaN and the
// infinities; the string "NaN" stays a string.
func TestNanAndTheInfinitiesRoundTrip(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		result, err := db.ExecuteParams(
			"RETURN $x AS x, $x IS TYPED FLOAT64 AS float",
			map[string]any{"x": value},
		)
		if err != nil {
			t.Fatalf("%v: %v", value, err)
		}
		row := result.Rows[0]
		back, ok := row["x"].(float64)
		if !ok || row["float"] != true {
			t.Fatalf("%v came back as %#v (float %v)", value, row["x"], row["float"])
		}
		if !(math.IsNaN(value) && math.IsNaN(back)) && back != value {
			t.Fatalf("%v came back as %v", value, back)
		}
	}
	// Inside a list.
	result, err := db.ExecuteParams(
		"RETURN $xs[0] IS TYPED FLOAT64 AS first, $xs[1] AS second",
		map[string]any{"xs": []any{math.NaN(), math.Inf(1)}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Rows[0]["first"] != true || result.Rows[0]["second"] != math.Inf(1) {
		t.Fatalf("a list of special floats: %#v", result.Rows[0])
	}
}

func TestTheStringNanStaysAString(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	for _, text := range []string{"NaN", "Infinity", "-Infinity"} {
		result, err := db.ExecuteParams(
			"RETURN $x AS x, $x IS TYPED STRING AS str",
			map[string]any{"x": text},
		)
		if err != nil {
			t.Fatalf("%v: %v", text, err)
		}
		if row := result.Rows[0]; row["x"] != text || row["str"] != true {
			t.Fatalf("%q came back as %#v (string %v)", text, row["x"], row["str"])
		}
	}
}

// A whole float32 is a FLOAT64 too.
func TestAWholeFloat32StaysAFloat(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	result, err := db.ExecuteParams("RETURN $x IS TYPED FLOAT64 AS float", map[string]any{"x": float32(3)})
	if err != nil || result.Rows[0]["float"] != true {
		t.Fatalf("float32(3): %v %v", result, err)
	}
}
