package grafeo

import (
	"encoding/json"
	"math"
	"testing"
)

// A whole float64 parameter is a FLOAT64: encoding/json writes float64(3) as
// "3", which read as an INT64, and float64(1<<63) as "9223372036854775808",
// which read as a DECIMAL. Each must arrive as a float with its value, alone
// and inside a list and a map.
func TestAWholeFloatParameterStaysAFloat(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	for _, value := range []float64{0, 3, -20, 1e6, math.Pow(2, 53), math.Pow(2, 63), -math.Pow(2, 63), 1e300} {
		result, err := db.ExecuteParams(
			"RETURN $x AS x, $x IS TYPED FLOAT64 AS float, $x IS TYPED INT64 AS int",
			map[string]any{"x": value},
		)
		if err != nil {
			t.Fatalf("%v: %v", value, err)
		}
		row := result.Rows[0]
		if row["float"] != true || row["int"] != false {
			t.Fatalf("%v arrived as %#v (float %v, int %v)", value, row["x"], row["float"], row["int"])
		}
		number, ok := row["x"].(json.Number)
		back, err := number.Float64()
		if !ok || err != nil || back != value {
			t.Fatalf("%v came back as %#v", value, row["x"])
		}
	}
	// Inside a list and a map.
	result, err := db.ExecuteParams(
		"RETURN $xs[0] IS TYPED FLOAT64 AS first, $m.w IS TYPED FLOAT64 AS w",
		map[string]any{"xs": []any{float64(3), 1.5}, "m": map[string]any{"w": float64(20)}},
	)
	if err != nil {
		t.Fatal(err)
	}
	row := result.Rows[0]
	if row["first"] != true || row["w"] != true {
		t.Fatalf("nested whole floats: first %v, w %v", row["first"], row["w"])
	}
	// An int parameter stays an INT64.
	result, err = db.ExecuteParams("RETURN $x IS TYPED INT64 AS int", map[string]any{"x": 3})
	if err != nil || result.Rows[0]["int"] != true {
		t.Fatalf("an int stays an INT64: %v %v", result, err)
	}
}
