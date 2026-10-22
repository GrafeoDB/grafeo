package grafeo

import (
	"encoding/json"
	"math/big"
	"strings"
	"testing"
)

// An integer parameter or property past 64 bits is an exact decimal, up to
// 38 digits: one that fits 64 bits stays INT64, one past it but within 38
// digits (a uint64 or a *big.Int) arrives as a DECIMAL with scale 0 (it used
// to be a float that lost its last digits), and one past 38 digits is an
// error naming the parameter and the value.
func TestAnIntegerPast64BitsIsAnExactDecimal(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	pow := func(base, exp int64) *big.Int {
		return new(big.Int).Exp(big.NewInt(base), big.NewInt(exp), nil)
	}
	tooWide := pow(10, 38)
	largest := new(big.Int).Sub(tooWide, big.NewInt(1))
	below := new(big.Int).Sub(new(big.Int).Neg(pow(2, 63)), big.NewInt(1))
	for _, value := range []any{uint64(1) << 63, ^uint64(0), pow(2, 64), below, largest} {
		result, err := db.ExecuteParams(
			"RETURN $x AS x, $x IS TYPED INT64 AS int, $x IS TYPED DECIMAL AS decimal",
			map[string]any{"x": value},
		)
		if err != nil {
			t.Fatalf("%v: %v", value, err)
		}
		row := result.Rows[0]
		decimal, ok := row["x"].(Decimal)
		if !ok || decimal.String() != toDigits(value) {
			t.Fatalf("%v came back as %#v", value, row["x"])
		}
		if row["int"] != false || row["decimal"] != true {
			t.Fatalf("%v: int %v, decimal %v", value, row["int"], row["decimal"])
		}
	}

	result, err := db.ExecuteParams("RETURN $x IS TYPED INT64 AS int", map[string]any{"x": uint64(1<<63 - 1)})
	if err != nil || result.Rows[0]["int"] != true {
		t.Fatalf("a uint64 that fits 64 bits: %v %v", result, err)
	}

	_, err = db.ExecuteParams("RETURN $x AS x", map[string]any{"x": tooWide})
	if err == nil || !strings.Contains(err.Error(), "parameter 'x'") ||
		!strings.Contains(err.Error(), tooWide.String()) || !strings.Contains(err.Error(), "38 digits") {
		t.Fatalf("expected an error naming the parameter and the value, got %v", err)
	}

	if _, err := db.CreateNode([]string{"N"}, map[string]any{"v": int64(5)}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.CreateNode([]string{"N"}, map[string]any{"v": pow(2, 64)}); err != nil {
		t.Fatal(err)
	}
	result, err = db.Execute("MATCH (n:N) RETURN sum(n.v) AS total")
	if err != nil {
		t.Fatal(err)
	}
	if total, ok := result.Rows[0]["total"].(Decimal); !ok || total.String() != "18446744073709551621" {
		t.Fatalf("the sum is not exact: %#v", result.Rows[0]["total"])
	}
	if _, err := db.CreateNode([]string{"N"}, map[string]any{"v": tooWide}); err == nil ||
		!strings.Contains(err.Error(), "property 'v'") {
		t.Fatalf("expected an error naming the property, got %v", err)
	}
}

// A float parameter keeps its type and its value through the JSON the binding
// sends: JSON is read keeping each number's text, and a number with a
// fraction or an exponent is a FLOAT64 as before.
func TestAFloatParameterStaysAFloatWithItsValue(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	for _, value := range []float64{1.5, -0.25, 150.5, 2.5e300} {
		result, err := db.ExecuteParams(
			"RETURN $x AS x, $x IS TYPED FLOAT64 AS float",
			map[string]any{"x": value},
		)
		if err != nil {
			t.Fatalf("%v: %v", value, err)
		}
		row := result.Rows[0]
		number, ok := row["x"].(json.Number)
		back, err := number.Float64()
		if !ok || err != nil || back != value || row["float"] != true {
			t.Fatalf("%v came back as %#v (float %v)", value, row["x"], row["float"])
		}
	}
}

func toDigits(value any) string {
	switch v := value.(type) {
	case uint64:
		return new(big.Int).SetUint64(v).String()
	case *big.Int:
		return v.String()
	}
	return ""
}
