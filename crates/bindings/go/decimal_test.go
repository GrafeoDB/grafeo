package grafeo

import (
	"encoding/json"
	"testing"
)

// A decimal parameter comes back from arithmetic as a Decimal with its scale.
func TestDecimalParameterRoundTrip(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	a, _ := ParseDecimal("0.10")
	b, _ := ParseDecimal("0.2")
	result, err := db.ExecuteParams("RETURN $a + $b AS total", map[string]any{"a": a, "b": b})
	if err != nil {
		t.Fatal(err)
	}
	total, ok := result.Rows[0]["total"].(Decimal)
	if !ok {
		t.Fatalf("expected a Decimal, got %T %v", result.Rows[0]["total"], result.Rows[0]["total"])
	}
	if total.String() != "0.30" {
		t.Errorf("expected 0.30, got %s", total)
	}
	if total.Rat().FloatString(2) != "0.30" {
		t.Errorf("expected the rational 0.30, got %s", total.Rat().FloatString(2))
	}
}

// Decimals compare by value whatever their scale.
func TestDecimalEqualityIgnoresScale(t *testing.T) {
	x, _ := ParseDecimal("1.50")
	y, _ := ParseDecimal("1.5")
	z, _ := ParseDecimal("1.05")
	if !x.Equal(y) {
		t.Error("1.50 should equal 1.5")
	}
	if x.Equal(z) {
		t.Error("1.50 should not equal 1.05")
	}
}

// Text that is not a decimal is refused; the tag writes the text as given.
func TestDecimalParsingAndEncoding(t *testing.T) {
	for _, bad := range []string{"", "1.2.3", "1e5", "abc"} {
		if _, err := ParseDecimal(bad); err == nil {
			t.Errorf("%q should be refused", bad)
		}
	}
	d, _ := ParseDecimal("-12.50")
	data, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `{"$decimal":"-12.50"}` {
		t.Errorf("unexpected encoding %s", data)
	}
}

// Only the tag alone is a decimal; a map with other keys stays a map, and
// tags inside lists and maps are decoded too.
func TestDecodeDecimalTags(t *testing.T) {
	decoded := decodeTags([]any{
		map[string]any{"$decimal": "1.5"},
		map[string]any{"$decimal": "1", "other": json.Number("2")},
		map[string]any{"inner": map[string]any{"$decimal": "3"}},
	})
	items := decoded.([]any)
	if d, ok := items[0].(Decimal); !ok || d.String() != "1.5" {
		t.Errorf("expected Decimal 1.5, got %T %v", items[0], items[0])
	}
	if _, ok := items[1].(map[string]any); !ok {
		t.Errorf("expected a map, got %T", items[1])
	}
	inner := items[2].(map[string]any)["inner"]
	if d, ok := inner.(Decimal); !ok || d.String() != "3" {
		t.Errorf("expected Decimal 3 inside the map, got %T %v", inner, inner)
	}
}
