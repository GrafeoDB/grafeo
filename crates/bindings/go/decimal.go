package grafeo

import (
	"encoding/json"
	"errors"
	"math/big"
	"regexp"
)

// Decimal is an exact decimal from Grafeo: up to 38 digits, kept as its text
// with the scale it was written with ("1.50"). Compare decimals with Equal,
// which ignores the scale; == compares the text. As a query parameter it is
// passed exactly.
type Decimal struct {
	text string
}

var decimalPattern = regexp.MustCompile(`^[+-]?(\d+(\.\d*)?|\.\d+)$`)

// ParseDecimal reads [+-]digits[.digits].
func ParseDecimal(text string) (Decimal, error) {
	if !decimalPattern.MatchString(text) {
		return Decimal{}, errors.New("grafeo: not a decimal number: " + text)
	}
	return Decimal{text: text}, nil
}

// String returns the text with its scale, as Grafeo writes it.
func (d Decimal) String() string {
	if d.text == "" {
		return "0"
	}
	return d.text
}

// Rat returns the exact value as a rational number.
func (d Decimal) Rat() *big.Rat {
	r, ok := new(big.Rat).SetString(d.String())
	if !ok {
		return new(big.Rat)
	}
	return r
}

// Equal reports whether two decimals have the same value, whatever their scale.
func (d Decimal) Equal(other Decimal) bool {
	return d.Rat().Cmp(other.Rat()) == 0
}

// MarshalJSON writes the {"$decimal": "1.50"} tag the engine reads.
func (d Decimal) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]string{"$decimal": d.String()})
}

// decodeTags turns every {"$decimal": "..."} map (the tag as its only key)
// inside value into a Decimal, and every {"$term": {...}} map into an
// RdfTerm, through lists and maps.
func decodeTags(value any) any {
	switch v := value.(type) {
	case map[string]any:
		if text, ok := v["$decimal"].(string); ok && len(v) == 1 {
			if d, err := ParseDecimal(text); err == nil {
				return d
			}
		}
		if term, ok := rdfTermOf(v); ok {
			return term
		}
		for key, item := range v {
			v[key] = decodeTags(item)
		}
		return v
	case []any:
		for i, item := range v {
			v[i] = decodeTags(item)
		}
		return v
	default:
		return value
	}
}
