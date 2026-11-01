package grafeo

import (
	"encoding/json"
	"math"
	"strconv"
	"strings"
)

// marshalJSON encodes v as encoding/json does, but writes every float64 and
// float32 as a JSON float: encoding/json writes float64(3) as "3", which
// Grafeo reads as an INT64, and float64(1<<63) as "9223372036854775808", a
// DECIMAL. A whole float keeps a fraction or an exponent ("3.0",
// "9.223372036854776e+18") and arrives as a FLOAT64, alone and inside lists
// and maps.
func marshalJSON(v any) ([]byte, error) {
	return json.Marshal(jsonFloats(v))
}

// jsonFloats is v with each float replaced by its JSON float text.
func jsonFloats(v any) any {
	switch x := v.(type) {
	case float64:
		return floatNumber(x)
	case float32:
		return floatNumber(float64(x))
	case map[string]any:
		m := make(map[string]any, len(x))
		for k, e := range x {
			m[k] = jsonFloats(e)
		}
		return m
	case []any:
		s := make([]any, len(x))
		for i, e := range x {
			s[i] = jsonFloats(e)
		}
		return s
	case []float64:
		s := make([]any, len(x))
		for i, e := range x {
			s[i] = floatNumber(e)
		}
		return s
	case []float32:
		s := make([]any, len(x))
		for i, e := range x {
			s[i] = floatNumber(float64(e))
		}
		return s
	default:
		return v
	}
}

// floatNumber is f as a JSON number that reads back as a float; NaN and the
// infinities, which JSON cannot write, as the {"$float": "NaN"} tag.
func floatNumber(f float64) any {
	switch {
	case math.IsNaN(f):
		return map[string]string{"$float": "NaN"}
	case math.IsInf(f, 1):
		return map[string]string{"$float": "Infinity"}
	case math.IsInf(f, -1):
		return map[string]string{"$float": "-Infinity"}
	}
	text := strconv.FormatFloat(f, 'g', -1, 64)
	if !strings.ContainsAny(text, ".eE") {
		text += ".0"
	}
	return json.Number(text)
}
