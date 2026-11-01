package grafeo

import (
	"encoding"
	"encoding/base64"
	"encoding/json"
	"math"
	"reflect"
	"strconv"
	"strings"
	"time"
)

// marshalJSON encodes v as encoding/json does, but writes every []byte as
// the {"$bytes": "..."} tag, every time.Time as a TIMESTAMP or ZONED DATETIME
// tag (see timestampTag) where encoding/json writes its text, a STRING, every
// time.Duration as the {"$duration": "..."} tag where encoding/json writes its
// nanoseconds, an INT64, and every float64 and float32 as a JSON float: encoding/json writes float64(3) as "3", which
// Grafeo reads as an INT64, and float64(1<<63) as "9223372036854775808", a
// DECIMAL. A whole float keeps a fraction or an exponent ("3.0",
// "9.223372036854776e+18") and arrives as a FLOAT64, alone and inside lists
// and maps. A pointer to any of these arrives as its value would, and a nil
// one as null.
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
	case []byte:
		// encoding/json writes []byte as a base64 string, which Grafeo reads
		// as a STRING; the $bytes tag arrives as BYTES. A nil slice stays
		// null, as encoding/json writes it.
		if x == nil {
			return nil
		}
		return map[string]string{"$bytes": base64.StdEncoding.EncodeToString(x)}
	case time.Time:
		return timestampTag(x)
	case *time.Time:
		// Before json.Marshaler, which *time.Time is: the value's tag, as
		// for a time.Time
		if x == nil {
			return nil
		}
		return timestampTag(*x)
	case time.Duration:
		return DurationOf(x)
	case json.Marshaler:
		// A value that writes its own JSON (Decimal, json.RawMessage)
		return v
	case encoding.TextMarshaler:
		// A value encoding/json writes as its text (uuid.UUID, net.IP), which
		// the typed containers below would take apart into numbers or bytes
		return v
	default:
		return typedContainer(v)
	}
}

// typedContainer is v with its elements' floats and byte slices tagged when
// v is a typed slice, array or string-keyed map ([][]byte, map[string][]byte,
// []float64 inside a struct-free container), which the cases above do not
// name, and the value a pointer points to (*time.Duration, *float64), so that
// it arrives as the type the value would; any other value is v.
func typedContainer(v any) any {
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Pointer:
		if rv.IsNil() {
			return nil
		}
		return jsonFloats(rv.Elem().Interface())
	case reflect.Slice:
		if rv.IsNil() {
			return nil
		}
		if rv.Type().Elem().Kind() == reflect.Uint8 {
			// A named byte slice type: bytes, as []byte is
			return jsonFloats(rv.Bytes())
		}
		fallthrough
	case reflect.Array:
		s := make([]any, rv.Len())
		for i := range s {
			s[i] = jsonFloats(rv.Index(i).Interface())
		}
		return s
	case reflect.Map:
		if rv.Type().Key().Kind() != reflect.String {
			return v
		}
		if rv.IsNil() {
			return nil
		}
		m := make(map[string]any, rv.Len())
		iter := rv.MapRange()
		for iter.Next() {
			m[iter.Key().String()] = jsonFloats(iter.Value().Interface())
		}
		return m
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
