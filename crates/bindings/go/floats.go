package grafeo

import (
	"encoding"
	"encoding/base64"
	"encoding/json"
	"errors"
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
	w := walker{path: map[container]struct{}{}}
	return w.value(v)
}

// container identifies a map, slice or pointer by what it holds: a value
// that holds itself meets the same container again on its own path.
type container struct {
	kind reflect.Kind
	ptr  uintptr
	len  int
}

// walker tracks the containers on the path from the root to the value it
// is at, so a value that holds itself stops at its first repeat instead of
// being walked without end (and branching at every level when it holds
// itself twice). Pointers are followed as encoding/json follows them, with
// no limit on how many there are.
type walker struct {
	path map[container]struct{}
}

// cycle stands in for a value that holds itself: json.Marshal fails on it,
// as it fails on such a value itself.
type cycle struct{}

func (cycle) MarshalJSON() ([]byte, error) {
	return nil, errors.New("grafeo: a parameter value holds itself")
}

// enter marks rv as on the path, or reports that it already is. An empty
// map or slice holds nothing, so it is not tracked.
func (w *walker) enter(rv reflect.Value) (container, bool) {
	key := container{kind: rv.Kind(), ptr: rv.Pointer()}
	if rv.Kind() != reflect.Pointer {
		key.len = rv.Len()
		if key.len == 0 {
			return key, true
		}
	}
	if _, seen := w.path[key]; seen {
		return key, false
	}
	w.path[key] = struct{}{}
	return key, true
}

func (w *walker) leave(key container) {
	delete(w.path, key)
}

// value is v with each float replaced by its JSON float text.
func (w *walker) value(v any) any {
	switch x := v.(type) {
	case float64:
		return floatNumber(x)
	case float32:
		return floatNumber(float64(x))
	case map[string]any:
		key, ok := w.enter(reflect.ValueOf(x))
		if !ok {
			return cycle{}
		}
		defer w.leave(key)
		m := make(map[string]any, len(x))
		for k, e := range x {
			m[k] = w.value(e)
		}
		return m
	case []any:
		key, ok := w.enter(reflect.ValueOf(x))
		if !ok {
			return cycle{}
		}
		defer w.leave(key)
		s := make([]any, len(x))
		for i, e := range x {
			s[i] = w.value(e)
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
		return w.typedContainer(v)
	}
}

// typedContainer is v with its elements' floats and byte slices tagged when
// v is a typed slice, array or string-keyed map ([][]byte, map[string][]byte,
// []float64 inside a struct-free container), which the cases above do not
// name, and the value a pointer points to (*time.Duration, *float64), so that
// it arrives as the type the value would; any other value is v.
func (w *walker) typedContainer(v any) any {
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Pointer:
		if rv.IsNil() {
			return nil
		}
		key, ok := w.enter(rv)
		if !ok {
			return cycle{}
		}
		defer w.leave(key)
		return w.value(rv.Elem().Interface())
	case reflect.Slice:
		if rv.IsNil() {
			return nil
		}
		if rv.Type().Elem().Kind() == reflect.Uint8 {
			// A named byte slice type: bytes, as []byte is
			return w.value(rv.Bytes())
		}
		key, ok := w.enter(rv)
		if !ok {
			return cycle{}
		}
		defer w.leave(key)
		return w.elements(rv)
	case reflect.Array:
		return w.elements(rv)
	case reflect.Map:
		if rv.Type().Key().Kind() != reflect.String {
			return v
		}
		if rv.IsNil() {
			return nil
		}
		key, ok := w.enter(rv)
		if !ok {
			return cycle{}
		}
		defer w.leave(key)
		m := make(map[string]any, rv.Len())
		iter := rv.MapRange()
		for iter.Next() {
			m[iter.Key().String()] = w.value(iter.Value().Interface())
		}
		return m
	default:
		return v
	}
}

// elements is the walked elements of the slice or array rv.
func (w *walker) elements(rv reflect.Value) []any {
	s := make([]any, rv.Len())
	for i := range s {
		s[i] = w.value(rv.Index(i).Interface())
	}
	return s
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
