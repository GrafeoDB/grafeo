package grafeo

import (
	"bytes"
	"testing"
)

// allBytes is every byte value once, 0 to 255.
func allBytes() []byte {
	b := make([]byte, 256)
	for i := range b {
		b[i] = byte(i)
	}
	return b
}

// A []byte parameter travels in the {"$bytes": "..."} tag and arrives as
// BYTES (encoding/json wrote a base64 string, a STRING), and BYTES come back
// as a []byte with every byte equal: a few bytes, none, all 256 values.
func TestBytesRoundTrip(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	for _, value := range [][]byte{{0, 1, 2, 255}, {}, allBytes()} {
		result, err := db.ExecuteParams(
			"RETURN $b AS b, $b IS TYPED BYTES AS bytes, $b IS TYPED STRING AS str",
			map[string]any{"b": value},
		)
		if err != nil {
			t.Fatalf("%d bytes: %v", len(value), err)
		}
		row := result.Rows[0]
		back, ok := row["b"].([]byte)
		if !ok || row["bytes"] != true || row["str"] != false {
			t.Fatalf("%d bytes came back as %#v (bytes %v, string %v)", len(value), row["b"], row["bytes"], row["str"])
		}
		if !bytes.Equal(back, value) {
			t.Fatalf("%d bytes came back as %v", len(value), back)
		}
	}
}

// Stored as a property and read back, inside a list and a map too.
func TestStoredBytesComeBackAsBytes(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	all := allBytes()
	if _, err := db.ExecuteParams(
		"INSERT (:Blob {name: 'Mia', data: $b, parts: $l})",
		map[string]any{"b": all, "l": []any{[]byte{7}, []byte{}}},
	); err != nil {
		t.Fatal(err)
	}
	result, err := db.Execute("MATCH (n:Blob) RETURN n.data AS data, n.parts AS parts")
	if err != nil {
		t.Fatal(err)
	}
	row := result.Rows[0]
	if data, ok := row["data"].([]byte); !ok || !bytes.Equal(data, all) {
		t.Fatalf("the stored bytes came back as %#v", row["data"])
	}
	parts, ok := row["parts"].([]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("the list came back as %#v", row["parts"])
	}
	if first, ok := parts[0].([]byte); !ok || !bytes.Equal(first, []byte{7}) {
		t.Fatalf("the first part came back as %#v", parts[0])
	}
	if second, ok := parts[1].([]byte); !ok || len(second) != 0 {
		t.Fatalf("the empty part came back as %#v", parts[1])
	}

	result, err = db.ExecuteParams(
		"RETURN $m.k IS TYPED BYTES AS inner",
		map[string]any{"m": map[string]any{"k": []byte{1, 2}}},
	)
	if err != nil || result.Rows[0]["inner"] != true {
		t.Fatalf("bytes inside a map: %v %v", result, err)
	}
}

// A string that looks like base64 stays a string, and a $bytes map that is
// not base64, or has a second key, stays a map.
func TestBytesLookalikesStayWhatTheyAre(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	result, err := db.ExecuteParams(
		"RETURN $s AS s, $s IS TYPED STRING AS str",
		map[string]any{"s": "AAEC/w=="},
	)
	if err != nil || result.Rows[0]["s"] != "AAEC/w==" || result.Rows[0]["str"] != true {
		t.Fatalf("a base64-looking string: %v %v", result, err)
	}
	for _, m := range []map[string]any{
		{"$bytes": "not base64!"},
		{"$bytes": "AAEC/w"},
		{"$bytes": "AAEC_w=="},
		{"$bytes": "AAEC/x=="},
		{"$bytes": "AAEC /w=="},
		{"$bytes": "AAEC\n/w=="},
		{"$bytes": "AAEC/w==", "name": "Gus"},
	} {
		result, err := db.ExecuteParams(
			"RETURN $m AS m, $m IS TYPED BYTES AS bytes",
			map[string]any{"m": m},
		)
		if err != nil {
			t.Fatalf("%v: %v", m, err)
		}
		row := result.Rows[0]
		back, ok := row["m"].(map[string]any)
		if !ok || row["bytes"] != false || len(back) != len(m) || back["$bytes"] != m["$bytes"] {
			t.Errorf("%q came back as %#v (bytes %v)", m["$bytes"], row["m"], row["bytes"])
		}
	}
}

// A nil []byte is null, as encoding/json writes it; an empty slice is an
// empty BYTES.
func TestNilBytesIsNullAndEmptyBytesIsBytes(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var nilBytes []byte
	result, err := db.ExecuteParams(
		"RETURN $n IS NULL AS null, $e IS TYPED BYTES AS bytes, $e AS e",
		map[string]any{"n": nilBytes, "e": []byte{}},
	)
	if err != nil {
		t.Fatal(err)
	}
	row := result.Rows[0]
	if row["null"] != true {
		t.Errorf("a nil []byte arrives as null, got %#v", row["null"])
	}
	if e, ok := row["e"].([]byte); row["bytes"] != true || !ok || len(e) != 0 {
		t.Errorf("an empty []byte arrives as empty BYTES, got %#v (bytes %v)", row["e"], row["bytes"])
	}
}
