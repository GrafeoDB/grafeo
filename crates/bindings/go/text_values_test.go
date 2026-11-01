package grafeo

import (
	"encoding/hex"
	"encoding/json"
	"net"
	"testing"
)

// shortID is a fixed-size byte array that encodes itself as text, as
// uuid.UUID does.
type shortID [4]byte

func (id shortID) MarshalText() ([]byte, error) {
	return []byte(hex.EncodeToString(id[:])), nil
}

// A parameter whose type writes itself as text (encoding.TextMarshaler)
// arrives as that text, alone and inside a map and a list: a named byte
// array is not a list of numbers and a net.IP is not BYTES.
func TestTextMarshalersArriveAsTheirText(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	ip := net.ParseIP("10.0.0.1").To4()
	id := shortID{0xde, 0xad, 0xbe, 0xef}
	result, err := db.ExecuteParams(
		"RETURN $ip AS ip, $id AS id, $ip IS TYPED STRING AS ip_text, $id IS TYPED STRING AS id_text, "+
			"$m.ip AS in_map, $l[0] AS in_list",
		map[string]any{"ip": ip, "id": id, "m": map[string]any{"ip": ip}, "l": []any{id}},
	)
	if err != nil {
		t.Fatal(err)
	}
	row := result.Rows[0]
	for column, want := range map[string]any{
		"ip": "10.0.0.1", "id": "deadbeef", "ip_text": true, "id_text": true,
		"in_map": "10.0.0.1", "in_list": "deadbeef",
	} {
		if row[column] != want {
			t.Errorf("%s: got %#v, want %#v", column, row[column], want)
		}
	}
}

// A streamed row gives its numbers as json.Number, as Execute does, so an
// integer above 2^53 keeps every digit.
func TestStreamedNumbersKeepEveryDigit(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	const query = "RETURN 9007199254740993 AS big, 1.5e0 AS half"
	materialized, err := db.Execute(query)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := db.ExecuteStream(query)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	rows, err := stream.Collect()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("got %d streamed rows", len(rows))
	}
	for column, want := range map[string]json.Number{"big": "9007199254740993", "half": "1.5"} {
		if rows[0][column] != want {
			t.Errorf("streamed %s: got %#v, want %#v", column, rows[0][column], want)
		}
		if materialized.Rows[0][column] != want {
			t.Errorf("executed %s: got %#v, want %#v", column, materialized.Rows[0][column], want)
		}
	}
}
