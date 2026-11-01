package grafeo

import (
	"bytes"
	"encoding/json"
	"math"
	"strconv"
	"testing"
)

// tagged holds a stored BYTES, DECIMAL and NaN, as every read must return
// them: a []byte, a Decimal and a float64 NaN, never the {"$bytes": ...},
// {"$decimal": ...} or {"$float": ...} map they travel in.
type tagged struct {
	data   []byte
	amount Decimal
}

func newTagged(t *testing.T) tagged {
	t.Helper()
	amount, err := ParseDecimal("12.50")
	if err != nil {
		t.Fatal(err)
	}
	return tagged{data: []byte{0, 1, 2, 255}, amount: amount}
}

// check fails unless properties hold the three values decoded.
func (want tagged) check(t *testing.T, where string, properties map[string]any) {
	t.Helper()
	if data, ok := properties["data"].([]byte); !ok || !bytes.Equal(data, want.data) {
		t.Errorf("%s: data came back as %#v", where, properties["data"])
	}
	if amount, ok := properties["amount"].(Decimal); !ok || !amount.Equal(want.amount) {
		t.Errorf("%s: amount came back as %#v", where, properties["amount"])
	}
	if ratio, ok := properties["ratio"].(float64); !ok || !math.IsNaN(ratio) {
		t.Errorf("%s: ratio came back as %#v", where, properties["ratio"])
	}
}

// idOf reads an id(...) column, a json.Number.
func idOf(t *testing.T, value any) uint64 {
	t.Helper()
	number, ok := value.(json.Number)
	if !ok {
		t.Fatalf("an id came back as %#v", value)
	}
	id, err := strconv.ParseUint(number.String(), 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	return id
}

// storeTagged stores a node and an edge with the three values and returns
// their ids.
func storeTagged(t *testing.T, db *Database, want tagged) (node, edge uint64) {
	t.Helper()
	result, err := db.ExecuteParams(
		"INSERT (a:Blob {name: 'Mia', data: $data, amount: $amount, ratio: $ratio})"+
			"-[e:HOLDS {data: $data, amount: $amount, ratio: $ratio}]->(:Blob {name: 'Butch'}) "+
			"RETURN id(a) AS node, id(e) AS edge",
		map[string]any{"data": want.data, "amount": want.amount, "ratio": math.NaN()},
	)
	if err != nil {
		t.Fatal(err)
	}
	return idOf(t, result.Rows[0]["node"]), idOf(t, result.Rows[0]["edge"])
}

// GetNode and GetEdge decode the tags in the properties they return.
func TestGetNodeAndGetEdgeDecodeTags(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	want := newTagged(t)
	nodeID, edgeID := storeTagged(t, db, want)

	node, err := db.GetNode(nodeID)
	if err != nil {
		t.Fatal(err)
	}
	want.check(t, "GetNode", node.Properties)
	if node.Properties["name"] != "Mia" {
		t.Errorf("GetNode: name came back as %#v", node.Properties["name"])
	}
	edge, err := db.GetEdge(edgeID)
	if err != nil {
		t.Fatal(err)
	}
	want.check(t, "GetEdge", edge.Properties)
}

// A stream's rows decode their tags, as Execute's do.
func TestStreamRowsDecodeTags(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	want := newTagged(t)
	storeTagged(t, db, want)

	const query = "MATCH (n:Blob {name: 'Mia'}) RETURN n.data AS data, n.amount AS amount, n.ratio AS ratio"
	result, err := db.Execute(query)
	if err != nil {
		t.Fatal(err)
	}
	want.check(t, "Execute", result.Rows[0])

	stream, err := db.ExecuteStream(query)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	row, err := stream.Next()
	if err != nil || row == nil {
		t.Fatalf("stream: %v %v", row, err)
	}
	want.check(t, "stream", row)

	// A list of bytes inside a streamed row is decoded too.
	listStream, err := db.ExecuteStream("MATCH (n:Blob {name: 'Mia'}) RETURN [n.data, n.data] AS both")
	if err != nil {
		t.Fatal(err)
	}
	defer listStream.Close()
	row, err = listStream.Next()
	if err != nil || row == nil {
		t.Fatalf("list stream: %v %v", row, err)
	}
	both, ok := row["both"].([]any)
	if !ok || len(both) != 2 {
		t.Fatalf("the list came back as %#v", row["both"])
	}
	for _, item := range both {
		if data, ok := item.([]byte); !ok || !bytes.Equal(data, want.data) {
			t.Errorf("a list item came back as %#v", item)
		}
	}
}

// The reads of an AsOfView decode their tags: GetNode, GetEdge,
// GetNodeProperty and its query results.
func TestAsOfReadsDecodeTags(t *testing.T) {
	db, err := OpenInMemoryWithHistory(HistoryForever)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	want := newTagged(t)
	nodeID, edgeID := storeTagged(t, db, want)
	then := db.CurrentEpoch()
	// Later values, which the view must not see.
	if _, err := db.Execute("MATCH (n:Blob {name: 'Mia'})-[e:HOLDS]->() SET n.data = NULL, e.data = NULL"); err != nil {
		t.Fatal(err)
	}

	view, err := db.AsOf(AtEpoch(then))
	if err != nil {
		t.Fatal(err)
	}
	defer view.Close()

	node, err := view.GetNode(nodeID)
	if err != nil {
		t.Fatal(err)
	}
	want.check(t, "AsOfView.GetNode", node.Properties)
	edge, err := view.GetEdge(edgeID)
	if err != nil {
		t.Fatal(err)
	}
	want.check(t, "AsOfView.GetEdge", edge.Properties)

	properties := map[string]any{}
	for _, key := range []string{"data", "amount", "ratio"} {
		value, err := view.GetNodeProperty(nodeID, key)
		if err != nil {
			t.Fatal(err)
		}
		properties[key] = value
	}
	want.check(t, "AsOfView.GetNodeProperty", properties)

	result, err := view.Execute("MATCH (n:Blob {name: 'Mia'}) RETURN n.data AS data, n.amount AS amount, n.ratio AS ratio")
	if err != nil {
		t.Fatal(err)
	}
	want.check(t, "AsOfView.Execute", result.Rows[0])
	result, err = db.ExecuteAsOf(
		"MATCH (n:Blob {name: 'Mia'}) RETURN n.data AS data, n.amount AS amount, n.ratio AS ratio",
		AtEpoch(then), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	want.check(t, "ExecuteAsOf", result.Rows[0])
}

// A one-column row named "$bytes", "$decimal" or "$float" holding a matching
// string is a row, not a tag: only values are decoded, never the row.
func TestARowNamedLikeATagStaysARow(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	for _, c := range []struct{ column, text string }{
		{"$bytes", "AQID"},
		{"$decimal", "1.50"},
		{"$float", "NaN"},
	} {
		query := "RETURN '" + c.text + "' AS `" + c.column + "`"
		result, err := db.Execute(query)
		if err != nil {
			t.Fatalf("%s: %v", query, err)
		}
		if got := result.Rows[0][c.column]; got != c.text {
			t.Errorf("%s: the column came back as %#v", query, got)
		}
		stream, err := db.ExecuteStream(query)
		if err != nil {
			t.Fatalf("stream %s: %v", query, err)
		}
		row, err := stream.Next()
		stream.Close()
		if err != nil || row == nil || row[c.column] != c.text || len(row) != 1 {
			t.Errorf("stream %s: the row came back as %#v (%v)", query, row, err)
		}
	}
}

// Blob is a named byte slice type, which encoding/json writes as base64 as
// it does []byte.
type Blob []byte

// Typed containers of byte slices and floats tag what they hold, nested, as
// []any and map[string]any do: [][]byte, map[string][]byte, a named byte
// slice type, typed slices and maps of floats.
func TestTypedContainersTagTheirBytesAndFloats(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	result, err := db.ExecuteParams(
		"RETURN $list[0] IS TYPED BYTES AS listed, $list[1] IS TYPED BYTES AS empty, "+
			"$map.k IS TYPED BYTES AS mapped, $blob IS TYPED BYTES AS named, "+
			"$nested[0][0] IS TYPED BYTES AS nested, "+
			"$floats.x IS TYPED FLOAT64 AS whole, $floatList[0] IS TYPED FLOAT64 AS listedFloat, "+
			"$floatList[1] IS TYPED FLOAT64 AS nan, "+
			"$list AS list, $blob AS blob",
		map[string]any{
			"list":      [][]byte{{1, 2}, {}},
			"map":       map[string][]byte{"k": {3}},
			"blob":      Blob{4, 5},
			"nested":    [][][]byte{{{6}}},
			"floats":    map[string]float64{"x": 3},
			"floatList": []float32{2, float32(math.NaN())},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	row := result.Rows[0]
	for _, column := range []string{"listed", "empty", "mapped", "named", "nested", "whole", "listedFloat", "nan"} {
		if row[column] != true {
			t.Errorf("%s: %#v", column, row[column])
		}
	}
	list, ok := row["list"].([]any)
	if !ok || len(list) != 2 {
		t.Fatalf("the list came back as %#v", row["list"])
	}
	if first, ok := list[0].([]byte); !ok || !bytes.Equal(first, []byte{1, 2}) {
		t.Errorf("the first item came back as %#v", list[0])
	}
	if blob, ok := row["blob"].([]byte); !ok || !bytes.Equal(blob, []byte{4, 5}) {
		t.Errorf("the named slice came back as %#v", row["blob"])
	}
}

// A value that writes its own JSON passes unchanged: a json.RawMessage
// parameter is its JSON, and a Decimal inside a typed map is a DECIMAL.
func TestJSONMarshalersPassUnchanged(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	amount, err := ParseDecimal("1.50")
	if err != nil {
		t.Fatal(err)
	}
	result, err := db.ExecuteParams(
		"RETURN $raw.k[1] AS second, $raw.s AS s, $amounts.a IS TYPED DECIMAL AS decimal",
		map[string]any{
			"raw":     json.RawMessage(`{"k": [1, 2], "s": "Paris"}`),
			"amounts": map[string]Decimal{"a": amount},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	row := result.Rows[0]
	if second, ok := row["second"].(json.Number); !ok || second.String() != "2" {
		t.Errorf("the raw message's list came back as %#v", row["second"])
	}
	if row["s"] != "Paris" {
		t.Errorf("the raw message's string came back as %#v", row["s"])
	}
	if row["decimal"] != true {
		t.Errorf("a Decimal in a typed map: %#v", row["decimal"])
	}
}

// A $bytes map whose value is not a string stays a map, as a malformed
// string does (see TestBytesLookalikesStayWhatTheyAre).
func TestANumericBytesMapStaysAMap(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	for _, value := range []any{7, true, nil, []any{"AQID"}} {
		result, err := db.ExecuteParams(
			"RETURN $m AS m, $m IS TYPED BYTES AS bytes",
			map[string]any{"m": map[string]any{"$bytes": value}},
		)
		if err != nil {
			t.Fatalf("%#v: %v", value, err)
		}
		row := result.Rows[0]
		back, ok := row["m"].(map[string]any)
		if !ok || row["bytes"] != false || len(back) != 1 {
			t.Errorf("{$bytes: %#v} came back as %#v (bytes %v)", value, row["m"], row["bytes"])
			continue
		}
		if _, isBytes := back["$bytes"].([]byte); isBytes {
			t.Errorf("{$bytes: %#v}: the value was decoded as bytes", value)
		}
	}
}
