package grafeo

import (
	"encoding/json"
	"testing"
	"time"
)

// typedBack runs RETURN $v IS TYPED <gqlType>, $v and returns whether the
// parameter arrived as that type and the value it came back as.
func typedBack(t *testing.T, db *Database, gqlType string, value any) (bool, any) {
	t.Helper()
	result, err := db.ExecuteParams(
		"RETURN $v IS TYPED "+gqlType+" AS typed, $v AS back", map[string]any{"v": value})
	if err != nil {
		t.Fatalf("%s %#v: %v", gqlType, value, err)
	}
	typed, _ := result.Rows[0]["typed"].(bool)
	return typed, result.Rows[0]["back"]
}

// must is value, for a call that cannot fail here.
func must[T any](value T, err error) T {
	if err != nil {
		panic(err)
	}
	return value
}

// Each temporal value goes in as its GQL type and comes back as the Go value
// it went in as, and that value goes in again as the same type.
func TestTemporalValuesRoundTripAsTheirTypes(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	cet := time.FixedZone("CET", 3600)
	cases := []struct {
		gqlType string
		value   any
		same    func(back any) bool
	}{
		{"LOCAL DATETIME", time.Date(2024, 3, 5, 10, 30, 0, 123456000, time.UTC), nil},
		{"TIMESTAMP", time.Date(1969, 7, 20, 20, 17, 40, 654321000, time.UTC), nil},
		{"ZONED DATETIME", time.Date(2024, 3, 5, 10, 30, 0, 250000000, cet), nil},
		{"ZONED DATETIME", time.Date(2024, 3, 5, 10, 30, 0, 0, time.FixedZone("", 0)), nil},
		{"ZONED DATETIME", time.Date(2024, 3, 5, 10, 30, 0, 0, time.FixedZone("", -(5*3600+1800))), nil},
		{"DATE", Date{Year: 2024, Month: time.February, Day: 29}, nil},
		{"DATE", Date{Year: -44, Month: time.March, Day: 15}, nil},
		{"TIME", must(NewLocalTime(14, 30, 15, 500000000)), nil},
		{"TIME", must(NewZonedTime(12, 30, 0, 0, 7200)), nil},
		{"TIME", must(NewZonedTime(23, 59, 59, 999999999, 0)), nil},
		{"DURATION", Duration{Months: 14, Days: 3, Nanos: 4*nanosPerHour + 5*nanosPerMinute + 6*nanosPerSecond + 5e8}, nil},
		{"DURATION", Duration{Months: -1, Days: -2, Nanos: -3}, nil},
		{"DURATION", Duration{Days: 1}, nil},
		{"DURATION", Duration{}, nil},
	}
	for _, c := range cases {
		typed, back := typedBack(t, db, c.gqlType, c.value)
		if !typed {
			t.Errorf("%#v did not arrive as %s", c.value, c.gqlType)
		}
		if !sameTemporal(c.value, back) {
			t.Errorf("%s %#v came back as %#v", c.gqlType, c.value, back)
			continue
		}
		again, _ := typedBack(t, db, c.gqlType, back)
		if !again {
			t.Errorf("%#v, read back, did not arrive as %s again", back, c.gqlType)
		}
	}
}

// sameTemporal reports whether back is want: a time.Time at the same instant
// with the same offset (a UTC one for UTC), or the same Date, Time or Duration.
func sameTemporal(want, back any) bool {
	wantTime, ok := want.(time.Time)
	if !ok {
		return want == back
	}
	backTime, ok := back.(time.Time)
	if !ok || !backTime.Equal(wantTime) {
		return false
	}
	_, wantOffset := wantTime.Zone()
	_, backOffset := backTime.Zone()
	return wantOffset == backOffset && (wantTime.Location() == time.UTC) == (backTime.Location() == time.UTC)
}

// A time in the local zone, as time.Now gives it, is a TIMESTAMP, and comes
// back in UTC; a time.Duration is a DURATION of its nanoseconds.
func TestLocalTimesAndTimeDurations(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	local := time.Date(2024, 3, 5, 10, 30, 0, 123456789, time.Local)
	typed, back := typedBack(t, db, "TIMESTAMP", local)
	if !typed {
		t.Errorf("a local time.Time did not arrive as a TIMESTAMP")
	}
	backTime, ok := back.(time.Time)
	if !ok || !backTime.Equal(local.Truncate(time.Microsecond)) || backTime.Location() != time.UTC {
		t.Errorf("a local time.Time came back as %#v", back)
	}
	typed, back = typedBack(t, db, "DURATION", 90*time.Minute+500*time.Microsecond)
	if !typed || back != (Duration{Nanos: int64(90*time.Minute + 500*time.Microsecond)}) {
		t.Errorf("a time.Duration arrived typed %v, came back as %#v", typed, back)
	}
	if d, ok := back.(Duration).TimeDuration(); !ok || d != 90*time.Minute+500*time.Microsecond {
		t.Errorf("TimeDuration gave %v, %v", d, ok)
	}
	if _, ok := (Duration{Days: 1}).TimeDuration(); ok {
		t.Errorf("P1D has no fixed length, but TimeDuration gave ok")
	}
}

// Temporal values a query makes come back typed, and go in again as their
// type, alone and inside lists and maps.
func TestTemporalResultsDecodeInListsAndMaps(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	result, err := db.Execute("RETURN DATE '2024-03-05' AS d, TIME '12:30:00+02:00' AS zt, " +
		"TIME '08:15:30.25' AS lt, DURATION 'P1Y2M3DT4H5M6.5S' AS dur, " +
		"ZONED DATETIME '2024-03-05T10:30:00.5+05:30' AS zdt, " +
		"local_datetime('2024-03-05T10:30:00.000001') AS ldt, " +
		"[DATE '2024-03-05', DATE '2024-03-06'] AS dates, " +
		"{at: TIME '12:30:00Z', took: DURATION 'PT1M'} AS m")
	if err != nil {
		t.Fatal(err)
	}
	row := result.Rows[0]
	if row["d"] != (Date{Year: 2024, Month: time.March, Day: 5}) {
		t.Errorf("DATE came back as %#v", row["d"])
	}
	zt, ok := row["zt"].(Time)
	if offset, zoned := zt.Offset(); !ok || zt.Hour() != 12 || zt.Minute() != 30 || !zoned || offset != 7200 {
		t.Errorf("TIME with an offset came back as %#v", row["zt"])
	}
	lt, ok := row["lt"].(Time)
	if _, zoned := lt.Offset(); !ok || zoned || lt.String() != "08:15:30.25" || lt.Nanosecond() != 250000000 {
		t.Errorf("TIME without an offset came back as %#v", row["lt"])
	}
	wantDuration := Duration{Months: 14, Days: 3, Nanos: 4*nanosPerHour + 5*nanosPerMinute + 6*nanosPerSecond + 5e8}
	if row["dur"] != wantDuration {
		t.Errorf("DURATION came back as %#v", row["dur"])
	}
	zdt, ok := row["zdt"].(time.Time)
	if _, offset := zdt.Zone(); !ok || offset != 5*3600+1800 ||
		!zdt.Equal(time.Date(2024, 3, 5, 5, 0, 0, 500000000, time.UTC)) {
		t.Errorf("ZONED DATETIME came back as %#v", row["zdt"])
	}
	ldt, ok := row["ldt"].(time.Time)
	if !ok || ldt.Location() != time.UTC || !ldt.Equal(time.Date(2024, 3, 5, 10, 30, 0, 1000, time.UTC)) {
		t.Errorf("LOCAL DATETIME came back as %#v", row["ldt"])
	}
	dates, ok := row["dates"].([]any)
	if !ok || len(dates) != 2 || dates[1] != (Date{Year: 2024, Month: time.March, Day: 6}) {
		t.Errorf("a list of dates came back as %#v", row["dates"])
	}
	m, ok := row["m"].(map[string]any)
	if !ok || m["took"] != (Duration{Nanos: nanosPerMinute}) {
		t.Errorf("a map came back as %#v", row["m"])
	}
	if at, ok := m["at"].(Time); !ok || at.String() != "12:30:00Z" {
		t.Errorf("a time in a map came back as %#v", m["at"])
	}

	// The values read go back in as their types.
	again, err := db.ExecuteParams("RETURN $d IS TYPED DATE AND $zt IS TYPED TIME AND "+
		"$lt IS TYPED TIME AND $dur IS TYPED DURATION AND $zdt IS TYPED ZONED DATETIME AND "+
		"$ldt IS TYPED LOCAL DATETIME AND $dates IS TYPED LIST<DATE> AND "+
		"$m.at IS TYPED TIME AND $m.took IS TYPED DURATION AS typed, "+
		"$zt = TIME '12:30:00+02:00' AND $lt = TIME '08:15:30.25' AND "+
		"$zdt = ZONED DATETIME '2024-03-05T10:30:00.5+05:30' AS equal",
		map[string]any(row))
	if err != nil {
		t.Fatal(err)
	}
	if again.Rows[0]["typed"] != true || again.Rows[0]["equal"] != true {
		t.Errorf("read values went back as %#v", again.Rows[0])
	}
}

// Stored temporal properties come back typed from GetNode, a stream and an
// AsOf read.
func TestStoredTemporalPropertiesDecodeEverywhere(t *testing.T) {
	db, err := OpenInMemoryWithHistory(HistoryForever)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	born := Date{Year: 1994, Month: time.May, Day: 12}
	wakes := must(NewZonedTime(7, 30, 0, 0, 3600))
	result, err := db.ExecuteParams(
		"INSERT (p:Person {name: 'Shosanna', born: $born, wakes: $wakes, seen: $seen}) RETURN id(p) AS id",
		map[string]any{"born": born, "wakes": wakes, "seen": []any{time.Date(2024, 3, 5, 10, 30, 0, 1000, time.UTC)}})
	if err != nil {
		t.Fatal(err)
	}
	id := idOf(t, result.Rows[0]["id"])
	check := func(where string, props map[string]any) {
		t.Helper()
		if props["born"] != born || props["wakes"] != wakes {
			t.Errorf("%s: born %#v, wakes %#v", where, props["born"], props["wakes"])
		}
		seen, ok := props["seen"].([]any)
		if !ok || len(seen) != 1 {
			t.Errorf("%s: seen %#v", where, props["seen"])
			return
		}
		if at, ok := seen[0].(time.Time); !ok || !at.Equal(time.Date(2024, 3, 5, 10, 30, 0, 1000, time.UTC)) {
			t.Errorf("%s: seen %#v", where, seen[0])
		}
	}
	node, err := db.GetNode(id)
	if err != nil {
		t.Fatal(err)
	}
	check("GetNode", node.Properties)

	stream, err := db.ExecuteStream("MATCH (p:Person) RETURN p.born AS born, p.wakes AS wakes, p.seen AS seen")
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	rows, err := stream.Collect()
	if err != nil {
		t.Fatal(err)
	}
	check("stream", rows[0])

	view, err := db.AsOf(AtEpoch(db.CurrentEpoch()))
	if err != nil {
		t.Fatal(err)
	}
	defer view.Close()
	props := map[string]any{}
	for _, key := range []string{"born", "wakes", "seen"} {
		if props[key], err = view.GetNodeProperty(id, key); err != nil {
			t.Fatal(err)
		}
	}
	check("AsOf", props)
}

// The texts read and written agree with the engine's: each parses to its
// value and writes back as the same text.
func TestTemporalTexts(t *testing.T) {
	for _, text := range []string{"2024-01-15", "-0044-03-15", "0001-01-01"} {
		d, err := ParseDate(text)
		if err != nil || d.String() != text {
			t.Errorf("date %q: %v %v", text, d, err)
		}
	}
	for _, text := range []string{"2024-02-30", "2024-13-01", "2024-1-1x", "", "+2024-01-01"} {
		if _, err := ParseDate(text); err == nil {
			t.Errorf("date %q was read", text)
		}
	}
	for _, text := range []string{"14:30:00", "14:30:00.5", "00:00:00.000000001", "12:30:00Z", "12:30:00+02:00", "12:30:00-05:30"} {
		v, err := ParseTime(text)
		if err != nil || v.String() != text {
			t.Errorf("time %q: %v %v", text, v, err)
		}
	}
	if v, err := ParseTime("14:30"); err != nil || v.String() != "14:30:00" {
		t.Errorf("time 14:30: %v %v", v, err)
	}
	for _, text := range []string{"24:00:00", "12:60:00", "12:30:00.", "12", "12:30:00+02:60", "noon"} {
		if _, err := ParseTime(text); err == nil {
			t.Errorf("time %q was read", text)
		}
	}
	for _, text := range []string{"P1Y2M3DT4H5M6.5S", "PT0S", "-P1M2DT0.000000003S", "P1D", "PT24H", "P1M-2D"} {
		v, err := ParseDuration(text)
		if err != nil || v.String() != text {
			t.Errorf("duration %q: %#v %v", text, v, err)
		}
	}
	if v, err := ParseDuration("P2W"); err != nil || v != (Duration{Days: 14}) {
		t.Errorf("duration P2W: %#v %v", v, err)
	}
	for _, text := range []string{"P", "1D", "P1X", "PT1D", "P+1D"} {
		if _, err := ParseDuration(text); err == nil {
			t.Errorf("duration %q was read", text)
		}
	}
	if _, err := NewZonedTime(12, 0, 0, 0, 90); err == nil {
		t.Errorf("an offset of 90 seconds was taken")
	}
	if _, err := json.Marshal(Date{}); err == nil {
		t.Errorf("the zero Date marshaled")
	}
}

// A time.Time in a location whose offset has seconds (a historic local mean
// time) keeps its instant, at its offset's whole minutes.
func TestOffsetWithSecondsKeepsTheInstant(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	amsterdam1900 := time.Date(1900, 6, 1, 12, 0, 0, 0, time.FixedZone("AMT", 1172))
	typed, back := typedBack(t, db, "ZONED DATETIME", amsterdam1900)
	backTime, ok := back.(time.Time)
	if _, offset := backTime.Zone(); !typed || !ok || !backTime.Equal(amsterdam1900) || offset != 1140 {
		t.Errorf("came back typed %v as %#v", typed, back)
	}
}
