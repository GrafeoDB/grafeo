package grafeo

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
)

// Temporal values, both ways: a time.Time is a TIMESTAMP (in UTC or the
// local zone) or a ZONED DATETIME (in any other location, its offset kept), a
// time.Duration a DURATION, and GQL's DATE, TIME and DURATION, which no Go
// type holds without loss, come back as Date, Time and Duration. Each goes
// back as the type it came as.

const (
	nanosPerSecond = int64(time.Second)
	nanosPerMinute = int64(time.Minute)
	nanosPerHour   = int64(time.Hour)
	nanosPerDay    = 24 * nanosPerHour
)

// timestampTag is t as the {"$timestamp_us": n} tag, its microseconds since
// the Unix epoch (a finer part dropped), when t is in UTC or the local zone;
// otherwise the {"$zoned_datetime": "..."} tag with t's offset at that
// instant.
func timestampTag(t time.Time) any {
	if loc := t.Location(); loc == time.UTC || loc == time.Local {
		return map[string]int64{"$timestamp_us": t.UnixMicro()}
	}
	return map[string]string{"$zoned_datetime": zonedText(t)}
}

// zonedText is t as a ZONED DATETIME writes it, to the microsecond. GQL
// offsets are whole minutes: an offset with seconds (a historic local mean
// time) is written as its whole minutes, with the same instant.
func zonedText(t time.Time) string {
	_, offset := t.Zone()
	if rest := offset % 60; rest != 0 {
		t = t.In(time.FixedZone("", offset-rest))
	}
	return t.Truncate(time.Microsecond).Format("2006-01-02T15:04:05.999999Z07:00")
}

// zonedTime is the time.Time of a ZONED DATETIME's text, in a fixed zone of
// its offset (never time.UTC, so it goes back as a ZONED DATETIME).
func zonedTime(text string) (time.Time, bool) {
	t, err := time.Parse(time.RFC3339Nano, text)
	if err != nil {
		return time.Time{}, false
	}
	_, offset := t.Zone()
	return t.In(time.FixedZone("", offset)), true
}

// timestampTime is the UTC time.Time of a $timestamp_us tag's value: a
// json.Number from a result, or a float64 where the JSON was read without
// UseNumber.
func timestampTime(value any) (time.Time, bool) {
	var micros int64
	switch v := value.(type) {
	case json.Number:
		n, err := v.Int64()
		if err != nil {
			return time.Time{}, false
		}
		micros = n
	case float64:
		if v != math.Trunc(v) || math.Abs(v) > 1<<53 {
			return time.Time{}, false
		}
		micros = int64(v)
	default:
		return time.Time{}, false
	}
	return time.UnixMicro(micros).UTC(), true
}

// Date is a GQL DATE: a day of the calendar, without a time or zone. The zero
// Date is not a valid date.
type Date struct {
	Year  int
	Month time.Month
	Day   int
}

// DateOf returns the date of t in t's location.
func DateOf(t time.Time) Date {
	year, month, day := t.Date()
	return Date{Year: year, Month: month, Day: day}
}

// ParseDate reads a date as GQL writes it: YYYY-MM-DD, with a leading minus
// for a year before year 0.
func ParseDate(text string) (Date, error) {
	invalid := errors.New("grafeo: not a date: " + text)
	body, negative := strings.CutPrefix(text, "-")
	parts := strings.SplitN(body, "-", 3)
	if len(parts) != 3 {
		return Date{}, invalid
	}
	var numbers [3]int
	for i, part := range parts {
		n, ok := digits(part)
		if !ok {
			return Date{}, invalid
		}
		numbers[i] = n
	}
	if negative {
		numbers[0] = -numbers[0]
	}
	d := Date{Year: numbers[0], Month: time.Month(numbers[1]), Day: numbers[2]}
	if !d.valid() {
		return Date{}, invalid
	}
	return d, nil
}

// digits is the number a text of ASCII digits names, and false for any other
// text: an empty one, or one with a sign or a space.
func digits(text string) (int, bool) {
	if text == "" || strings.Trim(text, "0123456789") != "" {
		return 0, false
	}
	n, err := strconv.Atoi(text)
	return n, err == nil
}

// valid reports whether d names a day of the calendar.
func (d Date) valid() bool {
	return d.Month >= time.January && d.Month <= time.December && d.Day >= 1 &&
		DateOf(d.In(time.UTC)) == d
}

// In returns midnight at the start of d in loc.
func (d Date) In(loc *time.Location) time.Time {
	return time.Date(d.Year, d.Month, d.Day, 0, 0, 0, 0, loc)
}

// String returns the date as GQL writes it: 2024-01-15.
func (d Date) String() string {
	if d.Year < 0 {
		return fmt.Sprintf("-%04d-%02d-%02d", -d.Year, int(d.Month), d.Day)
	}
	return fmt.Sprintf("%04d-%02d-%02d", d.Year, int(d.Month), d.Day)
}

// MarshalJSON writes the {"$date": "2024-01-15"} tag the engine reads.
func (d Date) MarshalJSON() ([]byte, error) {
	if !d.valid() {
		return nil, fmt.Errorf("grafeo: %04d-%02d-%02d is not a date", d.Year, int(d.Month), d.Day)
	}
	return json.Marshal(map[string]string{"$date": d.String()})
}

// Time is a GQL TIME: a time of day to the nanosecond, with a UTC offset
// (14:30:00+02:00) or without one (a local time, 14:30:00). Two times are ==
// when they have the same clock reading and offset. The zero Time is
// midnight, local.
type Time struct {
	nanos  int64 // since midnight
	offset int32 // seconds east of UTC, when zoned
	zoned  bool
}

// NewLocalTime returns the time of day without an offset.
func NewLocalTime(hour, minute, second, nanosecond int) (Time, error) {
	if hour < 0 || hour > 23 || minute < 0 || minute > 59 || second < 0 || second > 59 ||
		nanosecond < 0 || nanosecond >= int(nanosPerSecond) {
		return Time{}, fmt.Errorf("grafeo: %02d:%02d:%02d.%09d is not a time of day",
			hour, minute, second, nanosecond)
	}
	return Time{
		nanos: int64(hour)*nanosPerHour + int64(minute)*nanosPerMinute +
			int64(second)*nanosPerSecond + int64(nanosecond),
	}, nil
}

// NewZonedTime returns the time of day at a UTC offset, in seconds east of
// UTC: whole minutes, less than a day.
func NewZonedTime(hour, minute, second, nanosecond, offsetSeconds int) (Time, error) {
	t, err := NewLocalTime(hour, minute, second, nanosecond)
	if err != nil {
		return Time{}, err
	}
	if offsetSeconds%60 != 0 || offsetSeconds <= -int(nanosPerDay/nanosPerSecond) ||
		offsetSeconds >= int(nanosPerDay/nanosPerSecond) {
		return Time{}, fmt.Errorf("grafeo: an offset of %d seconds is not whole minutes within a day",
			offsetSeconds)
	}
	t.offset = int32(offsetSeconds)
	t.zoned = true
	return t, nil
}

// TimeOf returns the clock reading of t with its offset. GQL offsets are
// whole minutes: for an offset with seconds (a historic local mean time) it
// is the clock reading at the offset's whole minutes, at the same instant.
func TimeOf(t time.Time) Time {
	if _, offset := t.Zone(); offset%60 != 0 {
		t = t.In(time.FixedZone("", offset-offset%60))
	}
	_, offset := t.Zone()
	hour, minute, second := t.Clock()
	zoned, err := NewZonedTime(hour, minute, second, t.Nanosecond(), offset)
	if err != nil {
		// An offset of a day or more: the clock reading in UTC
		return TimeOf(t.UTC())
	}
	return zoned
}

// ParseTime reads a time as GQL writes it: HH:MM[:SS[.fraction]], then Z,
// +HH:MM or -HH:MM for an offset.
func ParseTime(text string) (Time, error) {
	invalid := errors.New("grafeo: not a time: " + text)
	body, offset, zoned := text, 0, false
	if rest, ok := strings.CutSuffix(text, "Z"); ok {
		body, zoned = rest, true
	} else if n := len(text); n >= 6 && (text[n-6] == '+' || text[n-6] == '-') && text[n-3] == ':' {
		hours, okHours := digits(text[n-5 : n-3])
		minutes, okMinutes := digits(text[n-2:])
		if !okHours || !okMinutes || minutes > 59 {
			return Time{}, invalid
		}
		offset = hours*3600 + minutes*60
		if text[n-6] == '-' {
			offset = -offset
		}
		body, zoned = text[:n-6], true
	}
	clock, fraction, hasFraction := strings.Cut(body, ".")
	fields := strings.Split(clock, ":")
	if len(fields) < 2 || len(fields) > 3 {
		return Time{}, invalid
	}
	var hms [3]int
	for i, field := range fields {
		n, ok := digits(field)
		if !ok {
			return Time{}, invalid
		}
		hms[i] = n
	}
	nanos := 0
	if hasFraction {
		if fraction == "" || strings.Trim(fraction, "0123456789") != "" {
			return Time{}, invalid
		}
		digits := (fraction + "000000000")[:9]
		nanos, _ = strconv.Atoi(digits)
	}
	var t Time
	var err error
	if zoned {
		t, err = NewZonedTime(hms[0], hms[1], hms[2], nanos, offset)
	} else {
		t, err = NewLocalTime(hms[0], hms[1], hms[2], nanos)
	}
	if err != nil {
		return Time{}, invalid
	}
	return t, nil
}

// Hour returns the hour, 0 to 23.
func (t Time) Hour() int { return int(t.nanos / nanosPerHour) }

// Minute returns the minute, 0 to 59.
func (t Time) Minute() int { return int(t.nanos % nanosPerHour / nanosPerMinute) }

// Second returns the second, 0 to 59.
func (t Time) Second() int { return int(t.nanos % nanosPerMinute / nanosPerSecond) }

// Nanosecond returns the fraction of the second in nanoseconds.
func (t Time) Nanosecond() int { return int(t.nanos % nanosPerSecond) }

// Offset returns the UTC offset in seconds east of UTC, and false for a
// local time.
func (t Time) Offset() (seconds int, ok bool) { return int(t.offset), t.zoned }

// String returns the time as GQL writes it: 14:30:00, 14:30:00.5+02:00,
// 14:30:00Z.
func (t Time) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%02d:%02d:%02d", t.Hour(), t.Minute(), t.Second())
	if ns := t.Nanosecond(); ns != 0 {
		b.WriteString("." + strings.TrimRight(fmt.Sprintf("%09d", ns), "0"))
	}
	if t.zoned {
		b.WriteString(offsetText(int(t.offset)))
	}
	return b.String()
}

// offsetText is an offset as GQL writes it: Z, +02:00, -05:30.
func offsetText(seconds int) string {
	if seconds == 0 {
		return "Z"
	}
	sign := '+'
	if seconds < 0 {
		sign, seconds = '-', -seconds
	}
	return fmt.Sprintf("%c%02d:%02d", sign, seconds/3600, seconds%3600/60)
}

// MarshalJSON writes the {"$time": "14:30:00+02:00"} tag the engine reads.
func (t Time) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]string{"$time": t.String()})
}

// Duration is a GQL DURATION: months, days and nanoseconds, kept apart as a
// month and a day have no fixed length (P1M is not P30D, and P1D is not
// PT24H). A time.Duration parameter is a Duration of nanoseconds alone.
type Duration struct {
	Months int64
	Days   int64
	Nanos  int64
}

// DurationOf returns d as a Duration of nanoseconds.
func DurationOf(d time.Duration) Duration {
	return Duration{Nanos: int64(d)}
}

// TimeDuration returns the duration as a time.Duration, and false when it
// has months or days, which have no fixed length.
func (d Duration) TimeDuration() (time.Duration, bool) {
	return time.Duration(d.Nanos), d.Months == 0 && d.Days == 0
}

// ParseDuration reads an ISO 8601 duration as GQL writes it:
// P[nY][nM][nW][nD][T[nH][nM][n[.fraction]S]], with a leading minus for a
// negative one.
func ParseDuration(text string) (Duration, error) {
	invalid := errors.New("grafeo: not a duration: " + text)
	body, negative := strings.CutPrefix(text, "-")
	body, ok := strings.CutPrefix(body, "P")
	if !ok {
		return Duration{}, invalid
	}
	var d Duration
	inTime, start, content := false, 0, false
	for i := 0; i < len(body); i++ {
		c := body[i]
		if c == 'T' && !inTime && start == i {
			inTime, start = true, i+1
			continue
		}
		designators := "YMWD"
		if inTime {
			designators = "HMS"
		}
		if !strings.ContainsRune(designators, rune(c)) {
			continue
		}
		number := body[start:i]
		var fits bool
		if c == 'S' {
			nanos, err := secondsNanos(number)
			if err != nil {
				return Duration{}, invalid
			}
			d.Nanos, fits = checkedAdd(d.Nanos, nanos)
		} else {
			n, err := strconv.ParseInt(number, 10, 64)
			if err != nil || number[0] == '+' {
				return Duration{}, invalid
			}
			switch {
			case c == 'Y':
				fits = addProduct(&d.Months, n, 12)
			case c == 'M' && !inTime:
				d.Months, fits = checkedAdd(d.Months, n)
			case c == 'W':
				fits = addProduct(&d.Days, n, 7)
			case c == 'D':
				d.Days, fits = checkedAdd(d.Days, n)
			case c == 'H':
				fits = addProduct(&d.Nanos, n, nanosPerHour)
			default:
				fits = addProduct(&d.Nanos, n, nanosPerMinute)
			}
		}
		if !fits {
			// A component past 64 bits: refused, not wrapped around
			return Duration{}, invalid
		}
		start, content = i+1, true
	}
	if !content || start != len(body) {
		return Duration{}, invalid
	}
	if negative {
		if d.Months == math.MinInt64 || d.Days == math.MinInt64 || d.Nanos == math.MinInt64 {
			return Duration{}, invalid
		}
		d = Duration{Months: -d.Months, Days: -d.Days, Nanos: -d.Nanos}
	}
	return d, nil
}

// checkedAdd is a + b, and false when the sum does not fit 64 bits.
func checkedAdd(a, b int64) (int64, bool) {
	sum := a + b
	return sum, (b >= 0) == (sum >= a)
}

// checkedMul is a * b, and false when the product does not fit 64 bits.
func checkedMul(a, b int64) (int64, bool) {
	if a == 0 || b == 0 {
		return 0, true
	}
	product := a * b
	fits := product/b == a && !(a == -1 && b == math.MinInt64) && !(b == -1 && a == math.MinInt64)
	return product, fits
}

// addProduct adds n * unit to *total, and is false when the product or the
// sum does not fit 64 bits.
func addProduct(total *int64, n, unit int64) bool {
	product, fits := checkedMul(n, unit)
	if !fits {
		return false
	}
	*total, fits = checkedAdd(*total, product)
	return fits
}

// secondsNanos reads the seconds of a duration, n or n.fraction, as the
// engine does: the fraction's digits (up to 9) as nanoseconds, its sign its
// own, so that the text the engine writes for any duration reads back as it.
func secondsNanos(number string) (int64, error) {
	whole, fraction, hasFraction := strings.Cut(number, ".")
	n, err := strconv.ParseInt(whole, 10, 64)
	if err != nil || whole[0] == '+' {
		return 0, errors.New("not a number")
	}
	nanos, fits := checkedMul(n, nanosPerSecond)
	if hasFraction {
		if len(fraction) > 9 {
			fraction = fraction[:9]
		}
		f, err := strconv.ParseInt(fraction, 10, 64)
		if err != nil || fraction[0] == '+' {
			return 0, errors.New("not a number")
		}
		var sumFits bool
		nanos, sumFits = checkedAdd(nanos, f*int64(math.Pow10(9-len(fraction))))
		fits = fits && sumFits
	}
	if !fits {
		return 0, errors.New("seconds past 64 bits of nanoseconds")
	}
	return nanos, nil
}

// String returns the duration as GQL writes it: P1Y2M3DT4H5M6.5S, PT0S.
func (d Duration) String() string {
	months, days, nanos := d.Months, d.Days, d.Nanos
	negative := (months < 0 || days < 0 || nanos < 0) && months <= 0 && days <= 0 && nanos <= 0
	if negative {
		months, days, nanos = -months, -days, -nanos
	}
	var b strings.Builder
	if negative {
		b.WriteByte('-')
	}
	b.WriteByte('P')
	if years := months / 12; years != 0 {
		fmt.Fprintf(&b, "%dY", years)
	}
	if m := months % 12; m != 0 {
		fmt.Fprintf(&b, "%dM", m)
	}
	if days != 0 {
		fmt.Fprintf(&b, "%dD", days)
	}
	hours := nanos / nanosPerHour
	minutes := nanos % nanosPerHour / nanosPerMinute
	seconds := nanos % nanosPerMinute / nanosPerSecond
	fraction := nanos % nanosPerSecond
	if nanos != 0 {
		b.WriteByte('T')
		if hours != 0 {
			fmt.Fprintf(&b, "%dH", hours)
		}
		if minutes != 0 {
			fmt.Fprintf(&b, "%dM", minutes)
		}
		if fraction != 0 {
			fmt.Fprintf(&b, "%d.%sS", seconds, strings.TrimRight(fmt.Sprintf("%09d", fraction), "0"))
		} else if seconds != 0 {
			fmt.Fprintf(&b, "%dS", seconds)
		}
	}
	if !negative && months == 0 && days == 0 && nanos == 0 {
		b.WriteString("T0S")
	}
	return b.String()
}

// MarshalJSON writes the {"$duration": "P1DT2H"} tag the engine reads.
func (d Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]string{"$duration": d.String()})
}

// temporalOf is the value a temporal tag stands for (the tag as the map's
// only key, its content of the tag's type), and false for any other map.
func temporalOf(m map[string]any) (any, bool) {
	if len(m) != 1 {
		return nil, false
	}
	if value, ok := m["$timestamp_us"]; ok {
		t, ok := timestampTime(value)
		return t, ok
	}
	for key, value := range m {
		text, ok := value.(string)
		if !ok {
			return nil, false
		}
		switch key {
		case "$zoned_datetime":
			t, ok := zonedTime(text)
			return t, ok
		case "$date":
			d, err := ParseDate(text)
			return d, err == nil
		case "$time":
			t, err := ParseTime(text)
			return t, err == nil
		case "$duration":
			d, err := ParseDuration(text)
			return d, err == nil
		}
	}
	return nil, false
}
