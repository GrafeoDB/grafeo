package grafeo

import (
	"errors"
	"strings"
	"testing"
)

// A database error carries its GQLSTATUS (#639): 22012 for a division by
// zero, 22003 for a value out of range, 42001 for a syntax error. It still
// matches ErrDatabase and reads as before.
func TestErrorsCarryTheirGQLStatus(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	cases := []struct {
		query, status, message string
	}{
		{"RETURN 1 / 0 AS x", "22012", "division by zero: 1 / 0"},
		{"RETURN 9223372036854775807 + 1 AS x", "22003", "numeric value out of range"},
		{"MATCH (n RETURN n", "42001", "GRAFEO-Q001"},
	}
	for _, c := range cases {
		_, err := db.Execute(c.query)
		if err == nil {
			t.Fatalf("%s: expected an error", c.query)
		}
		var grafeoErr *Error
		if !errors.As(err, &grafeoErr) {
			t.Fatalf("%s: %T is no *grafeo.Error", c.query, err)
		}
		if grafeoErr.GQLStatus != c.status {
			t.Errorf("%s: GQLStatus %q, want %q", c.query, grafeoErr.GQLStatus, c.status)
		}
		if !errors.Is(err, ErrDatabase) {
			t.Errorf("%s: does not match ErrDatabase", c.query)
		}
		if !strings.HasPrefix(err.Error(), "grafeo: ") || !strings.Contains(err.Error(), c.message) {
			t.Errorf("%s: message %q", c.query, err.Error())
		}
	}
}
