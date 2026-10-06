// Package grafeo_test discovers and executes .gtest spec files through the Go
// grafeo bindings. Each .gtest file becomes a top-level subtest, and each test
// case within becomes a nested subtest (Go table-driven test pattern).
package grafeo_test

import (
	"bufio"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"

	grafeo "github.com/GrafeoDB/grafeo/crates/bindings/go"
)

// ---------------------------------------------------------------------------
// Compiled features (cached once per test run)
// ---------------------------------------------------------------------------

// compiledFeatures is populated in TestMain and holds the features compiled
// into the grafeo-c library (e.g. "gql", "cypher", "algos").
var compiledFeatures = map[string]bool{}

// runnerCapabilities lists capabilities provided by this runner itself
// (not compiled into the native library). For example, Go natively supports
// 64-bit integers, so "int64-safe" is always available.
var runnerCapabilities = map[string]bool{"int64-safe": true}

func TestMain(m *testing.M) {
	db, err := grafeo.OpenInMemory()
	if err == nil {
		info, err := db.Info()
		if err == nil {
			for _, f := range info.Features {
				compiledFeatures[f] = true
			}
		}
		db.Close()
	}
	os.Exit(m.Run())
}

// hasFeature checks if a feature or compound language key is available.
// Handles compound keys like "graphql-rdf" by checking each part.
func hasFeature(key string) bool {
	if compiledFeatures[key] || runnerCapabilities[key] {
		return true
	}
	// Compound language key: "graphql-rdf" requires both "graphql" and "rdf"
	if key == "graphql-rdf" {
		return compiledFeatures["graphql"] && compiledFeatures["rdf"]
	}
	return false
}

// ---------------------------------------------------------------------------
// .gtest schema types
// ---------------------------------------------------------------------------

// GtestFile is the top-level structure of a .gtest file.
type GtestFile struct {
	Meta  Meta
	Tests []TestCase
}

// Meta holds file-level metadata such as the query language and dataset.
type Meta struct {
	Language string
	Model    string
	Section  string
	Title    string
	Dataset  string
	Requires []string
	Tags     []string
}

// TestCase represents a single test within a .gtest file.
type TestCase struct {
	Name       string
	Query      string
	Statements []string
	Setup      []string
	Params     map[string]string
	Tags       []string
	Requires   []string
	Skip       string
	Language   string
	Dataset    string
	Expect     Expect
	Variants   map[string]string
}

// Expect holds the expected result assertions for a test case.
type Expect struct {
	Rows      [][]string
	Ordered   bool
	Count     *int
	Empty     bool
	Error     *string
	Hash      *string
	Precision *int
	Columns   []string
}

// ---------------------------------------------------------------------------
// Language dispatch
// ---------------------------------------------------------------------------

// dispatchKey maps the language name from a .gtest file to the string accepted
// by db.ExecuteLanguage (which ultimately calls the C FFI).
func dispatchKey(lang string) string {
	switch lang {
	case "sql-pgq", "sql_pgq":
		return "sql"
	case "":
		return "gql"
	default:
		return lang
	}
}

// executeQuery runs a query in the specified language.
func executeQuery(db *grafeo.Database, language, query string) (*grafeo.QueryResult, error) {
	key := dispatchKey(language)
	if key == "gql" {
		return db.Execute(query)
	}
	return db.ExecuteLanguage(key, query, "")
}

// executeQueryWithParams runs a query in the specified language with JSON params.
func executeQueryWithParams(db *grafeo.Database, language, query, paramsJSON string) (*grafeo.QueryResult, error) {
	key := dispatchKey(language)
	if key == "gql" {
		return db.ExecuteWithParams(query, paramsJSON)
	}
	return db.ExecuteLanguage(key, query, paramsJSON)
}

// coerceParamsToJSON converts the string-typed params map from gtest files into
// a JSON string with properly typed values (a list or map written as JSON, int,
// float, bool, or string).
func coerceParamsToJSON(params map[string]string) string {
	if len(params) == 0 {
		return ""
	}
	typed := make(map[string]interface{}, len(params))
	for k, v := range params {
		trimmed := strings.TrimSpace(v)
		if (strings.HasPrefix(trimmed, "[") || strings.HasPrefix(trimmed, "{")) && json.Valid([]byte(trimmed)) {
			typed[k] = json.RawMessage(trimmed)
		} else if n, err := strconv.Atoi(v); err == nil {
			typed[k] = n
		} else if f, err := strconv.ParseFloat(v, 64); err == nil {
			typed[k] = f
		} else if v == "true" {
			typed[k] = true
		} else if v == "false" {
			typed[k] = false
		} else {
			typed[k] = v
		}
	}
	data, err := json.Marshal(typed)
	if err != nil {
		return ""
	}
	return string(data)
}

// ---------------------------------------------------------------------------
// Canonical value serialization
// ---------------------------------------------------------------------------

// valueToString converts a Go interface{} value (as returned by grafeo Row)
// to its canonical string representation for comparison.
func valueToString(v interface{}) string {
	if v == nil {
		return "null"
	}
	switch val := v.(type) {
	case json.Number:
		// Try int64 first to preserve full precision for large integers,
		// fall back to float64 for decimal values.
		if i, err := val.Int64(); err == nil {
			return strconv.FormatInt(i, 10)
		}
		if f, err := val.Float64(); err == nil {
			if math.IsNaN(f) {
				return "NaN"
			}
			if math.IsInf(f, 1) {
				return "Infinity"
			}
			if math.IsInf(f, -1) {
				return "-Infinity"
			}
			if f == math.Trunc(f) && math.Abs(f) < (1<<53) {
				return strconv.FormatInt(int64(f), 10)
			}
			return strconv.FormatFloat(f, 'f', -1, 64)
		}
		return val.String()
	case bool:
		if val {
			return "true"
		}
		return "false"
	case float64:
		if math.IsNaN(val) {
			return "NaN"
		}
		if math.IsInf(val, 1) {
			return "Infinity"
		}
		if math.IsInf(val, -1) {
			return "-Infinity"
		}
		// Match Rust Display: whole floats drop .0
		if val == math.Trunc(val) && math.Abs(val) < (1<<53) {
			return strconv.FormatInt(int64(val), 10)
		}
		return strconv.FormatFloat(val, 'f', -1, 64)
	case int:
		return strconv.Itoa(val)
	case int64:
		return strconv.FormatInt(val, 10)
	case string:
		return val
	case []interface{}:
		parts := make([]string, len(val))
		for i, item := range val {
			parts[i] = valueToString(item)
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case map[string]interface{}:
		// Temporal type-tagged values from C FFI JSON: {"$date": "2024-06-15"}
		if len(val) == 1 {
			for k, v := range val {
				switch k {
				case "$date", "$time", "$datetime", "$timestamp", "$zoned_datetime":
					return fmt.Sprintf("%v", v)
				case "$duration":
					return fmt.Sprintf("%v", v)
				}
			}
		}
		// Duration: {months, days, nanos} -> ISO 8601
		if len(val) == 3 {
			if m, ok1 := val["months"]; ok1 {
				if d, ok2 := val["days"]; ok2 {
					if n, ok3 := val["nanos"]; ok3 {
						return durationToISO(int64(toFloat(m)), int64(toFloat(d)), int64(toFloat(n)))
					}
				}
			}
		}
		entries := make([]string, 0, len(val))
		for k, item := range val {
			entries = append(entries, k+": "+valueToString(item))
		}
		sort.Strings(entries)
		return "{" + strings.Join(entries, ", ") + "}"
	default:
		return fmt.Sprintf("%v", val)
	}
}

// toFloat extracts a numeric value from an interface{}.
func toFloat(v interface{}) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	case int64:
		return float64(n)
	default:
		return 0
	}
}

// durationToISO converts {months, days, nanos} to ISO 8601 duration format.
func durationToISO(totalMonths, days, nanos int64) string {
	years := totalMonths / 12
	months := totalMonths % 12
	hours := nanos / 3_600_000_000_000
	rem := nanos % 3_600_000_000_000
	minutes := rem / 60_000_000_000
	rem = rem % 60_000_000_000
	seconds := rem / 1_000_000_000
	subNanos := rem % 1_000_000_000

	var b strings.Builder
	b.WriteString("P")
	if years != 0 {
		fmt.Fprintf(&b, "%dY", years)
	}
	if months != 0 {
		fmt.Fprintf(&b, "%dM", months)
	}
	if days != 0 {
		fmt.Fprintf(&b, "%dD", days)
	}

	var timePart strings.Builder
	if hours != 0 {
		fmt.Fprintf(&timePart, "%dH", hours)
	}
	if minutes != 0 {
		fmt.Fprintf(&timePart, "%dM", minutes)
	}
	if seconds != 0 || subNanos != 0 {
		if subNanos != 0 {
			frac := strings.TrimRight(fmt.Sprintf("%09d", subNanos), "0")
			fmt.Fprintf(&timePart, "%d.%sS", seconds, frac)
		} else {
			fmt.Fprintf(&timePart, "%dS", seconds)
		}
	}
	if timePart.Len() > 0 {
		b.WriteString("T")
		b.WriteString(timePart.String())
	}

	result := b.String()
	if result == "P" {
		return "P0D"
	}
	return result
}

// ---------------------------------------------------------------------------
// Result extraction
// ---------------------------------------------------------------------------

// resultToRows extracts canonical string rows from a QueryResult, in column
// order. If cols is nil, result.Columns is used.
func resultToRows(result *grafeo.QueryResult, cols []string) [][]string {
	if cols == nil {
		cols = result.Columns
	}
	rows := make([][]string, len(result.Rows))
	for i, row := range result.Rows {
		cells := make([]string, len(cols))
		for j, col := range cols {
			cells[j] = valueToString(row[col])
		}
		rows[i] = cells
	}
	return rows
}

// expectedRows returns the already-canonical rows (identity since Rows is [][]string).
func expectedRows(rows [][]string) [][]string {
	out := make([][]string, len(rows))
	for i, row := range rows {
		cells := make([]string, len(row))
		for j, cell := range row {
			cells[j] = cell
		}
		out[i] = cells
	}
	return out
}

// ---------------------------------------------------------------------------
// Assertions
// ---------------------------------------------------------------------------

func assertRowsSorted(t *testing.T, result *grafeo.QueryResult, expected [][]string) {
	t.Helper()
	actual := resultToRows(result, nil)
	sortRows(actual)
	sortRows(expected)
	assertRowsEqual(t, actual, expected, "sorted")
}

func assertRowsOrdered(t *testing.T, result *grafeo.QueryResult, expected [][]string) {
	t.Helper()
	actual := resultToRows(result, nil)
	assertRowsEqual(t, actual, expected, "ordered")
}

func assertRowsEqual(t *testing.T, actual, expected [][]string, mode string) {
	t.Helper()
	if len(actual) != len(expected) {
		t.Fatalf("Row count mismatch (%s): got %d, expected %d\nActual:   %v\nExpected: %v",
			mode, len(actual), len(expected), actual, expected)
	}
	for i := range actual {
		if len(actual[i]) != len(expected[i]) {
			t.Fatalf("Column count mismatch at %s row %d: got %d cols, expected %d\nActual:   %v\nExpected: %v",
				mode, i, len(actual[i]), len(expected[i]), actual[i], expected[i])
		}
		for j := range actual[i] {
			if actual[i][j] != expected[i][j] {
				t.Fatalf("Mismatch at %s row %d, col %d: got %q, expected %q\nFull actual row:   %v\nFull expected row: %v",
					mode, i, j, actual[i][j], expected[i][j], actual[i], expected[i])
			}
		}
	}
}

func assertRowsWithPrecision(t *testing.T, result *grafeo.QueryResult, expected [][]string, precision int) {
	t.Helper()
	actual := resultToRows(result, nil)
	tolerance := math.Pow(10, float64(-precision))

	if len(actual) != len(expected) {
		t.Fatalf("Row count mismatch: got %d, expected %d", len(actual), len(expected))
	}
	for i := range actual {
		if len(actual[i]) != len(expected[i]) {
			t.Fatalf("Column count mismatch at row %d: got %d, expected %d",
				i, len(actual[i]), len(expected[i]))
		}
		for j := range actual[i] {
			af, aErr := strconv.ParseFloat(actual[i][j], 64)
			ef, eErr := strconv.ParseFloat(expected[i][j], 64)
			if aErr == nil && eErr == nil {
				if math.Abs(af-ef) >= tolerance {
					t.Fatalf("Float mismatch at row %d, col %d: got %v, expected %v (tolerance %v)",
						i, j, af, ef, tolerance)
				}
			} else if actual[i][j] != expected[i][j] {
				t.Fatalf("Mismatch at row %d, col %d: got %q, expected %q",
					i, j, actual[i][j], expected[i][j])
			}
		}
	}
}

func assertCount(t *testing.T, result *grafeo.QueryResult, expected int) {
	t.Helper()
	actual := len(result.Rows)
	if actual != expected {
		t.Fatalf("Row count mismatch: got %d, expected %d", actual, expected)
	}
}

func assertEmpty(t *testing.T, result *grafeo.QueryResult) {
	t.Helper()
	actual := len(result.Rows)
	if actual != 0 {
		t.Fatalf("Expected empty result, got %d row(s)", actual)
	}
}

func assertColumns(t *testing.T, result *grafeo.QueryResult, expected []string) {
	t.Helper()
	actual := result.Columns
	if len(actual) != len(expected) {
		t.Fatalf("Column count mismatch: got %v, expected %v", actual, expected)
	}
	for i := range actual {
		if actual[i] != expected[i] {
			t.Fatalf("Column mismatch at index %d: got %q, expected %q\nFull actual:   %v\nFull expected: %v",
				i, actual[i], expected[i], actual, expected)
		}
	}
}

func assertHash(t *testing.T, result *grafeo.QueryResult, expectedHash string) {
	t.Helper()
	rows := resultToRows(result, nil)
	sortRows(rows)

	h := md5.New()
	for _, row := range rows {
		h.Write([]byte(strings.Join(row, "|")))
		h.Write([]byte("\n"))
	}
	actualHash := fmt.Sprintf("%x", h.Sum(nil))
	if actualHash != expectedHash {
		t.Fatalf("Hash mismatch: got %q, expected %q\nRows: %v", actualHash, expectedHash, rows)
	}
}

// sortRows sorts a slice of string slices lexicographically.
func sortRows(rows [][]string) {
	sort.Slice(rows, func(i, j int) bool {
		a, b := rows[i], rows[j]
		for k := 0; k < len(a) && k < len(b); k++ {
			if a[k] != b[k] {
				return a[k] < b[k]
			}
		}
		return len(a) < len(b)
	})
}

// ---------------------------------------------------------------------------
// Dataset loading
// ---------------------------------------------------------------------------

func loadDataset(t *testing.T, db *grafeo.Database, datasetName string, datasetsDir string) {
	t.Helper()
	setupPath := filepath.Join(datasetsDir, datasetName+".setup")
	f, err := os.Open(setupPath)
	if err != nil {
		t.Fatalf("Dataset file not found: %s", setupPath)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if _, err := db.Execute(line); err != nil {
			t.Fatalf("Failed to load dataset line %q: %v", line, err)
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("Error reading dataset %s: %v", setupPath, err)
	}
}

// ---------------------------------------------------------------------------
// Repo root discovery
// ---------------------------------------------------------------------------

// repoRoot walks up from the test file directory until it finds a directory
// containing Cargo.toml (the workspace root).
func repoRoot() string {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		panic("cannot determine test file location")
	}
	dir := filepath.Dir(file)
	for {
		if _, err := os.Stat(filepath.Join(dir, "Cargo.toml")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			panic("could not find repo root (Cargo.toml) from " + filepath.Dir(file))
		}
		dir = parent
	}
}

// ---------------------------------------------------------------------------
// File discovery
// ---------------------------------------------------------------------------

// findGtestFiles recursively finds all .gtest files under dir, excluding the
// runners subdirectory.
func findGtestFiles(dir string) ([]string, error) {
	var files []string
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		// Skip the runners directory
		if info.IsDir() && info.Name() == "runners" {
			return filepath.SkipDir
		}
		if !info.IsDir() && strings.HasSuffix(info.Name(), ".gtest") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	return files, nil
}

// ---------------------------------------------------------------------------
// Line-based .gtest parser (ported from Node.js parser.mjs)
// ---------------------------------------------------------------------------

// parseContext holds the parser state: the lines of the file and the current
// line index.
type parseContext struct {
	lines []string
	idx   int
}

// parseGtestFile reads and parses a .gtest file using a line-based parser
// (no YAML library). This mirrors the Node.js parser.mjs implementation.
func parseGtestFile(path string) (*GtestFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", path, err)
	}
	content := strings.ReplaceAll(string(data), "\r\n", "\n")
	lines := strings.Split(content, "\n")
	ctx := &parseContext{lines: lines, idx: 0}

	skipBlankAndComments(ctx)
	meta, err := parseMeta(ctx)
	if err != nil {
		return nil, fmt.Errorf("parsing meta in %s: %w", path, err)
	}
	skipBlankAndComments(ctx)
	tests, err := parseTests(ctx)
	if err != nil {
		return nil, fmt.Errorf("parsing tests in %s: %w", path, err)
	}
	return &GtestFile{Meta: meta, Tests: tests}, nil
}

// ---------------------------------------------------------------------------
// Meta block
// ---------------------------------------------------------------------------

func parseMeta(ctx *parseContext) (Meta, error) {
	meta := Meta{
		Language: "gql",
		Dataset:  "empty",
	}
	if err := expectLine(ctx, "meta:"); err != nil {
		return meta, err
	}
	for ctx.idx < len(ctx.lines) {
		skipBlankAndComments(ctx)
		if ctx.idx >= len(ctx.lines) {
			break
		}
		line := ctx.lines[ctx.idx]
		if line == "" || (!strings.HasPrefix(line, " ") && !strings.HasPrefix(line, "\t")) {
			break
		}
		kv := parseKV(strings.TrimSpace(line))
		if kv == nil {
			ctx.idx++
			continue
		}
		key, value := kv[0], kv[1]
		switch key {
		case "language":
			meta.Language = value
		case "model":
			meta.Model = value
		case "section":
			meta.Section = unquote(value)
		case "title":
			meta.Title = value
		case "dataset":
			meta.Dataset = value
		case "requires":
			meta.Requires = parseYamlList(value)
		case "tags":
			meta.Tags = parseYamlList(value)
		}
		ctx.idx++
	}
	return meta, nil
}

// ---------------------------------------------------------------------------
// Tests list
// ---------------------------------------------------------------------------

func parseTests(ctx *parseContext) ([]TestCase, error) {
	skipBlankAndComments(ctx)
	if err := expectLine(ctx, "tests:"); err != nil {
		return nil, err
	}
	var tests []TestCase
	for ctx.idx < len(ctx.lines) {
		skipBlankAndComments(ctx)
		if ctx.idx >= len(ctx.lines) {
			break
		}
		trimmed := strings.TrimSpace(ctx.lines[ctx.idx])
		if strings.HasPrefix(trimmed, "- name:") {
			tc := parseSingleTest(ctx)
			tests = append(tests, tc)
		} else {
			break
		}
	}
	return tests, nil
}

func parseSingleTest(ctx *parseContext) TestCase {
	tc := TestCase{
		Params:   make(map[string]string),
		Variants: make(map[string]string),
	}

	// First line: "- name: xxx"
	first := strings.TrimSpace(ctx.lines[ctx.idx])
	kv := parseKV(first[2:]) // strip "- "
	if kv != nil {
		tc.Name = unquote(kv[1])
	}
	ctx.idx++

	for ctx.idx < len(ctx.lines) {
		line := ctx.lines[ctx.idx]
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "#") {
			ctx.idx++
			continue
		}
		if strings.HasPrefix(trimmed, "- name:") {
			break
		}
		if trimmed == "" {
			ctx.idx++
			continue
		}

		kv2 := parseKV(trimmed)
		if kv2 == nil {
			ctx.idx++
			continue
		}
		key, value := kv2[0], kv2[1]
		switch key {
		case "query":
			if value == "|" {
				tc.Query = parseBlockScalar(ctx)
			} else {
				tc.Query = unquote(value)
				ctx.idx++
			}
		case "skip":
			tc.Skip = unquote(value)
			ctx.idx++
		case "language":
			tc.Language = unquote(value)
			ctx.idx++
		case "dataset":
			tc.Dataset = unquote(value)
			ctx.idx++
		case "setup":
			ctx.idx++
			tc.Setup = parseStringList(ctx)
		case "statements":
			ctx.idx++
			tc.Statements = parseStringList(ctx)
		case "tags":
			tc.Tags = parseYamlList(value)
			ctx.idx++
		case "requires":
			tc.Requires = parseYamlList(value)
			ctx.idx++
		case "params":
			ctx.idx++
			tc.Params = parseMap(ctx, 6)
		case "expect":
			ctx.idx++
			tc.Expect = parseExpectBlock(ctx)
		case "variants":
			ctx.idx++
			tc.Variants = parseMap(ctx, 6)
		default:
			ctx.idx++
		}
	}
	return tc
}

// ---------------------------------------------------------------------------
// Expect block
// ---------------------------------------------------------------------------

func parseExpectBlock(ctx *parseContext) Expect {
	exp := Expect{}
	for ctx.idx < len(ctx.lines) {
		line := ctx.lines[ctx.idx]
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "#") {
			ctx.idx++
			continue
		}
		if strings.HasPrefix(trimmed, "- name:") {
			break
		}
		if !strings.HasPrefix(line, " ") && !strings.HasPrefix(line, "\t") && trimmed != "" {
			break
		}
		if trimmed == "" {
			ctx.idx++
			continue
		}

		kv := parseKV(trimmed)
		if kv == nil {
			break
		}
		key, value := kv[0], kv[1]
		switch key {
		case "ordered":
			exp.Ordered = value == "true"
			ctx.idx++
		case "count":
			if v, err := strconv.Atoi(value); err == nil {
				exp.Count = &v
			}
			ctx.idx++
		case "empty":
			exp.Empty = value == "true"
			ctx.idx++
		case "error":
			s := unquote(value)
			exp.Error = &s
			ctx.idx++
		case "hash":
			s := unquote(value)
			exp.Hash = &s
			ctx.idx++
		case "precision":
			if v, err := strconv.Atoi(value); err == nil {
				exp.Precision = &v
			}
			ctx.idx++
		case "columns":
			exp.Columns = parseYamlList(value)
			ctx.idx++
		case "rows":
			ctx.idx++
			exp.Rows = parseRows(ctx)
		default:
			ctx.idx++
		}
	}
	return exp
}

func parseRows(ctx *parseContext) [][]string {
	var rows [][]string
	for ctx.idx < len(ctx.lines) {
		trimmed := strings.TrimSpace(ctx.lines[ctx.idx])
		if strings.HasPrefix(trimmed, "#") {
			ctx.idx++
			continue
		}
		if trimmed == "" {
			ctx.idx++
			continue
		}
		if strings.HasPrefix(trimmed, "- [") {
			rows = append(rows, parseInlineList(trimmed[2:]))
			ctx.idx++
		} else {
			break
		}
	}
	return rows
}

// ---------------------------------------------------------------------------
// Primitives (ported from Node.js parser.mjs)
// ---------------------------------------------------------------------------

// parseKV splits a string on the first unquoted colon. Returns [key, value] or
// nil if no valid split is found. Respects single and double quotes so that
// strings like "{name: 'Alix'}" inside a query value are not misinterpreted.
func parseKV(s string) []string {
	inSingle := false
	inDouble := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == '\'' && !inDouble {
			inSingle = !inSingle
		} else if c == '"' && !inSingle {
			inDouble = !inDouble
		} else if c == ':' && !inSingle && !inDouble {
			key := strings.TrimSpace(s[:i])
			value := strings.TrimSpace(s[i+1:])
			if key != "" {
				return []string{key, value}
			}
		}
	}
	return nil
}

// unquote strips surrounding single or double quotes and handles YAML-level
// escapes (quotes and backslashes). Does NOT process \n or \t: those are GQL
// string escapes handled by the engine's parser.
func unquote(s string) string {
	s = strings.TrimSpace(s)
	if len(s) >= 2 {
		if (s[0] == '"' && s[len(s)-1] == '"') || (s[0] == '\'' && s[len(s)-1] == '\'') {
			inner := s[1 : len(s)-1]
			inner = strings.ReplaceAll(inner, `\\`, "\x00")
			inner = strings.ReplaceAll(inner, `\"`, `"`)
			inner = strings.ReplaceAll(inner, `\'`, `'`)
			inner = strings.ReplaceAll(inner, "\x00", `\`)
			return inner
		}
	}
	return s
}

// parseYamlList parses an inline YAML list like "[a, b, c]" into a string slice.
// Returns a single-element slice for bare values.
func parseYamlList(s string) []string {
	s = strings.TrimSpace(s)
	if s == "[]" || s == "" {
		return nil
	}
	if strings.HasPrefix(s, "[") && strings.HasSuffix(s, "]") {
		inner := s[1 : len(s)-1]
		parts := strings.Split(inner, ",")
		var result []string
		for _, p := range parts {
			v := unquote(strings.TrimSpace(p))
			if v != "" {
				result = append(result, v)
			}
		}
		return result
	}
	return []string{unquote(s)}
}

// parseInlineList parses a row value like "[val1, val2, {key: val}]" with
// support for nested brackets and braces.
func parseInlineList(s string) []string {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "[") || !strings.HasSuffix(s, "]") {
		return []string{unquote(s)}
	}
	inner := s[1 : len(s)-1]
	var items []string
	var current strings.Builder
	depth := 0
	inSingle := false
	inDouble := false
	for _, c := range inner {
		switch {
		case c == '\'' && !inDouble && depth == 0:
			inSingle = !inSingle
			current.WriteRune(c)
		case c == '"' && !inSingle && depth == 0:
			inDouble = !inDouble
			current.WriteRune(c)
		case (c == '[' || c == '{') && !inSingle && !inDouble:
			depth++
			current.WriteRune(c)
		case (c == ']' || c == '}') && !inSingle && !inDouble:
			depth--
			current.WriteRune(c)
		case c == ',' && depth == 0 && !inSingle && !inDouble:
			items = append(items, unquote(strings.TrimSpace(current.String())))
			current.Reset()
		default:
			current.WriteRune(c)
		}
	}
	if strings.TrimSpace(current.String()) != "" {
		items = append(items, unquote(strings.TrimSpace(current.String())))
	}
	return items
}

// parseStringList parses a YAML-style "- item" list. Items with "|" trigger
// block scalar parsing.
func parseStringList(ctx *parseContext) []string {
	var items []string
	for ctx.idx < len(ctx.lines) {
		trimmed := strings.TrimSpace(ctx.lines[ctx.idx])
		if strings.HasPrefix(trimmed, "#") {
			ctx.idx++
			continue
		}
		if trimmed == "" {
			ctx.idx++
			continue
		}
		if strings.HasPrefix(trimmed, "- ") {
			value := trimmed[2:]
			if value == "|" {
				items = append(items, parseBlockScalar(ctx))
			} else {
				items = append(items, unquote(value))
				ctx.idx++
			}
		} else {
			break
		}
	}
	return items
}

// parseMap parses indented key-value pairs into a map. Stops when indentation
// drops below minIndent or a new test starts.
func parseMap(ctx *parseContext, minIndent int) map[string]string {
	m := make(map[string]string)
	for ctx.idx < len(ctx.lines) {
		line := ctx.lines[ctx.idx]
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "#") || trimmed == "" {
			ctx.idx++
			continue
		}
		if strings.HasPrefix(trimmed, "- name:") {
			break
		}
		indent := len(line) - len(strings.TrimLeft(line, " \t"))
		if indent < minIndent {
			break
		}
		kv := parseKV(trimmed)
		if kv != nil {
			if kv[1] == "|" {
				m[kv[0]] = parseBlockScalar(ctx)
			} else {
				m[kv[0]] = unquote(kv[1])
				ctx.idx++
			}
		} else {
			break
		}
	}
	return m
}

// parseBlockScalar parses a YAML block scalar ("|") by collecting subsequent
// indented lines. The result is trimmed of trailing whitespace.
func parseBlockScalar(ctx *parseContext) string {
	ctx.idx++ // skip the "|" line
	if ctx.idx >= len(ctx.lines) {
		return ""
	}
	blockIndent := len(ctx.lines[ctx.idx]) - len(strings.TrimLeft(ctx.lines[ctx.idx], " \t"))
	var parts []string
	for ctx.idx < len(ctx.lines) {
		line := ctx.lines[ctx.idx]
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			parts = append(parts, "")
			ctx.idx++
			continue
		}
		indent := len(line) - len(strings.TrimLeft(line, " \t"))
		if indent < blockIndent {
			break
		}
		parts = append(parts, line[blockIndent:])
		ctx.idx++
	}
	return strings.TrimRight(strings.Join(parts, "\n"), " \t\n\r")
}

// skipBlankAndComments advances past empty lines and comment lines (starting
// with #).
func skipBlankAndComments(ctx *parseContext) {
	for ctx.idx < len(ctx.lines) {
		trimmed := strings.TrimSpace(ctx.lines[ctx.idx])
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			ctx.idx++
		} else {
			break
		}
	}
}

// expectLine asserts that the current line (after skipping blanks/comments)
// matches the expected string, then advances.
func expectLine(ctx *parseContext, expected string) error {
	skipBlankAndComments(ctx)
	if ctx.idx >= len(ctx.lines) {
		return fmt.Errorf("expected %q at line %d, got <EOF>", expected, ctx.idx+1)
	}
	actual := strings.TrimSpace(ctx.lines[ctx.idx])
	if actual != expected {
		return fmt.Errorf("expected %q at line %d, got %q", expected, ctx.idx+1, actual)
	}
	ctx.idx++
	return nil
}

// ---------------------------------------------------------------------------
// Main test function
// ---------------------------------------------------------------------------

func TestSpec(t *testing.T) {
	root := repoRoot()
	specDir := filepath.Join(root, "tests", "spec")
	datasetsDir := filepath.Join(specDir, "datasets")

	gtestFiles, err := findGtestFiles(specDir)
	if err != nil {
		t.Fatalf("Failed to discover .gtest files: %v", err)
	}
	if len(gtestFiles) == 0 {
		t.Fatal("No .gtest files found under tests/spec/")
	}

	for _, filePath := range gtestFiles {
		relPath, _ := filepath.Rel(specDir, filePath)
		relPath = filepath.ToSlash(relPath)

		t.Run(relPath, func(t *testing.T) {
			gf, err := parseGtestFile(filePath)
			if err != nil {
				t.Fatalf("Parse error: %v", err)
			}

			for _, tc := range gf.Tests {
				tc := tc // capture loop variable

				if len(tc.Variants) > 0 {
					// Rosetta: one subtest per variant language
					for lang, query := range tc.Variants {
						lang, query := lang, query // capture
						testName := tc.Name + "[" + lang + "]"
						t.Run(testName, func(t *testing.T) {
							runSingleTest(t, gf, tc, lang, query, datasetsDir)
						})
					}
				} else {
					t.Run(tc.Name, func(t *testing.T) {
						runSingleTest(t, gf, tc, "", "", datasetsDir)
					})
				}
			}
		})
	}
}

// runSingleTest executes a single test case from a .gtest file.
// If variantLang is non-empty, it overrides the file's meta.language and
// variantQuery overrides tc.Query.
func runSingleTest(t *testing.T, gf *GtestFile, tc TestCase, variantLang, variantQuery string, datasetsDir string) {
	t.Helper()

	// Skip if test has a skip field
	if tc.Skip != "" {
		t.Skipf("skipped in .gtest: %s", tc.Skip)
	}

	// Determine language: per-test override > rosetta variant > file meta
	language := variantLang
	if language == "" && tc.Language != "" {
		language = tc.Language
	}
	if language == "" {
		language = gf.Meta.Language
	}

	// Check language availability
	if language != "gql" && language != "" {
		langKey := strings.ReplaceAll(language, "_", "-")
		if !hasFeature(langKey) {
			t.Skipf("language '%s' not available in this build", langKey)
		}
	}

	// Check requires: skip if this build lacks the feature
	for _, req := range gf.Meta.Requires {
		key := strings.ReplaceAll(req, "_", "-")
		if !hasFeature(key) {
			t.Skipf("feature '%s' not available in this build", key)
		}
	}
	for _, req := range tc.Requires {
		key := strings.ReplaceAll(req, "_", "-")
		if !hasFeature(key) {
			t.Skipf("required capability '%s' not available", key)
		}
	}

	// Fresh database per test
	db, err := grafeo.OpenInMemory()
	if err != nil {
		t.Fatalf("Failed to open in-memory database: %v", err)
	}
	defer db.Close()

	// Load dataset (per-test override takes priority)
	effectiveDataset := gf.Meta.Dataset
	if tc.Dataset != "" {
		effectiveDataset = tc.Dataset
	}
	if effectiveDataset != "" && effectiveDataset != "empty" {
		loadDataset(t, db, effectiveDataset, datasetsDir)
	}

	// Run setup queries in the file's declared language
	setupLang := gf.Meta.Language
	for _, setupQ := range tc.Setup {
		if _, err := executeQuery(db, setupLang, setupQ); err != nil {
			t.Fatalf("Setup query failed: %v\nQuery: %s", err, setupQ)
		}
	}

	// Determine query / statements
	query := variantQuery
	if query == "" {
		query = tc.Query
	}

	var queries []string
	if len(tc.Statements) > 0 {
		queries = tc.Statements
	} else if query != "" {
		queries = []string{query}
	} else if tc.Expect.Error != nil {
		// Allow empty-string queries for error tests (e.g. empty input)
		queries = []string{query}
	} else {
		t.Fatalf("No query or statements in test %q", tc.Name)
	}

	exp := tc.Expect

	// Coerce params to JSON for the last query
	paramsJSON := coerceParamsToJSON(tc.Params)

	// Error tests
	if exp.Error != nil {
		runErrorTest(t, db, language, queries, *exp.Error, paramsJSON)
		return
	}

	// Execute all-but-last as fire-and-forget (with params so $param works)
	for _, q := range queries[:len(queries)-1] {
		var fireErr error
		if paramsJSON != "" {
			_, fireErr = executeQueryWithParams(db, language, q, paramsJSON)
		} else {
			_, fireErr = executeQuery(db, language, q)
		}
		if fireErr != nil {
			if isUnsupportedLanguageError(fireErr) {
				t.Skipf("Language %q not available: %v", language, fireErr)
			}
			t.Fatalf("Statement failed: %v\nQuery: %s", fireErr, q)
		}
	}

	// Last query: capture result (with params if provided)
	var result *grafeo.QueryResult
	if paramsJSON != "" {
		result, err = executeQueryWithParams(db, language, queries[len(queries)-1], paramsJSON)
	} else {
		result, err = executeQuery(db, language, queries[len(queries)-1])
	}
	if err != nil {
		if isUnsupportedLanguageError(err) {
			t.Skipf("Language %q not available: %v", language, err)
		}
		t.Fatalf("Query failed: %v\nQuery: %s", err, queries[len(queries)-1])
	}

	// Column assertion (checked before value assertions)
	if len(exp.Columns) > 0 {
		assertColumns(t, result, exp.Columns)
	}

	// Value assertions
	expRows := expectedRows(exp.Rows)

	switch {
	case exp.Empty:
		assertEmpty(t, result)
	case exp.Count != nil:
		assertCount(t, result, *exp.Count)
	case exp.Hash != nil:
		assertHash(t, result, *exp.Hash)
	case len(expRows) > 0:
		if exp.Precision != nil {
			assertRowsWithPrecision(t, result, expRows, *exp.Precision)
		} else if exp.Ordered {
			assertRowsOrdered(t, result, expRows)
		} else {
			assertRowsSorted(t, result, expRows)
		}
	// If none of the above, the test just checks the query does not error
	}
}

// runErrorTest executes queries expecting the last one to produce an error.
func runErrorTest(t *testing.T, db *grafeo.Database, language string, queries []string, expectedSubstr string, paramsJSON string) {
	t.Helper()

	// Execute all-but-last normally (with params so $param works)
	for _, q := range queries[:len(queries)-1] {
		var setupErr error
		if paramsJSON != "" {
			_, setupErr = executeQueryWithParams(db, language, q, paramsJSON)
		} else {
			_, setupErr = executeQuery(db, language, q)
		}
		if setupErr != nil {
			if isUnsupportedLanguageError(setupErr) {
				t.Skipf("Language %q not available: %v", language, setupErr)
			}
			t.Fatalf("Pre-error statement failed unexpectedly: %v\nQuery: %s", setupErr, q)
		}
	}

	// Last query should fail (with params if provided)
	var err error
	if paramsJSON != "" {
		_, err = executeQueryWithParams(db, language, queries[len(queries)-1], paramsJSON)
	} else {
		_, err = executeQuery(db, language, queries[len(queries)-1])
	}
	if err == nil {
		t.Fatalf("Expected error containing %q but query succeeded", expectedSubstr)
	}

	errMsg := err.Error()
	if !strings.Contains(errMsg, expectedSubstr) {
		// Case-insensitive fallback
		if !strings.Contains(strings.ToLower(errMsg), strings.ToLower(expectedSubstr)) {
			t.Fatalf("Error %q does not contain %q", errMsg, expectedSubstr)
		}
	}
}

// isUnsupportedLanguageError checks if an error indicates the language is not
// compiled into the grafeo-c library.
func isUnsupportedLanguageError(err error) bool {
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "unsupported language") ||
		strings.Contains(msg, "not available") ||
		strings.Contains(msg, "not supported") ||
		strings.Contains(msg, "unknown language")
}
