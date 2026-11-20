// The openCypher TCK's value notation, for `values:`: parse a cell and compare it to a result
// value, type-exact. Mirrors the Rust runner's src/tck.rs; every runner checks itself against
// tests/spec/runners/values_cases.json (TestSpecValueCases below).

package grafeo_test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode"

	grafeo "github.com/GrafeoDB/grafeo/crates/bindings/go"
)

type tckKind int

const (
	tckNull tckKind = iota
	tckBool
	tckInt
	tckFloat
	tckString
	tckList
	tckMap
	tckNode
	tckRelationship
	tckPath
)

// tckCell is one cell of the TCK's value notation.
type tckCell struct {
	kind    tckKind
	b       bool
	i       int64
	f       float64
	s       string // a string, or a relationship's type
	items   []tckCell
	entries map[string]tckCell // a map, or an entity's properties
	labels  map[string]bool
	// A path: its nodes, relationships, and whether each relationship starts on its left
	nodes, rels []tckCell
	forward     []bool
}

type tckParser struct {
	text []rune
	at   int
}

func (p *tckParser) fail(what string) error {
	return fmt.Errorf("%s at %d in '%s'", what, p.at, string(p.text))
}

func (p *tckParser) skip() {
	for p.at < len(p.text) && unicode.IsSpace(p.text[p.at]) {
		p.at++
	}
}

func (p *tckParser) peek(token string) bool {
	p.skip()
	return strings.HasPrefix(string(p.text[p.at:]), token)
}

func (p *tckParser) take(token string) error {
	if !p.peek(token) {
		return p.fail("expected '" + token + "'")
	}
	p.at += len([]rune(token))
	return nil
}

func (p *tckParser) word(word string) bool {
	rest := string(p.text[p.at:])
	if !strings.HasPrefix(rest, word) {
		return false
	}
	after := []rune(rest[len(word):])
	if len(after) > 0 && (unicode.IsLetter(after[0]) || unicode.IsDigit(after[0])) {
		return false
	}
	p.at += len([]rune(word))
	return true
}

func (p *tckParser) name() (string, error) {
	p.skip()
	if p.peek("`") {
		end := strings.IndexRune(string(p.text[p.at+1:]), '`')
		if end < 0 {
			return "", p.fail("unterminated name")
		}
		name := []rune(string(p.text[p.at+1:])[:end])
		p.at += len(name) + 2
		return string(name), nil
	}
	start := p.at
	for p.at < len(p.text) && (unicode.IsLetter(p.text[p.at]) || unicode.IsDigit(p.text[p.at]) || p.text[p.at] == '_') {
		p.at++
	}
	if start == p.at {
		return "", p.fail("expected a name")
	}
	return string(p.text[start:p.at]), nil
}

func (p *tckParser) value() (tckCell, error) {
	p.skip()
	switch {
	case p.word("null"):
		return tckCell{kind: tckNull}, nil
	case p.word("true"):
		return tckCell{kind: tckBool, b: true}, nil
	case p.word("false"):
		return tckCell{kind: tckBool}, nil
	}
	for _, special := range []struct {
		word  string
		value float64
	}{{"NaN", math.NaN()}, {"-Inf", math.Inf(-1)}, {"Inf", math.Inf(1)}} {
		if strings.HasPrefix(string(p.text[p.at:]), special.word) {
			p.at += len(special.word)
			return tckCell{kind: tckFloat, f: special.value}, nil
		}
	}
	rest := string(p.text[p.at:])
	switch {
	case strings.HasPrefix(rest, "'"):
		s, err := p.str()
		return tckCell{kind: tckString, s: s}, err
	case strings.HasPrefix(rest, "{"):
		m, err := p.mapping()
		return tckCell{kind: tckMap, entries: m}, err
	case strings.HasPrefix(rest, "("):
		return p.node()
	case strings.HasPrefix(rest, "<"):
		return p.path()
	case strings.HasPrefix(rest, "[:"):
		return p.relationship()
	case strings.HasPrefix(rest, "["):
		if err := p.take("["); err != nil {
			return tckCell{}, err
		}
		items, err := p.items("]", p.value)
		return tckCell{kind: tckList, items: items}, err
	}
	return p.number()
}

func (p *tckParser) str() (string, error) {
	if err := p.take("'"); err != nil {
		return "", err
	}
	var out strings.Builder
	escapes := map[rune]rune{'\'': '\'', '"': '"', '\\': '\\', 'n': '\n', 't': '\t', 'r': '\r', 'b': '\b', 'f': '\f'}
	for p.at < len(p.text) {
		c := p.text[p.at]
		switch {
		case c == '\'':
			p.at++
			return out.String(), nil
		case c == '\\' && p.at+1 < len(p.text):
			code := p.text[p.at+1]
			if code == 'u' && p.at+6 <= len(p.text) {
				n, err := strconv.ParseUint(string(p.text[p.at+2:p.at+6]), 16, 32)
				if err != nil {
					return "", p.fail("bad \\u escape")
				}
				out.WriteRune(rune(n))
				p.at += 6
				continue
			}
			if r, ok := escapes[code]; ok {
				out.WriteRune(r)
			} else {
				out.WriteRune('\\')
				out.WriteRune(code)
			}
			p.at += 2
		default:
			out.WriteRune(c)
			p.at++
		}
	}
	return "", p.fail("unterminated string")
}

func (p *tckParser) number() (tckCell, error) {
	start := p.at
	for p.at < len(p.text) && strings.ContainsRune("+-0123456789.eE", p.text[p.at]) {
		p.at++
	}
	text := string(p.text[start:p.at])
	if text == "" {
		return tckCell{}, p.fail("expected a value")
	}
	if strings.ContainsAny(text, ".eE") {
		f, err := strconv.ParseFloat(text, 64)
		if err != nil {
			return tckCell{}, p.fail("bad float")
		}
		return tckCell{kind: tckFloat, f: f}, nil
	}
	i, err := strconv.ParseInt(text, 10, 64)
	if err != nil {
		return tckCell{}, p.fail("bad integer")
	}
	return tckCell{kind: tckInt, i: i}, nil
}

func (p *tckParser) items(closing string, item func() (tckCell, error)) ([]tckCell, error) {
	out := []tckCell{}
	if p.peek(closing) {
		return out, p.take(closing)
	}
	for {
		cell, err := item()
		if err != nil {
			return nil, err
		}
		out = append(out, cell)
		if p.peek(",") {
			_ = p.take(",")
			continue
		}
		return out, p.take(closing)
	}
}

func (p *tckParser) mapping() (map[string]tckCell, error) {
	if err := p.take("{"); err != nil {
		return nil, err
	}
	out := map[string]tckCell{}
	_, err := p.items("}", func() (tckCell, error) {
		key, err := p.name()
		if err != nil {
			return tckCell{}, err
		}
		if err := p.take(":"); err != nil {
			return tckCell{}, err
		}
		v, err := p.value()
		out[key] = v
		return v, err
	})
	return out, err
}

func (p *tckParser) properties() (map[string]tckCell, error) {
	if p.peek("{") {
		return p.mapping()
	}
	return map[string]tckCell{}, nil
}

func (p *tckParser) node() (tckCell, error) {
	if err := p.take("("); err != nil {
		return tckCell{}, err
	}
	labels := map[string]bool{}
	for p.peek(":") {
		_ = p.take(":")
		label, err := p.name()
		if err != nil {
			return tckCell{}, err
		}
		labels[label] = true
	}
	props, err := p.properties()
	if err != nil {
		return tckCell{}, err
	}
	return tckCell{kind: tckNode, labels: labels, entries: props}, p.take(")")
}

func (p *tckParser) relationship() (tckCell, error) {
	if err := p.take("["); err != nil {
		return tckCell{}, err
	}
	if err := p.take(":"); err != nil {
		return tckCell{}, err
	}
	relType, err := p.name()
	if err != nil {
		return tckCell{}, err
	}
	props, err := p.properties()
	if err != nil {
		return tckCell{}, err
	}
	return tckCell{kind: tckRelationship, s: relType, entries: props}, p.take("]")
}

func (p *tckParser) path() (tckCell, error) {
	if err := p.take("<"); err != nil {
		return tckCell{}, err
	}
	first, err := p.node()
	if err != nil {
		return tckCell{}, err
	}
	cell := tckCell{kind: tckPath, nodes: []tckCell{first}}
	for !p.peek(">") {
		forward := !p.peek("<-")
		opening, closing := "-", "->"
		if !forward {
			opening, closing = "<-", "-"
		}
		if err := p.take(opening); err != nil {
			return tckCell{}, err
		}
		rel, err := p.relationship()
		if err != nil {
			return tckCell{}, err
		}
		if err := p.take(closing); err != nil {
			return tckCell{}, err
		}
		next, err := p.node()
		if err != nil {
			return tckCell{}, err
		}
		cell.rels = append(cell.rels, rel)
		cell.forward = append(cell.forward, forward)
		cell.nodes = append(cell.nodes, next)
	}
	return cell, p.take(">")
}

// parseTck parses one cell of the TCK's value notation.
func parseTck(text string) (tckCell, error) {
	p := &tckParser{text: []rune(text)}
	cell, err := p.value()
	if err != nil {
		return cell, err
	}
	p.skip()
	if p.at != len(p.text) {
		return cell, p.fail("unexpected text")
	}
	return cell, nil
}

// tckEntity is a node or relationship as a result gives it.
type tckEntity struct {
	node           bool
	id             string
	labels         map[string]bool
	relType        string
	source, target string
	props          map[string]any
}

// entityOf is the one place that recognises entities, by the shape the binding gives them: a
// map with an integer `_id` and a `_labels` list is a node; one with `_id`, a string `_type`,
// `_source` and `_target` a relationship. A map the query builds with those keys comes back as
// an entity too: results have no typed node or relationship value.
func entityOf(value any) (tckEntity, bool) {
	m, ok := value.(map[string]any)
	if !ok {
		return tckEntity{}, false
	}
	id, ok := m["_id"].(json.Number)
	if !ok || strings.ContainsAny(string(id), ".eE") {
		return tckEntity{}, false
	}
	props := func(reserved ...string) map[string]any {
		out := map[string]any{}
		for k, v := range m {
			skip := false
			for _, r := range reserved {
				skip = skip || k == r
			}
			if !skip {
				out[k] = v
			}
		}
		return out
	}
	if labels, ok := m["_labels"].([]any); ok {
		set := map[string]bool{}
		for _, l := range labels {
			if s, ok := l.(string); ok {
				set[s] = true
			}
		}
		return tckEntity{node: true, id: string(id), labels: set, props: props("_id", "_labels")}, true
	}
	relType, ok1 := m["_type"].(string)
	source, ok2 := m["_source"].(json.Number)
	target, ok3 := m["_target"].(json.Number)
	if ok1 && ok2 && ok3 {
		return tckEntity{id: string(id), relType: relType, source: string(source), target: string(target),
			props: props("_id", "_type", "_source", "_target")}, true
	}
	return tckEntity{}, false
}

// pathOf is the nodes and relationships of a path, which the binding gives as {"$path": {...}}.
func pathOf(value any) (nodes, edges []any, ok bool) {
	m, isMap := value.(map[string]any)
	if !isMap || len(m) != 1 {
		return nil, nil, false
	}
	inner, isMap := m["$path"].(map[string]any)
	if !isMap {
		return nil, nil, false
	}
	nodes, ok1 := inner["nodes"].([]any)
	edges, ok2 := inner["edges"].([]any)
	return nodes, edges, ok1 && ok2
}

func propertiesMatch(expected map[string]tckCell, actual map[string]any, unordered bool) bool {
	if len(expected) != len(actual) {
		return false
	}
	for k, cell := range expected {
		v, ok := actual[k]
		if !ok || !matchTck(cell, v, unordered) {
			return false
		}
	}
	return true
}

func sameLabels(expected, actual map[string]bool) bool {
	if len(expected) != len(actual) {
		return false
	}
	for l := range expected {
		if !actual[l] {
			return false
		}
	}
	return true
}

func bagMatches(expected []tckCell, actual []any, unordered bool) bool {
	if len(expected) != len(actual) {
		return false
	}
	left := append([]any{}, actual...)
	for _, cell := range expected {
		hit := -1
		for i, v := range left {
			if matchTck(cell, v, unordered) {
				hit = i
				break
			}
		}
		if hit < 0 {
			return false
		}
		left = append(left[:hit], left[hit+1:]...)
	}
	return true
}

// matchTck reports whether a result value is the value cell writes, type-exact: a number keeps
// its JSON text, so an integer never matches a float.
func matchTck(cell tckCell, value any, unordered bool) bool {
	switch cell.kind {
	case tckNull:
		return value == nil
	case tckBool:
		b, ok := value.(bool)
		return ok && b == cell.b
	case tckInt:
		n, ok := value.(json.Number)
		if !ok || strings.ContainsAny(string(n), ".eE") {
			return false
		}
		i, err := n.Int64()
		return err == nil && i == cell.i
	case tckFloat:
		var f float64
		switch v := value.(type) {
		case json.Number:
			if !strings.ContainsAny(string(v), ".eE") {
				return false
			}
			parsed, err := v.Float64()
			if err != nil {
				return false
			}
			f = parsed
		case float64: // NaN and the infinities, which JSON carries as tags
			f = v
		default:
			return false
		}
		return f == cell.f || (math.IsNaN(f) && math.IsNaN(cell.f))
	case tckString:
		s, ok := value.(string)
		return ok && s == cell.s
	case tckList:
		items, ok := value.([]any)
		if !ok {
			return false
		}
		if unordered {
			return bagMatches(cell.items, items, unordered)
		}
		if len(items) != len(cell.items) {
			return false
		}
		for i, item := range cell.items {
			if !matchTck(item, items[i], unordered) {
				return false
			}
		}
		return true
	case tckMap:
		m, ok := value.(map[string]any)
		if !ok {
			return false
		}
		if _, isEntity := entityOf(value); isEntity {
			return false
		}
		if _, _, isPath := pathOf(value); isPath {
			return false
		}
		return propertiesMatch(cell.entries, m, unordered)
	case tckNode:
		e, ok := entityOf(value)
		return ok && e.node && sameLabels(cell.labels, e.labels) && propertiesMatch(cell.entries, e.props, unordered)
	case tckRelationship:
		e, ok := entityOf(value)
		return ok && !e.node && e.relType == cell.s && propertiesMatch(cell.entries, e.props, unordered)
	case tckPath:
		nodes, edges, ok := pathOf(value)
		if !ok || len(nodes) != len(cell.nodes) || len(edges) != len(cell.rels) {
			return false
		}
		for i, edge := range edges {
			e, ok := entityOf(edge)
			start, ok1 := entityOf(nodes[i])
			end, ok2 := entityOf(nodes[i+1])
			if !cell.forward[i] {
				start, end = end, start
			}
			if !ok || !ok1 || !ok2 || e.node || e.source != start.id || e.target != end.id {
				return false
			}
			if !matchTck(cell.rels[i], edge, unordered) {
				return false
			}
		}
		for i, node := range nodes {
			if !matchTck(cell.nodes[i], node, unordered) {
				return false
			}
		}
		return true
	}
	return false
}

// tableRows splits a `values:` table into rows of cell texts; `\|` is a pipe inside a cell.
func tableRows(table string) [][]string {
	var rows [][]string
	for _, line := range strings.Split(table, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		inner := strings.TrimSuffix(strings.TrimPrefix(line, "|"), "|")
		var cells []string
		var cell strings.Builder
		runes := []rune(inner)
		for i := 0; i < len(runes); i++ {
			switch {
			case runes[i] == '\\' && i+1 < len(runes) && runes[i+1] == '|':
				cell.WriteRune('|')
				i++
			case runes[i] == '|':
				cells = append(cells, strings.TrimSpace(cell.String()))
				cell.Reset()
			default:
				cell.WriteRune(runes[i])
			}
		}
		rows = append(rows, append(cells, strings.TrimSpace(cell.String())))
	}
	return rows
}

// isTemporal reports whether the binding gave value for a temporal.
func isTemporal(value any) bool {
	switch value.(type) {
	case time.Time, grafeo.Date, grafeo.Time, grafeo.Duration:
		return true
	}
	return false
}

// temporalsAsText is value with each temporal in it written as its Cypher toString(), the
// engine's own text: how a `values_dialect: tck` file compares temporals.
func temporalsAsText(db *grafeo.Database, value any) (any, error) {
	switch v := value.(type) {
	case []any:
		out := make([]any, len(v))
		for i, item := range v {
			text, err := temporalsAsText(db, item)
			if err != nil {
				return nil, err
			}
			out[i] = text
		}
		return out, nil
	case map[string]any:
		out := map[string]any{}
		for k, item := range v {
			text, err := temporalsAsText(db, item)
			if err != nil {
				return nil, err
			}
			out[k] = text
		}
		return out, nil
	}
	if !isTemporal(value) {
		return value, nil
	}
	result, err := db.ExecuteParams("RETURN toString($v) AS s", map[string]any{"v": value})
	if err != nil {
		return nil, err
	}
	return result.Rows[0]["s"], nil
}

// assertValues asserts the result holds the rows of table, a `values:` table whose first row
// names the columns: in order when ordered, otherwise as a bag.
func assertValues(t specT, db *grafeo.Database, result *grafeo.QueryResult, table string, ordered, unordered, tckDialect bool) {
	t.Helper()
	rows := tableRows(table)
	var header []string
	for _, name := range rows[0] {
		if name != "" {
			header = append(header, name)
		}
	}
	if strings.Join(result.Columns, "\x00") != strings.Join(header, "\x00") {
		t.Fatalf("Columns differ: %v, expected %v", result.Columns, header)
	}
	expected := make([][]tckCell, len(rows)-1)
	for i, row := range rows[1:] {
		for _, text := range row {
			cell, err := parseTck(text)
			if err != nil {
				t.Fatalf("%v", err)
			}
			expected[i] = append(expected[i], cell)
		}
	}
	actual := make([][]any, len(result.Rows))
	for i, row := range result.Rows {
		for _, column := range result.Columns {
			value := row[column]
			if tckDialect {
				text, err := temporalsAsText(db, value)
				if err != nil {
					t.Fatalf("toString failed: %v", err)
				}
				value = text
			}
			actual[i] = append(actual[i], value)
		}
	}
	rowMatches := func(cells []tckCell, values []any) bool {
		if len(cells) != len(values) {
			return false
		}
		for i, cell := range cells {
			if !matchTck(cell, values[i], unordered) {
				return false
			}
		}
		return true
	}
	holds := len(expected) == len(actual)
	if holds && ordered {
		for i := range expected {
			holds = holds && rowMatches(expected[i], actual[i])
		}
	} else if holds {
		left := append([][]any{}, actual...)
		for _, cells := range expected {
			hit := -1
			for i, values := range left {
				if rowMatches(cells, values) {
					hit = i
					break
				}
			}
			if hit < 0 {
				holds = false
				break
			}
			left = append(left[:hit], left[hit+1:]...)
		}
	}
	if !holds {
		mode := ""
		if ordered {
			mode = " (in order)"
		}
		t.Fatalf("Rows differ%s:\nactual:   %v\nexpected: %s", mode, actual, table)
	}
}

// graphObservation is what a graph holds, as the openCypher TCK observes side effects.
type graphObservation map[string]map[string]bool

// observeGraph reads node and relationship ids, the label names in use, and the properties as
// (entity, key, value) triples, with two GQL queries every build has.
func observeGraph(t specT, db *grafeo.Database) graphObservation {
	t.Helper()
	seen := graphObservation{"nodes": {}, "relationships": {}, "labels": {}, "properties": {}}
	for _, query := range []string{"MATCH (n) RETURN n", "MATCH ()-[r]->() RETURN r"} {
		result, err := db.Execute(query)
		if err != nil {
			t.Fatalf("observing the graph with %q failed: %v", query, err)
		}
		for _, row := range result.Rows {
			for _, value := range row {
				e, ok := entityOf(value)
				if !ok {
					continue
				}
				kind := "relationships"
				if e.node {
					kind = "nodes"
				}
				seen[kind][e.id] = true
				for label := range e.labels {
					seen["labels"][label] = true
				}
				for key, prop := range e.props {
					text, _ := json.Marshal(prop)
					seen["properties"][fmt.Sprintf("%s|%s|%s|%s", kind, e.id, key, text)] = true
				}
			}
		}
	}
	return seen
}

// assertSideEffects asserts the side effects from before to after are expected (an unnamed
// key is 0).
func assertSideEffects(t specT, before, after graphObservation, expected map[string]int) {
	t.Helper()
	var differ []string
	for _, key := range []string{"nodes", "relationships", "labels", "properties"} {
		added, removed := 0, 0
		for x := range after[key] {
			if !before[key][x] {
				added++
			}
		}
		for x := range before[key] {
			if !after[key][x] {
				removed++
			}
		}
		for _, c := range []struct {
			name  string
			count int
		}{{"+" + key, added}, {"-" + key, removed}} {
			if c.count != expected[c.name] {
				differ = append(differ, fmt.Sprintf("%s: %d, expected %d", c.name, c.count, expected[c.name]))
			}
		}
	}
	if len(differ) > 0 {
		sort.Strings(differ)
		t.Fatalf("Side effects differ: %s", strings.Join(differ, "; "))
	}
}

// ---------------------------------------------------------------------------
// The shared value cases (tests/spec/runners/values_cases.json)
// ---------------------------------------------------------------------------

// nativeValue is a neutral value of the case file as the Go binding returns it.
func nativeValue(t *testing.T, db *grafeo.Database, actual any) any {
	switch v := actual.(type) {
	case nil, bool:
		return v
	case []any:
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = nativeValue(t, db, item)
		}
		return out
	case map[string]any:
		for kind, inner := range v {
			props := func(p any) map[string]any {
				out := map[string]any{}
				for k, item := range p.(map[string]any) {
					out[k] = nativeValue(t, db, item)
				}
				return out
			}
			switch kind {
			case "int":
				return json.Number(inner.(string))
			case "float":
				switch inner.(string) {
				case "NaN":
					return math.NaN()
				case "Inf":
					return math.Inf(1)
				case "-Inf":
					return math.Inf(-1)
				}
				f, _ := strconv.ParseFloat(inner.(string), 64)
				text := strconv.FormatFloat(f, 'f', -1, 64)
				if !strings.ContainsAny(text, ".eE") {
					text += ".0"
				}
				return json.Number(text)
			case "string":
				return inner
			case "map":
				return props(inner)
			case "node":
				n := inner.(map[string]any)
				out := props(n["properties"])
				out["_id"] = json.Number(fmt.Sprint(n["id"]))
				out["_labels"] = n["labels"]
				return out
			case "relationship":
				r := inner.(map[string]any)
				out := props(r["properties"])
				out["_id"] = json.Number(fmt.Sprint(r["id"]))
				out["_type"] = r["type"]
				out["_source"] = json.Number(fmt.Sprint(r["source"]))
				out["_target"] = json.Number(fmt.Sprint(r["target"]))
				return out
			case "path":
				p := inner.(map[string]any)
				var nodes, edges []any
				for _, n := range p["nodes"].([]any) {
					nodes = append(nodes, nativeValue(t, db, map[string]any{"node": n}))
				}
				for _, r := range p["relationships"].([]any) {
					edges = append(edges, nativeValue(t, db, map[string]any{"relationship": r}))
				}
				return map[string]any{"$path": map[string]any{"nodes": nodes, "edges": edges}}
			case "cypher":
				result, err := db.ExecuteCypher("RETURN " + inner.(string) + " AS v")
				if err != nil {
					t.Fatalf("%v", err)
				}
				return result.Rows[0]["v"]
			}
			t.Fatalf("unknown neutral kind %s", kind)
		}
	}
	t.Fatalf("not a neutral value: %v", actual)
	return nil
}

func TestSpecValueCases(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "values_cases.json"))
	if err != nil {
		t.Fatalf("%v", err)
	}
	var file struct {
		Cases []struct {
			Cell           string            `json:"cell"`
			Actual         any               `json:"actual"`
			Match          bool              `json:"match"`
			Dialect        string            `json:"dialect"`
			UnorderedLists bool              `json:"unordered_lists"`
			RunnerGaps     map[string]string `json:"runner_gaps"`
		} `json:"cases"`
		RoundTrips []struct {
			Cypher     string            `json:"cypher"`
			Text       string            `json:"text"`
			RunnerGaps map[string]string `json:"runner_gaps"`
		} `json:"round_trips"`
		Params []struct {
			Param string         `json:"param"`
			Value map[string]any `json:"value"`
		} `json:"params"`
	}
	if err := json.Unmarshal(data, &file); err != nil {
		t.Fatalf("%v", err)
	}
	db, err := grafeo.OpenInMemory()
	if err != nil {
		t.Fatalf("%v", err)
	}
	defer db.Close()
	for _, c := range file.Cases {
		cell, err := parseTck(c.Cell)
		if err != nil {
			t.Fatalf("%s: %v", c.Cell, err)
		}
		value := nativeValue(t, db, c.Actual)
		if c.Dialect == "tck" {
			if value, err = temporalsAsText(db, value); err != nil {
				t.Fatalf("%v", err)
			}
		}
		// A gap of this binding gives the opposite verdict until fixed
		_, gap := c.RunnerGaps["go"]
		if matchTck(cell, value, c.UnorderedLists) != (c.Match != gap) {
			t.Errorf("%s vs %v: verdict differs from the shared case", c.Cell, c.Actual)
		}
	}
	for _, trip := range file.RoundTrips {
		value := nativeValue(t, db, map[string]any{"cypher": trip.Cypher})
		text, err := temporalsAsText(db, value)
		_, gap := trip.RunnerGaps["go"]
		if (err == nil && text == trip.Text) == gap {
			t.Errorf("round trip %s: %v (%v), gap %v", trip.Cypher, text, err, gap)
		}
	}
	for _, p := range file.Params {
		value, err := paramValue(p.Param)
		if err != nil {
			t.Fatalf("%v", err)
		}
		want := any(nil)
		if p.Value != nil {
			want = p.Value["string"]
		}
		if value != want {
			t.Errorf("param %s: %v, expected %v", p.Param, value, want)
		}
	}
}
