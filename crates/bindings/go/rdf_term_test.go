package grafeo

import (
	"encoding/json"
	"testing"
)

// SPARQL results hold exact RDF terms: kind, value, language tag and datatype.
func TestSparqlResultsHoldRdfTerms(t *testing.T) {
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.ExecuteSPARQL(`INSERT DATA {
		<http://example.org/apple> <http://example.org/label> "Pomme"@fr ;
			<http://example.org/price> "1.50"^^<http://www.w3.org/2001/XMLSchema#decimal> ;
			<http://example.org/note> "fresh" ;
			<http://example.org/grower> _:farm .
	}`); err != nil {
		t.Skipf("SPARQL unavailable in this build: %v", err)
	}
	result, err := db.ExecuteSPARQL(`SELECT ?p ?o WHERE { <http://example.org/apple> ?p ?o }`)
	if err != nil {
		t.Fatal(err)
	}
	if result.Kind != ResultRows {
		t.Errorf("a SELECT holds rows, got %d", result.Kind)
	}
	objects := map[string]RdfTerm{}
	for _, row := range result.Rows {
		predicate, ok := row["p"].(RdfTerm)
		if !ok || predicate.Type != IRI {
			t.Fatalf("a predicate is an IRI, got %T %v", row["p"], row["p"])
		}
		object, ok := row["o"].(RdfTerm)
		if !ok {
			t.Fatalf("an object is a term, got %T %v", row["o"], row["o"])
		}
		objects[predicate.Value] = object
	}
	want := map[string]RdfTerm{
		"http://example.org/label": {Type: Literal, Value: "Pomme", Language: "fr"},
		"http://example.org/price": {Type: Literal, Value: "1.50", Datatype: "http://www.w3.org/2001/XMLSchema#decimal"},
		"http://example.org/note":  {Type: Literal, Value: "fresh"},
	}
	for predicate, term := range want {
		if objects[predicate] != term {
			t.Errorf("%s: got %+v, want %+v", predicate, objects[predicate], term)
		}
	}
	if objects["http://example.org/label"].LiteralDatatype() != RDFLangString {
		t.Errorf("a tagged literal is an rdf:langString")
	}
	if grower := objects["http://example.org/grower"]; grower.Type != BlankNode || grower.Value == "" {
		t.Errorf("the grower is a blank node, got %+v", grower)
	}

	ask, err := db.ExecuteSPARQL(`ASK { ?s <http://example.org/label> "Pomme"@fr }`)
	if err != nil {
		t.Fatal(err)
	}
	if answer, ok := ask.Boolean(); ask.Kind != ResultBoolean || !ok || !answer {
		t.Errorf("ASK answers true, got kind %d %v %v", ask.Kind, answer, ok)
	}
	graph, err := db.ExecuteSPARQL(`CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }`)
	if err != nil {
		t.Fatal(err)
	}
	if graph.Kind != ResultGraph || len(graph.Rows) != 4 {
		t.Errorf("CONSTRUCT is a graph of 4 triples, got kind %d and %d rows", graph.Kind, len(graph.Rows))
	}
}

// An RdfTerm is written in the $term tag and read back as the same term.
func TestRdfTermJSONRoundTrip(t *testing.T) {
	for _, term := range []RdfTerm{
		NewIRI("http://example.org/Kraków"),
		NewBlankNode("b0"),
		NewLiteral("Apple"),
		NewLangLiteral("Pomme", "fr"),
		NewTypedLiteral("2026-01-15", "http://www.w3.org/2001/XMLSchema#date"),
	} {
		text, err := json.Marshal(term)
		if err != nil {
			t.Fatal(err)
		}
		var raw any
		if err := json.Unmarshal(text, &raw); err != nil {
			t.Fatal(err)
		}
		if back := decodeTags(raw); back != term {
			t.Errorf("%s came back as %#v", text, back)
		}
	}
	text, err := json.Marshal(NewLangLiteral("Pomme", "fr"))
	if err != nil {
		t.Fatal(err)
	}
	if want := `{"$term":{"type":"literal","value":"Pomme","xml:lang":"fr"}}`; string(text) != want {
		t.Errorf("got %s, want %s", text, want)
	}
}

// A map with the keys of a SPARQL JSON term ("type" and "value") is a map,
// not a term, both read from the C library and written as a parameter; so is
// a $term tag with other keys or content.
func TestMapsShapedLikeTermsStayMaps(t *testing.T) {
	for _, m := range []map[string]any{
		{"type": "uri", "value": "http://example.org/Paris"},
		{"type": "literal", "value": "Pomme", "xml:lang": "fr"},
		{"type": "bnode", "value": "b0"},
		{"type": "uri", "value": "x", "extra": 1.0},
		{"$term": map[string]any{"type": "uri", "value": "x"}, "other": 1.0},
		{"$term": map[string]any{"type": "page", "value": "x"}},
		{"$term": map[string]any{"type": "uri", "value": "x", "xml:lang": "en"}},
	} {
		if _, ok := decodeTags(m).(map[string]any); !ok {
			t.Errorf("%v is no term", m)
		}
	}

	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	place := map[string]any{"type": "uri", "value": "http://example.org/Paris"}
	result, err := db.ExecuteParams("RETURN $m AS m, $m.type AS kind", map[string]any{"m": place})
	if err != nil {
		t.Fatal(err)
	}
	back, ok := result.Rows[0]["m"].(map[string]any)
	if !ok || back["type"] != "uri" || back["value"] != "http://example.org/Paris" || len(back) != 2 {
		t.Errorf("the map comes back a map, got %T %v", result.Rows[0]["m"], result.Rows[0]["m"])
	}
	if result.Rows[0]["kind"] != "uri" {
		t.Errorf("the engine reads it as a map, got %v", result.Rows[0]["kind"])
	}
	term, err := db.ExecuteParams("RETURN $t AS t", map[string]any{"t": NewLangLiteral("Pomme", "fr")})
	if err != nil {
		t.Fatal(err)
	}
	if got := term.Rows[0]["t"]; got != NewLangLiteral("Pomme", "fr") {
		t.Errorf("a term parameter comes back the term, got %T %v", got, got)
	}
}

// Language tags compare without case: the constructors and the decoding keep
// a tag in lowercase, Equal compares any two terms so, and a literal of
// datatype xsd:string is the simple literal.
func TestRdfTermsNormalizeTagsAndXsdString(t *testing.T) {
	upper := NewLangLiteral("Pomme", "FR")
	if upper.Language != "fr" || upper != NewLangLiteral("Pomme", "fr") {
		t.Errorf("the tag is kept in lowercase, got %+v", upper)
	}
	if NewLangLiteral("Colour", "en-GB").Language != "en-gb" {
		t.Error("a tag with a region is kept in lowercase")
	}
	literal := RdfTerm{Type: Literal, Value: "Pomme", Language: "FR"}
	if !literal.Equal(NewLangLiteral("Pomme", "fr")) {
		t.Error("a struct literal with a tag in capitals is the term with the tag in lowercase")
	}
	if literal.Equal(NewLangLiteral("Pomme", "nl")) {
		t.Error("another tag is another term")
	}
	if NewTypedLiteral("Apple", XSDString) != NewLiteral("Apple") {
		t.Error("a literal of datatype xsd:string is the simple literal")
	}
	if !(RdfTerm{Type: Literal, Value: "Apple", Datatype: XSDString}).Equal(NewLiteral("Apple")) {
		t.Error("Equal reads xsd:string as the simple literal")
	}
	decoded := decodeTags(map[string]any{
		"$term": map[string]any{"type": "literal", "value": "Pomme", "xml:lang": "FR"},
	})
	if decoded != NewLangLiteral("Pomme", "fr") {
		t.Errorf("a decoded tag is kept in lowercase, got %#v", decoded)
	}
	text, err := json.Marshal(literal)
	if err != nil {
		t.Fatal(err)
	}
	if want := `{"$term":{"type":"literal","value":"Pomme","xml:lang":"fr"}}`; string(text) != want {
		t.Errorf("a parameter is sent with the tag in lowercase, got %s", text)
	}
}
