package grafeo

import (
	"encoding/json"
	"strings"
)

// RdfTermType is the kind of an RDF term.
type RdfTermType string

const (
	// IRI is an IRI.
	IRI RdfTermType = "uri"
	// Literal is a literal: its value with a language tag or a datatype.
	Literal RdfTermType = "literal"
	// BlankNode is a blank node: its value is the label, without "_:".
	BlankNode RdfTermType = "bnode"
)

// The datatypes of simple and language-tagged literals.
const (
	XSDString     = "http://www.w3.org/2001/XMLSchema#string"
	RDFLangString = "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString"
)

// RdfTerm is an RDF term as a SPARQL result holds it: an IRI, a blank node,
// or a literal with its language tag or datatype. As a query parameter it is
// that exact term.
//
// Language tags compare without case, so the constructors keep a tag in
// lowercase ("Pomme"@FR is "Pomme"@fr) and a literal of datatype xsd:string
// is the simple literal; Equal compares two terms that way, also terms built
// as struct literals.
type RdfTerm struct {
	Type  RdfTermType
	Value string
	// Language is the tag of a language-tagged literal, in lowercase, ""
	// otherwise.
	Language string
	// Datatype is the datatype IRI of a typed literal other than xsd:string,
	// "" otherwise.
	Datatype string
}

// NewIRI returns the IRI iri.
func NewIRI(iri string) RdfTerm {
	return RdfTerm{Type: IRI, Value: iri}
}

// NewBlankNode returns the blank node with label, without "_:".
func NewBlankNode(label string) RdfTerm {
	return RdfTerm{Type: BlankNode, Value: label}
}

// NewLiteral returns the simple literal value (of datatype xsd:string).
func NewLiteral(value string) RdfTerm {
	return RdfTerm{Type: Literal, Value: value}
}

// NewLangLiteral returns the literal value tagged with language, the tag in
// lowercase.
func NewLangLiteral(value, language string) RdfTerm {
	return RdfTerm{Type: Literal, Value: value, Language: language}.Normalized()
}

// NewTypedLiteral returns the literal value of datatype, the simple literal
// for xsd:string.
func NewTypedLiteral(value, datatype string) RdfTerm {
	return RdfTerm{Type: Literal, Value: value, Datatype: datatype}.Normalized()
}

// Normalized returns the term as the constructors make it: a literal's
// language tag in lowercase, no datatype for a tagged literal or one of
// datatype xsd:string, and no language or datatype for an IRI or a blank
// node.
func (t RdfTerm) Normalized() RdfTerm {
	if t.Type != Literal {
		return RdfTerm{Type: t.Type, Value: t.Value}
	}
	t.Language = strings.ToLower(t.Language)
	if t.Language != "" || t.Datatype == XSDString {
		t.Datatype = ""
	}
	return t
}

// Equal reports whether two terms are the same RDF term: the same kind and
// value, and for a literal the same language tag (without case) and
// datatype.
func (t RdfTerm) Equal(other RdfTerm) bool {
	return t.Normalized() == other.Normalized()
}

// String returns the term's text: the IRI, the blank node's label or the
// literal's lexical form.
func (t RdfTerm) String() string {
	return t.Value
}

// LiteralDatatype returns a literal's datatype as RDF reads it: rdf:langString
// for a tagged literal, xsd:string for a simple one, "" for an IRI or a blank
// node.
func (t RdfTerm) LiteralDatatype() string {
	switch {
	case t.Type != Literal:
		return ""
	case t.Language != "":
		return RDFLangString
	case t.Datatype == "":
		return XSDString
	default:
		return t.Datatype
	}
}

// termTag is the key of the tag an RDF term travels in as JSON.
const termTag = "$term"

// MarshalJSON writes the term in the JSON form the C library reads: the
// "$term" tag as the object's only key, holding "type" ("uri", "literal" or
// "bnode") and "value", and for a literal "xml:lang" or "datatype".
func (t RdfTerm) MarshalJSON() ([]byte, error) {
	t = t.Normalized()
	term := map[string]string{"type": string(t.Type), "value": t.Value}
	if t.Language != "" {
		term["xml:lang"] = t.Language
	} else if t.Datatype != "" {
		term["datatype"] = t.Datatype
	}
	return json.Marshal(map[string]any{termTag: term})
}

// rdfTermOf reads a map in the form MarshalJSON writes: the "$term" tag as
// the map's only key, holding "type" ("uri", "literal" or "bnode") and
// "value", and for a literal "xml:lang" or "datatype", and no other key. A
// map with these keys but no "$term" tag is no term.
func rdfTermOf(tagged map[string]any) (RdfTerm, bool) {
	v, ok := tagged[termTag].(map[string]any)
	if !ok || len(tagged) != 1 {
		return RdfTerm{}, false
	}
	typ, ok := v["type"].(string)
	if !ok {
		return RdfTerm{}, false
	}
	value, ok := v["value"].(string)
	if !ok {
		return RdfTerm{}, false
	}
	switch RdfTermType(typ) {
	case IRI, BlankNode:
		if len(v) != 2 {
			return RdfTerm{}, false
		}
		return RdfTerm{Type: RdfTermType(typ), Value: value}, true
	case Literal:
		term := RdfTerm{Type: Literal, Value: value}
		extra := 2
		if lang, ok := v["xml:lang"].(string); ok {
			term.Language = lang
			extra++
		}
		if datatype, ok := v["datatype"].(string); ok {
			term.Datatype = datatype
			extra++
		}
		if len(v) != extra || (term.Language != "" && term.Datatype != "") {
			return RdfTerm{}, false
		}
		return term.Normalized(), true
	default:
		return RdfTerm{}, false
	}
}

// ResultKind is what a QueryResult holds.
type ResultKind int32

const (
	// ResultRows is rows: SPARQL SELECT and updates, and every other language.
	ResultRows ResultKind = 0
	// ResultBoolean is one boolean: SPARQL ASK, one row with a "boolean" column.
	ResultBoolean ResultKind = 1
	// ResultGraph is an RDF graph: SPARQL CONSTRUCT and DESCRIBE, rows of
	// "subject", "predicate" and "object" terms.
	ResultGraph ResultKind = 2
)
