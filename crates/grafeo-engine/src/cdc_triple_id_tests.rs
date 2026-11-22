//! Golden values of the triple id (#679): the first 8 bytes, little endian, of SHA-256 over the
//! documented canonical encoding. Per term a one-byte tag (0 IRI, 1 blank node, 2 literal), then
//! as u64 little endian length plus UTF-8 bytes the IRI, the blank label (without `_:`), or the
//! literal's lexical form, datatype IRI and language tag (length 0 without one); then one tag
//! byte for the graph, 0 for the default graph and 1 for a named one, followed for a named graph
//! by its storage key the same way. So a named graph with an empty name (one an older release
//! made) does not share ids with the default graph. The values were computed outside
//! the engine from that description; the id is the same in every process and release.

use std::fmt::Write as _;

use grafeo_common::storage::log_record::{TermRecord, TripleRecord};

use super::{canonical_triple_bytes, triple_id};

const XSD: &str = "http://www.w3.org/2001/XMLSchema#";
const LANG_STRING: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString";

fn iri(text: &str) -> TermRecord {
    TermRecord::Iri(text.to_string())
}

fn literal(value: &str, datatype: &str, language: Option<&str>) -> TermRecord {
    TermRecord::Literal {
        value: value.to_string(),
        datatype: datatype.to_string(),
        language: language.map(str::to_string),
    }
}

fn triple(subject: TermRecord, predicate: TermRecord, object: TermRecord) -> TripleRecord {
    TripleRecord {
        subject,
        predicate,
        object,
    }
}

fn alix_knows_gus() -> TripleRecord {
    triple(
        iri("http://ex.org/alix"),
        iri("http://ex.org/knows"),
        iri("http://ex.org/gus"),
    )
}

#[test]
fn the_canonical_bytes_are_the_documented_ones() {
    let expected = "00120000000000000068747470\
                    3a2f2f65782e6f72672f616c6978\
                    00130000000000000068747470\
                    3a2f2f65782e6f72672f6b6e6f7773\
                    00110000000000000068747470\
                    3a2f2f65782e6f72672f677573\
                    00";
    let bytes = canonical_triple_bytes(&alix_knows_gus(), None).iter().fold(
        String::new(),
        |mut hex, byte| {
            write!(hex, "{byte:02x}").unwrap();
            hex
        },
    );
    assert_eq!(bytes, expected.replace(char::is_whitespace, ""));
}

#[test]
fn every_term_kind_has_its_golden_id() {
    let cases: [(&str, TripleRecord, Option<&str>, u64); 7] = [
        (
            "three IRIs",
            alix_knows_gus(),
            None,
            8_198_305_858_944_992_339,
        ),
        (
            "in a named graph",
            alix_knows_gus(),
            Some("people"),
            9_130_729_143_145_070_290,
        ),
        (
            "in a graph inside a schema",
            alix_knows_gus(),
            Some("travel/people"),
            14_748_202_074_365_181_223,
        ),
        (
            "in a named graph with an empty name",
            alix_knows_gus(),
            Some(""),
            2_743_233_947_580_612_434,
        ),
        (
            "a blank node and a plain literal",
            triple(
                TermRecord::Blank("b1".to_string()),
                iri("http://ex.org/name"),
                literal("Alix", &format!("{XSD}string"), None),
            ),
            None,
            358_890_899_194_566_529,
        ),
        (
            "a typed literal",
            triple(
                iri("http://ex.org/alix"),
                iri("http://ex.org/age"),
                literal("30", &format!("{XSD}integer"), None),
            ),
            None,
            10_121_511_245_416_736_000,
        ),
        (
            "a language-tagged literal",
            triple(
                iri("http://ex.org/alix"),
                iri("http://ex.org/greets"),
                literal("hallo", LANG_STRING, Some("nl")),
            ),
            None,
            14_679_000_264_070_144_688,
        ),
    ];
    for (what, record, graph, id) in cases {
        assert_eq!(triple_id(&record, graph), id, "{what}");
    }
}

/// The same triple in two graphs (an empty-named one included), and with a language tag or without
/// one, has two ids.
#[test]
fn the_graph_and_the_language_tag_are_part_of_the_id() {
    assert_ne!(
        triple_id(&alix_knows_gus(), None),
        triple_id(&alix_knows_gus(), Some("people"))
    );
    assert_ne!(
        triple_id(&alix_knows_gus(), None),
        triple_id(&alix_knows_gus(), Some("")),
        "the graph's tag keeps an empty-named graph apart from the default"
    );
    let tagged = |language: Option<&str>, datatype: &str| {
        triple(
            iri("http://ex.org/alix"),
            iri("http://ex.org/greets"),
            literal("hallo", datatype, language),
        )
    };
    assert_ne!(
        triple_id(&tagged(Some("nl"), LANG_STRING), None),
        triple_id(&tagged(None, &format!("{XSD}string")), None)
    );
}
