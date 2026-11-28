//! SPARQL triangles and 4-cycles through a person, in graph-bench's text (each KNOWS step the path
//! `(v:KNOWS|^v:KNOWS)`, the conditions on the id literals), count what a walk over the KNOWS
//! triples counts, however the patterns are ordered and joined.

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use std::collections::BTreeSet;
use std::fmt::Write as _;

use grafeo_engine::{GrafeoDB, RdfFormat};

const V: &str = "http://graph-bench.example/v/";

const PERSONS: usize = 60;

/// The KNOWS triples: four links out of each person, no link to itself.
fn knows() -> BTreeSet<(usize, usize)> {
    (0..PERSONS)
        .flat_map(|i| (0..4).map(move |k| (i, (i * 7 + k * 13 + 3) % PERSONS)))
        .filter(|(i, j)| i != j)
        .collect()
}

fn store() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let mut text = String::new();
    for i in 0..PERSONS {
        writeln!(
            text,
            "<{V}p{i}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{V}Person> .\n\
             <{V}p{i}> <{V}id> \"p{i}\" ."
        )
        .unwrap();
    }
    for (i, j) in knows() {
        writeln!(text, "<{V}p{i}> <{V}KNOWS> <{V}p{j}> .").unwrap();
    }
    db.load_rdf_text(&text, RdfFormat::NTriples, None).unwrap();
    db
}

/// Each step of a walk: one per KNOWS triple and direction.
fn steps(from: usize) -> Vec<usize> {
    knows()
        .into_iter()
        .filter_map(|(i, j)| {
            if i == from {
                Some(j)
            } else if j == from {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

fn id(person: usize) -> String {
    format!("p{person}")
}

fn count(db: &GrafeoDB, query: &str) -> usize {
    let result = db
        .execute_sparql(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    // An integer literal, `"7"^^<...#integer>`, or a plain number.
    let text = format!("{}", result.rows()[0][0]);
    let digits = text.split('"').nth(1).unwrap_or(&text);
    digits.parse().unwrap_or_else(|_| panic!("a count: {text}"))
}

#[test]
fn triangles_through_a_person_count_the_walks() {
    let db = store();
    for a in 0..10 {
        let mut expected = 0;
        for b in steps(a) {
            for c in steps(b) {
                if id(b) < id(c) {
                    expected += steps(c).iter().filter(|&&back| back == a).count();
                }
            }
        }
        let query = format!(
            "PREFIX v: <{V}>
             SELECT (COUNT(*) AS ?triangles) WHERE {{
               VALUES ?personId {{ \"{}\" }}
               ?a a v:Person ; v:id ?personId ; (v:KNOWS|^v:KNOWS) ?b .
               ?b a v:Person ; v:id ?bid ; (v:KNOWS|^v:KNOWS) ?c .
               ?c a v:Person ; v:id ?cid ; (v:KNOWS|^v:KNOWS) ?a .
               FILTER(?bid < ?cid)
             }}",
            id(a)
        );
        assert_eq!(count(&db, &query), expected, "triangles through p{a}");
    }
}

#[test]
fn four_cycles_through_a_person_count_the_walks() {
    let db = store();
    for a in 0..10 {
        let mut expected = 0;
        for b in steps(a) {
            for c in steps(b) {
                if c == a {
                    continue;
                }
                for d in steps(c) {
                    if id(b) < id(d) {
                        expected += steps(d).iter().filter(|&&back| back == a).count();
                    }
                }
            }
        }
        let query = format!(
            "PREFIX v: <{V}>
             SELECT (COUNT(*) AS ?cycles) WHERE {{
               VALUES ?personId {{ \"{}\" }}
               ?a a v:Person ; v:id ?personId ; (v:KNOWS|^v:KNOWS) ?b .
               ?b a v:Person ; v:id ?bid ; (v:KNOWS|^v:KNOWS) ?c .
               ?c a v:Person ; v:id ?cid ; (v:KNOWS|^v:KNOWS) ?d .
               ?d a v:Person ; v:id ?did ; (v:KNOWS|^v:KNOWS) ?a .
               FILTER(?bid < ?did && ?cid != ?personId)
             }}",
            id(a)
        );
        assert_eq!(count(&db, &query), expected, "4-cycles through p{a}");
    }
}
