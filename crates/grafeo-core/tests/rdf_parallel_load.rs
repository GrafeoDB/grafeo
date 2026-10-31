//! The parallel RDF load (0.6.1): N-Triples text of 1 MiB or more is parsed
//! in runs of whole lines at once, and a batch of 4,096 or more new triples
//! builds the store's six indexes at once. Both must give what the sequential
//! path gives:
//!
//! - the same triples in the same order as parsing each line alone, with
//!   CRLF, comments, blank lines, escapes, blank nodes, typed and tagged
//!   literals, and a last line without a newline;
//! - an error naming the document's line, counting comments and blank lines,
//!   in a later run too; with two bad lines, the first;
//! - a batch with duplicates inside it and against the store: the same new
//!   count, length and lookups (by subject, predicate, object and the three
//!   pairs) as inserting the triples one by one, just below, at and above
//!   4,096 new triples, and with the object index off.
//!
//! ```bash
//! cargo test -p grafeo-core --all-features --test rdf_parallel_load
//! ```

#![cfg(feature = "triple-store")]

use std::collections::{BTreeSet, HashSet};

use grafeo_core::graph::rdf::{
    NTriplesError, RdfStore, RdfStoreConfig, Term, Triple, TriplePattern, parse_ntriples,
    parse_ntriples_text,
};

/// Lines of every shape, `count` triples in all, and the line numbers
/// (1-based) of the triple lines.
fn document(count: usize) -> (String, Vec<usize>) {
    let mut text = String::new();
    let mut triple_lines = Vec::with_capacity(count);
    let mut line = 0;
    let mut push = |text: &mut String, content: &str, crlf: bool| {
        text.push_str(content);
        text.push_str(if crlf { "\r\n" } else { "\n" });
        line += 1;
        line
    };
    for i in 0..count {
        if i % 50 == 0 {
            push(&mut text, &format!("# people from {i}"), i % 100 == 0);
        }
        if i % 13 == 0 {
            push(&mut text, "", false);
        }
        if i % 29 == 0 {
            push(&mut text, "   ", true);
        }
        let subject = if i % 11 == 0 {
            format!("_:b{i}")
        } else {
            format!("<http://ex.org/p{}>", i % 997)
        };
        let object = match i % 6 {
            0 => format!("\"Person {i}\""),
            1 => format!("\"{}\"^^<http://www.w3.org/2001/XMLSchema#integer>", i % 70),
            2 => format!("\"Zo\u{eb} {i}\"@nl"),
            3 => format!("\"say \\\"hi\\\"\\n{i}\""),
            4 => format!("<http://ex.org/city/{}>", i % 5),
            _ => format!("_:b{}", i % 37),
        };
        let content = format!("{subject} <http://ex.org/p{}> {object} .", i % 7);
        triple_lines.push(push(&mut text, &content, i % 7 == 0));
    }
    // The last line without a line break.
    if text.ends_with("\r\n") {
        text.truncate(text.len() - 2);
    } else {
        text.truncate(text.len() - 1);
    }
    (text, triple_lines)
}

/// The document parsed one line at a time: each line is far below the
/// parallel threshold.
fn line_by_line(text: &str) -> Vec<Triple> {
    text.lines()
        .flat_map(|line| parse_ntriples_text(line).unwrap_or_else(|e| panic!("{line}: {e}")))
        .collect()
}

fn parse_error(result: Result<Vec<Triple>, NTriplesError>) -> (usize, String) {
    match result {
        Err(NTriplesError::Parse { line, content }) => (line, content),
        Err(other) => panic!("not a parse error: {other}"),
        Ok(triples) => panic!("{} triples, no error", triples.len()),
    }
}

/// The document with the triple lines `bad` (1-based document lines)
/// replaced by text that is no triple.
fn with_bad_lines(text: &str, bad: &[usize]) -> String {
    text.split_inclusive('\n')
        .enumerate()
        .map(|(index, line)| {
            if bad.contains(&(index + 1)) {
                let ending = &line[line.trim_end_matches(['\r', '\n']).len()..];
                format!("<http://ex.org/bad{}> not a triple{ending}", index + 1)
            } else {
                line.to_string()
            }
        })
        .collect()
}

#[test]
fn a_large_document_parses_as_its_lines_do() {
    let (text, triple_lines) = document(40_000);
    assert!(text.len() > 3 << 19, "{} bytes", text.len());
    let parsed = parse_ntriples_text(&text).unwrap();
    assert_eq!(parsed.len(), triple_lines.len());
    assert_eq!(parsed, line_by_line(&text), "same triples, same order");
    assert_eq!(parse_ntriples(text.as_bytes()).unwrap(), parsed);
}

#[test]
fn a_small_document_parses_as_its_lines_do() {
    let (text, triple_lines) = document(300);
    assert!(text.len() < 1 << 20);
    let parsed = parse_ntriples_text(&text).unwrap();
    assert_eq!(parsed.len(), triple_lines.len());
    assert_eq!(parsed, line_by_line(&text));
}

#[test]
fn an_error_names_the_document_line() {
    let (text, triple_lines) = document(40_000);
    let last = *triple_lines.last().unwrap();
    for bad in [
        triple_lines[3],
        triple_lines[triple_lines.len() / 2],
        triple_lines[triple_lines.len() - 7],
        last,
    ] {
        let (line, content) = parse_error(parse_ntriples_text(&with_bad_lines(&text, &[bad])));
        assert_eq!(line, bad, "{content}");
        assert!(
            content.contains(&format!("<http://ex.org/bad{bad}> not a triple")),
            "{content}"
        );
    }
}

#[test]
fn with_two_bad_lines_the_first_is_named() {
    let (text, triple_lines) = document(40_000);
    let n = triple_lines.len();
    for pair in [
        [triple_lines[n / 2], triple_lines[n - 3]],
        [triple_lines[2], triple_lines[n - 3]],
        [triple_lines[n / 3], triple_lines[n / 3 + 1]],
    ] {
        let (line, _) = parse_error(parse_ntriples_text(&with_bad_lines(&text, &pair)));
        assert_eq!(line, pair[0], "bad lines {pair:?}");
    }
}

#[test]
fn a_small_and_a_large_document_name_the_same_line() {
    let (text, triple_lines) = document(300);
    let bad = triple_lines[200];
    let small = with_bad_lines(&text, &[bad]);
    // The same lines, then enough good ones to pass the parallel threshold.
    let (tail, _) = document(40_000);
    let large = format!("{small}\n{tail}");
    assert!(large.len() > 3 << 19);
    assert_eq!(parse_error(parse_ntriples_text(&small)).0, bad);
    assert_eq!(parse_error(parse_ntriples_text(&large)).0, bad);
}

/// `count` distinct triples over a few hundred subjects, predicates and
/// objects, so the pair indexes hold lists.
fn distinct(count: usize, tag: &str) -> Vec<Triple> {
    (0..count)
        .map(|i| {
            let object = if i % 3 == 0 {
                Term::iri(format!("http://ex.org/{tag}o{}", i % 101))
            } else {
                Term::literal(format!("{tag} {i}"))
            };
            Triple::new(
                Term::iri(format!("http://ex.org/{tag}s{}", i % 313)),
                Term::iri(format!("http://ex.org/p{}", i % 7)),
                object,
            )
        })
        .collect()
}

fn names(triples: Vec<std::sync::Arc<Triple>>) -> BTreeSet<String> {
    triples.iter().map(|t| t.to_string()).collect()
}

/// Every lookup of `batched` equals the same lookup of `single`: each distinct subject,
/// predicate and object of `triples`, each pair, and each whole triple.
fn same_lookups(batched: &RdfStore, single: &RdfStore, triples: &[Triple]) {
    assert_eq!(batched.len(), single.len());
    assert_eq!(names(batched.triples()), names(single.triples()));
    let mut subjects = HashSet::new();
    let mut predicates = HashSet::new();
    let mut objects = HashSet::new();
    let mut patterns = HashSet::new();
    for triple in triples {
        let (s, p, o) = (triple.subject(), triple.predicate(), triple.object());
        subjects.insert(s.clone());
        predicates.insert(p.clone());
        objects.insert(o.clone());
        patterns.insert((Some(s.clone()), Some(p.clone()), None));
        patterns.insert((None, Some(p.clone()), Some(o.clone())));
        patterns.insert((Some(s.clone()), None, Some(o.clone())));
        patterns.insert((Some(s.clone()), Some(p.clone()), Some(o.clone())));
    }
    for s in &subjects {
        assert_eq!(
            names(batched.triples_with_subject(s)),
            names(single.triples_with_subject(s)),
            "{s}"
        );
    }
    for p in &predicates {
        assert_eq!(
            names(batched.triples_with_predicate(p)),
            names(single.triples_with_predicate(p)),
            "{p}"
        );
    }
    for o in &objects {
        assert_eq!(
            names(batched.triples_with_object(o)),
            names(single.triples_with_object(o)),
            "{o}"
        );
    }
    for (subject, predicate, object) in patterns {
        let pattern = TriplePattern {
            subject,
            predicate,
            object,
        };
        let found = names(batched.find(&pattern));
        assert!(!found.is_empty(), "{pattern:?}");
        assert_eq!(found, names(single.find(&pattern)), "{pattern:?}");
    }
}

/// A batch of `new` triples new to the store, each new one twice in the
/// batch, plus 500 the store holds already: the batch against one insert
/// per triple.
fn batch_against_one_by_one(new: usize, config: impl Fn() -> RdfStoreConfig) {
    let before = distinct(500, "old");
    let fresh = distinct(new, "new");
    let mut batch: Vec<Triple> = fresh.iter().chain(&before).cloned().collect();
    batch.extend(fresh.iter().rev().cloned());

    let batched = RdfStore::with_config(config());
    let one_by_one = RdfStore::with_config(config());
    for triple in &before {
        assert!(batched.insert(triple.clone()));
        assert!(one_by_one.insert(triple.clone()));
    }
    assert_eq!(batched.batch_insert(batch.clone()), new, "{new} new");
    let inserted = batch
        .iter()
        .filter(|triple| one_by_one.insert((*triple).clone()))
        .count();
    assert_eq!(inserted, new);
    assert_eq!(batched.len(), 500 + new);
    same_lookups(&batched, &one_by_one, &batch);
}

#[test]
fn a_batch_indexes_as_single_inserts_do() {
    for new in [4_095, 4_096, 6_000] {
        batch_against_one_by_one(new, RdfStoreConfig::default);
    }
}

#[test]
fn a_batch_without_an_object_index_indexes_as_single_inserts_do() {
    batch_against_one_by_one(6_000, || RdfStoreConfig {
        index_objects: false,
        ..RdfStoreConfig::default()
    });
}

#[test]
fn a_second_large_batch_adds_to_the_first() {
    let first = distinct(5_000, "a");
    let second = distinct(5_000, "b");
    let batched = RdfStore::new();
    let one_by_one = RdfStore::new();
    assert_eq!(batched.batch_insert(first.clone()), 5_000);
    assert_eq!(batched.batch_insert(second.clone()), 5_000);
    for triple in first.iter().chain(&second) {
        one_by_one.insert(triple.clone());
    }
    let all: Vec<Triple> = first.into_iter().chain(second).collect();
    same_lookups(&batched, &one_by_one, &all);
}
