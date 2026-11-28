//! A SPARQL cycle of four through a bound person is joined as two halves that meet at the person
//! opposite: each half reads the two-step paths on its side, where a left-deep chain reads every
//! three-step path. In every case, the split or not, the rows (b, c, d) are those a walk over the
//! KNOWS triples gives, as a bag: each KNOWS triple and direction one step, so a pair linked both
//! ways, a repeated `VALUES` row or a pattern written twice multiplies the rows as it should.

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use std::collections::BTreeSet;
use std::fmt::Write as _;

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB, RdfFormat};

const V: &str = "http://graph-bench.example/v/";
const PERSONS: usize = 40;
const STEP: &str = "(v:KNOWS|^v:KNOWS)";

/// Four links out of each person, a hub (`p0`) linked to every third person, and `p1` and `p2`
/// linked both ways.
fn knows() -> BTreeSet<(usize, usize)> {
    let mut knows: BTreeSet<(usize, usize)> = (0..PERSONS)
        .flat_map(|i| (0..4).map(move |k| (i, (i * 7 + k * 13 + 3) % PERSONS)))
        .filter(|(i, j)| i != j)
        .collect();
    knows.extend((3..PERSONS).step_by(3).map(|j| (0, j)));
    knows.extend([(1, 2), (2, 1)]);
    knows
}

/// The data: persons with an id, KNOWS, and one liked item each (`i % 5`).
fn text() -> String {
    let mut text = String::new();
    for i in 0..PERSONS {
        writeln!(
            text,
            "<{V}p{i}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{V}Person> .\n\
             <{V}p{i}> <{V}id> \"p{i}\" .\n\
             <{V}p{i}> <{V}likes> <{V}item{}> .",
            i % 5
        )
        .unwrap();
    }
    for (i, j) in knows() {
        writeln!(text, "<{V}p{i}> <{V}KNOWS> <{V}p{j}> .").unwrap();
    }
    text
}

fn store() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.load_rdf_text(&text(), RdfFormat::NTriples, None)
        .unwrap();
    db
}

/// The same data in the named graph `<urn:g>`, the default graph empty.
fn store_in_a_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.load_rdf_text(&text(), RdfFormat::NTriples, Some("urn:g"))
        .unwrap();
    db
}

/// The steps of a walk from `from`: one per KNOWS triple and direction.
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

/// Every walk a-b-c-d with what `close` counts of its closing from d to a, that many times.
fn walks(
    a: usize,
    close: impl Fn(usize, usize, usize, usize) -> usize,
) -> Vec<(usize, usize, usize)> {
    let mut rows = Vec::new();
    for b in steps(a) {
        for c in steps(b) {
            for d in steps(c) {
                for _ in 0..close(a, b, c, d) {
                    rows.push((b, c, d));
                }
            }
        }
    }
    rows.sort_unstable();
    rows
}

/// The walks a-b-c-d-a.
fn cycles(a: usize) -> Vec<(usize, usize, usize)> {
    walks(a, |a, _, _, d| {
        steps(d).iter().filter(|&&back| back == a).count()
    })
}

/// The person an IRI names.
fn person(value: &Value) -> usize {
    let text = format!("{value}");
    let digits: String = text
        .rsplit("/p")
        .next()
        .unwrap_or_default()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits
        .parse()
        .unwrap_or_else(|_| panic!("a person: {text}"))
}

fn query(body: &str) -> String {
    format!("PREFIX v: <{V}> SELECT ?b ?c ?d WHERE {{ {body} }}")
}

fn rows(db: &GrafeoDB, body: &str) -> Vec<(usize, usize, usize)> {
    let query = query(body);
    let result = db
        .execute_sparql(&query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    let mut rows: Vec<_> = result
        .rows()
        .iter()
        .map(|row| (person(&row[0]), person(&row[1]), person(&row[2])))
        .collect();
    rows.sort_unstable();
    rows
}

/// Whether the plan joins two halves: half d starts from the anchor's distinct rows, and none of
/// these queries asks for DISTINCT itself.
fn splits(db: &GrafeoDB, body: &str) -> bool {
    let query = format!("EXPLAIN {}", query(body));
    let result = db
        .execute_sparql(&query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    result
        .rows()
        .iter()
        .any(|row| format!("{}", row[0]).contains("Distinct"))
}

/// The 4-cycle through `p{a}` in graph-bench's text, with `close` as its last step from d to a.
fn cycle(anchor: &str, close: &str) -> String {
    format!(
        "{anchor}
         ?a a v:Person ; v:id ?personId ; {STEP} ?b .
         ?b a v:Person ; v:id ?bid ; {STEP} ?c .
         ?c a v:Person ; v:id ?cid ; {STEP} ?d .
         ?d a v:Person ; v:id ?did . {close}"
    )
}

fn anchored(a: usize) -> String {
    format!("VALUES ?personId {{ \"p{a}\" }}")
}

fn closing() -> String {
    format!("?d {STEP} ?a .")
}

#[test]
fn a_four_cycle_through_a_person_splits_and_gives_the_walks() {
    let db = store();
    // The hub, a person linked both ways, and two others.
    for a in [0, 1, 5, 17] {
        let body = cycle(&anchored(a), &closing());
        assert!(splits(&db, &body), "p{a} splits");
        assert_eq!(rows(&db, &body), cycles(a), "4-cycles through p{a}");
    }
}

#[test]
fn a_repeated_values_row_multiplies_the_split_rows_once() {
    let db = store();
    let body = cycle("VALUES ?personId { \"p5\" \"p5\" }", &closing());
    assert!(splits(&db, &body));
    let mut twice: Vec<_> = cycles(5).into_iter().flat_map(|row| [row, row]).collect();
    twice.sort_unstable();
    assert_eq!(rows(&db, &body), twice);
}

#[test]
fn a_written_union_closes_a_split_cycle() {
    let db = store();
    let body = cycle(&anchored(5), "{ ?d v:KNOWS ?a } UNION { ?a v:KNOWS ?d }");
    assert!(splits(&db, &body));
    assert_eq!(rows(&db, &body), cycles(5));
}

#[test]
fn a_cycle_in_a_named_graph_splits() {
    let db = store_in_a_graph();
    let fixed = format!("GRAPH <urn:g> {{ {} }}", cycle(&anchored(5), &closing()));
    assert!(splits(&db, &fixed));
    assert_eq!(rows(&db, &fixed), cycles(5));
    // A graph variable every pattern names joins the halves too.
    let variable = format!("GRAPH ?g {{ {} }}", cycle(&anchored(5), &closing()));
    assert!(splits(&db, &variable));
    assert_eq!(rows(&db, &variable), cycles(5));
}

/// The anchor outside `GRAPH ?g`: its `VALUES` names no graph. Where the anchor's own patterns
/// are inside, half d's anchor keeps `?g`; where the anchor is the `VALUES` alone, it does not,
/// and the halves join on the `?g` their edges bind. Both split, with the walks' rows.
#[test]
fn an_anchor_outside_the_graph_block_splits() {
    let db = store_in_a_graph();
    let inside = format!("{} GRAPH ?g {{ {} }}", anchored(5), cycle("", &closing()));
    assert!(splits(&db, &inside), "{inside}");
    assert_eq!(rows(&db, &inside), cycles(5));
    let alone = format!(
        "VALUES ?a {{ <{V}p5> }} GRAPH ?g {{ ?a {STEP} ?b . ?b {STEP} ?c . ?c {STEP} ?d . \
         ?d {STEP} ?a . }}"
    );
    assert!(splits(&db, &alone), "{alone}");
    assert_eq!(rows(&db, &alone), cycles(5));
}

#[test]
fn an_unanchored_cycle_keeps_its_plan_and_its_rows() {
    let db = store();
    let body = format!("?a {STEP} ?b . ?b {STEP} ?c . ?c {STEP} ?d . ?d {STEP} ?a .");
    assert!(!splits(&db, &body));
    let query = format!("PREFIX v: <{V}> SELECT ?a ?b ?c ?d WHERE {{ {body} }}");
    let got = db.execute_sparql(&query).unwrap().rows().len();
    let want: usize = (0..PERSONS).map(|a| cycles(a).len()).sum();
    assert_eq!(got, want);
}

#[test]
fn a_constant_anchor_keeps_its_plan_and_its_rows() {
    let db = store();
    let body = format!("<{V}p5> {STEP} ?b . ?b {STEP} ?c . ?c {STEP} ?d . ?d {STEP} <{V}p5> .");
    assert!(!splits(&db, &body));
    assert_eq!(rows(&db, &body), cycles(5));
}

#[test]
fn a_sequence_path_closing_step_keeps_its_plan_and_its_rows() {
    let db = store();
    let body = cycle(&anchored(5), "?d v:KNOWS/^v:KNOWS ?a .");
    assert!(!splits(&db, &body));
    let forward = knows();
    let want = walks(5, |a, _, _, d| {
        (0..PERSONS)
            .filter(|&x| forward.contains(&(d, x)) && forward.contains(&(a, x)))
            .count()
    });
    assert_eq!(rows(&db, &body), want);
}

#[test]
fn an_optional_or_minus_closing_step_keeps_its_plan_and_its_rows() {
    let db = store();
    let closes = |a: usize, d: usize| steps(d).iter().filter(|&&back| back == a).count();
    let optional = cycle(&anchored(5), &format!("OPTIONAL {{ ?d {STEP} ?a }}"));
    assert!(!splits(&db, &optional));
    assert_eq!(
        rows(&db, &optional),
        walks(5, |a, _, _, d| closes(a, d).max(1))
    );
    let minus = cycle(&anchored(5), &format!("MINUS {{ ?d {STEP} ?a }}"));
    assert!(!splits(&db, &minus));
    assert_eq!(
        rows(&db, &minus),
        walks(5, |a, _, _, d| usize::from(closes(a, d) == 0))
    );
}

#[test]
fn a_filter_on_one_side_keeps_the_rows() {
    let db = store();
    let body = cycle(
        &anchored(5),
        &format!("{} FILTER(?bid != \"p3\")", closing()),
    );
    let want: Vec<_> = cycles(5).into_iter().filter(|&(b, _, _)| b != 3).collect();
    assert_eq!(rows(&db, &body), want);
}

#[test]
fn a_five_cycle_keeps_its_plan_and_its_rows() {
    let db = store();
    let body = format!(
        "VALUES ?personId {{ \"p5\" }} ?a v:id ?personId ; {STEP} ?b . ?b {STEP} ?c .
         ?c {STEP} ?d . ?d {STEP} ?e . ?e {STEP} ?a ."
    );
    assert!(!splits(&db, &body));
    let mut want = Vec::new();
    for b in steps(5) {
        for c in steps(b) {
            for d in steps(c) {
                for e in steps(d) {
                    for _ in steps(e).iter().filter(|&&back| back == 5) {
                        want.push((b, c, d));
                    }
                }
            }
        }
    }
    want.sort_unstable();
    assert_eq!(rows(&db, &body), want);
}

/// A pattern that shares no variable with the cycle: the peel never takes a variable on no edge,
/// so what is left beside the cycle keeps the core from being a cycle of four. The rows are the
/// cycle's, once per row of the disconnected pattern.
#[test]
fn a_cycle_beside_a_disconnected_pattern_keeps_its_plan_and_its_rows() {
    let db = store();
    let out = |from: usize| -> Vec<usize> {
        knows()
            .into_iter()
            .filter_map(|(i, j)| (i == from).then_some(j))
            .collect()
    };
    let each = |times: usize| {
        let mut want: Vec<_> = cycles(5)
            .into_iter()
            .flat_map(|row| std::iter::repeat_n(row, times))
            .collect();
        want.sort_unstable();
        want
    };
    // A fresh variable that only a one-variable input names.
    let body = cycle(&anchored(5), &format!("{} ?x a v:Person .", closing()));
    assert!(!splits(&db, &body), "{body}");
    assert_eq!(rows(&db, &body), each(PERSONS));
    // A two-hop chain from another person.
    let body = cycle(
        &anchored(5),
        &format!(
            "{} ?y v:id \"p7\" ; v:KNOWS ?z . ?z v:KNOWS ?w .",
            closing()
        ),
    );
    assert!(!splits(&db, &body), "{body}");
    let two_hops = out(7).into_iter().map(|z| out(z).len()).sum();
    assert_eq!(rows(&db, &body), each(two_hops));
}

/// Two cycles of two, one through `p1` and one anywhere: four variables on two edges each, as
/// many edges as variables, and still no cycle of four, so the split's own check refuses it.
#[test]
fn two_disjoint_cycles_of_two_keep_their_plan_and_their_rows() {
    let db = store();
    let body =
        format!("?a v:id \"p1\" . ?a {STEP} ?b . ?b {STEP} ?a . ?c {STEP} ?d . ?d {STEP} ?c .");
    assert!(!splits(&db, &body), "{body}");
    let back = |from: usize, to: usize| steps(from).iter().filter(|&&s| s == to).count();
    let mut want = Vec::new();
    for b in steps(1) {
        for _ in 0..back(b, 1) {
            for c in 0..PERSONS {
                for d in steps(c) {
                    for _ in 0..back(d, c) {
                        want.push((b, c, d));
                    }
                }
            }
        }
    }
    want.sort_unstable();
    assert_eq!(rows(&db, &body), want);
}

/// A 4-cycle over a predicate of one triple per person fits the shape, and the estimate keeps the
/// chain: with one row per step the halves read more rows than the chain they would replace. The
/// split is taken only when it reads fewer rows.
#[test]
fn a_cycle_of_one_link_per_person_keeps_the_chain() {
    let db = GrafeoDB::new_in_memory();
    let mut ring = String::new();
    for i in 0..4 {
        writeln!(
            ring,
            "<{V}p{i}> <{V}id> \"p{i}\" .\n<{V}p{i}> <{V}KNOWS> <{V}p{}> .",
            (i + 1) % 4
        )
        .unwrap();
    }
    db.load_rdf_text(&ring, RdfFormat::NTriples, None).unwrap();
    let body = format!(
        "{} ?a v:id ?personId ; v:KNOWS ?b . ?b v:KNOWS ?c . ?c v:KNOWS ?d . ?d v:KNOWS ?a .",
        anchored(0)
    );
    assert!(!splits(&db, &body), "{body}");
    assert_eq!(rows(&db, &body), vec![(1, 2, 3)]);
}

#[test]
fn a_chord_keeps_the_plan_and_the_rows() {
    let db = store();
    let times = |from: usize, to: usize| steps(from).iter().filter(|&&s| s == to).count();
    // a-c.
    let body = cycle(&anchored(5), &format!("{} ?a {STEP} ?c .", closing()));
    assert!(!splits(&db, &body));
    let want = walks(5, |a, _, c, d| times(d, a) * times(a, c));
    assert_eq!(rows(&db, &body), want);
    // b and d through a shared liked item.
    let body = cycle(
        &anchored(5),
        &format!("{} ?b v:likes ?x . ?d v:likes ?x .", closing()),
    );
    assert!(!splits(&db, &body));
    let want = walks(5, |a, b, _, d| times(d, a) * usize::from(b % 5 == d % 5));
    assert_eq!(rows(&db, &body), want);
}

#[test]
fn an_edge_written_twice_keeps_the_plan_and_the_rows() {
    let db = store();
    let body = cycle(&anchored(5), &format!("{} ?a {STEP} ?b .", closing()));
    assert!(!splits(&db, &body));
    let times = |from: usize, to: usize| steps(from).iter().filter(|&&s| s == to).count();
    let want = walks(5, |a, b, _, d| times(d, a) * times(a, b));
    assert_eq!(rows(&db, &body), want);
}

#[test]
fn a_variable_predicate_edge_keeps_the_plan_and_the_rows() {
    let db = store();
    let body = cycle(&anchored(5), "?d ?p ?a .");
    assert!(!splits(&db, &body));
    let forward = knows();
    let want = walks(5, |a, _, _, d| usize::from(forward.contains(&(d, a))));
    assert_eq!(rows(&db, &body), want);
}

#[test]
fn both_ends_bound_keeps_the_plan_and_the_rows() {
    let db = store();
    let body = cycle("VALUES (?personId ?cid) { (\"p5\" \"p9\") }", &closing());
    assert!(!splits(&db, &body));
    let want: Vec<_> = cycles(5).into_iter().filter(|&(_, c, _)| c == 9).collect();
    assert_eq!(rows(&db, &body), want);
}

#[test]
fn the_halves_count_against_the_query_memory_limit() {
    let body = cycle(&anchored(0), &closing());
    let db = GrafeoDB::with_config(Config::in_memory().with_query_memory_limit(4 * 1024)).unwrap();
    db.load_rdf_text(&text(), RdfFormat::NTriples, None)
        .unwrap();
    assert!(splits(&db, &body));
    let error = db
        .execute_sparql(&query(&body))
        .expect_err("the halves pass the limit")
        .to_string();
    assert!(error.starts_with("GRAFEO-S001"), "{error}");
}
