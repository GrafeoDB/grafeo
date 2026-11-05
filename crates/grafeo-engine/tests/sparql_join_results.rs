//! SPARQL joins answer what the SPARQL 1.1 algebra says, whichever order they run in.
//!
//! Random triples and random groups of triple patterns and `OPTIONAL`s, among them cycles
//! (which plan as a multi-way join), unconnected parts (a real cross product) and variables
//! one side may leave unbound, against an evaluator written here from 18.2.2 and 18.5 of
//! the recommendation: `BGP`, `Join` and `LeftJoin` over solution mappings, nothing shared
//! with the engine. A failure prints its seed and a shrunken set of triples.
//!
//! `GRAFEO_RULES_SEEDS=2000` sweeps more seeds; `GRAFEO_RULES_SEED=17` runs one.

#![cfg(all(feature = "sparql", feature = "triple-store"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "test graphs of a few thousand nodes"
)]

use std::collections::BTreeMap;
use std::fmt::Write as _;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const FIXED_SEEDS: u64 = 40;
const QUERIES_PER_GRAPH: usize = 12;
const REPORTED: usize = 5;
/// The mappings one query may hold before it is given up.
const BUDGET: usize = 200_000;

struct Random(u64);

impl Random {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5851_F42D_4C95_7F2D)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % bound.max(1) as u64).unwrap()
    }

    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }
}

type Triple = [String; 3];

#[derive(Clone, Debug)]
enum Term {
    Variable(&'static str),
    Constant(String),
}

impl Term {
    fn text(&self) -> String {
        match self {
            Self::Variable(name) => format!("?{name}"),
            Self::Constant(iri) => format!("<{iri}>"),
        }
    }
}

#[derive(Clone, Debug)]
enum Element {
    Pattern([Term; 3]),
    Optional(Vec<Element>),
    /// A group of its own in the group, joined with what comes before.
    Group(Vec<Element>),
}

type Mapping = BTreeMap<&'static str, String>;

/// Over the budget: the query is given up, not answered.
struct TooLarge;

fn compatible(left: &Mapping, right: &Mapping) -> bool {
    left.iter()
        .all(|(name, value)| right.get(name).is_none_or(|other| other == value))
}

fn merged(left: &Mapping, right: &Mapping) -> Mapping {
    let mut out = left.clone();
    out.extend(right.iter().map(|(name, value)| (*name, value.clone())));
    out
}

/// `Join`: the merges of the compatible pairs, as a multiset.
fn join(left: &[Mapping], right: &[Mapping]) -> Result<Vec<Mapping>, TooLarge> {
    let mut out = Vec::new();
    for l in left {
        for r in right {
            if compatible(l, r) {
                out.push(merged(l, r));
                if out.len() > BUDGET {
                    return Err(TooLarge);
                }
            }
        }
    }
    Ok(out)
}

/// `LeftJoin` with the filter `true`: the join, and each left mapping compatible with none.
fn left_join(left: &[Mapping], right: &[Mapping]) -> Result<Vec<Mapping>, TooLarge> {
    let mut out = Vec::new();
    for l in left {
        let mut any = false;
        for r in right {
            if compatible(l, r) {
                out.push(merged(l, r));
                any = true;
            }
        }
        if !any {
            out.push(l.clone());
        }
        if out.len() > BUDGET {
            return Err(TooLarge);
        }
    }
    Ok(out)
}

/// The basic graph pattern: a solution per way to map its variables onto one triple each.
fn basic_graph_pattern(
    patterns: &[&[Term; 3]],
    triples: &[Triple],
) -> Result<Vec<Mapping>, TooLarge> {
    let mut mappings = vec![Mapping::new()];
    for pattern in patterns {
        let mut next = Vec::new();
        for mapping in &mappings {
            for triple in triples {
                let mut extended = mapping.clone();
                let matches = pattern.iter().zip(triple).all(|(term, value)| match term {
                    Term::Constant(iri) => iri == value,
                    Term::Variable(name) => {
                        if let Some(bound) = extended.get(name) {
                            bound == value
                        } else {
                            extended.insert(name, value.clone());
                            true
                        }
                    }
                });
                if matches {
                    next.push(extended);
                    if next.len() > BUDGET {
                        return Err(TooLarge);
                    }
                }
            }
        }
        mappings = next;
    }
    Ok(mappings)
}

/// A group, as 18.2.2 translates it: adjacent triple patterns form one basic graph pattern,
/// joined on to what comes before; an `OPTIONAL` left-joins.
fn group(elements: &[Element], triples: &[Triple]) -> Result<Vec<Mapping>, TooLarge> {
    let mut solutions = vec![Mapping::new()];
    let mut index = 0;
    while index < elements.len() {
        match &elements[index] {
            Element::Optional(inner) => {
                solutions = left_join(&solutions, &group(inner, triples)?)?;
                index += 1;
            }
            Element::Group(inner) => {
                solutions = join(&solutions, &group(inner, triples)?)?;
                index += 1;
            }
            Element::Pattern(_) => {
                let mut patterns = Vec::new();
                while let Some(Element::Pattern(pattern)) = elements.get(index) {
                    patterns.push(pattern);
                    index += 1;
                }
                let found = basic_graph_pattern(&patterns, triples)?;
                solutions = join(&solutions, &found)?;
            }
        }
    }
    Ok(solutions)
}

fn group_text(elements: &[Element]) -> String {
    let mut out = String::new();
    for element in elements {
        match element {
            Element::Pattern([s, p, o]) => {
                let _ = write!(out, "{} {} {} . ", s.text(), p.text(), o.text());
            }
            Element::Optional(inner) => {
                let _ = write!(out, "OPTIONAL {{ {}}} ", group_text(inner));
            }
            Element::Group(inner) => {
                let _ = write!(out, "{{ {}}} ", group_text(inner));
            }
        }
    }
    out
}

fn variables_of(elements: &[Element], into: &mut Vec<&'static str>) {
    for element in elements {
        match element {
            Element::Pattern(terms) => {
                for term in terms {
                    if let Term::Variable(name) = term
                        && !into.contains(name)
                    {
                        into.push(name);
                    }
                }
            }
            Element::Optional(inner) | Element::Group(inner) => variables_of(inner, into),
        }
    }
}

const NAMES: [&str; 7] = ["a", "b", "c", "d", "e", "f", "g"];

struct Builder<'a> {
    random: &'a mut Random,
}

impl Builder<'_> {
    fn predicate(&mut self) -> Term {
        if self.random.chance(6) {
            return Term::Variable("p");
        }
        Term::Constant(format!("urn:p{}", self.random.below(2)))
    }

    fn edge(&mut self, from: &'static str, to: &'static str) -> Element {
        Element::Pattern([Term::Variable(from), self.predicate(), Term::Variable(to)])
    }

    /// A pattern with one end a constant, now and then.
    fn pinned(&mut self, from: &'static str) -> Element {
        let node = format!("urn:n{}", self.random.below(5));
        Element::Pattern([Term::Variable(from), self.predicate(), Term::Constant(node)])
    }

    fn cycle(&mut self, names: &[&'static str]) -> Vec<Element> {
        (0..names.len())
            .map(|i| self.edge(names[i], names[(i + 1) % names.len()]))
            .collect()
    }

    fn query(&mut self) -> Vec<Element> {
        match self.random.below(13) {
            // Groups around a cycle, each with an OPTIONAL: their joins are the inputs of a
            // multi-way join, and a variable may be unbound on either side.
            10 => {
                let first = vec![
                    self.edge("a", "b"),
                    Element::Optional(vec![self.edge("b", "d")]),
                ];
                let second = vec![
                    self.edge("b", "c"),
                    Element::Optional(vec![self.edge("c", "d")]),
                ];
                let third = vec![self.edge("c", "a")];
                vec![
                    Element::Group(first),
                    Element::Group(second),
                    Element::Group(third),
                ]
            }
            // The variable the groups share is only ever optional.
            11 => {
                let first = vec![
                    self.edge("a", "b"),
                    Element::Optional(vec![self.edge("a", "d")]),
                ];
                let second = vec![
                    self.edge("c", "b"),
                    Element::Optional(vec![self.edge("c", "d")]),
                ];
                let third = vec![self.edge("a", "c")];
                vec![
                    Element::Group(first),
                    Element::Group(second),
                    Element::Group(third),
                ]
            }
            // An OPTIONAL with nothing before it, as a group.
            12 => {
                let first = vec![Element::Optional(vec![self.edge("a", "d")])];
                let second = vec![self.edge("a", "b")];
                let third = vec![self.edge("b", "c")];
                let fourth = vec![self.edge("c", "a")];
                vec![
                    Element::Group(first),
                    Element::Group(second),
                    Element::Group(third),
                    Element::Group(fourth),
                ]
            }
            // A triangle, a square.
            0 => self.cycle(&NAMES[..3]),
            1 => self.cycle(&NAMES[..4]),
            // A cycle with a tail and a pin: more inputs than the variables connect at once.
            2 => {
                let mut elements = self.cycle(&NAMES[..3]);
                elements.push(self.edge("c", "d"));
                elements.push(self.edge("d", "e"));
                elements.push(self.pinned("e"));
                elements
            }
            // The shape of the benchmark: two starts, joined late through a cycle.
            3 => vec![
                self.pinned("p1"),
                self.edge("p1", "t"),
                self.pinned("t"),
                self.edge("t", "n"),
                self.pinned("m"),
                self.edge("m", "t"),
                self.edge("m", "p1"),
            ],
            // Two cycles that share nothing: a real cross product.
            4 => {
                let mut elements = self.cycle(&NAMES[..3]);
                elements.extend(self.cycle(&NAMES[3..5]));
                elements
            }
            // A cycle and an unconnected pattern.
            5 => {
                let mut elements = self.cycle(&NAMES[..3]);
                elements.push(self.edge("d", "e"));
                elements
            }
            // A cycle, then an OPTIONAL on one of its variables, then a pattern on what the
            // OPTIONAL may leave unbound.
            6 => {
                let mut elements = self.cycle(&NAMES[..3]);
                elements.push(Element::Optional(vec![self.edge("c", "d")]));
                elements.push(self.edge("d", "e"));
                elements
            }
            // An OPTIONAL first: its variables are loosely shared with the cycle after it.
            7 => {
                let mut elements = vec![Element::Optional(vec![self.edge("a", "d")])];
                elements.extend(self.cycle(&NAMES[..3]));
                elements.push(self.edge("d", "b"));
                elements
            }
            // An OPTIONAL with a cycle inside, and a pattern after.
            8 => {
                let mut elements = vec![self.edge("a", "e")];
                elements.push(Element::Optional(self.cycle(&NAMES[..3])));
                elements.push(self.edge("b", "f"));
                elements
            }
            // Nested OPTIONAL blocks around a cycle.
            _ => {
                let inner = Element::Optional(vec![self.edge("d", "e")]);
                let mut elements = self.cycle(&NAMES[..3]);
                elements.push(Element::Optional(vec![self.edge("c", "d"), inner]));
                elements.push(self.edge("e", "a"));
                elements
            }
        }
    }
}

fn triples_text(triples: &[Triple]) -> String {
    let lines: Vec<String> = triples
        .iter()
        .map(|[s, p, o]| format!("<{s}> <{p}> <{o}> ."))
        .collect();
    lines.join("\n")
}

fn database(triples: &[Triple]) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    if !triples.is_empty() {
        db.execute_sparql(&format!("INSERT DATA {{\n{}\n}}", triples_text(triples)))
            .unwrap();
    }
    db
}

fn answer_of(db: &GrafeoDB, elements: &[Element]) -> Result<Vec<Vec<String>>, String> {
    let mut variables = Vec::new();
    variables_of(elements, &mut variables);
    let names: Vec<String> = variables.iter().map(|name| format!("?{name}")).collect();
    let query = format!(
        "SELECT {} WHERE {{ {}}}",
        names.join(" "),
        group_text(elements)
    );
    let result = db
        .execute_sparql(&query)
        .map_err(|error| error.to_string())?;
    let mut rows: Vec<Vec<String>> = result
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| match value {
                    Value::String(text) => text.to_string(),
                    Value::Null => "-".to_string(),
                    // An IRI prints as `RdfTerm(<urn:n1>)`.
                    other => format!("{other:?}")
                        .trim_start_matches("RdfTerm(<")
                        .trim_end_matches(">)")
                        .to_string(),
                })
                .collect()
        })
        .collect();
    rows.sort();
    Ok(rows)
}

fn expected_of(triples: &[Triple], elements: &[Element]) -> Option<Vec<Vec<String>>> {
    let mut variables = Vec::new();
    variables_of(elements, &mut variables);
    let solutions = group(elements, triples).ok()?;
    let mut rows: Vec<Vec<String>> = solutions
        .iter()
        .map(|mapping| {
            variables
                .iter()
                .map(|name| {
                    mapping
                        .get(name)
                        .cloned()
                        .unwrap_or_else(|| "-".to_string())
                })
                .collect()
        })
        .collect();
    rows.sort();
    Some(rows)
}

fn differs(triples: &[Triple], elements: &[Element]) -> bool {
    expected_of(triples, elements)
        .is_some_and(|expected| answer_of(&database(triples), elements).as_ref() != Ok(&expected))
}

fn report(seed: u64, triples: &[Triple], elements: &[Element]) -> String {
    let mut smallest = triples.to_vec();
    let mut changed = true;
    while changed {
        changed = false;
        for index in (0..smallest.len()).rev() {
            let mut smaller = smallest.clone();
            smaller.remove(index);
            if differs(&smaller, elements) {
                smallest = smaller;
                changed = true;
            }
        }
    }
    format!(
        "seed {seed}\n  query:    {}\n  expected: {:?}\n  got:      {:?}\n  triples:\n{}\n",
        group_text(elements),
        expected_of(&smallest, elements),
        answer_of(&database(&smallest), elements),
        triples_text(&smallest)
    )
}

fn seeds() -> Vec<u64> {
    if let Some(seed) = std::env::var("GRAFEO_RULES_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        return vec![seed];
    }
    let count = std::env::var("GRAFEO_RULES_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(FIXED_SEEDS);
    (0..count).collect()
}

#[test]
fn every_join_answers_what_the_algebra_says() {
    let mut differences = Vec::new();
    let (mut asked, mut with_rows, mut given_up, mut with_unbound) = (0, 0, 0, 0);
    for seed in seeds() {
        let mut random = Random::new(seed);
        let count = 8 + random.below(32);
        let mut triples: Vec<Triple> = Vec::new();
        for _ in 0..count {
            let triple = [
                format!("urn:n{}", random.below(5)),
                format!("urn:p{}", random.below(2)),
                format!("urn:n{}", random.below(5)),
            ];
            if !triples.contains(&triple) {
                triples.push(triple);
            }
        }
        let db = database(&triples);
        for _ in 0..QUERIES_PER_GRAPH {
            let elements = Builder {
                random: &mut random,
            }
            .query();
            let Some(expected) = expected_of(&triples, &elements) else {
                given_up += 1;
                continue;
            };
            asked += 1;
            with_rows += usize::from(!expected.is_empty());
            with_unbound += usize::from(expected.iter().flatten().any(|cell| cell == "-"));
            if answer_of(&db, &elements).as_ref() != Ok(&expected) && differences.len() < REPORTED {
                differences.push(report(seed, &triples, &elements));
            }
        }
    }
    eprintln!(
        "{asked} answers, {with_rows} with rows, {with_unbound} with unbound variables, {given_up} given up"
    );
    assert!(differences.is_empty(), "{}", differences.join("\n"));
    if seeds().len() >= FIXED_SEEDS as usize {
        assert!(
            with_rows * 2 > asked,
            "only {with_rows} of {asked} had rows"
        );
        assert!(
            with_unbound > 10,
            "only {with_unbound} had unbound variables"
        );
    }
}

/// Patterns that share a variable only through later ones are never crossed first: a join
/// of 3 small unconnected sets and their links finishes where their product would not.
#[test]
fn a_join_over_links_does_not_cross_the_small_sides() {
    let db = GrafeoDB::with_config(
        grafeo_engine::Config::in_memory().with_query_timeout(std::time::Duration::from_secs(20)),
    )
    .unwrap();
    let mut triples = String::new();
    for i in 0..700 {
        let _ = writeln!(triples, "<urn:a{i}> a <urn:A> .");
        let _ = writeln!(triples, "<urn:b{i}> a <urn:B> .");
        let _ = writeln!(triples, "<urn:c{i}> a <urn:C> .");
        // Links outnumber each set, so they are estimated last.
        for step in 0..3 {
            let j = (i + step) % 700;
            let _ = writeln!(triples, "<urn:a{i}> <urn:ab> <urn:b{j}> .");
            let _ = writeln!(triples, "<urn:b{i}> <urn:bc> <urn:c{j}> .");
            let _ = writeln!(triples, "<urn:c{i}> <urn:ca> <urn:a{j}> .");
        }
    }
    db.execute_sparql(&format!("INSERT DATA {{\n{triples}}}"))
        .unwrap();
    let result = db
        .execute_sparql(
            "SELECT (COUNT(*) AS ?n) WHERE {
               ?a a <urn:A> . ?b a <urn:B> . ?c a <urn:C> .
               ?a <urn:ab> ?b . ?b <urn:bc> ?c . ?c <urn:ca> ?a .
             }",
        )
        .unwrap();
    // A cycle a(i) -> b(i+s) -> c(i+s+t) -> a(i+s+t+u) with s+t+u % 700 = 0: only all zero.
    let count = format!("{:?}", result.rows()[0][0]);
    assert!(count.contains("\"700\""), "{count}");
}
