//! Path search prefixes and path modes of ISO/IEC 39075:2024 (GQL).
//!
//! A path pattern with a search prefix other than `ALL` is selective (16.6
//! `<path pattern prefix>`): its matches are partitioned by their two
//! endpoints, for each row the pattern starts from, and each partition keeps
//! its own selection. `ANY k` keeps k paths of each partition, `SHORTEST k`
//! the k shortest, `SHORTEST k GROUPS` every path of the k shortest lengths.
//! The prefix goes after the path variable (16.4 `<path pattern>`:
//! `p = ANY SHORTEST TRAIL (...)`), and the path mode of a search prefix
//! restricts the paths the search picks among.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test path_search_prefixes
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Alix, Gus, Vincent, Mia and Jules (`:Person`), with `:KNOWS` edges named
/// by their `w`: Alix->Gus 3, Alix->Vincent 19, Gus->Mia 88, Vincent->Mia
/// 38, Gus->Vincent 33 and Mia->Jules 83.
///
/// The directed paths from Alix: Gus [3]; Vincent [19], [3, 33]; Mia
/// [3, 88], [19, 38], [3, 33, 38]; Jules [3, 88, 83], [19, 38, 83],
/// [3, 33, 38, 83]. Undirected, Jules hangs off Mia alone.
fn diamonds() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {name: 'Alix'}), (gus:Person {name: 'Gus'}), \
         (vincent:Person {name: 'Vincent'}), (mia:Person {name: 'Mia'}), \
         (jules:Person {name: 'Jules'}), \
         (alix)-[:KNOWS {w: 3}]->(gus), (alix)-[:KNOWS {w: 19}]->(vincent), \
         (gus)-[:KNOWS {w: 88}]->(mia), (vincent)-[:KNOWS {w: 38}]->(mia), \
         (gus)-[:KNOWS {w: 33}]->(vincent), (mia)-[:KNOWS {w: 83}]->(jules)",
    )
    .unwrap();
    db
}

/// A value as text: strings bare, lists in brackets.
fn text(value: &Value) -> String {
    match value {
        Value::String(text) => text.to_string(),
        Value::Int64(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        Value::Null => "null".to_string(),
        Value::List(items) => format!(
            "[{}]",
            items.iter().map(text).collect::<Vec<_>>().join(", ")
        ),
        other => panic!("unexpected value {other:?}"),
    }
}

/// The rows of `query` as text, sorted.
fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<String>> {
    let result = db
        .execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    let mut rows: Vec<Vec<String>> = result
        .rows()
        .iter()
        .map(|row| row.iter().map(text).collect())
        .collect();
    rows.sort();
    rows
}

/// Checks the rows of the GQL `query` against `want`, as multisets.
fn assert_rows(db: &GrafeoDB, query: &str, want: &[&[&str]]) {
    let mut want: Vec<Vec<String>> = want
        .iter()
        .map(|row| row.iter().map(|cell| (*cell).to_string()).collect())
        .collect();
    want.sort();
    assert_eq!(rows(db, query), want, "GQL: {query}");
}

/// The error message of `query`, which must fail.
fn error(db: &GrafeoDB, query: &str) -> String {
    match db.execute(query) {
        Ok(result) => panic!("{query} should fail, returned {:?}", result.rows()),
        Err(error) => error.to_string(),
    }
}

// ---------------------------------------------------------------------------
// ANY keeps one path per pair of endpoints, not one row in all (16.6)
// ---------------------------------------------------------------------------

/// `MATCH ANY` returned a single row in total: a global LIMIT 1 over every
/// source and target. Each pair of endpoints is its own partition.
#[test]
fn any_keeps_one_path_for_each_pair_of_endpoints() {
    let db = diamonds();
    assert_rows(
        &db,
        "MATCH ANY (a:Person)-[:KNOWS]->{1,3}(b) RETURN a.name, b.name",
        &[
            &["Alix", "Gus"],
            &["Alix", "Vincent"],
            &["Alix", "Mia"],
            &["Alix", "Jules"],
            &["Gus", "Vincent"],
            &["Gus", "Mia"],
            &["Gus", "Jules"],
            &["Vincent", "Mia"],
            &["Vincent", "Jules"],
            &["Mia", "Jules"],
        ],
    );
}

/// Which path ANY keeps is up to the implementation; it is one path of the
/// pair, from its source to its target.
#[test]
fn any_binds_a_path_of_the_pair() {
    let db = diamonds();
    assert_rows(
        &db,
        "MATCH p = ANY (a:Person {name: 'Alix'})-[:KNOWS]->{1,3}(b) \
         RETURN b.name, [n IN nodes(p) | n.name][0], [n IN nodes(p) | n.name][length(p)]",
        &[
            &["Gus", "Alix", "Gus"],
            &["Vincent", "Alix", "Vincent"],
            &["Mia", "Alix", "Mia"],
            &["Jules", "Alix", "Jules"],
        ],
    );
}

/// `ANY k` keeps k paths of each pair (all of them when it has fewer); the
/// prefix at the MATCH and after the path variable mean the same.
#[test]
fn any_k_keeps_k_paths_for_each_pair() {
    let db = diamonds();
    let want: &[&[&str]] = &[
        &["Gus", "1"],
        &["Vincent", "2"],
        &["Mia", "2"],
        &["Jules", "2"],
    ];
    assert_rows(
        &db,
        "MATCH ANY 2 (a:Person {name: 'Alix'})-[:KNOWS]->{1,3}(b) RETURN b.name, count(*)",
        want,
    );
    assert_rows(
        &db,
        "MATCH p = ANY 2 (a:Person {name: 'Alix'})-[:KNOWS]->{1,3}(b) RETURN b.name, count(*)",
        want,
    );
    assert_rows(
        &db,
        "MATCH p = ANY 2 (a:Person {name: 'Alix'})-[:KNOWS]->{1,3}(b {name: 'Vincent'}) \
         RETURN [n IN nodes(p) | n.name]",
        &[&["[Alix, Vincent]"], &["[Alix, Gus, Vincent]"]],
    );
    // ANY 3 has room for every path of the pairs
    assert_rows(
        &db,
        "MATCH p = ANY 3 (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b {name: 'Mia'}) \
         RETURN [x IN e | x.w]",
        &[&["[3, 88]"], &["[19, 38]"], &["[3, 33, 38]"]],
    );
}

/// The per-pattern `p = ANY (...)` was ignored: every path came back.
#[test]
fn a_per_pattern_any_selects_too() {
    let db = diamonds();
    assert_rows(
        &db,
        "MATCH p = ANY (a:Person {name: 'Alix'})-[:KNOWS]->{1,3}(b) RETURN b.name, count(*)",
        &[
            &["Gus", "1"],
            &["Vincent", "1"],
            &["Mia", "1"],
            &["Jules", "1"],
        ],
    );
}

/// Each row the pattern starts from has its own partitions: two equal input
/// rows keep a path each.
#[test]
fn each_input_row_has_its_own_partitions() {
    let db = diamonds();
    assert_rows(
        &db,
        "UNWIND [3, 3] AS x MATCH ANY (a:Person {name: 'Alix'})-[:KNOWS]->{1,3}(b) \
         RETURN x, b.name",
        &[
            &["3", "Gus"],
            &["3", "Gus"],
            &["3", "Vincent"],
            &["3", "Vincent"],
            &["3", "Mia"],
            &["3", "Mia"],
            &["3", "Jules"],
            &["3", "Jules"],
        ],
    );
    // Gus and Vincent both know Mia: two rows of Mia
    assert_rows(
        &db,
        "MATCH (x:Person)-[:KNOWS]->(m:Person {name: 'Mia'}) WITH m \
         MATCH ANY (m)-[:KNOWS]->{1,2}(b) RETURN m.name, b.name",
        &[&["Mia", "Jules"], &["Mia", "Jules"]],
    );
}

/// The edge pattern's `WHERE` holds for every edge before ANY selects: the
/// first path to Mia starts with the edge of `w` 3, which the condition
/// leaves out, and Mia still has a path.
#[test]
fn an_edge_where_holds_before_any_selects() {
    let db = diamonds();
    assert_rows(
        &db,
        "MATCH p = ANY (a:Person {name: 'Alix'})-[e:KNOWS WHERE e.w > 10]->{1,3}(b) \
         RETURN b.name, [x IN e | x.w]",
        &[
            &["Vincent", "[19]"],
            &["Mia", "[19, 38]"],
            &["Jules", "[19, 38, 83]"],
        ],
    );
    assert_rows(
        &db,
        "MATCH ANY (a:Person {name: 'Alix'})-[e:KNOWS {w: 19}]->{1,3}(b) RETURN b.name",
        &[&["Vincent"]],
    );
}

/// The target's own conditions keep the partitions they hold for.
#[test]
fn a_target_condition_keeps_its_partitions() {
    let db = diamonds();
    assert_rows(
        &db,
        "MATCH ANY (a:Person {name: 'Alix'})-[:KNOWS]->{1,3}(b WHERE b.name <> 'Gus') \
         RETURN b.name",
        &[&["Vincent"], &["Mia"], &["Jules"]],
    );
    // Only Jules is a Musician: one path to him
    db.execute("MATCH (j:Person {name: 'Jules'}) SET j:Musician")
        .unwrap();
    assert_rows(
        &db,
        "MATCH p = ANY (a:Person {name: 'Alix'})-[:KNOWS]->{1,3}(b:Musician) \
         RETURN b.name, length(p)",
        &[&["Jules", "3"]],
    );
}

/// One edge without a quantifier: a pair with two parallel edges has two
/// paths, and ANY keeps one.
#[test]
fn any_over_one_edge_keeps_one_of_parallel_edges() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {name: 'Alix'}), (gus:Person {name: 'Gus'}), \
         (alix)-[:KNOWS {w: 3}]->(gus), (alix)-[:KNOWS {w: 19}]->(gus), \
         (gus)-[:KNOWS {w: 88}]->(alix)",
    )
    .unwrap();
    assert_rows(
        &db,
        "MATCH ANY (a)-[e:KNOWS]->(b) RETURN a.name, b.name",
        &[&["Alix", "Gus"], &["Gus", "Alix"]],
    );
    assert_rows(
        &db,
        "MATCH ANY 2 (a)-[e:KNOWS]->(b) RETURN a.name, b.name, e.w",
        &[
            &["Alix", "Gus", "3"],
            &["Alix", "Gus", "19"],
            &["Gus", "Alix", "88"],
        ],
    );
}

/// OPTIONAL MATCH keeps a source without a path.
#[test]
fn optional_any_keeps_a_source_without_a_path() {
    let db = diamonds();
    assert_rows(
        &db,
        "MATCH (a:Person) OPTIONAL MATCH ANY (a)-[:KNOWS]->{1,2}(b) RETURN a.name, b.name",
        &[
            &["Alix", "Gus"],
            &["Alix", "Vincent"],
            &["Alix", "Mia"],
            &["Gus", "Vincent"],
            &["Gus", "Mia"],
            &["Gus", "Jules"],
            &["Vincent", "Mia"],
            &["Vincent", "Jules"],
            &["Mia", "Jules"],
            &["Jules", "null"],
        ],
    );
}

/// A target bound before: the pair is fixed, ANY keeps its paths.
#[test]
fn any_to_a_bound_target() {
    let db = diamonds();
    assert_rows(
        &db,
        "MATCH (b:Person {name: 'Mia'}) \
         MATCH p = ANY 2 (a:Person {name: 'Alix'})-[:KNOWS]->{1,3}(b) RETURN count(*)",
        &[&["2"]],
    );
    assert_rows(
        &db,
        "MATCH (b:Person {name: 'Mia'}) \
         MATCH ANY (a:Person)-[:KNOWS]->{1,3}(b) RETURN a.name",
        &[&["Alix"], &["Gus"], &["Vincent"]],
    );
    // A cycle back to the source: undirected, Alix has a path to herself
    assert_rows(
        &db,
        "MATCH ANY (a:Person {name: 'Alix'})-[:KNOWS]-{1,3}(a) RETURN a.name",
        &[&["Alix"]],
    );
}

/// A number of paths below one is an error, as is one too large to read.
#[test]
fn any_zero_is_an_error() {
    let db = diamonds();
    let message = error(
        &db,
        "MATCH ANY 0 (a:Person)-[:KNOWS]->{1,3}(b) RETURN b.name",
    );
    assert!(message.contains("at least 1"), "{message}");
    let message = error(
        &db,
        "MATCH ANY 99999999999999999999999 (a:Person)-[:KNOWS]->{1,3}(b) RETURN b.name",
    );
    assert!(message.contains("number of paths"), "{message}");
}

/// A search over two edge patterns would need the node between them in its
/// partitions; it is an error until it works.
#[test]
fn any_over_two_edge_patterns_is_an_error() {
    let db = diamonds();
    let message = error(
        &db,
        "MATCH ANY (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN a.name, c.name",
    );
    assert!(message.contains("more than one edge pattern"), "{message}");
    // A parenthesized or alternated path pattern likewise
    for query in [
        "MATCH ANY ((a:Person)-[:KNOWS]->(b)){1,2} RETURN count(*)",
        "MATCH ANY (a:Person)-[:KNOWS]->(b) | (a:Person)-[:LIKES]->(b) RETURN count(*)",
    ] {
        let message = error(&db, query);
        assert!(message.contains("an ANY path search over a"), "{message}");
    }
}

/// A lone node is the one path of its partition, of length 0: every
/// selection keeps it.
#[test]
fn a_search_over_a_lone_node_keeps_the_node() {
    let db = diamonds();
    for prefix in ["ANY", "ANY SHORTEST", "SHORTEST 3 GROUPS"] {
        assert_rows(
            &db,
            &format!("MATCH {prefix} (a:Person) RETURN count(*)"),
            &[&["5"]],
        );
    }
}

// ---------------------------------------------------------------------------
// The path mode of a search prefix, and of a path pattern (16.6)
// ---------------------------------------------------------------------------

/// Undirected, Jules reaches himself and Mia again only over the edge
/// Jules-Mia twice or back through Mia: WALK reaches all five, TRAIL never
/// repeats the edge (Mia again by the loop Gus-Vincent), ACYCLIC never a
/// node.
#[test]
fn the_path_mode_of_any_restricts_its_paths() {
    let db = diamonds();
    let base = "(a:Person {name: 'Jules'})-[:KNOWS]-{2,4}(b) RETURN b.name";
    assert_rows(
        &db,
        &format!("MATCH ANY {base}"),
        &[&["Alix"], &["Gus"], &["Jules"], &["Mia"], &["Vincent"]],
    );
    assert_rows(
        &db,
        &format!("MATCH ANY TRAIL {base}"),
        &[&["Alix"], &["Gus"], &["Mia"], &["Vincent"]],
    );
    assert_rows(
        &db,
        &format!("MATCH p = ANY TRAIL {base}"),
        &[&["Alix"], &["Gus"], &["Mia"], &["Vincent"]],
    );
    assert_rows(
        &db,
        &format!("MATCH p = ANY ACYCLIC PATHS {base}"),
        &[&["Alix"], &["Gus"], &["Vincent"]],
    );
    // Alix has four simple paths from Jules, Gus and Vincent three each, and
    // a simple path may end where it starts: Jules, Mia, Jules
    assert_rows(
        &db,
        &format!("MATCH ANY 3 SIMPLE {base}"),
        &[
            &["Jules"],
            &["Alix"],
            &["Alix"],
            &["Alix"],
            &["Gus"],
            &["Gus"],
            &["Gus"],
            &["Vincent"],
            &["Vincent"],
            &["Vincent"],
        ],
    );
}

/// `p = TRAIL (...)` puts the path mode where ISO puts it, after the path
/// variable; it was a syntax error. From Jules to Mia undirected within four
/// edges: six walks, three trails, one acyclic path.
#[test]
fn a_path_mode_after_the_path_variable() {
    let db = diamonds();
    let pattern = "(a:Person {name: 'Jules'})-[:KNOWS]-{1,4}(b:Person {name: 'Mia'}) \
                   RETURN length(p)";
    assert_rows(
        &db,
        &format!("MATCH p = WALK {pattern}"),
        &[&["1"], &["3"], &["3"], &["3"], &["4"], &["4"]],
    );
    assert_rows(
        &db,
        &format!("MATCH p = TRAIL {pattern}"),
        &[&["1"], &["4"], &["4"]],
    );
    assert_rows(
        &db,
        &format!("MATCH p = TRAIL PATH {pattern}"),
        &[&["1"], &["4"], &["4"]],
    );
    assert_rows(
        &db,
        &format!("MATCH p = ALL TRAIL PATHS {pattern}"),
        &[&["1"], &["4"], &["4"]],
    );
    assert_rows(&db, &format!("MATCH p = ACYCLIC {pattern}"), &[&["1"]]);
    assert_rows(&db, &format!("MATCH p = SIMPLE {pattern}"), &[&["1"]]);
    // The mode of one path pattern holds for that pattern only: three trails
    // to Mia, then seven walks of two edges from her (four of them trails)
    assert_rows(
        &db,
        "MATCH p = TRAIL (a:Person {name: 'Jules'})-[:KNOWS]-{1,4}(b:Person {name: 'Mia'}), \
         q = (b)-[:KNOWS]-{2,2}(c) RETURN count(*)",
        &[&["21"]],
    );
}

// ---------------------------------------------------------------------------
// SHORTEST k and SHORTEST k GROUPS (16.6)
// ---------------------------------------------------------------------------

/// `SHORTEST k` returned one path; it keeps the k shortest of each pair.
#[test]
fn shortest_k_keeps_the_k_shortest_paths_of_each_pair() {
    let db = diamonds();
    assert_rows(
        &db,
        "MATCH p = SHORTEST 2 (a:Person {name: 'Alix'})-[:KNOWS]->{1,4}(b) \
         RETURN b.name, length(p)",
        &[
            &["Gus", "1"],
            &["Vincent", "1"],
            &["Vincent", "2"],
            &["Mia", "2"],
            &["Mia", "2"],
            &["Jules", "3"],
            &["Jules", "3"],
        ],
    );
    assert_rows(
        &db,
        "MATCH p = SHORTEST 2 (a:Person {name: 'Alix'})-[e:KNOWS]->{1,4}(b:Person {name: 'Mia'}) \
         RETURN [x IN e | x.w]",
        &[&["[3, 88]"], &["[19, 38]"]],
    );
    assert_rows(
        &db,
        "MATCH SHORTEST 3 p = (a:Person {name: 'Alix'})-[e:KNOWS]->{1,4}(b:Person {name: 'Jules'}) \
         RETURN [x IN e | x.w]",
        &[&["[3, 88, 83]"], &["[19, 38, 83]"], &["[3, 33, 38, 83]"]],
    );
    // More than there are: every path of the pair
    assert_rows(
        &db,
        "MATCH p = SHORTEST 19 PATHS (a:Person {name: 'Alix'})-[:KNOWS]->{1,4}(b:Person {name: 'Jules'}) \
         RETURN length(p)",
        &[&["3"], &["3"], &["4"]],
    );
}

/// `SHORTEST k GROUPS` returned one path; it keeps every path of the k
/// shortest lengths of each pair.
#[test]
fn shortest_k_groups_keeps_the_paths_of_the_k_shortest_lengths() {
    let db = diamonds();
    assert_rows(
        &db,
        "MATCH p = SHORTEST 2 GROUPS (a:Person {name: 'Alix'})-[:KNOWS]->{1,4}(b) \
         RETURN b.name, length(p)",
        &[
            &["Gus", "1"],
            &["Vincent", "1"],
            &["Vincent", "2"],
            &["Mia", "2"],
            &["Mia", "2"],
            &["Mia", "3"],
            &["Jules", "3"],
            &["Jules", "3"],
            &["Jules", "4"],
        ],
    );
    // SHORTEST GROUPS is one group: the shortest paths, as ALL SHORTEST
    assert_rows(
        &db,
        "MATCH p = SHORTEST GROUPS (a:Person {name: 'Alix'})-[e:KNOWS]->{1,4}(b:Person {name: 'Jules'}) \
         RETURN [x IN e | x.w]",
        &[&["[3, 88, 83]"], &["[19, 38, 83]"]],
    );
    assert_rows(
        &db,
        "MATCH SHORTEST 1 PATH GROUP p = (a:Person {name: 'Alix'})-[:KNOWS]->{1,4}(b:Person {name: 'Vincent'}) \
         RETURN length(p)",
        &[&["1"]],
    );
}

/// Undirected from Jules to Mia in two or more edges: the shortest walk goes
/// back over the edge Jules-Mia (3 edges), the shortest trails around the
/// loop Gus-Vincent (4 edges, two of them; two more of 5 edges pass Alix),
/// and no acyclic path has two.
#[test]
fn the_path_mode_of_a_shortest_search_restricts_its_paths() {
    let db = diamonds();
    let pattern = "(a:Person {name: 'Jules'})-[e:KNOWS]-{2,}(b:Person {name: 'Mia'})";
    assert_rows(
        &db,
        &format!("MATCH p = ANY SHORTEST {pattern} RETURN length(p)"),
        &[&["3"]],
    );
    assert_rows(
        &db,
        &format!("MATCH p = ANY SHORTEST TRAIL {pattern} RETURN length(p)"),
        &[&["4"]],
    );
    assert_rows(
        &db,
        &format!("MATCH ANY SHORTEST TRAIL p = {pattern} RETURN length(p)"),
        &[&["4"]],
    );
    assert_rows(
        &db,
        &format!("MATCH TRAIL p = ANY SHORTEST {pattern} RETURN length(p)"),
        &[&["4"]],
    );
    assert_rows(
        &db,
        &format!("MATCH p = ALL SHORTEST TRAIL {pattern} RETURN [x IN e | x.w]"),
        &[&["[83, 38, 33, 88]"], &["[83, 88, 33, 38]"]],
    );
    assert_rows(
        &db,
        &format!("MATCH p = SHORTEST 3 TRAIL {pattern} RETURN length(p)"),
        &[&["4"], &["4"], &["5"]],
    );
    assert_rows(
        &db,
        &format!("MATCH p = SHORTEST 2 TRAIL GROUPS {pattern} RETURN length(p)"),
        &[&["4"], &["4"], &["5"], &["5"]],
    );
    assert_rows(
        &db,
        &format!("MATCH p = ANY SHORTEST ACYCLIC {pattern} RETURN length(p)"),
        &[],
    );
    assert_rows(
        &db,
        &format!("MATCH p = ALL SHORTEST SIMPLE {pattern} RETURN length(p)"),
        &[],
    );
}

/// Two path modes for one path pattern contradict each other.
#[test]
fn two_path_modes_for_one_pattern_are_an_error() {
    let db = diamonds();
    let message = error(
        &db,
        "MATCH TRAIL ANY SHORTEST ACYCLIC (a:Person)-[:KNOWS]->+(b) RETURN b.name",
    );
    assert!(message.contains("path mode"), "{message}");
}
