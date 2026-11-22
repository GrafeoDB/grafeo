//! Work per query (#794): the engine's own count of the work each graph-bench query does, per
//! class (`grafeo_common::work`), checked against a baseline with no margin.
//!
//! Timings vary 5 to 15% between runs, so a 2% effect cannot be seen in them; these counts are
//! exact and the same in every run. A performance change states its effect in them, and this
//! bench gates CI on them.
//!
//! The data has graph-bench's shapes at a small scale, from a seeded generator, and is loaded
//! through the CSV bulk loader as graph-bench's adapter loads it (a named graph per dataset, an
//! index on `id`):
//!
//! - `snb`: `Person` (2,001, the last one without edges, string ids), `KNOWS` (36,000, 40% of
//!   them on the first 1% of the persons, so a few persons are hubs), `Tag`, `Post`, `Comment`
//!   (200,000: two full row groups and two partial ones), `HAS_INTEREST`, `HAS_TAG`,
//!   `HAS_CREATOR` and `REPLY_OF`, with graph-bench's properties and kinds;
//! - `graphalytics`: `V` (10,000, integer ids, every node of the graph) and `E` (each of 70,000
//!   pairs both ways, with hubs, no self-loop: two full row groups and a partial one);
//! - `multigraph`: graph-bench's fixture, `V` and `E` with self-loops and parallel edges.
//!
//! Every query runs in two states: right after the load (flat vectors), and after a seeded batch
//! of committed updates and deletes while an older transaction stays open, so the reads go
//! through stamped vectors.
//!
//! The bench sets the memory limit and the query timeout (the default limit follows the machine's
//! memory, and without a timeout no deadline is checked). Each statement runs on the calling
//! thread, so its whole tally is on one thread; no query here reaches the rayon pool.
//!
//! Only a build with `--cfg grafeo_work_counters` counts; it has its own target directory, so the
//! changed `RUSTFLAGS` do not rebuild the usual one:
//!
//! ```bash
//! RUSTFLAGS="--cfg grafeo_work_counters" CARGO_TARGET_DIR=C:/cargo-target/grafeo-work \
//!   cargo bench -p grafeo-engine --features cypher,sql-pgq,algos --bench work_per_query -- --check
//! ```
//!
//! `--check` fails when a count differs from `benches/work_per_query.baseline.json`, up or down,
//! or when a query or a class is in one and not the other. A change that moves work updates the
//! baseline in the same commit (`-- --update-baseline`), so a drop shows in review and a later
//! rise back cannot pass. The baseline records the target and the toolchain it was made with: on
//! another target the differences are reported without failing (the allocation classes follow the
//! platform's std), and a toolchain bump updates the baseline in the same commit. `--out <path>`
//! also writes the counts as JSON.

#![cfg_attr(
    grafeo_work_counters,
    expect(
        unsafe_code,
        reason = "GlobalAlloc is an unsafe trait; this one forwards every call to System"
    )
)]

#[cfg(grafeo_work_counters)]
mod support;

/// Without the cfg nothing is counted: say how to build the bench, and fail a check.
#[cfg(not(grafeo_work_counters))]
fn main() {
    eprintln!(
        "work_per_query counts only in a build with `--cfg grafeo_work_counters`:\n  \
         RUSTFLAGS=\"--cfg grafeo_work_counters\" CARGO_TARGET_DIR=<a target dir of its own> \
         cargo bench -p grafeo-engine --features cypher,sql-pgq,algos --bench work_per_query -- --check"
    );
    if std::env::args().any(|arg| arg == "--check") {
        std::process::exit(2);
    }
}

#[cfg(grafeo_work_counters)]
fn main() {
    counted::main();
}

#[cfg(grafeo_work_counters)]
mod counted {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::collections::{BTreeMap, BTreeSet, HashMap};
    use std::fmt::Write as _;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use grafeo_common::types::Value;
    use grafeo_common::work::{self, Work};
    use grafeo_engine::loader::{Column, ColumnKind, EdgeLoad, EndKey, NodeLoad};
    use grafeo_engine::{Config, GrafeoDB};

    use super::support::Xorshift;

    /// The seed of the data and of the second state's changes.
    const SEED: u64 = 0x5EED_0794_0001;
    /// The persons with edges; one more has none.
    const PERSONS: usize = 2_000;
    /// The `KNOWS` edges (graph-bench's 18 per person).
    const KNOWS: usize = 36_000;
    /// The comments: from node id 4,021 on, two full row groups of 65,536 and two partial ones.
    const COMMENTS: usize = 200_000;
    /// The `V` nodes of `graphalytics`.
    const VERTICES: usize = 10_000;
    /// The `V` pairs, each an `E` both ways: 140,000 edges.
    const PAIRS: usize = 70_000;
    /// Set, so the grant step does not follow the machine's memory.
    const MEMORY_LIMIT: usize = 4 << 30;
    /// Set, so the deadline is checked.
    const QUERY_TIMEOUT: Duration = Duration::from_secs(600);

    const DAY_MS: i64 = 86_400_000;
    /// 1940-01-01, 2010-01-01 and 2010-06-01 in milliseconds since the epoch.
    const JAN_1940_MS: i64 = -946_771_200_000;
    const JAN_2010_MS: i64 = 1_262_304_000_000;
    const JUN_2010_MS: i64 = 1_275_350_400_000;
    /// The window of `scan_filter`: 100 of the comments' 4,001 days.
    const FROM_MS: i64 = JUN_2010_MS + 1_000 * DAY_MS;
    const TO_MS: i64 = FROM_MS + 100 * DAY_MS;

    // ── The counting allocator ──────────────────────────────────────

    /// Forwards to [`System`] and counts each allocation and its bytes on the thread's tally.
    struct Counting;

    fn counted(pointer: *mut u8, bytes: usize) {
        if !pointer.is_null() {
            work::count(Work::Allocation, 1);
            work::count(
                Work::AllocatedBytes,
                u64::try_from(bytes).unwrap_or(u64::MAX),
            );
        }
    }

    // SAFETY: every method forwards to `System` with the caller's arguments, so `System`'s
    // guarantees are this allocator's; counting allocates nothing.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: the caller's contract for `alloc`, passed on.
            let pointer = unsafe { System.alloc(layout) };
            counted(pointer, layout.size());
            pointer
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            // SAFETY: the caller's contract for `alloc_zeroed`, passed on.
            let pointer = unsafe { System.alloc_zeroed(layout) };
            counted(pointer, layout.size());
            pointer
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            // SAFETY: the caller's contract for `dealloc`, passed on.
            unsafe { System.dealloc(pointer, layout) };
        }

        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            // SAFETY: the caller's contract for `realloc`, passed on.
            let moved = unsafe { System.realloc(pointer, layout, new_size) };
            counted(moved, new_size);
            moved
        }
    }

    #[global_allocator]
    static GLOBAL: Counting = Counting;

    // ── The data ────────────────────────────────────────────────────

    const FIRST_NAMES: [&str; 8] = [
        "Vincent", "Jules", "Mia", "Butch", "Django", "Shosanna", "Hans", "Beatrix",
    ];
    const LAST_NAMES: [&str; 6] = [
        "Vega",
        "Winnfield",
        "Wallace",
        "Coolidge",
        "Dreyfus",
        "Kiddo",
    ];
    const BROWSERS: [&str; 6] = [
        "Firefox",
        "Chrome",
        "Safari",
        "Edge",
        "Opera",
        "Internet Explorer",
    ];
    const TAGS: [&str; 20] = [
        "Photography",
        "Music",
        "Travel",
        "Sports",
        "Technology",
        "Art",
        "Science",
        "Movies",
        "Books",
        "Gaming",
        "Cooking",
        "Fashion",
        "Fitness",
        "Nature",
        "Politics",
        "Business",
        "Education",
        "Health",
        "History",
        "Philosophy",
    ];
    const POST_CONTENT: [&str; 4] = [
        "Just visited an amazing new restaurant in Amsterdam!",
        "Excited to start my new project at work today.",
        "Thinking about switching careers, any advice?",
        "Just finished reading an incredible book.",
    ];
    /// Graph-bench's comment texts: 23 to 37 characters, two with a comma.
    const COMMENT_CONTENT: [&str; 8] = [
        "Great post! Totally agree.",
        "Thanks for sharing this!",
        "I had a similar experience.",
        "Interesting perspective.",
        "That's awesome, congrats!",
        "I disagree, but respect your opinion.",
        "Love this! More please.",
        "Where was this? Looks amazing!",
    ];

    /// The nodes of one label, as CSV rows, `id` first.
    struct Nodes {
        label: &'static str,
        columns: &'static [(&'static str, ColumnKind)],
        rows: Vec<String>,
    }

    /// The edges of one type between two labels, as CSV rows of `src`, `dst` and the properties.
    struct Edges {
        edge_type: &'static str,
        source: &'static str,
        target: &'static str,
        key: ColumnKind,
        columns: &'static [(&'static str, ColumnKind)],
        rows: Vec<String>,
    }

    /// A named graph's data.
    struct Graph {
        name: &'static str,
        nodes: Vec<Nodes>,
        edges: Vec<Edges>,
    }

    /// A field as CSV writes it: quoted when it holds a comma, a quote or a line break.
    fn field(text: &str) -> String {
        if text.contains([',', '"', '\n']) {
            format!("\"{}\"", text.replace('"', "\"\""))
        } else {
            text.to_string()
        }
    }

    /// A day from `start`, at most `days` days later, in milliseconds.
    fn date(random: &mut Xorshift, start: i64, days: usize) -> i64 {
        start + DAY_MS * i64::try_from(random.below(days + 1)).expect("a few thousand days")
    }

    /// `count` of the numbers below `bound`, each once, in the order drawn.
    fn distinct(random: &mut Xorshift, count: usize, bound: usize) -> Vec<usize> {
        let mut numbers: Vec<usize> = (0..bound).collect();
        for at in 0..count {
            let other = at + random.below(bound - at);
            numbers.swap(at, other);
        }
        numbers.truncate(count);
        numbers
    }

    /// `pairs` edges among `nodes` nodes, no self-loop and no pair twice in either direction:
    /// 40% of them have one end among the first 1% of the nodes (graph-bench's `KNOWS`).
    fn hub_pairs(random: &mut Xorshift, nodes: usize, pairs: usize) -> Vec<(usize, usize)> {
        let hubs = (nodes / 100).max(10);
        let mut seen = BTreeSet::new();
        let mut found = Vec::with_capacity(pairs);
        let mut attempts = 0;
        while found.len() < pairs && attempts < pairs * 3 {
            attempts += 1;
            let (source, target) = if random.below(10) < 4 {
                if random.below(2) == 0 {
                    (random.below(hubs), random.below(nodes))
                } else {
                    (random.below(nodes), random.below(hubs))
                }
            } else {
                (random.below(nodes), random.below(nodes))
            };
            if source != target && seen.insert((source.min(target), source.max(target))) {
                found.push((source, target));
            }
        }
        assert_eq!(found.len(), pairs, "the generator finds every pair");
        found
    }

    fn snb(random: &mut Xorshift) -> Graph {
        use ColumnKind::{Int64, String as Text};
        let persons = (0..=PERSONS)
            .map(|i| {
                let gender = if random.below(2) == 0 {
                    "male"
                } else {
                    "female"
                };
                let first = FIRST_NAMES[random.below(FIRST_NAMES.len())];
                let last = LAST_NAMES[random.below(LAST_NAMES.len())];
                let birthday = date(random, JAN_1940_MS, 30_000);
                let created = date(random, JAN_2010_MS, 4_000);
                let ip = format!(
                    "{}.{}.{}.{}",
                    1 + random.below(255),
                    random.below(256),
                    random.below(256),
                    random.below(256)
                );
                let browser = field(BROWSERS[random.below(BROWSERS.len())]);
                format!("person_{i},{first},{last},{gender},{birthday},{created},{ip},{browser}")
            })
            .collect();
        let tags = TAGS
            .iter()
            .enumerate()
            .map(|(i, name)| format!("tag_{i},{name}"))
            .collect();
        let posts = (0..PERSONS)
            .map(|i| {
                let created = date(random, JAN_2010_MS, 4_000);
                // An empty field sets no property: an image post has no content.
                let (content, image) = if random.below(10) < 3 {
                    (
                        String::new(),
                        format!("photo{}.jpg", 1 + random.below(1_000)),
                    )
                } else {
                    let text = POST_CONTENT[random.below(POST_CONTENT.len())];
                    (field(text), String::new())
                };
                let length = if content.is_empty() { 0 } else { content.len() };
                format!("post_{i},{content},{image},{created},{length}")
            })
            .collect();
        let comments = (0..COMMENTS)
            .map(|i| {
                let text = COMMENT_CONTENT[random.below(COMMENT_CONTENT.len())];
                let created = date(random, JUN_2010_MS, 4_000);
                format!("comment_{i},{},{created},{}", field(text), text.len())
            })
            .collect();

        let knows = hub_pairs(random, PERSONS, KNOWS)
            .into_iter()
            .map(|(a, b)| {
                let created = date(random, JAN_2010_MS, 4_000);
                format!("person_{a},person_{b},{created}")
            })
            .collect();
        let mut interests = Vec::new();
        for i in 0..PERSONS {
            let count = 1 + random.below(5);
            for tag in distinct(random, count, TAGS.len()) {
                interests.push(format!("person_{i},tag_{tag}"));
            }
        }
        let (mut post_creators, mut post_tags) = (Vec::new(), Vec::new());
        for i in 0..PERSONS {
            post_creators.push(format!("post_{i},person_{}", random.below(PERSONS)));
            let count = 1 + random.below(3);
            for tag in distinct(random, count, TAGS.len()) {
                post_tags.push(format!("post_{i},tag_{tag}"));
            }
        }
        let (mut comment_creators, mut replies_to_posts, mut replies_to_comments) =
            (Vec::new(), Vec::new(), Vec::new());
        for i in 0..COMMENTS {
            comment_creators.push(format!("comment_{i},person_{}", random.below(PERSONS)));
            if i > 0 && random.below(10) < 3 {
                replies_to_comments.push(format!("comment_{i},comment_{}", random.below(i)));
            } else {
                replies_to_posts.push(format!("comment_{i},post_{}", random.below(PERSONS)));
            }
        }

        let edges = |edge_type,
                     source,
                     target,
                     columns: &'static [(&'static str, ColumnKind)],
                     rows: Vec<String>| Edges {
            edge_type,
            source,
            target,
            key: Text,
            columns,
            rows,
        };
        Graph {
            name: "snb",
            nodes: vec![
                Nodes {
                    label: "Person",
                    columns: &[
                        ("id", Text),
                        ("firstName", Text),
                        ("lastName", Text),
                        ("gender", Text),
                        ("birthday", Int64),
                        ("creationDate", Int64),
                        ("locationIP", Text),
                        ("browserUsed", Text),
                    ],
                    rows: persons,
                },
                Nodes {
                    label: "Tag",
                    columns: &[("id", Text), ("name", Text)],
                    rows: tags,
                },
                Nodes {
                    label: "Post",
                    columns: &[
                        ("id", Text),
                        ("content", Text),
                        ("imageFile", Text),
                        ("creationDate", Int64),
                        ("length", Int64),
                    ],
                    rows: posts,
                },
                Nodes {
                    label: "Comment",
                    columns: &[
                        ("id", Text),
                        ("content", Text),
                        ("creationDate", Int64),
                        ("length", Int64),
                    ],
                    rows: comments,
                },
            ],
            edges: vec![
                edges(
                    "KNOWS",
                    "Person",
                    "Person",
                    &[("creationDate", Int64)],
                    knows,
                ),
                edges("HAS_INTEREST", "Person", "Tag", &[], interests),
                edges("HAS_CREATOR", "Post", "Person", &[], post_creators),
                edges("HAS_TAG", "Post", "Tag", &[], post_tags),
                edges("HAS_CREATOR", "Comment", "Person", &[], comment_creators),
                edges("REPLY_OF", "Comment", "Post", &[], replies_to_posts),
                edges("REPLY_OF", "Comment", "Comment", &[], replies_to_comments),
            ],
        }
    }

    /// `V` and `E` from `rows` of pairs, as graph-bench's graphalytics and multigraph are.
    fn vertices_and_edges(name: &'static str, vertices: Vec<String>, edges: Vec<String>) -> Graph {
        Graph {
            name,
            nodes: vec![Nodes {
                label: "V",
                columns: &[("id", ColumnKind::Int64)],
                rows: vertices,
            }],
            edges: vec![Edges {
                edge_type: "E",
                source: "V",
                target: "V",
                key: ColumnKind::Int64,
                columns: &[],
                rows: edges,
            }],
        }
    }

    fn graphalytics(random: &mut Xorshift) -> Graph {
        let mut both_ways: Vec<(usize, usize)> = hub_pairs(random, VERTICES, PAIRS)
            .into_iter()
            .flat_map(|(a, b)| [(a, b), (b, a)])
            .collect();
        both_ways.sort_unstable();
        vertices_and_edges(
            "graphalytics",
            (0..VERTICES).map(|i| i.to_string()).collect(),
            both_ways.iter().map(|(a, b)| format!("{a},{b}")).collect(),
        )
    }

    /// Graph-bench's fixture: parallel edges, self-loops (one twice), an antiparallel pair and a
    /// vertex without edges.
    fn multigraph() -> Graph {
        let edges = [
            (1, 2),
            (1, 2),
            (2, 1),
            (2, 2),
            (2, 3),
            (3, 3),
            (3, 3),
            (3, 4),
            (4, 5),
            (5, 4),
            (5, 5),
        ];
        vertices_and_edges(
            "multigraph",
            (1..=6).map(|i: u32| i.to_string()).collect(),
            edges.iter().map(|(a, b)| format!("{a},{b}")).collect(),
        )
    }

    fn columns(columns: &[(&str, ColumnKind)]) -> Vec<Column> {
        columns
            .iter()
            .map(|&(name, kind)| Column {
                name: name.to_string(),
                kind,
            })
            .collect()
    }

    fn write_csv(path: &Path, columns: &[(&str, ColumnKind)], rows: &[String]) {
        let mut text = columns
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>()
            .join(",");
        text.push('\n');
        for row in rows {
            text.push_str(row);
            text.push('\n');
        }
        std::fs::write(path, text).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    }

    /// Loads `graph` into a named graph of its own, as graph-bench's adapter does: each label
    /// keyed by `id`, each edge's ends found by it, then an index on `id`.
    fn load(db: &GrafeoDB, dir: &Path, graph: &Graph) {
        db.create_graph(graph.name).expect("a new graph");
        db.set_current_graph(Some(graph.name)).expect("the graph");
        for nodes in &graph.nodes {
            let path = dir.join(format!("{}.{}.csv", graph.name, nodes.label));
            write_csv(&path, nodes.columns, &nodes.rows);
            let spec = NodeLoad {
                labels: vec![nodes.label.to_string()],
                key: Some("id".to_string()),
                columns: columns(nodes.columns),
                ..NodeLoad::default()
            };
            db.load_nodes_csv(&path, &spec)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        }
        for edges in &graph.edges {
            let path = dir.join(format!(
                "{}.{}.{}.{}.csv",
                graph.name, edges.edge_type, edges.source, edges.target
            ));
            let mut all = vec![("src", edges.key), ("dst", edges.key)];
            all.extend_from_slice(edges.columns);
            write_csv(&path, &all, &edges.rows);
            let end = |label: &str, column: &str| EndKey {
                label: label.to_string(),
                property: "id".to_string(),
                column: column.to_string(),
            };
            let spec = EdgeLoad {
                edge_type: edges.edge_type.to_string(),
                source: end(edges.source, "src"),
                target: end(edges.target, "dst"),
                columns: columns(&all),
                ..EdgeLoad::default()
            };
            db.load_edges_csv(&path, &spec)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        }
        db.create_property_index("id").expect("the id index");
    }

    // ── The queries ─────────────────────────────────────────────────

    #[derive(Clone, Copy)]
    enum Language {
        /// `execute_cypher`, as graph-bench runs its property-graph rows.
        Cypher,
        /// `execute_sql`, as graph-bench runs its SQL/PGQ rows.
        Sql,
        /// `execute`, for the algorithm's `CALL`.
        Gql,
    }

    #[derive(Clone, Copy)]
    enum Parameter {
        Text(&'static str),
        Int(i64),
    }

    /// A graph-bench row: its id, its dataset, and its text character for character, with its
    /// parameters fixed (`person_0` and `person_1` are hubs, `person_500` is not).
    struct Query {
        id: &'static str,
        graph: &'static str,
        language: Language,
        text: &'static str,
        parameters: &'static [(&'static str, Parameter)],
    }

    const HUB: &[(&str, Parameter)] = &[("personId", Parameter::Text("person_0"))];
    const PERSON: &[(&str, Parameter)] = &[("personId", Parameter::Text("person_500"))];
    const PAIR: &[(&str, Parameter)] = &[
        ("person1Id", Parameter::Text("person_1")),
        ("person2Id", Parameter::Text("person_2")),
    ];
    const WINDOW: &[(&str, Parameter)] = &[
        ("from", Parameter::Int(FROM_MS)),
        ("to", Parameter::Int(TO_MS)),
    ];
    const MIN_LENGTH: &[(&str, Parameter)] = &[("minLength", Parameter::Int(30))];
    const MIN: &[(&str, Parameter)] = &[("min", Parameter::Int(25))];
    const CUT: &[(&str, Parameter)] = &[("cut", Parameter::Int(1_000))];

    const fn query(
        id: &'static str,
        graph: &'static str,
        language: Language,
        text: &'static str,
        parameters: &'static [(&'static str, Parameter)],
    ) -> Query {
        Query {
            id,
            graph,
            language,
            text,
            parameters,
        }
    }

    use Language::{Cypher, Gql, Sql};

    /// The rows of the lane's items; a row is added as its item comes up.
    const QUERIES: &[Query] = &[
        query(
            "scan_label",
            "snb",
            Cypher,
            "MATCH (m:Comment) RETURN m.id AS id",
            &[],
        ),
        query(
            "scan_filter",
            "snb",
            Cypher,
            "MATCH (m:Comment)\nWHERE m.creationDate >= $from AND m.creationDate < $to\nRETURN m.id AS id",
            WINDOW,
        ),
        query(
            "scan_aggregate",
            "snb",
            Cypher,
            "MATCH (p:Person)\nRETURN p.browserUsed AS browser, p.gender AS gender, count(*) AS persons, \
             min(p.birthday) AS firstBirthday,\n       max(p.creationDate) AS lastCreated\nORDER BY browser, gender",
            &[],
        ),
        query(
            "scan_order_limit",
            "snb",
            Cypher,
            "MATCH (m:Comment)\nWHERE m.length >= $minLength\nRETURN m.id AS id, m.content AS content, \
             m.creationDate AS created\nORDER BY created DESC, id ASC\nLIMIT 20",
            MIN_LENGTH,
        ),
        query(
            "scan_filter_avg",
            "snb",
            Cypher,
            "MATCH (c:Comment)\nWHERE c.length > $min\nRETURN count(*) AS n, avg(c.length) AS mean",
            MIN,
        ),
        query(
            "scan_label_sql",
            "snb",
            Sql,
            "SELECT id\nFROM GRAPH_TABLE (snb\n  MATCH (m:Comment)\n  COLUMNS (m.id AS id)\n)",
            &[],
        ),
        query(
            "scan_filter_sql",
            "snb",
            Sql,
            "SELECT id\nFROM GRAPH_TABLE (snb\n  MATCH (m:Comment)\n  WHERE m.creationDate >= $from AND \
             m.creationDate < $to\n  COLUMNS (m.id AS id)\n)",
            WINDOW,
        ),
        query(
            "scan_aggregate_sql",
            "snb",
            Sql,
            "SELECT browser, gender, count(*) AS persons, min(birthday) AS firstBirthday, \
             max(creationDate) AS lastCreated\nFROM GRAPH_TABLE (snb\n  MATCH (p:Person)\n  COLUMNS \
             (p.browserUsed AS browser, p.gender AS gender, p.birthday AS birthday, p.creationDate \
             AS creationDate)\n)\nGROUP BY browser, gender\nORDER BY browser, gender",
            &[],
        ),
        query(
            "scan_order_limit_sql",
            "snb",
            Sql,
            "SELECT id, content, created\nFROM GRAPH_TABLE (snb\n  MATCH (m:Comment)\n  WHERE m.length \
             >= $minLength\n  COLUMNS (m.id AS id, m.content AS content, m.creationDate AS created)\n)\n\
             ORDER BY created DESC, id ASC\nLIMIT 20",
            MIN_LENGTH,
        ),
        query(
            "scan_filter_avg_sql",
            "snb",
            Sql,
            "SELECT count(*) AS n, avg(len) AS mean\nFROM GRAPH_TABLE (snb\n  MATCH (c:Comment)\n  \
             WHERE c.length > $min\n  COLUMNS (c.length AS len)\n)",
            MIN,
        ),
        query(
            "hop_1",
            "snb",
            Cypher,
            "MATCH (s:Person {id: $personId})-[:KNOWS]-(f:Person)\nRETURN count(DISTINCT f.id) AS persons",
            HUB,
        ),
        query(
            "hop_2",
            "snb",
            Cypher,
            "MATCH (s:Person {id: $personId})-[:KNOWS]-(:Person)-[:KNOWS]-(f:Person)\nWHERE f.id <> \
             $personId\nRETURN count(DISTINCT f.id) AS persons",
            HUB,
        ),
        query(
            "hop_var",
            "snb",
            Cypher,
            "MATCH (s:Person {id: $personId})-[:KNOWS*1..3]-(f:Person)\nWHERE f.id <> $personId\n\
             RETURN count(DISTINCT f.id) AS persons",
            HUB,
        ),
        query(
            "hop_1_sql",
            "snb",
            Sql,
            "SELECT count(DISTINCT fid) AS persons\nFROM GRAPH_TABLE (snb\n  MATCH \
             (s:Person)-[k:KNOWS]-(f:Person)\n  WHERE s.id = $personId\n  COLUMNS (f.id AS fid)\n)",
            HUB,
        ),
        query(
            "hop_2_sql",
            "snb",
            Sql,
            "SELECT count(DISTINCT fid) AS persons\nFROM GRAPH_TABLE (snb\n  MATCH \
             (s:Person)-[k1:KNOWS]-(m:Person)-[k2:KNOWS]-(f:Person)\n  WHERE s.id = $personId AND \
             f.id <> $personId\n  COLUMNS (f.id AS fid)\n)",
            HUB,
        ),
        query(
            "hop_var_sql",
            "snb",
            Sql,
            "SELECT count(DISTINCT fid) AS persons\nFROM GRAPH_TABLE (snb\n  MATCH \
             (s:Person)-[k:KNOWS]-{1,3}(f:Person)\n  WHERE s.id = $personId AND f.id <> $personId\n  \
             COLUMNS (f.id AS fid)\n)",
            HUB,
        ),
        query(
            "pattern_triangle",
            "snb",
            Cypher,
            "MATCH (a:Person {id: $personId})-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(a)\n\
             WHERE b.id < c.id\nRETURN count(*) AS triangles",
            PERSON,
        ),
        query(
            "pattern_square",
            "snb",
            Cypher,
            "MATCH (a:Person {id: $personId})-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-\
             (d:Person)-[:KNOWS]-(a)\nWHERE b.id < d.id AND c.id <> $personId\nRETURN count(*) AS cycles",
            PERSON,
        ),
        query(
            "pattern_common",
            "snb",
            Cypher,
            "MATCH (a:Person {id: $person1Id})-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person {id: \
             $person2Id})\nRETURN count(DISTINCT c.id) AS common",
            PAIR,
        ),
        query(
            "pattern_join",
            "snb",
            Cypher,
            "MATCH (p:Person)-[:HAS_INTEREST]->(t:Tag), (m:Post)-[:HAS_TAG]->(t), \
             (m)-[:HAS_CREATOR]->(p)\nRETURN t.name AS tag, count(*) AS posts\nORDER BY posts DESC, \
             tag ASC",
            &[],
        ),
        query(
            "pattern_triangle_hub",
            "snb",
            Cypher,
            "MATCH (a:Person {id: $personId})-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(a)\n\
             WHERE b.id < c.id\nRETURN count(*) AS triangles",
            HUB,
        ),
        query(
            "pattern_triangle_all",
            "snb",
            Cypher,
            "MATCH (a:Person)-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(a)\nWHERE a.id < b.id \
             AND b.id < c.id\nRETURN count(*) AS triangles",
            &[],
        ),
        query(
            "pattern_triangle_sql",
            "snb",
            Sql,
            "SELECT count(*) AS triangles\nFROM GRAPH_TABLE (snb\n  MATCH \
             (a:Person)-[k1:KNOWS]-(b:Person)-[k2:KNOWS]-(c:Person)-[k3:KNOWS]-(a:Person)\n  WHERE \
             a.id = $personId AND b.id < c.id\n  COLUMNS (c.id AS cid)\n)",
            PERSON,
        ),
        query(
            "pattern_square_sql",
            "snb",
            Sql,
            "SELECT count(*) AS cycles\nFROM GRAPH_TABLE (snb\n  MATCH \
             (a:Person)-[k1:KNOWS]-(b:Person)-[k2:KNOWS]-(c:Person)-[k3:KNOWS]-(d:Person)-[k4:KNOWS]-\
             (a:Person)\n  WHERE a.id = $personId AND b.id < d.id AND c.id <> $personId\n  COLUMNS \
             (c.id AS cid)\n)",
            PERSON,
        ),
        query(
            "pattern_common_sql",
            "snb",
            Sql,
            "SELECT count(DISTINCT cid) AS common\nFROM GRAPH_TABLE (snb\n  MATCH \
             (a:Person)-[k1:KNOWS]-(c:Person)-[k2:KNOWS]-(b:Person)\n  WHERE a.id = $person1Id AND \
             b.id = $person2Id\n  COLUMNS (c.id AS cid)\n)",
            PAIR,
        ),
        query(
            "pattern_join_sql",
            "snb",
            Sql,
            "SELECT tag, count(*) AS posts\nFROM GRAPH_TABLE (snb\n  MATCH \
             (p:Person)-[i:HAS_INTEREST]->(t:Tag), (m:Post)-[h:HAS_TAG]->(t:Tag), \
             (m:Post)-[c:HAS_CREATOR]->(p:Person)\n  COLUMNS (t.name AS tag)\n)\nGROUP BY tag\n\
             ORDER BY posts DESC, tag ASC",
            &[],
        ),
        query(
            "pm_two_hop_count",
            "graphalytics",
            Cypher,
            "MATCH (a:V)-[:E]->(b:V)-[:E]->(c:V) RETURN count(*) AS paths",
            &[],
        ),
        query(
            "pm_two_hop_top10",
            "graphalytics",
            Cypher,
            "MATCH (a:V)-[:E]->(b:V)-[:E]->(c:V) WHERE a.id < $cut RETURN a.id AS id, count(*) AS \
             paths ORDER BY paths DESC, id ASC LIMIT 10",
            CUT,
        ),
        query(
            "agg_out_degree_top10",
            "graphalytics",
            Cypher,
            "MATCH (a:V)-[:E]->(:V) RETURN a.id AS id, count(*) AS degree ORDER BY degree DESC, id \
             ASC LIMIT 10",
            &[],
        ),
        query(
            "graphalytics_wcc",
            "graphalytics",
            Gql,
            "CALL grafeo.connected_components() YIELD node_id, component_id \
             RETURN count(DISTINCT component_id) AS components",
            &[],
        ),
        query(
            "sem_trail_self_loop",
            "multigraph",
            Cypher,
            "MATCH (a:V)-[:E]->(b:V)-[:E]->(c:V) RETURN count(*) AS paths",
            &[],
        ),
    ];

    /// Per state and query (`"<state> <query>"`), the count of each class above 0.
    type Counts = BTreeMap<String, BTreeMap<String, u64>>;

    /// Runs every query once in `state`, each statement's whole tally taken around its call, and
    /// returns each query's first row as text.
    fn run_queries(db: &GrafeoDB, state: &str, counts: &mut Counts) -> Vec<(String, String)> {
        let mut answers = Vec::new();
        for query in QUERIES {
            db.set_current_graph(Some(query.graph)).expect("the graph");
            let parameters: HashMap<String, Value> = query
                .parameters
                .iter()
                .map(|&(name, value)| {
                    let value = match value {
                        Parameter::Text(text) => Value::from(text),
                        Parameter::Int(number) => Value::Int64(number),
                    };
                    (name.to_string(), value)
                })
                .collect();
            let _ = work::take();
            let result = match query.language {
                Cypher => db.execute_cypher_with_params(query.text, parameters),
                Sql => db.execute_sql_with_params(query.text, parameters),
                Gql => db.execute_with_params(query.text, parameters),
            };
            let done = work::take();
            let result = result.unwrap_or_else(|error| panic!("{state} {}: {error}", query.id));
            let key = format!("{state} {}", query.id);
            counts.insert(
                key.clone(),
                done.nonzero()
                    .map(|(work, count)| (work.name().to_string(), count))
                    .collect(),
            );
            let first = result
                .rows()
                .first()
                .map_or_else(|| "no rows".to_string(), |row| format!("{row:?}"));
            answers.push((key, first));
        }
        answers
    }

    /// `count` distinct ids, drawn from `bound` numbers, as a list of what `id` holds.
    fn ids(
        random: &mut Xorshift,
        count: usize,
        bound: usize,
        id: impl Fn(usize) -> Value,
    ) -> Value {
        Value::List(
            distinct(random, count, bound)
                .into_iter()
                .map(id)
                .collect::<Vec<_>>()
                .into(),
        )
    }

    /// The second state: an older transaction opens and reads, then a seeded batch of updates
    /// and deletes commits. The returned session keeps that transaction open.
    fn change(db: &GrafeoDB, random: &mut Xorshift) -> grafeo_engine::Session {
        let mut older = db.session();
        older.begin_transaction().expect("an older transaction");
        older
            .execute("MATCH (n) RETURN count(n)")
            .expect("its first read");
        let run = |graph: &str, text: &str, list: Value| {
            db.set_current_graph(Some(graph)).expect("the graph");
            db.execute_cypher_with_params(text, HashMap::from([("ids".to_string(), list)]))
                .unwrap_or_else(|error| panic!("{text}: {error}"));
        };
        let comment = |i: usize| Value::from(format!("comment_{i}"));
        let person = |i: usize| Value::from(format!("person_{i}"));
        let vertex = |i: usize| Value::Int64(i64::try_from(i).expect("a small id"));
        run(
            "snb",
            "MATCH (c:Comment) WHERE c.id IN $ids SET c.length = c.length + 1",
            ids(random, 2_000, COMMENTS, comment),
        );
        run(
            "snb",
            "MATCH (c:Comment) WHERE c.id IN $ids DETACH DELETE c",
            ids(random, 1_000, COMMENTS, comment),
        );
        run(
            "snb",
            "MATCH (p:Person) WHERE p.id IN $ids SET p.browserUsed = 'Opera'",
            ids(random, 100, PERSONS, person),
        );
        run(
            "snb",
            "MATCH (p:Person)-[k:KNOWS]->(:Person) WHERE p.id IN $ids DELETE k",
            ids(random, 50, PERSONS, person),
        );
        run(
            "graphalytics",
            "MATCH (a:V)-[e:E]->(:V) WHERE a.id IN $ids DELETE e",
            ids(random, 100, VERTICES, vertex),
        );
        older
    }

    /// Loads the data, runs every query in both states, and returns the counts and the answers.
    fn run() -> (Counts, Vec<(String, String)>) {
        let config = Config::in_memory()
            .with_memory_limit(MEMORY_LIMIT)
            .with_query_timeout(QUERY_TIMEOUT);
        let db = GrafeoDB::with_config(config).expect("a database");
        let mut random = Xorshift::new(SEED);
        let dir =
            std::env::temp_dir().join(format!("grafeo-work-per-query-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap_or_else(|error| panic!("{}: {error}", dir.display()));
        for graph in [snb(&mut random), graphalytics(&mut random), multigraph()] {
            load(&db, &dir, &graph);
        }
        let _ = std::fs::remove_dir_all(&dir);

        let mut counts = Counts::new();
        let mut answers = run_queries(&db, "loaded", &mut counts);
        let mut older = change(&db, &mut random);
        answers.extend(run_queries(&db, "changed", &mut counts));
        older.rollback().expect("the older transaction ends");
        let stamped = counts["changed scan_label"]
            .get(Work::StampedRead.name())
            .copied()
            .unwrap_or(0);
        assert!(
            stamped > 0,
            "the second state reads through stamped vectors: {:?}",
            counts["changed scan_label"]
        );
        (counts, answers)
    }

    // ── The baseline ────────────────────────────────────────────────

    /// The target triple and the toolchain, from the `rustc` that cargo runs here (the pinned
    /// one), or "unknown".
    fn platform() -> (String, String) {
        let text = std::process::Command::new("rustc")
            .arg("-vV")
            .output()
            .map_or_else(
                |_| String::new(),
                |output| String::from_utf8_lossy(&output.stdout).into_owned(),
            );
        let line = |prefix: &str| {
            text.lines()
                .find_map(|line| line.strip_prefix(prefix))
                .map_or_else(|| "unknown".to_string(), |value| value.trim().to_string())
        };
        (line("host:"), line("release:"))
    }

    struct Baseline {
        target: String,
        toolchain: String,
        counts: Counts,
    }

    fn baseline_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/work_per_query.baseline.json")
    }

    fn to_json(target: &str, toolchain: &str, counts: &Counts) -> String {
        let workload = serde_json::json!({
            "seed": format!("{SEED:#x}"),
            "persons": PERSONS + 1,
            "knows": KNOWS,
            "comments": COMMENTS,
            "vertices": VERTICES,
            "edges": PAIRS * 2,
            "memory limit": MEMORY_LIMIT,
            "query timeout seconds": QUERY_TIMEOUT.as_secs(),
        });
        let document = serde_json::json!({
            "target": target,
            "toolchain": toolchain,
            "workload": workload,
            "work": counts,
        });
        serde_json::to_string_pretty(&document).expect("the counts serialize") + "\n"
    }

    fn read_baseline() -> Result<Baseline, String> {
        let path = baseline_path();
        let fail = |error: &dyn std::fmt::Display| format!("{}: {error}", path.display());
        let text = std::fs::read_to_string(&path).map_err(|error| fail(&error))?;
        let document: serde_json::Value =
            serde_json::from_str(&text).map_err(|error| fail(&error))?;
        let text_of = |name: &str| {
            document
                .get(name)
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| fail(&format!("no \"{name}\" text")))
        };
        let counts = serde_json::from_value(document.get("work").cloned().unwrap_or_default())
            .map_err(|error| fail(&error))?;
        Ok(Baseline {
            target: text_of("target")?,
            toolchain: text_of("toolchain")?,
            counts,
        })
    }

    /// Every count that differs from the baseline, either way (a class missing on one side
    /// counts 0 there), and every query in one and not the other, as a line.
    fn differences(counts: &Counts, baseline: &Counts) -> Vec<String> {
        let mut found = Vec::new();
        for (query, now) in counts {
            let Some(before) = baseline.get(query) else {
                found.push(format!("{query}: not in the baseline"));
                continue;
            };
            let classes: BTreeSet<&String> = now.keys().chain(before.keys()).collect();
            for class in classes {
                let (now, before) = (
                    now.get(class).copied().unwrap_or(0),
                    before.get(class).copied().unwrap_or(0),
                );
                if now != before {
                    let way = if now > before { "up" } else { "down" };
                    found.push(format!("{query} {class}: {now}, {way} from {before}"));
                }
            }
        }
        for query in baseline.keys() {
            if !counts.contains_key(query) {
                found.push(format!("{query}: in the baseline, not run any more"));
            }
        }
        found
    }

    /// The comparison `--check` applies, on fixed counts: checked before every check, as a bench
    /// target runs no `#[test]`.
    fn assert_the_comparison_holds() {
        let counts_of = |entries: &[(&str, &[(&str, u64)])]| -> Counts {
            entries
                .iter()
                .map(|(query, classes)| {
                    let classes = classes
                        .iter()
                        .map(|&(class, count)| (class.to_string(), count))
                        .collect();
                    (query.to_string(), classes)
                })
                .collect()
        };
        let counts = counts_of(&[
            ("risen", &[("parses", 2)]),
            ("dropped", &[("parses", 1)]),
            ("same", &[("parses", 1), ("plans_built", 1)]),
            ("class gone", &[]),
            ("class new", &[("parses", 1)]),
            ("query new", &[("parses", 1)]),
        ]);
        let baseline = counts_of(&[
            ("risen", &[("parses", 1)]),
            ("dropped", &[("parses", 2)]),
            ("same", &[("parses", 1), ("plans_built", 1)]),
            ("class gone", &[("parses", 1)]),
            ("class new", &[]),
            ("query gone", &[("parses", 1)]),
        ]);
        let found = differences(&counts, &baseline);
        let named = |name: &str| found.iter().filter(|line| line.starts_with(name)).count();
        assert_eq!(named("risen "), 1, "one more fails: {found:?}");
        assert_eq!(named("dropped "), 1, "one fewer fails: {found:?}");
        assert_eq!(
            named("class gone "),
            1,
            "a class counted 0 now fails: {found:?}"
        );
        assert_eq!(
            named("class new "),
            1,
            "a class counted 0 before fails: {found:?}"
        );
        assert_eq!(
            named("query new:"),
            1,
            "a query the baseline lacks fails: {found:?}"
        );
        assert_eq!(
            named("query gone:"),
            1,
            "a query no longer run fails: {found:?}"
        );
        assert_eq!(named("same"), 0, "equal counts pass: {found:?}");
        assert_eq!(found.len(), 6, "{found:?}");
    }

    fn table(counts: &Counts, answers: &[(String, String)]) -> String {
        let mut out = String::from("| Query | Answer | Work |\n| --- | --- | --- |\n");
        for (query, answer) in answers {
            let work = counts[query]
                .iter()
                .map(|(class, count)| format!("{class}={count}"))
                .collect::<Vec<_>>()
                .join(" ");
            // Writing to a String cannot fail.
            let _ = writeln!(out, "| {query} | {answer} | {work} |");
        }
        out
    }

    pub fn main() {
        // `cargo bench` passes `--bench`; the bench's own flags follow `--`.
        let args: Vec<String> = std::env::args().skip(1).collect();
        let check = args.iter().any(|arg| arg == "--check");
        let update = args.iter().any(|arg| arg == "--update-baseline");
        let out = args
            .iter()
            .position(|arg| arg == "--out")
            .and_then(|at| args.get(at + 1))
            .map(PathBuf::from);

        let (counts, answers) = run();
        let (target, toolchain) = platform();
        let table = table(&counts, &answers);
        println!("Work per query on {target}, Rust {toolchain}\n\n{table}");
        let json = to_json(&target, &toolchain, &counts);
        if let Some(out) = out {
            std::fs::write(&out, &json)
                .unwrap_or_else(|error| panic!("{}: {error}", out.display()));
        }
        if update {
            let path = baseline_path();
            std::fs::write(&path, &json)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            println!("Baseline written to {}", path.display());
        }
        if !check {
            return;
        }
        assert_the_comparison_holds();
        let baseline = read_baseline().unwrap_or_else(|error| {
            eprintln!("no baseline to check against: {error}");
            std::process::exit(2)
        });
        let differences = differences(&counts, &baseline.counts);
        let summary = if differences.is_empty() {
            "Every count equals the baseline.".to_string()
        } else {
            format!(
                "{} counts differ from the baseline ({} on {}):\n  {}",
                differences.len(),
                baseline.toolchain,
                baseline.target,
                differences.join("\n  ")
            )
        };
        if let Ok(path) = std::env::var("GITHUB_STEP_SUMMARY") {
            let text = format!("## Work per query (#794)\n\n{summary}\n\n{table}\n");
            if let Err(error) = std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(&path)
                .and_then(|mut file| std::io::Write::write_all(&mut file, text.as_bytes()))
            {
                eprintln!("could not write the step summary {path}: {error}");
            }
        }
        if differences.is_empty() {
            println!("{summary}");
        } else if baseline.target != target {
            // The allocation classes follow the platform's std: only the baseline's target gates.
            println!("{summary}\nReported, not checked: this is {target}.");
        } else {
            eprintln!(
                "{summary}\nFor a change that is intended (a drop too, or a toolchain bump: this is \
                 Rust {toolchain}), run with --update-baseline and commit the baseline."
            );
            std::process::exit(1);
        }
    }
}
