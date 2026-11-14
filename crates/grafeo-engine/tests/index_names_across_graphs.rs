//! #723: an index name belongs to the database, and each graph's indexes are its own. A graph
//! lists a name only when it has the index the name stands for, `DROP INDEX` keeps a name while
//! another graph still has its index, and every drop that takes a name's last index (a
//! `DROP INDEX`, a direct drop through the API, a dropped graph) frees the name. Each case runs
//! for the four kinds of index a name can stand for, and `SHOW INDEXES` is read in GQL and Cypher.
//!
//! The oracle, unless a test names another: the listing in each graph, and whether the name still
//! exists, read as a `CREATE INDEX` of the name with another definition failing with "already
//! exists" (tried in a transaction that is rolled back).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main index_names_across_graphs::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

/// A kind of index a name can stand for, with the definition the tests give it.
#[derive(Clone, Copy, Debug)]
struct Kind {
    name: &'static str,
    label: &'static str,
    property: &'static str,
    using: &'static str,
}

const HASH: Kind = Kind {
    name: "by_id",
    label: "File",
    property: "id",
    using: "",
};
const BTREE: Kind = Kind {
    name: "by_rank",
    label: "File",
    property: "rank",
    using: " USING BTREE",
};
const TEXT: Kind = Kind {
    name: "doc_body",
    label: "Doc",
    property: "body",
    using: " USING TEXT",
};
const VECTOR: Kind = Kind {
    name: "doc_emb",
    label: "Doc",
    property: "emb",
    using: " USING VECTOR {dimensions: 3, metric: 'cosine'}",
};
const KINDS: [Kind; 4] = [HASH, BTREE, TEXT, VECTOR];

impl Kind {
    fn create(self) -> String {
        format!(
            "CREATE INDEX {} FOR (n:{}) ON (n.{}){}",
            self.name, self.label, self.property, self.using
        )
    }

    /// The name with a definition of its own, which fails while the name exists.
    fn other_definition(self) -> String {
        format!("CREATE INDEX {} FOR (n:Person) ON (n.nick)", self.name)
    }

    /// The row `SHOW INDEXES` gives a database that holds this index alone.
    fn row(self) -> [String; 4] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute(&self.create()).unwrap();
        let rows = listed(&session);
        assert_eq!(rows.len(), 1, "{self:?}: one row for one index: {rows:?}");
        rows[0].clone()
    }
}

fn run_all(session: &Session, statements: &[&str]) {
    for statement in statements {
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
}

fn rows_of(result: &grafeo_engine::database::QueryResult) -> Vec<[String; 4]> {
    let mut rows: Vec<[String; 4]> = result
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| match value {
                    Value::String(value) => value.to_string(),
                    Value::Null => "NULL".to_string(),
                    other => panic!("SHOW INDEXES lists strings and nulls: {other:?}"),
                })
                .collect::<Vec<_>>()
                .try_into()
                .expect("four columns")
        })
        .collect();
    rows.sort();
    rows
}

/// What `SHOW INDEXES` lists in the session's graph, sorted; GQL and Cypher agree.
fn listed(session: &Session) -> Vec<[String; 4]> {
    let gql = rows_of(&session.execute("SHOW INDEXES").unwrap());
    let cypher = rows_of(&session.execute_cypher("SHOW INDEXES").unwrap());
    assert_eq!(gql, cypher, "GQL and Cypher list the same");
    gql
}

/// The names `SHOW INDEXES` lists in the session's graph.
fn names(session: &Session) -> Vec<String> {
    listed(session)
        .into_iter()
        .map(|[name, ..]| name)
        .filter(|name| name != "NULL")
        .collect()
}

/// Whether `name` exists in the database: a `CREATE INDEX` of it with another definition fails
/// with "already exists" (in a transaction that is rolled back).
fn name_exists(db: &GrafeoDB, kind: Kind) -> bool {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let created = session.execute(&kind.other_definition());
    session.rollback().unwrap();
    match created {
        Ok(_) => false,
        Err(error) if error.to_string().contains("already exists") => true,
        Err(error) => panic!("{kind:?}: {error}"),
    }
}

fn use_graph(session: &Session, graph: &str) {
    session.execute(&format!("USE GRAPH {graph}")).unwrap();
}

/// Whether a lookup of `label` by `key` reads an index.
fn reads_index(session: &Session, label: &str, key: &str) -> bool {
    let query = format!("UNWIND [1] AS x MATCH (n:{label}) WHERE n.{key} = x RETURN n.{key}");
    let plan = session.execute(&format!("EXPLAIN {query}")).unwrap().rows()[0][0]
        .as_str()
        .unwrap()
        .to_string();
    plan.contains(&format!("[index: {key}]"))
}

/// A database with the graph `other` and `kind`'s index in the default graph and in `other`.
fn in_both(kind: Kind) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &["CREATE GRAPH other", &kind.create()]);
    use_graph(&session, "other");
    session.execute(&kind.create()).unwrap();
    db
}

fn status(error: &grafeo_common::utils::error::Error) -> String {
    GqlStatus::from(error).as_str().to_string()
}

// ---------------------------------------------------------------------------
// Listing (patch 1)
// ---------------------------------------------------------------------------

/// A graph without the index lists no row for its name, whichever graph has it; the API's
/// `list_indexes` agrees for the current graph.
#[test]
fn a_graph_without_the_index_lists_no_name() {
    for kind in KINDS {
        let row = kind.row();
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        run_all(&session, &["CREATE GRAPH other", &kind.create()]);
        assert_eq!(
            listed(&session),
            std::slice::from_ref(&row),
            "{kind:?}: the default graph"
        );
        use_graph(&session, "other");
        assert_eq!(
            listed(&session),
            Vec::<[String; 4]>::new(),
            "{kind:?}: other"
        );
        db.set_current_graph(Some("other")).unwrap();
        assert!(
            db.list_indexes()
                .iter()
                .all(|index| index.name != kind.name),
            "{kind:?}: list_indexes in other"
        );
        db.set_current_graph(None).unwrap();
        assert!(
            db.list_indexes()
                .iter()
                .any(|index| index.name == kind.name),
            "{kind:?}: list_indexes in the default graph"
        );

        // The index in a named graph only.
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        run_all(
            &session,
            &["CREATE GRAPH other", "USE GRAPH other", &kind.create()],
        );
        assert_eq!(listed(&session), [row], "{kind:?}: other has it");
        use_graph(&session, "default");
        assert_eq!(
            listed(&session),
            Vec::<[String; 4]>::new(),
            "{kind:?}: the default graph has none"
        );
    }
}

/// In a transaction, an index it creates is listed in its graph and in no other, before and after
/// the commit.
#[test]
fn a_queued_index_is_listed_in_its_graph_only() {
    for kind in KINDS {
        let row = kind.row();
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        session.execute("CREATE GRAPH other").unwrap();
        session.begin_transaction().unwrap();
        run_all(&session, &["USE GRAPH other", &kind.create()]);
        assert_eq!(
            listed(&session),
            std::slice::from_ref(&row),
            "{kind:?}: queued, in other"
        );
        use_graph(&session, "default");
        assert!(listed(&session).is_empty(), "{kind:?}: queued, default");
        session.commit().unwrap();
        assert!(listed(&session).is_empty(), "{kind:?}: committed, default");
        use_graph(&session, "other");
        assert_eq!(listed(&session), [row], "{kind:?}: committed, in other");
    }
}

/// The same `CREATE INDEX` twice in one transaction gives what two statements give.
#[test]
fn a_second_create_in_one_transaction_fails_as_two_statements_do() {
    for kind in KINDS {
        let outcomes = |in_transaction: bool| {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();
            if in_transaction {
                session.begin_transaction().unwrap();
            }
            let first = session
                .execute(&kind.create())
                .map(drop)
                .map_err(|e| status(&e));
            let second = session
                .execute(&kind.create())
                .map(drop)
                .map_err(|e| status(&e));
            (first, second)
        };
        assert_eq!(outcomes(true), outcomes(false), "{kind:?}");
    }
}

/// A Hash name whose index another graph serves with an index on the property of every label is
/// listed there by its name, not as an unnamed row.
#[test]
fn a_name_served_by_a_key_wide_index_is_listed_by_name() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &["CREATE GRAPH other", &HASH.create()]);
    use_graph(&session, "other");
    session.create_property_index("id").unwrap();
    assert_eq!(listed(&session), [HASH.row()]);
}

// ---------------------------------------------------------------------------
// DROP INDEX (patch 2)
// ---------------------------------------------------------------------------

/// The issue's case, for each kind: `DROP INDEX` in one graph keeps the name for the other, which
/// keeps listing it; the drop in the last graph frees the name for another definition.
#[test]
fn drop_index_keeps_a_name_another_graph_uses() {
    for kind in KINDS {
        let row = kind.row();
        let db = in_both(kind);
        let session = db.session();
        session
            .execute(&format!("DROP INDEX {}", kind.name))
            .unwrap();
        assert!(listed(&session).is_empty(), "{kind:?}: dropped here");
        use_graph(&session, "other");
        assert_eq!(
            listed(&session),
            std::slice::from_ref(&row),
            "{kind:?}: other keeps it"
        );
        if kind.using.is_empty() {
            assert!(reads_index(&session, kind.label, kind.property));
        }
        assert!(name_exists(&db, kind), "{kind:?}: the name stays");

        session
            .execute(&format!("DROP INDEX {}", kind.name))
            .unwrap();
        assert!(
            listed(&session).is_empty(),
            "{kind:?}: the last one dropped"
        );
        assert!(!name_exists(&db, kind), "{kind:?}: the name is free");
        session.execute(&kind.other_definition()).unwrap();
    }
}

/// A name over two properties stays while another graph has the index of one of them, and that
/// graph lists that property's row alone.
#[test]
fn a_name_over_two_properties_stays_for_one_property() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let create = "CREATE INDEX doc_ab FOR (n:Doc) ON (n.a, n.b)";
    run_all(&session, &["CREATE GRAPH other", create]);
    let both = listed(&session);
    assert_eq!(both.len(), 2, "{both:?}");
    use_graph(&session, "other");
    session.execute(create).unwrap();
    assert!(session.drop_property_index("b").unwrap());
    let a_row: Vec<[String; 4]> = both.iter().filter(|row| row[3] == "a").cloned().collect();
    assert_eq!(listed(&session), a_row, "other lists a alone");
    use_graph(&session, "default");
    session.execute("DROP INDEX doc_ab").unwrap();
    assert_eq!(listed(&session), Vec::<[String; 4]>::new());
    use_graph(&session, "other");
    assert_eq!(listed(&session), a_row, "other keeps the name for a");
    let refused = session
        .execute("CREATE INDEX doc_ab FOR (n:Person) ON (n.nick)")
        .unwrap_err();
    assert!(refused.to_string().contains("already exists"), "{refused}");
}

/// Inside a transaction a kept drop lists as the queue leaves it; a rollback restores both graphs,
/// a commit keeps the drop and the name.
#[test]
fn a_kept_drop_in_a_transaction() {
    for commit in [false, true] {
        let row = HASH.row();
        let db = in_both(HASH);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute("DROP INDEX by_id").unwrap();
        assert!(listed(&session).is_empty(), "queued drop, default");
        use_graph(&session, "other");
        assert_eq!(
            listed(&session),
            std::slice::from_ref(&row),
            "queued drop, other"
        );
        if commit {
            session.commit().unwrap();
        } else {
            session.rollback().unwrap();
        }
        assert_eq!(
            listed(&session),
            std::slice::from_ref(&row),
            "commit {commit}: other"
        );
        use_graph(&session, "default");
        assert_eq!(
            listed(&session).is_empty(),
            commit,
            "commit {commit}: default"
        );
        assert!(name_exists(&db, HASH), "commit {commit}: the name stays");
    }
}

/// A text or vector index created and dropped in one transaction leaves neither index nor name,
/// named or unnamed.
#[test]
fn an_index_created_and_dropped_in_one_transaction_leaves_nothing() {
    for kind in KINDS {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute(&kind.create()).unwrap();
        session
            .execute(&format!("DROP INDEX {}", kind.name))
            .unwrap();
        session.commit().unwrap();
        assert!(
            listed(&session).is_empty(),
            "{kind:?}: {:?}",
            listed(&session)
        );
        assert!(db.list_indexes().is_empty(), "{kind:?}: list_indexes");
        assert!(!name_exists(&db, kind), "{kind:?}");
    }
}

/// A transaction that drops a name's index in one graph, keeping the name, holds it: another
/// writer's drop of the name in the other graph fails with a write conflict while it is open, and
/// the other graph keeps its listing.
#[test]
fn two_drops_of_one_name_conflict() {
    let row = HASH.row();
    let db = in_both(HASH);
    let mut first = db.session();
    first.begin_transaction().unwrap();
    first.execute("DROP INDEX by_id").unwrap();

    let second = db.session();
    use_graph(&second, "other");
    let refused = second.execute("DROP INDEX by_id").unwrap_err();
    assert_eq!(status(&refused), "40000", "{refused}");

    let mut third = db.session();
    third.begin_transaction().unwrap();
    use_graph(&third, "other");
    let refused = third.execute("DROP INDEX by_id").unwrap_err();
    assert_eq!(status(&refused), "40000", "an open transaction: {refused}");
    third.rollback().unwrap();

    first.commit().unwrap();
    assert_eq!(listed(&second), [row], "other keeps its index and name");
    assert!(name_exists(&db, HASH));
}

/// A direct drop of an index whose name an open transaction holds fails with `GRAFEO-T001`.
#[test]
fn a_direct_drop_meets_the_claim_of_an_open_drop() {
    let db = in_both(VECTOR);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    use_graph(&session, "other");
    session.execute("DROP INDEX doc_emb").unwrap();
    let refused = db.drop_vector_index("Doc", "emb").unwrap_err();
    assert!(refused.to_string().contains("GRAFEO-T001"), "{refused}");
    session.rollback().unwrap();
    assert!(db.drop_vector_index("Doc", "emb").unwrap());
}

/// `DROP INDEX x` in a graph that has none of x's indexes fails while another graph has one, and
/// `IF EXISTS` changes nothing there; the error is the one of a name that does not exist.
#[test]
fn drop_index_in_a_graph_without_it_fails_while_another_has_it() {
    let row = HASH.row();
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &["CREATE GRAPH other", "USE GRAPH other", &HASH.create()],
    );
    use_graph(&session, "default");
    let missing = status(&session.execute("DROP INDEX no_such_index").unwrap_err());
    let refused = session.execute("DROP INDEX by_id").unwrap_err();
    assert_eq!(status(&refused), missing, "{refused}");
    assert!(
        refused.to_string().contains("does not exist in this graph"),
        "{refused}"
    );
    session.execute("DROP INDEX IF EXISTS by_id").unwrap();
    use_graph(&session, "other");
    assert_eq!(listed(&session), [row]);
    assert!(name_exists(&db, HASH));
}

/// A name no graph has an index for, as 0.6.1 could leave behind (its `DROP GRAPH` kept the
/// names of the indexes it dropped), is listed in no graph, and one `DROP INDEX` removes it. The
/// orphan is on `Doc.code`: the file's `kept` index is on `id` for every label (0.6.1 built it so),
/// so an orphan on `id` would have that index and be no orphan.
#[test]
fn an_orphan_name_from_0_6_1_is_cleared_by_one_drop() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/orphan_index_name/0.6.1.grafeo");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("orphan.grafeo");
    std::fs::copy(&fixture, &path).unwrap();
    let orphan = Kind {
        name: "orphan",
        label: "Doc",
        property: "code",
        using: "",
    };
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    assert_eq!(names(&session), ["kept"], "the orphan is listed nowhere");
    assert!(name_exists(&db, orphan), "the 0.6.1 file holds the name");
    session.execute("DROP INDEX IF EXISTS orphan").unwrap();
    session.execute("DROP INDEX orphan").unwrap_err();
    assert!(!name_exists(&db, orphan), "IF EXISTS removed it");
    drop(session);
    db.close().unwrap();

    std::fs::copy(&fixture, &path).unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    session.execute("DROP INDEX orphan").unwrap();
    assert!(!name_exists(&db, orphan));
    session.execute(&orphan.other_definition()).unwrap();
    assert_eq!(names(&session), ["kept", "orphan"]);
    drop(session);
    db.close().unwrap();
}

/// Two names on one index: `c1`, made on the key `by_id` has an index on, stands for that index in
/// every graph that has it. `DROP INDEX by_id` removes the name `by_id` from every graph and keeps
/// the index, which `c1` still stands for.
#[test]
fn a_second_name_keeps_the_index_and_the_dropped_name_goes_everywhere() {
    let db = in_both(HASH);
    let session = db.session();
    session
        .execute("CREATE INDEX c1 FOR (n:File) ON (n.id)")
        .unwrap();
    for graph in ["other", "default"] {
        use_graph(&session, graph);
        assert_eq!(names(&session), ["by_id", "c1"], "{graph}: both names");
    }
    session.execute("DROP INDEX by_id").unwrap();
    for graph in ["default", "other"] {
        use_graph(&session, graph);
        assert_eq!(names(&session), ["c1"], "{graph}");
        assert!(reads_index(&session, "File", "id"), "{graph}");
    }
    assert!(!name_exists(&db, HASH));
}

/// A graph the transaction created counts: its index keeps the name a drop in the default graph
/// would otherwise free.
#[test]
fn a_graph_made_in_the_transaction_keeps_the_name() {
    let row = HASH.row();
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.execute(&HASH.create()).unwrap();
    session.begin_transaction().unwrap();
    run_all(
        &session,
        &[
            "CREATE GRAPH later",
            "USE GRAPH later",
            &HASH.create(),
            "USE GRAPH default",
            "DROP INDEX by_id",
        ],
    );
    session.commit().unwrap();
    assert_eq!(listed(&session), Vec::<[String; 4]>::new());
    use_graph(&session, "later");
    assert_eq!(listed(&session), [row]);
    assert!(name_exists(&db, HASH));
}

/// The listings and the names survive a checkpoint and a reopen.
#[test]
fn the_names_survive_a_reopen() {
    let row = HASH.row();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("names.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        run_all(
            &session,
            &[
                "CREATE GRAPH other",
                &HASH.create(),
                &VECTOR.create(),
                "USE GRAPH other",
                &HASH.create(),
                "USE GRAPH default",
                "DROP INDEX by_id",
            ],
        );
        drop(session);
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_reopened_names(&db, &row);
    db.close().unwrap();
}

/// After a reopen of the sequence of `the_names_survive_a_reopen`: the default graph lists
/// `doc_emb`, `other` lists `by_id`, and both names exist.
fn assert_reopened_names(db: &GrafeoDB, row: &[String; 4]) {
    let session = db.session();
    assert_eq!(names(&session), ["doc_emb"], "the default graph");
    use_graph(&session, "other");
    assert_eq!(listed(&session), std::slice::from_ref(row), "other");
    assert!(name_exists(db, HASH));
    assert!(name_exists(db, VECTOR));
}

// ---------------------------------------------------------------------------
// The direct drops (patch 3)
// ---------------------------------------------------------------------------

/// The four direct drops, each on the index of the current graph.
fn direct_drops() -> Vec<(Kind, &'static str, fn(&GrafeoDB) -> bool)> {
    vec![
        (VECTOR, "drop_vector_index", |db| {
            db.drop_vector_index("Doc", "emb").unwrap()
        }),
        (TEXT, "drop_text_index", |db| {
            db.drop_text_index("Doc", "body").unwrap()
        }),
        (HASH, "GrafeoDB::drop_property_index", |db| {
            db.drop_property_index("id").unwrap()
        }),
        (HASH, "Session::drop_property_index", |db| {
            db.session().drop_property_index("id").unwrap()
        }),
    ]
}

/// A direct drop of a named index frees the name when no other graph has its index, and keeps it
/// when one does; a drop that finds no index changes nothing.
#[test]
fn a_direct_drop_frees_a_name_no_graph_uses() {
    for (kind, call, drop_index) in direct_drops() {
        let row = kind.row();

        let db = GrafeoDB::new_in_memory();
        db.session().execute(&kind.create()).unwrap();
        assert!(drop_index(&db), "{call}: found");
        assert!(listed(&db.session()).is_empty(), "{call}: one graph");
        assert!(!name_exists(&db, kind), "{call}: the name is free");
        db.session().execute(&kind.other_definition()).unwrap();

        let db = in_both(kind);
        assert!(drop_index(&db), "{call}: found in the default graph");
        assert!(listed(&db.session()).is_empty(), "{call}: dropped here");
        let session = db.session();
        use_graph(&session, "other");
        assert_eq!(
            listed(&session),
            std::slice::from_ref(&row),
            "{call}: other keeps it"
        );
        assert!(name_exists(&db, kind), "{call}: the name stays");

        // Only `other` has it now: a drop in the default graph finds nothing.
        assert!(!drop_index(&db), "{call}: nothing to drop");
        assert_eq!(listed(&session), [row], "{call}: unchanged");
        assert!(name_exists(&db, kind), "{call}: unchanged");
    }
}

// ---------------------------------------------------------------------------
// A dropped graph (patch 5)
// ---------------------------------------------------------------------------

/// Every way a graph is dropped frees the names of the indexes only it had, and keeps a name
/// another graph still has.
#[test]
fn a_dropped_graph_frees_the_names_only_it_used() {
    type Drop = fn(&GrafeoDB, &str);
    let ways: [(&str, Drop); 3] = [
        ("DROP GRAPH", |db, graph| {
            db.execute(&format!("DROP GRAPH {graph}")).unwrap();
        }),
        ("DROP GRAPH in a transaction", |db, graph| {
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session.execute(&format!("DROP GRAPH {graph}")).unwrap();
            session.commit().unwrap();
        }),
        ("GrafeoDB::drop_graph", |db, graph| {
            assert!(db.drop_graph(graph).unwrap());
        }),
    ];
    for (way, drop_graph) in ways {
        for kind in KINDS {
            let db = GrafeoDB::new_in_memory();
            run_all(
                &db.session(),
                &["CREATE GRAPH gone", "USE GRAPH gone", &kind.create()],
            );
            drop_graph(&db, "gone");
            assert!(!name_exists(&db, kind), "{way} {kind:?}: the name is free");
            assert!(listed(&db.session()).is_empty(), "{way} {kind:?}");

            let db = in_both(kind);
            drop_graph(&db, "other");
            assert!(
                name_exists(&db, kind),
                "{way} {kind:?}: the default keeps it"
            );
            assert_eq!(listed(&db.session()), [kind.row()], "{way} {kind:?}");
        }
    }
}

/// `DROP SCHEMA` drops the schema's default graph, the fifth way a graph goes: a name only that
/// graph used is freed, a name another graph still has stays, and in each case the listing (GQL and
/// Cypher) and the name lookup agree afterwards. A schema made again under the same name starts
/// with no index and no name.
#[test]
fn a_dropped_schema_frees_the_names_only_its_default_graph_used() {
    for statement in ["DROP SCHEMA staging", "DROP SCHEMA IF EXISTS staging"] {
        for kind in KINDS {
            let db = GrafeoDB::new_in_memory();
            let inside = db.session();
            run_all(
                &inside,
                &[
                    "CREATE SCHEMA staging",
                    "SESSION SET SCHEMA staging",
                    &kind.create(),
                ],
            );
            assert_eq!(names(&inside), [kind.name], "{statement} {kind:?}: before");
            let root = db.session();
            assert!(
                listed(&root).is_empty(),
                "{statement} {kind:?}: the root lists nothing"
            );
            root.execute(statement).unwrap();
            assert!(
                !name_exists(&db, kind),
                "{statement} {kind:?}: the name is free"
            );
            assert!(
                listed(&root).is_empty(),
                "{statement} {kind:?}: nothing listed"
            );
            let again = db.session();
            run_all(
                &again,
                &["CREATE SCHEMA staging", "SESSION SET SCHEMA staging"],
            );
            assert!(
                listed(&again).is_empty(),
                "{statement} {kind:?}: a new schema of the name has no index"
            );
            root.execute(&kind.other_definition())
                .unwrap_or_else(|error| {
                    panic!("{statement} {kind:?}: the name is reused: {error}")
                });

            let db = GrafeoDB::new_in_memory();
            let root = db.session();
            run_all(&root, &[&kind.create(), "CREATE SCHEMA staging"]);
            let inside = db.session();
            run_all(&inside, &["SESSION SET SCHEMA staging", &kind.create()]);
            assert_eq!(names(&inside), [kind.name], "{statement} {kind:?}: in both");
            root.execute(statement).unwrap();
            assert!(
                name_exists(&db, kind),
                "{statement} {kind:?}: the root graph keeps the name"
            );
            assert_eq!(
                listed(&root),
                [kind.row()],
                "{statement} {kind:?}: and lists it"
            );
        }
    }
}

/// SQL/PGQ drops a property graph with `DROP PROPERTY GRAPH` and with `CREATE OR REPLACE PROPERTY
/// GRAPH`: either frees the names of the indexes only that graph had. A replace is a drop and a
/// create, for index names too: the new graph lists what a fresh create of it lists, its key
/// rule's index unnamed.
#[cfg(feature = "sql-pgq")]
#[test]
fn a_dropped_property_graph_frees_the_names_only_it_used() {
    for drop_it in [
        "DROP PROPERTY GRAPH Shop",
        "CREATE OR REPLACE PROPERTY GRAPH Shop NODE TABLES (Item (sku VARCHAR PRIMARY KEY))",
    ] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute_sql(
                "CREATE PROPERTY GRAPH Shop NODE TABLES (Item (sku VARCHAR PRIMARY KEY, name VARCHAR))",
            )
            .unwrap();
        run_all(
            &session,
            &[
                "USE GRAPH Shop",
                "CREATE INDEX shop_name FOR (n:Item) ON (n.name)",
            ],
        );
        assert_eq!(names(&session), ["shop_name"], "{drop_it}: before");
        use_graph(&session, "default");
        session.execute_sql(drop_it).unwrap();
        let shop_name = Kind {
            name: "shop_name",
            label: "Item",
            property: "name",
            using: "",
        };
        assert!(!name_exists(&db, shop_name), "{drop_it}: the name is free");
        if drop_it.starts_with("CREATE OR REPLACE") {
            // A drop and a create: the new graph lists what a fresh create of it lists, the key
            // rule's index unnamed.
            let fresh = GrafeoDB::new_in_memory();
            let fresh_session = fresh.session();
            fresh_session
                .execute_sql(&drop_it.replace("CREATE OR REPLACE", "CREATE"))
                .unwrap();
            use_graph(&fresh_session, "Shop");
            use_graph(&session, "Shop");
            assert_eq!(
                listed(&session),
                listed(&fresh_session),
                "{drop_it}: as after a fresh create"
            );
            assert!(names(&session).is_empty(), "{drop_it}: no name is listed");
        }
    }
}

#[cfg(feature = "testing-crash-injection")]
mod crash {
    use super::*;
    use std::path::{Path, PathBuf};

    const PATH_VAR: &str = "GRAFEO_INDEX_NAMES_CRASH_PATH";
    const CASE_VAR: &str = "GRAFEO_INDEX_NAMES_CRASH_CASE";
    const POINT_VAR: &str = "GRAFEO_INDEX_NAMES_CRASH_POINT";
    /// The child's exit code when the drop completed before the crash point.
    const COMPLETED: i32 = 10;

    /// The drops the sweep crashes in: a direct drop of a named Hash index, and `DROP INDEX` of
    /// the last graph's index.
    const CASES: [&str; 2] = ["drop_property_index", "drop_index"];

    /// Builds the database at `GRAFEO_INDEX_NAMES_CRASH_PATH` with `by_id` in the default graph
    /// alone and closes it, then reopens it and runs the case's drop with a crash at point
    /// `GRAFEO_INDEX_NAMES_CRASH_POINT`: the process exits there with the database open (exit
    /// code 0), or with `COMPLETED` when the drop finished first.
    #[test]
    fn crash_child() {
        let Some(path) = std::env::var_os(PATH_VAR) else {
            return;
        };
        let path = PathBuf::from(path);
        let case = std::env::var(CASE_VAR).unwrap();
        let point: u64 = std::env::var(POINT_VAR).unwrap().parse().unwrap();
        {
            let db = GrafeoDB::open(&path).unwrap();
            run_all(
                &db.session(),
                &[
                    "CREATE GRAPH other",
                    &HASH.create(),
                    "INSERT (:File {id: 1})",
                ],
            );
            db.close().unwrap();
        }
        let db = GrafeoDB::open(&path).unwrap();
        std::panic::set_hook(Box::new(|info| {
            eprintln!("{info}");
            std::process::exit(0);
        }));
        grafeo_common::testing::crash::enable_crash_at(point);
        match case.as_str() {
            "drop_property_index" => {
                db.drop_property_index("id").unwrap();
            }
            "drop_index" => {
                db.execute("DROP INDEX by_id").unwrap();
            }
            other => panic!("no case {other}"),
        }
        grafeo_common::testing::crash::disable_crash();
        std::process::exit(COMPLETED);
    }

    fn crash(path: &Path, case: &str, point: u64) -> Option<i32> {
        grafeo_common::testing::child_process::run(
            grafeo_common::child_test!("crash_child")
                .env(PATH_VAR, path)
                .env(CASE_VAR, case)
                .env(POINT_VAR, point.to_string()),
        )
        .unwrap()
        .code()
    }

    /// A crash at any point of either drop leaves the name and the index changed together: after
    /// the reopen the default graph lists `by_id` exactly when it reads the index, and the name
    /// exists exactly then.
    #[test]
    fn a_crash_in_a_drop_changes_the_name_and_the_index_together() {
        let dir = tempfile::tempdir().unwrap();
        for case in CASES {
            let mut crashed = 0;
            for point in 1..=64u64 {
                let path = dir.path().join(format!("{case}-{point}.grafeo"));
                match crash(&path, case, point) {
                    Some(0) => crashed += 1,
                    Some(COMPLETED) => break,
                    other => panic!("{case} point {point}: the child failed with {other:?}"),
                }
                let db = GrafeoDB::open(&path).unwrap();
                let session = db.session();
                let has = reads_index(&session, "File", "id");
                assert_eq!(
                    names(&session) == ["by_id"],
                    has,
                    "{case} point {point}: listed exactly when the index is there"
                );
                assert_eq!(
                    name_exists(&db, HASH),
                    has,
                    "{case} point {point}: the name"
                );
                drop(session);
                db.close().unwrap();
            }
            assert!(crashed > 0, "{case}: no crash point was reached");
        }
    }
}
