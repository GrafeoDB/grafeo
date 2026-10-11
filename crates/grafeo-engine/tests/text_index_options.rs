//! A text index keeps its options (#351): the BM25 parameters k1 and b, the
//! tokenizer and the stop words. Scores follow k1 and b as BM25 says, the
//! default options score as before, and the options come back whatever
//! path a reopen takes: the index section restored after `close()`, the
//! index built from the data after a crash (from the WAL), or after a
//! checkpoint that left the index sections out because a transaction had
//! open changes. Copies (`to_memory`, `save`) and `rebuild_text_index` keep
//! them, and `CREATE INDEX ... USING TEXT {...}` sets them.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test text_index_options
//! ```

#![cfg(all(feature = "text-index", feature = "lpg", feature = "gql"))]

use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_engine::{GrafeoDB, TextIndexOptions, TokenizerKind};

/// The options of the notes' index in these tests: none of them the default.
fn options() -> TextIndexOptions {
    TextIndexOptions::new()
        .with_k1(0.3)
        .with_b(0.19)
        .with_tokenizer(TokenizerKind::CjkBigram)
        .with_stop_words(["住在", "и"])
}

/// Notes in Chinese and Russian: Alix lives in Berlin, Gus in Amsterdam,
/// Vincent in Berlin, Mia and Jules are in Prague.
const NOTES: [&str; 4] = [
    "INSERT (:Note {owner: 'Alix', body: '阿利克斯住在柏林'})",
    "INSERT (:Note {owner: 'Gus', body: '古斯住在阿姆斯特丹'})",
    "INSERT (:Note {owner: 'Vincent', body: 'Винсент живёт в Берлине'})",
    "INSERT (:Note {owner: 'Mia', body: 'Мия и Жюль в Праге'})",
];

/// The queries whose results show the options at work: a Chinese word
/// inside a sentence (the CJK bigram tokenizer), two stop words, a Russian
/// word in capitals, and a pair of characters in two notes.
const QUERIES: [&str; 5] = ["柏林", "住在", "и", "БЕРЛИНЕ", "斯特 克斯"];

fn insert_notes(db: &GrafeoDB) {
    for statement in NOTES {
        db.execute(statement).unwrap();
    }
}

fn owner(db: &GrafeoDB, node: NodeId) -> String {
    let found = db.get_node(node).expect("a found note exists");
    match found.properties.get(&PropertyKey::new("owner")) {
        Some(Value::String(owner)) => owner.to_string(),
        other => panic!("note {node:?} has the owner {other:?}"),
    }
}

/// What each of [`QUERIES`] finds in the notes: the owners and the bits of
/// their scores, by owner.
fn found(db: &GrafeoDB) -> Vec<(&'static str, Vec<(String, u64)>)> {
    QUERIES
        .into_iter()
        .map(|query| {
            let mut hits: Vec<(String, u64)> = db
                .text_search("Note", "body", query, 10, None)
                .unwrap()
                .into_iter()
                .map(|(node, score)| (owner(db, node), score.to_bits()))
                .collect();
            hits.sort();
            (query, hits)
        })
        .collect()
}

/// What [`found`] gives for the notes indexed with [`options`] in memory:
/// what every reopen must give back, score for score.
fn expected() -> Vec<(&'static str, Vec<(String, u64)>)> {
    let db = GrafeoDB::new_in_memory();
    insert_notes(&db);
    db.create_text_index_with("Note", "body", options())
        .unwrap();
    let found = found(&db);
    let owners = |query: &str| -> Vec<&str> {
        found
            .iter()
            .find(|(asked, _)| *asked == query)
            .map(|(_, hits)| hits.iter().map(|(owner, _)| owner.as_str()).collect())
            .unwrap_or_default()
    };
    assert_eq!(owners("柏林"), ["Alix"], "a word inside a Chinese sentence");
    assert_eq!(owners("住在"), Vec::<&str>::new(), "a stop word");
    assert_eq!(owners("и"), Vec::<&str>::new(), "a Russian stop word");
    assert_eq!(owners("БЕРЛИНЕ"), ["Vincent"], "queries are lowercased");
    assert_eq!(owners("斯特 克斯"), ["Alix", "Gus"]);
    found
}

/// Asserts that `db` holds the notes' index with [`options`], and that it
/// finds what [`expected`] says, with the same scores.
fn assert_kept(db: &GrafeoDB, after: &str) {
    assert_eq!(
        db.text_index_options("Note", "body"),
        Some(options()),
        "{after}: the index keeps its options"
    );
    assert_eq!(
        found(db),
        expected(),
        "{after}: the same results and scores"
    );
}

/// The scores of the term "berlin" in three notes for BM25 with `k1` and
/// `b`, by node: Alix in Amsterdam (3 terms), Gus in Berlin (2), Alix
/// three times in Berlin (4).
fn berlin_scores(options: Option<TextIndexOptions>) -> Vec<f64> {
    let db = GrafeoDB::new_in_memory();
    for statement in [
        "INSERT (:Note {owner: 'Alix', body: 'Alix Amsterdam Amsterdam'})",
        "INSERT (:Note {owner: 'Gus', body: 'Gus Berlin'})",
        "INSERT (:Note {owner: 'Butch', body: 'Alix Berlin Berlin Berlin'})",
    ] {
        db.execute(statement).unwrap();
    }
    match options {
        Some(options) => db.create_text_index_with("Note", "body", options).unwrap(),
        None => db.create_text_index("Note", "body").unwrap(),
    }
    let mut hits: Vec<(String, f64)> = db
        .text_search("Note", "body", "Berlin", 10, None)
        .unwrap()
        .into_iter()
        .map(|(node, score)| (owner(&db, node), score))
        .collect();
    hits.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        hits.iter()
            .map(|(owner, _)| owner.as_str())
            .collect::<Vec<_>>(),
        ["Butch", "Gus"]
    );
    hits.into_iter().map(|(_, score)| score).collect()
}

/// BM25 of "berlin": 3 notes of 3 terms on average, 2 with the term, so
/// idf = ln(1 + 1.5 / 2.5); Gus's note has it once in 2 terms, Butch's three
/// times in 4: idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / 3)).
#[test]
fn scores_follow_k1_and_b_as_bm25_says() {
    for (k1, b, butch, gus) in [
        (1.2, 0.75, 0.689_338_656_227_079, 0.544_214_728_600_325_5),
        (0.0, 0.75, 0.470_003_629_245_735_6, 0.470_003_629_245_735_6),
        (2.0, 0.0, 0.846_006_532_642_324_1, 0.470_003_629_245_735_6),
        (1.2, 1.0, 0.674_353_033_265_620_8, 0.574_448_880_189_232_6),
    ] {
        let scores = berlin_scores(Some(TextIndexOptions::new().with_k1(k1).with_b(b)));
        for (score, expected) in scores.iter().zip([butch, gus]) {
            assert!(
                (score - expected).abs() < 1e-12,
                "k1 {k1}, b {b}: {score}, BM25 says {expected}"
            );
        }
    }
}

#[test]
fn the_default_options_score_as_before() {
    let scores = berlin_scores(None);
    assert_eq!(
        scores,
        berlin_scores(Some(TextIndexOptions::new().with_k1(1.2).with_b(0.75))),
        "create_text_index is create_text_index_with the defaults"
    );
    assert!(
        (scores[0] - 0.689_338_656_227_079).abs() < 1e-12,
        "{scores:?}"
    );
    let db = GrafeoDB::new_in_memory();
    insert_notes(&db);
    db.create_text_index("Note", "body").unwrap();
    assert_eq!(
        db.text_index_options("Note", "body"),
        Some(TextIndexOptions::new())
    );
    assert_eq!(
        db.text_search("Note", "body", "柏林", 10, None).unwrap(),
        [],
        "the simple tokenizer keeps a Chinese sentence as one term"
    );
    assert_eq!(
        db.text_index_options("Note", "title"),
        None,
        "no such index"
    );
}

#[test]
fn options_out_of_range_are_refused_and_create_nothing() {
    let db = GrafeoDB::new_in_memory();
    insert_notes(&db);
    for (options, says) in [
        (TextIndexOptions::new().with_k1(-0.3), "k1"),
        (TextIndexOptions::new().with_k1(f64::NAN), "k1"),
        (TextIndexOptions::new().with_b(1.88), "b"),
    ] {
        let error = db
            .create_text_index_with("Note", "body", options)
            .unwrap_err();
        assert_eq!(error.error_code().as_str(), "GRAFEO-V001", "{error}");
        assert!(error.to_string().contains(says), "{error}");
    }
    for (statement, says) in [
        (
            "CREATE INDEX notes FOR (n:Note) ON (n.body) USING TEXT {tokenizer: 'jieba'}",
            "Unknown tokenizer 'jieba'. Use: simple, standard, cjk_bigram",
        ),
        (
            "CREATE INDEX notes FOR (n:Note) ON (n.body) USING TEXT {k1: -3}",
            "k1",
        ),
        (
            "CREATE INDEX notes FOR (n:Note) ON (n.body) USING TEXT {b: 19}",
            "b must be a number from 0 to 1",
        ),
    ] {
        let error = db.execute(statement).unwrap_err();
        assert_eq!(
            error.error_code().as_str(),
            "GRAFEO-V001",
            "{statement}: {error}"
        );
        assert!(error.to_string().contains(says), "{statement}: {error}");
    }
    assert_eq!(
        db.text_index_options("Note", "body"),
        None,
        "no index was made"
    );
    assert!(
        db.execute("SHOW INDEXES").unwrap().rows().is_empty(),
        "no index name was registered"
    );
}

#[test]
fn rebuild_text_index_keeps_the_options() {
    let db = GrafeoDB::new_in_memory();
    insert_notes(&db);
    db.create_text_index_with("Note", "body", options())
        .unwrap();
    db.rebuild_text_index("Note", "body").unwrap();
    assert_kept(&db, "rebuild_text_index");
}

#[test]
fn create_index_using_text_takes_the_options() {
    let db = GrafeoDB::new_in_memory();
    insert_notes(&db);
    db.execute(
        "CREATE INDEX notes FOR (n:Note) ON (n.body) USING TEXT \
         {k1: 0.3, b: 0.19, tokenizer: 'CJK_BIGRAM', stop_words: ['И', '住在']}",
    )
    .unwrap();
    assert_kept(&db, "CREATE INDEX ... USING TEXT {...}");
}

/// What a database file keeps: reopens, crash recovery and copies.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod persistence {
    use std::path::{Path, PathBuf};

    use grafeo_common::testing::child_process;
    use grafeo_engine::{Config, DurabilityMode, GrafeoDB};

    use super::{assert_kept, insert_notes, options};

    fn open(path: &Path) -> GrafeoDB {
        GrafeoDB::with_config(Config::persistent(path).with_wal_durability(DurabilityMode::Sync))
            .unwrap()
    }

    fn db_path(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("notes.grafeo")
    }

    /// After `close()` the file holds the index section and the catalog
    /// record: the reopen restores the index from its section. `to_memory` and
    /// `save` copy both.
    #[test]
    fn the_options_survive_close_and_reopen_and_copies() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir);
        let db = open(&path);
        insert_notes(&db);
        db.create_text_index_with("Note", "body", options())
            .unwrap();
        assert_kept(&db, "before close");
        assert_kept(&db.to_memory().unwrap(), "to_memory");
        let saved = dir.path().join("saved.grafeo");
        db.save(&saved).unwrap();
        db.close().unwrap();
        drop(db);

        let db = open(&path);
        assert_kept(&db, "close and reopen");
        db.close().unwrap();
        drop(db);
        let copy = open(&saved);
        assert_kept(&copy, "save");
        copy.close().unwrap();
    }

    /// The index made by the GQL statement survives a reopen too.
    #[test]
    fn the_options_of_create_index_survive_close_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir);
        let db = open(&path);
        insert_notes(&db);
        db.execute(
            "CREATE INDEX notes FOR (n:Note) ON (n.body) USING TEXT \
             {k1: 0.3, b: 0.19, tokenizer: 'cjk_bigram', stop_words: ['и', '住在']}",
        )
        .unwrap();
        db.close().unwrap();
        drop(db);
        let db = open(&path);
        assert_kept(&db, "CREATE INDEX, close and reopen");
        db.close().unwrap();
    }

    /// `close()` while a transaction has changes in the default graph leaves
    /// the index sections out of the file: the reopen builds the index from the
    /// data, with the options of the catalog.
    #[test]
    fn the_options_survive_a_close_with_open_changes() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir);
        let db = open(&path);
        insert_notes(&db);
        db.create_text_index_with("Note", "body", options())
            .unwrap();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute("INSERT (:Note {owner: 'Jules', body: '朱尔斯住在柏林'})")
            .unwrap();
        db.close().unwrap();
        drop(session);
        drop(db);

        let db = open(&path);
        assert_kept(&db, "close with an open transaction, reopen");
        db.close().unwrap();
    }

    const SCENARIO_VAR: &str = "GRAFEO_TEXT_OPTIONS_SCENARIO";
    const PATH_VAR: &str = "GRAFEO_TEXT_OPTIONS_PATH";

    /// Runs `scenario` in a child process that exits without closing the
    /// database (a crash: no checkpoint at close, no destructors).
    fn crash_after(scenario: &str, path: &Path) {
        let status = child_process::run(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "persistence::crash_child", "--nocapture"])
                .env(SCENARIO_VAR, scenario)
                .env(PATH_VAR, path),
        )
        .unwrap();
        assert!(status.success(), "scenario {scenario} failed");
    }

    /// Child-process entry for [`crash_after`]; a no-op when run directly.
    #[test]
    fn crash_child() {
        let Ok(scenario) = std::env::var(SCENARIO_VAR) else {
            return;
        };
        let path = PathBuf::from(std::env::var_os(PATH_VAR).unwrap());
        // Built outside any crash point: nothing unwinds into its `Drop`.
        let db = open(&path);
        insert_notes(&db);
        match scenario.as_str() {
            "created" => db
                .create_text_index_with("Note", "body", options())
                .unwrap(),
            "created_by_gql" => {
                db.execute(
                    "CREATE INDEX notes FOR (n:Note) ON (n.body) USING TEXT \
                     {k1: 0.3, b: 0.19, tokenizer: 'cjk_bigram', stop_words: ['и', '住在']}",
                )
                .unwrap();
            }
            "checkpoint_with_open_changes" => {
                db.create_text_index_with("Note", "body", options())
                    .unwrap();
                let mut session = db.session();
                session.begin_transaction().unwrap();
                session
                    .execute("INSERT (:Note {owner: 'Jules', body: '朱尔斯住在柏林'})")
                    .unwrap();
                db.wal_checkpoint().unwrap();
                // The transaction stays open: the crash loses it.
                std::mem::forget(session);
            }
            other => panic!("unknown scenario {other}"),
        }
        // Crash: no close(), no destructors.
        std::process::exit(0);
    }

    /// The index created right before a crash comes back from the WAL: the
    /// replay applies its catalog record and builds it from the data, with its
    /// options.
    #[test]
    fn the_options_survive_a_crash() {
        for scenario in ["created", "created_by_gql"] {
            let dir = tempfile::tempdir().unwrap();
            let path = db_path(&dir);
            crash_after(scenario, &path);
            let db = open(&path);
            assert_kept(&db, &format!("a crash after {scenario}"));
            db.close().unwrap();
        }
    }

    /// A checkpoint while a transaction has changes in the default graph
    /// leaves the index sections out of the image: after the crash the reopen
    /// builds the index from the data, with the options of the catalog.
    #[test]
    fn the_options_survive_a_checkpoint_with_open_changes_and_a_crash() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir);
        crash_after("checkpoint_with_open_changes", &path);
        let db = open(&path);
        assert_kept(&db, "a checkpoint with open changes, then a crash");
        db.close().unwrap();
    }
}
