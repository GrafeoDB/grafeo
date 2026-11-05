//! A rule created while a writer is open holds against what that writer wrote.
//!
//! Tests first, from the issue's done-when: a transaction's commit fails with `GRAFEO-V001` when
//! its writes break a UNIQUE or NOT NULL rule another session created since they were written,
//! also after a reopen; of two writers with a UNIQUE created between them, the first succeeds and
//! the second fails; a writer whose writes meet the new rule commits. Today all of these
//! commit, and a committed row breaks a committed rule (also on the 0.6.1 release).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main rules_against_open_writers::
//! ```

#![cfg(feature = "gql")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

const UNIQUE: &str = "CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) UNIQUE";
const NOT_NULL: &str = "CREATE CONSTRAINT c1 FOR (n:File) ON (n.must) NOT NULL";

fn with_a_file(db: &GrafeoDB) {
    db.session()
        .execute("INSERT (:File {id: 'a', must: 1})")
        .unwrap();
}

fn count(session: &Session, query: &str) -> usize {
    session.execute(query).unwrap().row_count()
}

fn files(db: &GrafeoDB) -> usize {
    count(&db.session(), "MATCH (f:File) RETURN f.id")
}

/// `(name, the rule, a write it refuses, rows that show the write stood)`.
const BROKEN: [(&str, &str, &str, &str); 2] = [
    (
        "UNIQUE",
        UNIQUE,
        "INSERT (:File {id: 'a', must: 2})",
        "MATCH (f:File) WHERE f.must = 2 RETURN f.id",
    ),
    (
        "NOT NULL",
        NOT_NULL,
        "INSERT (:File {id: 'b'})",
        "MATCH (f:File) WHERE f.id = 'b' RETURN f.id",
    ),
];

fn is_invalid_value(error: &grafeo_common::utils::error::Error) -> bool {
    error.to_string().contains("GRAFEO-V001")
}

#[test]
fn a_commit_fails_when_its_writes_break_a_new_rule() {
    for (name, rule, write, stood) in BROKEN {
        let db = GrafeoDB::new_in_memory();
        with_a_file(&db);
        let mut writer = db.session();
        writer.begin_transaction().unwrap();
        writer.execute(write).unwrap();
        db.session().execute(rule).unwrap();
        let error = writer
            .commit()
            .err()
            .unwrap_or_else(|| panic!("{name}: the commit passed"));
        assert!(is_invalid_value(&error), "{name}: {error}");
        assert_eq!(count(&db.session(), stood), 0, "{name}: the write stands");
        assert_eq!(files(&db), 1, "{name}");
    }
}

/// A failed commit rolls back the whole transaction, not only the row that broke the rule, and
/// the session goes on.
#[test]
fn a_failed_commit_undoes_all_of_its_writes() {
    for (name, rule, write, _) in BROKEN {
        let db = GrafeoDB::new_in_memory();
        with_a_file(&db);
        let mut writer = db.session();
        writer.begin_transaction().unwrap();
        writer.execute("INSERT (:Other {name: 'Vincent'})").unwrap();
        writer.execute(write).unwrap();
        writer
            .execute("MATCH (f:File {id: 'a'}) SET f.touched = true")
            .unwrap();
        db.session().execute(rule).unwrap();
        assert!(writer.commit().is_err(), "{name}");
        let reader = db.session();
        assert_eq!(count(&reader, "MATCH (o:Other) RETURN o.name"), 0, "{name}");
        assert_eq!(
            count(&reader, "MATCH (f:File) WHERE f.touched = true RETURN f.id"),
            0,
            "{name}"
        );
        writer.execute("INSERT (:Other {name: 'Mia'})").unwrap();
        assert_eq!(count(&writer, "MATCH (o:Other) RETURN o.name"), 1, "{name}");
    }
}

#[test]
fn it_holds_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    for (number, (name, rule, write, stood)) in BROKEN.into_iter().enumerate() {
        let path = dir.path().join(format!("rule-{number}.grafeo"));
        let db = GrafeoDB::open(&path).unwrap();
        with_a_file(&db);
        let mut writer = db.session();
        writer.begin_transaction().unwrap();
        writer.execute(write).unwrap();
        db.session().execute(rule).unwrap();
        let error = writer.commit().err().unwrap_or_else(|| panic!("{name}"));
        assert!(is_invalid_value(&error), "{name}: {error}");
        drop(writer);
        db.close().unwrap();

        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(count(&db.session(), stood), 0, "{name}: the write stands");
        assert_eq!(files(&db), 1, "{name}");
        // The rule is still there, and refuses the same write.
        assert!(
            db.session().execute(write).is_err(),
            "{name}: the rule went"
        );
        db.close().unwrap();
    }
}

const PATH_VAR: &str = "GRAFEO_RULES_WRITERS_PATH";
const CASE_VAR: &str = "GRAFEO_RULES_WRITERS_CASE";

/// Child-process entry: the failed commit, then the process dies with no close.
#[test]
fn crash_child() {
    let Some(path) = std::env::var_os(PATH_VAR) else {
        return;
    };
    let number: usize = std::env::var(CASE_VAR).unwrap().parse().unwrap();
    let (_, rule, write, _) = BROKEN[number];
    let db = GrafeoDB::open(std::path::PathBuf::from(path)).unwrap();
    with_a_file(&db);
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer.execute(write).unwrap();
    db.session().execute(rule).unwrap();
    assert!(writer.commit().is_err());
    std::process::exit(0);
}

/// A failed commit leaves nothing in the log: a crash right after it replays the rule alone.
#[test]
fn a_crash_after_the_failed_commit_replays_no_part_of_it() {
    let dir = tempfile::tempdir().unwrap();
    for (number, (name, _, write, stood)) in BROKEN.into_iter().enumerate() {
        let path = dir.path().join(format!("crash-{number}.grafeo"));
        let status = grafeo_common::testing::child_process::run(
            grafeo_common::child_test!("crash_child")
                .env(PATH_VAR, &path)
                .env(CASE_VAR, number.to_string()),
        )
        .unwrap();
        assert!(status.success(), "{name}: the child process failed");
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(count(&db.session(), stood), 0, "{name}: the write stands");
        assert!(
            db.session().execute(write).is_err(),
            "{name}: the rule went"
        );
        db.close().unwrap();
    }
}

/// Two writers of one value with a UNIQUE created between them: the first commit passes, as
/// nothing else holds the value, and the second fails.
#[test]
fn of_two_writers_with_a_rule_between_them_the_second_fails() {
    let db = GrafeoDB::new_in_memory();
    let mut first = db.session();
    let mut second = db.session();
    first.begin_transaction().unwrap();
    second.begin_transaction().unwrap();
    first.execute("INSERT (:File {id: 'x', must: 1})").unwrap();
    second.execute("INSERT (:File {id: 'x', must: 2})").unwrap();
    db.session().execute(UNIQUE).unwrap();
    first.commit().unwrap();
    let error = second
        .commit()
        .expect_err("two rows of one value under a UNIQUE rule");
    assert!(is_invalid_value(&error), "{error}");
    let rows = db
        .session()
        .execute("MATCH (f:File {id: 'x'}) RETURN f.must")
        .unwrap();
    assert_eq!(rows.rows(), &[vec![Value::Int64(1)]]);
}

/// A writer whose writes meet the new rule commits, whichever rule, and its writes stand.
#[test]
fn a_writer_that_meets_the_rule_commits() {
    for (name, rule, write, stood) in [
        (
            "another value",
            UNIQUE,
            "INSERT (:File {id: 'b', must: 2})",
            "MATCH (f:File {id: 'b'}) RETURN f.id",
        ),
        (
            "the required property",
            NOT_NULL,
            "INSERT (:File {id: 'b', must: 2})",
            "MATCH (f:File {id: 'b'}) RETURN f.id",
        ),
        (
            "another label",
            UNIQUE,
            "INSERT (:Other {id: 'a'})",
            "MATCH (o:Other {id: 'a'}) RETURN o.id",
        ),
        (
            "another label, no property",
            NOT_NULL,
            "INSERT (:Other {id: 'a'})",
            "MATCH (o:Other {id: 'a'}) RETURN o.id",
        ),
        (
            "no property of the unique rule",
            UNIQUE,
            "INSERT (:File {must: 3})",
            "MATCH (f:File {must: 3}) RETURN f.must",
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        with_a_file(&db);
        let mut writer = db.session();
        writer.begin_transaction().unwrap();
        writer.execute(write).unwrap();
        db.session().execute(rule).unwrap();
        writer
            .commit()
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(count(&db.session(), stood), 1, "{name}");
    }
}

/// A transaction that only reads, or wrote elsewhere, is not disturbed.
#[test]
fn a_reader_commits() {
    let db = GrafeoDB::new_in_memory();
    with_a_file(&db);
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    assert_eq!(count(&reader, "MATCH (f:File) RETURN f.id"), 1);
    db.session().execute(UNIQUE).unwrap();
    reader.commit().unwrap();
}

/// Writes other than an insert break a rule too: a `SET` to a taken value, a `REMOVE` of the
/// required property, a label added to a node that does not meet it.
#[test]
fn every_kind_of_write_is_checked() {
    for (name, setup, write, rule) in [
        (
            "SET to a taken value",
            "INSERT (:File {id: 'b', must: 2})",
            "MATCH (f:File {id: 'b'}) SET f.id = 'a'",
            UNIQUE,
        ),
        (
            "REMOVE of the required property",
            "INSERT (:File {id: 'b', must: 2})",
            "MATCH (f:File {id: 'b'}) REMOVE f.must",
            NOT_NULL,
        ),
        (
            "a label on a node with a taken value",
            "INSERT (:Draft {id: 'a'})",
            "MATCH (d:Draft) SET d:File",
            UNIQUE,
        ),
        (
            "a label on a node without the property",
            "INSERT (:Draft {id: 'c'})",
            "MATCH (d:Draft) SET d:File",
            NOT_NULL,
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        with_a_file(&db);
        db.session().execute(setup).unwrap();
        let mut writer = db.session();
        writer.begin_transaction().unwrap();
        writer.execute(write).unwrap();
        db.session().execute(rule).unwrap();
        let error = writer
            .commit()
            .err()
            .unwrap_or_else(|| panic!("{name}: the commit passed"));
        assert!(is_invalid_value(&error), "{name}: {error}");
    }
}

/// What the transaction's own writes clear does not count against it: the old holder of the value
/// is deleted or changed in the same transaction, so the end state meets the rule.
#[test]
fn the_end_state_of_the_transaction_decides() {
    for (name, writes) in [
        (
            "deleted the holder",
            vec![
                "MATCH (f:File {id: 'a'}) DELETE f",
                "INSERT (:File {id: 'a', must: 2})",
            ],
        ),
        (
            "moved the holder",
            vec![
                "MATCH (f:File {id: 'a'}) SET f.id = 'old'",
                "INSERT (:File {id: 'a', must: 2})",
            ],
        ),
        (
            "wrote and took back",
            vec![
                "INSERT (:File {id: 'a', must: 2})",
                "MATCH (f:File {id: 'a', must: 2}) DELETE f",
            ],
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        with_a_file(&db);
        let mut writer = db.session();
        writer.begin_transaction().unwrap();
        for write in writes {
            writer.execute(write).unwrap();
        }
        db.session().execute(UNIQUE).unwrap();
        writer
            .commit()
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let holders = count(&db.session(), "MATCH (f:File {id: 'a'}) RETURN f.must");
        assert_eq!(holders, 1, "{name}");
    }
}

/// A rule dropped before the commit no longer counts: only rules that stand at the commit.
#[test]
fn a_rule_gone_by_the_commit_does_not_count() {
    let db = GrafeoDB::new_in_memory();
    with_a_file(&db);
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer.execute("INSERT (:File {id: 'a', must: 2})").unwrap();
    db.session().execute(UNIQUE).unwrap();
    db.session().execute("DROP CONSTRAINT c1").unwrap();
    writer.commit().unwrap();
    assert_eq!(files(&db), 2);
}

/// A savepoint rollback takes back the breaking write, and the commit passes.
#[test]
fn a_write_taken_back_at_a_savepoint_does_not_count() {
    let db = GrafeoDB::new_in_memory();
    with_a_file(&db);
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer.execute("INSERT (:Other {name: 'Mia'})").unwrap();
    writer.savepoint("s").unwrap();
    writer.execute("INSERT (:File {id: 'a', must: 2})").unwrap();
    db.session().execute(UNIQUE).unwrap();
    writer.rollback_to_savepoint("s").unwrap();
    writer.commit().unwrap();
    assert_eq!(files(&db), 1);
    assert_eq!(count(&db.session(), "MATCH (o:Other) RETURN o.name"), 1);
}
