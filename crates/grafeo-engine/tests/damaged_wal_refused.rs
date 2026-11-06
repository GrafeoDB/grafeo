//! Class test 1, the WAL (class-tests design, 2026-11-05): damaged data is refused with a named
//! error, never a panic and never a silent default. #425 ruled what that means for a WAL: damage
//! before its end fails the open, naming the segment, and keeps the WAL; a salvaging open replays
//! the commits before the damage; a torn tail is not damage. The WAL's reader adds what a tail is:
//! bytes that were never durable, a partial group or "an unsynced hole before unsynced groups".
//! A group is durable once a later group's prologue records a synced position past it.
//!
//! A child writes one transaction per kind of record the WAL holds (`LogRecord`: each `DataOp`,
//! each `StandaloneOp`, each catalog record put and dropped, each RDF graph operation, as the
//! catalogues in `tests/catalogue/` list them) over two segments, and crashes; once syncing every
//! commit and once with the default durability. Every frame of every group is then damaged in
//! turn (a byte of its header and a byte of its payload; with the default durability only the
//! tail that was never durable and the frame before it, as syncing checks the rest), and the
//! database opened three ways:
//!
//! - **read-write and read-only:** damage in a durable frame fails the open with
//!   `Error::Corruption` naming its segment, and leaves the WAL as it was; damage in the tail
//!   that was never durable is cut, so the open reads the state of the WAL cut where the damaged
//!   group starts.
//! - **salvaging:** it reads that state too, wherever the damage is.
//!
//! A state is read whole: every node and edge of both graphs, every `SHOW` of the catalog, the
//! triples. Never a panic.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main damaged_wal_refused::
//! ```

#![cfg(all(
    feature = "gql",
    feature = "sparql",
    feature = "vector-index",
    feature = "wal",
    feature = "grafeo-file",
    not(miri)
))]

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use grafeo_common::storage::{LogRecord, read_log_records};
use grafeo_common::testing::child_process;
use grafeo_common::utils::error::{Error, Result};
use grafeo_engine::{Config, DurabilityMode, GrafeoDB};
use grafeo_storage::wal::{FRAME_HEADER_BYTES, FRAME_PROLOGUE_BYTES, SEGMENT_HEADER_BYTES};

const DATA_OPS: &str = include_str!("../../../tests/catalogue/data_op.txt");
const STANDALONE_OPS: &str = include_str!("../../../tests/catalogue/standalone_op.txt");
const RDF_GRAPH_OPS: &str = include_str!("../../../tests/catalogue/rdf_graph_op.txt");
const CATALOG_RECORDS: &str = include_str!("../../../tests/catalogue/catalog_record.txt");
const PATH_VAR: &str = "GRAFEO_DAMAGED_WAL_PATH";
/// Set for a child that syncs every commit.
const SYNC_VAR: &str = "GRAFEO_DAMAGED_WAL_SYNC";

fn names(file: &'static str) -> Vec<&'static str> {
    file.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect()
}

/// What the child writes, a transaction each, before and after it rotates the WAL.
const FIRST_SEGMENT: &[&str] = &[
    "INSERT (:Person {name: 'Alix', age: 30})-[:KNOWS {since: 2001}]->(:Person {name: 'Gus'})",
    "MATCH (n:Person {name: 'Gus'}) SET n.age = 40",
    "MATCH ()-[e:KNOWS]->() SET e.weight = 3",
    "MATCH (n:Person {name: 'Alix'}) REMOVE n.age",
    "MATCH ()-[e:KNOWS]->() REMOVE e.since",
    "MATCH (n:Person {name: 'Gus'}) SET n:Pilot",
    "MATCH (n:Person {name: 'Gus'}) REMOVE n:Pilot",
    "CREATE NODE TYPE City (name STRING NOT NULL)",
    "CREATE EDGE TYPE ROUTE CONNECTING (City) TO (City) (km INT64)",
    "CREATE GRAPH TYPE travel (NODE TYPE City, EDGE TYPE ROUTE)",
    "ALTER NODE TYPE City ADD PROPERTY country STRING",
    "ALTER EDGE TYPE ROUTE ADD PROPERTY note STRING",
    "ALTER GRAPH TYPE travel ADD NODE TYPE Harbour",
    "CREATE INDEX person_name FOR (p:Person) ON (p.name)",
    "INSERT (:Doc {body: 'graph database', emb: vector([1.0, 0.0])})",
    "CREATE VECTOR INDEX doc_emb ON :Doc(emb)",
    "CREATE CONSTRAINT person_email FOR (p:Person) ON (p.email) UNIQUE",
    "CREATE SCHEMA archive",
    "CREATE PROCEDURE people() RETURNS (name STRING) AS { MATCH (p:Person) RETURN p.name AS name }",
    "CREATE GRAPH trips TYPED travel",
];

const SECOND_SEGMENT: &[&str] = &[
    "USE GRAPH trips INSERT (:City {name: 'Prague'})",
    "CREATE GRAPH spare",
    "DROP GRAPH spare",
    "DROP PROCEDURE people",
    "DROP SCHEMA archive",
    "DROP CONSTRAINT person_email",
    "DROP INDEX person_name",
    "DROP INDEX doc_emb",
    "CREATE NODE TYPE Spare (x INT64)",
    "DROP NODE TYPE Spare",
    "CREATE EDGE TYPE SPARE_EDGE CONNECTING (City) TO (City)",
    "DROP EDGE TYPE SPARE_EDGE",
    "CREATE GRAPH TYPE spare_type (NODE TYPE City)",
    "DROP GRAPH TYPE spare_type",
    "MATCH ()-[e:KNOWS]->() DELETE e",
    "MATCH (n:Person {name: 'Gus'}) DELETE n",
];

const SPARQL: &[&str] = &[
    "INSERT DATA { <urn:alix> <urn:knows> <urn:gus> . <urn:gus> <urn:age> 40 }",
    "DELETE DATA { <urn:gus> <urn:age> 40 }",
    "CREATE GRAPH <urn:g>",
    "INSERT DATA { GRAPH <urn:g> { <urn:a> <urn:b> <urn:c> } }",
    "COPY <urn:g> TO <urn:h>",
    "ADD <urn:g> TO <urn:h>",
    "MOVE <urn:h> TO <urn:i>",
    "CLEAR GRAPH <urn:g>",
    "DROP GRAPH <urn:g>",
];

/// The child: writes the transactions over two segments, rolls one back, and crashes.
#[test]
fn wal_writer_child() {
    let Some(path) = std::env::var_os(PATH_VAR) else {
        return;
    };
    let config = Config::persistent(Path::new(&path));
    let config = if std::env::var_os(SYNC_VAR).is_some() {
        config.with_wal_durability(DurabilityMode::Sync)
    } else {
        config
    };
    let db = GrafeoDB::with_config(config).unwrap();
    db.wal_checkpoint().unwrap();
    for statement in FIRST_SEGMENT {
        db.execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    db.wal().unwrap().rotate().unwrap();
    for statement in SECOND_SEGMENT {
        db.execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    for statement in SPARQL {
        db.execute_sparql(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Ghost'})").unwrap();
    session.rollback().unwrap();
    db.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    // A crash: no close, no checkpoint, no destructors.
    std::process::exit(0);
}

fn sidecar(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.wal", path.display()))
}

/// The segment files, in log order.
fn segments(path: &Path) -> Vec<PathBuf> {
    let mut segments: Vec<PathBuf> = std::fs::read_dir(sidecar(path))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|file| {
            file.file_name()
                .and_then(|name| name.to_str())
                .and_then(grafeo_storage::wal::parse_segment_file_name)
                .is_some()
        })
        .collect();
    segments.sort();
    segments
}

/// Every file under `dir`, relative, with its bytes.
fn files_in(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(at) = pending.pop() {
        for entry in std::fs::read_dir(&at).unwrap() {
            let entry = entry.unwrap().path();
            if entry.is_dir() {
                pending.push(entry);
            } else {
                let bytes = std::fs::read(&entry).unwrap();
                files.push((entry.strip_prefix(dir).unwrap().to_path_buf(), bytes));
            }
        }
    }
    files.sort();
    files
}

/// Writes `files` into `dir`, emptied first.
fn restore(dir: &Path, files: &[(PathBuf, Vec<u8>)]) {
    if dir.exists() {
        std::fs::remove_dir_all(dir).unwrap();
    }
    for (relative, bytes) in files {
        let file = dir.join(relative);
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(file, bytes).unwrap();
    }
}

/// One frame: its segment, where it starts, its log position, its payload's length, and its
/// group.
#[derive(Clone, Copy, Debug)]
struct Frame {
    segment: usize,
    offset: usize,
    lsn: u64,
    payload: usize,
    group: usize,
}

/// The frames of the segments, the kinds of record each group holds, and the synced position each
/// group's prologue records.
fn frames(segment_files: &[PathBuf]) -> (Vec<Frame>, Vec<Vec<String>>, Vec<u64>) {
    let mut frames = Vec::new();
    let mut groups: Vec<Vec<String>> = Vec::new();
    let mut synced = Vec::new();
    for (segment, file) in segment_files.iter().enumerate() {
        let bytes = std::fs::read(file).unwrap();
        let mut at = SEGMENT_HEADER_BYTES;
        while at + FRAME_HEADER_BYTES <= bytes.len() {
            let length =
                usize::try_from(u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())).unwrap();
            let flags = bytes[at + 24];
            if length == 0 && flags == 0 {
                break;
            }
            let payload = &bytes[at + FRAME_HEADER_BYTES..at + FRAME_HEADER_BYTES + length];
            if flags & 1 != 0 {
                groups.push(Vec::new());
                synced.push(u64::from_le_bytes(payload[..8].try_into().unwrap()));
            }
            frames.push(Frame {
                segment,
                offset: at,
                lsn: u64::from_le_bytes(bytes[at + 8..at + 16].try_into().unwrap()),
                payload: length,
                group: groups.len() - 1,
            });
            // A frame's payload is a run of whole records, after the prologue in a group's first.
            let records = if flags & 1 != 0 {
                &payload[FRAME_PROLOGUE_BYTES..]
            } else {
                payload
            };
            let kinds = groups.last_mut().unwrap();
            read_log_records(records, &mut |record| {
                kinds.push(kind_of(&record));
                Ok(())
            })
            .unwrap();
            at += FRAME_HEADER_BYTES + length;
        }
    }
    (frames, groups, synced)
}

/// A record's kind as the catalogues name it: `Data/CreateNode`, `Standalone/CreateGraph`,
/// `PutCatalog/NodeType`, `DropCatalog/Index`, `RdfGraph/Copy`, or `GroupBegin`.
fn kind_of(record: &LogRecord) -> String {
    let shown = |debug: String, words: usize| {
        debug
            .split(|c: char| !c.is_alphanumeric())
            .filter(|word| !word.is_empty())
            .take(words)
            .collect::<Vec<_>>()
            .join("/")
    };
    match record {
        LogRecord::GroupBegin(_) => "GroupBegin".to_string(),
        LogRecord::Data { op, .. } => format!("Data/{}", shown(format!("{op:?}"), 1)),
        LogRecord::Standalone(op) => {
            let debug = format!("{op:?}");
            let nested = ["PutCatalog", "DropCatalog", "RdfGraph"]
                .iter()
                .any(|outer| debug.starts_with(outer));
            if nested {
                shown(debug, 2)
            } else {
                format!("Standalone/{}", shown(debug, 1))
            }
        }
    }
}

/// Every kind the catalogues name, as [`kind_of`] writes it.
fn catalogued() -> Vec<String> {
    let mut kinds = Vec::new();
    kinds.extend(names(DATA_OPS).iter().map(|op| format!("Data/{op}")));
    kinds.extend(
        names(STANDALONE_OPS)
            .iter()
            .filter(|op| !["PutCatalog", "DropCatalog", "RdfGraph"].contains(op))
            .map(|op| format!("Standalone/{op}")),
    );
    for record in names(CATALOG_RECORDS) {
        kinds.push(format!("PutCatalog/{record}"));
        kinds.push(format!("DropCatalog/{record}"));
    }
    kinds.extend(
        names(RDF_GRAPH_OPS)
            .iter()
            .map(|op| format!("RdfGraph/{op}")),
    );
    kinds
}

/// The rows of `result`, each written out, sorted: a `SHOW` lists in no fixed order.
fn sorted(result: grafeo_engine::database::QueryResult) -> String {
    let mut rows: Vec<String> = result.rows().iter().map(|row| format!("{row:?}")).collect();
    rows.sort();
    rows.join("; ")
}

/// The state a reader sees, written out.
fn read_whole(db: &GrafeoDB) -> Result<Vec<String>> {
    let mut read = Vec::new();
    for query in [
        "MATCH (n) RETURN labels(n), properties(n)",
        "MATCH (a)-[e]->(b) RETURN a.name, type(e), properties(e), b.name",
        "SHOW NODE TYPES",
        "SHOW EDGE TYPES",
        "SHOW GRAPH TYPES",
        "SHOW CONSTRAINTS",
        "SHOW INDEXES",
        "SHOW GRAPHS",
        "SHOW SCHEMAS",
    ] {
        read.push(sorted(db.execute(query)?));
    }
    if sorted(db.execute("SHOW GRAPHS")?).contains("trips") {
        read.push(sorted(db.execute(
            "USE GRAPH trips MATCH (n) RETURN labels(n), properties(n)",
        )?));
    }
    read.push(sorted(
        db.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")?,
    ));
    Ok(read)
}

/// How an open went.
enum Opened {
    Read(Vec<String>),
    Refused(PathBuf),
    Failed(Error),
    Panicked(String),
}

impl Opened {
    fn show(&self) -> String {
        match self {
            Opened::Read(read) => format!("opened, read {} parts", read.len()),
            Opened::Refused(file) => format!("refused as a corruption of {}", file.display()),
            Opened::Failed(error) => format!("failed: {error:?}"),
            Opened::Panicked(message) => format!("panicked: {message}"),
        }
    }
}

fn open(path: &Path, how: &str) -> Opened {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let db = match how {
            "read-only" => GrafeoDB::open_read_only(path)?,
            "salvaging" => GrafeoDB::with_config(Config::persistent(path).with_wal_salvage())?,
            _ => GrafeoDB::open(path)?,
        };
        read_whole(&db)
    }));
    match outcome {
        Ok(Ok(read)) => Opened::Read(read),
        Ok(Err(Error::Corruption(corruption))) if corruption.file.is_some() => {
            Opened::Refused(corruption.file.unwrap())
        }
        Ok(Err(error)) => Opened::Failed(error),
        Err(panic) => Opened::Panicked(
            panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_default(),
        ),
    }
}

/// The crashed database: its directory's files, the frames and the record kinds of each group.
struct Crashed {
    files: Vec<(PathBuf, Vec<u8>)>,
    segments: Vec<PathBuf>,
    frames: Vec<Frame>,
    groups: Vec<Vec<String>>,
    synced: Vec<u64>,
}

impl Crashed {
    /// Whether `frame` was durable: a later group records a synced position past it.
    fn durable(&self, frame: &Frame) -> bool {
        self.synced
            .iter()
            .enumerate()
            .any(|(group, synced)| group > frame.group && *synced > frame.lsn)
    }
}

/// The database a child left, syncing every commit when `sync`.
fn crashed(dir: &Path, sync: bool) -> Crashed {
    let path = dir.join("db.grafeo");
    std::fs::create_dir_all(dir).unwrap();
    GrafeoDB::open(&path).unwrap().close().unwrap();
    let mut child = grafeo_common::child_test!("wal_writer_child");
    child.env(PATH_VAR, &path);
    if sync {
        child.env(SYNC_VAR, "1");
    }
    let status = child_process::run(&mut child).unwrap();
    assert!(status.success(), "the child failed");
    let segments = segments(&path);
    assert_eq!(segments.len(), 2, "{segments:?}");
    let (frames, groups, synced) = frames(&segments);
    Crashed {
        files: files_in(dir),
        segments,
        frames,
        groups,
        synced,
    }
}

/// Kinds of record the child cannot write, with the reason.
const NOT_WRITTEN: &[(&str, &str)] = &[
    (
        "DropCatalog/GraphBinding",
        "no statement logs it: a typed graph's binding goes with the graph's own DropGraph",
    ),
    (
        "PutCatalog/StoredConfig",
        "the stored settings go into the catalog section at a checkpoint, never into the WAL",
    ),
    (
        "DropCatalog/StoredConfig",
        "the stored settings go into the catalog section at a checkpoint, never into the WAL",
    ),
];

#[test]
fn every_record_of_the_catalogue_is_in_the_wal() {
    let dir = tempfile::tempdir().unwrap();
    let crashed = crashed(dir.path(), false);
    let written: Vec<String> = crashed.groups.into_iter().flatten().collect();
    let missing: Vec<String> = catalogued()
        .into_iter()
        .filter(|name| !written.contains(name))
        .filter(|name| !NOT_WRITTEN.iter().any(|(n, _)| n == name))
        .collect();
    assert!(
        missing.is_empty(),
        "kinds the WAL does not hold: {missing:?}; write them in the child, or list them in \
         NOT_WRITTEN with the reason"
    );
}

#[test]
fn a_damaged_wal_is_refused_or_salvaged_up_to_the_damage() {
    let mut wrong = Vec::new();
    let mut checked = 0;
    for sync in [true, false] {
        damage_every_frame(sync, &mut wrong, &mut checked);
    }
    assert!(checked > 200, "{checked} checked");
    assert!(
        wrong.is_empty(),
        "{} of {checked} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// The newest segment emptied (blank, or its header torn) where the previous segment's frames end
/// exactly at its name is damage, not a rotation leftover: since 0.6.2 a writer never leaves a
/// blank segment name (a leftover is ignored only when its name lies strictly inside the previous
/// segment's frames, #725). Read-write and read-only refuse it, naming the segment and
/// `wal_salvage`, and keep it; a salvaging open then opens with exactly the commits before it.
#[test]
fn an_emptied_segment_where_the_previous_one_ends_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let crashed = crashed(&dir.path().join("pristine"), true);
    let work = dir.path().join("work");
    let path = work.join("db.grafeo");
    let second = crashed.segments[1].file_name().unwrap();
    // The commits before the second segment: the state with it gone.
    restore(&work, &crashed.files);
    std::fs::remove_file(work.join("db.grafeo.wal").join(second)).unwrap();
    let before = match open(&path, "read-write") {
        Opened::Read(read) => read,
        other => panic!("the WAL without its second segment: {}", other.show()),
    };
    for (emptied, keep) in [("blank", 0), ("a torn header", 12)] {
        let empty = |segment: &Path| {
            let mut bytes = std::fs::read(segment).unwrap();
            bytes.truncate(keep);
            std::fs::write(segment, &bytes).unwrap();
            bytes
        };
        for how in ["read-write", "read-only"] {
            restore(&work, &crashed.files);
            let segment = work.join("db.grafeo.wal").join(second);
            let bytes = empty(&segment);
            let files = files_in(&work);
            match open(&path, how) {
                Opened::Refused(file) => assert!(
                    file.ends_with(second),
                    "{emptied}, {how}: refused naming {}",
                    file.display()
                ),
                other => panic!("{emptied}, {how}: {}", other.show()),
            }
            assert_eq!(
                std::fs::read(&segment).unwrap(),
                bytes,
                "{emptied}, {how}: the segment is kept"
            );
            if how == "read-only" {
                assert!(
                    files_in(&work) == files,
                    "{emptied}: a read-only refusal changes nothing"
                );
            }
        }
        restore(&work, &crashed.files);
        empty(&work.join("db.grafeo.wal").join(second));
        let message = match GrafeoDB::open(&path) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("{emptied}: opened"),
        };
        assert!(
            message.contains("wal_salvage"),
            "{emptied}: names wal_salvage: {message}"
        );
        match open(&path, "salvaging") {
            Opened::Read(read) => assert_eq!(
                read, before,
                "{emptied}: salvage keeps every commit before the emptied segment"
            ),
            other => panic!("{emptied}, salvaging: {}", other.show()),
        }
        match open(&path, "read-write") {
            Opened::Read(read) => assert_eq!(read, before, "{emptied}: an open after the salvage"),
            other => panic!("{emptied}: an open after the salvage: {}", other.show()),
        }
    }
}

/// A database whose only segment is emptied (blank, or its header torn) is refused too: no writer
/// since 0.6.2 leaves one, whatever comes before it (#725). Read-write and read-only refuse it,
/// naming it, and the read-only open changes nothing.
#[test]
fn an_emptied_only_segment_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let crashed = crashed(&dir.path().join("pristine"), true);
    let work = dir.path().join("work");
    let path = work.join("db.grafeo");
    let (first, second) = (
        crashed.segments[0].file_name().unwrap(),
        crashed.segments[1].file_name().unwrap(),
    );
    for (emptied, keep) in [("blank", 0), ("a torn header", 12)] {
        for how in ["read-write", "read-only"] {
            restore(&work, &crashed.files);
            let wal = work.join("db.grafeo.wal");
            std::fs::remove_file(wal.join(second)).unwrap();
            let mut bytes = std::fs::read(wal.join(first)).unwrap();
            bytes.truncate(keep);
            std::fs::write(wal.join(first), &bytes).unwrap();
            let files = files_in(&work);
            match open(&path, how) {
                Opened::Refused(file) => assert!(
                    file.ends_with(first),
                    "{emptied}, {how}: refused naming {}",
                    file.display()
                ),
                other => panic!("{emptied}, {how}: {}", other.show()),
            }
            if how == "read-only" {
                assert!(
                    files_in(&work) == files,
                    "{emptied}: a read-only refusal changes nothing"
                );
            }
        }
    }
}

/// Damages every frame of a crashed WAL in turn, with `sync` as the child's durability.
fn damage_every_frame(sync: bool, wrong: &mut Vec<String>, checked: &mut usize) {
    let dir = tempfile::tempdir().unwrap();
    let crashed = crashed(&dir.path().join("pristine"), sync);
    if sync {
        let tail: Vec<&Frame> = crashed
            .frames
            .iter()
            .filter(|frame| !crashed.durable(frame))
            .collect();
        assert!(
            tail.iter()
                .all(|frame| frame.group == crashed.groups.len() - 1),
            "syncing every commit, every group but the last is durable: {tail:?}"
        );
    }
    let work = dir.path().join("work");
    let path = work.join("db.grafeo");
    let last_group = crashed.groups.len() - 1;
    // The state of the WAL cut where each group starts.
    let mut cut_at = Vec::new();
    for group in 0..=last_group {
        let first = crashed
            .frames
            .iter()
            .find(|frame| frame.group == group)
            .unwrap();
        restore(&work, &crashed.files);
        let segment = work
            .join("db.grafeo.wal")
            .join(crashed.segments[first.segment].file_name().unwrap());
        let mut bytes = std::fs::read(&segment).unwrap();
        bytes.truncate(first.offset);
        std::fs::write(&segment, bytes).unwrap();
        for later in &crashed.segments[first.segment + 1..] {
            std::fs::remove_file(work.join("db.grafeo.wal").join(later.file_name().unwrap()))
                .unwrap();
        }
        match open(&path, "read-write") {
            Opened::Read(read) => cut_at.push(read),
            other => panic!("the WAL cut before group {group}: {}", other.show()),
        }
    }
    // Syncing every commit checks every durable frame; the default durability adds the tail that
    // was never durable, and the last durable frame before it.
    let boundary = crashed
        .frames
        .iter()
        .rposition(|frame| crashed.durable(frame));
    for (index, frame) in crashed.frames.iter().enumerate() {
        let durable = crashed.durable(frame);
        if !sync && durable && Some(index) != boundary {
            continue;
        }
        let segment_name = crashed.segments[frame.segment].file_name().unwrap();
        for (part, at) in [
            ("header", frame.offset + 10),
            (
                "payload",
                frame.offset + FRAME_HEADER_BYTES + frame.payload / 2,
            ),
        ] {
            let what = format!(
                "{}, group {} ({:?}){}, frame at {} of segment {}, its {part}",
                if sync {
                    "syncing every commit"
                } else {
                    "default durability"
                },
                frame.group,
                crashed.groups[frame.group],
                if durable {
                    ", durable"
                } else {
                    ", not yet durable"
                },
                frame.offset,
                frame.segment + 1
            );
            for how in ["read-write", "read-only", "salvaging"] {
                restore(&work, &crashed.files);
                let segment = work.join("db.grafeo.wal").join(segment_name);
                let mut bytes = std::fs::read(&segment).unwrap();
                bytes[at] ^= 0x5A;
                std::fs::write(&segment, &bytes).unwrap();
                let wal_before = files_in(&work.join("db.grafeo.wal"));
                let opened = open(&path, how);
                *checked += 1;
                let holds = match (&opened, how) {
                    (Opened::Read(read), "salvaging") => *read == cut_at[frame.group],
                    (Opened::Read(read), _) => !durable && *read == cut_at[frame.group],
                    (Opened::Refused(file), "read-write" | "read-only") => {
                        durable
                            && file.file_name() == Some(segment_name)
                            && files_in(&work.join("db.grafeo.wal")) == wal_before
                    }
                    _ => false,
                };
                if !holds {
                    // Which state it opened to: the WAL cut before which group, if any.
                    let state = match &opened {
                        Opened::Read(read) => {
                            cut_at.iter().position(|cut| cut == read).map_or_else(
                                || " (no cut state: something else)".to_string(),
                                |group| format!(" (the state of the WAL cut before group {group})"),
                            )
                        }
                        _ => String::new(),
                    };
                    wrong.push(format!("{what}, {how}: {}{state}", opened.show()));
                }
            }
        }
    }
}
