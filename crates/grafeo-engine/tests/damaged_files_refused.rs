//! Class test 1, the database file (class-tests design, 2026-11-05): damaged data is refused with
//! a named error, never a panic and never a silent default.
//!
//! The files hold every section of the catalogue (`tests/catalogue/section_type.txt`): three this
//! build writes (an LPG database with indexes and history, one compacted and then changed, an RDF
//! database with its ring) and released files for the sections only older releases wrote (the
//! compacted store, the search index sections). Every chunk of every section is damaged in turn,
//! and each damaged copy is opened read-only and read-write and read whole (every node, edge and
//! triple, the index lookups, the history):
//!
//! - **On disk**, which the chunk's checksum sees: a byte flipped at the chunk's first, middle and
//!   last byte, its first page zeroed, the file cut where it starts. The open, or a read after it,
//!   must fail with `Error::Corruption` naming the file, or read exactly what the undamaged file
//!   reads (a reader that rebuilds a derived section loses nothing).
//! - **Behind a valid checksum** (the chunk written again, damaged, by a checkpoint): emptied, cut
//!   to half, given trailing bytes, a byte flipped, zeroed. Such a chunk can be a valid encoding
//!   (no records, or another value), so a reader cannot always tell; the rule is no panic and no
//!   internal error. An error that names the section (`Serialization`, "written by another
//!   version") is a refusal: the reader cannot tell damage from a newer release's bytes.
//!
//! Every damage of every chunk runs, in CI too: the whole sweep takes seconds.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main damaged_files_refused::
//! ```

#![cfg(all(
    feature = "grafeo-file",
    feature = "gql",
    feature = "sparql",
    feature = "ring-index",
    feature = "vector-index",
    feature = "text-index"
))]

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use grafeo_common::storage::{ChunkMeta, Section, SectionSink, SectionSource, SectionType};
use grafeo_common::types::EpochId;
use grafeo_common::utils::error::{Error, Result};
use grafeo_engine::{Config, GrafeoDB, GraphModel, HistoryRetention};
use grafeo_storage::file::legacy::{LegacyContents, LegacyFile};
use grafeo_storage::file::v3::ImageReader;
use grafeo_storage::file::{CheckpointHeader, GrafeoFileManager};

const SECTIONS: &str = include_str!("../../../tests/catalogue/section_type.txt");

fn names(file: &'static str) -> Vec<&'static str> {
    file.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect()
}

/// The files the test damages.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// Data, a property index, a vector and a text index, and history.
    Lpg,
    /// Compacted, then a node deleted and one added.
    Compacted,
    /// Triples, with the ring index.
    Rdf,
    /// A released LPG file of `tests/fixtures`.
    ReleasedLpg(&'static str),
}

const KINDS: [Kind; 5] = [
    Kind::Lpg,
    Kind::Compacted,
    Kind::Rdf,
    // The compacted store and its deletions, which 0.5 wrote.
    Kind::ReleasedLpg("released/0.5.44/compacted.grafeo"),
    // The vector and text index sections, which 0.6.0 development builds wrote.
    Kind::ReleasedLpg("search-indexes/0.6.0-dev/documents.grafeo"),
];

fn config(kind: Kind, path: &Path) -> Config {
    let config = Config::persistent(path);
    match kind {
        Kind::Lpg => config.with_history_retention(HistoryRetention::Forever),
        Kind::Compacted | Kind::ReleasedLpg(_) => config,
        Kind::Rdf => config.with_graph_model(GraphModel::Rdf),
    }
}

/// Builds the file of `kind` at `path`, and returns the epoch whose view the history read uses.
fn build(kind: Kind, path: &Path) -> EpochId {
    if let Kind::ReleasedLpg(fixture) = kind {
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(fixture);
        std::fs::copy(source, path).unwrap();
        return EpochId::new(0);
    }
    let mut db = GrafeoDB::with_config(config(kind, path)).unwrap();
    let mut early = EpochId::new(0);
    match kind {
        Kind::Lpg => {
            db.execute(
                "INSERT (:Person {name: 'Vincent', age: 40})-[:KNOWS {since: 1994}]->\
                 (:Person {name: 'Jules', age: 38}), \
                 (:Doc {title: 'graphs', body: 'graph database engine', emb: vector([1.0, 0.0])}), \
                 (:Doc {title: 'web', body: 'python web framework', emb: vector([0.0, 1.0])})",
            )
            .unwrap();
            db.create_property_index("name").unwrap();
            db.execute("CREATE VECTOR INDEX doc_emb ON :Doc(emb)")
                .unwrap();
            db.execute("CREATE INDEX doc_body FOR (n:Doc) ON (n.body) USING TEXT")
                .unwrap();
            early = db.current_epoch();
            db.execute("MATCH (n:Person {name: 'Jules'}) SET n.age = 39")
                .unwrap();
        }
        Kind::Compacted => {
            db.execute(
                "INSERT (:Person {name: 'Mia', age: 26})-[:KNOWS]->(:Person {name: 'Butch', age: 33}), \
                 (:Person {name: 'Marcus', age: 50})",
            )
            .unwrap();
            db.compact().unwrap();
            db.execute("MATCH (n:Person {name: 'Marcus'}) DETACH DELETE n")
                .unwrap();
            db.execute("INSERT (:Person {name: 'Django', age: 30})")
                .unwrap();
        }
        Kind::Rdf => {
            db.execute_sparql(
                "INSERT DATA { <urn:vincent> <urn:knows> <urn:jules> . \
                 <urn:vincent> <urn:name> \"Vincent\" . <urn:jules> <urn:age> 38 }",
            )
            .unwrap();
            db.rdf_store().rebuild_ring();
        }
        Kind::ReleasedLpg(_) => unreachable!("copied above"),
    }
    db.close().unwrap();
    early
}

/// Everything a reader of `kind` can read, written out: a damaged file that opens must read it
/// all the same, or fail with a corruption.
fn read_whole(db: &GrafeoDB, kind: Kind, early: EpochId) -> Result<Vec<String>> {
    let rows = |result: grafeo_engine::database::QueryResult| format!("{:?}", result.rows());
    let mut read = Vec::new();
    match kind {
        Kind::Lpg | Kind::Compacted | Kind::ReleasedLpg(_) => {
            for query in [
                "MATCH (n) RETURN labels(n), properties(n) ORDER BY id(n)",
                "MATCH (a)-[e]->(b) RETURN id(a), type(e), properties(e), id(b) ORDER BY id(e)",
                "MATCH (n:Person) WHERE n.name = 'Jules' RETURN n.age",
            ] {
                read.push(rows(db.execute(query)?));
            }
            if kind == Kind::Lpg {
                let near = db.vector_search("Doc", "emb", &[1.0, 0.1], 1, None, None)?;
                read.push(format!(
                    "{:?}",
                    near.iter().map(|hit| hit.0).collect::<Vec<_>>()
                ));
                let found = db.text_search("Doc", "body", "graph", 5, None)?;
                read.push(format!(
                    "{:?}",
                    found.iter().map(|hit| hit.0).collect::<Vec<_>>()
                ));
                let session = db.session();
                session.set_viewing_epoch(early);
                read.push(rows(session.execute(
                    "MATCH (n:Person) RETURN n.name, n.age ORDER BY n.name",
                )?));
            }
            if kind == Kind::ReleasedLpg("search-indexes/0.6.0-dev/documents.grafeo") {
                let near =
                    db.vector_search("Document", "embedding", &[3.0, 19.0, 88.0], 1, None, None)?;
                read.push(format!(
                    "{:?}",
                    near.iter().map(|hit| hit.0).collect::<Vec<_>>()
                ));
                let found = db.text_search("Document", "content", "canals", 5, None)?;
                read.push(format!(
                    "{:?}",
                    found.iter().map(|hit| hit.0).collect::<Vec<_>>()
                ));
            }
        }
        Kind::Rdf => {
            read.push(rows(db.execute_sparql(
                "SELECT ?s ?p ?o WHERE { ?s ?p ?o } ORDER BY ?s ?p ?o",
            )?));
        }
    }
    Ok(read)
}

/// One chunk of the file: its section, its index among the section's chunks, and where it is.
#[derive(Clone, Copy, Debug)]
struct Chunk {
    section: SectionType,
    index: usize,
    of: usize,
    offset: u64,
    length: u64,
    /// In a file 0.5 wrote, which this build reads but cannot write: damaged on disk only.
    legacy: bool,
}

/// The sections of a file 0.5 wrote, each one blob, found in the file by its bytes.
fn legacy_chunks(path: &Path) -> Vec<Chunk> {
    let LegacyContents::Sections(sections) =
        LegacyFile::open(path, None).unwrap().contents().unwrap()
    else {
        panic!("{}: a 0.5 file with sections", path.display());
    };
    let bytes = std::fs::read(path).unwrap();
    sections
        .into_iter()
        .filter(|(_, data)| !data.is_empty())
        .map(|(section, data)| {
            let at = bytes
                .windows(data.len())
                .position(|window| window == data.as_slice())
                .expect("a 0.5 section is stored as it reads");
            Chunk {
                section,
                index: 0,
                of: 1,
                offset: u64::try_from(at).unwrap(),
                length: u64::try_from(data.len()).unwrap(),
                legacy: true,
            }
        })
        .collect()
}

fn chunks(path: &Path) -> Vec<Chunk> {
    let Ok(manager) = GrafeoFileManager::open_read_only(path, None) else {
        return legacy_chunks(path);
    };
    let root = manager.active_header().root;
    drop(manager);
    let file = std::fs::File::open(path).unwrap();
    let reader = ImageReader::open(&file, root, None).unwrap();
    let mut seen: Vec<(SectionType, usize)> = Vec::new();
    let mut found = Vec::new();
    for entry in reader.entries() {
        let index = match seen.iter_mut().find(|(s, _)| *s == entry.section_type) {
            Some((_, count)) => {
                *count += 1;
                *count - 1
            }
            None => {
                seen.push((entry.section_type, 1));
                0
            }
        };
        found.push(Chunk {
            section: entry.section_type,
            index,
            of: 0,
            offset: entry.offset,
            length: entry.length,
            legacy: false,
        });
    }
    for chunk in &mut found {
        chunk.of = seen.iter().find(|(s, _)| *s == chunk.section).unwrap().1;
    }
    found
}

/// Damage on disk, which the chunk's checksum sees.
#[derive(Clone, Copy, Debug)]
enum OnDisk {
    FlipFirst,
    FlipMiddle,
    FlipLast,
    ZeroFirstPage,
    CutHere,
}

/// Damage behind a valid checksum, written by a checkpoint.
#[derive(Clone, Copy, Debug)]
enum Decoded {
    Empty,
    Half,
    Trailing,
    FlipFirst,
    FlipMiddle,
    FlipLast,
    Zeros,
}

impl Decoded {
    fn apply(self, bytes: &mut Vec<u8>) {
        let flip = |bytes: &mut Vec<u8>, at: usize| {
            if let Some(byte) = bytes.get_mut(at) {
                *byte ^= 0x5A;
            }
        };
        let last = bytes.len().saturating_sub(1);
        match self {
            Decoded::Empty => bytes.clear(),
            Decoded::Half => bytes.truncate(bytes.len() / 2),
            Decoded::Trailing => bytes.extend_from_slice(&[0xFF; 8]),
            Decoded::FlipFirst => flip(bytes, 0),
            Decoded::FlipMiddle => flip(bytes, bytes.len() / 2),
            Decoded::FlipLast => flip(bytes, last),
            Decoded::Zeros => bytes.iter_mut().for_each(|byte| *byte = 0),
        }
    }
}

fn damage_on_disk(path: &Path, chunk: Chunk, damage: OnDisk) {
    let mut bytes = std::fs::read(path).unwrap();
    let start = usize::try_from(chunk.offset).unwrap();
    let length = usize::try_from(chunk.length).unwrap();
    match damage {
        OnDisk::FlipFirst => bytes[start] ^= 0x5A,
        OnDisk::FlipMiddle => bytes[start + length / 2] ^= 0x5A,
        OnDisk::FlipLast => bytes[start + length - 1] ^= 0x5A,
        OnDisk::ZeroFirstPage => {
            let end = (start + 4096).min(bytes.len());
            bytes[start..end].iter_mut().for_each(|byte| *byte = 0);
        }
        OnDisk::CutHere => bytes.truncate(start),
    }
    std::fs::write(path, bytes).unwrap();
}

/// A section written back chunk for chunk, as it was read.
struct Replayed {
    section_type: SectionType,
    version: u8,
    chunks: Vec<(ChunkMeta, Vec<u8>)>,
}

impl Section for Replayed {
    fn section_type(&self) -> SectionType {
        self.section_type
    }

    fn version(&self) -> u8 {
        self.version
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        Err(Error::Internal("a replayed section writes chunks".into()))
    }

    fn deserialize(&mut self, _data: &[u8]) -> Result<()> {
        Err(Error::Internal("a replayed section is not read".into()))
    }

    fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
        for (meta, bytes) in &self.chunks {
            sink.write_chunk(*meta, bytes)?;
        }
        Ok(())
    }

    fn read_from(&mut self, _source: &dyn SectionSource) -> Result<()> {
        Err(Error::Internal("a replayed section is not read".into()))
    }

    fn is_dirty(&self) -> bool {
        true
    }

    fn mark_clean(&self) {}

    fn memory_usage(&self) -> usize {
        0
    }
}

/// Checkpoints the file again with every section of its active image, chunk for chunk, the
/// `index`th chunk of `section` damaged by `damage`.
fn damage_decoded(path: &Path, section: SectionType, index: usize, damage: Decoded) {
    let manager = GrafeoFileManager::open(path, None).unwrap();
    let mut sections: Vec<Replayed> = manager
        .read_image(|image| {
            let mut sections = Vec::new();
            for section_type in (0..=u8::MAX).filter_map(SectionType::from_u8) {
                let Some(source) = image.section_source(section_type) else {
                    continue;
                };
                let chunks = (0..source.chunks().len())
                    .map(|at| Ok((source.chunks()[at], source.fetch(at)?.to_vec())))
                    .collect::<Result<Vec<_>>>()?;
                sections.push(Replayed {
                    section_type,
                    version: source.section_version(),
                    chunks,
                });
            }
            Ok(sections)
        })
        .unwrap();
    let target = sections
        .iter_mut()
        .find(|replayed| replayed.section_type == section)
        .unwrap();
    damage.apply(&mut target.chunks[index].1);
    let active = manager.active_header();
    let sections: Vec<&dyn Section> = sections.iter().map(|s| s as &dyn Section).collect();
    manager
        .write_checkpoint(
            &sections,
            &CheckpointHeader {
                checkpoint_lsn: active.checkpoint_lsn,
                epoch: active.epoch,
                last_transaction_id: active.last_transaction_id,
                node_count: active.node_count,
                edge_count: active.edge_count,
            },
        )
        .unwrap();
    manager.close().unwrap();
}

/// What opening and reading a damaged copy gave.
enum Found {
    /// It read what the undamaged file reads.
    Same,
    /// A corruption naming the file.
    Refused,
    /// It read something else.
    Other(Vec<String>),
    /// Another error.
    Failed(Error),
    Panicked(String),
}

impl Found {
    fn show(&self) -> String {
        match self {
            Found::Same => "read the same".to_string(),
            Found::Refused => "refused as a corruption of the file".to_string(),
            Found::Other(read) => format!("read something else: {read:?}"),
            Found::Failed(error) => format!("failed: {error:?}"),
            Found::Panicked(message) => format!("panicked: {message}"),
        }
    }
}

fn open_and_read(
    path: &Path,
    kind: Kind,
    early: EpochId,
    whole: &[String],
) -> Vec<(&'static str, Found)> {
    let mut found = Vec::new();
    for mode in ["read-only", "read-write"] {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let db = match mode {
                "read-only" => GrafeoDB::open_read_only(path)?,
                _ => GrafeoDB::with_config(config(kind, path))?,
            };
            read_whole(&db, kind, early)
        }));
        found.push((
            mode,
            match outcome {
                Ok(Ok(read)) if read == whole => Found::Same,
                Ok(Ok(read)) => Found::Other(read),
                Ok(Err(Error::Corruption(corruption)))
                    if corruption.file.as_deref() == Some(path) =>
                {
                    Found::Refused
                }
                Ok(Err(error)) => Found::Failed(error),
                Err(panic) => Found::Panicked(
                    panic
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_string()))
                        .unwrap_or_default(),
                ),
            },
        ));
    }
    found
}

struct Built {
    kind: Kind,
    path: PathBuf,
    early: EpochId,
    whole: Vec<String>,
}

fn built(dir: &Path) -> Vec<Built> {
    KINDS
        .iter()
        .enumerate()
        .map(|(n, &kind)| {
            let path = dir.join(format!("file-{n}.grafeo"));
            let early = build(kind, &path);
            let db = GrafeoDB::open_read_only(&path).unwrap();
            let whole = read_whole(&db, kind, early).unwrap();
            drop(db);
            Built {
                kind,
                path,
                early,
                whole,
            }
        })
        .collect()
}

/// Sections no file of this test holds, with the reason.
const NOT_WRITTEN: &[(&str, &str)] = &[(
    "PropertyIndex",
    "no release writes it: property indexes are built from the rows at load, and the type is      named for memory tiering (`Config::with_section_tier`) only",
)];

#[test]
fn every_section_of_the_catalogue_is_in_a_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut written = Vec::new();
    for file in built(dir.path()) {
        written.extend(
            chunks(&file.path)
                .iter()
                .map(|chunk| format!("{:?}", chunk.section)),
        );
    }
    let missing: Vec<&str> = names(SECTIONS)
        .into_iter()
        .filter(|name| !written.iter().any(|w| w == name))
        .filter(|name| !NOT_WRITTEN.iter().any(|(n, _)| n == name))
        .collect();
    assert!(
        missing.is_empty(),
        "sections no file holds: {missing:?}; build them, or list them in NOT_WRITTEN with the reason"
    );
}

#[test]
fn a_damaged_file_is_refused_or_reads_the_same() {
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    let mut checked = 0;
    for file in built(dir.path()) {
        let pristine = std::fs::read(&file.path).unwrap();
        let copy = dir.path().join("damaged.grafeo");
        for chunk in chunks(&file.path) {
            let at = format!(
                "{:?} {:?} chunk {} of {}",
                file.kind,
                chunk.section,
                chunk.index + 1,
                chunk.of
            );
            for damage in [
                OnDisk::FlipFirst,
                OnDisk::FlipMiddle,
                OnDisk::FlipLast,
                OnDisk::ZeroFirstPage,
                OnDisk::CutHere,
            ] {
                std::fs::write(&copy, &pristine).unwrap();
                damage_on_disk(&copy, chunk, damage);
                for (mode, found) in open_and_read(&copy, file.kind, file.early, &file.whole) {
                    checked += 1;
                    if !matches!(found, Found::Same | Found::Refused) {
                        wrong.push(format!(
                            "{at}, {damage:?} on disk, {mode}: {}",
                            found.show()
                        ));
                    }
                }
            }
            for damage in [
                Decoded::Empty,
                Decoded::Half,
                Decoded::Trailing,
                Decoded::FlipFirst,
                Decoded::FlipMiddle,
                Decoded::FlipLast,
                Decoded::Zeros,
            ] {
                if chunk.legacy {
                    continue;
                }
                std::fs::write(&copy, &pristine).unwrap();
                damage_decoded(&copy, chunk.section, chunk.index, damage);
                for (mode, found) in open_and_read(&copy, file.kind, file.early, &file.whole) {
                    checked += 1;
                    let holds = match &found {
                        Found::Panicked(_) | Found::Failed(Error::Internal(_)) => false,
                        Found::Same | Found::Refused | Found::Other(_) | Found::Failed(_) => true,
                    };
                    if !holds {
                        wrong.push(format!(
                            "{at}, {damage:?} behind its checksum, {mode}: {}",
                            found.show()
                        ));
                    }
                }
            }
        }
    }
    assert!(checked > 100, "{checked} checked");
    assert!(
        wrong.is_empty(),
        "{} of {checked} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
