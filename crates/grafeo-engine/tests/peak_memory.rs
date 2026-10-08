//! A checkpoint and an open hold memory that does not grow with the database
//! (#429).
//!
//! Each section streams its chunks: a checkpoint writes them into the file as
//! the section produces them, and an open reads them one at a time. What the
//! two hold beyond the database is then a few chunks, plus the exceptions the
//! container format's "Memory" section documents (sorted ids and references,
//! 8 or 16 bytes per entity, and the 48-byte directory entry of each chunk).
//! A checkpoint that held a whole section, or an open that read one whole,
//! would hold as much again as that section.
//!
//! The global allocator of this test binary counts the bytes it hands out,
//! the highest count since a reset and the largest single request, and
//! forwards every call to the system allocator. Each test builds a database
//! twice, at two sizes ten times apart, and measures:
//!
//! - `close()`, which writes the final checkpoint: the peak above what was
//!   held before it (the database);
//! - the open of the file: the peak above what the open leaves held (the
//!   database it built).
//!
//! From the smaller to the larger database, that extra peak may grow by
//! twice what the documented exceptions grow by (a vector grown one push at
//! a time holds up to twice its length), plus a slack of four chunks (see
//! [`Scale`]); the open's also by a sixteenth of what it keeps more, for the
//! maps it builds (see [`MAP_GROWTH_DIVISOR`]). Measured on Windows with all
//! features and with the shipped set (`full`, `compact-store`,
//! `arrow-export`), the checkpoint's extra peak grew by 0.20 to 0.40 MiB, at
//! most 47 bytes for each person added (a node and an edge with four values,
//! for which the Memory section allows 48), and the open's not at all; a
//! section held whole made it grow by 2.5 MiB (the vector topology) to 37
//! MiB (the RDF triples).
//!
//! The tests write with chunk caps of 1,024 rows and 64 KiB ([`SCALE`]), so
//! each smaller database already spans tens of chunks, and each test runs
//! in a few seconds in a debug build. The default caps
//! allow 65,536 rows a chunk, and a chunk being written holds its rows as
//! values until it is encoded: for a column of small values, such as the
//! node labels, that is several MiB, bounded by the row cap and not by the
//! database (the checkpoint of the smaller database of the test at the
//! default caps holds 14 MiB above it). That test ([`DEFAULT_SCALE`]) needs
//! databases of a million rows to see past it, so it runs in release builds
//! only, before a release.
//!
//! The counters are global, so the tests that measure take turns (nextest
//! runs each test in a process of its own anyway).
//!
//! The tests build their databases in batches of 1,000 entities ([`BATCH`]),
//! one commit each: what a checkpoint and an open hold does not depend on
//! how many commits built the database, and a commit per entity runs out of
//! memory on Windows in a build with grafeo-core's `tiered-storage` (see
//! [`BATCH`]).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test peak_memory -- --nocapture
//! ```

#![cfg(all(feature = "lpg", feature = "grafeo-file", feature = "wal"))]
#![expect(
    unsafe_code,
    reason = "GlobalAlloc is an unsafe trait; this only forwards to System"
)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, MutexGuard, PoisonError};

use grafeo_common::storage::ChunkCaps;
use grafeo_common::testing::chunk_caps::with_chunk_caps;
use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_engine::database::BatchEdge;
use grafeo_engine::{Config, DurabilityMode, GrafeoDB};
use grafeo_storage::file::GrafeoFileManager;

// --- Counting ---------------------------------------------------------------------

/// Counts the bytes held through the global allocator, the most held since
/// the last [`reset`](counting::reset), and the largest single request.
mod counting {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CURRENT: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);
    static LARGEST: AtomicUsize = AtomicUsize::new(0);

    /// Forwards every call to [`System`] and counts what it hands out.
    pub struct Counting;

    fn requested(size: usize) {
        LARGEST.fetch_max(size, Ordering::Relaxed);
    }

    fn grew(by: usize) {
        let now = CURRENT.fetch_add(by, Ordering::Relaxed).wrapping_add(by);
        PEAK.fetch_max(now, Ordering::Relaxed);
    }

    fn shrank(by: usize) {
        CURRENT.fetch_sub(by, Ordering::Relaxed);
    }

    // SAFETY: every method passes its arguments unchanged to `System`, which
    // meets the trait's contract, and returns what `System` returned. The
    // counting touches atomics only and allocates nothing.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            requested(layout.size());
            // SAFETY: the caller meets `alloc`'s contract, which is `System`'s.
            let block = unsafe { System.alloc(layout) };
            if !block.is_null() {
                grew(layout.size());
            }
            block
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            requested(layout.size());
            // SAFETY: as in `alloc`.
            let block = unsafe { System.alloc_zeroed(layout) };
            if !block.is_null() {
                grew(layout.size());
            }
            block
        }

        unsafe fn dealloc(&self, block: *mut u8, layout: Layout) {
            // SAFETY: `block` was handed out by this allocator, so by
            // `System`, with `layout`.
            unsafe { System.dealloc(block, layout) };
            shrank(layout.size());
        }

        unsafe fn realloc(&self, block: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            requested(new_size);
            // SAFETY: as in `dealloc`, and the caller meets `realloc`'s
            // contract for `new_size`.
            let moved = unsafe { System.realloc(block, layout, new_size) };
            if !moved.is_null() {
                if new_size >= layout.size() {
                    grew(new_size - layout.size());
                } else {
                    shrank(layout.size() - new_size);
                }
            }
            moved
        }
    }

    /// The bytes held now.
    pub fn current() -> usize {
        CURRENT.load(Ordering::Relaxed)
    }

    /// The most bytes held since the last [`reset`].
    pub fn peak() -> usize {
        PEAK.load(Ordering::Relaxed)
    }

    /// The largest single request since the last [`reset`].
    pub fn largest() -> usize {
        LARGEST.load(Ordering::Relaxed)
    }

    /// Starts a measurement: the peak from what is held now, the largest
    /// request from 0.
    pub fn reset() {
        PEAK.store(current(), Ordering::Relaxed);
        LARGEST.store(0, Ordering::Relaxed);
    }
}

#[global_allocator]
static COUNTING: counting::Counting = counting::Counting;

/// The counters are global: the tests that measure take turns.
static MEASURING: Mutex<()> = Mutex::new(());

fn measuring() -> MutexGuard<'static, ()> {
    MEASURING.lock().unwrap_or_else(PoisonError::into_inner)
}

/// What a call held, in bytes.
#[derive(Debug, Clone, Copy)]
struct Measured {
    before: usize,
    after: usize,
    peak: usize,
    largest: usize,
}

impl Measured {
    /// The most the call held beyond what was held before it.
    fn above_before(&self) -> usize {
        self.peak.saturating_sub(self.before)
    }

    /// The most the call held beyond what it left held.
    fn above_after(&self) -> usize {
        self.peak.saturating_sub(self.after)
    }

    /// What the call left held.
    fn kept(&self) -> usize {
        self.after.saturating_sub(self.before)
    }
}

/// Runs `call` and measures what it held.
fn measure<T>(call: impl FnOnce() -> T) -> (T, Measured) {
    let before = counting::current();
    counting::reset();
    let result = call();
    let measured = Measured {
        before,
        after: counting::current(),
        peak: counting::peak(),
        largest: counting::largest(),
    };
    (result, measured)
}

fn mib(bytes: usize) -> String {
    format!("{:.2} MiB", bytes as f64 / f64::from(1 << 20))
}

// --- The bound --------------------------------------------------------------------

/// The chunk caps a test writes with and the sizes of its two databases.
#[derive(Debug, Clone, Copy)]
struct Scale {
    caps: ChunkCaps,
    /// The entities of the smaller database; the larger holds ten times as
    /// many.
    smaller: usize,
    /// What the extra peak may grow by besides the documented exceptions:
    /// four chunks of the byte cap. The chunks being written or read are
    /// bounded by the caps, but their high-water mark differs between two
    /// databases.
    slack: usize,
}

/// Caps of 1,024 rows and 64 KiB, and 2,000 entities against 20,000: the
/// smaller database spans two row groups and tens of chunks.
const SCALE: Scale = Scale {
    caps: ChunkCaps {
        max_rows: 1_024,
        max_bytes: 64 << 10,
    },
    smaller: 2_000,
    slack: 4 * (64 << 10),
};

/// The default caps (65,536 rows and 1 MiB), and two row groups against
/// twenty.
const DEFAULT_SCALE: Scale = Scale {
    caps: ChunkCaps::DEFAULT,
    smaller: 2 * 65_536,
    slack: 4 << 20,
};

/// The bytes the directory holds for each chunk (the Memory section names
/// them for a checkpoint; an open reads the same directory).
const DIRECTORY_ENTRY: usize = 48;

/// The bytes per entity of a test database that the container format's
/// Memory section lets each step hold, besides the chunks and the directory.
#[derive(Debug, Clone, Copy)]
struct Documented {
    checkpoint: usize,
    open: usize,
}

/// One database, built, checkpointed by `close()` and opened again.
#[derive(Debug)]
struct Run {
    entities: usize,
    /// Chunks of the image `close()` wrote.
    chunks: usize,
    file: u64,
    close: Measured,
    open: Measured,
}

impl Run {
    /// What the documented exceptions hold for this database, at `per_entity`
    /// bytes per entity.
    fn documented(&self, per_entity: usize) -> usize {
        per_entity * self.entities + DIRECTORY_ENTRY * self.chunks
    }

    fn describe(&self) -> String {
        format!(
            "{} entities, {} chunks, file {}: close held {} above the database \
             (largest request {}); open kept {} and held {} above it (largest request {})",
            self.entities,
            self.chunks,
            mib(usize::try_from(self.file).unwrap_or(usize::MAX)),
            mib(self.close.above_before()),
            mib(self.close.largest),
            mib(self.open.kept()),
            mib(self.open.above_after()),
            mib(self.open.largest),
        )
    }
}

/// Builds a database of `entities` entities with `build`, checkpoints it
/// with `close()` under `caps`, opens it again and hands it to `check`,
/// measuring the close and the open.
fn run(
    caps: ChunkCaps,
    entities: usize,
    build: &dyn Fn(&GrafeoDB, usize),
    check: &dyn Fn(&GrafeoDB, usize),
) -> Run {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("peak.grafeo");
    let config = || Config::persistent(&path).with_wal_durability(DurabilityMode::NoSync);

    let db = GrafeoDB::with_config(config()).unwrap();
    build(&db, entities);
    // `close()` builds and writes the sections on this thread, so with the
    // caps of this thread.
    let (closed, close) = with_chunk_caps(caps, || measure(|| db.close()));
    closed.unwrap();
    drop(db);

    let file = std::fs::metadata(&path).unwrap().len();
    let chunks = chunk_count(&path);
    let (db, open) = measure(|| GrafeoDB::with_config(config()).unwrap());
    check(&db, entities);
    db.close().unwrap();

    Run {
        entities,
        chunks,
        file,
        close,
        open,
    }
}

/// The chunks of the active image of the file at `path`.
fn chunk_count(path: &Path) -> usize {
    let manager = GrafeoFileManager::open_read_only(path, None).unwrap();
    let chunks = manager.image_stats().unwrap().chunks;
    manager.close().unwrap();
    chunks
}

/// The part of what an open keeps more that its extra peak may grow by:
/// a sixteenth. An open builds the store's maps as it reads, and a map that
/// grows holds its old table (half the new one) until it has moved: a
/// sawtooth in the database's size. With people of 200-byte biographies at
/// the test caps it measured 0, 3.5, 0, 3.4, 0 and 11.6 MiB at 20,000,
/// 60,000, 100,000, 140,000, 200,000 and 260,000 people, at most 2.3% of
/// what the open kept.
const MAP_GROWTH_DIVISOR: usize = 16;

/// Builds and measures a database of `scale.smaller` entities and one of ten
/// times as many, and asserts that the extra peak of the close and of the
/// open grows by at most twice what the `documented` exceptions grow by,
/// plus `scale.slack`, and for the open also [`MAP_GROWTH_DIVISOR`]'s part
/// of what it keeps more.
fn assert_flat(
    what: &str,
    scale: Scale,
    documented: Documented,
    build: &dyn Fn(&GrafeoDB, usize),
    check: &dyn Fn(&GrafeoDB, usize),
) {
    let _measuring = measuring();
    let small = run(scale.caps, scale.smaller, build, check);
    let large = run(scale.caps, 10 * scale.smaller, build, check);
    let report = format!(
        "{what}\n  smaller: {}\n  larger:  {}",
        small.describe(),
        large.describe()
    );
    eprintln!("{report}");

    let allowed = |per_entity: usize| {
        2 * large
            .documented(per_entity)
            .saturating_sub(small.documented(per_entity))
            + scale.slack
    };
    let close_growth = large
        .close
        .above_before()
        .saturating_sub(small.close.above_before());
    let close_allowed = allowed(documented.checkpoint);
    assert!(
        close_growth <= close_allowed,
        "the checkpoint of {what} held {} more above the database at ten times the size, \
         more than the {} the documented exceptions allow\n{report}",
        mib(close_growth),
        mib(close_allowed),
    );
    let open_growth = large
        .open
        .above_after()
        .saturating_sub(small.open.above_after());
    let open_allowed = allowed(documented.open)
        + large.open.kept().saturating_sub(small.open.kept()) / MAP_GROWTH_DIVISOR;
    assert!(
        open_growth <= open_allowed,
        "the open of {what} held {} more above the database at ten times the size, more \
         than the {} the documented exceptions and growing maps allow\n{report}",
        mib(open_growth),
        mib(open_allowed),
    );
}

// --- Data -------------------------------------------------------------------------

const PEOPLE: [&str; 5] = ["Alix", "Gus", "Vincent", "Mia", "Jules"];
const CITIES: [&str; 4] = ["Amsterdam", "Berlin", "Paris", "Prague"];

/// The entities a test adds per commit. In a build with grafeo-core's
/// `tiered-storage` (which `--all-features` on the workspace turns on),
/// every commit's epoch gets an arena with a first chunk of 1 MiB that lives
/// as long as the database (#433). Windows commits that memory: built with a
/// direct call per entity, a database of 20,000 entities held 20 GiB, and
/// the tests ran out of memory on CI. In batches of 1,000 they hold 20 MiB of
/// arenas.
const BATCH: usize = 1_000;

/// Hands `items` to `add` in batches of [`BATCH`].
fn in_batches<T>(items: impl Iterator<Item = T>, mut add: impl FnMut(Vec<T>)) {
    let mut items = items.peekable();
    while items.peek().is_some() {
        add(items.by_ref().take(BATCH).collect());
    }
}

/// A text of `length` bytes that no other `index` gives.
fn text_of(index: usize, length: usize) -> String {
    let mut text = format!(
        "{} {index} lives in {}. ",
        PEOPLE[index % PEOPLE.len()],
        CITIES[index % CITIES.len()]
    );
    while text.len() < length {
        text.push_str("Alix and Gus walk from Amsterdam to Berlin, Paris and Prague. ");
    }
    text.truncate(length);
    text
}

/// The next number of a fixed sequence (a linear congruential generator).
#[cfg(any(feature = "vector-index", feature = "text-index"))]
fn next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state >> 33
}

/// The bytes of a literal of the RDF test: the data dominates the triples.
#[cfg(feature = "triple-store")]
const LITERAL: usize = 2_048;

/// 1,000 people against 10,000, with biographies of [`BIO`] bytes: 4 and 40
/// MiB of them. Fewer entities than the other tests: in a build with
/// grafeo-core's `tiered-storage` (which `--all-features` on the workspace
/// turns on), an open puts every node and edge record into the arena of one
/// epoch, which uses only its first chunk of 1 MiB (#433), and 10,000
/// people take about 800 KB of records.
const PEOPLE_SCALE: Scale = Scale {
    smaller: 1_000,
    ..SCALE
};

/// The bytes of a biography: the data dominates the people.
const BIO: usize = 4_096;

/// Per person, a node with three values and an edge with one: the sorted ids
/// of each table (8 bytes per node or edge) and of each property column (8
/// bytes per value). An open holds what it builds.
const PERSON: Documented = Documented {
    checkpoint: 8 + 8 + 4 * 8,
    open: 0,
};

/// Adds `people` people with a name, an age and a biography of `bio` bytes,
/// each knowing the one before.
fn add_people(db: &GrafeoDB, people: usize, bio: usize) {
    let mut persons: Vec<NodeId> = Vec::with_capacity(people);
    in_batches(
        (0..people).map(|index| {
            HashMap::from([
                (
                    PropertyKey::from("name"),
                    Value::from(PEOPLE[index % PEOPLE.len()]),
                ),
                (
                    PropertyKey::from("age"),
                    Value::from(i64::try_from(19 + index % 88).unwrap()),
                ),
                (PropertyKey::from("bio"), Value::from(text_of(index, bio))),
            ])
        }),
        |batch| persons.extend(db.batch_create_nodes_with_props("Person", batch).unwrap()),
    );
    in_batches(
        persons.windows(2).map(|pair| {
            BatchEdge::new(pair[0], pair[1], "KNOWS").with_properties([("since", 1988_i64)])
        }),
        |batch| {
            db.batch_create_edges(batch).unwrap();
        },
    );
}

/// Checks that the people [`add_people`] added came back.
fn check_people(db: &GrafeoDB, people: usize, bio: usize) {
    assert_eq!(db.node_count(), people, "every person came back");
    assert_eq!(db.edge_count(), people - 1, "every KNOWS edge came back");
    let last = NodeId::new(u64::try_from(people - 1).unwrap());
    assert_eq!(
        db.get_node(last)
            .and_then(|node| node.get_property("bio").cloned()),
        Some(Value::from(text_of(people - 1, bio))),
        "the last biography came back"
    );
}

// --- Tests ------------------------------------------------------------------------

/// People with a biography of [`BIO`] bytes (see [`PEOPLE_SCALE`]).
#[test]
fn checkpoint_and_open_memory_does_not_grow_with_the_lpg_store() {
    assert_flat(
        "the LPG store",
        PEOPLE_SCALE,
        PERSON,
        &|db, people| add_people(db, people, BIO),
        &|db, people| check_people(db, people, BIO),
    );
}

/// The same with the default chunk caps: 131,072 people against 1,310,720,
/// with biographies of 200 bytes (a row group of 65,536 people is then about
/// 13 MiB of biographies, 13 chunks of the byte cap). The larger database
/// holds about 2 GiB and the test writes about 1 GiB, so it is ignored in
/// normal runs; in a release build it takes a minute and a half. Run it
/// before a release, with the shipped features (not with grafeo-core's
/// `tiered-storage`, whose arena cannot open that many people yet, see
/// [`PEOPLE_SCALE`]):
///
/// ```bash
/// cargo test --release -p grafeo-engine --features full,compact-store,arrow-export \
///     --test peak_memory -- --ignored --nocapture
/// ```
#[test]
#[ignore = "release only: about 2 GiB of memory and 90 seconds in a release build"]
fn checkpoint_and_open_memory_does_not_grow_with_the_lpg_store_at_the_default_caps() {
    assert_flat(
        "the LPG store at the default caps",
        DEFAULT_SCALE,
        PERSON,
        &|db, people| add_people(db, people, 200),
        &|db, people| check_people(db, people, 200),
    );
}

/// Documents with an embedding of 16 dimensions, under an HNSW index whose
/// topology (up to 32 neighbors a node) outweighs the embeddings.
#[cfg(feature = "vector-index")]
#[test]
fn checkpoint_and_open_memory_does_not_grow_with_a_vector_index() {
    assert_flat(
        "a vector index",
        SCALE,
        // A sorted reference to each node of the topology (16 bytes), and
        // the LPG section's sorted ids of the nodes and of the embedding
        // column (8 bytes each).
        Documented {
            checkpoint: 16 + 8 + 8,
            open: 0,
        },
        &|db, documents| {
            let mut state = 88;
            in_batches(
                (0..documents).map(|_| {
                    (0..16)
                        .map(|_| (next(&mut state) % 1_000) as f32 / 1_000.0)
                        .collect()
                }),
                |batch| {
                    db.batch_create_nodes("Doc", "embedding", batch).unwrap();
                },
            );
            db.create_vector_index(
                "Doc",
                "embedding",
                Some(16),
                Some("euclidean"),
                Some(16),
                Some(32),
                None,
            )
            .unwrap();
        },
        &|db, documents| {
            let index = db
                .store()
                .get_vector_index("Doc", "embedding")
                .expect("the vector index came back");
            assert_eq!(index.len(), documents, "every node is in the index");
        },
    );
}

/// Documents of 40 words from a vocabulary of 1,024, under a text index
/// whose posting lists outweigh the texts.
#[cfg(feature = "text-index")]
#[test]
fn checkpoint_and_open_memory_does_not_grow_with_a_text_index() {
    assert_flat(
        "a text index",
        SCALE,
        // Writing: a copy of the document lengths (16 bytes a document), a
        // sorted copy of the posting list being written (16 bytes a posting,
        // at most one per document), and the LPG section's sorted ids of the
        // nodes and of the text column (8 bytes each); the references to the
        // terms (16 bytes a term) stay the same, as the vocabulary does.
        // Reading: each document's length and what is left of it to cover
        // (16 bytes a document).
        Documented {
            checkpoint: 16 + 16 + 8 + 8,
            open: 16,
        },
        &|db, documents| {
            use std::fmt::Write as _;

            let mut state = 19;
            in_batches(
                (0..documents).map(|_| {
                    let mut body = String::new();
                    for _ in 0..40 {
                        let term = usize::try_from(next(&mut state) % 1_024).unwrap();
                        let city = CITIES[term % CITIES.len()].to_lowercase();
                        write!(body, "{city}{term} ").unwrap();
                    }
                    HashMap::from([(PropertyKey::from("body"), Value::from(body))])
                }),
                |batch| {
                    db.batch_create_nodes_with_props("Doc", batch).unwrap();
                },
            );
            db.create_text_index("Doc", "body").unwrap();
        },
        &|db, documents| {
            assert_eq!(db.node_count(), documents, "every document came back");
            let found = db.text_search("Doc", "body", "amsterdam88", 3).unwrap();
            assert_eq!(found.len(), 3, "the text index came back: {found:?}");
        },
    );
}

/// Triples whose objects are literals of [`LITERAL`] bytes.
#[cfg(feature = "triple-store")]
#[test]
fn checkpoint_and_open_memory_does_not_grow_with_rdf_triples() {
    use grafeo_core::graph::rdf::{Term, Triple};

    assert_flat(
        "RDF triples",
        SCALE,
        // A reference to every triple, sorted (8 bytes a triple).
        Documented {
            checkpoint: 8,
            open: 0,
        },
        &|db, triples| {
            let batch = (0..triples).map(|index| {
                Triple::new(
                    Term::iri(format!("http://grafeo.dev/person/{index}")),
                    Term::iri("http://grafeo.dev/bio"),
                    Term::literal(text_of(index, LITERAL)),
                )
            });
            assert_eq!(db.batch_insert_rdf(batch).unwrap(), triples);
        },
        &|db, triples| {
            assert_eq!(db.rdf_store().len(), triples, "every triple came back");
        },
    );
}

/// A text index stream that claims far more than it holds allocates only
/// for what it holds: a reader that trusted a count with an allocation
/// would ask for 256 MiB here, which a machine that commits memory lazily
/// grants without a sign.
#[cfg(feature = "text-index")]
#[test]
fn a_text_index_stream_that_claims_huge_counts_allocates_only_for_what_it_holds() {
    use std::sync::Arc;

    use grafeo_common::storage::{
        ChunkMeta, ImageSource, MemoryImage, Section, SectionSink, SectionType,
    };
    use grafeo_core::index::text::{BM25Config, InvertedIndex, TextIndexSection};
    use parking_lot::RwLock;

    /// 2^24 documents or postings of 16 bytes each, or a term of 2^28
    /// bytes: 256 MiB if allocated up front.
    const HUGE_COUNT: u64 = 1 << 24;
    const HUGE_LENGTH: u32 = 1 << 28;
    /// Far below the 256 MiB, far above what reading the few bytes present
    /// needs.
    const BOUND: usize = 1 << 20;

    fn index(text: &str) -> Arc<RwLock<InvertedIndex>> {
        let mut index = InvertedIndex::new(BM25Config::default());
        if !text.is_empty() {
            index.insert(NodeId::new(3), text);
        }
        Arc::new(RwLock::new(index))
    }

    /// The stream header: BM25 parameters, the total length, the document
    /// count and the term count.
    fn header(total_length: u64, documents: u64, terms: u64) -> Vec<u8> {
        let mut stream = Vec::new();
        stream.extend_from_slice(&1.2_f64.to_le_bytes());
        stream.extend_from_slice(&0.75_f64.to_le_bytes());
        stream.extend_from_slice(&total_length.to_le_bytes());
        stream.extend_from_slice(&documents.to_le_bytes());
        stream.extend_from_slice(&terms.to_le_bytes());
        stream
    }

    /// Node 3 as the one document, of length 3.
    fn one_document(stream: &mut Vec<u8>) {
        stream.extend_from_slice(&3_u64.to_le_bytes());
        stream.extend_from_slice(&3_u32.to_le_bytes());
    }

    // The metadata chunk a checkpoint writes for an index on `Doc.body`.
    let written = TextIndexSection::new(vec![("Doc:body".to_string(), index("Paris"))]);
    let image = MemoryImage::from_sections(&[&written as &dyn Section]).unwrap();
    let meta = image
        .section_source(SectionType::TextIndex)
        .expect("the image holds the text index section")
        .fetch(0)
        .unwrap();

    let mut documents = header(3, HUGE_COUNT, 1);
    one_document(&mut documents);

    let mut postings = header(3, 1, 1);
    one_document(&mut postings);
    postings.extend_from_slice(&5_u32.to_le_bytes());
    postings.extend_from_slice(b"paris");
    postings.extend_from_slice(&HUGE_COUNT.to_le_bytes());
    postings.extend_from_slice(&3_u64.to_le_bytes());
    postings.extend_from_slice(&3_u32.to_le_bytes());

    let mut term = header(3, 1, 1);
    one_document(&mut term);
    term.extend_from_slice(&HUGE_LENGTH.to_le_bytes());
    term.extend_from_slice(b"paris");

    let _measuring = measuring();
    for (what, claimed, stream) in [
        ("a document count", HUGE_COUNT, documents),
        ("a posting count", HUGE_COUNT, postings),
        ("a term length", u64::from(HUGE_LENGTH), term),
    ] {
        let mut image = MemoryImage::new();
        image
            .begin_section(SectionType::TextIndex, written.version())
            .unwrap();
        image.write_chunk(ChunkMeta::meta(), &meta).unwrap();
        image
            .write_chunk(ChunkMeta::stream_piece(0, 0, 0), &stream)
            .unwrap();
        let source = image.section_source(SectionType::TextIndex).unwrap();
        let mut section = TextIndexSection::new(vec![("Doc:body".to_string(), index(""))]);

        let (read, measured) = measure(|| section.read_from(&*source));
        eprintln!("{what}: largest request {} bytes", measured.largest);
        let error = read
            .expect_err("the stream ends before what it claims")
            .to_string();
        assert!(
            error.contains("ends before"),
            "{what}: the read fails where the stream ends: {error}"
        );
        assert!(
            measured.largest < BOUND,
            "{what} of {claimed}: the read asked for {} bytes at once",
            measured.largest
        );
    }
}
