//! A read on a graph of a closed type allocates what the same read does on a graph with none: it has
//! nothing to check, so the type's size must not reach it. Counted, so it holds in any build, and
//! the type has nine node types and thirteen edge types, as a real schema does.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test typed_read_allocations
//! ```

#![allow(unsafe_code, reason = "a global allocator that counts")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

struct Counting;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

// SAFETY: forwards every call to the system allocator and only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller's contract, passed on.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract, passed on.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const TYPE: &str = "CREATE GRAPH TYPE large (\
    NODE TYPE Person (id STRING NOT NULL, firstName STRING, lastName STRING, gender STRING, birthday INT64, \
    creationDate INT64, locationIP STRING, browserUsed STRING), \
    NODE TYPE City (id STRING NOT NULL, name STRING, country STRING), \
    NODE TYPE Tag (id STRING NOT NULL, name STRING), \
    NODE TYPE Country (id STRING NOT NULL, name STRING), \
    NODE TYPE University (id STRING NOT NULL, name STRING), \
    NODE TYPE Company (id STRING NOT NULL, name STRING), \
    NODE TYPE Forum (id STRING NOT NULL, title STRING, creationDate INT64), \
    NODE TYPE Post (id STRING NOT NULL, content STRING, imageFile STRING, creationDate INT64, length INT64), \
    NODE TYPE Comment (id STRING NOT NULL, content STRING, creationDate INT64, length INT64), \
    EDGE TYPE KNOWS (creationDate INT64), EDGE TYPE LIVES_IN (), EDGE TYPE HAS_INTEREST (), EDGE TYPE IS_PART_OF (), \
    EDGE TYPE STUDY_AT (classYear INT64), EDGE TYPE WORK_AT (workFrom INT64), EDGE TYPE IS_LOCATED_IN (), \
    EDGE TYPE HAS_CREATOR (), EDGE TYPE HAS_TAG (), EDGE TYPE CONTAINER_OF (), EDGE TYPE REPLY_OF (), \
    EDGE TYPE HAS_MODERATOR (), EDGE TYPE HAS_MEMBER (joinDate INT64))";

/// How many allocations a read on the typed graph may make over the plain one's: the name of the
/// type's lookup, and no more.
const MARGIN: u64 = 8;

/// A graph with the same nodes, closed by the type or not, and the same index.
fn graph(typed: bool) -> (GrafeoDB, Session) {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    if typed {
        session.execute(TYPE).unwrap();
        session.execute("CREATE GRAPH g TYPED large").unwrap();
    } else {
        session.execute("CREATE GRAPH g").unwrap();
    }
    session.execute("USE GRAPH g").unwrap();
    for at in 0..50 {
        session
            .execute(&format!(
                "INSERT (:Person {{id: 'person_{at}', firstName: 'Alix', lastName: 'Gus'}})"
            ))
            .unwrap();
    }
    session
        .execute("CREATE INDEX person_id FOR (n:Person) ON (n.id)")
        .unwrap();
    (db, session)
}

/// What the statement allocates on the last of its runs, with its plan kept by then.
fn allocations(session: &Session, query: &str, params: &HashMap<String, Value>) -> u64 {
    for _ in 0..3 {
        std::hint::black_box(session.execute_with_params(query, params.clone()).unwrap());
    }
    let cloned = params.clone();
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let result = session.execute_with_params(query, cloned).unwrap();
    let after = ALLOCATIONS.load(Ordering::Relaxed);
    std::hint::black_box(result);
    after - before
}

/// The one test of this binary: other tests' allocations would count.
#[test]
fn a_read_on_a_closed_type_allocates_what_it_does_on_no_type() {
    let point = HashMap::from([("v".to_string(), Value::from("person_7"))]);
    let none = HashMap::new();
    let (plain_db, plain) = graph(false);
    let (typed_db, typed) = graph(true);
    for (name, query, params) in [
        ("no data", "RETURN 1 AS one", &none),
        (
            "a point read",
            "MATCH (n:Person {id: $v}) RETURN n.firstName",
            &point,
        ),
        (
            "a label scan",
            "MATCH (n:Person) RETURN count(n) AS c",
            &none,
        ),
    ] {
        let without = allocations(&plain, query, params);
        let with = allocations(&typed, query, params);
        assert!(
            with <= without + MARGIN,
            "{name}: {with} allocations on the closed type, {without} on none"
        );
    }
    drop((plain, typed, plain_db, typed_db));
}
