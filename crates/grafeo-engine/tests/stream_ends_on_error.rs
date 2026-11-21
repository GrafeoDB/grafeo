//! A stream that returned an error ends: a statement that fails partway returns its rows up to
//! the error, then the error, and no row after it however often it is read. Before, a read after
//! the error went on with the rows the operators still held or could still compute.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main stream_ends_on_error::
//! ```

#![cfg(feature = "gql")]

use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;

/// A division by zero at row 3,000 of 5,000: the first chunk (2,048 rows) comes, the next one
/// fails, and the rows after it never come.
#[test]
fn a_stream_that_failed_returns_no_more_rows() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let mut stream = session
        .execute_streaming("UNWIND range(1, 5000) AS i RETURN 10 / (i - 3000) AS q")
        .unwrap();
    let mut delivered = 0;
    let error = loop {
        match stream.next_chunk() {
            Ok(Some(chunk)) => delivered += chunk.selected_indices().count(),
            Ok(None) => panic!("the stream ended after {delivered} rows without the error"),
            Err(error) => break error,
        }
    };
    assert!(
        delivered < 3000,
        "{delivered} rows before the error at row 3,000"
    );
    assert_eq!(GqlStatus::from(&error).as_str(), "22012", "{error}");
    for _ in 0..10 {
        match stream.next_chunk() {
            Ok(None) => {}
            Ok(Some(chunk)) => panic!(
                "{} rows after the error, {delivered} before it",
                chunk.selected_indices().count()
            ),
            Err(error) => panic!("a second error after the first: {error}"),
        }
    }
}
