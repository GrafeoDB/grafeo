//! The harness's expected failures (0.6.2): a `.gtest` case marked `xfail:` (its `spec:` and
//! `issue:`) passes as an expected failure only when it fails the way a spec case fails: on one of
//! the generated assertions or on an error from one of its statements, both of which raise a
//! `SpecFailure`. A case that passes fails the run (XPASS, naming the spec and the issue), and any
//! other panic (an engine bug, a harness bug) goes on unchanged, so it is never taken for the
//! expected failure.
//!
//! The parse rules: `xfail:` needs both `spec:` and `issue:`, a case is never both `skip:` and
//! `xfail:`, and the probe-only key `probe: runner_error` is accepted only in a file under a
//! `spec-probes` directory.
//!
//! ```bash
//! cargo test -p grafeo-spec-tests --test xfail_harness
//! ```

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind, panic_any};

use grafeo_engine::GrafeoDB;
use grafeo_spec_tests::gtest::parse_gtest_text;
use grafeo_spec_tests::{
    SpecFailure, assert_rows_sorted, assert_warnings, execute_query, run_xfail, spec_failure,
};

const SPEC: &str = "ISO/IEC 39075 20.2";
const ISSUE: &str = "#748";

/// The payload of `f`'s panic.
fn payload_of(f: impl FnOnce()) -> Box<dyn Any + Send> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Err(payload) => payload,
        Ok(()) => panic!("it did not fail"),
    }
}

/// The text of the `SpecFailure` `f` fails with.
fn spec_failure_text(f: impl FnOnce()) -> String {
    match payload_of(f).downcast::<SpecFailure>() {
        Ok(failure) => failure.0,
        Err(other) => panic!("a SpecFailure, not {}", describe(&*other)),
    }
}

fn describe(payload: &(dyn Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|text| (*text).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a payload of another type".to_string())
}

// ---------------------------------------------------------------------------
// run_xfail
// ---------------------------------------------------------------------------

#[test]
fn a_spec_failure_is_the_expected_failure() {
    assert_eq!(
        run_xfail(SPEC, ISSUE, || spec_failure("rows differ".to_string())),
        Ok(())
    );
}

#[test]
fn a_failing_assertion_is_the_expected_failure() {
    let db = GrafeoDB::new_in_memory();
    let result = db.execute("RETURN 1 AS v").unwrap();
    assert_eq!(
        run_xfail(SPEC, ISSUE, || assert_rows_sorted(
            &result,
            &[vec!["2".to_string()]]
        )),
        Ok(())
    );
    assert_eq!(
        run_xfail(SPEC, ISSUE, || assert_warnings(&result, &["01X01"])),
        Ok(())
    );
}

#[test]
fn a_statement_error_is_the_expected_failure() {
    let db = GrafeoDB::new_in_memory();
    assert_eq!(
        run_xfail(SPEC, ISSUE, || execute_query(
            &db,
            "gql",
            "RETURN 1 / 0 AS v"
        )),
        Ok(())
    );
}

/// A case marked xfail that passes fails the run, and the message says which spec and which issue
/// it was held against, so the mark can come off.
#[test]
fn a_passing_case_is_an_xpass_naming_the_spec_and_the_issue() {
    let db = GrafeoDB::new_in_memory();
    let message = run_xfail(SPEC, ISSUE, || execute_query(&db, "gql", "RETURN 1 AS v"))
        .expect_err("a case that passes is an XPASS");
    for part in ["XPASS", SPEC, ISSUE] {
        assert!(message.contains(part), "names {part}: {message}");
    }
}

/// Any other panic is not the expected failure: it goes on with its payload unchanged, whatever its
/// type.
#[test]
fn any_other_panic_goes_on_unchanged() {
    let payload = payload_of(|| {
        drop(run_xfail(SPEC, ISSUE, || panic!("an engine bug")));
    });
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"an engine bug"));

    let payload = payload_of(|| {
        // A value the compiler cannot fold into the format string, so the payload is a String
        // (`panic!` with only literal text carries a `&'static str`).
        let at = std::hint::black_box(19);
        drop(run_xfail(SPEC, ISSUE, || panic!("a harness bug at {at}")));
    });
    assert_eq!(
        payload.downcast_ref::<String>().map(String::as_str),
        Some("a harness bug at 19")
    );

    let payload = payload_of(|| {
        drop(run_xfail(SPEC, ISSUE, || panic_any(88_u8)));
    });
    assert_eq!(payload.downcast_ref::<u8>(), Some(&88));
}

// ---------------------------------------------------------------------------
// The helpers raise SpecFailure, with their own text
// ---------------------------------------------------------------------------

/// Each assertion helper and the statement-error path fail with a `SpecFailure` that carries the
/// assertion's text, so an ordinary failing case still says what differed.
#[test]
fn the_helpers_fail_with_a_spec_failure_carrying_their_text() {
    let db = GrafeoDB::new_in_memory();
    let result = db.execute("RETURN 1 AS v").unwrap();

    let rows = spec_failure_text(|| assert_rows_sorted(&result, &[vec!["2".to_string()]]));
    assert!(rows.contains("Mismatch") && rows.contains("'1'"), "{rows}");

    let warnings = spec_failure_text(|| assert_warnings(&result, &["01X01"]));
    assert!(
        warnings.contains("Warning mismatch") && warnings.contains("01X01"),
        "{warnings}"
    );

    let statement = spec_failure_text(|| execute_query(&db, "gql", "RETURN 1 / 0 AS v"));
    assert!(
        statement.contains("Query failed") && statement.contains("RETURN 1 / 0 AS v"),
        "{statement}"
    );

    let direct = spec_failure_text(|| spec_failure("Gus differs".to_string()));
    assert_eq!(direct, "Gus differs");
}

// ---------------------------------------------------------------------------
// The parse rules
// ---------------------------------------------------------------------------

const META: &str = "meta:\n  model: lpg\n  section: probes\n\ntests:\n";

fn gtest(case: &str) -> String {
    format!("{META}{case}")
}

#[test]
fn an_xfail_case_parses_with_its_spec_and_issue() {
    let text = gtest(
        "  - name: marked
    xfail:
      spec: \"ISO/IEC 39075 20.2\"
      issue: \"fix in 0.6.2\"
    query: \"RETURN 1 AS v\"
    expect:
      rows:
        - [2]

  - name: unmarked
    query: \"RETURN 1 AS v\"
    expect:
      rows:
        - [1]
",
    );
    let file = parse_gtest_text("tests/spec/lpg/gql/probe.gtest", &text).unwrap();
    let [marked, unmarked] = &file.tests[..] else {
        panic!("two cases: {}", file.tests.len())
    };
    let xfail = marked.xfail.as_ref().expect("marked is an xfail");
    assert_eq!(
        (xfail.spec.as_str(), xfail.issue.as_str()),
        ("ISO/IEC 39075 20.2", "fix in 0.6.2")
    );
    assert!(marked.skip.is_none());
    assert!(unmarked.xfail.is_none());
}

/// A case is either skipped or expected to fail, never both; an xfail names both its spec and its
/// issue. Each refusal names the case.
#[test]
fn a_case_both_skipped_and_xfail_or_an_xfail_missing_a_key_is_refused() {
    let cases = [
        (
            "skip and xfail",
            "  - name: both_marks
    skip: \"not yet\"
    xfail:
      spec: \"ISO/IEC 39075 20.2\"
      issue: \"#748\"
    query: \"RETURN 1 AS v\"
",
            "both_marks",
        ),
        (
            "no spec",
            "  - name: without_spec
    xfail:
      issue: \"#748\"
    query: \"RETURN 1 AS v\"
",
            "without_spec",
        ),
        (
            "no issue",
            "  - name: without_issue
    xfail:
      spec: \"ISO/IEC 39075 20.2\"
    query: \"RETURN 1 AS v\"
",
            "without_issue",
        ),
    ];
    for (what, case, name) in cases {
        let error =
            parse_gtest_text("tests/spec/lpg/gql/probe.gtest", &gtest(case)).expect_err(what);
        assert!(error.contains(name), "{what}: names the case: {error}");
    }
}

/// The probe-only key is refused in any file outside a `spec-probes` directory, so no spec case can
/// use it to fake a harness failure; under `spec-probes` it parses.
#[test]
fn the_runner_error_probe_parses_only_under_spec_probes() {
    let text = gtest(
        "  - name: a_runner_error
    xfail:
      spec: \"the harness itself\"
      issue: \"#748\"
    query: \"RETURN 1 AS v\"
    probe: runner_error
",
    );
    for path in [
        "tests/spec-probes/xfail/runner_error/probe.gtest",
        "H:\\GrafeoDB\\grafeo\\tests\\spec-probes\\xfail\\runner_error\\probe.gtest",
    ] {
        parse_gtest_text(path, &text).unwrap_or_else(|error| panic!("{path}: {error}"));
    }
    for path in [
        "tests/spec/lpg/gql/probe.gtest",
        "tests/spec/spec-probes-notes/probe.gtest",
        "tests/my-spec-probes/probe.gtest",
    ] {
        let error = parse_gtest_text(path, &text).expect_err(path);
        assert!(error.contains(path), "names the file: {error}");
    }
}
