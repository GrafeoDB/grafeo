//! `CALL grafeo.procedures()` lists, for each procedure, whether it writes (`writes`, a BOOLEAN)
//! and what it reaches outside the database (`external_access`: 'none', 'reads' or 'writes'),
//! as each procedure declares them. The listing must say what CALL enforces: a writing
//! procedure is refused on a ReadOnly session, and one that reaches outside is refused on a
//! session without external access. The test reads the registry, so a procedure added later,
//! or one a build profile leaves out, is covered without a change.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main procedures_declare_what_call_enforces::
//! ```

#![cfg(feature = "gql")]

use std::collections::BTreeMap;

use grafeo_common::types::Value;
use grafeo_common::utils::error::Error;
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;
use grafeo_engine::auth::{ExternalAccess, Identity, Role};
use grafeo_engine::procedures::BuiltinProcedures;

/// The listing's name for an external access.
fn listed_access(access: ExternalAccess) -> &'static str {
    match access {
        ExternalAccess::None => "none",
        ExternalAccess::Reads => "reads",
        ExternalAccess::Writes => "writes",
        #[allow(
            unreachable_patterns,
            reason = "the enum is non_exhaustive outside its crate; a new kind needs a name here"
        )]
        other => panic!("no listed name for {other:?}"),
    }
}

/// What `CALL grafeo.procedures()` lists: name to (writes, external_access).
fn listing(db: &GrafeoDB) -> BTreeMap<String, (bool, String)> {
    db.execute(
        "CALL grafeo.procedures() YIELD name, writes, external_access \
         RETURN name, writes, external_access",
    )
    .unwrap()
    .rows()
    .iter()
    .map(|row| match (&row[0], &row[1], &row[2]) {
        (Value::String(name), Value::Bool(writes), Value::String(access)) => {
            (name.to_string(), (*writes, access.to_string()))
        }
        other => panic!("a listed row of name, BOOLEAN and string: {other:?}"),
    })
    .collect()
}

#[test]
fn the_listing_has_the_two_columns_after_the_four() {
    let db = GrafeoDB::new_in_memory();
    let result = db.execute("CALL grafeo.procedures()").unwrap();
    assert_eq!(
        result.columns,
        [
            "name",
            "description",
            "parameters",
            "output_columns",
            "writes",
            "external_access"
        ]
    );
}

/// Every procedure of the registry is listed, with what it declares, and nothing else is.
#[test]
fn the_listing_says_what_each_procedure_declares() {
    let db = GrafeoDB::new_in_memory();
    let listed = listing(&db);
    let registry = BuiltinProcedures::new().list();
    assert_eq!(
        listed.keys().cloned().collect::<Vec<_>>(),
        registry
            .iter()
            .map(|info| info.name.clone())
            .collect::<Vec<_>>()
    );
    for info in &registry {
        assert_eq!(
            listed[&info.name],
            (info.writes, listed_access(info.external_access).to_string()),
            "{}",
            info.name
        );
    }
}

/// The status a ReadOnly session's write is refused with.
fn role_denial(db: &GrafeoDB) -> String {
    let error = db
        .session_with_role(Role::ReadOnly)
        .execute("INSERT (:Denied)")
        .expect_err("a ReadOnly session may not write");
    GqlStatus::from(&error).as_str().to_string()
}

fn status(result: &Result<(), Error>) -> Option<String> {
    result
        .as_ref()
        .err()
        .map(|error| GqlStatus::from(error).as_str().to_string())
}

fn names_external_access(result: &Result<(), Error>) -> bool {
    result
        .as_ref()
        .is_err_and(|error| error.to_string().contains("external access"))
}

/// For every procedure: refused on a ReadOnly session (which may reach outside, so only the
/// role refuses) exactly when the listing says it writes; refused without external access
/// exactly when the listing says it reaches outside. The refusals come before the arguments
/// are checked, so a call with none tells.
#[test]
fn call_enforces_what_the_listing_says() {
    let db = GrafeoDB::new_in_memory();
    let denial = role_denial(&db);
    assert_eq!(denial, "42501", "the role denial's status");
    let reader = db.session_with_identity(
        Identity::new("reader", [Role::ReadOnly]).with_external_access(true),
    );
    let inside = db.session_with_role(Role::ReadWrite);
    let mut wrong = Vec::new();
    for (name, (writes, access)) in listing(&db) {
        let call = format!("CALL {name}()");

        let as_reader = reader.execute(&call).map(|_| ());
        let refused_for_the_role = status(&as_reader).as_deref() == Some(denial.as_str())
            && !names_external_access(&as_reader);
        if refused_for_the_role != writes {
            wrong.push(format!(
                "{name}: listed writes {writes}, a ReadOnly session got {as_reader:?}"
            ));
        }
        if writes {
            let error = as_reader.as_ref().unwrap_err();
            if error.error_code().as_str() != "GRAFEO-Q007" {
                wrong.push(format!("{name}: the role refusal is {error}"));
            }
        }

        let without_access = inside.execute(&call).map(|_| ());
        let refused_for_access = names_external_access(&without_access);
        if refused_for_access {
            let error = without_access.as_ref().unwrap_err();
            if error.error_code().as_str() != "GRAFEO-Q008"
                || GqlStatus::from(error).as_str() != "42501"
            {
                wrong.push(format!("{name}: the access refusal is {error}"));
            }
        }
        if refused_for_access != (access != "none") {
            wrong.push(format!(
                "{name}: listed external_access {access}, a session without it got {without_access:?}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The two known kinds keep their declaration: the load procedures write and read files; the
/// labels listing does neither.
#[test]
fn the_load_procedures_write_and_read_outside() {
    let db = GrafeoDB::new_in_memory();
    let listed = listing(&db);
    for load in ["grafeo.load_nodes_csv", "grafeo.load_edges_csv"] {
        let (writes, access) = &listed[load];
        assert!(*writes, "{load} writes");
        assert_eq!(access, "reads", "{load} reads a file");
    }
    assert_eq!(listed["grafeo.labels"], (false, "none".to_string()));
}

/// A refusal of permission is 42501, told apart by its code: GRAFEO-Q007 for a write the role
/// refuses, GRAFEO-Q008 for external access refused; a mistake in the query stays what it was,
/// an unknown variable GRAFEO-Q002 with 42002 (invalid reference).
#[test]
fn permission_refusals_have_their_own_codes() {
    let db = GrafeoDB::new_in_memory();
    let write = db
        .session_with_role(Role::ReadOnly)
        .execute("INSERT (:Denied)")
        .expect_err("a ReadOnly session may not write");
    assert_eq!(
        (
            write.error_code().as_str(),
            GqlStatus::from(&write).as_str()
        ),
        ("GRAFEO-Q007", "42501"),
        "{write}"
    );
    let outside = db
        .session_with_role(Role::ReadWrite)
        .execute("CALL grafeo.load_nodes_csv({path: 'people.csv', labels: ['P']})")
        .expect_err("no external access");
    assert_eq!(
        (
            outside.error_code().as_str(),
            GqlStatus::from(&outside).as_str()
        ),
        ("GRAFEO-Q008", "42501"),
        "{outside}"
    );
    let unknown = db
        .execute("RETURN nobody.name")
        .expect_err("an unknown variable");
    assert_eq!(
        (
            unknown.error_code().as_str(),
            GqlStatus::from(&unknown).as_str()
        ),
        ("GRAFEO-Q002", "42002"),
        "{unknown}"
    );
}
