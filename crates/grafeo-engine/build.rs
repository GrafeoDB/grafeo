//! Sets the `query_language` cfg when the build has a query language, so
//! the code only queries use is compiled only where a query can run, and the
//! `lpg_language` cfg when it has one that queries the property graph (all
//! but SPARQL), for the code only those use (the LPG planner), and the
//! `rdf_language` cfg when it has the triple store and a language that
//! queries it (SPARQL, GraphQL), for the code that processes those queries,
//! and the `stored_procedures` cfg when it has GQL, for the code that runs a
//! stored procedure's GQL body (`CREATE PROCEDURE` needs only GQL, and so does
//! its `CALL`, whose procedure registry is in every build).

fn main() {
    // What this prints depends only on the features: without this line Cargo
    // reruns it, and so rebuilds the crate, whenever any file of the package
    // changes, a test file too.
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(query_language)");
    println!("cargo::rustc-check-cfg=cfg(lpg_language)");
    println!("cargo::rustc-check-cfg=cfg(rdf_language)");
    println!("cargo::rustc-check-cfg=cfg(stored_procedures)");
    let enabled = |language: &str| std::env::var_os(format!("CARGO_FEATURE_{language}")).is_some();
    let lpg_languages = ["GQL", "CYPHER", "GREMLIN", "GRAPHQL", "SQL_PGQ"];
    if lpg_languages.iter().any(|language| enabled(language)) {
        println!("cargo::rustc-cfg=lpg_language");
    }
    if enabled("SPARQL") || lpg_languages.iter().any(|language| enabled(language)) {
        println!("cargo::rustc-cfg=query_language");
    }
    if enabled("TRIPLE_STORE") && (enabled("SPARQL") || enabled("GRAPHQL")) {
        println!("cargo::rustc-cfg=rdf_language");
    }
    if enabled("GQL") {
        println!("cargo::rustc-cfg=stored_procedures");
    }
}
