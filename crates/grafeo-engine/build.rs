//! Sets the `query_language` cfg when the build has a query language, so
//! the code only queries use is compiled only where a query can run.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(query_language)");
    let languages = ["GQL", "CYPHER", "SPARQL", "GREMLIN", "GRAPHQL", "SQL_PGQ"];
    if languages
        .iter()
        .any(|language| std::env::var_os(format!("CARGO_FEATURE_{language}")).is_some())
    {
        println!("cargo::rustc-cfg=query_language");
    }
}
