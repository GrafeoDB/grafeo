//! Shared adversarial topology for optional-match path controls.

use std::collections::BTreeSet;

use grafeo_engine::GrafeoDB;

/// The shared diamond, cycle, self-loop, parallel edges and unreachable component.
pub const EDGES: &[(&str, &str, &str)] = &[
    ("s", "a", "REL"),
    ("s", "b", "REL"),
    ("a", "d", "REL"),
    ("b", "d", "REL"),
    ("d", "e", "REL"),
    ("e", "f", "REL"),
    ("f", "e", "REL"),
    ("b", "g", "REL"),
    ("g", "g", "REL"),
    ("a", "h", "REL"),
    ("h", "d", "REL"),
    ("s", "x", "OTHER"),
    ("s", "x", "OTHER"),
    ("u", "v", "REL"),
];

pub fn node_ids() -> BTreeSet<&'static str> {
    EDGES
        .iter()
        .flat_map(|(from, to, _)| [*from, *to])
        .collect()
}

pub fn adversarial_path_graph(indexed: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for name in node_ids() {
        db.execute(&format!("INSERT (:Node {{id: '{name}'}})"))
            .unwrap();
    }
    if indexed {
        db.create_property_index("id").unwrap();
    }
    for (slot, (from, to, kind)) in EDGES.iter().enumerate() {
        db.execute(&format!(
            "MATCH (a:Node {{id: '{from}'}}), (b:Node {{id: '{to}'}}) \
             INSERT (a)-[:{kind} {{cost: 1, slot: {slot}}}]->(b)"
        ))
        .unwrap();
    }
    db
}

/// The same extra parallel REL edge used by the original free-endpoint oracle.
pub fn add_parallel_rel(db: &GrafeoDB) {
    let slot = EDGES.len();
    db.execute(&format!(
        "MATCH (s:Node {{id: 's'}}), (a:Node {{id: 'a'}}) \
         INSERT (s)-[:REL {{cost: 1, slot: {slot}}}]->(a)"
    ))
    .unwrap();
}
