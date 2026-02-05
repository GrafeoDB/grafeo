//! Database schema command.

use std::path::Path;

use anyhow::Result;
use comfy_table::Cell;
use grafeo_engine::{GrafeoDB, SchemaInfo};
use serde::Serialize;

use crate::OutputFormat;
use crate::output::{self, Format};

/// Schema output for LPG mode.
#[derive(Serialize)]
struct LpgSchemaOutput {
    mode: String,
    labels: Vec<LabelOutput>,
    edge_types: Vec<EdgeTypeOutput>,
    property_keys: Vec<String>,
}

/// Label information.
#[derive(Serialize)]
struct LabelOutput {
    name: String,
    count: usize,
}

/// Edge type information.
#[derive(Serialize)]
struct EdgeTypeOutput {
    name: String,
    count: usize,
}

/// Schema output for RDF mode.
#[derive(Serialize)]
struct RdfSchemaOutput {
    mode: String,
    predicates: Vec<PredicateOutput>,
    named_graphs: Vec<String>,
    subject_count: usize,
    object_count: usize,
}

/// Predicate information.
#[derive(Serialize)]
struct PredicateOutput {
    iri: String,
    count: usize,
}

/// Run the schema command.
pub fn run(path: &Path, format: OutputFormat, quiet: bool) -> Result<()> {
    let db = GrafeoDB::open(path)?;
    let schema = db.schema();

    let fmt: Format = format.into();

    match schema {
        SchemaInfo::Lpg(lpg) => {
            let output = LpgSchemaOutput {
                mode: "LPG".to_string(),
                labels: lpg
                    .labels
                    .iter()
                    .map(|l| LabelOutput {
                        name: l.name.clone(),
                        count: l.count,
                    })
                    .collect(),
                edge_types: lpg
                    .edge_types
                    .iter()
                    .map(|e| EdgeTypeOutput {
                        name: e.name.clone(),
                        count: e.count,
                    })
                    .collect(),
                property_keys: lpg.property_keys,
            };

            match fmt {
                Format::Json => {
                    if !quiet {
                        println!("{}", serde_json::to_string_pretty(&output)?);
                    }
                }
                Format::Table => {
                    if !quiet {
                        println!("Mode: LPG (Labeled Property Graph)\n");

                        // Labels table
                        let mut table = output::create_table();
                        output::add_header(&mut table, &["Label", "Count"]);
                        for label in &output.labels {
                            table.add_row(vec![Cell::new(&label.name), Cell::new(label.count)]);
                        }
                        println!("{table}\n");

                        // Edge types table
                        let mut table = output::create_table();
                        output::add_header(&mut table, &["Edge Type", "Count"]);
                        for edge_type in &output.edge_types {
                            table.add_row(vec![
                                Cell::new(&edge_type.name),
                                Cell::new(edge_type.count),
                            ]);
                        }
                        println!("{table}\n");

                        // Property keys table
                        let mut table = output::create_table();
                        output::add_header(&mut table, &["Property Keys"]);
                        for key in &output.property_keys {
                            table.add_row(vec![Cell::new(key)]);
                        }
                        println!("{table}");
                    }
                }
            }
        }
        SchemaInfo::Rdf(rdf) => {
            let output = RdfSchemaOutput {
                mode: "RDF".to_string(),
                predicates: rdf
                    .predicates
                    .iter()
                    .map(|p| PredicateOutput {
                        iri: p.iri.clone(),
                        count: p.count,
                    })
                    .collect(),
                named_graphs: rdf.named_graphs,
                subject_count: rdf.subject_count,
                object_count: rdf.object_count,
            };

            match fmt {
                Format::Json => {
                    if !quiet {
                        println!("{}", serde_json::to_string_pretty(&output)?);
                    }
                }
                Format::Table => {
                    if !quiet {
                        println!("Mode: RDF (Triple Store)\n");

                        // Stats
                        println!(
                            "Subjects: {}, Objects: {}\n",
                            output.subject_count, output.object_count
                        );

                        // Predicates table
                        let mut table = output::create_table();
                        output::add_header(&mut table, &["Predicate", "Count"]);
                        for pred in &output.predicates {
                            table.add_row(vec![Cell::new(&pred.iri), Cell::new(pred.count)]);
                        }
                        println!("{table}\n");

                        // Named graphs table
                        let mut table = output::create_table();
                        output::add_header(&mut table, &["Named Graphs"]);
                        for graph in &output.named_graphs {
                            table.add_row(vec![Cell::new(graph)]);
                        }
                        println!("{table}");
                    }
                }
            }
        }
    }

    Ok(())
}
