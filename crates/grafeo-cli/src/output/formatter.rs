//! Shared formatting utilities for CLI output.

use std::io::{self, Write};

use grafeo_engine::memory_usage::MemoryUsage;

/// Writes a hierarchical memory breakdown to `writer`.
///
/// Mirrors the structure of `db.memory_usage()`: total on top, then one block
/// per component (store, indexes, MVCC, caches, string pool, buffer manager,
/// RDF, CDC). Zero-valued or feature-disabled components are skipped so
/// default embedded builds stay terse.
///
/// # Errors
///
/// Returns any I/O error produced by the underlying writer.
pub fn format_memory<W: Write>(writer: &mut W, usage: &MemoryUsage) -> io::Result<()> {
    writeln!(
        writer,
        "Total:           {}",
        format_bytes(usage.total_bytes)
    )?;

    if usage.store.total_bytes > 0 {
        writeln!(writer)?;
        writeln!(
            writer,
            "Store:           {}",
            format_bytes(usage.store.total_bytes)
        )?;
        if usage.store.nodes_bytes > 0 {
            writeln!(
                writer,
                "  nodes:         {}",
                format_bytes(usage.store.nodes_bytes)
            )?;
        }
        if usage.store.edges_bytes > 0 {
            writeln!(
                writer,
                "  edges:         {}",
                format_bytes(usage.store.edges_bytes)
            )?;
        }
        if usage.store.node_properties_bytes > 0 {
            writeln!(
                writer,
                "  node props:    {}",
                format_bytes(usage.store.node_properties_bytes)
            )?;
        }
        if usage.store.edge_properties_bytes > 0 {
            writeln!(
                writer,
                "  edge props:    {}",
                format_bytes(usage.store.edge_properties_bytes)
            )?;
        }
    }

    if usage.indexes.total_bytes > 0 {
        writeln!(writer)?;
        writeln!(
            writer,
            "Indexes:         {}",
            format_bytes(usage.indexes.total_bytes)
        )?;
        for idx in &usage.indexes.vector_indexes {
            writeln!(
                writer,
                "  vector[{}]: {} ({} items)",
                idx.name,
                format_bytes(idx.bytes),
                idx.item_count
            )?;
        }
        for idx in &usage.indexes.text_indexes {
            writeln!(
                writer,
                "  text[{}]: {} ({} items)",
                idx.name,
                format_bytes(idx.bytes),
                idx.item_count
            )?;
        }
    }

    if usage.mvcc.total_bytes > 0 {
        writeln!(writer)?;
        writeln!(
            writer,
            "MVCC:            {}",
            format_bytes(usage.mvcc.total_bytes)
        )?;
        if usage.mvcc.average_chain_depth > 0.0 {
            writeln!(
                writer,
                "  avg chain:     {:.2}",
                usage.mvcc.average_chain_depth
            )?;
        }
    }

    if usage.caches.total_bytes > 0 || usage.caches.cached_plan_count > 0 {
        writeln!(writer)?;
        writeln!(
            writer,
            "Caches:          {} ({} plans)",
            format_bytes(usage.caches.total_bytes),
            usage.caches.cached_plan_count
        )?;
    }

    if usage.string_pool.total_bytes > 0 {
        writeln!(writer)?;
        writeln!(
            writer,
            "String pool:     {}",
            format_bytes(usage.string_pool.total_bytes)
        )?;
    }

    if usage.buffer_manager.allocated_bytes > 0 || usage.buffer_manager.budget_bytes > 0 {
        writeln!(writer)?;
        writeln!(
            writer,
            "Buffer manager:  {} / {} budget",
            format_bytes(usage.buffer_manager.allocated_bytes),
            format_bytes(usage.buffer_manager.budget_bytes)
        )?;
    }

    if !usage.rdf.is_empty() {
        writeln!(writer)?;
        writeln!(
            writer,
            "RDF:             {} ({} triples, {} named graphs)",
            format_bytes(usage.rdf.total_bytes),
            usage.rdf.triple_count,
            usage.rdf.named_graph_count
        )?;
        if usage.rdf.term_dictionary_bytes > 0 {
            writeln!(
                writer,
                "  term dict:     {}",
                format_bytes(usage.rdf.term_dictionary_bytes)
            )?;
        }
        if usage.rdf.ring_index_bytes > 0 {
            writeln!(
                writer,
                "  ring index:    {}",
                format_bytes(usage.rdf.ring_index_bytes)
            )?;
        }
    }

    if !usage.cdc.is_empty() {
        writeln!(writer)?;
        writeln!(
            writer,
            "CDC:             {} ({} events, {} entities)",
            format_bytes(usage.cdc.total_bytes),
            usage.cdc.event_count,
            usage.cdc.entity_count
        )?;
    }

    Ok(())
}

/// Prints a hierarchical memory breakdown to stdout.
///
/// Thin wrapper over [`format_memory`] for the REPL call site. Swallows
/// pipe-closed errors (e.g., `grafeo ... | head`) instead of propagating.
pub fn print_memory(usage: &MemoryUsage) {
    let mut stdout = io::stdout().lock();
    let _ = format_memory(&mut stdout, usage);
}

/// Format bytes as a human-readable string.
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} bytes")
    }
}

/// Format a duration in milliseconds as a human-readable string.
pub fn format_duration_ms(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else if ms >= 1.0 {
        format!("{:.1}ms", ms)
    } else {
        format!("{:.0}us", ms * 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_engine::memory_usage::NamedMemory;

    #[test]
    fn test_format_bytes_bytes() {
        assert_eq!(format_bytes(0), "0 bytes");
        assert_eq!(format_bytes(1), "1 bytes");
        assert_eq!(format_bytes(512), "512 bytes");
        assert_eq!(format_bytes(1023), "1023 bytes");
    }

    #[test]
    fn test_format_bytes_kilobytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(10240), "10.00 KB");
        assert_eq!(format_bytes(1024 * 1024 - 1), "1024.00 KB");
    }

    #[test]
    fn test_format_bytes_megabytes() {
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 5), "5.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 100), "100.00 MB");
    }

    #[test]
    fn test_format_bytes_gigabytes() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_bytes(1024 * 1024 * 1024 * 2), "2.00 GB");
        assert_eq!(
            format_bytes(1024 * 1024 * 1024 + 512 * 1024 * 1024),
            "1.50 GB"
        );
    }

    #[test]
    fn test_format_duration_microseconds() {
        assert_eq!(format_duration_ms(0.5), "500us");
        assert_eq!(format_duration_ms(0.001), "1us");
    }

    #[test]
    fn test_format_duration_milliseconds() {
        assert_eq!(format_duration_ms(1.0), "1.0ms");
        assert_eq!(format_duration_ms(42.5), "42.5ms");
        assert_eq!(format_duration_ms(999.9), "999.9ms");
    }

    #[test]
    fn test_format_duration_seconds() {
        assert_eq!(format_duration_ms(1000.0), "1.00s");
        assert_eq!(format_duration_ms(2500.0), "2.50s");
    }

    fn render(usage: &MemoryUsage) -> String {
        let mut buf = Vec::new();
        format_memory(&mut buf, usage).expect("in-memory writer never fails");
        String::from_utf8(buf).expect("output is UTF-8")
    }

    #[test]
    fn empty_usage_prints_only_total() {
        let out = render(&MemoryUsage::default());
        assert_eq!(out, "Total:           0 bytes\n");
    }

    #[test]
    fn total_is_always_the_first_line() {
        let mut usage = MemoryUsage::default();
        usage.total_bytes = 4096;
        usage.rdf.total_bytes = 4096;
        usage.rdf.triple_count = 10;
        usage.rdf.triples_and_indexes_bytes = 4096;
        let out = render(&usage);
        let first = out.lines().next().unwrap();
        assert!(
            first.starts_with("Total:"),
            "expected first line to start with 'Total:', got {first:?}"
        );
    }

    #[test]
    fn store_section_only_emits_populated_subfields() {
        // nodes_bytes and edges_bytes set; property bytes zero: only those two sub-lines.
        let mut usage = MemoryUsage::default();
        usage.store.total_bytes = 3072;
        usage.store.nodes_bytes = 2048;
        usage.store.edges_bytes = 1024;
        let out = render(&usage);
        assert!(out.contains("Store:           3.00 KB"));
        assert!(out.contains("  nodes:         2.00 KB"));
        assert!(out.contains("  edges:         1.00 KB"));
        assert!(
            !out.contains("node props"),
            "should not emit node-props line when zero"
        );
        assert!(
            !out.contains("edge props"),
            "should not emit edge-props line when zero"
        );
    }

    #[test]
    fn rdf_section_hidden_when_empty() {
        let usage = store_of_one_kilobyte();
        let out = render(&usage);
        assert!(
            !out.contains("RDF:"),
            "RDF: line must not appear for empty RDF memory, got:\n{out}"
        );
    }

    #[test]
    fn rdf_section_reports_triple_and_graph_counts() {
        let mut usage = MemoryUsage::default();
        usage.rdf.total_bytes = 5000;
        usage.rdf.triple_count = 42;
        usage.rdf.triples_and_indexes_bytes = 5000;
        usage.rdf.named_graph_count = 3;
        let out = render(&usage);
        // Specific shape: "RDF:             <bytes> (N triples, M named graphs)"
        assert!(
            out.contains("42 triples"),
            "triple count not embedded in RDF line: {out}"
        );
        assert!(
            out.contains("3 named graphs"),
            "named-graph count not embedded in RDF line: {out}"
        );
    }

    #[test]
    fn rdf_sub_lines_gate_on_nonzero_values() {
        let mut usage = MemoryUsage::default();
        usage.rdf.total_bytes = 100;
        usage.rdf.triple_count = 1;
        usage.rdf.triples_and_indexes_bytes = 100;
        let without_sub = render(&usage);
        assert!(!without_sub.contains("term dict"));
        assert!(!without_sub.contains("ring index"));

        usage.rdf.term_dictionary_bytes = 200;
        usage.rdf.ring_index_bytes = 300;
        let with_sub = render(&usage);
        assert!(with_sub.contains("  term dict:     200 bytes"));
        assert!(with_sub.contains("  ring index:    300 bytes"));
    }

    #[test]
    fn cdc_section_hidden_when_empty() {
        let usage = store_of_one_kilobyte();
        let out = render(&usage);
        assert!(!out.contains("CDC:"));
    }

    #[test]
    fn cdc_section_reports_event_and_entity_counts() {
        let mut usage = MemoryUsage::default();
        usage.cdc.total_bytes = 2048;
        usage.cdc.event_count = 7;
        usage.cdc.entity_count = 4;
        let out = render(&usage);
        assert!(out.contains("7 events"));
        assert!(out.contains("4 entities"));
    }

    #[test]
    fn index_entries_include_name_and_item_count() {
        let mut usage = MemoryUsage::default();
        usage.indexes.total_bytes = 4096;
        usage.indexes.vector_indexes = vec![NamedMemory {
            name: "person.embedding".to_string(),
            bytes: 3072,
            item_count: 150,
        }];
        usage.indexes.text_indexes = vec![NamedMemory {
            name: "person.bio".to_string(),
            bytes: 1024,
            item_count: 50,
        }];
        let out = render(&usage);
        assert!(
            out.contains("vector[person.embedding]"),
            "vector index name missing: {out}"
        );
        assert!(
            out.contains("(150 items)"),
            "vector item count missing: {out}"
        );
        assert!(
            out.contains("text[person.bio]"),
            "text index name missing: {out}"
        );
        assert!(out.contains("(50 items)"), "text item count missing: {out}");
    }

    #[test]
    fn caches_line_appears_with_plan_count_even_when_bytes_zero() {
        // A registry may report cached_plan_count > 0 before any bytes are
        // accounted for (e.g., Arc-shared entries). The line must still appear.
        let mut usage = MemoryUsage::default();
        usage.caches.total_bytes = 0;
        usage.caches.cached_plan_count = 12;
        let out = render(&usage);
        assert!(
            out.contains("Caches:"),
            "Caches line missing when plan_count > 0: {out}"
        );
        assert!(out.contains("(12 plans)"));
    }

    #[test]
    fn buffer_manager_line_gates_on_allocated_or_budget() {
        let hidden = render(&MemoryUsage::default());
        assert!(!hidden.contains("Buffer manager"));

        let mut usage = MemoryUsage::default();
        usage.buffer_manager.budget_bytes = 1024;
        let with_budget = render(&usage);
        assert!(with_budget.contains("Buffer manager:"));
        assert!(with_budget.contains("0 bytes / 1.00 KB budget"));
    }

    #[test]
    fn mvcc_avg_chain_line_gates_on_nonzero() {
        let mut usage = MemoryUsage::default();
        usage.mvcc.total_bytes = 512;
        usage.mvcc.average_chain_depth = 0.0;
        let zero_chain = render(&usage);
        assert!(zero_chain.contains("MVCC:"));
        assert!(!zero_chain.contains("avg chain"));

        usage.mvcc.average_chain_depth = 2.75;
        let with_chain = render(&usage);
        assert!(with_chain.contains("  avg chain:     2.75"));
    }

    #[test]
    fn string_pool_line_gates_on_nonzero() {
        let hidden = render(&MemoryUsage::default());
        assert!(!hidden.contains("String pool"));

        let mut usage = MemoryUsage::default();
        usage.string_pool.total_bytes = 2048;
        let shown = render(&usage);
        assert!(shown.contains("String pool:     2.00 KB"));
    }

    /// A store of 1 KB of nodes, and nothing else.
    fn store_of_one_kilobyte() -> MemoryUsage {
        let mut usage = MemoryUsage::default();
        usage.total_bytes = 1024;
        usage.store.total_bytes = 1024;
        usage.store.nodes_bytes = 1024;
        usage
    }
}
