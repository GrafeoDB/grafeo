//! Function dispatch per row (#540), before and after. Not committed.
use grafeo_engine::GrafeoDB;
use std::time::Instant;

#[test]
fn dispatch() {
    let db = GrafeoDB::new_in_memory();
    db.execute("UNWIND range(1, 200000) AS i INSERT (:P {s: 'Amsterdam', x: i})")
        .unwrap();
    for (label, q) in [
        ("n.s", "MATCH (n:P) RETURN n.s"),
        ("abs(n.x)", "MATCH (n:P) RETURN abs(n.x)"),
        ("toUpper(n.s)", "MATCH (n:P) RETURN toUpper(n.s)"),
        (
            "year(date(...))",
            "MATCH (n:P) RETURN year(date('2026-10-17'))",
        ),
        (
            "WHERE toUpper",
            "MATCH (n:P) WHERE toUpper(n.s) = 'AMSTERDAM' RETURN count(*)",
        ),
    ] {
        db.execute(q).unwrap();
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            db.execute(q).unwrap();
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        eprintln!(
            "{label:18} {best:8.2} ms  {:.0} ns/row",
            best * 1e6 / 200000.0
        );
    }
}
