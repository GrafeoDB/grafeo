//! SHACL constraint evaluation.
//!
//! Each constraint type produces zero or more `ValidationResult` entries
//! when the constraint is violated.

use std::cmp::Ordering;
use std::collections::HashSet;

use crate::graph::rdf::overlay::RdfView;
use crate::graph::rdf::{Literal, RdfStore, Term, TriplePattern};

use super::path::evaluate_path;
use super::report::ValidationResult;
use super::shape::{Constraint, NodeKindValue, PropertyPath, SH, Severity, Shape};

/// Context for constraint evaluation.
pub struct EvalContext<'a> {
    /// The focus node being validated.
    pub focus_node: &'a Term,
    /// The shape that owns this constraint.
    pub shape: &'a Shape,
    /// The property path (for property shapes, None for node shapes).
    pub path: Option<&'a PropertyPath>,
    /// The data graph.
    pub data_graph: &'a RdfStore,
    /// All parsed shapes (for recursive references).
    pub all_shapes: &'a [Shape],
    /// Visited (focus_node, shape_id) pairs for cycle detection.
    pub visited: &'a mut HashSet<(Term, Term)>,
}

/// Evaluates a single constraint against the given value nodes.
///
/// Returns validation results for each violation found.
pub fn evaluate_constraint(
    constraint: &Constraint,
    value_nodes: &[Term],
    ctx: &mut EvalContext<'_>,
) -> Vec<ValidationResult> {
    match constraint {
        // -- Value type --
        Constraint::Class(class) => eval_class(class, value_nodes, ctx),
        Constraint::Datatype(dt) => eval_datatype(dt, value_nodes, ctx),
        Constraint::NodeKind(kind) => eval_node_kind(*kind, value_nodes, ctx),

        // -- Cardinality --
        Constraint::MinCount(n) => eval_min_count(*n, value_nodes, ctx),
        Constraint::MaxCount(n) => eval_max_count(*n, value_nodes, ctx),

        // -- Value range --
        Constraint::MinExclusive(bound) => {
            eval_range(bound, value_nodes, ctx, "minExclusive", |ord| {
                ord == Ordering::Greater
            })
        }
        Constraint::MaxExclusive(bound) => {
            eval_range(bound, value_nodes, ctx, "maxExclusive", |ord| {
                ord == Ordering::Less
            })
        }
        Constraint::MinInclusive(bound) => {
            eval_range(bound, value_nodes, ctx, "minInclusive", |ord| {
                ord != Ordering::Less
            })
        }
        Constraint::MaxInclusive(bound) => {
            eval_range(bound, value_nodes, ctx, "maxInclusive", |ord| {
                ord != Ordering::Greater
            })
        }

        // -- String --
        Constraint::MinLength(n) => eval_min_length(*n, value_nodes, ctx),
        Constraint::MaxLength(n) => eval_max_length(*n, value_nodes, ctx),
        Constraint::Pattern { pattern, flags } => {
            eval_pattern(pattern, flags.as_deref(), value_nodes, ctx)
        }
        Constraint::LanguageIn(langs) => eval_language_in(langs, value_nodes, ctx),
        Constraint::UniqueLang => eval_unique_lang(value_nodes, ctx),

        // -- Property pair --
        Constraint::Equals(path_iri) => eval_equals(path_iri, value_nodes, ctx),
        Constraint::Disjoint(path_iri) => eval_disjoint(path_iri, value_nodes, ctx),
        Constraint::LessThan(path_iri) => eval_less_than(path_iri, value_nodes, ctx, false),
        Constraint::LessThanOrEquals(path_iri) => eval_less_than(path_iri, value_nodes, ctx, true),

        // -- Logical --
        Constraint::Not(shape) => eval_not(shape, ctx),
        Constraint::And(shapes) => eval_and(shapes, ctx),
        Constraint::Or(shapes) => eval_or(shapes, ctx),
        Constraint::Xone(shapes) => eval_xone(shapes, ctx),

        // -- Shape-based --
        Constraint::ShapeNode(shape) => eval_shape_node(shape, value_nodes, ctx),
        Constraint::QualifiedValueShape {
            shape,
            min_count,
            max_count,
            disjoint,
        } => eval_qualified(shape, *min_count, *max_count, *disjoint, value_nodes, ctx),

        // -- Other --
        Constraint::Closed { ignored_properties } => eval_closed(ignored_properties, ctx),
        Constraint::HasValue(value) => eval_has_value(value, value_nodes, ctx),
        Constraint::In(allowed) => eval_in(allowed, value_nodes, ctx),

        // -- SPARQL (handled by engine, not core) --
        Constraint::Sparql(_) => Vec::new(),
    }
}

// =========================================================================
// Result builder helper
// =========================================================================

fn result(
    ctx: &EvalContext<'_>,
    component: &str,
    value: Option<Term>,
    message: String,
) -> ValidationResult {
    ValidationResult {
        focus_node: ctx.focus_node.clone(),
        source_constraint_component: format!("{}{component}ConstraintComponent", SH::NS),
        source_shape: ctx.shape.id().clone(),
        value,
        result_path: ctx.path.cloned(),
        severity: ctx.shape.severity(),
        message: Some(message),
    }
}

// =========================================================================
// Value type constraints
// =========================================================================

fn eval_class(class: &Term, value_nodes: &[Term], ctx: &EvalContext<'_>) -> Vec<ValidationResult> {
    let rdf_type = Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    let mut results = Vec::new();
    for vn in value_nodes {
        let has_type = !ctx
            .data_graph
            .find(
                &RdfView::Latest,
                &TriplePattern {
                    subject: Some(vn.clone()),
                    predicate: Some(rdf_type.clone()),
                    object: Some(class.clone()),
                },
            )
            .is_empty();
        if !has_type {
            results.push(result(
                ctx,
                "Class",
                Some(vn.clone()),
                format!("Value {vn} is not an instance of {class}"),
            ));
        }
    }
    results
}

fn eval_datatype(dt: &Term, value_nodes: &[Term], ctx: &EvalContext<'_>) -> Vec<ValidationResult> {
    let expected = match dt {
        Term::Iri(iri) => iri.as_str(),
        _ => return Vec::new(),
    };
    let mut results = Vec::new();
    for vn in value_nodes {
        let ok = match vn {
            Term::Literal(lit) => lit.datatype() == expected,
            _ => false,
        };
        if !ok {
            results.push(result(
                ctx,
                "Datatype",
                Some(vn.clone()),
                format!("Value {vn} does not have datatype {expected}"),
            ));
        }
    }
    results
}

fn eval_node_kind(
    kind: NodeKindValue,
    value_nodes: &[Term],
    ctx: &EvalContext<'_>,
) -> Vec<ValidationResult> {
    let mut results = Vec::new();
    for vn in value_nodes {
        let ok = match kind {
            NodeKindValue::Iri => vn.is_iri(),
            NodeKindValue::BlankNode => vn.is_blank_node(),
            NodeKindValue::Literal => vn.is_literal(),
            NodeKindValue::BlankNodeOrIri => vn.is_blank_node() || vn.is_iri(),
            NodeKindValue::BlankNodeOrLiteral => vn.is_blank_node() || vn.is_literal(),
            NodeKindValue::IriOrLiteral => vn.is_iri() || vn.is_literal(),
        };
        if !ok {
            results.push(result(
                ctx,
                "NodeKind",
                Some(vn.clone()),
                format!("Value {vn} does not match node kind {kind:?}"),
            ));
        }
    }
    results
}

// =========================================================================
// Cardinality constraints
// =========================================================================

fn eval_min_count(n: usize, value_nodes: &[Term], ctx: &EvalContext<'_>) -> Vec<ValidationResult> {
    if value_nodes.len() < n {
        vec![result(
            ctx,
            "MinCount",
            None,
            format!("Expected at least {n} value(s), got {}", value_nodes.len()),
        )]
    } else {
        Vec::new()
    }
}

fn eval_max_count(n: usize, value_nodes: &[Term], ctx: &EvalContext<'_>) -> Vec<ValidationResult> {
    if value_nodes.len() > n {
        vec![result(
            ctx,
            "MaxCount",
            None,
            format!("Expected at most {n} value(s), got {}", value_nodes.len()),
        )]
    } else {
        Vec::new()
    }
}

// =========================================================================
// Value range constraints
// =========================================================================

fn eval_range(
    bound: &Term,
    value_nodes: &[Term],
    ctx: &EvalContext<'_>,
    name: &str,
    check: impl Fn(Ordering) -> bool,
) -> Vec<ValidationResult> {
    let mut results = Vec::new();
    for vn in value_nodes {
        match compare_terms(vn, bound) {
            Some(ord) if check(ord) => {}
            _ => {
                results.push(result(
                    ctx,
                    name,
                    Some(vn.clone()),
                    format!("Value {vn} violates {name} {bound}"),
                ));
            }
        }
    }
    results
}

/// Returns true if the literal has a numeric XSD datatype.
fn is_numeric_datatype(lit: &Literal) -> bool {
    matches!(
        lit.datatype(),
        Literal::XSD_INTEGER | Literal::XSD_DECIMAL | Literal::XSD_DOUBLE
    )
}

/// Compares two RDF terms for ordering.
///
/// Numeric literals (with numeric XSD datatypes) compare numerically,
/// string literals compare lexicographically. Returns `None` if the terms
/// are not comparable.
fn compare_terms(a: &Term, b: &Term) -> Option<Ordering> {
    match (a, b) {
        (Term::Literal(la), Term::Literal(lb)) => {
            // Only apply numeric comparison when both have numeric datatypes
            if is_numeric_datatype(la)
                && is_numeric_datatype(lb)
                && let (Some(da), Some(db)) = (la.as_double(), lb.as_double())
            {
                return da.partial_cmp(&db);
            }
            // Fall back to lexicographic comparison of values
            Some(la.value().cmp(lb.value()))
        }
        _ => None,
    }
}

// =========================================================================
// String constraints
// =========================================================================

fn term_string_len(term: &Term) -> Option<usize> {
    match term {
        Term::Literal(lit) => Some(lit.value().chars().count()),
        Term::Iri(iri) => Some(iri.as_str().chars().count()),
        _ => None,
    }
}

fn eval_min_length(n: usize, value_nodes: &[Term], ctx: &EvalContext<'_>) -> Vec<ValidationResult> {
    let mut results = Vec::new();
    for vn in value_nodes {
        if let Some(len) = term_string_len(vn)
            && len < n
        {
            results.push(result(
                ctx,
                "MinLength",
                Some(vn.clone()),
                format!("String length {len} is less than minimum {n}"),
            ));
        }
    }
    results
}

fn eval_max_length(n: usize, value_nodes: &[Term], ctx: &EvalContext<'_>) -> Vec<ValidationResult> {
    let mut results = Vec::new();
    for vn in value_nodes {
        if let Some(len) = term_string_len(vn)
            && len > n
        {
            results.push(result(
                ctx,
                "MaxLength",
                Some(vn.clone()),
                format!("String length {len} exceeds maximum {n}"),
            ));
        }
    }
    results
}

fn eval_pattern(
    pattern: &str,
    flags: Option<&str>,
    value_nodes: &[Term],
    ctx: &EvalContext<'_>,
) -> Vec<ValidationResult> {
    // Build regex pattern with flags
    let regex_pattern = if let Some(f) = flags {
        format!("(?{f}){pattern}")
    } else {
        pattern.to_string()
    };

    #[cfg(feature = "regex")]
    let re = regex::Regex::new(&regex_pattern);
    #[cfg(all(not(feature = "regex"), feature = "regex-lite"))]
    let re = regex_lite::Regex::new(&regex_pattern);
    #[cfg(all(not(feature = "regex"), not(feature = "regex-lite")))]
    let re: Result<(), String> = Err("No regex feature enabled".to_string());

    let Ok(re) = re else {
        return vec![result(
            ctx,
            "Pattern",
            None,
            format!("Invalid regex pattern: {pattern}"),
        )];
    };

    let mut results = Vec::new();
    for vn in value_nodes {
        let text = match vn {
            Term::Literal(lit) => lit.value(),
            Term::Iri(iri) => iri.as_str(),
            _ => continue,
        };
        #[cfg(any(feature = "regex", feature = "regex-lite"))]
        if !re.is_match(text) {
            results.push(result(
                ctx,
                "Pattern",
                Some(vn.clone()),
                format!("Value \"{text}\" does not match pattern \"{pattern}\""),
            ));
        }
    }
    results
}

fn eval_language_in(
    langs: &[String],
    value_nodes: &[Term],
    ctx: &EvalContext<'_>,
) -> Vec<ValidationResult> {
    let mut results = Vec::new();
    for vn in value_nodes {
        let ok = match vn {
            Term::Literal(lit) => {
                if let Some(lang) = lit.language() {
                    let lang_lower = lang.to_lowercase();
                    langs.iter().any(|allowed| {
                        let allowed_lower = allowed.to_lowercase();
                        lang_lower == allowed_lower
                            || lang_lower.starts_with(&format!("{allowed_lower}-"))
                    })
                } else {
                    false
                }
            }
            _ => false,
        };
        if !ok {
            results.push(result(
                ctx,
                "LanguageIn",
                Some(vn.clone()),
                "Language tag not in allowed list".to_string(),
            ));
        }
    }
    results
}

fn eval_unique_lang(value_nodes: &[Term], ctx: &EvalContext<'_>) -> Vec<ValidationResult> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut has_duplicate = false;
    for vn in value_nodes {
        if let Term::Literal(lit) = vn
            && let Some(lang) = lit.language()
        {
            let lang_lower = lang.to_lowercase();
            if !lang_lower.is_empty() && !seen.insert(lang_lower) {
                has_duplicate = true;
            }
        }
    }
    if has_duplicate {
        vec![result(
            ctx,
            "UniqueLang",
            None,
            "Duplicate language tags found".to_string(),
        )]
    } else {
        Vec::new()
    }
}

// =========================================================================
// Property pair constraints
// =========================================================================

fn eval_equals(
    path_iri: &Term,
    value_nodes: &[Term],
    ctx: &EvalContext<'_>,
) -> Vec<ValidationResult> {
    let comparison_path = PropertyPath::Predicate(path_iri.clone());
    let other_values: HashSet<Term> =
        evaluate_path(&comparison_path, ctx.focus_node, ctx.data_graph)
            .into_iter()
            .collect();
    let value_set: HashSet<Term> = value_nodes.iter().cloned().collect();

    let mut results = Vec::new();
    for vn in &value_set {
        if !other_values.contains(vn) {
            results.push(result(
                ctx,
                "Equals",
                Some(vn.clone()),
                format!("Value {vn} not found in {path_iri} values"),
            ));
        }
    }
    for ov in &other_values {
        if !value_set.contains(ov) {
            results.push(result(
                ctx,
                "Equals",
                Some(ov.clone()),
                format!("Value {ov} from {path_iri} not in shape values"),
            ));
        }
    }
    results
}

fn eval_disjoint(
    path_iri: &Term,
    value_nodes: &[Term],
    ctx: &EvalContext<'_>,
) -> Vec<ValidationResult> {
    let comparison_path = PropertyPath::Predicate(path_iri.clone());
    let other_values: HashSet<Term> =
        evaluate_path(&comparison_path, ctx.focus_node, ctx.data_graph)
            .into_iter()
            .collect();

    let mut results = Vec::new();
    for vn in value_nodes {
        if other_values.contains(vn) {
            results.push(result(
                ctx,
                "Disjoint",
                Some(vn.clone()),
                format!("Value {vn} also appears in {path_iri}"),
            ));
        }
    }
    results
}

fn eval_less_than(
    path_iri: &Term,
    value_nodes: &[Term],
    ctx: &EvalContext<'_>,
    or_equals: bool,
) -> Vec<ValidationResult> {
    let name = if or_equals {
        "LessThanOrEquals"
    } else {
        "LessThan"
    };
    let comparison_path = PropertyPath::Predicate(path_iri.clone());
    let other_values = evaluate_path(&comparison_path, ctx.focus_node, ctx.data_graph);

    let mut results = Vec::new();
    for vn in value_nodes {
        for ov in &other_values {
            let ok = match compare_terms(vn, ov) {
                Some(Ordering::Less) => true,
                Some(Ordering::Equal) => or_equals,
                _ => false,
            };
            if !ok {
                results.push(result(
                    ctx,
                    name,
                    Some(vn.clone()),
                    format!("Value {vn} is not {name} {ov}"),
                ));
            }
        }
    }
    results
}

// =========================================================================
// Logical constraints
// =========================================================================

fn eval_not(inner_shape: &Shape, ctx: &mut EvalContext<'_>) -> Vec<ValidationResult> {
    let inner_results = evaluate_shape_for_node(inner_shape, ctx.focus_node, ctx);
    // If the inner shape conforms (no violations), that's a violation of sh:not
    let inner_conforms = inner_results
        .iter()
        .all(|r| r.severity != Severity::Violation);
    if inner_conforms {
        vec![result(
            ctx,
            "Not",
            None,
            "Focus node conforms to shape that should not match".to_string(),
        )]
    } else {
        Vec::new()
    }
}

fn eval_and(shapes: &[Shape], ctx: &mut EvalContext<'_>) -> Vec<ValidationResult> {
    let mut results = Vec::new();
    for shape in shapes {
        let inner = evaluate_shape_for_node(shape, ctx.focus_node, ctx);
        let conforms = inner.iter().all(|r| r.severity != Severity::Violation);
        if !conforms {
            results.push(result(
                ctx,
                "And",
                None,
                "Focus node does not conform to all shapes in sh:and".to_string(),
            ));
            break;
        }
    }
    results
}

fn eval_or(shapes: &[Shape], ctx: &mut EvalContext<'_>) -> Vec<ValidationResult> {
    for shape in shapes {
        let inner = evaluate_shape_for_node(shape, ctx.focus_node, ctx);
        let conforms = inner.iter().all(|r| r.severity != Severity::Violation);
        if conforms {
            return Vec::new();
        }
    }
    vec![result(
        ctx,
        "Or",
        None,
        "Focus node does not conform to any shape in sh:or".to_string(),
    )]
}

fn eval_xone(shapes: &[Shape], ctx: &mut EvalContext<'_>) -> Vec<ValidationResult> {
    let conforming_count = shapes
        .iter()
        .filter(|shape| {
            let inner = evaluate_shape_for_node(shape, ctx.focus_node, ctx);
            inner.iter().all(|r| r.severity != Severity::Violation)
        })
        .count();

    if conforming_count == 1 {
        Vec::new()
    } else {
        vec![result(
            ctx,
            "Xone",
            None,
            format!("Focus node conforms to {conforming_count} shapes (expected exactly 1)"),
        )]
    }
}

// =========================================================================
// Shape-based constraints
// =========================================================================

fn eval_shape_node(
    inner_shape: &Shape,
    value_nodes: &[Term],
    ctx: &mut EvalContext<'_>,
) -> Vec<ValidationResult> {
    let mut results = Vec::new();
    for vn in value_nodes {
        let inner = evaluate_shape_for_node(inner_shape, vn, ctx);
        let conforms = inner.iter().all(|r| r.severity != Severity::Violation);
        if !conforms {
            results.push(result(
                ctx,
                "Node",
                Some(vn.clone()),
                format!("Value {vn} does not conform to referenced shape"),
            ));
        }
    }
    results
}

fn eval_qualified(
    inner_shape: &Shape,
    min_count: Option<usize>,
    max_count: Option<usize>,
    disjoint: bool,
    value_nodes: &[Term],
    ctx: &mut EvalContext<'_>,
) -> Vec<ValidationResult> {
    // Collect sibling qualified shapes when disjoint is enabled
    let sibling_shapes: Vec<&Shape> = if disjoint {
        ctx.shape
            .constraints()
            .iter()
            .filter_map(|c| {
                if let Constraint::QualifiedValueShape { shape, .. } = c {
                    // Exclude the current shape (compare by id)
                    if shape.id() != inner_shape.id() {
                        return Some(shape.as_ref());
                    }
                }
                None
            })
            .collect()
    } else {
        Vec::new()
    };

    let conforming = value_nodes
        .iter()
        .filter(|vn| {
            let inner = evaluate_shape_for_node(inner_shape, vn, ctx);
            let conforms = inner.iter().all(|r| r.severity != Severity::Violation);
            if !conforms {
                return false;
            }
            // When disjoint, exclude values that also conform to a sibling shape
            if disjoint {
                for sibling in &sibling_shapes {
                    let sibling_results = evaluate_shape_for_node(sibling, vn, ctx);
                    if sibling_results
                        .iter()
                        .all(|r| r.severity != Severity::Violation)
                    {
                        return false;
                    }
                }
            }
            true
        })
        .count();

    let mut results = Vec::new();
    if let Some(min) = min_count
        && conforming < min
    {
        results.push(result(
            ctx,
            "QualifiedMinCount",
            None,
            format!("Only {conforming} value(s) conform (minimum {min})"),
        ));
    }
    if let Some(max) = max_count
        && conforming > max
    {
        results.push(result(
            ctx,
            "QualifiedMaxCount",
            None,
            format!("{conforming} value(s) conform (maximum {max})"),
        ));
    }
    results
}

// =========================================================================
// Other constraints
// =========================================================================

fn eval_closed(ignored: &[Term], ctx: &EvalContext<'_>) -> Vec<ValidationResult> {
    // Collect allowed predicates from the shape's property shapes
    let mut allowed: HashSet<Term> = HashSet::new();
    if let Shape::Node(ns) = ctx.shape {
        for ps in &ns.property_shapes {
            if let PropertyPath::Predicate(pred) = &ps.path {
                allowed.insert(pred.clone());
            }
        }
    }
    for ign in ignored {
        allowed.insert(ign.clone());
    }
    // rdf:type is always implicitly allowed
    allowed.insert(Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"));

    // Check all outgoing predicates of the focus node
    let mut results = Vec::new();
    for triple in ctx
        .data_graph
        .triples_with_subject(&RdfView::Latest, ctx.focus_node)
    {
        if !allowed.contains(triple.predicate()) {
            results.push(result(
                ctx,
                "Closed",
                Some(triple.predicate().clone()),
                format!(
                    "Predicate {} is not allowed by closed shape",
                    triple.predicate()
                ),
            ));
        }
    }
    results
}

fn eval_has_value(
    value: &Term,
    value_nodes: &[Term],
    ctx: &EvalContext<'_>,
) -> Vec<ValidationResult> {
    if value_nodes.contains(value) {
        Vec::new()
    } else {
        vec![result(
            ctx,
            "HasValue",
            None,
            format!("Required value {value} not found"),
        )]
    }
}

fn eval_in(allowed: &[Term], value_nodes: &[Term], ctx: &EvalContext<'_>) -> Vec<ValidationResult> {
    let mut results = Vec::new();
    for vn in value_nodes {
        if !allowed.contains(vn) {
            results.push(result(
                ctx,
                "In",
                Some(vn.clone()),
                format!("Value {vn} is not in the allowed list"),
            ));
        }
    }
    results
}

// =========================================================================
// Recursive shape evaluation helper
// =========================================================================

/// Evaluates a shape against a focus node, returning validation results.
///
/// Used internally for recursive constraints (sh:not, sh:and, sh:or, sh:node).
fn evaluate_shape_for_node(
    shape: &Shape,
    focus_node: &Term,
    parent_ctx: &mut EvalContext<'_>,
) -> Vec<ValidationResult> {
    // Cycle detection
    let key = (focus_node.clone(), shape.id().clone());
    if !parent_ctx.visited.insert(key.clone()) {
        // Already evaluating this (focus_node, shape) pair: treat as conforming
        return Vec::new();
    }

    let mut results = Vec::new();

    match shape {
        Shape::Node(ns) => {
            // Evaluate node-level constraints
            let mut ctx = EvalContext {
                focus_node,
                shape,
                path: None,
                data_graph: parent_ctx.data_graph,
                all_shapes: parent_ctx.all_shapes,
                visited: parent_ctx.visited,
            };
            for constraint in &ns.constraints {
                let value_nodes = vec![focus_node.clone()];
                results.extend(evaluate_constraint(constraint, &value_nodes, &mut ctx));
            }
            // Evaluate nested property shapes
            for ps in &ns.property_shapes {
                let path_values = evaluate_path(&ps.path, focus_node, parent_ctx.data_graph);
                let ps_shape = Shape::Property(ps.clone());
                let mut ps_ctx = EvalContext {
                    focus_node,
                    shape: &ps_shape,
                    path: Some(&ps.path),
                    data_graph: parent_ctx.data_graph,
                    all_shapes: parent_ctx.all_shapes,
                    visited: parent_ctx.visited,
                };
                for constraint in &ps.constraints {
                    results.extend(evaluate_constraint(constraint, &path_values, &mut ps_ctx));
                }
            }
        }
        Shape::Property(ps) => {
            let path_values = evaluate_path(&ps.path, focus_node, parent_ctx.data_graph);
            let mut ctx = EvalContext {
                focus_node,
                shape,
                path: Some(&ps.path),
                data_graph: parent_ctx.data_graph,
                all_shapes: parent_ctx.all_shapes,
                visited: parent_ctx.visited,
            };
            for constraint in &ps.constraints {
                results.extend(evaluate_constraint(constraint, &path_values, &mut ctx));
            }
        }
    }

    // Remove cycle guard so the same pair can be visited from a different path
    parent_ctx.visited.remove(&key);

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::rdf::{RdfStore, Triple};

    fn data_store() -> RdfStore {
        let store = RdfStore::new();
        let rdf_type = Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
        let person = Term::iri("http://ex.org/Person");
        let name = Term::iri("http://ex.org/name");
        let age = Term::iri("http://ex.org/age");
        let alix = Term::iri("http://ex.org/alix");

        store.insert(Triple::new(alix.clone(), rdf_type, person));
        store.insert(Triple::new(alix.clone(), name, Term::literal("Alix")));
        store.insert(Triple::new(
            alix,
            age,
            Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        store
    }

    fn dummy_shape() -> Shape {
        use super::super::shape::{NodeShape, Severity};
        Shape::Node(NodeShape {
            id: Term::iri("http://ex.org/TestShape"),
            targets: Vec::new(),
            property_shapes: Vec::new(),
            constraints: Vec::new(),
            deactivated: false,
            severity: Severity::Violation,
            messages: Vec::new(),
        })
    }

    fn make_ctx<'a>(
        focus: &'a Term,
        shape: &'a Shape,
        store: &'a RdfStore,
        visited: &'a mut HashSet<(Term, Term)>,
    ) -> EvalContext<'a> {
        EvalContext {
            focus_node: focus,
            shape,
            path: None,
            data_graph: store,
            all_shapes: &[],
            visited,
        }
    }

    #[test]
    fn class_valid() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_class(
            &Term::iri("http://ex.org/Person"),
            std::slice::from_ref(&alix),
            &ctx,
        );
        assert!(results.is_empty());
    }

    #[test]
    fn class_invalid() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_class(
            &Term::iri("http://ex.org/Animal"),
            std::slice::from_ref(&alix),
            &ctx,
        );
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn datatype_valid() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let val = Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer");
        let dt = Term::iri("http://www.w3.org/2001/XMLSchema#integer");
        let results = eval_datatype(&dt, &[val], &ctx);
        assert!(results.is_empty());
    }

    #[test]
    fn datatype_mismatch() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let val = Term::literal("hello"); // xsd:string
        let dt = Term::iri("http://www.w3.org/2001/XMLSchema#integer");
        let results = eval_datatype(&dt, &[val], &ctx);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn node_kind_iri_valid() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_node_kind(NodeKindValue::Iri, std::slice::from_ref(&alix), &ctx);
        assert!(results.is_empty());
    }

    #[test]
    fn node_kind_iri_invalid() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let val = Term::literal("hello");
        let results = eval_node_kind(NodeKindValue::Iri, &[val], &ctx);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn min_count_pass() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_min_count(1, &[Term::literal("a")], &ctx);
        assert!(results.is_empty());
    }

    #[test]
    fn min_count_fail() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_min_count(2, &[Term::literal("a")], &ctx);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn max_count_pass() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_max_count(2, &[Term::literal("a")], &ctx);
        assert!(results.is_empty());
    }

    #[test]
    fn max_count_fail() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_max_count(0, &[Term::literal("a")], &ctx);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn min_inclusive_pass() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let val = Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer");
        let bound = Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer");
        let results = eval_range(&bound, &[val], &ctx, "minInclusive", |ord| {
            ord != Ordering::Less
        });
        assert!(results.is_empty());
    }

    #[test]
    fn min_exclusive_fail() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let val = Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer");
        let bound = Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer");
        let results = eval_range(&bound, &[val], &ctx, "minExclusive", |ord| {
            ord == Ordering::Greater
        });
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn min_length_pass() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_min_length(3, &[Term::literal("Alix")], &ctx);
        assert!(results.is_empty());
    }

    #[test]
    fn min_length_fail() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_min_length(10, &[Term::literal("Alix")], &ctx);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn has_value_present() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let val = Term::literal("Alix");
        let results = eval_has_value(&val, std::slice::from_ref(&val), &ctx);
        assert!(results.is_empty());
    }

    #[test]
    fn has_value_absent() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_has_value(&Term::literal("Missing"), &[Term::literal("Alix")], &ctx);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn in_valid() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let allowed = vec![Term::literal("a"), Term::literal("b"), Term::literal("c")];
        let results = eval_in(&allowed, &[Term::literal("b")], &ctx);
        assert!(results.is_empty());
    }

    #[test]
    fn in_invalid() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let allowed = vec![Term::literal("a"), Term::literal("b")];
        let results = eval_in(&allowed, &[Term::literal("z")], &ctx);
        assert_eq!(results.len(), 1);
    }

    // =====================================================================
    // compare_terms datatype guard tests
    // =====================================================================

    #[test]
    fn test_compare_terms_numeric_datatypes() {
        // Two xsd:integer literals should compare numerically: 10 > 9
        let ten = Term::typed_literal("10", Literal::XSD_INTEGER);
        let nine = Term::typed_literal("9", Literal::XSD_INTEGER);
        assert_eq!(compare_terms(&ten, &nine), Some(Ordering::Greater));
        assert_eq!(compare_terms(&nine, &ten), Some(Ordering::Less));
    }

    #[test]
    fn test_compare_terms_string_with_numeric_value() {
        // xsd:string literals compare lexicographically, NOT numerically.
        // Lexicographically "42" < "9" because '4' < '9'.
        let forty_two = Term::typed_literal("42", Literal::XSD_STRING);
        let nine = Term::typed_literal("9", Literal::XSD_STRING);
        assert_eq!(compare_terms(&forty_two, &nine), Some(Ordering::Less));
    }

    #[test]
    fn test_compare_terms_mixed_numeric_string() {
        // One xsd:integer and one xsd:string: only one is numeric, so
        // the comparison falls back to lexicographic ordering.
        let integer_val = Term::typed_literal("10", Literal::XSD_INTEGER);
        let string_val = Term::typed_literal("9", Literal::XSD_STRING);
        // Lexicographically "10" < "9" because '1' < '9'.
        assert_eq!(
            compare_terms(&integer_val, &string_val),
            Some(Ordering::Less)
        );
    }

    // =====================================================================
    // minLength / maxLength character count (non-ASCII) tests
    // =====================================================================

    #[test]
    fn test_min_length_non_ascii() {
        // "café" is 4 characters but 5 UTF-8 bytes.
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();

        let cafe = Term::literal("café");

        // sh:minLength 4 should pass (4 chars >= 4)
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_min_length(4, std::slice::from_ref(&cafe), &ctx);
        assert!(results.is_empty(), "café (4 chars) should pass minLength 4");

        // sh:minLength 5 should fail (4 chars < 5)
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_min_length(5, &[cafe], &ctx);
        assert_eq!(results.len(), 1, "café (4 chars) should fail minLength 5");
    }

    #[test]
    fn test_max_length_non_ascii() {
        // "Ωmega" is 5 characters but 6 UTF-8 bytes (Ω is 2 bytes).
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();

        let omega = Term::literal("\u{03a9}mega");

        // sh:maxLength 5 should pass (5 chars <= 5)
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_max_length(5, std::slice::from_ref(&omega), &ctx);
        assert!(
            results.is_empty(),
            "Ωmega (5 chars) should pass maxLength 5"
        );

        // sh:maxLength 4 should fail (5 chars > 4)
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_max_length(4, &[omega], &ctx);
        assert_eq!(results.len(), 1, "Ωmega (5 chars) should fail maxLength 4");
    }

    // =====================================================================
    // QualifiedValueShape disjoint tests
    // =====================================================================

    #[test]
    fn test_qualified_disjoint_excludes_overlapping() {
        use super::super::shape::{Constraint, NodeShape, PropertyShape, Severity};

        // Setup: a data graph with Alix who has two scores, 80 and 95.
        let store = RdfStore::new();
        let alix = Term::iri("http://ex.org/alix");
        let score_pred = Term::iri("http://ex.org/score");
        store.insert(Triple::new(
            alix.clone(),
            score_pred.clone(),
            Term::typed_literal("80", Literal::XSD_INTEGER),
        ));
        store.insert(Triple::new(
            alix.clone(),
            score_pred.clone(),
            Term::typed_literal("95", Literal::XSD_INTEGER),
        ));

        // Sibling shape A: sh:minInclusive 70, sh:maxInclusive 100
        // Matches values in [70..100], so both 80 and 95 conform.
        let sibling_shape_a = Shape::Node(NodeShape {
            id: Term::iri("http://ex.org/ShapeA"),
            targets: Vec::new(),
            property_shapes: Vec::new(),
            constraints: vec![
                Constraint::MinInclusive(Term::typed_literal("70", Literal::XSD_INTEGER)),
                Constraint::MaxInclusive(Term::typed_literal("100", Literal::XSD_INTEGER)),
            ],
            deactivated: false,
            severity: Severity::Violation,
            messages: Vec::new(),
        });

        // Sibling shape B: sh:minInclusive 90, sh:maxInclusive 100
        // Matches values in [90..100], so only 95 conforms (not 80).
        let sibling_shape_b = Shape::Node(NodeShape {
            id: Term::iri("http://ex.org/ShapeB"),
            targets: Vec::new(),
            property_shapes: Vec::new(),
            constraints: vec![
                Constraint::MinInclusive(Term::typed_literal("90", Literal::XSD_INTEGER)),
                Constraint::MaxInclusive(Term::typed_literal("100", Literal::XSD_INTEGER)),
            ],
            deactivated: false,
            severity: Severity::Violation,
            messages: Vec::new(),
        });

        // The owning property shape has two QualifiedValueShape constraints
        // with disjoint=true. Shape A claims [70..100], shape B claims [90..100].
        // Value 95 conforms to BOTH, so disjoint mode should exclude it from
        // each shape's count. Value 80 conforms to A but not B, so A should
        // count it. Net: shape A conforming=1 (only 80), shape B conforming=0.
        let prop_shape = PropertyShape {
            id: Term::iri("http://ex.org/ScoreShape"),
            path: PropertyPath::Predicate(score_pred),
            targets: Vec::new(),
            constraints: vec![
                Constraint::QualifiedValueShape {
                    shape: Box::new(sibling_shape_a.clone()),
                    min_count: Some(2),
                    max_count: None,
                    disjoint: true,
                },
                Constraint::QualifiedValueShape {
                    shape: Box::new(sibling_shape_b.clone()),
                    min_count: Some(1),
                    max_count: None,
                    disjoint: true,
                },
            ],
            deactivated: false,
            severity: Severity::Violation,
            messages: Vec::new(),
            name: None,
            description: None,
        };

        let owner_shape = Shape::Property(prop_shape);
        let value_nodes = vec![
            Term::typed_literal("80", Literal::XSD_INTEGER),
            Term::typed_literal("95", Literal::XSD_INTEGER),
        ];
        let mut visited = HashSet::new();

        // Evaluate shape A with disjoint=true.
        // 80 conforms to A and does NOT conform to sibling B, so counted.
        // 95 conforms to A and ALSO conforms to sibling B, so excluded.
        // conforming=1, qualifiedMinCount=2 => violation expected.
        let mut ctx_a = EvalContext {
            focus_node: &alix,
            shape: &owner_shape,
            path: None,
            data_graph: &store,
            all_shapes: &[],
            visited: &mut visited,
        };
        let results_a = eval_qualified(
            &sibling_shape_a,
            Some(2),
            None,
            true,
            &value_nodes,
            &mut ctx_a,
        );
        assert!(
            !results_a.is_empty(),
            "Shape A with disjoint should report a violation: only 1 non-overlapping value, but minCount=2"
        );

        // Evaluate shape B with disjoint=true.
        // 80 does not conform to B, so not counted.
        // 95 conforms to B and ALSO conforms to sibling A, so excluded.
        // conforming=0, qualifiedMinCount=1 => violation expected.
        let mut ctx_b = EvalContext {
            focus_node: &alix,
            shape: &owner_shape,
            path: None,
            data_graph: &store,
            all_shapes: &[],
            visited: &mut visited,
        };
        let results_b = eval_qualified(
            &sibling_shape_b,
            Some(1),
            None,
            true,
            &value_nodes,
            &mut ctx_b,
        );
        assert!(
            !results_b.is_empty(),
            "Shape B with disjoint should report a violation: 0 non-overlapping values, but minCount=1"
        );
    }

    // =====================================================================
    // Pattern constraint
    // =====================================================================

    #[test]
    fn pattern_match_passes() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_pattern("^[A-Za-z]+$", None, &[Term::literal("Alix")], &ctx);
        assert!(results.is_empty(), "Alix should match ^[A-Za-z]+$");
    }

    #[test]
    fn pattern_no_match_fails() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_pattern("^[0-9]+$", None, &[Term::literal("Alix")], &ctx);
        assert_eq!(results.len(), 1);
        assert!(results[0].source_constraint_component.contains("Pattern"));
        assert!(
            results[0]
                .message
                .as_ref()
                .unwrap()
                .contains("does not match")
        );
    }

    #[test]
    fn pattern_with_case_insensitive_flag() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        // Without flag: "alix" does not match ^ALIX$
        let results = eval_pattern("^ALIX$", None, &[Term::literal("alix")], &ctx);
        assert_eq!(results.len(), 1, "case-sensitive should fail");

        // With flag: "alix" matches ^ALIX$ case-insensitively
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_pattern("^ALIX$", Some("i"), &[Term::literal("alix")], &ctx);
        assert!(results.is_empty(), "case-insensitive should pass");
    }

    // =====================================================================
    // Language-in constraint
    // =====================================================================

    #[test]
    fn language_in_allowed_tag() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let val = Term::lang_literal("Alix", "en");
        let results = eval_language_in(&["en".to_string(), "de".to_string()], &[val], &ctx);
        assert!(results.is_empty(), "en should be in [en, de]");
    }

    #[test]
    fn language_in_subtag_match() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let val = Term::lang_literal("Alix", "en-US");
        let results = eval_language_in(&["en".to_string()], &[val], &ctx);
        assert!(results.is_empty(), "en-US should match base language en");
    }

    #[test]
    fn language_in_disallowed_tag() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let val = Term::lang_literal("Alix", "fr");
        let results = eval_language_in(&["en".to_string(), "de".to_string()], &[val], &ctx);
        assert_eq!(results.len(), 1);
        assert!(
            results[0]
                .source_constraint_component
                .contains("LanguageIn")
        );
    }

    #[test]
    fn language_in_no_lang_tag_fails() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let val = Term::literal("Alix"); // no language tag
        let results = eval_language_in(&["en".to_string()], &[val], &ctx);
        assert_eq!(
            results.len(),
            1,
            "literal without lang tag should fail languageIn"
        );
    }

    // =====================================================================
    // Unique-lang constraint
    // =====================================================================

    #[test]
    fn unique_lang_distinct_tags() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let values = vec![
            Term::lang_literal("Alix", "en"),
            Term::lang_literal("Alix", "de"),
        ];
        let results = eval_unique_lang(&values, &ctx);
        assert!(results.is_empty(), "distinct lang tags should pass");
    }

    #[test]
    fn unique_lang_duplicate_tags() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let values = vec![
            Term::lang_literal("Alix", "en"),
            Term::lang_literal("Alex", "en"),
        ];
        let results = eval_unique_lang(&values, &ctx);
        assert_eq!(results.len(), 1);
        assert!(
            results[0]
                .source_constraint_component
                .contains("UniqueLang")
        );
    }

    // =====================================================================
    // Equals constraint
    // =====================================================================

    #[test]
    fn equals_matching_values() {
        let store = RdfStore::new();
        let alix = Term::iri("http://ex.org/alix");
        let name = Term::iri("http://ex.org/name");
        let label = Term::iri("http://ex.org/label");
        store.insert(Triple::new(alix.clone(), name, Term::literal("Alix")));
        store.insert(Triple::new(alix.clone(), label, Term::literal("Alix")));

        let shape = dummy_shape();
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_equals(
            &Term::iri("http://ex.org/label"),
            &[Term::literal("Alix")],
            &ctx,
        );
        assert!(results.is_empty(), "name and label both have 'Alix'");
    }

    #[test]
    fn equals_mismatched_values() {
        let store = RdfStore::new();
        let alix = Term::iri("http://ex.org/alix");
        let name = Term::iri("http://ex.org/name");
        let label = Term::iri("http://ex.org/label");
        store.insert(Triple::new(alix.clone(), name, Term::literal("Alix")));
        store.insert(Triple::new(alix.clone(), label, Term::literal("Alex")));

        let shape = dummy_shape();
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_equals(
            &Term::iri("http://ex.org/label"),
            &[Term::literal("Alix")],
            &ctx,
        );
        assert_eq!(
            results.len(),
            2,
            "Alix not in label values and Alex not in name values"
        );
        assert!(
            results
                .iter()
                .all(|r| r.source_constraint_component.contains("Equals"))
        );
    }

    // =====================================================================
    // Disjoint constraint
    // =====================================================================

    #[test]
    fn disjoint_no_overlap() {
        let store = RdfStore::new();
        let alix = Term::iri("http://ex.org/alix");
        store.insert(Triple::new(
            alix.clone(),
            Term::iri("http://ex.org/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            alix.clone(),
            Term::iri("http://ex.org/nick"),
            Term::literal("Al"),
        ));

        let shape = dummy_shape();
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_disjoint(
            &Term::iri("http://ex.org/nick"),
            &[Term::literal("Alix")],
            &ctx,
        );
        assert!(results.is_empty(), "Alix and Al are disjoint");
    }

    #[test]
    fn disjoint_with_overlap() {
        let store = RdfStore::new();
        let alix = Term::iri("http://ex.org/alix");
        store.insert(Triple::new(
            alix.clone(),
            Term::iri("http://ex.org/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            alix.clone(),
            Term::iri("http://ex.org/nick"),
            Term::literal("Alix"),
        ));

        let shape = dummy_shape();
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_disjoint(
            &Term::iri("http://ex.org/nick"),
            &[Term::literal("Alix")],
            &ctx,
        );
        assert_eq!(results.len(), 1);
        assert!(results[0].source_constraint_component.contains("Disjoint"));
        assert_eq!(results[0].value.as_ref().unwrap(), &Term::literal("Alix"));
    }

    // =====================================================================
    // Less-than constraint
    // =====================================================================

    #[test]
    fn less_than_passes() {
        let store = RdfStore::new();
        let alix = Term::iri("http://ex.org/alix");
        let start = Term::iri("http://ex.org/start");
        let end = Term::iri("http://ex.org/end");
        store.insert(Triple::new(
            alix.clone(),
            start,
            Term::typed_literal("10", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        store.insert(Triple::new(
            alix.clone(),
            end,
            Term::typed_literal("20", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let shape = dummy_shape();
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let val = Term::typed_literal("10", "http://www.w3.org/2001/XMLSchema#integer");
        let results = eval_less_than(&Term::iri("http://ex.org/end"), &[val], &ctx, false);
        assert!(results.is_empty(), "10 < 20 should pass");
    }

    #[test]
    fn less_than_fails() {
        let store = RdfStore::new();
        let alix = Term::iri("http://ex.org/alix");
        let end = Term::iri("http://ex.org/end");
        store.insert(Triple::new(
            alix.clone(),
            end,
            Term::typed_literal("5", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let shape = dummy_shape();
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let val = Term::typed_literal("10", "http://www.w3.org/2001/XMLSchema#integer");
        let results = eval_less_than(&Term::iri("http://ex.org/end"), &[val], &ctx, false);
        assert_eq!(results.len(), 1, "10 < 5 should fail");
        assert!(results[0].source_constraint_component.contains("LessThan"));
    }

    #[test]
    fn less_than_or_equals_boundary() {
        let store = RdfStore::new();
        let alix = Term::iri("http://ex.org/alix");
        let end = Term::iri("http://ex.org/end");
        store.insert(Triple::new(
            alix.clone(),
            end,
            Term::typed_literal("10", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let shape = dummy_shape();
        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let val = Term::typed_literal("10", "http://www.w3.org/2001/XMLSchema#integer");
        // lessThan: 10 < 10 fails
        let results = eval_less_than(
            &Term::iri("http://ex.org/end"),
            std::slice::from_ref(&val),
            &ctx,
            false,
        );
        assert_eq!(results.len(), 1, "10 < 10 should fail for strict lessThan");

        // lessThanOrEquals: 10 <= 10 passes
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_less_than(&Term::iri("http://ex.org/end"), &[val], &ctx, true);
        assert!(results.is_empty(), "10 <= 10 should pass");
    }

    // =====================================================================
    // Logical constraints (not, and, or, xone)
    // =====================================================================

    #[allow(dead_code)]
    fn int_datatype_shape() -> Shape {
        use super::super::shape::{NodeShape, Severity};
        Shape::Node(NodeShape {
            id: Term::iri("http://ex.org/IntShape"),
            targets: Vec::new(),
            property_shapes: Vec::new(),
            constraints: vec![Constraint::Datatype(Term::iri(
                "http://www.w3.org/2001/XMLSchema#integer",
            ))],
            deactivated: false,
            severity: Severity::Violation,
            messages: Vec::new(),
        })
    }

    fn string_node_kind_shape() -> Shape {
        use super::super::shape::{NodeShape, Severity};
        Shape::Node(NodeShape {
            id: Term::iri("http://ex.org/LitShape"),
            targets: Vec::new(),
            property_shapes: Vec::new(),
            constraints: vec![Constraint::NodeKind(NodeKindValue::Literal)],
            deactivated: false,
            severity: Severity::Violation,
            messages: Vec::new(),
        })
    }

    #[test]
    fn not_passes_when_inner_fails() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let mut ctx = make_ctx(&alix, &shape, &store, &mut visited);
        // alix is an IRI, not a literal; inner shape requires Literal nodeKind
        let inner = string_node_kind_shape();
        let results = eval_not(&inner, &mut ctx);
        assert!(
            results.is_empty(),
            "sh:not should pass when inner shape fails (alix is IRI, not Literal)"
        );
    }

    #[test]
    fn not_fails_when_inner_passes() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let mut ctx = make_ctx(&alix, &shape, &store, &mut visited);
        // alix is an IRI; inner shape requires IRI nodeKind
        let inner = {
            use super::super::shape::{NodeShape, Severity};
            Shape::Node(NodeShape {
                id: Term::iri("http://ex.org/IriShape"),
                targets: Vec::new(),
                property_shapes: Vec::new(),
                constraints: vec![Constraint::NodeKind(NodeKindValue::Iri)],
                deactivated: false,
                severity: Severity::Violation,
                messages: Vec::new(),
            })
        };
        let results = eval_not(&inner, &mut ctx);
        assert_eq!(
            results.len(),
            1,
            "sh:not should fail when inner shape passes"
        );
        assert!(results[0].source_constraint_component.contains("Not"));
    }

    #[test]
    fn and_all_pass() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let mut ctx = make_ctx(&alix, &shape, &store, &mut visited);
        // alix is IRI: both IRI-nodeKind and BlankNodeOrIRI should pass
        let shapes = vec![
            {
                use super::super::shape::{NodeShape, Severity};
                Shape::Node(NodeShape {
                    id: Term::iri("http://ex.org/S1"),
                    targets: Vec::new(),
                    property_shapes: Vec::new(),
                    constraints: vec![Constraint::NodeKind(NodeKindValue::Iri)],
                    deactivated: false,
                    severity: Severity::Violation,
                    messages: Vec::new(),
                })
            },
            {
                use super::super::shape::{NodeShape, Severity};
                Shape::Node(NodeShape {
                    id: Term::iri("http://ex.org/S2"),
                    targets: Vec::new(),
                    property_shapes: Vec::new(),
                    constraints: vec![Constraint::NodeKind(NodeKindValue::BlankNodeOrIri)],
                    deactivated: false,
                    severity: Severity::Violation,
                    messages: Vec::new(),
                })
            },
        ];
        let results = eval_and(&shapes, &mut ctx);
        assert!(
            results.is_empty(),
            "sh:and should pass when all shapes conform"
        );
    }

    #[test]
    fn and_one_fails() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let mut ctx = make_ctx(&alix, &shape, &store, &mut visited);
        // alix is IRI: IRI passes but Literal fails
        let shapes = vec![
            {
                use super::super::shape::{NodeShape, Severity};
                Shape::Node(NodeShape {
                    id: Term::iri("http://ex.org/S1"),
                    targets: Vec::new(),
                    property_shapes: Vec::new(),
                    constraints: vec![Constraint::NodeKind(NodeKindValue::Iri)],
                    deactivated: false,
                    severity: Severity::Violation,
                    messages: Vec::new(),
                })
            },
            string_node_kind_shape(),
        ];
        let results = eval_and(&shapes, &mut ctx);
        assert_eq!(results.len(), 1);
        assert!(results[0].source_constraint_component.contains("And"));
    }

    #[test]
    fn or_one_passes() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let mut ctx = make_ctx(&alix, &shape, &store, &mut visited);
        // alix is IRI: Literal fails but IRI passes
        let shapes = vec![string_node_kind_shape(), {
            use super::super::shape::{NodeShape, Severity};
            Shape::Node(NodeShape {
                id: Term::iri("http://ex.org/IriShape"),
                targets: Vec::new(),
                property_shapes: Vec::new(),
                constraints: vec![Constraint::NodeKind(NodeKindValue::Iri)],
                deactivated: false,
                severity: Severity::Violation,
                messages: Vec::new(),
            })
        }];
        let results = eval_or(&shapes, &mut ctx);
        assert!(
            results.is_empty(),
            "sh:or should pass when at least one shape conforms"
        );
    }

    #[test]
    fn or_none_pass() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let mut ctx = make_ctx(&alix, &shape, &store, &mut visited);
        // alix is IRI: both Literal and BlankNode fail
        let shapes = vec![string_node_kind_shape(), {
            use super::super::shape::{NodeShape, Severity};
            Shape::Node(NodeShape {
                id: Term::iri("http://ex.org/BnShape"),
                targets: Vec::new(),
                property_shapes: Vec::new(),
                constraints: vec![Constraint::NodeKind(NodeKindValue::BlankNode)],
                deactivated: false,
                severity: Severity::Violation,
                messages: Vec::new(),
            })
        }];
        let results = eval_or(&shapes, &mut ctx);
        assert_eq!(results.len(), 1);
        assert!(results[0].source_constraint_component.contains("Or"));
    }

    #[test]
    fn xone_exactly_one() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let mut ctx = make_ctx(&alix, &shape, &store, &mut visited);
        // alix is IRI: IRI passes, Literal fails -> exactly 1
        let shapes = vec![
            {
                use super::super::shape::{NodeShape, Severity};
                Shape::Node(NodeShape {
                    id: Term::iri("http://ex.org/IriShape"),
                    targets: Vec::new(),
                    property_shapes: Vec::new(),
                    constraints: vec![Constraint::NodeKind(NodeKindValue::Iri)],
                    deactivated: false,
                    severity: Severity::Violation,
                    messages: Vec::new(),
                })
            },
            string_node_kind_shape(),
        ];
        let results = eval_xone(&shapes, &mut ctx);
        assert!(
            results.is_empty(),
            "sh:xone should pass with exactly 1 conforming"
        );
    }

    #[test]
    fn xone_two_conform_fails() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let mut ctx = make_ctx(&alix, &shape, &store, &mut visited);
        // alix is IRI: both IRI and BlankNodeOrIRI pass -> 2 conforming
        let shapes = vec![
            {
                use super::super::shape::{NodeShape, Severity};
                Shape::Node(NodeShape {
                    id: Term::iri("http://ex.org/S1"),
                    targets: Vec::new(),
                    property_shapes: Vec::new(),
                    constraints: vec![Constraint::NodeKind(NodeKindValue::Iri)],
                    deactivated: false,
                    severity: Severity::Violation,
                    messages: Vec::new(),
                })
            },
            {
                use super::super::shape::{NodeShape, Severity};
                Shape::Node(NodeShape {
                    id: Term::iri("http://ex.org/S2"),
                    targets: Vec::new(),
                    property_shapes: Vec::new(),
                    constraints: vec![Constraint::NodeKind(NodeKindValue::BlankNodeOrIri)],
                    deactivated: false,
                    severity: Severity::Violation,
                    messages: Vec::new(),
                })
            },
        ];
        let results = eval_xone(&shapes, &mut ctx);
        assert_eq!(results.len(), 1);
        assert!(results[0].message.as_ref().unwrap().contains('2'));
    }

    // =====================================================================
    // Shape-node constraint
    // =====================================================================

    #[test]
    fn shape_node_value_conforms() {
        let store = RdfStore::new();
        let rdf_type = Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
        let alix = Term::iri("http://ex.org/alix");
        let gus = Term::iri("http://ex.org/gus");
        store.insert(Triple::new(
            alix.clone(),
            rdf_type.clone(),
            Term::iri("http://ex.org/Person"),
        ));
        store.insert(Triple::new(
            gus.clone(),
            rdf_type,
            Term::iri("http://ex.org/Person"),
        ));

        let shape = dummy_shape();
        let inner = {
            use super::super::shape::{NodeShape, Severity};
            Shape::Node(NodeShape {
                id: Term::iri("http://ex.org/IriShape"),
                targets: Vec::new(),
                property_shapes: Vec::new(),
                constraints: vec![Constraint::NodeKind(NodeKindValue::Iri)],
                deactivated: false,
                severity: Severity::Violation,
                messages: Vec::new(),
            })
        };

        let mut visited = HashSet::new();
        let mut ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_shape_node(&inner, &[gus], &mut ctx);
        assert!(
            results.is_empty(),
            "gus is an IRI, should conform to IRI shape"
        );
    }

    #[test]
    fn shape_node_value_fails() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let mut ctx = make_ctx(&alix, &shape, &store, &mut visited);
        // Validate a literal against an IRI-only shape
        let inner = {
            use super::super::shape::{NodeShape, Severity};
            Shape::Node(NodeShape {
                id: Term::iri("http://ex.org/IriShape"),
                targets: Vec::new(),
                property_shapes: Vec::new(),
                constraints: vec![Constraint::NodeKind(NodeKindValue::Iri)],
                deactivated: false,
                severity: Severity::Violation,
                messages: Vec::new(),
            })
        };
        let results = eval_shape_node(&inner, &[Term::literal("not-an-iri")], &mut ctx);
        assert_eq!(results.len(), 1);
        assert!(results[0].source_constraint_component.contains("Node"));
    }

    // =====================================================================
    // Closed constraint
    // =====================================================================

    #[test]
    fn closed_all_allowed() {
        let store = RdfStore::new();
        let alix = Term::iri("http://ex.org/alix");
        let rdf_type = Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
        let name = Term::iri("http://ex.org/name");
        store.insert(Triple::new(
            alix.clone(),
            rdf_type,
            Term::iri("http://ex.org/Person"),
        ));
        store.insert(Triple::new(
            alix.clone(),
            name.clone(),
            Term::literal("Alix"),
        ));

        use super::super::shape::{NodeShape, PropertyShape, Severity};
        let shape = Shape::Node(NodeShape {
            id: Term::iri("http://ex.org/ClosedShape"),
            targets: Vec::new(),
            property_shapes: vec![PropertyShape {
                id: Term::blank("p"),
                path: PropertyPath::Predicate(name),
                targets: Vec::new(),
                constraints: Vec::new(),
                deactivated: false,
                severity: Severity::Violation,
                messages: Vec::new(),
                name: None,
                description: None,
            }],
            constraints: Vec::new(),
            deactivated: false,
            severity: Severity::Violation,
            messages: Vec::new(),
        });

        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_closed(&[], &ctx);
        assert!(
            results.is_empty(),
            "only rdf:type and name used, both allowed"
        );
    }

    #[test]
    fn closed_extra_property_violates() {
        let store = RdfStore::new();
        let alix = Term::iri("http://ex.org/alix");
        let name = Term::iri("http://ex.org/name");
        let age = Term::iri("http://ex.org/age");
        store.insert(Triple::new(
            alix.clone(),
            name.clone(),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            alix.clone(),
            age,
            Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        use super::super::shape::{NodeShape, PropertyShape, Severity};
        let shape = Shape::Node(NodeShape {
            id: Term::iri("http://ex.org/ClosedShape"),
            targets: Vec::new(),
            property_shapes: vec![PropertyShape {
                id: Term::blank("p"),
                path: PropertyPath::Predicate(name),
                targets: Vec::new(),
                constraints: Vec::new(),
                deactivated: false,
                severity: Severity::Violation,
                messages: Vec::new(),
                name: None,
                description: None,
            }],
            constraints: Vec::new(),
            deactivated: false,
            severity: Severity::Violation,
            messages: Vec::new(),
        });

        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_closed(&[], &ctx);
        assert_eq!(
            results.len(),
            1,
            "age predicate should violate closed shape"
        );
        assert!(results[0].source_constraint_component.contains("Closed"));
        assert_eq!(
            results[0].value.as_ref().unwrap(),
            &Term::iri("http://ex.org/age")
        );
    }

    #[test]
    fn closed_with_ignored_properties() {
        let store = RdfStore::new();
        let alix = Term::iri("http://ex.org/alix");
        let name = Term::iri("http://ex.org/name");
        let age = Term::iri("http://ex.org/age");
        store.insert(Triple::new(
            alix.clone(),
            name.clone(),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            alix.clone(),
            age.clone(),
            Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        use super::super::shape::{NodeShape, PropertyShape, Severity};
        let shape = Shape::Node(NodeShape {
            id: Term::iri("http://ex.org/ClosedShape"),
            targets: Vec::new(),
            property_shapes: vec![PropertyShape {
                id: Term::blank("p"),
                path: PropertyPath::Predicate(name),
                targets: Vec::new(),
                constraints: Vec::new(),
                deactivated: false,
                severity: Severity::Violation,
                messages: Vec::new(),
                name: None,
                description: None,
            }],
            constraints: Vec::new(),
            deactivated: false,
            severity: Severity::Violation,
            messages: Vec::new(),
        });

        let mut visited = HashSet::new();
        let ctx = make_ctx(&alix, &shape, &store, &mut visited);
        let results = eval_closed(&[age], &ctx);
        assert!(results.is_empty(), "age in ignoredProperties should pass");
    }

    // =====================================================================
    // evaluate_constraint dispatcher
    // =====================================================================

    #[test]
    fn evaluate_constraint_dispatches_correctly() {
        let store = data_store();
        let shape = dummy_shape();
        let alix = Term::iri("http://ex.org/alix");
        let mut visited = HashSet::new();
        let mut ctx = make_ctx(&alix, &shape, &store, &mut visited);

        // MinCount via the dispatcher
        let results =
            evaluate_constraint(&Constraint::MinCount(1), &[Term::literal("a")], &mut ctx);
        assert!(results.is_empty());

        let results =
            evaluate_constraint(&Constraint::MinCount(5), &[Term::literal("a")], &mut ctx);
        assert_eq!(results.len(), 1);

        // MaxLength via the dispatcher
        let results = evaluate_constraint(
            &Constraint::MaxLength(10),
            &[Term::literal("short")],
            &mut ctx,
        );
        assert!(results.is_empty());

        // HasValue via the dispatcher
        let results = evaluate_constraint(
            &Constraint::HasValue(Term::literal("a")),
            &[Term::literal("a"), Term::literal("b")],
            &mut ctx,
        );
        assert!(results.is_empty());

        // SPARQL constraint returns empty (no executor in core tests)
        let results = evaluate_constraint(
            &Constraint::Sparql(super::super::shape::SparqlConstraint {
                select: "SELECT ?this WHERE { ?this ?p ?o }".to_string(),
                message: None,
                prefixes: Vec::new(),
                deactivated: false,
            }),
            &[],
            &mut ctx,
        );
        assert!(
            results.is_empty(),
            "SPARQL constraints should be no-op in core"
        );
    }
}
