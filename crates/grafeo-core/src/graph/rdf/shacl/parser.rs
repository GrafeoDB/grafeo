//! SHACL shape parser.
//!
//! Reads shape definitions from an RDF store by traversing triples that use
//! the SHACL vocabulary. Produces a list of `Shape` values suitable for
//! validation.

use std::collections::HashSet;

use crate::graph::rdf::{GraphAt, Term, TriplePattern};

use super::shape::{
    Constraint, NodeKindValue, NodeShape, PrefixDeclaration, PropertyPath, PropertyShape, RDF, SH,
    Severity, ShaclError, Shape, SparqlConstraint, Target,
};

/// Constructs a `TriplePattern` from optional subject, predicate, and object.
fn pat(subject: Option<&Term>, predicate: Option<&Term>, object: Option<&Term>) -> TriplePattern {
    TriplePattern {
        subject: subject.cloned(),
        predicate: predicate.cloned(),
        object: object.cloned(),
    }
}

/// Parses all SHACL shapes from the given shapes graph.
///
/// Finds shapes by `rdf:type sh:NodeShape` / `sh:PropertyShape` and by
/// implicit detection (subjects with SHACL constraint properties).
///
/// # Errors
///
/// Returns `ShaclError::InvalidShape` if a shape is structurally invalid
/// (e.g., a property shape without `sh:path`).
pub fn parse_shapes(shapes_graph: GraphAt<'_>) -> Result<Vec<Shape>, ShaclError> {
    let rdf_type = Term::iri(RDF::TYPE);

    // Collect all explicitly typed shapes
    let mut shape_ids: HashSet<Term> = HashSet::new();

    let node_shape_type = Term::iri(SH::NODE_SHAPE);
    for triple in shapes_graph.find(&pat(None, Some(&rdf_type), Some(&node_shape_type))) {
        shape_ids.insert(triple.subject().clone());
    }

    let prop_shape_type = Term::iri(SH::PROPERTY_SHAPE);
    for triple in shapes_graph.find(&pat(None, Some(&rdf_type), Some(&prop_shape_type))) {
        shape_ids.insert(triple.subject().clone());
    }

    // Detect implicit shapes: subjects with sh:targetClass, sh:property, or sh:path
    // that are not already identified
    let implicit_predicates = [
        SH::TARGET_CLASS,
        SH::TARGET_NODE,
        SH::TARGET_SUBJECTS_OF,
        SH::TARGET_OBJECTS_OF,
        SH::PROPERTY,
    ];
    for pred_iri in &implicit_predicates {
        let pred = Term::iri(*pred_iri);
        for triple in shapes_graph.triples_with_predicate(&pred) {
            shape_ids.insert(triple.subject().clone());
        }
    }

    // Parse each shape
    let mut shapes = Vec::new();
    let mut visiting = HashSet::new();
    for shape_id in &shape_ids {
        let shape = parse_shape(shapes_graph, shape_id, &shape_ids, &mut visiting)?;
        shapes.push(shape);
    }

    Ok(shapes)
}

/// Parses a single shape from the shapes graph.
fn parse_shape(
    graph: GraphAt<'_>,
    shape_id: &Term,
    all_shape_ids: &HashSet<Term>,
    visiting: &mut HashSet<Term>,
) -> Result<Shape, ShaclError> {
    if !visiting.insert(shape_id.clone()) {
        return Err(ShaclError::InvalidShape(format!(
            "Cyclic shape reference detected at {shape_id}"
        )));
    }

    let result = parse_shape_inner(graph, shape_id, all_shape_ids, visiting);
    visiting.remove(shape_id);
    result
}

fn parse_shape_inner(
    graph: GraphAt<'_>,
    shape_id: &Term,
    all_shape_ids: &HashSet<Term>,
    visiting: &mut HashSet<Term>,
) -> Result<Shape, ShaclError> {
    let rdf_type = Term::iri(RDF::TYPE);

    // Determine if this is a property shape (has sh:path)
    let path_pred = Term::iri(SH::PATH);
    let path_triples = graph.find(&pat(Some(shape_id), Some(&path_pred), None));

    if let Some(path_triple) = path_triples.first() {
        // Property shape
        let path = parse_path(graph, path_triple.object())?;
        let targets = parse_targets(graph, shape_id);
        let constraints = parse_constraints(graph, shape_id, all_shape_ids, visiting)?;
        let severity = parse_severity(graph, shape_id);
        let deactivated = parse_deactivated(graph, shape_id);
        let messages = parse_messages(graph, shape_id);
        let name = parse_string_property(graph, shape_id, SH::NAME);
        let description = parse_string_property(graph, shape_id, SH::DESCRIPTION);

        Ok(Shape::Property(PropertyShape {
            id: shape_id.clone(),
            path,
            targets,
            constraints,
            deactivated,
            severity,
            messages,
            name,
            description,
        }))
    } else {
        // Check explicit type
        let is_prop_shape = !graph
            .find(&pat(
                Some(shape_id),
                Some(&rdf_type),
                Some(&Term::iri(SH::PROPERTY_SHAPE)),
            ))
            .is_empty();

        if is_prop_shape {
            return Err(ShaclError::InvalidShape(format!(
                "Property shape {} is missing sh:path",
                shape_id
            )));
        }

        // Node shape
        let targets = parse_targets(graph, shape_id);
        let constraints = parse_constraints(graph, shape_id, all_shape_ids, visiting)?;
        let property_shapes = parse_property_shapes(graph, shape_id, all_shape_ids, visiting)?;
        let severity = parse_severity(graph, shape_id);
        let deactivated = parse_deactivated(graph, shape_id);
        let messages = parse_messages(graph, shape_id);

        Ok(Shape::Node(NodeShape {
            id: shape_id.clone(),
            targets,
            property_shapes,
            constraints,
            deactivated,
            severity,
            messages,
        }))
    }
}

// =========================================================================
// Target parsing
// =========================================================================

fn parse_targets(graph: GraphAt<'_>, shape_id: &Term) -> Vec<Target> {
    let mut targets = Vec::new();

    let target_class = Term::iri(SH::TARGET_CLASS);
    for triple in graph.find(&pat(Some(shape_id), Some(&target_class), None)) {
        targets.push(Target::Class(triple.object().clone()));
    }

    let target_node = Term::iri(SH::TARGET_NODE);
    for triple in graph.find(&pat(Some(shape_id), Some(&target_node), None)) {
        targets.push(Target::Node(triple.object().clone()));
    }

    let target_subjects_of = Term::iri(SH::TARGET_SUBJECTS_OF);
    for triple in graph.find(&pat(Some(shape_id), Some(&target_subjects_of), None)) {
        targets.push(Target::SubjectsOf(triple.object().clone()));
    }

    let target_objects_of = Term::iri(SH::TARGET_OBJECTS_OF);
    for triple in graph.find(&pat(Some(shape_id), Some(&target_objects_of), None)) {
        targets.push(Target::ObjectsOf(triple.object().clone()));
    }

    targets
}

// =========================================================================
// Property shape parsing (nested under node shapes)
// =========================================================================

fn parse_property_shapes(
    graph: GraphAt<'_>,
    node_shape_id: &Term,
    all_shape_ids: &HashSet<Term>,
    visiting: &mut HashSet<Term>,
) -> Result<Vec<PropertyShape>, ShaclError> {
    let property_pred = Term::iri(SH::PROPERTY);
    let mut result = Vec::new();

    for triple in graph.find(&pat(Some(node_shape_id), Some(&property_pred), None)) {
        let prop_id = triple.object();
        let path_pred = Term::iri(SH::PATH);
        let path_triples = graph.find(&pat(Some(prop_id), Some(&path_pred), None));

        let path = match path_triples.first() {
            Some(pt) => parse_path(graph, pt.object())?,
            None => {
                return Err(ShaclError::InvalidShape(format!(
                    "Property shape {} is missing sh:path",
                    prop_id
                )));
            }
        };

        let constraints = parse_constraints(graph, prop_id, all_shape_ids, visiting)?;
        let severity = parse_severity(graph, prop_id);
        let deactivated = parse_deactivated(graph, prop_id);
        let messages = parse_messages(graph, prop_id);
        let name = parse_string_property(graph, prop_id, SH::NAME);
        let description = parse_string_property(graph, prop_id, SH::DESCRIPTION);

        result.push(PropertyShape {
            id: prop_id.clone(),
            path,
            targets: Vec::new(), // Inherited from parent node shape
            constraints,
            deactivated,
            severity,
            messages,
            name,
            description,
        });
    }

    Ok(result)
}

// =========================================================================
// Path parsing
// =========================================================================

/// Parses a SHACL property path from a term.
///
/// Simple paths are IRIs. Complex paths are blank nodes with path modifier
/// properties (`sh:inversePath`, `sh:alternativePath`, etc.) or RDF lists
/// (sequence paths).
fn parse_path(graph: GraphAt<'_>, term: &Term) -> Result<PropertyPath, ShaclError> {
    // Simple predicate path: the term is an IRI
    if term.is_iri() {
        return Ok(PropertyPath::Predicate(term.clone()));
    }

    // Complex path: the term is a blank node with path modifier properties
    if term.is_blank_node() {
        // sh:inversePath
        let inverse_pred = Term::iri(SH::INVERSE_PATH);
        let inverse = graph.find(&pat(Some(term), Some(&inverse_pred), None));
        if let Some(t) = inverse.first() {
            let inner = parse_path(graph, t.object())?;
            return Ok(PropertyPath::Inverse(Box::new(inner)));
        }

        // sh:alternativePath
        let alt_pred = Term::iri(SH::ALTERNATIVE_PATH);
        let alt = graph.find(&pat(Some(term), Some(&alt_pred), None));
        if let Some(t) = alt.first() {
            let items = collect_rdf_list(graph, t.object());
            let mut paths = Vec::new();
            for item in &items {
                paths.push(parse_path(graph, item)?);
            }
            return Ok(PropertyPath::Alternative(paths));
        }

        // sh:zeroOrMorePath
        let zom_pred = Term::iri(SH::ZERO_OR_MORE_PATH);
        let zom = graph.find(&pat(Some(term), Some(&zom_pred), None));
        if let Some(t) = zom.first() {
            let inner = parse_path(graph, t.object())?;
            return Ok(PropertyPath::ZeroOrMore(Box::new(inner)));
        }

        // sh:oneOrMorePath
        let oom_pred = Term::iri(SH::ONE_OR_MORE_PATH);
        let oom = graph.find(&pat(Some(term), Some(&oom_pred), None));
        if let Some(t) = oom.first() {
            let inner = parse_path(graph, t.object())?;
            return Ok(PropertyPath::OneOrMore(Box::new(inner)));
        }

        // sh:zeroOrOnePath
        let zoo_pred = Term::iri(SH::ZERO_OR_ONE_PATH);
        let zoo = graph.find(&pat(Some(term), Some(&zoo_pred), None));
        if let Some(t) = zoo.first() {
            let inner = parse_path(graph, t.object())?;
            return Ok(PropertyPath::ZeroOrOne(Box::new(inner)));
        }

        // Sequence path: blank node is the head of an RDF list
        let first_pred = Term::iri(RDF::FIRST);
        let has_first = !graph
            .find(&pat(Some(term), Some(&first_pred), None))
            .is_empty();
        if has_first {
            let items = collect_rdf_list(graph, term);
            let mut paths = Vec::new();
            for item in &items {
                paths.push(parse_path(graph, item)?);
            }
            return Ok(PropertyPath::Sequence(paths));
        }
    }

    Err(ShaclError::InvalidPath(format!(
        "Cannot parse property path from term: {term}"
    )))
}

// =========================================================================
// Constraint parsing
// =========================================================================

fn parse_constraints(
    graph: GraphAt<'_>,
    shape_id: &Term,
    all_shape_ids: &HashSet<Term>,
    visiting: &mut HashSet<Term>,
) -> Result<Vec<Constraint>, ShaclError> {
    let mut constraints = Vec::new();

    // Value type constraints
    parse_term_constraints(
        graph,
        shape_id,
        SH::CLASS,
        &mut constraints,
        Constraint::Class,
    );
    parse_term_constraints(
        graph,
        shape_id,
        SH::DATATYPE,
        &mut constraints,
        Constraint::Datatype,
    );
    parse_node_kind_constraints(graph, shape_id, &mut constraints)?;

    // Cardinality
    if let Some(n) = parse_integer_property(graph, shape_id, SH::MIN_COUNT)
        && let Ok(count) = usize::try_from(n)
    {
        constraints.push(Constraint::MinCount(count));
    }
    if let Some(n) = parse_integer_property(graph, shape_id, SH::MAX_COUNT)
        && let Ok(count) = usize::try_from(n)
    {
        constraints.push(Constraint::MaxCount(count));
    }

    // Value range
    parse_term_constraints(
        graph,
        shape_id,
        SH::MIN_EXCLUSIVE,
        &mut constraints,
        Constraint::MinExclusive,
    );
    parse_term_constraints(
        graph,
        shape_id,
        SH::MAX_EXCLUSIVE,
        &mut constraints,
        Constraint::MaxExclusive,
    );
    parse_term_constraints(
        graph,
        shape_id,
        SH::MIN_INCLUSIVE,
        &mut constraints,
        Constraint::MinInclusive,
    );
    parse_term_constraints(
        graph,
        shape_id,
        SH::MAX_INCLUSIVE,
        &mut constraints,
        Constraint::MaxInclusive,
    );

    // String constraints
    if let Some(n) = parse_integer_property(graph, shape_id, SH::MIN_LENGTH)
        && let Ok(len) = usize::try_from(n)
    {
        constraints.push(Constraint::MinLength(len));
    }
    if let Some(n) = parse_integer_property(graph, shape_id, SH::MAX_LENGTH)
        && let Ok(len) = usize::try_from(n)
    {
        constraints.push(Constraint::MaxLength(len));
    }
    parse_pattern_constraint(graph, shape_id, &mut constraints);
    parse_language_in_constraint(graph, shape_id, &mut constraints);
    if parse_boolean_property(graph, shape_id, SH::UNIQUE_LANG) {
        constraints.push(Constraint::UniqueLang);
    }

    // Property pair
    parse_term_constraints(
        graph,
        shape_id,
        SH::EQUALS,
        &mut constraints,
        Constraint::Equals,
    );
    parse_term_constraints(
        graph,
        shape_id,
        SH::DISJOINT,
        &mut constraints,
        Constraint::Disjoint,
    );
    parse_term_constraints(
        graph,
        shape_id,
        SH::LESS_THAN,
        &mut constraints,
        Constraint::LessThan,
    );
    parse_term_constraints(
        graph,
        shape_id,
        SH::LESS_THAN_OR_EQUALS,
        &mut constraints,
        Constraint::LessThanOrEquals,
    );

    // Logical constraints
    parse_logical_constraints(graph, shape_id, all_shape_ids, visiting, &mut constraints)?;

    // Shape-based: sh:node
    let node_pred = Term::iri(SH::NODE);
    for triple in graph.find(&pat(Some(shape_id), Some(&node_pred), None)) {
        let inner = parse_inline_shape(graph, triple.object(), all_shape_ids, visiting)?;
        constraints.push(Constraint::ShapeNode(Box::new(inner)));
    }

    // sh:qualifiedValueShape
    parse_qualified_value_shape(graph, shape_id, all_shape_ids, visiting, &mut constraints)?;

    // Other constraints
    parse_closed_constraint(graph, shape_id, &mut constraints);
    parse_term_constraints(
        graph,
        shape_id,
        SH::HAS_VALUE,
        &mut constraints,
        Constraint::HasValue,
    );
    parse_in_constraint(graph, shape_id, &mut constraints);

    // SPARQL constraints
    parse_sparql_constraints(graph, shape_id, &mut constraints)?;

    Ok(constraints)
}

// =========================================================================
// Constraint parsing helpers
// =========================================================================

fn parse_term_constraints(
    graph: GraphAt<'_>,
    shape_id: &Term,
    predicate_iri: &str,
    constraints: &mut Vec<Constraint>,
    constructor: fn(Term) -> Constraint,
) {
    let pred = Term::iri(predicate_iri);
    for triple in graph.find(&pat(Some(shape_id), Some(&pred), None)) {
        constraints.push(constructor(triple.object().clone()));
    }
}

fn parse_node_kind_constraints(
    graph: GraphAt<'_>,
    shape_id: &Term,
    constraints: &mut Vec<Constraint>,
) -> Result<(), ShaclError> {
    let pred = Term::iri(SH::NODE_KIND);
    for triple in graph.find(&pat(Some(shape_id), Some(&pred), None)) {
        // sh:nodeKind values must be IRIs per the SHACL spec
        let iri_str = match triple.object() {
            Term::Iri(iri) => iri.as_str(),
            _ => {
                return Err(ShaclError::InvalidShape(
                    "sh:nodeKind value must be an IRI, not a literal or blank node".to_string(),
                ));
            }
        };
        let value = match iri_str {
            SH::BLANK_NODE => NodeKindValue::BlankNode,
            SH::IRI => NodeKindValue::Iri,
            SH::LITERAL => NodeKindValue::Literal,
            SH::BLANK_NODE_OR_IRI => NodeKindValue::BlankNodeOrIri,
            SH::BLANK_NODE_OR_LITERAL => NodeKindValue::BlankNodeOrLiteral,
            SH::IRI_OR_LITERAL => NodeKindValue::IriOrLiteral,
            other => {
                return Err(ShaclError::InvalidShape(format!(
                    "Unknown sh:nodeKind value: {other}"
                )));
            }
        };
        constraints.push(Constraint::NodeKind(value));
    }
    Ok(())
}

fn parse_pattern_constraint(
    graph: GraphAt<'_>,
    shape_id: &Term,
    constraints: &mut Vec<Constraint>,
) {
    let pattern_pred = Term::iri(SH::PATTERN);
    for triple in graph.find(&pat(Some(shape_id), Some(&pattern_pred), None)) {
        if let Some(pattern) = term_string_value(triple.object()) {
            let flags_pred = Term::iri(SH::FLAGS);
            let flags = graph
                .find(&pat(Some(shape_id), Some(&flags_pred), None))
                .first()
                .and_then(|t| term_string_value(t.object()));
            constraints.push(Constraint::Pattern { pattern, flags });
        }
    }
}

fn parse_language_in_constraint(
    graph: GraphAt<'_>,
    shape_id: &Term,
    constraints: &mut Vec<Constraint>,
) {
    let pred = Term::iri(SH::LANGUAGE_IN);
    for triple in graph.find(&pat(Some(shape_id), Some(&pred), None)) {
        let items = collect_rdf_list(graph, triple.object());
        let langs: Vec<String> = items.iter().filter_map(term_string_value).collect();
        if !langs.is_empty() {
            constraints.push(Constraint::LanguageIn(langs));
        }
    }
}

fn parse_logical_constraints(
    graph: GraphAt<'_>,
    shape_id: &Term,
    all_shape_ids: &HashSet<Term>,
    visiting: &mut HashSet<Term>,
    constraints: &mut Vec<Constraint>,
) -> Result<(), ShaclError> {
    // sh:not
    let not_pred = Term::iri(SH::NOT);
    for triple in graph.find(&pat(Some(shape_id), Some(&not_pred), None)) {
        let inner = parse_inline_shape(graph, triple.object(), all_shape_ids, visiting)?;
        constraints.push(Constraint::Not(Box::new(inner)));
    }

    // sh:and
    let and_pred = Term::iri(SH::AND);
    for triple in graph.find(&pat(Some(shape_id), Some(&and_pred), None)) {
        let items = collect_rdf_list(graph, triple.object());
        let mut shapes = Vec::new();
        for item in &items {
            shapes.push(parse_inline_shape(graph, item, all_shape_ids, visiting)?);
        }
        constraints.push(Constraint::And(shapes));
    }

    // sh:or
    let or_pred = Term::iri(SH::OR);
    for triple in graph.find(&pat(Some(shape_id), Some(&or_pred), None)) {
        let items = collect_rdf_list(graph, triple.object());
        let mut shapes = Vec::new();
        for item in &items {
            shapes.push(parse_inline_shape(graph, item, all_shape_ids, visiting)?);
        }
        constraints.push(Constraint::Or(shapes));
    }

    // sh:xone
    let xone_pred = Term::iri(SH::XONE);
    for triple in graph.find(&pat(Some(shape_id), Some(&xone_pred), None)) {
        let items = collect_rdf_list(graph, triple.object());
        let mut shapes = Vec::new();
        for item in &items {
            shapes.push(parse_inline_shape(graph, item, all_shape_ids, visiting)?);
        }
        constraints.push(Constraint::Xone(shapes));
    }

    Ok(())
}

fn parse_qualified_value_shape(
    graph: GraphAt<'_>,
    shape_id: &Term,
    all_shape_ids: &HashSet<Term>,
    visiting: &mut HashSet<Term>,
    constraints: &mut Vec<Constraint>,
) -> Result<(), ShaclError> {
    let qvs_pred = Term::iri(SH::QUALIFIED_VALUE_SHAPE);
    for triple in graph.find(&pat(Some(shape_id), Some(&qvs_pred), None)) {
        let inner = parse_inline_shape(graph, triple.object(), all_shape_ids, visiting)?;
        let min_count = parse_integer_property(graph, shape_id, SH::QUALIFIED_MIN_COUNT)
            .and_then(|n| usize::try_from(n).ok());
        let max_count = parse_integer_property(graph, shape_id, SH::QUALIFIED_MAX_COUNT)
            .and_then(|n| usize::try_from(n).ok());
        let disjoint = parse_boolean_property(graph, shape_id, SH::QUALIFIED_VALUE_SHAPES_DISJOINT);

        constraints.push(Constraint::QualifiedValueShape {
            shape: Box::new(inner),
            min_count,
            max_count,
            disjoint,
        });
    }
    Ok(())
}

fn parse_closed_constraint(graph: GraphAt<'_>, shape_id: &Term, constraints: &mut Vec<Constraint>) {
    if parse_boolean_property(graph, shape_id, SH::CLOSED) {
        let ignored_pred = Term::iri(SH::IGNORED_PROPERTIES);
        let ignored = graph
            .find(&pat(Some(shape_id), Some(&ignored_pred), None))
            .first()
            .map(|t| collect_rdf_list(graph, t.object()))
            .unwrap_or_default();
        constraints.push(Constraint::Closed {
            ignored_properties: ignored,
        });
    }
}

fn parse_in_constraint(graph: GraphAt<'_>, shape_id: &Term, constraints: &mut Vec<Constraint>) {
    let pred = Term::iri(SH::IN);
    for triple in graph.find(&pat(Some(shape_id), Some(&pred), None)) {
        let items = collect_rdf_list(graph, triple.object());
        constraints.push(Constraint::In(items));
    }
}

fn parse_sparql_constraints(
    graph: GraphAt<'_>,
    shape_id: &Term,
    constraints: &mut Vec<Constraint>,
) -> Result<(), ShaclError> {
    let sparql_pred = Term::iri(SH::SPARQL);
    for triple in graph.find(&pat(Some(shape_id), Some(&sparql_pred), None)) {
        let constraint_node = triple.object();

        // sh:select (required)
        let select_pred = Term::iri(SH::SELECT);
        let select = graph
            .find(&pat(Some(constraint_node), Some(&select_pred), None))
            .first()
            .and_then(|t| term_string_value(t.object()));

        let Some(select) = select else {
            return Err(ShaclError::InvalidShape(format!(
                "sh:sparql constraint on {} is missing sh:select",
                shape_id
            )));
        };

        // sh:message (optional)
        let message = parse_string_property(graph, constraint_node, SH::MESSAGE);

        // sh:prefixes (optional, may be multiple)
        let prefixes = parse_prefix_declarations(graph, constraint_node);

        let deactivated = parse_deactivated(graph, constraint_node);

        constraints.push(Constraint::Sparql(SparqlConstraint {
            select,
            message,
            prefixes,
            deactivated,
        }));
    }
    Ok(())
}

fn parse_prefix_declarations(graph: GraphAt<'_>, constraint_node: &Term) -> Vec<PrefixDeclaration> {
    let mut declarations = Vec::new();
    let prefixes_pred = Term::iri(SH::PREFIXES);
    let declare_pred = Term::iri(SH::DECLARE);
    let prefix_pred = Term::iri(SH::PREFIX_DECL);
    let namespace_pred = Term::iri(SH::NAMESPACE);

    for pref_triple in graph.find(&pat(Some(constraint_node), Some(&prefixes_pred), None)) {
        let pref_node = pref_triple.object();
        for decl_triple in graph.find(&pat(Some(pref_node), Some(&declare_pred), None)) {
            let decl_node = decl_triple.object();
            let prefix = graph
                .find(&pat(Some(decl_node), Some(&prefix_pred), None))
                .first()
                .and_then(|t| term_string_value(t.object()));
            let namespace = graph
                .find(&pat(Some(decl_node), Some(&namespace_pred), None))
                .first()
                .and_then(|t| term_string_value(t.object()));

            if let (Some(prefix), Some(namespace)) = (prefix, namespace) {
                declarations.push(PrefixDeclaration { prefix, namespace });
            }
        }
    }

    declarations
}

/// Parses an inline shape reference (for sh:not, sh:and, sh:or, sh:node, etc.).
///
/// If the term refers to a known shape, it is parsed recursively.
/// Otherwise, it is parsed as an anonymous inline shape.
fn parse_inline_shape(
    graph: GraphAt<'_>,
    term: &Term,
    all_shape_ids: &HashSet<Term>,
    visiting: &mut HashSet<Term>,
) -> Result<Shape, ShaclError> {
    // Parse the shape definition from the graph (whether or not it was pre-identified)
    parse_shape(graph, term, all_shape_ids, visiting)
}

// =========================================================================
// Metadata helpers
// =========================================================================

fn parse_severity(graph: GraphAt<'_>, shape_id: &Term) -> Severity {
    let pred = Term::iri(SH::SEVERITY);
    graph
        .find(&pat(Some(shape_id), Some(&pred), None))
        .first()
        .and_then(|t| term_as_str(t.object()))
        .map_or(Severity::Violation, |s| match s {
            SH::SEVERITY_WARNING => Severity::Warning,
            SH::SEVERITY_INFO => Severity::Info,
            _ => Severity::Violation,
        })
}

fn parse_deactivated(graph: GraphAt<'_>, shape_id: &Term) -> bool {
    parse_boolean_property(graph, shape_id, SH::DEACTIVATED)
}

fn parse_messages(graph: GraphAt<'_>, shape_id: &Term) -> Vec<String> {
    let pred = Term::iri(SH::MESSAGE);
    graph
        .find(&pat(Some(shape_id), Some(&pred), None))
        .iter()
        .filter_map(|t| term_string_value(t.object()))
        .collect()
}

// =========================================================================
// Primitive helpers
// =========================================================================

fn parse_string_property(
    graph: GraphAt<'_>,
    subject: &Term,
    predicate_iri: &str,
) -> Option<String> {
    let pred = Term::iri(predicate_iri);
    graph
        .find(&pat(Some(subject), Some(&pred), None))
        .first()
        .and_then(|t| term_string_value(t.object()))
}

fn parse_integer_property(graph: GraphAt<'_>, subject: &Term, predicate_iri: &str) -> Option<i64> {
    let pred = Term::iri(predicate_iri);
    graph
        .find(&pat(Some(subject), Some(&pred), None))
        .first()
        .and_then(|t| match t.object() {
            Term::Literal(lit) => lit.as_integer(),
            _ => None,
        })
}

fn parse_boolean_property(graph: GraphAt<'_>, subject: &Term, predicate_iri: &str) -> bool {
    let pred = Term::iri(predicate_iri);
    graph
        .find(&pat(Some(subject), Some(&pred), None))
        .first()
        .and_then(|t| term_string_value(t.object()))
        .is_some_and(|s| s == "true" || s == "1")
}

/// Returns the string representation of a term (IRI value or literal value).
fn term_as_str(term: &Term) -> Option<&str> {
    match term {
        Term::Iri(iri) => Some(iri.as_str()),
        Term::Literal(lit) => Some(lit.value()),
        _ => None,
    }
}

/// Returns the string value of a term (literal value preferred, IRI as fallback).
fn term_string_value(term: &Term) -> Option<String> {
    match term {
        Term::Literal(lit) => Some(lit.value().to_string()),
        Term::Iri(iri) => Some(iri.as_str().to_string()),
        _ => None,
    }
}

// =========================================================================
// RDF list traversal
// =========================================================================

/// Collects the elements of an RDF list (rdf:first/rdf:rest chain) starting
/// from the given head node.
pub(super) fn collect_rdf_list(graph: GraphAt<'_>, head: &Term) -> Vec<Term> {
    let first_pred = Term::iri(RDF::FIRST);
    let rest_pred = Term::iri(RDF::REST);
    let nil = Term::iri(RDF::NIL);

    let mut items = Vec::new();
    let mut current = head.clone();

    // Safety bound to prevent infinite loops on malformed lists
    for _ in 0..10_000 {
        if current == nil {
            break;
        }

        // Get rdf:first value
        let first = graph.find(&pat(Some(&current), Some(&first_pred), None));
        if let Some(t) = first.first() {
            items.push(t.object().clone());
        } else {
            break;
        }

        // Follow rdf:rest
        let rest = graph.find(&pat(Some(&current), Some(&rest_pred), None));
        match rest.first() {
            Some(t) => current = t.object().clone(),
            None => break,
        }
    }

    items
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::rdf::{RdfStore, Term, Triple};

    /// The view the tests read through.
    static LATEST: crate::graph::rdf::RdfView =
        crate::graph::rdf::RdfView::latest(crate::graph::rdf::LatestUse::Test);

    fn shapes_store() -> RdfStore {
        let store = RdfStore::new();

        // Define a simple person shape
        let shape = Term::iri("http://example.org/PersonShape");
        let rdf_type = Term::iri(RDF::TYPE);
        let node_shape = Term::iri(SH::NODE_SHAPE);

        store.insert(Triple::new(shape.clone(), rdf_type, node_shape));

        // Target class
        let target_class = Term::iri(SH::TARGET_CLASS);
        let person = Term::iri("http://example.org/Person");
        store.insert(Triple::new(shape.clone(), target_class, person));

        // Property shape: name (minCount 1, maxCount 1)
        let prop_node = Term::blank("name_prop");
        let property_pred = Term::iri(SH::PROPERTY);
        store.insert(Triple::new(shape.clone(), property_pred, prop_node.clone()));

        let path_pred = Term::iri(SH::PATH);
        let name_iri = Term::iri("http://example.org/name");
        store.insert(Triple::new(prop_node.clone(), path_pred, name_iri));

        let min_count = Term::iri(SH::MIN_COUNT);
        store.insert(Triple::new(
            prop_node.clone(),
            min_count,
            Term::typed_literal("1", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let max_count = Term::iri(SH::MAX_COUNT);
        store.insert(Triple::new(
            prop_node,
            max_count,
            Term::typed_literal("1", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        store
    }

    #[test]
    fn parse_simple_node_shape() {
        let store = shapes_store();
        let shapes = parse_shapes(store.at(&LATEST)).unwrap();

        // Should find the PersonShape
        let person_shapes: Vec<_> = shapes
            .iter()
            .filter(|s| {
                matches!(s.id(), Term::Iri(iri) if iri.as_str() == "http://example.org/PersonShape")
            })
            .collect();
        assert_eq!(person_shapes.len(), 1);

        let shape = &person_shapes[0];
        assert!(!shape.is_deactivated());
        assert_eq!(shape.severity(), Severity::Violation);
    }

    #[test]
    fn parse_target_class() {
        let store = shapes_store();
        let shapes = parse_shapes(store.at(&LATEST)).unwrap();

        let person_shape = shapes
            .iter()
            .find(|s| {
                matches!(s.id(), Term::Iri(iri) if iri.as_str() == "http://example.org/PersonShape")
            })
            .unwrap();

        let targets = person_shape.targets();
        assert_eq!(targets.len(), 1);
        assert!(
            matches!(&targets[0], Target::Class(t) if matches!(t, Term::Iri(iri) if iri.as_str() == "http://example.org/Person"))
        );
    }

    #[test]
    fn parse_property_shape_with_cardinality() {
        let store = shapes_store();
        let shapes = parse_shapes(store.at(&LATEST)).unwrap();

        let person_shape = shapes
            .iter()
            .find(|s| {
                matches!(s.id(), Term::Iri(iri) if iri.as_str() == "http://example.org/PersonShape")
            })
            .unwrap();

        if let Shape::Node(ns) = person_shape {
            assert_eq!(ns.property_shapes.len(), 1);
            let prop = &ns.property_shapes[0];
            assert!(
                matches!(&prop.path, PropertyPath::Predicate(Term::Iri(iri)) if iri.as_str() == "http://example.org/name")
            );

            let has_min = prop
                .constraints
                .iter()
                .any(|c| matches!(c, Constraint::MinCount(1)));
            let has_max = prop
                .constraints
                .iter()
                .any(|c| matches!(c, Constraint::MaxCount(1)));
            assert!(has_min, "Should have minCount 1");
            assert!(has_max, "Should have maxCount 1");
        } else {
            panic!("Expected NodeShape");
        }
    }

    #[test]
    fn parse_inverse_path() {
        let store = RdfStore::new();
        let shape = Term::iri("http://example.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::PROPERTY_SHAPE),
        ));

        // sh:path [ sh:inversePath <http://example.org/knows> ]
        let path_bnode = Term::blank("path1");
        store.insert(Triple::new(shape, Term::iri(SH::PATH), path_bnode.clone()));
        store.insert(Triple::new(
            path_bnode,
            Term::iri(SH::INVERSE_PATH),
            Term::iri("http://example.org/knows"),
        ));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        assert_eq!(shapes.len(), 1);
        if let Shape::Property(ps) = &shapes[0] {
            assert!(
                matches!(&ps.path, PropertyPath::Inverse(inner) if matches!(inner.as_ref(), PropertyPath::Predicate(_)))
            );
        } else {
            panic!("Expected PropertyShape");
        }
    }

    #[test]
    fn parse_severity_warning() {
        let store = RdfStore::new();
        let shape = Term::iri("http://example.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::SEVERITY),
            Term::iri(SH::SEVERITY_WARNING),
        ));
        store.insert(Triple::new(
            shape,
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://example.org/x"),
        ));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        assert_eq!(shapes[0].severity(), Severity::Warning);
    }

    #[test]
    fn parse_deactivated_shape() {
        let store = RdfStore::new();
        let shape = Term::iri("http://example.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::DEACTIVATED),
            Term::typed_literal("true", "http://www.w3.org/2001/XMLSchema#boolean"),
        ));
        store.insert(Triple::new(
            shape,
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://example.org/x"),
        ));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        assert!(shapes[0].is_deactivated());
    }

    #[test]
    fn parse_message() {
        let store = RdfStore::new();
        let shape = Term::iri("http://example.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::MESSAGE),
            Term::literal("Name is required"),
        ));
        store.insert(Triple::new(
            shape,
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://example.org/x"),
        ));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        assert_eq!(shapes[0].messages(), &["Name is required"]);
    }

    #[test]
    fn parse_implicit_shape() {
        let store = RdfStore::new();
        // No rdf:type, just has sh:targetClass
        let shape = Term::iri("http://example.org/S");
        store.insert(Triple::new(
            shape,
            Term::iri(SH::TARGET_CLASS),
            Term::iri("http://example.org/Person"),
        ));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        assert_eq!(shapes.len(), 1);
        // Should be parsed as a node shape (no sh:path)
        assert!(matches!(shapes[0], Shape::Node(_)));
    }

    #[test]
    fn parse_node_kind_constraint() {
        let store = RdfStore::new();
        let shape = Term::iri("http://example.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::NODE_KIND),
            Term::iri(SH::IRI),
        ));
        store.insert(Triple::new(
            shape,
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://example.org/x"),
        ));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        let constraints = shapes[0].constraints();
        assert!(matches!(
            constraints[0],
            Constraint::NodeKind(NodeKindValue::Iri)
        ));
    }

    #[test]
    fn parse_in_constraint() {
        let store = RdfStore::new();
        let shape = Term::iri("http://example.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://example.org/x"),
        ));

        // sh:in ( "a" "b" "c" ) — build an RDF list
        let list1 = Term::blank("l1");
        let list2 = Term::blank("l2");
        let list3 = Term::blank("l3");
        let nil = Term::iri(RDF::NIL);

        store.insert(Triple::new(shape, Term::iri(SH::IN), list1.clone()));
        store.insert(Triple::new(
            list1.clone(),
            Term::iri(RDF::FIRST),
            Term::literal("a"),
        ));
        store.insert(Triple::new(list1, Term::iri(RDF::REST), list2.clone()));
        store.insert(Triple::new(
            list2.clone(),
            Term::iri(RDF::FIRST),
            Term::literal("b"),
        ));
        store.insert(Triple::new(list2, Term::iri(RDF::REST), list3.clone()));
        store.insert(Triple::new(
            list3.clone(),
            Term::iri(RDF::FIRST),
            Term::literal("c"),
        ));
        store.insert(Triple::new(list3, Term::iri(RDF::REST), nil));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        let constraints = shapes[0].constraints();
        let in_constraint = constraints.iter().find(|c| matches!(c, Constraint::In(_)));
        assert!(in_constraint.is_some());
        if let Constraint::In(items) = in_constraint.unwrap() {
            assert_eq!(items.len(), 3);
        }
    }

    #[test]
    fn missing_path_on_property_shape_is_error() {
        let store = RdfStore::new();
        let shape = Term::iri("http://example.org/S");
        store.insert(Triple::new(
            shape,
            Term::iri(RDF::TYPE),
            Term::iri(SH::PROPERTY_SHAPE),
        ));
        // No sh:path

        let result = parse_shapes(store.at(&LATEST));
        assert!(result.is_err());
    }

    // =================================================================
    // Cycle detection tests
    // =================================================================

    #[test]
    fn test_cyclic_shape_reference_detected() {
        let store = RdfStore::new();

        // Shape A: a node shape with sh:node pointing to shape B
        let shape_a = Term::iri("http://example.org/ShapeA");
        let shape_b = Term::iri("http://example.org/ShapeB");

        store.insert(Triple::new(
            shape_a.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape_a.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://example.org/x"),
        ));
        store.insert(Triple::new(
            shape_a.clone(),
            Term::iri(SH::NODE),
            shape_b.clone(),
        ));

        // Shape B: a node shape with sh:node pointing back to shape A
        store.insert(Triple::new(
            shape_b.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape_b.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://example.org/y"),
        ));
        store.insert(Triple::new(shape_b, Term::iri(SH::NODE), shape_a));

        let result = parse_shapes(store.at(&LATEST));
        assert!(result.is_err(), "Should detect cyclic reference");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("Cyclic"),
            "Error should mention 'Cyclic', got: {err_msg}"
        );
    }

    #[test]
    fn test_self_referencing_shape_detected() {
        let store = RdfStore::new();

        // Shape A has sh:not pointing to itself
        let shape_a = Term::iri("http://example.org/SelfShape");

        store.insert(Triple::new(
            shape_a.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape_a.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://example.org/x"),
        ));
        store.insert(Triple::new(shape_a.clone(), Term::iri(SH::NOT), shape_a));

        let result = parse_shapes(store.at(&LATEST));
        assert!(result.is_err(), "Should detect self-referencing shape");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("Cyclic"),
            "Error should mention 'Cyclic', got: {err_msg}"
        );
    }

    // =================================================================
    // sh:nodeKind IRI validation tests
    // =================================================================

    #[test]
    fn test_node_kind_rejects_literal_value() {
        let store = RdfStore::new();
        let shape = Term::iri("http://example.org/S");

        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://example.org/x"),
        ));
        // Provide sh:nodeKind as a literal string instead of an IRI
        store.insert(Triple::new(
            shape,
            Term::iri(SH::NODE_KIND),
            Term::literal("http://www.w3.org/ns/shacl#IRI"),
        ));

        let result = parse_shapes(store.at(&LATEST));
        assert!(result.is_err(), "Literal sh:nodeKind should be rejected");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("must be an IRI"),
            "Error should mention 'must be an IRI', got: {err_msg}"
        );
    }

    #[test]
    fn test_node_kind_accepts_iri_value() {
        let store = RdfStore::new();
        let shape = Term::iri("http://example.org/S");

        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://example.org/x"),
        ));
        // Correctly provide sh:nodeKind as an IRI
        store.insert(Triple::new(
            shape,
            Term::iri(SH::NODE_KIND),
            Term::iri(SH::IRI),
        ));

        let shapes = parse_shapes(store.at(&LATEST)).expect("IRI-valued sh:nodeKind should parse");
        let constraints = shapes[0].constraints();
        assert!(
            constraints
                .iter()
                .any(|c| matches!(c, Constraint::NodeKind(NodeKindValue::Iri))),
            "Should contain NodeKind::Iri constraint"
        );
    }

    // =================================================================
    // Logical constraint parsing with visiting
    // =================================================================

    #[test]
    fn test_parse_sh_not_with_inline_shape() {
        let store = RdfStore::new();
        let shape = Term::iri("http://example.org/OuterShape");
        let inner = Term::blank("inner_not");

        // Outer shape
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::TARGET_CLASS),
            Term::iri("http://example.org/Person"),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::NOT),
            inner.clone(),
        ));

        // Inline shape referenced by sh:not, with a minCount constraint
        store.insert(Triple::new(
            inner.clone(),
            Term::iri(SH::MIN_COUNT),
            Term::typed_literal("2", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let shapes =
            parse_shapes(store.at(&LATEST)).expect("sh:not with inline shape should parse");

        // Find the outer shape (the one with a target)
        let outer = shapes
            .iter()
            .find(|s| {
                matches!(s.id(), Term::Iri(iri) if iri.as_str() == "http://example.org/OuterShape")
            })
            .expect("Should find OuterShape");

        let has_not = outer
            .constraints()
            .iter()
            .any(|c| matches!(c, Constraint::Not(_)));
        assert!(has_not, "OuterShape should have a sh:not constraint");
    }

    #[test]
    fn test_parse_sh_and_with_list() {
        let store = RdfStore::new();
        let shape = Term::iri("http://example.org/CombinedShape");

        // Two inline shapes referenced in the RDF list
        let inner_a = Term::blank("andA");
        let inner_b = Term::blank("andB");

        // Main shape
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::TARGET_CLASS),
            Term::iri("http://example.org/Person"),
        ));

        // sh:and points to the head of an RDF list
        let list1 = Term::blank("list1");
        let list2 = Term::blank("list2");
        let nil = Term::iri(RDF::NIL);

        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::AND),
            list1.clone(),
        ));
        store.insert(Triple::new(
            list1.clone(),
            Term::iri(RDF::FIRST),
            inner_a.clone(),
        ));
        store.insert(Triple::new(list1, Term::iri(RDF::REST), list2.clone()));
        store.insert(Triple::new(
            list2.clone(),
            Term::iri(RDF::FIRST),
            inner_b.clone(),
        ));
        store.insert(Triple::new(list2, Term::iri(RDF::REST), nil));

        // Give each inline shape a distinct constraint so we can verify both are parsed
        store.insert(Triple::new(
            inner_a,
            Term::iri(SH::MIN_COUNT),
            Term::typed_literal("1", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        store.insert(Triple::new(
            inner_b,
            Term::iri(SH::MAX_COUNT),
            Term::typed_literal("5", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let shapes = parse_shapes(store.at(&LATEST)).expect("sh:and with RDF list should parse");

        let combined = shapes
            .iter()
            .find(|s| {
                matches!(s.id(), Term::Iri(iri) if iri.as_str() == "http://example.org/CombinedShape")
            })
            .expect("Should find CombinedShape");

        let and_constraint = combined
            .constraints()
            .iter()
            .find(|c| matches!(c, Constraint::And(_)));
        assert!(and_constraint.is_some(), "Should have a sh:and constraint");

        if let Some(Constraint::And(members)) = and_constraint {
            assert_eq!(members.len(), 2, "sh:and list should contain two shapes");
        }
    }

    #[test]
    fn parse_zero_or_more_path() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::PROPERTY_SHAPE),
        ));
        let path_bnode = Term::blank("p1");
        store.insert(Triple::new(shape, Term::iri(SH::PATH), path_bnode.clone()));
        store.insert(Triple::new(
            path_bnode,
            Term::iri(SH::ZERO_OR_MORE_PATH),
            Term::iri("http://ex.org/knows"),
        ));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        assert_eq!(shapes.len(), 1);
        if let Shape::Property(ps) = &shapes[0] {
            assert!(
                matches!(&ps.path, PropertyPath::ZeroOrMore(inner) if matches!(inner.as_ref(), PropertyPath::Predicate(_)))
            );
        } else {
            panic!("Expected PropertyShape");
        }
    }

    #[test]
    fn parse_one_or_more_path() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::PROPERTY_SHAPE),
        ));
        let path_bnode = Term::blank("p1");
        store.insert(Triple::new(shape, Term::iri(SH::PATH), path_bnode.clone()));
        store.insert(Triple::new(
            path_bnode,
            Term::iri(SH::ONE_OR_MORE_PATH),
            Term::iri("http://ex.org/knows"),
        ));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        if let Shape::Property(ps) = &shapes[0] {
            assert!(matches!(&ps.path, PropertyPath::OneOrMore(_)));
        } else {
            panic!("Expected PropertyShape");
        }
    }

    #[test]
    fn parse_zero_or_one_path() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::PROPERTY_SHAPE),
        ));
        let path_bnode = Term::blank("p1");
        store.insert(Triple::new(shape, Term::iri(SH::PATH), path_bnode.clone()));
        store.insert(Triple::new(
            path_bnode,
            Term::iri(SH::ZERO_OR_ONE_PATH),
            Term::iri("http://ex.org/knows"),
        ));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        if let Shape::Property(ps) = &shapes[0] {
            assert!(matches!(&ps.path, PropertyPath::ZeroOrOne(_)));
        } else {
            panic!("Expected PropertyShape");
        }
    }

    #[test]
    fn parse_sequence_path() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::PROPERTY_SHAPE),
        ));

        // sh:path is an RDF list (sequence path): ( ex:knows ex:name )
        let l1 = Term::blank("l1");
        let l2 = Term::blank("l2");
        let nil = Term::iri(RDF::NIL);
        store.insert(Triple::new(shape, Term::iri(SH::PATH), l1.clone()));
        store.insert(Triple::new(
            l1.clone(),
            Term::iri(RDF::FIRST),
            Term::iri("http://ex.org/knows"),
        ));
        store.insert(Triple::new(l1, Term::iri(RDF::REST), l2.clone()));
        store.insert(Triple::new(
            l2.clone(),
            Term::iri(RDF::FIRST),
            Term::iri("http://ex.org/name"),
        ));
        store.insert(Triple::new(l2, Term::iri(RDF::REST), nil));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        if let Shape::Property(ps) = &shapes[0] {
            if let PropertyPath::Sequence(steps) = &ps.path {
                assert_eq!(steps.len(), 2);
            } else {
                panic!("Expected Sequence path, got {:?}", ps.path);
            }
        } else {
            panic!("Expected PropertyShape");
        }
    }

    #[test]
    fn parse_alternative_path() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::PROPERTY_SHAPE),
        ));

        // sh:path [ sh:alternativePath ( ex:name ex:label ) ]
        let path_bnode = Term::blank("p1");
        store.insert(Triple::new(shape, Term::iri(SH::PATH), path_bnode.clone()));
        let l1 = Term::blank("al1");
        let l2 = Term::blank("al2");
        let nil = Term::iri(RDF::NIL);
        store.insert(Triple::new(
            path_bnode,
            Term::iri(SH::ALTERNATIVE_PATH),
            l1.clone(),
        ));
        store.insert(Triple::new(
            l1.clone(),
            Term::iri(RDF::FIRST),
            Term::iri("http://ex.org/name"),
        ));
        store.insert(Triple::new(l1, Term::iri(RDF::REST), l2.clone()));
        store.insert(Triple::new(
            l2.clone(),
            Term::iri(RDF::FIRST),
            Term::iri("http://ex.org/label"),
        ));
        store.insert(Triple::new(l2, Term::iri(RDF::REST), nil));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        if let Shape::Property(ps) = &shapes[0] {
            if let PropertyPath::Alternative(alts) = &ps.path {
                assert_eq!(alts.len(), 2);
            } else {
                panic!("Expected Alternative path, got {:?}", ps.path);
            }
        } else {
            panic!("Expected PropertyShape");
        }
    }

    #[test]
    fn parse_pattern_with_flags() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://ex.org/x"),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::PATTERN),
            Term::literal("^[a-z]+$"),
        ));
        store.insert(Triple::new(shape, Term::iri(SH::FLAGS), Term::literal("i")));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        let c = shapes[0]
            .constraints()
            .iter()
            .find(|c| matches!(c, Constraint::Pattern { .. }));
        assert!(c.is_some());
        if let Some(Constraint::Pattern { pattern, flags }) = c {
            assert_eq!(pattern, "^[a-z]+$");
            assert_eq!(flags.as_deref(), Some("i"));
        }
    }

    #[test]
    fn parse_language_in() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://ex.org/x"),
        ));

        let l1 = Term::blank("ll1");
        let l2 = Term::blank("ll2");
        let nil = Term::iri(RDF::NIL);
        store.insert(Triple::new(shape, Term::iri(SH::LANGUAGE_IN), l1.clone()));
        store.insert(Triple::new(
            l1.clone(),
            Term::iri(RDF::FIRST),
            Term::literal("en"),
        ));
        store.insert(Triple::new(l1, Term::iri(RDF::REST), l2.clone()));
        store.insert(Triple::new(
            l2.clone(),
            Term::iri(RDF::FIRST),
            Term::literal("de"),
        ));
        store.insert(Triple::new(l2, Term::iri(RDF::REST), nil));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        let c = shapes[0]
            .constraints()
            .iter()
            .find(|c| matches!(c, Constraint::LanguageIn(_)));
        assert!(c.is_some());
        if let Some(Constraint::LanguageIn(langs)) = c {
            assert_eq!(langs, &["en", "de"]);
        }
    }

    #[test]
    fn parse_unique_lang() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://ex.org/x"),
        ));
        store.insert(Triple::new(
            shape,
            Term::iri(SH::UNIQUE_LANG),
            Term::typed_literal("true", "http://www.w3.org/2001/XMLSchema#boolean"),
        ));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        assert!(
            shapes[0]
                .constraints()
                .iter()
                .any(|c| matches!(c, Constraint::UniqueLang))
        );
    }

    #[test]
    fn parse_closed_with_ignored() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://ex.org/x"),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::CLOSED),
            Term::typed_literal("true", "http://www.w3.org/2001/XMLSchema#boolean"),
        ));

        let l1 = Term::blank("ig1");
        let nil = Term::iri(RDF::NIL);
        store.insert(Triple::new(
            shape,
            Term::iri(SH::IGNORED_PROPERTIES),
            l1.clone(),
        ));
        store.insert(Triple::new(
            l1.clone(),
            Term::iri(RDF::FIRST),
            Term::iri("http://ex.org/age"),
        ));
        store.insert(Triple::new(l1, Term::iri(RDF::REST), nil));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        let c = shapes[0]
            .constraints()
            .iter()
            .find(|c| matches!(c, Constraint::Closed { .. }));
        assert!(c.is_some());
        if let Some(Constraint::Closed { ignored_properties }) = c {
            assert_eq!(ignored_properties.len(), 1);
        }
    }

    #[test]
    fn parse_qualified_value_shape() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://ex.org/x"),
        ));

        let inner_shape = Term::blank("inner");
        store.insert(Triple::new(
            inner_shape.clone(),
            Term::iri(SH::NODE_KIND),
            Term::iri(SH::IRI),
        ));

        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::QUALIFIED_VALUE_SHAPE),
            inner_shape,
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::QUALIFIED_MIN_COUNT),
            Term::typed_literal("1", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        store.insert(Triple::new(
            shape,
            Term::iri(SH::QUALIFIED_MAX_COUNT),
            Term::typed_literal("3", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        let c = shapes[0]
            .constraints()
            .iter()
            .find(|c| matches!(c, Constraint::QualifiedValueShape { .. }));
        assert!(c.is_some());
        if let Some(Constraint::QualifiedValueShape {
            min_count,
            max_count,
            ..
        }) = c
        {
            assert_eq!(*min_count, Some(1));
            assert_eq!(*max_count, Some(3));
        }
    }

    #[test]
    fn parse_not_constraint() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://ex.org/x"),
        ));

        let inner = Term::blank("notShape");
        store.insert(Triple::new(
            inner.clone(),
            Term::iri(SH::NODE_KIND),
            Term::iri(SH::LITERAL),
        ));
        store.insert(Triple::new(shape, Term::iri(SH::NOT), inner));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        assert!(
            shapes[0]
                .constraints()
                .iter()
                .any(|c| matches!(c, Constraint::Not(_)))
        );
    }

    #[test]
    fn parse_or_constraint() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://ex.org/x"),
        ));

        let s1 = Term::blank("or1");
        let s2 = Term::blank("or2");
        store.insert(Triple::new(
            s1.clone(),
            Term::iri(SH::NODE_KIND),
            Term::iri(SH::IRI),
        ));
        store.insert(Triple::new(
            s2.clone(),
            Term::iri(SH::NODE_KIND),
            Term::iri(SH::LITERAL),
        ));

        let l1 = Term::blank("orl1");
        let l2 = Term::blank("orl2");
        let nil = Term::iri(RDF::NIL);
        store.insert(Triple::new(shape, Term::iri(SH::OR), l1.clone()));
        store.insert(Triple::new(l1.clone(), Term::iri(RDF::FIRST), s1));
        store.insert(Triple::new(l1, Term::iri(RDF::REST), l2.clone()));
        store.insert(Triple::new(l2.clone(), Term::iri(RDF::FIRST), s2));
        store.insert(Triple::new(l2, Term::iri(RDF::REST), nil));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        let c = shapes[0]
            .constraints()
            .iter()
            .find(|c| matches!(c, Constraint::Or(_)));
        assert!(c.is_some());
        if let Some(Constraint::Or(members)) = c {
            assert_eq!(members.len(), 2);
        }
    }

    #[test]
    fn parse_xone_constraint() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://ex.org/x"),
        ));

        let s1 = Term::blank("x1");
        store.insert(Triple::new(
            s1.clone(),
            Term::iri(SH::NODE_KIND),
            Term::iri(SH::IRI),
        ));

        let l1 = Term::blank("xl1");
        let nil = Term::iri(RDF::NIL);
        store.insert(Triple::new(shape, Term::iri(SH::XONE), l1.clone()));
        store.insert(Triple::new(l1.clone(), Term::iri(RDF::FIRST), s1));
        store.insert(Triple::new(l1, Term::iri(RDF::REST), nil));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        assert!(
            shapes[0]
                .constraints()
                .iter()
                .any(|c| matches!(c, Constraint::Xone(_)))
        );
    }

    #[test]
    fn parse_sparql_constraint() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://ex.org/x"),
        ));

        let sparql_node = Term::blank("sq1");
        store.insert(Triple::new(
            shape,
            Term::iri(SH::SPARQL),
            sparql_node.clone(),
        ));
        store.insert(Triple::new(
            sparql_node.clone(),
            Term::iri(SH::SELECT),
            Term::literal("SELECT $this WHERE { $this ?p ?o }"),
        ));
        store.insert(Triple::new(
            sparql_node,
            Term::iri(SH::MESSAGE),
            Term::literal("Custom violation"),
        ));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        let c = shapes[0]
            .constraints()
            .iter()
            .find(|c| matches!(c, Constraint::Sparql(_)));
        assert!(c.is_some());
        if let Some(Constraint::Sparql(sc)) = c {
            assert!(sc.select.contains("$this"));
            assert_eq!(sc.message.as_deref(), Some("Custom violation"));
        }
    }

    #[test]
    fn parse_sparql_missing_select_errors() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://ex.org/x"),
        ));

        let sparql_node = Term::blank("sq1");
        store.insert(Triple::new(shape, Term::iri(SH::SPARQL), sparql_node));
        // Missing sh:select

        let result = parse_shapes(store.at(&LATEST));
        assert!(result.is_err(), "sh:sparql without sh:select should error");
    }

    #[test]
    fn parse_equals_disjoint_less_than() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://ex.org/x"),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::EQUALS),
            Term::iri("http://ex.org/label"),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::DISJOINT),
            Term::iri("http://ex.org/nick"),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::LESS_THAN),
            Term::iri("http://ex.org/end"),
        ));
        store.insert(Triple::new(
            shape,
            Term::iri(SH::LESS_THAN_OR_EQUALS),
            Term::iri("http://ex.org/max"),
        ));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        let cs = shapes[0].constraints();
        assert!(cs.iter().any(|c| matches!(c, Constraint::Equals(_))));
        assert!(cs.iter().any(|c| matches!(c, Constraint::Disjoint(_))));
        assert!(cs.iter().any(|c| matches!(c, Constraint::LessThan(_))));
        assert!(
            cs.iter()
                .any(|c| matches!(c, Constraint::LessThanOrEquals(_)))
        );
    }

    #[test]
    fn parse_value_range_constraints() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://ex.org/x"),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::MIN_EXCLUSIVE),
            Term::typed_literal("0", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::MAX_EXCLUSIVE),
            Term::typed_literal("100", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::MIN_INCLUSIVE),
            Term::typed_literal("1", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        store.insert(Triple::new(
            shape,
            Term::iri(SH::MAX_INCLUSIVE),
            Term::typed_literal("99", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        let cs = shapes[0].constraints();
        assert!(cs.iter().any(|c| matches!(c, Constraint::MinExclusive(_))));
        assert!(cs.iter().any(|c| matches!(c, Constraint::MaxExclusive(_))));
        assert!(cs.iter().any(|c| matches!(c, Constraint::MinInclusive(_))));
        assert!(cs.iter().any(|c| matches!(c, Constraint::MaxInclusive(_))));
    }

    #[test]
    fn parse_min_max_length() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://ex.org/x"),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::MIN_LENGTH),
            Term::typed_literal("3", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        store.insert(Triple::new(
            shape,
            Term::iri(SH::MAX_LENGTH),
            Term::typed_literal("50", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        let cs = shapes[0].constraints();
        assert!(cs.iter().any(|c| matches!(c, Constraint::MinLength(3))));
        assert!(cs.iter().any(|c| matches!(c, Constraint::MaxLength(50))));
    }

    #[test]
    fn parse_has_value() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://ex.org/x"),
        ));
        store.insert(Triple::new(
            shape,
            Term::iri(SH::HAS_VALUE),
            Term::literal("required"),
        ));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        assert!(
            shapes[0]
                .constraints()
                .iter()
                .any(|c| matches!(c, Constraint::HasValue(_)))
        );
    }

    #[test]
    fn parse_shape_node_constraint() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::NODE_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::TARGET_NODE),
            Term::iri("http://ex.org/x"),
        ));

        let inner = Term::blank("nodeRef");
        store.insert(Triple::new(
            inner.clone(),
            Term::iri(SH::NODE_KIND),
            Term::iri(SH::IRI),
        ));
        store.insert(Triple::new(shape, Term::iri(SH::NODE), inner));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        assert!(
            shapes[0]
                .constraints()
                .iter()
                .any(|c| matches!(c, Constraint::ShapeNode(_)))
        );
    }

    #[test]
    fn parse_name_and_description() {
        let store = RdfStore::new();
        let shape = Term::iri("http://ex.org/S");
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(RDF::TYPE),
            Term::iri(SH::PROPERTY_SHAPE),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::PATH),
            Term::iri("http://ex.org/name"),
        ));
        store.insert(Triple::new(
            shape.clone(),
            Term::iri(SH::NAME),
            Term::literal("Full Name"),
        ));
        store.insert(Triple::new(
            shape,
            Term::iri(SH::DESCRIPTION),
            Term::literal("The person's full name"),
        ));

        let shapes = parse_shapes(store.at(&LATEST)).unwrap();
        if let Shape::Property(ps) = &shapes[0] {
            assert_eq!(ps.name.as_deref(), Some("Full Name"));
            assert_eq!(ps.description.as_deref(), Some("The person's full name"));
        } else {
            panic!("Expected PropertyShape");
        }
    }
}
