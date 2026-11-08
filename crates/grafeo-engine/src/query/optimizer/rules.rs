//! The planner rules a session can turn off.
//!
//! A rule changes how a query runs, not its rows, so a test can run one query
//! with a rule on and off (and against a brute-force answer) and compare:
//!
//! - When both succeed, the rows are the same.
//! - A check that cannot raise a condition may run earlier, so with the rule on
//!   a query may succeed where it fails with the rule off, on a row another
//!   conjunct rejects, and which condition is raised may differ (ISO/IEC 39075
//!   5.3.2.4 leaves the order of evaluation to the implementation, US008).
//! - A rule never moves an expression that can raise ahead of a check written
//!   before it: a query that succeeds with the rule off does not fail with it
//!   on. A check that cannot raise may move earlier past one that can, never later.
//!
//! Every rule is on outside such tests; this is no tuning surface and may change
//! in any release. The rules are read while planning, each guards one shortcut,
//! and off it falls back to a scan and a filter.
//!
//! With every rule off a query still scans a label, runs its filters (pushed down
//! as far as they go), expands, joins by hash, projects away unread columns, and
//! answers `EXISTS` with a semi-join or per row. Vector and text index searches,
//! which a query asks for by name, also stay.

/// A planner rule that [`PlannerRules`] turns on or off.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlannerRule {
    /// The cost-based order of a pattern's joins; off, they run as written.
    JoinOrder,
    /// An expand whose far end a filter pins starts at that end.
    SoughtEnd,
    /// Counts from the adjacency and factorized chains over expands; off,
    /// every row is built, then counted.
    FactorizedExecution,
    /// A count of one label's nodes from the store's own count.
    LabelCount,
    /// A variable-length expand whose rows only reach a consumer that ignores
    /// duplicates runs as a reachability search; off, every walk is listed.
    Reachability,
    /// A hop that closes a cycle looks up the edges to its known target
    /// instead of listing every edge.
    ExpandInto,
    /// A hop whose far end a key pins finds that end first: nothing to read
    /// when no node holds the key, a lookup when one does.
    PinnedFarEnd,
    /// The last two hops of a cycle intersect the lists of the two bound nodes
    /// they meet between, instead of walking one of them.
    IntersectExpand,
    /// A filter on an indexed property, over a scan with no input, reads the
    /// nodes its value or `IN` list finds in the index while planning.
    IndexLookup,
    /// A range filter on a property reads the nodes in that range.
    IndexRange,
    /// A filter on an indexed property, over a scan per input row, looks the
    /// key up per row.
    KeySeek,
    /// A filter on `id()` looks the node up per row.
    IdSeek,
    /// A filter a zone map proves false reads nothing.
    ZoneMaps,
    /// A filter that equates the scanned node's values with the row's scans
    /// once and joins by hash.
    ValueJoin,
    /// `EXISTS` and `COUNT` of one hop from a row's node read that node's edges
    /// inline.
    ExistsEdgeCheck,
    /// An equality on a key no index serves reads the key's column instead
    /// of filtering every node of the scan.
    KeyColumnLookup,
    /// An `EXISTS` that a run without the row answers runs once; off, every `EXISTS` runs per row,
    /// which must give the same answers.
    ExistsRunOnce,
}

impl PlannerRule {
    /// Every rule.
    pub const ALL: [Self; 17] = [
        Self::JoinOrder,
        Self::SoughtEnd,
        Self::FactorizedExecution,
        Self::LabelCount,
        Self::Reachability,
        Self::ExpandInto,
        Self::PinnedFarEnd,
        Self::IntersectExpand,
        Self::IndexLookup,
        Self::IndexRange,
        Self::KeySeek,
        Self::IdSeek,
        Self::ZoneMaps,
        Self::ValueJoin,
        Self::ExistsEdgeCheck,
        Self::KeyColumnLookup,
        Self::ExistsRunOnce,
    ];

    /// The rule's bit in a [`PlannerRules`].
    const fn bit(self) -> u32 {
        1 << self as u32
    }
}

/// Which [`PlannerRule`]s a session's queries are planned with.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlannerRules(u32);

impl PlannerRules {
    /// Every rule on, as every session starts.
    #[must_use]
    pub const fn all() -> Self {
        let mut bits = 0;
        let mut index = 0;
        while index < PlannerRule::ALL.len() {
            bits |= PlannerRule::ALL[index].bit();
            index += 1;
        }
        Self(bits)
    }

    /// Every rule off: the plain plan.
    #[must_use]
    pub const fn plain() -> Self {
        Self(0)
    }

    /// These rules with `rule` turned on or off.
    #[must_use]
    pub const fn with(self, rule: PlannerRule, enabled: bool) -> Self {
        if enabled {
            Self(self.0 | rule.bit())
        } else {
            Self(self.0 & !rule.bit())
        }
    }

    /// Whether `rule` is on.
    #[must_use]
    pub const fn has(self, rule: PlannerRule) -> bool {
        self.0 & rule.bit() != 0
    }
}

impl Default for PlannerRules {
    fn default() -> Self {
        Self::all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_rule_turns_on_and_off_alone() {
        for rule in PlannerRule::ALL {
            let off = PlannerRules::all().with(rule, false);
            let on = PlannerRules::plain().with(rule, true);
            for other in PlannerRule::ALL {
                assert_eq!(off.has(other), other != rule);
                assert_eq!(on.has(other), other == rule);
            }
        }
    }

    /// What a session's `PROFILE` of `query` prints, and its one count.
    #[cfg(feature = "gql")]
    fn profile(session: &crate::session::Session, query: &str) -> (String, i64) {
        let plan = session.execute(&format!("PROFILE {query}")).unwrap();
        let plan = format!("{:?}", plan.rows());
        let count = session.execute(query).unwrap().rows()[0][0].clone();
        (plan, count.as_int64().unwrap())
    }

    /// Off, the same query lists its rows where it read counts, and answers
    /// the same.
    #[cfg(feature = "gql")]
    #[test]
    fn a_rule_that_is_off_changes_the_plan_and_not_the_answer() {
        let db = crate::GrafeoDB::new_in_memory();
        let mut plain = db.session();
        plain.set_planner_rules(PlannerRules::all().with(PlannerRule::FactorizedExecution, false));
        let usual = db.session();
        usual
            .execute(
                "INSERT (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person {name: 'Gus'}),
                        (b)-[:KNOWS]->(c:Person {name: 'Vincent'}),
                        (b)-[:KNOWS]->(:Person {name: 'Jules'}),
                        (c)-[:KNOWS]->(a)",
            )
            .unwrap();
        let query = "MATCH (:Person)-[:KNOWS]->(:Person)-[:KNOWS]->() RETURN count(*)";
        let (plain_plan, plain_count) = profile(&plain, query);
        let (usual_plan, usual_count) = profile(&usual, query);
        assert!(!plain_plan.contains("EdgeCount"), "{plain_plan}");
        assert!(usual_plan.contains("EdgeCount"), "{usual_plan}");
        assert_eq!(plain_count, 4);
        assert_eq!(usual_count, 4);
    }
}
