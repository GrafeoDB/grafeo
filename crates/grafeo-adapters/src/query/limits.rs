//! Limits every parser enforces on the statements it accepts.
//!
//! Every stage after the parser (translation, binding, optimization,
//! planning, expression evaluation) walks a statement recursively, one stack
//! frame or more per level of nesting. A statement nested deeper than the
//! stack allows used to overflow it: the process aborted, and a WebAssembly
//! module trapped and stayed broken (#573). The parsers bound the nesting
//! instead, so a deeper statement fails with an error that names the limit.

/// How deep a statement may nest.
///
/// An expression nests one level in the expression, list, map or clause it
/// is part of (so do parentheses around it), and so do `NOT`, a sign and
/// each operator of a chain of other binary operators (`a + b + c` nests two
/// levels, as `(a + b) + c`). A chain of `AND`, `OR` or `XOR` is read as a
/// balanced tree, so it nests as many levels as the tree is deep: 14 for
/// 10,000 terms. A function call, `CASE` and a subquery nest one more level,
/// as parsing and planning them take more stack. A long flat list
/// (`[1, 2, 3, ...]`) nests one level however long it is. A deeper statement
/// fails with a syntax error naming this limit.
///
/// The limit keeps the stack a statement can take at every stage well below
/// the smallest stack Grafeo runs on (1 MiB: the main thread on Windows and
/// the WebAssembly builds): about 470 KiB at the most on x86-64, 415 KiB in
/// WebAssembly. The engine's tests run every kind of nesting at the limit on
/// a 640 KiB stack.
pub const MAX_NESTING_DEPTH: u32 = 64;

#[cfg(any(
    feature = "gql",
    feature = "cypher",
    feature = "sparql",
    feature = "gremlin",
    feature = "graphql",
    feature = "sql-pgq"
))]
/// The message of the error a parser returns for a statement that nests
/// deeper than [`MAX_NESTING_DEPTH`].
#[must_use]
pub(crate) fn nesting_error_message() -> String {
    format!("Maximum nesting depth of {MAX_NESTING_DEPTH} exceeded")
}

#[cfg(any(
    feature = "gql",
    feature = "cypher",
    feature = "sparql",
    feature = "gremlin",
    feature = "graphql",
    feature = "sql-pgq"
))]
/// How deep the statement a parser reads nests, as [`MAX_NESTING_DEPTH`]
/// counts it.
///
/// A parser [`enter`](Self::enter)s a level around each construct it parses
/// recursively, and brackets each loop that builds a chain of binary
/// operators with [`begin_chain`](Self::begin_chain) and
/// [`end_chain`](Self::end_chain). A chain is built without recursion, but
/// every later stage recurses into the tree it builds: for an operator that
/// is not associative the parser builds it from the left, as deep as the
/// chain is long, and calls [`link`](Self::link) for each operator; for `AND`,
/// `OR` and `XOR` it collects the operands with
/// [`take_operand`](Self::take_operand) and builds a balanced tree with
/// [`join_balanced`](Self::join_balanced). A parser that backtracks restores a
/// copy taken where it started, as it restores its lexer.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Nesting {
    /// The levels the parser is inside now.
    depth: u32,
    /// The deepest level the syntax parsed since the innermost open chain
    /// began reaches.
    reached: u32,
}

#[cfg(any(
    feature = "gql",
    feature = "cypher",
    feature = "sparql",
    feature = "sql-pgq"
))]
/// A chain of binary operators being parsed (see [`Nesting::begin_chain`]).
#[derive(Debug)]
#[must_use = "a chain ends with `Nesting::end_chain`"]
pub(crate) struct Chain {
    /// The deepest level reached before the chain began.
    outer: u32,
}

#[cfg(any(
    feature = "gql",
    feature = "cypher",
    feature = "sparql",
    feature = "gremlin",
    feature = "graphql",
    feature = "sql-pgq"
))]
impl Nesting {
    /// Enters one level of nesting. Returns `false` when that is deeper than
    /// [`MAX_NESTING_DEPTH`]: the parser then fails.
    #[must_use]
    pub(crate) fn enter(&mut self) -> bool {
        self.depth += 1;
        self.reached = self.reached.max(self.depth);
        self.depth <= MAX_NESTING_DEPTH
    }

    /// Leaves the level [`enter`](Self::enter) entered.
    pub(crate) fn exit(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }
}

/// The chains of binary operators, which the languages with infix operators
/// parse.
#[cfg(any(
    feature = "gql",
    feature = "cypher",
    feature = "sparql",
    feature = "sql-pgq"
))]
impl Nesting {
    /// Begins a chain of binary operators at the current level, before its
    /// first operand is parsed.
    pub(crate) fn begin_chain(&mut self) -> Chain {
        Chain {
            outer: std::mem::replace(&mut self.reached, self.depth),
        }
    }

    /// Counts the operator that joins the chain so far with the operand just
    /// parsed: it is one level above the deeper of the two. Returns `false`
    /// when that is deeper than [`MAX_NESTING_DEPTH`]: the parser then fails.
    #[must_use]
    pub(crate) fn link(&mut self) -> bool {
        self.reached += 1;
        self.reached <= MAX_NESTING_DEPTH
    }

    /// Ends `chain`: what encloses it reaches at least as deep as the chain.
    pub(crate) fn end_chain(&mut self, chain: Chain) {
        self.reached = self.reached.max(chain.outer);
    }

    /// The deepest level the operand just parsed reaches, for
    /// [`join_balanced`](Self::join_balanced); the next operand of the chain
    /// starts again at the current level.
    pub(crate) fn take_operand(&mut self) -> u32 {
        std::mem::replace(&mut self.reached, self.depth)
    }

    /// Joins the operands of a chain of one associative operator (`AND`,
    /// `OR`, `XOR`) into a balanced tree, in their order, with `join`, each
    /// with the level [`take_operand`](Self::take_operand) returned for it.
    /// Returns `None` when the tree is deeper than [`MAX_NESTING_DEPTH`]: the
    /// parser then fails.
    ///
    /// Every stage after the parser recurses into the tree, so a balanced one
    /// keeps a chain of any length shallow; evaluating it in order gives what
    /// evaluating it from the left does, as the operator is associative.
    pub(crate) fn join_balanced<T>(
        &mut self,
        operands: Vec<(T, u32)>,
        join: impl FnMut(T, T) -> T,
    ) -> Option<T> {
        let (tree, level) = balanced(operands, join)?;
        self.reached = self.reached.max(level);
        (level <= MAX_NESTING_DEPTH).then_some(tree)
    }
}

#[cfg(any(
    feature = "gql",
    feature = "cypher",
    feature = "sparql",
    feature = "sql-pgq"
))]
/// Joins `operands` pairwise, level by level, into a balanced tree that keeps
/// their order, and returns it with its level: one above the deeper of the
/// two it joins, for each join. `None` for no operands.
fn balanced<T>(operands: Vec<(T, u32)>, mut join: impl FnMut(T, T) -> T) -> Option<(T, u32)> {
    let mut layer = operands;
    while layer.len() > 1 {
        let mut joined = Vec::with_capacity(layer.len().div_ceil(2));
        let mut operands = layer.into_iter();
        while let Some((left, left_level)) = operands.next() {
            joined.push(match operands.next() {
                Some((right, right_level)) => (join(left, right), left_level.max(right_level) + 1),
                None => (left, left_level),
            });
        }
        layer = joined;
    }
    layer.pop()
}

#[cfg(all(
    test,
    any(
        feature = "gql",
        feature = "cypher",
        feature = "sparql",
        feature = "sql-pgq"
    )
))]
mod tests {
    use super::*;

    /// The deepest level `parse` reaches, starting at the top.
    fn reached(parse: impl FnOnce(&mut Nesting) -> bool) -> Option<u32> {
        let mut nesting = Nesting::default();
        let chain = nesting.begin_chain();
        let ok = parse(&mut nesting);
        let reached = nesting.reached;
        nesting.end_chain(chain);
        ok.then_some(reached)
    }

    #[test]
    fn a_chain_is_as_deep_as_its_operators() {
        // a OR b OR c: two operators.
        let depth = reached(|n| {
            let chain = n.begin_chain();
            let ok = n.link() && n.link();
            n.end_chain(chain);
            ok
        });
        assert_eq!(depth, Some(2));
    }

    #[test]
    fn a_chain_over_a_parenthesized_operand_counts_both() {
        // (a OR b) AND c: the parentheses, the OR inside, the AND outside.
        let depth = reached(|n| {
            let and = n.begin_chain();
            let ok = n.enter() && {
                let or = n.begin_chain();
                let ok = n.link();
                n.end_chain(or);
                ok
            };
            n.exit();
            let ok = ok && n.link();
            n.end_chain(and);
            ok
        });
        assert_eq!(depth, Some(3));
    }

    #[test]
    fn siblings_take_the_deeper_one_not_their_sum() {
        // f((a OR b), (c OR d)): two arguments of depth 2 each, in a call.
        let depth = reached(|n| {
            let ok = n.enter();
            let mut ok = ok;
            for _ in 0..2 {
                ok &= n.enter();
                let or = n.begin_chain();
                ok &= n.link();
                n.end_chain(or);
                n.exit();
            }
            n.exit();
            ok
        });
        assert_eq!(depth, Some(3));
    }

    /// Joins `count` leaves `0..count` at level 0 into a tree, written as text.
    fn balanced_leaves(count: usize) -> Option<(String, u32)> {
        let leaves = (0..count).map(|leaf| (leaf.to_string(), 0)).collect();
        balanced(leaves, |left, right| format!("({left} {right})"))
    }

    #[test]
    fn an_associative_chain_is_a_balanced_tree_in_order() {
        assert_eq!(balanced_leaves(0), None);
        assert_eq!(balanced_leaves(1), Some(("0".to_string(), 0)));
        assert_eq!(balanced_leaves(3), Some(("((0 1) 2)".to_string(), 2)));
        assert_eq!(
            balanced_leaves(5),
            Some(("(((0 1) (2 3)) 4)".to_string(), 3))
        );
        let (tree, level) = balanced_leaves(10_000).unwrap();
        assert_eq!(level, 14, "ceil(log2(10,000)) levels");
        let leaves: Vec<usize> = tree
            .split(|c: char| !c.is_ascii_digit())
            .filter(|leaf| !leaf.is_empty())
            .map(|leaf| leaf.parse().unwrap())
            .collect();
        assert_eq!(leaves, (0..10_000).collect::<Vec<_>>(), "in their order");
    }

    #[test]
    fn a_deep_operand_of_a_balanced_chain_counts_where_it_is() {
        // a OR b OR (c) OR d: the parenthesized operand reaches level 1, two
        // joins above it.
        let mut nesting = Nesting::default();
        let chain = nesting.begin_chain();
        let mut operands = Vec::new();
        for leaf in 0..4 {
            if leaf == 2 {
                assert!(nesting.enter());
                nesting.exit();
            }
            operands.push((leaf, nesting.take_operand()));
        }
        let tree = nesting.join_balanced(operands, |left, right| left + right);
        assert_eq!(tree, Some(6));
        assert_eq!(nesting.reached, 3);
        nesting.end_chain(chain);
        // Two operands one level below the limit join at the limit, three
        // beyond it.
        let at_limit = vec![(0, MAX_NESTING_DEPTH - 1); 2];
        assert_eq!(
            Nesting::default().join_balanced(at_limit, |l, _| l),
            Some(0)
        );
        let beyond = vec![(0, MAX_NESTING_DEPTH - 1); 3];
        assert_eq!(Nesting::default().join_balanced(beyond, |l, _| l), None);
    }

    #[test]
    fn the_limit_is_inclusive() {
        let at_limit = reached(|n| {
            let chain = n.begin_chain();
            let ok = (0..MAX_NESTING_DEPTH).all(|_| n.link());
            n.end_chain(chain);
            ok
        });
        assert_eq!(at_limit, Some(MAX_NESTING_DEPTH));
        let beyond = reached(|n| {
            let chain = n.begin_chain();
            let ok = (0..=MAX_NESTING_DEPTH).all(|_| n.link());
            n.end_chain(chain);
            ok
        });
        assert_eq!(beyond, None);
        let mut nesting = Nesting::default();
        assert!((0..MAX_NESTING_DEPTH).all(|_| nesting.enter()));
        assert!(!nesting.enter(), "one level beyond the limit");
    }
}
