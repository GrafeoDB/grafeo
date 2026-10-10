//! Native RDF endpoint reachability for outer `+` and `*` property paths.

/// A finite path step inside an outer closure, available to logical IR even
/// when the native RDF store is not compiled.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum PathStep {
    /// Follow this predicate forwards.
    Predicate(String),
    /// Reverse the entire inner step.
    Inverse(Box<Self>),
    /// Follow each step in order.
    Sequence(Vec<Self>),
    /// Follow any of these steps.
    Alternative(Vec<Self>),
}

/// The active graph of a property path.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RdfPathGraph {
    /// The physical default graph.
    Default,
    /// A merged default graph from FROM; an empty list is an empty graph.
    Union(Vec<String>),
    /// One existing named graph.
    Named(String),
    /// Named graphs separately, optionally restricted by FROM NAMED.
    NamedGraphs(Option<Vec<String>>),
}

#[cfg(feature = "triple-store")]
mod physical {
    use super::super::{DEFAULT_PATH_SEARCH_BUDGET, Operator, OperatorError, OperatorResult};
    use super::{PathStep, RdfPathGraph};
    use crate::execution::{DataChunk, ValueVector};
    use crate::graph::rdf::path_budget::{PathBudget, term_payload};
    use crate::graph::rdf::{RdfStore, Term, TriplePattern};
    use grafeo_common::types::{LogicalType, TransactionId};
    use grafeo_common::utils::hash::FxHashSet;
    use std::collections::VecDeque;
    use std::mem::size_of;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// Internal cross-crate control for borrowing a transaction's RDF net.
    /// It never calls a global memory manager while a store lock is held.
    #[doc(hidden)]
    pub struct RdfPathReadControl<'a> {
        budget: &'a mut PathBudget,
    }

    impl RdfPathReadControl<'_> {
        /// Checks the query deadline while scanning even rejected candidates.
        ///
        /// # Errors
        /// Returns the query timeout when its deadline expires.
        pub fn poll(&mut self) -> Result<(), OperatorError> {
            self.budget.check()
        }

        /// The next bounded wait for a transaction lock.
        ///
        /// # Errors
        /// Returns the query timeout when its deadline expires.
        pub fn lock_wait(&mut self) -> Result<Duration, OperatorError> {
            self.budget.lock_wait()
        }
    }

    /// Borrowed view of one graph's last-write net. Its lifetime is scoped to
    /// the overlay callback, so no transaction reference escapes its lock.
    #[doc(hidden)]
    pub trait RdfPathPendingGraph {
        /// The last write of a triple, if it has one in this graph.
        fn state(&self, triple: &crate::graph::rdf::Triple) -> Option<bool>;

        /// Visits matching pending insertions once, polling rejected entries.
        /// The callback only updates query-local state, never storage.
        ///
        /// # Errors
        /// Returns a deadline or callback error.
        fn visit_present(
            &self,
            pattern: &TriplePattern,
            control: &mut RdfPathReadControl<'_>,
            visit: &mut dyn FnMut(
                &crate::graph::rdf::Triple,
                &mut RdfPathReadControl<'_>,
            ) -> Result<(), OperatorError>,
        ) -> Result<(), OperatorError>;
    }

    /// Internal adapter from an engine transaction to the native path reader.
    /// Lock order is transaction changes, then one store index. Callbacks may
    /// only update query-local state and must not recursively access storage.
    #[doc(hidden)]
    pub trait RdfPathReadOverlay: Send + Sync {
        /// Borrows one graph's net under a deadline-aware transaction lock.
        ///
        /// # Errors
        /// Returns a deadline or callback error.
        fn with_graph(
            &self,
            graph: Option<&str>,
            control: &mut RdfPathReadControl<'_>,
            read: &mut dyn FnMut(
                &dyn RdfPathPendingGraph,
                &mut RdfPathReadControl<'_>,
            ) -> Result<(), OperatorError>,
        ) -> Result<(), OperatorError>;

        /// Visits names with pending insertions without allocating a list.
        ///
        /// # Errors
        /// Returns a deadline or callback error.
        fn visit_graph_names(
            &self,
            control: &mut RdfPathReadControl<'_>,
            visit: &mut dyn FnMut(&str, &mut RdfPathReadControl<'_>) -> Result<(), OperatorError>,
        ) -> Result<(), OperatorError>;

        /// The adapter's own allocation, excluding transaction-owned state.
        fn retained_bytes(&self) -> usize;
    }

    /// Endpoint bindings, graph selection and output shape of a native path.
    #[derive(Debug, Clone)]
    pub struct RdfPathConfig {
        /// Bound subject, or a free endpoint.
        pub subject: Option<Term>,
        /// Bound object, or a free endpoint.
        pub object: Option<Term>,
        /// Subject variable, if any.
        pub subject_var: Option<String>,
        /// Object variable, if any.
        pub object_var: Option<String>,
        /// Graph variable, if any.
        pub graph_var: Option<String>,
        /// True for `+`, false for `*`.
        pub min_hops: bool,
        /// The finite step repeated by the closure.
        pub path: PathStep,
        /// Active default or named graphs.
        pub graph: RdfPathGraph,
        /// Emit datatype companions; language companions are always emitted.
        pub companions: bool,
        /// Pending operations visible to this transaction.
        pub transaction_id: Option<TransactionId>,
        /// Maximum rows returned by one next call.
        pub chunk_capacity: usize,
    }

    /// Resumable breadth-first search over exact Terms and finite automaton states.
    /// Only one origin's traversal is retained; endpoint pairs stream as chunks.
    pub struct RdfPathOperator {
        store: Arc<RdfStore>,
        config: RdfPathConfig,
        budget: usize,
        deadline: Option<Instant>,
        read_overlay: Option<Arc<dyn RdfPathReadOverlay>>,
        state: Option<State>,
        exhausted: bool,
    }

    impl RdfPathOperator {
        /// Creates an operator. Charged traversal state is allocated on first next.
        #[must_use]
        pub fn new(store: Arc<RdfStore>, config: RdfPathConfig) -> Self {
            Self {
                store,
                config,
                budget: DEFAULT_PATH_SEARCH_BUDGET,
                deadline: None,
                read_overlay: None,
                state: None,
                exhausted: false,
            }
        }

        /// Narrows the default path cap to the available query budget.
        #[must_use]
        pub fn with_memory_budget(mut self, budget: usize) -> Self {
            self.budget = budget.min(DEFAULT_PATH_SEARCH_BUDGET);
            self
        }

        /// Shares the query deadline, including lookup and lock waits.
        #[must_use]
        pub fn with_deadline(mut self, deadline: Option<Instant>) -> Self {
            self.deadline = deadline;
            self
        }

        /// Attaches the engine's borrowed change-set reader for this query.
        #[doc(hidden)]
        #[must_use]
        pub fn with_read_overlay(mut self, overlay: Option<Arc<dyn RdfPathReadOverlay>>) -> Self {
            self.read_overlay = overlay;
            self
        }

        /// Unique endpoint columns, their companions, then a unique graph column.
        #[must_use]
        pub fn columns(&self) -> Vec<String> {
            let vars = endpoint_vars(&self.config);
            let mut columns: Vec<String> =
                vars.iter().flatten().map(|var| (*var).to_owned()).collect();
            for var in vars.iter().flatten() {
                columns.push(format!("__lang_{var}"));
                if self.config.companions {
                    columns.push(format!("__datatype_{var}"));
                }
            }
            if let Some(graph) = unique_graph_var(&self.config) {
                columns.push(graph.to_owned());
            }
            columns
        }

        fn next_chunk(&mut self) -> OperatorResult {
            if self.state.is_none() {
                self.state = Some(State::new(
                    &self.store,
                    &self.config,
                    self.budget,
                    self.deadline,
                    self.read_overlay.clone(),
                )?);
            }
            let state = self.state.as_mut().expect("RDF path state initialized");
            let vars = endpoint_vars(&self.config);
            let count = vars.iter().flatten().count() * (2 + usize::from(self.config.companions))
                + usize::from(unique_graph_var(&self.config).is_some());
            let capacity = self.config.chunk_capacity.max(1);
            // Existing ValueVector allocation is bounded before it runs. Use the
            // larger generic Value slot as a conservative bound on string slots.
            let base = count
                .checked_mul(capacity)
                .and_then(|n| n.checked_mul(size_of::<grafeo_common::types::Value>()))
                .and_then(|n| {
                    n.checked_add(count * (size_of::<ValueVector>() + size_of::<LogicalType>()))
                })
                .ok_or_else(output_allocation_error)?;
            state.budget.charge(base)?;
            let mut output_charge = base;
            let mut chunk = DataChunk::with_capacity(&vec![LogicalType::String; count], capacity);
            let mut rows = 0;
            while rows < capacity {
                let Some((subject, object, graph)) = state.next_row(&self.config)? else {
                    break;
                };
                write_row(
                    &mut chunk,
                    &self.config,
                    &subject,
                    &object,
                    graph.as_ref(),
                    &mut state.budget,
                    &mut output_charge,
                )?;
                rows += 1;
            }
            chunk.set_count(rows); // A fully bound path can have zero columns.
            state.budget.release(output_charge);
            if state.finished {
                self.state = None;
                self.exhausted = true;
            }
            if rows == 0 { Ok(None) } else { Ok(Some(chunk)) }
        }
    }

    impl Operator for RdfPathOperator {
        fn next(&mut self) -> OperatorResult {
            if self.exhausted {
                return Ok(None);
            }
            let result = self.next_chunk();
            if result.is_err() {
                self.state = None;
                self.exhausted = true;
            }
            result
        }

        fn reset(&mut self) {
            self.state = None;
            self.exhausted = false;
        }
        fn name(&self) -> &'static str {
            "RdfPropertyPath"
        }
        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    fn endpoint_vars(config: &RdfPathConfig) -> [Option<&str>; 2] {
        [
            config.subject_var.as_deref(),
            config
                .object_var
                .as_deref()
                .filter(|object| Some(*object) != config.subject_var.as_deref()),
        ]
    }

    fn unique_graph_var(config: &RdfPathConfig) -> Option<&str> {
        config.graph_var.as_deref().filter(|graph| {
            Some(*graph) != config.subject_var.as_deref()
                && Some(*graph) != config.object_var.as_deref()
        })
    }

    struct Transition {
        from: usize,
        to: usize,
        predicate: Option<Term>,
        inverse: bool,
    }
    struct Nfa {
        transitions: Vec<Transition>,
        start: usize,
        accept: usize,
    }
    struct CompileTask<'a> {
        step: &'a PathStep,
        reverse: bool,
        from: usize,
        to: usize,
    }

    impl Nfa {
        fn compile(
            path: &PathStep,
            reverse: bool,
            budget: &mut PathBudget,
        ) -> Result<Self, OperatorError> {
            let mut tasks = Vec::new();
            budget.reserve_vec(&mut tasks, 1)?;
            tasks.push(CompileTask {
                step: path,
                reverse,
                from: 0,
                to: 1,
            });
            let mut transitions = Vec::new();
            let mut states = 2usize;
            while let Some(task) = tasks.pop() {
                budget.check()?;
                match task.step {
                    PathStep::Predicate(predicate) => {
                        budget.charge(predicate.capacity())?;
                        budget.reserve_vec(&mut transitions, 1)?;
                        budget.charge(predicate.len().saturating_add(2 * size_of::<usize>()))?;
                        transitions.push(Transition {
                            from: task.from,
                            to: task.to,
                            predicate: Some(Term::iri(predicate.as_str())),
                            inverse: task.reverse,
                        });
                    }
                    PathStep::Inverse(inner) => {
                        budget.charge(size_of::<PathStep>())?;
                        budget.reserve_vec(&mut tasks, 1)?;
                        tasks.push(CompileTask {
                            step: inner,
                            reverse: !task.reverse,
                            from: task.from,
                            to: task.to,
                        });
                    }
                    PathStep::Sequence(steps) => {
                        if steps.is_empty() {
                            return Err(OperatorError::InvalidValue(
                                "an RDF path sequence cannot be empty".into(),
                            ));
                        }
                        budget.charge(steps.capacity().saturating_mul(size_of::<PathStep>()))?;
                        let mut from = task.from;
                        for position in 0..steps.len() {
                            budget.check()?;
                            let to = if position + 1 == steps.len() {
                                task.to
                            } else {
                                let state = states;
                                states =
                                    states.checked_add(1).ok_or_else(output_allocation_error)?;
                                state
                            };
                            let index = if task.reverse {
                                steps.len() - position - 1
                            } else {
                                position
                            };
                            budget.reserve_vec(&mut tasks, 1)?;
                            tasks.push(CompileTask {
                                step: &steps[index],
                                reverse: task.reverse,
                                from,
                                to,
                            });
                            from = to;
                        }
                    }
                    PathStep::Alternative(steps) => {
                        if steps.is_empty() {
                            return Err(OperatorError::InvalidValue(
                                "an RDF path alternative cannot be empty".into(),
                            ));
                        }
                        budget.charge(steps.capacity().saturating_mul(size_of::<PathStep>()))?;
                        for step in steps {
                            budget.check()?;
                            budget.reserve_vec(&mut tasks, 1)?;
                            tasks.push(CompileTask {
                                step,
                                reverse: task.reverse,
                                from: task.from,
                                to: task.to,
                            });
                        }
                    }
                }
            }
            let task_bytes = tasks
                .capacity()
                .saturating_mul(size_of::<CompileTask<'_>>());
            drop(tasks);
            budget.release(task_bytes);
            budget.reserve_vec(&mut transitions, 1)?;
            transitions.push(Transition {
                from: 1,
                to: 0,
                predicate: None,
                inverse: false,
            });
            Ok(Self {
                transitions,
                start: 0,
                accept: 1,
            })
        }
    }

    struct GraphSource {
        name: Option<Term>,
        store: Option<Arc<RdfStore>>,
    }

    impl GraphSource {
        fn visit_matches(
            &self,
            pattern: &TriplePattern,
            transaction_id: Option<TransactionId>,
            overlay: Option<&dyn RdfPathReadOverlay>,
            budget: &mut PathBudget,
            visit: &mut dyn FnMut(
                &crate::graph::rdf::Triple,
                &mut PathBudget,
            ) -> Result<(), OperatorError>,
        ) -> Result<(), OperatorError> {
            let Some(overlay) = overlay else {
                return match &self.store {
                    Some(store) => {
                        store.visit_matches_with_pending(pattern, transaction_id, budget, visit)
                    }
                    None => Ok(()),
                };
            };
            let name = self
                .name
                .as_ref()
                .and_then(Term::as_iri)
                .map(|iri| iri.as_str());
            let mut control = RdfPathReadControl { budget };
            overlay.with_graph(name, &mut control, &mut |pending, control| {
                pending.visit_present(pattern, control, &mut |triple, control| {
                    visit(triple, control.budget)
                })?;
                if let Some(store) = &self.store {
                    store.visit_path_base(pattern, control.budget, &mut |triple, budget| {
                        // A pending insertion was emitted above; a pending
                        // deletion suppresses the committed triple entirely.
                        if pending.state(triple).is_none() {
                            visit(triple, budget)?;
                        }
                        Ok(())
                    })?;
                }
                Ok(())
            })
        }
    }

    struct GraphContext {
        name: Option<Term>,
        sources: Vec<GraphSource>,
    }

    fn add_source(
        sources: &mut Vec<GraphSource>,
        name: Option<&str>,
        store: Option<&Arc<RdfStore>>,
        budget: &mut PathBudget,
    ) -> Result<(), OperatorError> {
        for source in sources.iter_mut() {
            budget.check()?;
            if source
                .name
                .as_ref()
                .and_then(Term::as_iri)
                .map(|iri| iri.as_str())
                == name
            {
                if source.store.is_none() {
                    source.store = store.cloned();
                }
                return Ok(());
            }
        }
        budget.reserve_vec(sources, 1)?;
        if let Some(name) = name {
            budget.charge(name.len().saturating_add(2 * size_of::<usize>()))?;
        }
        sources.push(GraphSource {
            name: name.map(Term::iri),
            store: store.cloned(),
        });
        Ok(())
    }

    fn add_named_context(
        contexts: &mut Vec<GraphContext>,
        name: &str,
        store: Option<&Arc<RdfStore>>,
        budget: &mut PathBudget,
    ) -> Result<(), OperatorError> {
        for context in contexts.iter_mut() {
            budget.check()?;
            if context
                .name
                .as_ref()
                .and_then(Term::as_iri)
                .is_some_and(|iri| iri.as_str() == name)
            {
                return add_source(&mut context.sources, Some(name), store, budget);
            }
        }
        let mut sources = Vec::new();
        add_source(&mut sources, Some(name), store, budget)?;
        budget.reserve_vec(contexts, 1)?;
        budget.charge(name.len().saturating_add(2 * size_of::<usize>()))?;
        contexts.push(GraphContext {
            name: Some(Term::iri(name)),
            sources,
        });
        Ok(())
    }

    fn selects_name(
        names: &[String],
        name: &str,
        budget: &mut PathBudget,
    ) -> Result<bool, OperatorError> {
        for selected in names {
            budget.check()?;
            if selected == name {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn graph_contexts(
        store: &Arc<RdfStore>,
        config: &RdfPathConfig,
        overlay: Option<&dyn RdfPathReadOverlay>,
        budget: &mut PathBudget,
    ) -> Result<Vec<GraphContext>, OperatorError> {
        let mut contexts: Vec<GraphContext> = Vec::new();
        match &config.graph {
            RdfPathGraph::Default | RdfPathGraph::Union(_) => {
                let mut sources = Vec::new();
                if let RdfPathGraph::Union(names) = &config.graph {
                    store.visit_path_graphs(Some(names), budget, &mut |name, graph, budget| {
                        add_source(&mut sources, Some(name), Some(graph), budget)
                    })?;
                } else {
                    add_source(&mut sources, None, Some(store), budget)?;
                }
                budget.reserve_vec(&mut contexts, 1)?;
                contexts.push(GraphContext {
                    name: None,
                    sources,
                });
            }
            RdfPathGraph::Named(_) | RdfPathGraph::NamedGraphs(_) => {
                let names = match &config.graph {
                    RdfPathGraph::Named(name) => Some(std::slice::from_ref(name)),
                    RdfPathGraph::NamedGraphs(names) => names.as_deref(),
                    _ => unreachable!(),
                };
                store.visit_path_graphs(names, budget, &mut |name, graph, budget| {
                    add_named_context(&mut contexts, name, Some(graph), budget)
                })?;
            }
        }
        // The named-store lock is released before borrowing changes. The
        // callback below allocates only query-local inventory, never a graph.
        if let Some(overlay) = overlay
            && !matches!(config.graph, RdfPathGraph::Default)
        {
            let mut control = RdfPathReadControl { budget };
            overlay.visit_graph_names(&mut control, &mut |name, control| {
                let budget = &mut *control.budget;
                match &config.graph {
                    RdfPathGraph::Default => {}
                    RdfPathGraph::Union(names) => {
                        if selects_name(names, name, budget)? {
                            add_source(&mut contexts[0].sources, Some(name), None, budget)?;
                        }
                    }
                    RdfPathGraph::Named(selected) => {
                        if selected == name {
                            add_named_context(&mut contexts, name, None, budget)?;
                        }
                    }
                    RdfPathGraph::NamedGraphs(names) => {
                        if match names {
                            Some(names) => selects_name(names, name, budget)?,
                            None => true,
                        } {
                            add_named_context(&mut contexts, name, None, budget)?;
                        }
                    }
                }
                Ok(())
            })?;
        }
        Ok(contexts)
    }

    #[derive(Clone, PartialEq, Eq, Hash)]
    struct StateKey {
        term: Term,
        state: usize,
    }
    struct Search {
        origin: Term,
        queue: VecDeque<StateKey>,
        visited: FxHashSet<StateKey>,
        zero_pending: bool,
    }

    impl Search {
        fn new(
            origin: &Term,
            start: usize,
            zero: bool,
            budget: &mut PathBudget,
        ) -> Result<Self, OperatorError> {
            budget.charge(term_payload(origin))?;
            let mut search = Self {
                origin: origin.clone(),
                queue: VecDeque::new(),
                visited: FxHashSet::default(),
                zero_pending: zero,
            };
            search.enqueue(origin, start, budget)?;
            Ok(search)
        }

        fn enqueue(
            &mut self,
            term: &Term,
            state: usize,
            budget: &mut PathBudget,
        ) -> Result<(), OperatorError> {
            budget.check()?;
            let key = StateKey {
                term: term.clone(),
                state,
            };
            if self.visited.contains(&key) {
                return Ok(());
            }
            budget.charge(term_payload(term).saturating_mul(2))?;
            budget.reserve_set(&mut self.visited, 1)?;
            budget.reserve_queue(&mut self.queue, 1)?;
            self.visited.insert(key.clone());
            self.queue.push_back(key);
            Ok(())
        }

        fn bytes(&self) -> usize {
            term_payload(&self.origin)
                .saturating_add(self.queue.capacity().saturating_mul(size_of::<StateKey>()))
                .saturating_add(
                    self.queue
                        .iter()
                        .map(|key| term_payload(&key.term))
                        .sum::<usize>(),
                )
                .saturating_add(self.visited.allocation_size())
                .saturating_add(
                    self.visited
                        .iter()
                        .map(|key| term_payload(&key.term))
                        .sum::<usize>(),
                )
        }
    }

    struct State {
        budget: PathBudget,
        nfa: Nfa,
        contexts: Vec<GraphContext>,
        read_overlay: Option<Arc<dyn RdfPathReadOverlay>>,
        graph_index: usize,
        origins: Vec<Term>,
        origins_ready: bool,
        origin_index: usize,
        search: Option<Search>,
        reverse: bool,
        finished: bool,
    }

    impl State {
        fn new(
            store: &Arc<RdfStore>,
            config: &RdfPathConfig,
            limit: usize,
            deadline: Option<Instant>,
            read_overlay: Option<Arc<dyn RdfPathReadOverlay>>,
        ) -> Result<Self, OperatorError> {
            let mut budget = PathBudget::new(limit, deadline);
            budget.charge(size_of::<Self>())?;
            if let Some(overlay) = &read_overlay {
                budget.charge(overlay.retained_bytes())?;
            }
            for var in [&config.subject_var, &config.object_var, &config.graph_var]
                .into_iter()
                .flatten()
            {
                budget.charge(var.capacity())?;
            }
            for term in [&config.subject, &config.object].into_iter().flatten() {
                budget.charge(term_payload(term))?;
            }
            match &config.graph {
                RdfPathGraph::Named(name) => budget.charge(name.capacity())?,
                RdfPathGraph::Union(names) | RdfPathGraph::NamedGraphs(Some(names)) => {
                    budget.charge(names.capacity().saturating_mul(size_of::<String>()))?;
                    for name in names {
                        budget.charge(name.capacity())?;
                    }
                }
                _ => {}
            }
            let reverse = config.subject.is_none() && config.object.is_some();
            let nfa = Nfa::compile(&config.path, reverse, &mut budget)?;
            let contexts = graph_contexts(store, config, read_overlay.as_deref(), &mut budget)?;
            Ok(Self {
                budget,
                nfa,
                contexts,
                read_overlay,
                graph_index: 0,
                origins: Vec::new(),
                origins_ready: false,
                origin_index: 0,
                search: None,
                reverse,
                finished: false,
            })
        }

        fn prepare_origins(&mut self, config: &RdfPathConfig) -> Result<(), OperatorError> {
            let bound = if self.reverse {
                config.object.as_ref()
            } else {
                config.subject.as_ref()
            };
            if let Some(term) = bound {
                self.budget.reserve_vec(&mut self.origins, 1)?;
                self.budget.charge(term_payload(term))?;
                self.origins.push(term.clone());
            } else {
                let mut unique = FxHashSet::default();
                for source in &self.contexts[self.graph_index].sources {
                    source.visit_matches(
                        &TriplePattern::any(),
                        config.transaction_id,
                        self.read_overlay.as_deref(),
                        &mut self.budget,
                        &mut |triple, budget| {
                            for term in [triple.subject(), triple.object()] {
                                budget.check()?;
                                if !unique.contains(term) {
                                    budget.charge(term_payload(term))?;
                                    budget.reserve_set(&mut unique, 1)?;
                                    unique.insert(term.clone());
                                }
                            }
                            Ok(())
                        },
                    )?;
                }
                self.budget.reserve_vec(&mut self.origins, unique.len())?;
                let table_bytes = unique.allocation_size();
                for term in unique {
                    self.origins.push(term);
                }
                self.budget.release(table_bytes);
            }
            self.origins_ready = true;
            Ok(())
        }

        fn release_origins(&mut self) {
            let bytes = self
                .origins
                .capacity()
                .saturating_mul(size_of::<Term>())
                .saturating_add(self.origins.iter().map(term_payload).sum::<usize>());
            self.origins = Vec::new();
            self.budget.release(bytes);
            self.origins_ready = false;
            self.origin_index = 0;
        }

        fn release_search(&mut self) {
            if let Some(search) = self.search.take() {
                let bytes = search.bytes();
                drop(search);
                self.budget.release(bytes);
            }
        }

        fn next_row(
            &mut self,
            config: &RdfPathConfig,
        ) -> Result<Option<(Term, Term, Option<Term>)>, OperatorError> {
            loop {
                self.budget.check()?;
                if self.graph_index >= self.contexts.len() {
                    self.finished = true;
                    return Ok(None);
                }
                if self.search.is_none() {
                    if !self.origins_ready {
                        self.prepare_origins(config)?;
                    }
                    if self.origin_index >= self.origins.len() {
                        self.release_origins();
                        self.graph_index += 1;
                        continue;
                    }
                    self.search = Some(Search::new(
                        &self.origins[self.origin_index],
                        self.nfa.start,
                        !config.min_hops,
                        &mut self.budget,
                    )?);
                    self.origin_index += 1;
                }
                let context = &self.contexts[self.graph_index];
                let search = self.search.as_mut().expect("RDF origin initialized");
                if search.zero_pending {
                    search.zero_pending = false;
                    if row_matches(
                        config,
                        &search.origin,
                        &search.origin,
                        context.name.as_ref(),
                    ) {
                        return Ok(Some((
                            search.origin.clone(),
                            search.origin.clone(),
                            context.name.clone(),
                        )));
                    }
                }
                let Some(key) = search.queue.pop_front() else {
                    self.release_search();
                    continue;
                };
                // Complete this state's transitions before yielding an accepted
                // endpoint, so resumption never loses its next repetition.
                for transition in &self.nfa.transitions {
                    self.budget.check()?;
                    if transition.from != key.state {
                        continue;
                    }
                    if let Some(predicate) = &transition.predicate {
                        let pattern = if transition.inverse {
                            TriplePattern {
                                subject: None,
                                predicate: Some(predicate.clone()),
                                object: Some(key.term.clone()),
                            }
                        } else {
                            TriplePattern {
                                subject: Some(key.term.clone()),
                                predicate: Some(predicate.clone()),
                                object: None,
                            }
                        };
                        for source in &context.sources {
                            source.visit_matches(
                                &pattern,
                                config.transaction_id,
                                self.read_overlay.as_deref(),
                                &mut self.budget,
                                &mut |triple, budget| {
                                    let term = if transition.inverse {
                                        triple.subject()
                                    } else {
                                        triple.object()
                                    };
                                    search.enqueue(term, transition.to, budget)
                                },
                            )?;
                        }
                    } else {
                        search.enqueue(&key.term, transition.to, &mut self.budget)?;
                    }
                }
                self.budget.release(term_payload(&key.term));
                if key.state == self.nfa.accept && (config.min_hops || key.term != search.origin) {
                    let (subject, object) = if self.reverse {
                        (&key.term, &search.origin)
                    } else {
                        (&search.origin, &key.term)
                    };
                    if row_matches(config, subject, object, context.name.as_ref()) {
                        return Ok(Some((
                            subject.clone(),
                            object.clone(),
                            context.name.clone(),
                        )));
                    }
                }
            }
        }
    }

    fn row_matches(
        config: &RdfPathConfig,
        subject: &Term,
        object: &Term,
        graph: Option<&Term>,
    ) -> bool {
        if config
            .subject
            .as_ref()
            .is_some_and(|bound| bound != subject)
            || config.object.as_ref().is_some_and(|bound| bound != object)
        {
            return false;
        }
        if config.subject_var.is_some()
            && config.subject_var == config.object_var
            && subject != object
        {
            return false;
        }
        if let Some(var) = &config.graph_var {
            let Some(graph) = graph else {
                return false;
            };
            if config.subject_var.as_ref() == Some(var) && graph != subject {
                return false;
            }
            if config.object_var.as_ref() == Some(var) && graph != object {
                return false;
            }
        }
        true
    }

    fn output_allocation_error() -> OperatorError {
        OperatorError::LimitExceeded(
            "RDF path output allocation exceeds the available memory".into(),
        )
    }

    fn push_text(
        column: &mut ValueVector,
        prefix: &str,
        text: &str,
        budget: &mut PathBudget,
        output_charge: &mut usize,
    ) -> Result<(), OperatorError> {
        let len = prefix
            .len()
            .checked_add(text.len())
            .ok_or_else(output_allocation_error)?;
        let bytes = len
            .checked_mul(2)
            .and_then(|len| len.checked_add(64))
            .ok_or_else(output_allocation_error)?;
        budget.charge(bytes)?;
        *output_charge = output_charge
            .checked_add(bytes)
            .ok_or_else(output_allocation_error)?;
        let mut value = String::new();
        value
            .try_reserve_exact(len)
            .map_err(|_| output_allocation_error())?;
        value.push_str(prefix);
        value.push_str(text);
        column.push_string(value);
        Ok(())
    }

    fn push_term(
        column: &mut ValueVector,
        term: &Term,
        budget: &mut PathBudget,
        output_charge: &mut usize,
    ) -> Result<(), OperatorError> {
        match term {
            Term::Iri(iri) => push_text(column, "", iri.as_str(), budget, output_charge),
            Term::BlankNode(node) => push_text(column, "_:", node.id(), budget, output_charge),
            Term::Literal(literal) => push_text(column, "", literal.value(), budget, output_charge),
        }
    }

    fn write_row(
        chunk: &mut DataChunk,
        config: &RdfPathConfig,
        subject: &Term,
        object: &Term,
        graph: Option<&Term>,
        budget: &mut PathBudget,
        output_charge: &mut usize,
    ) -> Result<(), OperatorError> {
        let vars = endpoint_vars(config);
        let terms = [subject, object];
        let mut column = 0;
        for (var, term) in vars.iter().zip(terms) {
            if var.is_some() {
                push_term(
                    chunk.column_mut(column).expect("RDF endpoint column"),
                    term,
                    budget,
                    output_charge,
                )?;
                column += 1;
            }
        }
        for (var, term) in vars.iter().zip(terms) {
            if var.is_some() {
                let literal = term.as_literal();
                push_text(
                    chunk.column_mut(column).expect("RDF language column"),
                    "",
                    literal.and_then(|lit| lit.language()).unwrap_or(""),
                    budget,
                    output_charge,
                )?;
                column += 1;
                if config.companions {
                    push_text(
                        chunk.column_mut(column).expect("RDF datatype column"),
                        "",
                        literal.map_or("", |lit| lit.datatype()),
                        budget,
                        output_charge,
                    )?;
                    column += 1;
                }
            }
        }
        if unique_graph_var(config).is_some() {
            push_term(
                chunk.column_mut(column).expect("RDF graph column"),
                graph.expect("named graph row"),
                budget,
                output_charge,
            )?;
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::graph::rdf::{Term, Triple};
        use std::sync::Arc;

        fn config() -> RdfPathConfig {
            RdfPathConfig {
                subject: Some(Term::iri("s")),
                object: None,
                subject_var: None,
                object_var: Some("o".into()),
                graph_var: None,
                min_hops: false,
                path: PathStep::Predicate("p".into()),
                graph: RdfPathGraph::Default,
                companions: false,
                transaction_id: None,
                chunk_capacity: 1,
            }
        }

        fn chain() -> Arc<crate::graph::rdf::RdfStore> {
            let store = Arc::new(crate::graph::rdf::RdfStore::new());
            for (s, o) in [("s", "a"), ("a", "b"), ("b", "s")] {
                store.insert(Triple::new(Term::iri(s), Term::iri("p"), Term::iri(o)));
            }
            store
        }

        #[test]
        fn reset_releases_search_and_replays_all_endpoints() {
            let mut path = RdfPathOperator::new(chain(), config());
            assert_eq!(path.next().unwrap().unwrap().len(), 1);
            assert!(path.state.is_some());
            path.reset();
            assert!(path.state.is_none());
            let mut rows = 0;
            while let Some(chunk) = path.next().unwrap() {
                rows += chunk.len();
            }
            assert_eq!(rows, 3);
            assert!(path.state.is_none());
        }

        #[test]
        fn memory_error_releases_state_without_returning_a_partial_chunk() {
            let mut cfg = config();
            cfg.chunk_capacity = 16;
            let mut path = RdfPathOperator::new(chain(), cfg).with_memory_budget(1);
            assert!(matches!(
                path.next(),
                Err(super::OperatorError::LimitExceeded(_))
            ));
            assert!(path.state.is_none());
            assert!(path.next().unwrap().is_none());
        }

        #[test]
        fn deadline_after_bfs_started_discards_chunk_and_releases_state() {
            let store = Arc::new(RdfStore::new());
            let mut subject = Term::iri("s");
            for n in 0..64 {
                let object = Term::iri(format!("n{n}"));
                store.insert(Triple::new(subject, Term::iri("p"), object.clone()));
                subject = object;
            }
            let mut cfg = config();
            cfg.min_hops = true;
            cfg.object_var = None;
            cfg.chunk_capacity = 64;
            let mut path = RdfPathOperator::new(Arc::clone(&store), cfg);
            let idle_references = Arc::strong_count(&store);
            path.state = Some(
                State::new(&path.store, &path.config, path.budget, path.deadline, None).unwrap(),
            );
            let state = path.state.as_mut().unwrap();
            let (_, object, _) = state.next_row(&path.config).unwrap().unwrap();
            assert_eq!(object, Term::iri("n0"));
            let search = state.search.as_ref().unwrap();
            assert!(search.visited.contains(&StateKey {
                term: Term::iri("n0"),
                state: state.nfa.start,
            }));
            assert!(!search.queue.is_empty());
            assert!(state.budget.used() > 0);
            // The first real edge and its accepting state have already run.
            // Allow resumed work, then expire during the remaining 63 hops.
            state.budget.expire_after_polls(50);
            assert!(matches!(path.next(), Err(OperatorError::Timeout)));
            assert!(path.state.is_none());
            assert_eq!(Arc::strong_count(&store), idle_references);
            assert!(path.next().unwrap().is_none());
            path.reset();
            let mut rows = 0;
            while let Some(chunk) = path.next().unwrap() {
                rows += chunk.len();
            }
            assert_eq!(rows, 64);
            assert!(path.state.is_none());
            assert_eq!(Arc::strong_count(&store), idle_references);
        }

        #[test]
        fn frontier_growth_after_initialization_discards_chunk_and_releases_state() {
            let store = Arc::new(RdfStore::new());
            for n in 0..256 {
                store.insert(Triple::new(
                    Term::iri("s"),
                    Term::iri("p"),
                    Term::iri(format!("n{n}")),
                ));
            }
            let mut cfg = config();
            // No output allocation can consume the fixed budget: the
            // reflexive row is buffered, then the fanout must grow the live
            // frontier/visited set beyond the same cap that admitted setup.
            cfg.object_var = None;
            cfg.chunk_capacity = 16;
            let cap = 8 * 1024;
            let mut path = RdfPathOperator::new(Arc::clone(&store), cfg).with_memory_budget(cap);
            let idle_references = Arc::strong_count(&store);
            let mut state =
                State::new(&path.store, &path.config, path.budget, path.deadline, None).unwrap();
            state.prepare_origins(&path.config).unwrap();
            state.search = Some(
                Search::new(&state.origins[0], state.nfa.start, true, &mut state.budget).unwrap(),
            );
            state.origin_index = 1;
            let search = state.search.as_ref().unwrap();
            assert_eq!(search.queue.len(), 1);
            assert_eq!(search.visited.len(), 1);
            assert!(search.zero_pending);
            let initialized_bytes = state.budget.used();
            assert!(initialized_bytes > 0);
            assert!(initialized_bytes <= cap);
            path.state = Some(state);
            assert!(matches!(path.next(), Err(OperatorError::LimitExceeded(_))));
            assert!(path.state.is_none());
            assert_eq!(Arc::strong_count(&store), idle_references);
            assert!(path.next().unwrap().is_none());
            path.reset();
            path = path.with_memory_budget(DEFAULT_PATH_SEARCH_BUDGET);
            let mut rows = 0;
            while let Some(chunk) = path.next().unwrap() {
                rows += chunk.len();
            }
            assert_eq!(rows, 257);
            assert!(path.state.is_none());
            assert_eq!(Arc::strong_count(&store), idle_references);
        }

        #[test]
        fn bound_reflexive_match_has_cardinality_with_no_output_columns() {
            let mut cfg = config();
            cfg.subject = Some(Term::literal("absent"));
            cfg.object = cfg.subject.clone();
            cfg.object_var = None;
            let mut path = RdfPathOperator::new(chain(), cfg);
            let chunk = path.next().unwrap().unwrap();
            assert_eq!(chunk.column_count(), 0);
            assert_eq!(chunk.len(), 1);
            assert!(path.next().unwrap().is_none());
        }
    }
}

#[cfg(feature = "triple-store")]
pub use physical::{
    RdfPathConfig, RdfPathOperator, RdfPathPendingGraph, RdfPathReadControl, RdfPathReadOverlay,
};
