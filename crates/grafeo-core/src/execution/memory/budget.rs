//! The operator memory of builds with spilling: one statement's cap
//! ([`QueryMemory`]), the grants its operators take under it and the buffer
//! manager's limit, and the memory consumers of the operators that spill.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use grafeo_common::memory::buffer::{BufferManager, MemoryGrant, MemoryRegion, PressureLevel};
use grafeo_common::utils::error::{Error, LimitBound, LimitRefusal};

/// One statement's operator memory: its cap (the query memory limit, never
/// above the buffer manager's budget) and what its operators hold under it.
///
/// Every operator of the statement shares it (through clones of one
/// [`OperatorMemoryContext`]), so the cap bounds what they hold together, not
/// each on its own. The buffer manager's limit binds besides it: a grant is
/// taken from both.
#[derive(Debug)]
pub struct QueryMemory {
    /// The cap, in bytes.
    limit: usize,
    /// The step the operators' grants grow by under this cap: the buffer
    /// manager's, smaller for a small cap so its rounding does not eat it.
    step: usize,
    /// What the statement's operators hold under the cap.
    used: AtomicUsize,
}

impl QueryMemory {
    /// A statement's memory under `limit` bytes, whose grants grow by at most
    /// `step` bytes at a time (a 64th of the cap, from 4 KiB).
    #[must_use]
    pub fn new(limit: usize, step: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            step: step.min((limit / 64).max(4 << 10)).max(1),
            used: AtomicUsize::new(0),
        })
    }

    /// The cap, in bytes.
    #[must_use]
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// What the statement's operators hold under the cap.
    #[must_use]
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    /// Counts `bytes` more if the cap allows them; the refusal names the
    /// query memory limit with the statement's figures.
    fn take(&self, bytes: usize) -> Result<(), LimitRefusal> {
        let as_u64 = |value: usize| u64::try_from(value).unwrap_or(u64::MAX);
        self.used
            .try_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|total| *total <= self.limit)
            })
            .map(drop)
            .map_err(|used| LimitRefusal::Memory {
                bound: LimitBound::QueryMemoryLimit,
                requested: as_u64(bytes),
                used: as_u64(used),
                limit: as_u64(self.limit),
                detail: None,
            })
    }

    /// Gives back `bytes` taken before.
    fn give_back(&self, bytes: usize) {
        if bytes > 0 {
            self.used.fetch_sub(bytes, Ordering::AcqRel);
        }
    }
}

/// An operator's grant: what it holds under its statement's cap and the
/// buffer manager's limit, grown by steps through
/// [`OperatorMemoryContext::grow`] and given back to both when it is
/// released or dropped (the operator ended, or failed).
#[derive(Debug, Default)]
pub struct OperatorGrant {
    /// The buffer manager's grant.
    grant: Option<MemoryGrant>,
    /// The statement whose cap counts it.
    query: Option<Arc<QueryMemory>>,
    /// The bytes it holds.
    held: usize,
}

impl OperatorGrant {
    /// The bytes it holds.
    #[must_use]
    pub fn size(&self) -> usize {
        self.held
    }
}

impl Drop for OperatorGrant {
    fn drop(&mut self) {
        if let Some(query) = self.query.take() {
            query.give_back(self.held);
        }
    }
}

/// An operator's memory consumer, registered with the buffer manager under a
/// name of its own (see [`OperatorMemoryContext::consumer_name`]) and
/// unregistered when this drops: one operator's end, or its failure, never
/// unregisters another's.
pub struct ConsumerRegistration {
    /// The manager it is registered with.
    buffer_manager: Arc<BufferManager>,
    /// The consumer's name, its own.
    name: String,
}

impl ConsumerRegistration {
    /// The consumer's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for ConsumerRegistration {
    fn drop(&mut self) {
        self.buffer_manager.unregister_consumer(&self.name);
    }
}

/// The memory of an operator that materializes its input and cannot spill (a
/// hash join's build side, a distinct's seen keys, an eager or collecting
/// operator, a set operation, an in-memory sort, a pull aggregate): it counts
/// what the operator holds, a chunk or a table resize at a time, and takes a
/// grant a step at a time. At the statement's cap or the memory limit the
/// operator fails with `GRAFEO-S001` naming the bound that refused. Built
/// outside a budgeted statement it counts nothing.
#[derive(Default)]
pub struct OperatorMemory {
    /// What it counts under its statement; `None`: unbudgeted.
    counted: Option<Counted>,
}

/// What an operator of a budgeted statement holds.
struct Counted {
    /// The statement's context.
    ctx: OperatorMemoryContext,
    /// What it holds.
    grant: OperatorGrant,
    /// What the operator counts it holds.
    bytes: usize,
}

impl OperatorMemory {
    /// The memory of an operator of the statement `ctx` serves.
    #[must_use]
    pub fn new(ctx: OperatorMemoryContext) -> Self {
        Self {
            counted: Some(Counted {
                ctx,
                grant: OperatorGrant::default(),
                bytes: 0,
            }),
        }
    }

    /// The memory of an operator that counts nothing (built outside a
    /// budgeted statement).
    #[must_use]
    pub const fn unbudgeted() -> Self {
        Self { counted: None }
    }

    /// Counts `bytes` more (a chunk the operator keeps).
    ///
    /// # Errors
    ///
    /// The refusal of the query memory limit or the memory limit
    /// (`GRAFEO-S001`); the grant stays as it was.
    #[inline]
    pub fn add(&mut self, bytes: usize) -> Result<(), crate::execution::operators::OperatorError> {
        match &mut self.counted {
            Some(counted) => counted.set(counted.bytes.saturating_add(bytes)),
            None => Ok(()),
        }
    }

    /// Counts `bytes` in all (a figure the operator keeps, such as a hash
    /// table's capacity). Within the grant it costs a compare.
    ///
    /// # Errors
    ///
    /// As [`add`](Self::add).
    #[inline]
    pub fn set(&mut self, bytes: usize) -> Result<(), crate::execution::operators::OperatorError> {
        match &mut self.counted {
            Some(counted) => counted.set(bytes),
            None => Ok(()),
        }
    }

    /// Counts the bytes `bytes` computes more, computed only when it counts
    /// (a figure that reads values, such as [`chunk_bytes`](super::chunk_bytes)).
    ///
    /// # Errors
    ///
    /// As [`add`](Self::add).
    #[inline]
    pub fn add_with(
        &mut self,
        bytes: impl FnOnce() -> usize,
    ) -> Result<(), crate::execution::operators::OperatorError> {
        match &mut self.counted {
            Some(counted) => counted.set(counted.bytes.saturating_add(bytes())),
            None => Ok(()),
        }
    }

    /// Counts the bytes `bytes` computes in all, computed only when it
    /// counts.
    ///
    /// # Errors
    ///
    /// As [`add`](Self::add).
    #[inline]
    pub fn set_with(
        &mut self,
        bytes: impl FnOnce() -> usize,
    ) -> Result<(), crate::execution::operators::OperatorError> {
        match &mut self.counted {
            Some(counted) => counted.set(bytes()),
            None => Ok(()),
        }
    }

    /// Whether it counts: built in a budgeted statement. An operator reads
    /// what its values hold on the heap only when it does.
    #[must_use]
    #[inline]
    pub fn counts(&self) -> bool {
        self.counted.is_some()
    }

    /// What the operator counts it holds.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.counted.as_ref().map_or(0, |counted| counted.bytes)
    }

    /// The operator let go of its rows to start over (a reset): it counts 0
    /// again and keeps its grant, so a rerun (a correlated subquery per outer
    /// row) grows within it with a compare instead of taking it again. The
    /// grant is what one run needed at most; it goes back at
    /// [`release`](Self::release) or when the operator drops.
    pub fn clear(&mut self) {
        if let Some(counted) = &mut self.counted {
            counted.bytes = 0;
        }
    }

    /// Gives back what it holds, its grant too (it spilled, or is done).
    pub fn release(&mut self) {
        if let Some(counted) = &mut self.counted {
            counted.bytes = 0;
            counted.ctx.release(&mut counted.grant);
        }
    }

    /// The bytes its grant holds (0 when it counts nothing).
    #[must_use]
    pub fn granted(&self) -> usize {
        self.counted
            .as_ref()
            .map_or(0, |counted| counted.grant.size())
    }
}

impl Counted {
    /// Counts `bytes` in all.
    fn set(&mut self, bytes: usize) -> Result<(), crate::execution::operators::OperatorError> {
        self.bytes = bytes;
        if bytes <= self.grant.size() {
            return Ok(());
        }
        self.ctx.grow(&mut self.grant, bytes).map_err(Into::into)
    }
}

/// Memory context passed to the operators of one statement: the statement's
/// memory ([`QueryMemory`], which every clone shares), the buffer manager
/// their grants come from, and the per-query spill manager when the open has
/// a spill path.
///
/// Lightweight and `Clone`-able (a few `Arc` references). Operators capture it
/// at construction from the statement's scope (see [`enter`](super::enter) and
/// [`capture`](super::capture)); the spillable push operators get it from the pipeline
/// conversion, which runs in the same scope.
#[derive(Clone)]
pub struct OperatorMemoryContext {
    /// The buffer manager for grants, pressure queries and consumer registration.
    buffer_manager: Arc<BufferManager>,
    /// Per-query spill manager for creating spill files (`None`: this open
    /// has no spill path, so its operators cannot spill).
    spill_manager: Option<Arc<crate::execution::spill::SpillManager>>,
    /// The statement's memory, shared by its operators.
    query: Arc<QueryMemory>,
}

impl OperatorMemoryContext {
    /// Creates the context of one statement, capped by the buffer manager's
    /// budget until [`with_query_limit`](Self::with_query_limit) sets a lower
    /// cap.
    #[must_use]
    pub fn new(
        buffer_manager: Arc<BufferManager>,
        spill_manager: Option<Arc<crate::execution::spill::SpillManager>>,
    ) -> Self {
        let query = QueryMemory::new(buffer_manager.budget(), buffer_manager.grant_step());
        Self {
            buffer_manager,
            spill_manager,
            query,
        }
    }

    /// Sets the query memory limit this statement's operators are bounded by
    /// together, never above the buffer manager's budget. A builder: it gives
    /// the statement a fresh count, so it is called before the context is
    /// handed out.
    #[must_use]
    pub fn with_query_limit(mut self, limit: usize) -> Self {
        self.query = QueryMemory::new(
            limit.min(self.buffer_manager.budget()),
            self.buffer_manager.grant_step(),
        );
        self
    }

    /// The query memory limit: what one statement's operators may hold together.
    #[must_use]
    pub fn query_limit(&self) -> usize {
        self.query.limit()
    }

    /// The statement's memory.
    #[must_use]
    pub fn query(&self) -> &Arc<QueryMemory> {
        &self.query
    }

    /// Grows `grant` to cover `counted` bytes, a step at a time. Past what it
    /// holds: the statement's cap first (one atomic), then the buffer
    /// manager's grant, and when that has no room its eviction cycle and the
    /// grant once more (an operator holds no store lock). Within what it holds
    /// it costs a compare.
    ///
    /// # Errors
    ///
    /// The refusal of the bound that refused, [`LimitBound::QueryMemoryLimit`]
    /// or [`LimitBound::MemoryLimit`] (`GRAFEO-S001`); the grant is as it was.
    pub fn grow(&self, grant: &mut OperatorGrant, counted: usize) -> Result<(), Error> {
        if counted <= grant.held {
            return Ok(());
        }
        let exact = counted - grant.held;
        // Rounded up to the step, so the next chunks cost a compare; when the
        // rounding alone would pass the cap, the exact bytes are asked for.
        let rounded = counted
            .checked_next_multiple_of(self.query.step)
            .map_or(exact, |wanted| wanted - grant.held);
        let more = if self.query.take(rounded).is_ok() {
            rounded
        } else {
            self.query.take(exact)?;
            exact
        };
        let held = grant.grant.get_or_insert_with(|| {
            self.buffer_manager
                .empty_grant(MemoryRegion::ExecutionBuffers)
        });
        // The rounded bytes, else the exact ones, else the exact ones after the
        // eviction cycle made room.
        let granted = if held.try_grow(more) {
            more
        } else if (more > exact && held.try_grow(exact))
            || (self.buffer_manager.make_room(exact) && held.try_grow(exact))
        {
            self.query.give_back(more - exact);
            exact
        } else {
            self.query.give_back(more);
            return Err(self
                .buffer_manager
                .refusal(LimitBound::MemoryLimit, exact)
                .into());
        };
        grant.held += granted;
        grant.query.get_or_insert_with(|| Arc::clone(&self.query));
        Ok(())
    }

    /// Gives back what `grant` holds, to the statement's cap and the buffer
    /// manager (after a spill, or at the end).
    pub fn release(&self, grant: &mut OperatorGrant) {
        grant.grant = None;
        if let Some(query) = &grant.query {
            query.give_back(grant.held);
        }
        grant.held = 0;
    }

    /// Returns the current pressure level.
    #[must_use]
    pub fn pressure_level(&self) -> PressureLevel {
        self.buffer_manager.pressure_level()
    }

    /// Returns whether system pressure warrants spilling (High or Critical).
    #[must_use]
    pub fn should_spill(&self) -> bool {
        self.pressure_level().should_spill()
    }

    /// Registers a memory consumer with the buffer manager, unregistered when
    /// the returned registration drops. Its name must be its own (see
    /// [`consumer_name`](Self::consumer_name)): unregistering goes by name.
    #[must_use = "the consumer is unregistered when the registration drops"]
    pub fn register_consumer(
        &self,
        consumer: Arc<dyn grafeo_common::memory::buffer::MemoryConsumer>,
    ) -> ConsumerRegistration {
        let name = consumer.name().to_string();
        self.buffer_manager.register_consumer(consumer);
        ConsumerRegistration {
            buffer_manager: Arc::clone(&self.buffer_manager),
            name,
        }
    }

    /// A consumer name no other operator of this process has: `base`, `#`
    /// and a number (`SpillableSortPush#7`).
    #[must_use]
    pub fn consumer_name(base: &str) -> String {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let number = NEXT.fetch_add(1, Ordering::Relaxed);
        format!("{base}#{number}")
    }

    /// Returns a reference to the buffer manager.
    #[must_use]
    pub fn buffer_manager(&self) -> &Arc<BufferManager> {
        &self.buffer_manager
    }

    /// Returns the spill manager, when this open has a spill path.
    #[must_use]
    pub fn spill_manager(&self) -> Option<&Arc<crate::execution::spill::SpillManager>> {
        self.spill_manager.as_ref()
    }
}

/// The memory of the statement being planned and converted on this thread, as
/// its scope sets it (see [`enter`](super::enter)).
#[derive(Clone)]
pub enum StatementMemory {
    /// The statement's context: its operators take grants against it.
    Budgeted(OperatorMemoryContext),
    /// None, on purpose: a site that plans operators that never run (an
    /// `EXPLAIN`), or that has no statement (say why where it is used). Its
    /// operators take no grants.
    Unbudgeted,
}

impl StatementMemory {
    /// `context`'s memory, or [`Unbudgeted`](Self::Unbudgeted) without one (a
    /// database without a buffer manager).
    #[must_use]
    pub fn of(context: Option<OperatorMemoryContext>) -> Self {
        context.map_or(Self::Unbudgeted, Self::Budgeted)
    }
}

/// The context of the statement planned on this thread, while a budgeted one
/// is (see [`enter`](super::enter)); `None` otherwise.
#[must_use]
pub fn current() -> Option<OperatorMemoryContext> {
    match super::statement()? {
        StatementMemory::Budgeted(ctx) => Some(ctx),
        StatementMemory::Unbudgeted => None,
    }
}

/// What a materializing operator that cannot spill captures at construction:
/// its memory under the statement planned on this thread (see [`enter`](super::enter)); one
/// that counts nothing when that statement is unbudgeted or the operator is
/// built outside any statement (a unit test).
#[must_use]
pub fn capture() -> OperatorMemory {
    current().map_or_else(OperatorMemory::unbudgeted, OperatorMemory::new)
}
