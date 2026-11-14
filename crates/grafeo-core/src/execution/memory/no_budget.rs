//! The operator memory of builds without spilling: operators take no grants,
//! so what they hold of it is a zero-sized type whose methods do nothing.

/// The memory of an operator that materializes its input: in a build without
/// spilling it counts nothing.
#[derive(Debug, Default, Clone, Copy)]
pub struct OperatorMemory;

impl OperatorMemory {
    /// The memory of an operator that counts nothing.
    #[must_use]
    pub const fn unbudgeted() -> Self {
        Self
    }

    /// Counts nothing.
    ///
    /// # Errors
    ///
    /// None in a build without spilling.
    #[inline]
    pub fn add(&mut self, _bytes: usize) -> Result<(), crate::execution::operators::OperatorError> {
        Ok(())
    }

    /// Counts nothing.
    ///
    /// # Errors
    ///
    /// None in a build without spilling.
    #[inline]
    pub fn set(&mut self, _bytes: usize) -> Result<(), crate::execution::operators::OperatorError> {
        Ok(())
    }

    /// Counts nothing; `bytes` is never computed.
    ///
    /// # Errors
    ///
    /// None in a build without spilling.
    #[inline]
    pub fn add_with(
        &mut self,
        _bytes: impl FnOnce() -> usize,
    ) -> Result<(), crate::execution::operators::OperatorError> {
        Ok(())
    }

    /// Counts nothing; `bytes` is never computed.
    ///
    /// # Errors
    ///
    /// None in a build without spilling.
    #[inline]
    pub fn set_with(
        &mut self,
        _bytes: impl FnOnce() -> usize,
    ) -> Result<(), crate::execution::operators::OperatorError> {
        Ok(())
    }

    /// Whether it counts: never in a build without spilling.
    #[must_use]
    #[inline]
    pub fn counts(&self) -> bool {
        false
    }

    /// What it counts: nothing.
    #[must_use]
    #[inline]
    pub fn bytes(&self) -> usize {
        0
    }

    /// Gives back nothing.
    #[inline]
    pub fn clear(&mut self) {}

    /// Gives back nothing.
    #[inline]
    pub fn release(&mut self) {}

    /// What its grant holds: nothing.
    #[must_use]
    #[inline]
    pub fn granted(&self) -> usize {
        0
    }
}

/// The memory of the statement being planned and converted on this thread:
/// without spilling, always unbudgeted.
#[derive(Debug, Clone)]
pub enum StatementMemory {
    /// No grants.
    Unbudgeted,
}

/// What a materializing operator captures at construction: without spilling,
/// memory that counts nothing.
#[must_use]
#[inline]
pub fn capture() -> OperatorMemory {
    OperatorMemory
}
