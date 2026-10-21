//! Admission before grouped aggregate growth.

use super::super::accumulator::DECIMAL_SCRATCH_BYTES;
use super::{
    AggregateExpr, AggregateState, HashableValue, OperatorError, Value, compare_values,
    compare_values_total, value_to_f64,
};
use crate::execution::memory::{
    GroupedMemoryOwner, checked_grouped_bytes, grouped_hash_table_bytes, grouped_memory_limit,
};
use crate::execution::spill::{
    value_hash_scratch_bytes, value_representative_admitted, value_retained_bytes,
};
use std::collections::HashSet;
use std::fmt::{self, Write};

fn add(a: usize, b: usize) -> Result<usize, OperatorError> {
    a.checked_add(b).ok_or_else(grouped_memory_limit)
}

fn arc_slice_bytes<T>(count: usize) -> Result<usize, OperatorError> {
    // Rust 1.99 alloc/src/sync.rs: ArcInner is two atomic words followed
    // by the payload, extended and padded to the combined alignment.
    let payload = std::alloc::Layout::array::<T>(count).map_err(|_| grouped_memory_limit())?;
    std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
        .extend(payload)
        .map(|(layout, _)| layout.pad_to_align().size())
        .map_err(|_| grouped_memory_limit())
}

// Declared before temporary values, so those values drop before their grants.
// If a fallible allocator succeeds for one container then fails for another,
// retain the successfully replaced capacity, without committing an operand.
struct Admission<'a> {
    owner: &'a mut GroupedMemoryOwner,
    retained: usize,
}

impl Drop for Admission<'_> {
    fn drop(&mut self) {
        if self.owner.size() != self.retained {
            let result = self.owner.try_resize(self.retained);
            debug_assert!(result.is_ok(), "cleanup only releases admitted bytes");
        }
    }
}

// Release temporary conversion workspace without disturbing capacity retained
// by the inner admission guard, including during unwinding.
struct Scratch<'a> {
    owner: &'a mut GroupedMemoryOwner,
    bytes: usize,
}

impl Drop for Scratch<'_> {
    fn drop(&mut self) {
        let result = self.owner.try_resize(self.owner.size() - self.bytes);
        debug_assert!(result.is_ok(), "scratch cleanup only releases bytes");
    }
}

#[derive(Default)]
struct Growth {
    old: usize,
    new: usize,
    capacity: usize,
}

impl Growth {
    fn vector<T>(values: &Vec<T>) -> Result<Self, OperatorError> {
        if values.len() < values.capacity() {
            return Ok(Self::default());
        }
        let capacity = values
            .capacity()
            .checked_mul(2)
            .ok_or_else(grouped_memory_limit)?
            .max(4);
        Ok(Self {
            old: checked_grouped_bytes(values.capacity(), std::mem::size_of::<T>())?,
            new: checked_grouped_bytes(capacity, std::mem::size_of::<T>())?,
            capacity,
        })
    }

    fn reserve<T>(
        &self,
        values: &mut Vec<T>,
        admission: &mut Admission<'_>,
    ) -> Result<(), OperatorError> {
        if self.new != 0 {
            values
                .try_reserve_exact(self.capacity - values.len())
                .map_err(|_| grouped_memory_limit())?;
            admission.retained = add(admission.retained, self.new - self.old)?;
        }
        Ok(())
    }
}

#[derive(Default)]
struct Prepared {
    identity: Option<HashableValue>,
    formatted: Option<String>,
    table: Growth,
    vector: Growth,
    release: usize,
    decimal_promotion: Option<AggregateState>,
}

fn seen(state: &AggregateState) -> Option<&HashSet<HashableValue>> {
    match state {
        AggregateState::CountDistinct(_, seen)
        | AggregateState::SumIntDistinct(_, _, seen)
        | AggregateState::SumFloatDistinct(_, _, _, seen)
        | AggregateState::AvgDistinct(_, _, seen)
        | AggregateState::CollectDistinct(_, seen)
        | AggregateState::GroupConcatDistinct(_, _, seen)
        | AggregateState::Distinct { seen, .. } => Some(seen),
        _ => None,
    }
}

fn seen_mut(state: &mut AggregateState) -> Option<&mut HashSet<HashableValue>> {
    match state {
        AggregateState::CountDistinct(_, seen)
        | AggregateState::SumIntDistinct(_, _, seen)
        | AggregateState::SumFloatDistinct(_, _, _, seen)
        | AggregateState::AvgDistinct(_, _, seen)
        | AggregateState::CollectDistinct(_, seen)
        | AggregateState::GroupConcatDistinct(_, _, seen)
        | AggregateState::Distinct { seen, .. } => Some(seen),
        _ => None,
    }
}

fn body(state: &AggregateState) -> &AggregateState {
    match state {
        AggregateState::Distinct { inner, .. } => inner,
        state => state,
    }
}

fn body_mut(state: &mut AggregateState) -> &mut AggregateState {
    match state {
        AggregateState::Distinct { inner, .. } => inner,
        state => state,
    }
}

fn format_operand(value: &Value, output: &mut impl Write) -> fmt::Result {
    match value {
        Value::String(value) => output.write_str(value),
        Value::Int64(value) => write!(output, "{value}"),
        Value::Float64(value) => write!(output, "{value}"),
        Value::Bool(value) => write!(output, "{value}"),
        Value::Null => Ok(()),
        value => write!(output, "{value:?}"),
    }
}

struct CountBytes(usize);
impl Write for CountBytes {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        self.0 = self.0.checked_add(value.len()).ok_or(fmt::Error)?;
        Ok(())
    }
}

fn prepare_body(
    state: &AggregateState,
    value: Option<&Value>,
    prepared: &mut Prepared,
    owner: &mut GroupedMemoryOwner,
) -> Result<(), OperatorError> {
    let Some(value) = value else {
        return Ok(());
    };
    let state = body(state);
    let replacement = match state {
        AggregateState::Min(current) => current
            .as_ref()
            .is_none_or(|old| compare_values_total(value, old).is_lt()),
        AggregateState::Max(current) => current
            .as_ref()
            .is_none_or(|old| compare_values_total(value, old).is_gt()),
        AggregateState::MinOfRdfLiterals(current) => current
            .as_ref()
            .is_none_or(|old| compare_values(value, old) == Some(std::cmp::Ordering::Less)),
        AggregateState::MaxOfRdfLiterals(current) => current
            .as_ref()
            .is_none_or(|old| compare_values(value, old) == Some(std::cmp::Ordering::Greater)),
        AggregateState::First(current) | AggregateState::Sample(current) => current.is_none(),
        AggregateState::Last(_) => true,
        _ => false,
    };
    if replacement {
        let bytes = value_retained_bytes(value, owner)?;
        if bytes != 0 {
            owner.try_grow(bytes)?;
        }
        let old = match state {
            AggregateState::Min(old)
            | AggregateState::Max(old)
            | AggregateState::MinOfRdfLiterals(old)
            | AggregateState::MaxOfRdfLiterals(old)
            | AggregateState::First(old)
            | AggregateState::Last(old)
            | AggregateState::Sample(old) => old.as_ref(),
            _ => None,
        };
        if let Some(old) = old {
            prepared.release = add(prepared.release, value_retained_bytes(old, owner)?)?;
        }
    }
    match state {
        AggregateState::Collect(values) | AggregateState::CollectDistinct(values, _) => {
            let bytes = value_retained_bytes(value, owner)?;
            if bytes != 0 {
                owner.try_grow(bytes)?;
            }
            prepared.vector = Growth::vector(values)?;
        }
        AggregateState::PercentileDisc { values, .. }
        | AggregateState::PercentileCont { values, .. }
            if value_to_f64(value).is_some() =>
        {
            prepared.vector = Growth::vector(values)?;
        }
        AggregateState::GroupConcat(values, _)
        | AggregateState::GroupConcatDistinct(values, _, _) => {
            // Value::Debug delegates Time/Duration/ZonedDatetime to Display;
            // their fractional-digit format! calls allocate a temporary
            // String. Lists/maps can reach these displays recursively, but
            // only one fractional buffer is live at a time. Duration's
            // mixed-sign case has at most ten bytes (nine digits and sign).
            // Rust 1.99 fmt::Arguments::estimated_capacity starts these
            // placeholder-only strings at zero. RawVec<u8> starts at eight
            // and grows to max(2*old, required): before a growth old <= 9,
            // new <= 18, hence at most 27 bytes with realloc overlap.
            // Hold this scratch before BOTH counting and output formatting.
            if matches!(
                value,
                Value::Time(_)
                    | Value::Duration(_)
                    | Value::ZonedDatetime(_)
                    | Value::List(_)
                    | Value::Map(_)
            ) {
                const FRACTION_SCRATCH: usize = 9 + 18;
                owner.try_grow(FRACTION_SCRATCH)?;
                prepared.release = add(prepared.release, FRACTION_SCRATCH)?;
            }
            // Two formatting passes avoid guessing a Debug expansion factor.
            let mut count = CountBytes(0);
            format_operand(value, &mut count).map_err(|_| grouped_memory_limit())?;
            if count.0 != 0 {
                owner.try_grow(count.0)?;
            }
            let mut formatted = String::new();
            formatted
                .try_reserve_exact(count.0)
                .map_err(|_| grouped_memory_limit())?;
            format_operand(value, &mut formatted).map_err(|_| grouped_memory_limit())?;
            prepared.formatted = Some(formatted);
            prepared.vector = Growth::vector(values)?;
        }
        _ => {}
    }
    if prepared.vector.new != 0 {
        owner.try_grow(prepared.vector.new)?;
        prepared.release = add(prepared.release, prepared.vector.old)?;
    }
    Ok(())
}

impl AggregateState {
    /// Admits output ownership and temporary finalization buffers while the
    /// group still owns its state. Scalar finalizers need no reservation.
    pub(crate) fn finalize_accounted(
        &self,
        owner: &mut GroupedMemoryOwner,
        expr: Option<&AggregateExpr>,
    ) -> Result<Value, OperatorError> {
        if let AggregateState::Distinct { inner, .. } = self {
            return inner.finalize_accounted(owner, expr);
        }
        let original = owner.size();
        let mut admission = Admission {
            owner,
            retained: original,
        };
        let (retained, temporary) = match self {
            AggregateState::Collect(values) | AggregateState::CollectDistinct(values, _) => {
                let mut retained = arc_slice_bytes::<Value>(values.len())?;
                for value in values {
                    retained = add(retained, value_retained_bytes(value, admission.owner)?)?;
                }
                (
                    retained,
                    checked_grouped_bytes(values.len(), std::mem::size_of::<Value>())?,
                )
            }
            AggregateState::PercentileDisc { values, .. }
            | AggregateState::PercentileCont { values, .. } => {
                // Rust 1.99 stable sorting of f64 uses at most len slots of
                // heap scratch, in addition to the cloned input vector.
                (
                    0,
                    checked_grouped_bytes(values.len(), 2 * std::mem::size_of::<f64>())?,
                )
            }
            AggregateState::GroupConcat(values, separator)
            | AggregateState::GroupConcatDistinct(values, separator, _) => {
                let mut length =
                    checked_grouped_bytes(values.len().saturating_sub(1), separator.len())?;
                for value in values {
                    length = add(length, value.len())?;
                }
                // arcstr 1.2.0: two words then UTF-8; String and ArcStr coexist.
                (add(length, 2 * std::mem::size_of::<usize>())?, length)
            }
            AggregateState::Min(value)
            | AggregateState::Max(value)
            | AggregateState::MinOfRdfLiterals(value)
            | AggregateState::MaxOfRdfLiterals(value)
            | AggregateState::First(value)
            | AggregateState::Last(value)
            | AggregateState::Sample(value) => (
                match value {
                    Some(value) => value_retained_bytes(value, admission.owner)?,
                    None => 0,
                },
                0,
            ),
            _ => (0, 0),
        };
        let additional = add(retained, temporary)?;
        if additional != 0 {
            admission.owner.try_grow(additional)?;
        }
        let result = match expr {
            Some(expr) => self.finalize_for(expr)?,
            None => self.finalize()?,
        };
        admission.retained = add(original, retained)?;
        Ok(result)
    }

    /// Updates a state only after its retained payload, replacement storage
    /// and identity workspace have been admitted. Arithmetic uses the same
    /// update routines as the non-spilling path.
    pub(crate) fn update_accounted(
        &mut self,
        value: Option<Value>,
        second: Option<Value>,
        owner: &mut GroupedMemoryOwner,
    ) -> Result<(), OperatorError> {
        if matches!(value, Some(Value::Null)) {
            return Ok(());
        }
        // Containers can reach Decimal formatting/comparison recursively.
        // Do not traverse them just to decide whether scratch is necessary.
        let may_format_decimal = |value: &Value| {
            matches!(
                value,
                Value::Decimal(_) | Value::List(_) | Value::Map(_) | Value::Path { .. }
            )
        };
        let old_extreme = match body(self) {
            Self::Min(value)
            | Self::Max(value)
            | Self::MinOfRdfLiterals(value)
            | Self::MaxOfRdfLiterals(value) => value.as_ref(),
            _ => None,
        };
        let needs_scratch = value.as_ref().is_some_and(may_format_decimal)
            || second.as_ref().is_some_and(may_format_decimal)
            || old_extreme.is_some_and(may_format_decimal)
            || matches!(body(self), Self::SumDecimal(..) | Self::AvgDecimal(..));
        if !needs_scratch {
            return self.update_admitted_inner(value, second, owner);
        }
        owner.try_grow(DECIMAL_SCRATCH_BYTES)?;
        let scratch = Scratch {
            owner,
            bytes: DECIMAL_SCRATCH_BYTES,
        };
        self.update_admitted_inner(value, second, scratch.owner)
    }

    fn update_admitted_inner(
        &mut self,
        value: Option<Value>,
        second: Option<Value>,
        owner: &mut GroupedMemoryOwner,
    ) -> Result<(), OperatorError> {
        if matches!(value, Some(Value::Null)) {
            return Ok(());
        }
        // The common numeric states retain no new payload and allocate nothing.
        if matches!(
            self,
            Self::Count(_)
                | Self::SumInt(..)
                | Self::SumFloat(..)
                | Self::Avg(..)
                | Self::SumDecimal(..)
                | Self::AvgDecimal(..)
                | Self::StdDev { .. }
                | Self::StdDevPop { .. }
                | Self::Variance { .. }
                | Self::VariancePop { .. }
                | Self::Bivariate { .. }
        ) {
            if matches!(self, Self::Bivariate { .. }) {
                self.update_bivariate(value, second);
            } else {
                self.update(value);
            }
            return Ok(());
        }
        let binary = matches!(body(self), Self::Bivariate { .. });
        if binary
            && (value.as_ref().is_none_or(Value::is_null)
                || second.as_ref().is_none_or(Value::is_null))
        {
            return Ok(());
        }
        let original = owner.size();
        let mut admission = Admission {
            owner,
            retained: original,
        };
        let mut prepared = Prepared::default();
        if let (Some(seen), Some(operand)) = (seen(self), value.as_ref()) {
            let representative = if binary {
                // The pair's temporary Vec and Arc coexist during conversion.
                let pair_bytes = add(
                    checked_grouped_bytes(2, std::mem::size_of::<Value>())?,
                    arc_slice_bytes::<Value>(2)?,
                )?;
                admission.owner.try_grow(pair_bytes)?;
                let pair = Value::List(
                    vec![
                        operand.clone(),
                        second.as_ref().expect("checked second operand").clone(),
                    ]
                    .into(),
                );
                let representative = value_representative_admitted(&pair, admission.owner)?;
                drop(pair);
                admission
                    .owner
                    .try_resize(admission.owner.size() - pair_bytes)?;
                representative
            } else {
                value_representative_admitted(operand, admission.owner)?
            };
            prepared.identity = Some(HashableValue::from_representative(representative));
            let identity = prepared.identity.as_ref().expect("prepared identity");
            let scratch = value_hash_scratch_bytes(identity.representative(), admission.owner)?;
            if scratch != 0 {
                admission.owner.try_grow(scratch)?;
            }
            prepared.release = scratch;
            if seen.contains(identity) {
                return Ok(());
            }
            if seen.len() == seen.capacity() {
                prepared.table = Growth {
                    old: grouped_hash_table_bytes::<HashableValue>(seen.capacity())?,
                    new: grouped_hash_table_bytes::<HashableValue>(add(seen.len(), 1)?)?,
                    capacity: 0,
                };
                // Rehashing visits existing identities. Measure workspace only
                // at table growth, rather than rescanning the set every row.
                let mut rehash_scratch = 0;
                for existing in seen {
                    rehash_scratch = rehash_scratch.max(value_hash_scratch_bytes(
                        existing.representative(),
                        admission.owner,
                    )?);
                }
                admission
                    .owner
                    .try_grow(add(prepared.table.new, rehash_scratch)?)?;
                prepared.release = add(prepared.release, add(prepared.table.old, rehash_scratch)?)?;
            }
        }
        if !binary {
            prepare_body(self, value.as_ref(), &mut prepared, admission.owner)?;
        }
        if matches!(value, Some(Value::Decimal(_))) {
            let scalar = match self {
                Self::SumIntDistinct(sum, count, _) => Some(Self::SumInt(*sum, *count)),
                Self::AvgDistinct(sum, count, _) => Some(Self::Avg(*sum, *count)),
                _ => None,
            };
            if let Some(mut scalar) = scalar {
                // The operand is a Decimal (Copy), so this clone has no heap.
                scalar.update(value.clone());
                if matches!(scalar, Self::SumDecimal(..) | Self::AvgDecimal(..)) {
                    admission.owner.try_grow(std::mem::size_of::<Self>())?;
                    prepared.decimal_promotion = Some(scalar);
                }
            }
        }
        let final_retained = admission
            .owner
            .size()
            .checked_sub(prepared.release)
            .ok_or_else(grouped_memory_limit)?;
        // Everything which can refuse admission has now run. Capacity changes
        // use fallible allocation; retain each successful change on errors.
        if prepared.table.new != 0 {
            seen_mut(self)
                .expect("distinct table")
                .try_reserve(1)
                .map_err(|_| grouped_memory_limit())?;
            admission.retained = add(admission.retained, prepared.table.new - prepared.table.old)?;
        }
        match body_mut(self) {
            Self::Collect(values) | Self::CollectDistinct(values, _) => {
                prepared.vector.reserve(values, &mut admission)?;
            }
            Self::GroupConcat(values, _) | Self::GroupConcatDistinct(values, _, _) => {
                prepared.vector.reserve(values, &mut admission)?;
            }
            Self::PercentileDisc { values, .. } | Self::PercentileCont { values, .. } => {
                prepared.vector.reserve(values, &mut admission)?;
            }
            _ => {}
        }
        if let Some(identity) = prepared.identity.take() {
            seen_mut(self).expect("distinct identity").insert(identity);
        }
        if let Some(promoted) = prepared.decimal_promotion.take() {
            let inner = Box::new(promoted); // Its allocation was admitted above.
            let seen = std::mem::take(seen_mut(self).expect("distinct promotion"));
            *self = Self::Distinct { seen, inner };
            admission.retained = final_retained;
            return Ok(());
        }
        match self {
            Self::CountDistinct(count, _) => {
                if value.is_some() {
                    *count += 1;
                }
            }
            Self::SumIntDistinct(sum, count, seen) => {
                // Delegate numeric promotion and RDF numeric parsing unchanged.
                let mut scalar = Self::SumInt(*sum, *count);
                scalar.update(value);
                match scalar {
                    Self::SumInt(new_sum, new_count) => {
                        *sum = new_sum;
                        *count = new_count;
                    }
                    Self::SumFloat(new_sum, compensation, new_count) => {
                        *self = Self::SumFloatDistinct(
                            new_sum,
                            compensation,
                            new_count,
                            std::mem::take(seen),
                        );
                    }
                    _ => unreachable!("SUM integer update only promotes to float"),
                }
            }
            Self::SumFloatDistinct(sum, compensation, count, _) => {
                let mut scalar = Self::SumFloat(*sum, *compensation, *count);
                scalar.update(value);
                if let Self::SumFloat(new_sum, new_compensation, new_count) = scalar {
                    *sum = new_sum;
                    *compensation = new_compensation;
                    *count = new_count;
                }
            }
            Self::AvgDistinct(sum, count, _) => {
                let mut scalar = Self::Avg(*sum, *count);
                scalar.update(value);
                if let Self::Avg(new_sum, new_count) = scalar {
                    *sum = new_sum;
                    *count = new_count;
                }
            }
            Self::CollectDistinct(values, _) => {
                if let Some(value) = value {
                    values.push(value);
                }
            }
            Self::GroupConcat(values, _) | Self::GroupConcatDistinct(values, _, _) => {
                if let Some(formatted) = prepared.formatted.take() {
                    values.push(formatted);
                }
            }
            Self::Distinct { inner, .. } => {
                if binary {
                    inner.update_bivariate(value, second);
                } else if let Self::GroupConcat(values, _) = inner.as_mut() {
                    if let Some(formatted) = prepared.formatted.take() {
                        values.push(formatted);
                    }
                } else {
                    inner.update(value);
                }
            }
            _ => self.update(value),
        }
        admission.retained = final_retained;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::AggregateFunction;
    use super::*;
    use grafeo_common::memory::buffer::{BufferManager, BufferManagerConfig};

    fn owner(budget: usize) -> GroupedMemoryOwner {
        let manager = BufferManager::new(BufferManagerConfig {
            budget,
            hard_limit_fraction: 1.0,
            ..BufferManagerConfig::default()
        });
        GroupedMemoryOwner::try_new(manager, 0).unwrap()
    }

    #[test]
    fn accounted_collect_denial_precedes_mutation() {
        let mut state = AggregateState::Collect(Vec::new());
        let mut memory = owner(0);
        assert!(
            state
                .update_accounted(Some(Value::Int64(1)), None, &mut memory)
                .is_err()
        );
        assert!(
            matches!(state, AggregateState::Collect(ref values) if values.is_empty() && values.capacity() == 0)
        );
        assert_eq!(memory.size(), 0);
    }

    #[test]
    fn accounted_finalize_denies_collect_output_before_allocating() {
        let state = AggregateState::Collect(vec![Value::Int64(1), Value::Int64(2)]);
        let mut memory = owner(0);
        assert!(state.finalize_accounted(&mut memory, None).is_err());
        assert_eq!(memory.size(), 0);
        assert!(matches!(state, AggregateState::Collect(ref values) if values.len() == 2));
    }

    #[test]
    fn accounted_distinct_denial_preserves_identity_set() {
        let mut state = AggregateState::CountDistinct(0, HashSet::new());
        let mut memory = owner(0);
        assert!(
            state
                .update_accounted(Some(Value::Int64(1)), None, &mut memory)
                .is_err()
        );
        assert!(
            matches!(state, AggregateState::CountDistinct(0, ref seen) if seen.is_empty() && seen.capacity() == 0)
        );
        assert_eq!(memory.size(), 0);
    }

    #[test]
    fn accounted_nested_distinct_duplicate_releases_temporary_bytes() {
        let mut state = AggregateState::CountDistinct(0, HashSet::new());
        let mut memory = owner(1 << 20);
        state
            .update_accounted(
                Some(Value::List(vec![Value::Int64(3)].into())),
                None,
                &mut memory,
            )
            .unwrap();
        let retained = memory.size();
        assert!(retained > 0);
        state
            .update_accounted(
                Some(Value::List(vec![Value::Float64(3.0)].into())),
                None,
                &mut memory,
            )
            .unwrap();
        assert_eq!(memory.size(), retained);
        assert_eq!(state.finalize().unwrap(), Value::Int64(1));
        state
            .update_accounted(
                Some(Value::List(vec![Value::Int64(4)].into())),
                None,
                &mut memory,
            )
            .unwrap();
        assert_eq!(state.finalize().unwrap(), Value::Int64(2));
        let manager = memory.manager().clone();
        drop(state);
        drop(memory);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn accounted_distinct_statistics_continue_with_same_identity() {
        let mut state =
            AggregateState::new(AggregateFunction::PercentileDisc, true, Some(0.5), None);
        let mut legacy = state.clone();
        let mut memory = owner(1 << 20);
        for value in [
            Value::Int64(3),
            Value::Float64(3.0),
            Value::Int64(7),
            Value::Int64(11),
        ] {
            legacy.update(Some(value.clone()));
            state
                .update_accounted(Some(value), None, &mut memory)
                .unwrap();
        }
        assert_eq!(state.finalize().unwrap(), legacy.finalize().unwrap());
        assert!(memory.size() > 0);
    }

    #[test]
    fn accounted_last_denial_keeps_previous_value() {
        let mut state = AggregateState::Last(Some(Value::Int64(7)));
        let mut memory = owner(0);
        assert!(
            state
                .update_accounted(Some(Value::from("new payload")), None, &mut memory)
                .is_err()
        );
        assert_eq!(state.finalize().unwrap(), Value::Int64(7));
        assert_eq!(memory.size(), 0);
    }

    #[test]
    fn accounted_concat_preserves_display_and_distinct_identity() {
        let mut state = AggregateState::new(AggregateFunction::GroupConcat, true, None, Some("|"));
        let mut memory = owner(1 << 20);
        for value in [
            Value::Int64(3),
            Value::Float64(3.0),
            Value::Bool(true),
            Value::Null,
            Value::List(vec![Value::Int64(2)].into()),
        ] {
            state
                .update_accounted(Some(value), None, &mut memory)
                .unwrap();
        }
        assert_eq!(
            state.finalize().unwrap(),
            Value::from("3|true|List([Int64(2)])")
        );
    }

    #[test]
    fn accounted_bivariate_distinct_counts_pairs_and_skips_null() {
        let mut state = AggregateState::new(AggregateFunction::RegrCount, true, None, None);
        let mut memory = owner(1 << 20);
        state
            .update_accounted(Some(Value::Int64(3)), Some(Value::Int64(7)), &mut memory)
            .unwrap();
        let retained = memory.size();
        state
            .update_accounted(
                Some(Value::Float64(3.0)),
                Some(Value::Float64(7.0)),
                &mut memory,
            )
            .unwrap();
        assert_eq!(memory.size(), retained);
        state
            .update_accounted(Some(Value::Null), Some(Value::Int64(9)), &mut memory)
            .unwrap();
        state
            .update_accounted(Some(Value::Int64(7)), Some(Value::Int64(3)), &mut memory)
            .unwrap();
        assert_eq!(state.finalize().unwrap(), Value::Int64(2));
    }
    #[test]
    fn accounted_nonempty_distinct_refusal_preserves_continuation() {
        use grafeo_common::memory::buffer::MemoryRegion;
        let mut state = AggregateState::CountDistinct(0, HashSet::new());
        let mut memory = owner(1 << 20);
        state
            .update_accounted(Some(Value::Int64(3)), None, &mut memory)
            .unwrap();
        let retained = memory.size();
        let manager = memory.manager().clone();
        let blocker = manager
            .try_allocate(manager.budget() - retained, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let nested = Value::List(vec![Value::Int64(3)].into());
        assert!(
            state
                .update_accounted(Some(nested.clone()), None, &mut memory)
                .is_err()
        );
        assert_eq!(memory.size(), retained);
        assert_eq!(state.finalize().unwrap(), Value::Int64(1));
        drop(blocker);
        state
            .update_accounted(Some(nested), None, &mut memory)
            .unwrap();
        state
            .update_accounted(
                Some(Value::List(vec![Value::Float64(3.0)].into())),
                None,
                &mut memory,
            )
            .unwrap();
        assert_eq!(state.finalize().unwrap(), Value::Int64(2));
        drop(state);
        drop(memory);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn accounted_all_29_states_preserve_results_through_growth_and_output() {
        let states = [
            AggregateState::Count(0),
            AggregateState::CountDistinct(0, HashSet::new()),
            AggregateState::SumInt(0, 0),
            AggregateState::SumIntDistinct(0, 0, HashSet::new()),
            AggregateState::SumFloat(0.0, 0.0, 0),
            AggregateState::SumFloatDistinct(0.0, 0.0, 0, HashSet::new()),
            AggregateState::Avg(0.0, 0),
            AggregateState::AvgDistinct(0.0, 0, HashSet::new()),
            AggregateState::Min(None),
            AggregateState::Max(None),
            AggregateState::MinOfRdfLiterals(None),
            AggregateState::MaxOfRdfLiterals(None),
            AggregateState::First(None),
            AggregateState::Last(None),
            AggregateState::Collect(Vec::new()),
            AggregateState::CollectDistinct(Vec::new(), HashSet::new()),
            AggregateState::StdDev {
                count: 0,
                mean: 0.0,
                m2: 0.0,
            },
            AggregateState::StdDevPop {
                count: 0,
                mean: 0.0,
                m2: 0.0,
            },
            AggregateState::PercentileDisc {
                values: Vec::new(),
                percentile: 0.5,
            },
            AggregateState::PercentileCont {
                values: Vec::new(),
                percentile: 0.5,
            },
            AggregateState::GroupConcat(Vec::new(), "|".into()),
            AggregateState::GroupConcatDistinct(Vec::new(), "|".into(), HashSet::new()),
            AggregateState::Sample(None),
            AggregateState::Variance {
                count: 0,
                mean: 0.0,
                m2: 0.0,
            },
            AggregateState::VariancePop {
                count: 0,
                mean: 0.0,
                m2: 0.0,
            },
            AggregateState::new(AggregateFunction::RegrCount, false, None, None),
            AggregateState::new(AggregateFunction::PercentileCont, true, Some(0.5), None),
            AggregateState::SumDecimal(Some(grafeo_common::types::Decimal::from_i64(0)), 0),
            AggregateState::AvgDecimal(Some(grafeo_common::types::Decimal::from_i64(0)), 0),
        ];
        assert_eq!(states.len(), 29);
        for (variant, mut state) in states.into_iter().enumerate() {
            let binary = matches!(state, AggregateState::Bivariate { .. });
            let mut legacy = state.clone();
            let mut memory = owner(1 << 24);
            let operands = (0..64).map(|index| Some(Value::Int64(index % 40))).chain([
                Some(Value::Float64(3.0)),
                Some(Value::List(vec![Value::Int64(3)].into())),
                Some(Value::List(vec![Value::Float64(3.0)].into())),
                Some(Value::Null),
                None,
            ]);
            for value in operands {
                let second = binary.then_some(Value::Int64(2));
                if binary {
                    legacy.update_bivariate(value.clone(), second.clone());
                } else {
                    legacy.update(value.clone());
                }
                state.update_accounted(value, second, &mut memory).unwrap();
            }
            let manager = memory.manager().clone();
            let mut output = GroupedMemoryOwner::try_new(manager.clone(), 0).unwrap();
            let result = state.finalize_accounted(&mut output, None).unwrap();
            assert_eq!(result, legacy.finalize().unwrap(), "variant {variant}");
            match variant {
                0 => assert_eq!(result, Value::Int64(68)),
                1 => assert_eq!(result, Value::Int64(41)),
                3 => assert_eq!(result, Value::Int64(780)),
                25 => assert_eq!(result, Value::Int64(65)),
                _ => {}
            }
            drop(result);
            drop(output);
            drop(state);
            drop(memory);
            assert_eq!(manager.allocated(), 0, "variant {variant}");
        }
    }
    #[test]
    fn accounted_temporal_concat_denies_format_scratch_before_replacing_state() {
        use grafeo_common::memory::buffer::MemoryRegion;
        use grafeo_common::types::Time;
        let mut state = AggregateState::GroupConcat(Vec::new(), "|".into());
        let mut memory = owner(1 << 20);
        state
            .update_accounted(Some(Value::Int64(3)), None, &mut memory)
            .unwrap();
        let retained = memory.size();
        let manager = memory.manager().clone();
        // The final string fits, but its fractional-digit formatting buffer
        // must coexist with it. The existing vector already has a free slot.
        let expected = "Time(00:00:00.1)";
        let blocker = manager
            .try_allocate(
                manager.budget() - retained - expected.len(),
                MemoryRegion::ExecutionBuffers,
            )
            .unwrap();
        let time = Value::Time(Time::from_hms_nano(0, 0, 0, 100_000_000).unwrap());
        assert!(
            state
                .update_accounted(Some(time.clone()), None, &mut memory)
                .is_err()
        );
        assert_eq!(state.finalize().unwrap(), Value::from("3"));
        assert_eq!(memory.size(), retained);
        drop(blocker);
        state
            .update_accounted(Some(time), None, &mut memory)
            .unwrap();
        assert_eq!(state.finalize().unwrap(), Value::from("3|Time(00:00:00.1)"));
        drop(state);
        drop(memory);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn accounted_decimal_distinct_promotion_retains_identity_and_box() {
        use grafeo_common::types::Decimal;
        for function in [AggregateFunction::Sum, AggregateFunction::Avg] {
            let mut state = AggregateState::new(function, true, None, None);
            let mut legacy = state.clone();
            let mut memory = owner(1 << 20);
            for value in [
                Value::Int64(3),
                Value::Decimal(Decimal::parse("1.50").unwrap()),
            ] {
                legacy.update(Some(value.clone()));
                state
                    .update_accounted(Some(value), None, &mut memory)
                    .unwrap();
            }
            assert!(matches!(&state, AggregateState::Distinct { seen, inner }
                if seen.len() == 2 && matches!(inner.as_ref(),
                    AggregateState::SumDecimal(Some(_), 2) | AggregateState::AvgDecimal(Some(_), 2))));
            let retained = memory.size();
            assert!(retained >= std::mem::size_of::<AggregateState>());
            // A Decimal and float with the same numeric identity remain one
            // value after the original specialized state became a wrapper.
            legacy.update(Some(Value::Float64(1.5)));
            state
                .update_accounted(Some(Value::Float64(1.5)), None, &mut memory)
                .unwrap();
            assert_eq!(memory.size(), retained);
            let next = Value::Decimal(Decimal::parse("2.50").unwrap());
            legacy.update(Some(next.clone()));
            state
                .update_accounted(Some(next), None, &mut memory)
                .unwrap();
            let expected = if matches!(function, AggregateFunction::Sum) {
                Decimal::from_i64(7)
            } else {
                Decimal::from_i64(7)
                    .checked_div(&Decimal::from_i64(3))
                    .unwrap()
            };
            assert_eq!(state.finalize().unwrap(), Value::Decimal(expected));
            assert_eq!(state.finalize().unwrap(), legacy.finalize().unwrap());
            let manager = memory.manager().clone();
            drop(state);
            drop(memory);
            assert_eq!(manager.allocated(), 0);
        }
    }

    #[test]
    fn accounted_decimal_promotion_denial_keeps_nonempty_state() {
        use grafeo_common::memory::buffer::MemoryRegion;
        use grafeo_common::types::Decimal;
        for function in [AggregateFunction::Sum, AggregateFunction::Avg] {
            let mut state = AggregateState::new(function, true, None, None);
            let mut memory = owner(1 << 20);
            state
                .update_accounted(Some(Value::Int64(3)), None, &mut memory)
                .unwrap();
            let before = state.finalize().unwrap();
            let retained = memory.size();
            let manager = memory.manager().clone();
            // The identity table has room and whole Decimal normalization
            // needs no conversion. Leave all outer scratch, but one byte
            // less than the new wrapper's Box allocation requires.
            let headroom = DECIMAL_SCRATCH_BYTES + std::mem::size_of::<AggregateState>() - 1;
            let held = manager
                .try_allocate(
                    manager.budget() - retained - headroom,
                    MemoryRegion::ExecutionBuffers,
                )
                .unwrap();
            assert!(
                state
                    .update_accounted(
                        Some(Value::Decimal(Decimal::from_i64(7))),
                        None,
                        &mut memory,
                    )
                    .is_err()
            );
            assert_eq!(state.finalize().unwrap(), before);
            assert_eq!(seen(&state).unwrap().len(), 1);
            assert_eq!(memory.size(), retained);
            drop(held);
            state
                .update_accounted(
                    Some(Value::Decimal(Decimal::from_i64(7))),
                    None,
                    &mut memory,
                )
                .unwrap();
            assert_eq!(seen(&state).unwrap().len(), 2);
            drop(state);
            drop(memory);
            assert_eq!(manager.allocated(), 0);
        }
    }

    #[test]
    fn accounted_nested_decimal_identity_releases_duplicate_scratch() {
        use grafeo_common::types::Decimal;
        let mut state = AggregateState::CountDistinct(0, HashSet::new());
        let mut memory = owner(1 << 20);
        let nested = |value| Value::List(vec![Value::List(vec![value].into())].into());
        state
            .update_accounted(Some(nested(Value::Float64(1.5))), None, &mut memory)
            .unwrap();
        let retained = memory.size();
        state
            .update_accounted(
                Some(nested(Value::Decimal(Decimal::parse("1.50").unwrap()))),
                None,
                &mut memory,
            )
            .unwrap();
        assert_eq!(state.finalize().unwrap(), Value::Int64(1));
        assert_eq!(memory.size(), retained);
        let manager = memory.manager().clone();
        drop(state);
        drop(memory);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn accounted_decimal_comparison_scratch_denial_preserves_extreme() {
        use grafeo_common::types::Decimal;
        let original = Value::Decimal(Decimal::parse("0.1").unwrap());
        let mut state = AggregateState::Min(Some(original.clone()));
        let mut memory = owner(DECIMAL_SCRATCH_BYTES - 1);
        // Comparing with the smallest positive float calls from_f64, whose
        // fixed notation exceeds 300 bytes. Refusal precedes that conversion.
        assert!(
            state
                .update_accounted(Some(Value::Float64(f64::from_bits(1))), None, &mut memory)
                .is_err()
        );
        assert_eq!(state.finalize().unwrap(), original);
        assert_eq!(memory.size(), 0);
        let mut admitted = owner(DECIMAL_SCRATCH_BYTES);
        state
            .update_accounted(Some(Value::Float64(f64::from_bits(1))), None, &mut admitted)
            .unwrap();
        assert_eq!(state.finalize().unwrap(), Value::Float64(f64::from_bits(1)));
        assert_eq!(admitted.size(), 0);
    }

    #[test]
    fn accounted_decimal_concat_scratch_and_finalization_errors_release_grants() {
        use grafeo_common::types::Decimal;
        let mut concat = AggregateState::GroupConcat(Vec::new(), "|".into());
        let decimal =
            Value::Decimal(Decimal::parse("-0.00000000000000000000000000000000000001").unwrap());
        let mut denied = owner(DECIMAL_SCRATCH_BYTES - 1);
        assert!(
            concat
                .update_accounted(Some(decimal.clone()), None, &mut denied)
                .is_err()
        );
        assert_eq!(concat.finalize().unwrap(), Value::from(""));
        assert_eq!(denied.size(), 0);
        let mut memory = owner(1 << 20);
        let mut legacy = concat.clone();
        legacy.update(Some(decimal.clone()));
        concat
            .update_accounted(Some(decimal), None, &mut memory)
            .unwrap();
        assert_eq!(concat.finalize().unwrap(), legacy.finalize().unwrap());
        let manager = memory.manager().clone();
        let mut output = GroupedMemoryOwner::try_new(manager.clone(), 19).unwrap();
        for state in [
            AggregateState::SumInt(i128::from(i64::MAX) + 1, 2),
            AggregateState::SumDecimal(None, 2),
            AggregateState::AvgDecimal(None, 2),
        ] {
            assert_eq!(
                state
                    .finalize_accounted(&mut output, None)
                    .unwrap_err()
                    .to_string(),
                state.finalize().unwrap_err().to_string()
            );
            assert_eq!(output.size(), 19);
        }
        drop(output);
        drop(concat);
        drop(memory);
        assert_eq!(manager.allocated(), 0);
    }
}
