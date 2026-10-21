//! Spill records of a group's live aggregate state.
//!
//! A record keeps everything an accumulator needs to go on (counts, sums,
//! compensation, DISTINCT identities, collected values), so a group reloaded
//! from a spilled partition aggregates later rows as if it never left memory.
//! The operator that writes a record reads it back within the same query: the
//! bytes are temporary, not a database format.

use std::collections::HashSet;
use std::io::{Read, Write};

use grafeo_common::types::Value;

use super::GroupState;
use crate::execution::memory::{
    GroupedMemoryOwner, grouped_hash_table_bytes, grouped_io_error, grouped_memory_limit,
};
use crate::execution::operators::OperatorError;
use crate::execution::operators::accumulator::{AggregateFunction, AggregateState, HashableValue};
use crate::execution::spill::{deserialize_value, serialize_value};
use crate::execution::spill::{
    deserialize_value_admitted, deserialize_value_representative_admitted, value_hash_scratch_bytes,
};

/// The tag of each accumulator state in a record.
mod tags {
    pub const COUNT: u8 = 0;
    pub const SUM_INT: u8 = 1;
    pub const SUM_FLOAT: u8 = 2;
    pub const AVG: u8 = 3;
    pub const MIN: u8 = 4;
    pub const MAX: u8 = 5;
    pub const FIRST: u8 = 6;
    pub const LAST: u8 = 7;
    pub const COLLECT: u8 = 8;
    pub const COUNT_DISTINCT: u8 = 9;
    pub const SUM_INT_DISTINCT: u8 = 10;
    pub const SUM_FLOAT_DISTINCT: u8 = 11;
    pub const AVG_DISTINCT: u8 = 12;
    pub const COLLECT_DISTINCT: u8 = 13;
    pub const GROUP_CONCAT: u8 = 14;
    pub const GROUP_CONCAT_DISTINCT: u8 = 15;
    pub const STDDEV: u8 = 16;
    pub const STDDEV_POP: u8 = 17;
    pub const VARIANCE: u8 = 18;
    pub const VARIANCE_POP: u8 = 19;
    pub const PERCENTILE_DISC: u8 = 20;
    pub const PERCENTILE_CONT: u8 = 21;
    pub const BIVARIATE: u8 = 22;
    pub const SAMPLE: u8 = 23;
    pub const DISTINCT: u8 = 24;
    pub const MIN_OF_RDF_LITERALS: u8 = 25;
    pub const MAX_OF_RDF_LITERALS: u8 = 26;
    pub const SUM_DECIMAL: u8 = 27;
    pub const AVG_DECIMAL: u8 = 28;
}

fn invalid(message: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}

fn write_len(w: &mut dyn Write, len: usize) -> std::io::Result<()> {
    let len = u64::try_from(len).map_err(|_| invalid("aggregate length exceeds u64"))?;
    w.write_all(&len.to_le_bytes())
}

fn read_u64(r: &mut dyn Read) -> std::io::Result<u64> {
    let mut bytes = [0; 8];
    r.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_i64(r: &mut dyn Read) -> std::io::Result<i64> {
    let mut bytes = [0; 8];
    r.read_exact(&mut bytes)?;
    Ok(i64::from_le_bytes(bytes))
}

fn read_i128(r: &mut dyn Read) -> std::io::Result<i128> {
    let mut bytes = [0; 16];
    r.read_exact(&mut bytes)?;
    Ok(i128::from_le_bytes(bytes))
}

fn read_f64(r: &mut dyn Read) -> std::io::Result<f64> {
    let mut bytes = [0; 8];
    r.read_exact(&mut bytes)?;
    Ok(f64::from_le_bytes(bytes))
}

fn read_tag(r: &mut dyn Read) -> std::io::Result<u8> {
    let mut tag = [0];
    r.read_exact(&mut tag)?;
    Ok(tag[0])
}

fn read_flag(r: &mut dyn Read) -> std::io::Result<bool> {
    match read_tag(r)? {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(invalid("invalid aggregate presence flag")),
    }
}

fn read_len(r: &mut dyn Read) -> std::io::Result<usize> {
    usize::try_from(read_u64(r)?).map_err(|_| invalid("aggregate length exceeds usize"))
}

fn write_string(w: &mut dyn Write, value: &str) -> std::io::Result<()> {
    write_len(w, value.len())?;
    w.write_all(value.as_bytes())
}

fn read_string(r: &mut dyn Read) -> std::io::Result<String> {
    let mut bytes = vec![0; read_len(r)?];
    r.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|_| invalid("aggregate string is not UTF-8"))
}

/// Writes a DISTINCT identity as its representative value, which the
/// identity is read back from.
fn write_identity(w: &mut dyn Write, value: &HashableValue) -> std::io::Result<()> {
    serialize_value(value.representative(), w).map(|_| ())
}

fn read_identity(r: &mut dyn Read) -> std::io::Result<HashableValue> {
    deserialize_value(r).map(HashableValue::from)
}

fn write_seen(w: &mut dyn Write, seen: &HashSet<HashableValue>) -> std::io::Result<()> {
    write_len(w, seen.len())?;
    for value in seen {
        write_identity(w, value)?;
    }
    Ok(())
}

fn read_seen(r: &mut dyn Read) -> std::io::Result<HashSet<HashableValue>> {
    let len = read_len(r)?;
    (0..len).map(|_| read_identity(r)).collect()
}

fn bivariate_tag(kind: AggregateFunction) -> std::io::Result<u8> {
    match kind {
        AggregateFunction::CovarSamp => Ok(0),
        AggregateFunction::CovarPop => Ok(1),
        AggregateFunction::Corr => Ok(2),
        AggregateFunction::RegrSlope => Ok(3),
        AggregateFunction::RegrIntercept => Ok(4),
        AggregateFunction::RegrR2 => Ok(5),
        AggregateFunction::RegrCount => Ok(6),
        AggregateFunction::RegrSxx => Ok(7),
        AggregateFunction::RegrSyy => Ok(8),
        AggregateFunction::RegrSxy => Ok(9),
        AggregateFunction::RegrAvgx => Ok(10),
        AggregateFunction::RegrAvgy => Ok(11),
        _ => Err(invalid("non-bivariate function in aggregate state")),
    }
}

fn read_bivariate_kind(r: &mut dyn Read) -> std::io::Result<AggregateFunction> {
    match read_tag(r)? {
        0 => Ok(AggregateFunction::CovarSamp),
        1 => Ok(AggregateFunction::CovarPop),
        2 => Ok(AggregateFunction::Corr),
        3 => Ok(AggregateFunction::RegrSlope),
        4 => Ok(AggregateFunction::RegrIntercept),
        5 => Ok(AggregateFunction::RegrR2),
        6 => Ok(AggregateFunction::RegrCount),
        7 => Ok(AggregateFunction::RegrSxx),
        8 => Ok(AggregateFunction::RegrSyy),
        9 => Ok(AggregateFunction::RegrSxy),
        10 => Ok(AggregateFunction::RegrAvgx),
        11 => Ok(AggregateFunction::RegrAvgy),
        _ => Err(invalid("unknown bivariate aggregate function")),
    }
}

fn read_values(r: &mut dyn Read) -> std::io::Result<Vec<Value>> {
    let len = read_len(r)?;
    (0..len).map(|_| deserialize_value(r)).collect()
}

/// Writes a group's key values and the live state of its accumulators.
pub(super) fn serialize_group_state(state: &GroupState, w: &mut dyn Write) -> std::io::Result<()> {
    write_len(w, state.key_values.len())?;
    for value in &state.key_values {
        serialize_value(value, w)?;
    }
    write_len(w, state.accumulators.len())?;
    for accumulator in &state.accumulators {
        write_accumulator(w, accumulator)?;
    }
    Ok(())
}

/// Writes one accumulator: its tag, its values, then the identities a
/// DISTINCT state has seen. A `Distinct` wrapper writes the state it wraps
/// as its values.
fn write_accumulator(w: &mut dyn Write, accumulator: &AggregateState) -> std::io::Result<()> {
    let tag = match accumulator {
        AggregateState::Count(_) => tags::COUNT,
        AggregateState::CountDistinct(..) => tags::COUNT_DISTINCT,
        AggregateState::SumInt(..) => tags::SUM_INT,
        AggregateState::SumIntDistinct(..) => tags::SUM_INT_DISTINCT,
        AggregateState::SumFloat(..) => tags::SUM_FLOAT,
        AggregateState::SumFloatDistinct(..) => tags::SUM_FLOAT_DISTINCT,
        AggregateState::Avg(..) => tags::AVG,
        AggregateState::SumDecimal(..) => tags::SUM_DECIMAL,
        AggregateState::AvgDecimal(..) => tags::AVG_DECIMAL,
        AggregateState::AvgDistinct(..) => tags::AVG_DISTINCT,
        AggregateState::Min(_) => tags::MIN,
        AggregateState::Max(_) => tags::MAX,
        AggregateState::MinOfRdfLiterals(_) => tags::MIN_OF_RDF_LITERALS,
        AggregateState::MaxOfRdfLiterals(_) => tags::MAX_OF_RDF_LITERALS,
        AggregateState::First(_) => tags::FIRST,
        AggregateState::Last(_) => tags::LAST,
        AggregateState::Collect(_) => tags::COLLECT,
        AggregateState::CollectDistinct(..) => tags::COLLECT_DISTINCT,
        AggregateState::StdDev { .. } => tags::STDDEV,
        AggregateState::StdDevPop { .. } => tags::STDDEV_POP,
        AggregateState::Variance { .. } => tags::VARIANCE,
        AggregateState::VariancePop { .. } => tags::VARIANCE_POP,
        AggregateState::PercentileDisc { .. } => tags::PERCENTILE_DISC,
        AggregateState::PercentileCont { .. } => tags::PERCENTILE_CONT,
        AggregateState::GroupConcat(..) => tags::GROUP_CONCAT,
        AggregateState::GroupConcatDistinct(..) => tags::GROUP_CONCAT_DISTINCT,
        AggregateState::Sample(_) => tags::SAMPLE,
        AggregateState::Bivariate { .. } => tags::BIVARIATE,
        AggregateState::Distinct { .. } => tags::DISTINCT,
    };
    w.write_all(&[tag])?;
    match accumulator {
        AggregateState::Count(count) | AggregateState::CountDistinct(count, _) => {
            w.write_all(&count.to_le_bytes())?;
        }
        AggregateState::SumInt(sum, count) | AggregateState::SumIntDistinct(sum, count, _) => {
            w.write_all(&sum.to_le_bytes())?;
            w.write_all(&count.to_le_bytes())?;
        }
        AggregateState::SumFloat(sum, compensation, count)
        | AggregateState::SumFloatDistinct(sum, compensation, count, _) => {
            w.write_all(&sum.to_le_bytes())?;
            w.write_all(&compensation.to_le_bytes())?;
            w.write_all(&count.to_le_bytes())?;
        }
        AggregateState::Avg(sum, count) | AggregateState::AvgDistinct(sum, count, _) => {
            w.write_all(&sum.to_le_bytes())?;
            w.write_all(&count.to_le_bytes())?;
        }
        // The sum (absent once past 38 digits), then the count.
        AggregateState::SumDecimal(total, count) | AggregateState::AvgDecimal(total, count) => {
            w.write_all(&[u8::from(total.is_some())])?;
            if let Some(total) = total {
                serialize_value(&Value::Decimal(*total), w)?;
            }
            w.write_all(&count.to_le_bytes())?;
        }
        AggregateState::Min(value)
        | AggregateState::Max(value)
        | AggregateState::MinOfRdfLiterals(value)
        | AggregateState::MaxOfRdfLiterals(value)
        | AggregateState::First(value)
        | AggregateState::Last(value)
        | AggregateState::Sample(value) => {
            w.write_all(&[u8::from(value.is_some())])?;
            if let Some(value) = value {
                serialize_value(value, w)?;
            }
        }
        AggregateState::Collect(values) | AggregateState::CollectDistinct(values, _) => {
            write_len(w, values.len())?;
            for value in values {
                serialize_value(value, w)?;
            }
        }
        AggregateState::GroupConcat(values, separator)
        | AggregateState::GroupConcatDistinct(values, separator, _) => {
            write_len(w, values.len())?;
            for value in values {
                write_string(w, value)?;
            }
            write_string(w, separator)?;
        }
        AggregateState::StdDev { count, mean, m2 }
        | AggregateState::StdDevPop { count, mean, m2 }
        | AggregateState::Variance { count, mean, m2 }
        | AggregateState::VariancePop { count, mean, m2 } => {
            w.write_all(&count.to_le_bytes())?;
            w.write_all(&mean.to_le_bytes())?;
            w.write_all(&m2.to_le_bytes())?;
        }
        AggregateState::PercentileDisc { values, percentile }
        | AggregateState::PercentileCont { values, percentile } => {
            write_len(w, values.len())?;
            for value in values {
                w.write_all(&value.to_le_bytes())?;
            }
            w.write_all(&percentile.to_le_bytes())?;
        }
        AggregateState::Bivariate {
            kind,
            count,
            mean_x,
            mean_y,
            m2_x,
            m2_y,
            c_xy,
        } => {
            w.write_all(&[bivariate_tag(*kind)?])?;
            w.write_all(&count.to_le_bytes())?;
            for value in [mean_x, mean_y, m2_x, m2_y, c_xy] {
                w.write_all(&value.to_le_bytes())?;
            }
        }
        AggregateState::Distinct { inner, .. } => {
            if matches!(**inner, AggregateState::Distinct { .. }) {
                return Err(invalid("a DISTINCT aggregate state wraps another"));
            }
            write_accumulator(w, inner)?;
        }
    }
    match accumulator {
        AggregateState::CountDistinct(_, seen)
        | AggregateState::SumIntDistinct(_, _, seen)
        | AggregateState::SumFloatDistinct(_, _, _, seen)
        | AggregateState::AvgDistinct(_, _, seen)
        | AggregateState::CollectDistinct(_, seen)
        | AggregateState::GroupConcatDistinct(_, _, seen)
        | AggregateState::Distinct { seen, .. } => write_seen(w, seen)?,
        _ => {}
    }
    Ok(())
}

/// Reads what [`serialize_group_state`] wrote: the group with its accumulators
/// live, ready for more rows.
pub(super) fn deserialize_group_state(r: &mut dyn Read) -> std::io::Result<GroupState> {
    let key_values = read_values(r)?;
    let num_accumulators = read_len(r)?;
    let mut accumulators = Vec::with_capacity(num_accumulators);
    for _ in 0..num_accumulators {
        accumulators.push(read_accumulator(r)?);
    }
    Ok(GroupState {
        key_values,
        accumulators,
    })
}

/// Reads what [`write_accumulator`] wrote.
fn read_accumulator(r: &mut dyn Read) -> std::io::Result<AggregateState> {
    let tag = read_tag(r)?;
    read_tagged_accumulator(r, tag)
}

/// Reads the rest of an accumulator whose tag is `tag`.
fn read_tagged_accumulator(r: &mut dyn Read, tag: u8) -> std::io::Result<AggregateState> {
    Ok(match tag {
        tags::COUNT => AggregateState::Count(read_i64(r)?),
        tags::COUNT_DISTINCT => AggregateState::CountDistinct(read_i64(r)?, read_seen(r)?),
        tags::SUM_INT | tags::SUM_INT_DISTINCT => {
            // The 128-bit sum (16 bytes), as the writer writes it.
            let sum = read_i128(r)?;
            let count = read_i64(r)?;
            if tag == tags::SUM_INT {
                AggregateState::SumInt(sum, count)
            } else {
                AggregateState::SumIntDistinct(sum, count, read_seen(r)?)
            }
        }
        tags::SUM_FLOAT | tags::SUM_FLOAT_DISTINCT => {
            let sum = read_f64(r)?;
            let compensation = read_f64(r)?;
            let count = read_i64(r)?;
            if tag == tags::SUM_FLOAT {
                AggregateState::SumFloat(sum, compensation, count)
            } else {
                AggregateState::SumFloatDistinct(sum, compensation, count, read_seen(r)?)
            }
        }
        tags::AVG | tags::AVG_DISTINCT => {
            let sum = read_f64(r)?;
            let count = read_i64(r)?;
            if tag == tags::AVG {
                AggregateState::Avg(sum, count)
            } else {
                AggregateState::AvgDistinct(sum, count, read_seen(r)?)
            }
        }
        tags::SUM_DECIMAL | tags::AVG_DECIMAL => {
            let total = if read_flag(r)? {
                match deserialize_value(r)? {
                    Value::Decimal(total) => Some(total),
                    _ => return Err(invalid("a decimal aggregate state holds another value")),
                }
            } else {
                None
            };
            let count = read_i64(r)?;
            if tag == tags::SUM_DECIMAL {
                AggregateState::SumDecimal(total, count)
            } else {
                AggregateState::AvgDecimal(total, count)
            }
        }
        tags::MIN
        | tags::MAX
        | tags::MIN_OF_RDF_LITERALS
        | tags::MAX_OF_RDF_LITERALS
        | tags::FIRST
        | tags::LAST
        | tags::SAMPLE => {
            let value = if read_flag(r)? {
                Some(deserialize_value(r)?)
            } else {
                None
            };
            match tag {
                tags::MIN => AggregateState::Min(value),
                tags::MAX => AggregateState::Max(value),
                tags::MIN_OF_RDF_LITERALS => AggregateState::MinOfRdfLiterals(value),
                tags::MAX_OF_RDF_LITERALS => AggregateState::MaxOfRdfLiterals(value),
                tags::FIRST => AggregateState::First(value),
                tags::LAST => AggregateState::Last(value),
                _ => AggregateState::Sample(value),
            }
        }
        tags::COLLECT | tags::COLLECT_DISTINCT => {
            let values = read_values(r)?;
            if tag == tags::COLLECT {
                AggregateState::Collect(values)
            } else {
                AggregateState::CollectDistinct(values, read_seen(r)?)
            }
        }
        tags::GROUP_CONCAT | tags::GROUP_CONCAT_DISTINCT => {
            let len = read_len(r)?;
            let values = (0..len)
                .map(|_| read_string(r))
                .collect::<std::io::Result<Vec<_>>>()?;
            let separator = read_string(r)?;
            if tag == tags::GROUP_CONCAT {
                AggregateState::GroupConcat(values, separator)
            } else {
                AggregateState::GroupConcatDistinct(values, separator, read_seen(r)?)
            }
        }
        tags::STDDEV | tags::STDDEV_POP | tags::VARIANCE | tags::VARIANCE_POP => {
            let count = read_i64(r)?;
            let mean = read_f64(r)?;
            let m2 = read_f64(r)?;
            match tag {
                tags::STDDEV => AggregateState::StdDev { count, mean, m2 },
                tags::STDDEV_POP => AggregateState::StdDevPop { count, mean, m2 },
                tags::VARIANCE => AggregateState::Variance { count, mean, m2 },
                _ => AggregateState::VariancePop { count, mean, m2 },
            }
        }
        tags::PERCENTILE_DISC | tags::PERCENTILE_CONT => {
            let len = read_len(r)?;
            let values = (0..len)
                .map(|_| read_f64(r))
                .collect::<std::io::Result<Vec<_>>>()?;
            let percentile = read_f64(r)?;
            if tag == tags::PERCENTILE_DISC {
                AggregateState::PercentileDisc { values, percentile }
            } else {
                AggregateState::PercentileCont { values, percentile }
            }
        }
        tags::BIVARIATE => AggregateState::Bivariate {
            kind: read_bivariate_kind(r)?,
            count: read_i64(r)?,
            mean_x: read_f64(r)?,
            mean_y: read_f64(r)?,
            m2_x: read_f64(r)?,
            m2_y: read_f64(r)?,
            c_xy: read_f64(r)?,
        },
        tags::DISTINCT => {
            let inner_tag = read_tag(r)?;
            if inner_tag == tags::DISTINCT {
                return Err(invalid("a DISTINCT aggregate state wraps another"));
            }
            AggregateState::Distinct {
                inner: Box::new(read_tagged_accumulator(r, inner_tag)?),
                seen: read_seen(r)?,
            }
        }
        _ => return Err(invalid("unknown aggregate state tag")),
    })
}

struct GroupAdmission<'a> {
    owner: &'a mut GroupedMemoryOwner,
    original: usize,
    keep: bool,
}

impl Drop for GroupAdmission<'_> {
    fn drop(&mut self) {
        if !self.keep {
            let _ = self.owner.try_resize(self.original);
        }
    }
}

fn admitted_buffer<T>(
    length: usize,
    owner: &mut GroupedMemoryOwner,
) -> Result<Vec<T>, OperatorError> {
    let layout = std::alloc::Layout::array::<T>(length)
        .map_err(|_| grouped_io_error(invalid("invalid aggregate capacity")))?;
    owner.try_grow(layout.size())?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| grouped_memory_limit())?;
    Ok(values)
}

fn read_values_admitted(
    r: &mut dyn Read,
    owner: &mut GroupedMemoryOwner,
) -> Result<Vec<Value>, OperatorError> {
    let length = read_len(r).map_err(grouped_io_error)?;
    let mut values = admitted_buffer(length, owner)?;
    for _ in 0..length {
        values.push(deserialize_value_admitted(r, owner)?);
    }
    Ok(values)
}

fn read_string_admitted(
    r: &mut dyn Read,
    owner: &mut GroupedMemoryOwner,
) -> Result<String, OperatorError> {
    let length = read_len(r).map_err(grouped_io_error)?;
    let mut bytes = admitted_buffer::<u8>(length, owner)?;
    bytes.resize(length, 0);
    r.read_exact(&mut bytes).map_err(grouped_io_error)?;
    String::from_utf8(bytes).map_err(|_| grouped_io_error(invalid("aggregate string is not UTF-8")))
}

fn read_seen_admitted(
    r: &mut dyn Read,
    owner: &mut GroupedMemoryOwner,
) -> Result<HashSet<HashableValue>, OperatorError> {
    let length = read_len(r).map_err(grouped_io_error)?;
    owner.try_grow(grouped_hash_table_bytes::<HashableValue>(length)?)?;
    let mut seen = HashSet::new();
    seen.try_reserve(length)
        .map_err(|_| grouped_memory_limit())?;
    for _ in 0..length {
        let value = deserialize_value_representative_admitted(r, owner)?;
        let scratch = value_hash_scratch_bytes(&value, owner)?;
        let retained_size = owner.size();
        owner.try_grow(scratch)?;
        seen.insert(HashableValue::from_representative(value));
        // Hash/sort scratch has dropped before returning this temporary grant.
        owner.try_resize(retained_size)?;
    }
    Ok(seen)
}

/// Reads the unchanged live-state bytes with the caller's reservation.
/// Decode refusal leaves its original grant intact, so the partition keeps
/// cleanup authority over the spill file and can retry the same record.
pub(super) fn deserialize_group_state_admitted(
    r: &mut dyn Read,
    owner: &mut GroupedMemoryOwner,
) -> Result<GroupState, OperatorError> {
    let original = owner.size();
    let mut admission = GroupAdmission {
        owner,
        original,
        keep: false,
    };
    let result = (|| {
        let key_values = read_values_admitted(r, admission.owner)?;
        let length = read_len(r).map_err(grouped_io_error)?;
        let mut accumulators = admitted_buffer(length, admission.owner)?;
        for _ in 0..length {
            let tag = read_tag(r).map_err(grouped_io_error)?;
            accumulators.push(read_tagged_accumulator_admitted(r, tag, admission.owner)?);
        }
        Ok(GroupState {
            key_values,
            accumulators,
        })
    })();
    admission.keep = result.is_ok();
    result
}

fn read_tagged_accumulator_admitted(
    r: &mut dyn Read,
    tag: u8,
    owner: &mut GroupedMemoryOwner,
) -> Result<AggregateState, OperatorError> {
    Ok(match tag {
        tags::COUNT => AggregateState::Count(read_i64(r).map_err(grouped_io_error)?),
        tags::COUNT_DISTINCT => AggregateState::CountDistinct(
            read_i64(r).map_err(grouped_io_error)?,
            read_seen_admitted(r, owner)?,
        ),
        tags::SUM_INT | tags::SUM_INT_DISTINCT => {
            let sum = read_i128(r).map_err(grouped_io_error)?;
            let count = read_i64(r).map_err(grouped_io_error)?;
            if tag == tags::SUM_INT {
                AggregateState::SumInt(sum, count)
            } else {
                AggregateState::SumIntDistinct(sum, count, read_seen_admitted(r, owner)?)
            }
        }
        tags::SUM_FLOAT | tags::SUM_FLOAT_DISTINCT => {
            let sum = read_f64(r).map_err(grouped_io_error)?;
            let compensation = read_f64(r).map_err(grouped_io_error)?;
            let count = read_i64(r).map_err(grouped_io_error)?;
            if tag == tags::SUM_FLOAT {
                AggregateState::SumFloat(sum, compensation, count)
            } else {
                AggregateState::SumFloatDistinct(
                    sum,
                    compensation,
                    count,
                    read_seen_admitted(r, owner)?,
                )
            }
        }
        tags::AVG | tags::AVG_DISTINCT => {
            let sum = read_f64(r).map_err(grouped_io_error)?;
            let count = read_i64(r).map_err(grouped_io_error)?;
            if tag == tags::AVG {
                AggregateState::Avg(sum, count)
            } else {
                AggregateState::AvgDistinct(sum, count, read_seen_admitted(r, owner)?)
            }
        }
        tags::SUM_DECIMAL | tags::AVG_DECIMAL => {
            let total = if read_flag(r).map_err(grouped_io_error)? {
                match deserialize_value_admitted(r, owner)? {
                    Value::Decimal(total) => Some(total),
                    _ => {
                        return Err(grouped_io_error(invalid(
                            "a decimal aggregate state holds another value",
                        )));
                    }
                }
            } else {
                None
            };
            let count = read_i64(r).map_err(grouped_io_error)?;
            if tag == tags::SUM_DECIMAL {
                AggregateState::SumDecimal(total, count)
            } else {
                AggregateState::AvgDecimal(total, count)
            }
        }
        tags::MIN
        | tags::MAX
        | tags::MIN_OF_RDF_LITERALS
        | tags::MAX_OF_RDF_LITERALS
        | tags::FIRST
        | tags::LAST
        | tags::SAMPLE => {
            let value = if read_flag(r).map_err(grouped_io_error)? {
                Some(deserialize_value_admitted(r, owner)?)
            } else {
                None
            };
            match tag {
                tags::MIN => AggregateState::Min(value),
                tags::MAX => AggregateState::Max(value),
                tags::MIN_OF_RDF_LITERALS => AggregateState::MinOfRdfLiterals(value),
                tags::MAX_OF_RDF_LITERALS => AggregateState::MaxOfRdfLiterals(value),
                tags::FIRST => AggregateState::First(value),
                tags::LAST => AggregateState::Last(value),
                _ => AggregateState::Sample(value),
            }
        }
        tags::COLLECT | tags::COLLECT_DISTINCT => {
            let values = read_values_admitted(r, owner)?;
            if tag == tags::COLLECT {
                AggregateState::Collect(values)
            } else {
                AggregateState::CollectDistinct(values, read_seen_admitted(r, owner)?)
            }
        }
        tags::GROUP_CONCAT | tags::GROUP_CONCAT_DISTINCT => {
            let len = read_len(r).map_err(grouped_io_error)?;
            let mut values = admitted_buffer::<String>(len, owner)?;
            for _ in 0..len {
                values.push(read_string_admitted(r, owner)?);
            }
            let separator = read_string_admitted(r, owner)?;
            if tag == tags::GROUP_CONCAT {
                AggregateState::GroupConcat(values, separator)
            } else {
                AggregateState::GroupConcatDistinct(
                    values,
                    separator,
                    read_seen_admitted(r, owner)?,
                )
            }
        }
        tags::STDDEV | tags::STDDEV_POP | tags::VARIANCE | tags::VARIANCE_POP => {
            let count = read_i64(r).map_err(grouped_io_error)?;
            let mean = read_f64(r).map_err(grouped_io_error)?;
            let m2 = read_f64(r).map_err(grouped_io_error)?;
            match tag {
                tags::STDDEV => AggregateState::StdDev { count, mean, m2 },
                tags::STDDEV_POP => AggregateState::StdDevPop { count, mean, m2 },
                tags::VARIANCE => AggregateState::Variance { count, mean, m2 },
                _ => AggregateState::VariancePop { count, mean, m2 },
            }
        }
        tags::PERCENTILE_DISC | tags::PERCENTILE_CONT => {
            let len = read_len(r).map_err(grouped_io_error)?;
            let mut values = admitted_buffer::<f64>(len, owner)?;
            for _ in 0..len {
                values.push(read_f64(r).map_err(grouped_io_error)?);
            }
            let percentile = read_f64(r).map_err(grouped_io_error)?;
            if tag == tags::PERCENTILE_DISC {
                AggregateState::PercentileDisc { values, percentile }
            } else {
                AggregateState::PercentileCont { values, percentile }
            }
        }
        tags::BIVARIATE => AggregateState::Bivariate {
            kind: read_bivariate_kind(r).map_err(grouped_io_error)?,
            count: read_i64(r).map_err(grouped_io_error)?,
            mean_x: read_f64(r).map_err(grouped_io_error)?,
            mean_y: read_f64(r).map_err(grouped_io_error)?,
            m2_x: read_f64(r).map_err(grouped_io_error)?,
            m2_y: read_f64(r).map_err(grouped_io_error)?,
            c_xy: read_f64(r).map_err(grouped_io_error)?,
        },
        tags::DISTINCT => {
            let inner_tag = read_tag(r).map_err(grouped_io_error)?;
            if inner_tag == tags::DISTINCT {
                return Err(grouped_io_error(invalid(
                    "a DISTINCT aggregate state wraps another",
                )));
            }
            owner.try_grow(std::mem::size_of::<AggregateState>())?;
            AggregateState::Distinct {
                inner: Box::new(read_tagged_accumulator_admitted(r, inner_tag, owner)?),
                seen: read_seen_admitted(r, owner)?,
            }
        }
        _ => return Err(grouped_io_error(invalid("unknown aggregate state tag"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A decimal sum or average spills and comes back with its exact sum
    /// (scale kept) or its overflow, and its count.
    #[test]
    fn decimal_states_round_trip() {
        use grafeo_common::types::Decimal;
        let total = Decimal::parse("-12.50").unwrap();
        let group = GroupState {
            key_values: Vec::new(),
            accumulators: vec![
                AggregateState::SumDecimal(Some(total), 3),
                AggregateState::AvgDecimal(Some(total), 4),
                AggregateState::SumDecimal(None, 7),
            ],
        };
        let mut bytes = Vec::new();
        serialize_group_state(&group, &mut bytes).unwrap();
        let back = deserialize_group_state(&mut bytes.as_slice()).unwrap();
        let described: Vec<String> = back
            .accumulators
            .iter()
            .map(|state| match state {
                AggregateState::SumDecimal(total, count) => format!("sum {total:?} {count}"),
                AggregateState::AvgDecimal(total, count) => format!("avg {total:?} {count}"),
                other => format!("{other:?}"),
            })
            .collect();
        assert_eq!(
            described,
            [
                "sum Some(Decimal(-12.50)) 3",
                "avg Some(Decimal(-12.50)) 4",
                "sum None 7"
            ]
        );
    }

    fn admitted_owner(budget: usize) -> GroupedMemoryOwner {
        use grafeo_common::memory::buffer::{BufferManager, BufferManagerConfig};
        let manager = BufferManager::new(BufferManagerConfig {
            hard_limit_fraction: 1.0,
            ..BufferManagerConfig::with_budget(budget)
        });
        GroupedMemoryOwner::try_new(manager, 0).unwrap()
    }

    #[test]
    fn admitted_numeric_states_preserve_current_bytes_and_resume() {
        use grafeo_common::types::Decimal;

        fn assert_states_equal(actual: &GroupState, expected: &GroupState) {
            assert_eq!(actual.key_values, expected.key_values);
            assert_eq!(actual.accumulators.len(), expected.accumulators.len());
            for (actual, expected) in actual.accumulators.iter().zip(&expected.accumulators) {
                match (actual, expected) {
                    (
                        AggregateState::SumInt(actual_sum, actual_count),
                        AggregateState::SumInt(expected_sum, expected_count),
                    ) => assert_eq!((actual_sum, actual_count), (expected_sum, expected_count)),
                    (
                        AggregateState::SumIntDistinct(actual_sum, actual_count, actual_seen),
                        AggregateState::SumIntDistinct(expected_sum, expected_count, expected_seen),
                    ) => {
                        assert_eq!((actual_sum, actual_count), (expected_sum, expected_count));
                        assert_eq!(actual_seen, expected_seen);
                    }
                    (
                        AggregateState::SumDecimal(actual_total, actual_count),
                        AggregateState::SumDecimal(expected_total, expected_count),
                    )
                    | (
                        AggregateState::AvgDecimal(actual_total, actual_count),
                        AggregateState::AvgDecimal(expected_total, expected_count),
                    ) => {
                        assert_eq!(actual_count, expected_count);
                        assert_eq!(
                            actual_total.map(|total| (total.mantissa(), total.scale())),
                            expected_total.map(|total| (total.mantissa(), total.scale()))
                        );
                    }
                    _ => panic!("unexpected numeric states: {actual:?}, {expected:?}"),
                }
                match (actual.finalize(), expected.finalize()) {
                    (Ok(actual), Ok(expected)) => assert_eq!(actual, expected),
                    (Err(actual), Err(expected)) => {
                        assert_eq!(actual.to_string(), expected.to_string());
                    }
                    other => panic!("different finalization outcomes: {other:?}"),
                }
            }
        }

        let past = i128::from(i64::MAX) + 2;
        let total = Decimal::parse("-12.50").unwrap();
        let seen = HashSet::from([HashableValue::from(Value::Int64(i64::MAX))]);
        let group = GroupState {
            key_values: vec![Value::Decimal(total)],
            accumulators: vec![
                AggregateState::SumInt(past, 3),
                AggregateState::SumIntDistinct(past, 3, seen),
                AggregateState::SumDecimal(Some(total), 3),
                AggregateState::AvgDecimal(Some(total), 4),
                AggregateState::SumDecimal(None, 7),
                AggregateState::AvgDecimal(None, 7),
            ],
        };
        let mut bytes = Vec::new();
        serialize_group_state(&group, &mut bytes).unwrap();
        let mut legacy = deserialize_group_state(&mut bytes.as_slice()).unwrap();
        let mut owner = admitted_owner(1024 * 1024);
        let mut admitted =
            deserialize_group_state_admitted(&mut bytes.as_slice(), &mut owner).unwrap();
        let mut encoded = Vec::new();
        serialize_group_state(&admitted, &mut encoded).unwrap();
        assert_eq!(
            encoded, bytes,
            "admission must preserve current spill bytes"
        );
        assert!(matches!(admitted.accumulators[0], AggregateState::SumInt(sum, 3) if sum == past));
        for (left, right) in admitted
            .accumulators
            .iter_mut()
            .zip(&mut legacy.accumulators)
        {
            left.update(Some(Value::Int64(-2)));
            right.update(Some(Value::Int64(-2)));
        }
        assert_eq!(
            admitted.accumulators[0].finalize().unwrap(),
            Value::Int64(i64::MAX)
        );
        assert_eq!(
            admitted.accumulators[1].finalize().unwrap(),
            Value::Int64(i64::MAX)
        );
        assert!(matches!(
            admitted.accumulators[4],
            AggregateState::SumDecimal(None, 8)
        ));
        assert!(matches!(
            admitted.accumulators[5],
            AggregateState::AvgDecimal(None, 8)
        ));
        let mut continued = Vec::new();
        serialize_group_state(&admitted, &mut continued).unwrap();
        assert_states_equal(&admitted, &legacy);
        let resumed = deserialize_group_state(&mut continued.as_slice()).unwrap();
        assert_states_equal(&resumed, &legacy);
        let mut resumed_owner = admitted_owner(1024 * 1024);
        let admitted_resumed =
            deserialize_group_state_admitted(&mut continued.as_slice(), &mut resumed_owner)
                .unwrap();
        assert_states_equal(&admitted_resumed, &legacy);
    }

    #[test]
    fn admitted_malformed_group_restores_original_reservation() {
        let mut bytes = Vec::new();
        bytes.extend(1u64.to_le_bytes());
        serialize_value(&Value::List(vec![Value::Int64(1)].into()), &mut bytes).unwrap();
        bytes.extend(1u64.to_le_bytes());
        bytes.push(255);
        let mut owner = admitted_owner(1024 * 1024);
        owner.try_grow(32).unwrap();
        assert!(deserialize_group_state_admitted(&mut bytes.as_slice(), &mut owner).is_err());
        assert_eq!(owner.size(), 32);
        assert_eq!(owner.manager().allocated(), 32);
    }

    #[test]
    fn admitted_distinct_reload_preserves_nested_identity() {
        let identity = HashableValue::from(Value::List(vec![Value::Float64(3.0)].into()));
        let mut seen = HashSet::new();
        seen.insert(identity.clone());
        let group = GroupState {
            key_values: vec![Value::Int64(7)],
            accumulators: vec![AggregateState::CountDistinct(1, seen)],
        };
        let mut bytes = Vec::new();
        serialize_group_state(&group, &mut bytes).unwrap();
        let mut owner = admitted_owner(1024 * 1024);
        let mut decoded =
            deserialize_group_state_admitted(&mut bytes.as_slice(), &mut owner).unwrap();
        match decoded.accumulators.pop().unwrap() {
            AggregateState::CountDistinct(count, seen) => {
                assert_eq!(count, 1);
                assert!(seen.contains(&identity));
            }
            _ => panic!("wrong live accumulator"),
        }
        assert!(owner.size() > 0);
    }

    #[test]
    fn nested_declared_capacity_is_rejected_without_panicking() {
        // One key: a list containing a vector with an impossible capacity.
        // The fixture is small; refusal must precede any payload allocation.
        let mut bytes = Vec::new();
        bytes.extend(1u64.to_le_bytes());
        bytes.push(7); // Value::List
        bytes.extend(1u64.to_le_bytes());
        bytes.push(9); // Value::Vector
        bytes.extend(u64::MAX.to_le_bytes());

        let decoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            deserialize_group_state(&mut bytes.as_slice())
        }));
        let error = decoded
            .expect("an impossible nested capacity must not panic")
            .err()
            .expect("an impossible nested capacity must be refused");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    /// A DISTINCT state wraps the state of the aggregate itself, never
    /// another DISTINCT state: a record that claims one is refused, without
    /// following it (a run of DISTINCT tags would otherwise recurse), and
    /// none is written.
    #[test]
    fn a_distinct_state_that_wraps_another_is_refused() {
        let mut bytes = Vec::new();
        write_len(&mut bytes, 0).unwrap(); // no key values
        write_len(&mut bytes, 1).unwrap(); // one accumulator
        bytes.extend([tags::DISTINCT; 64]);
        let error = deserialize_group_state(&mut bytes.as_slice())
            .err()
            .expect("a DISTINCT state in a DISTINCT state is refused");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);

        let distinct = |inner| AggregateState::Distinct {
            seen: HashSet::new(),
            inner: Box::new(inner),
        };
        let group = GroupState {
            key_values: Vec::new(),
            accumulators: vec![distinct(distinct(AggregateState::new(
                AggregateFunction::StdDev,
                false,
                None,
                None,
            )))],
        };
        assert!(serialize_group_state(&group, &mut Vec::new()).is_err());
    }

    #[test]
    fn admitted_live_variants_resume_with_owned_state() {
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

        let functions = [
            AggregateFunction::Count,
            AggregateFunction::CountNonNull,
            AggregateFunction::Sum,
            AggregateFunction::Avg,
            AggregateFunction::Min,
            AggregateFunction::Max,
            AggregateFunction::First,
            AggregateFunction::Last,
            AggregateFunction::Collect,
            AggregateFunction::StdDev,
            AggregateFunction::StdDevPop,
            AggregateFunction::Variance,
            AggregateFunction::VariancePop,
            AggregateFunction::PercentileDisc,
            AggregateFunction::PercentileCont,
            AggregateFunction::GroupConcat,
            AggregateFunction::Sample,
            AggregateFunction::CovarSamp,
            AggregateFunction::CovarPop,
            AggregateFunction::Corr,
            AggregateFunction::RegrSlope,
            AggregateFunction::RegrIntercept,
            AggregateFunction::RegrR2,
            AggregateFunction::RegrCount,
            AggregateFunction::RegrSxx,
            AggregateFunction::RegrSyy,
            AggregateFunction::RegrSxy,
            AggregateFunction::RegrAvgx,
            AggregateFunction::RegrAvgy,
        ];
        for function in functions {
            for distinct in [false, true] {
                for floats in [false, true] {
                    let operand = |value: i32| {
                        if function == AggregateFunction::GroupConcat {
                            Value::from(format!("item-{value}"))
                        } else if floats {
                            Value::Float64(f64::from(value))
                        } else {
                            Value::Int64(i64::from(value))
                        }
                    };
                    let update = |state: &mut AggregateState, y, x| {
                        if state.is_bivariate() {
                            state.update_bivariate(Some(operand(y)), Some(operand(x)));
                        } else {
                            state.update(Some(operand(y)));
                        }
                    };
                    let mut populated =
                        AggregateState::new(function, distinct, Some(0.25), Some("|"));
                    update(&mut populated, 1, 2);
                    update(&mut populated, 3, 4);
                    let mut uninterrupted = GroupState {
                        key_values: vec![Value::from("continuation")],
                        // The empty state also exercises optional None payloads,
                        // empty buffers and empty DISTINCT sets on reload.
                        accumulators: vec![
                            populated,
                            AggregateState::new(function, distinct, Some(0.25), Some("|")),
                        ],
                    };
                    let mut bytes = Vec::new();
                    serialize_group_state(&uninterrupted, &mut bytes).unwrap();
                    let mut owner = admitted_owner(1024 * 1024);
                    owner.try_grow(32).unwrap();
                    let manager = std::sync::Arc::clone(owner.manager());
                    let mut restored =
                        deserialize_group_state_admitted(&mut bytes.as_slice(), &mut owner)
                            .unwrap();
                    assert_eq!(restored.key_values, uninterrupted.key_values);
                    assert!(owner.size() > 32, "the decoded key remains owned");
                    if function == AggregateFunction::GroupConcat {
                        match &restored.accumulators[0] {
                            AggregateState::GroupConcat(values, separator)
                            | AggregateState::GroupConcatDistinct(values, separator, _) => {
                                assert_eq!(values.len(), 2);
                                assert_eq!(values[0], "item-1");
                                assert_eq!(values[1], "item-3");
                                assert_eq!(separator, "|");
                            }
                            other => panic!("expected a populated string state, got {other:?}"),
                        }
                    }

                    for (actual, expected) in restored
                        .accumulators
                        .iter()
                        .zip(&uninterrupted.accumulators)
                    {
                        assert_eq!(
                            std::mem::discriminant(actual),
                            std::mem::discriminant(expected)
                        );
                        assert_eq!(seen(actual), seen(expected));
                        assert_eq!(actual.finalize().unwrap(), expected.finalize().unwrap());
                    }
                    for (y, x) in [(3, 4), (5, 8), (9, 16)] {
                        for (actual, expected) in restored
                            .accumulators
                            .iter_mut()
                            .zip(&mut uninterrupted.accumulators)
                        {
                            update(expected, y, x);
                            let second = actual.is_bivariate().then(|| operand(x));
                            actual
                                .update_accounted(Some(operand(y)), second, &mut owner)
                                .unwrap();
                            assert_eq!(seen(actual), seen(expected));
                            assert_eq!(
                                actual.finalize().unwrap(),
                                expected.finalize().unwrap(),
                                "{function:?}, distinct={distinct}, floats={floats}, y={y}"
                            );
                        }
                        assert_eq!(manager.allocated(), owner.size());
                    }

                    let mut continued = Vec::new();
                    serialize_group_state(&restored, &mut continued).unwrap();
                    let mut resumed_owner = admitted_owner(1024 * 1024);
                    resumed_owner.try_grow(32).unwrap();
                    let resumed_manager = std::sync::Arc::clone(resumed_owner.manager());
                    let resumed = deserialize_group_state_admitted(
                        &mut continued.as_slice(),
                        &mut resumed_owner,
                    )
                    .unwrap();
                    assert_eq!(resumed.key_values, uninterrupted.key_values);
                    for (actual, expected) in
                        resumed.accumulators.iter().zip(&uninterrupted.accumulators)
                    {
                        assert_eq!(seen(actual), seen(expected));
                        let output = actual.finalize_accounted(&mut resumed_owner, None).unwrap();
                        assert_eq!(output, expected.finalize().unwrap());
                        drop(output);
                    }
                    assert_eq!(resumed_manager.allocated(), resumed_owner.size());
                    drop(resumed);
                    resumed_owner.try_resize(32).unwrap();
                    assert_eq!(resumed_manager.allocated(), 32);
                    drop(resumed_owner);
                    assert_eq!(resumed_manager.allocated(), 0);
                    drop(restored);
                    owner.try_resize(32).unwrap();
                    assert_eq!(manager.allocated(), 32);
                    drop(owner);
                    assert_eq!(manager.allocated(), 0);
                }
            }
        }
    }
}
