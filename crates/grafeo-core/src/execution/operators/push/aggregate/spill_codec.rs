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
use crate::execution::operators::accumulator::{AggregateFunction, AggregateState, HashableValue};
use crate::execution::spill::{deserialize_value, serialize_value};

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
            let sum = read_i64(r)?;
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
