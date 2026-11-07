//! A long constant list on the right of `IN`, as a set: a row reads its needle's few candidates
//! rather than every item, and `=` still decides each, so the answer is the walk's.

use grafeo_common::types::Value;
use grafeo_common::utils::hash::FxHashMap;
use smallvec::SmallVec;

use super::HashKey;
use super::value_utils::ValueKinds;

/// The fewest items a list needs to become a set: below this the walk is as cheap as building one.
const MIN_ITEMS: usize = 16;

/// The items of a constant list, by the key under which `=` finds them (see
/// [`HashKey::for_equality`]): values it finds equal share a key, and values it tells apart may too.
#[derive(Debug)]
pub struct InSet {
    items: FxHashMap<HashKey, SmallVec<[Value; 1]>>,
    holds_null: bool,
    /// The kinds of the items that are not null.
    kinds: ValueKinds,
}

impl InSet {
    /// The set of a list of `items`, when it is long enough and holds only integers, floats,
    /// strings, booleans and nulls. Another item keeps the walk, whose rules it has of its own.
    #[must_use]
    pub fn of(items: &[&Value]) -> Option<Self> {
        if items.len() < MIN_ITEMS {
            return None;
        }
        let mut set = Self {
            items: FxHashMap::default(),
            holds_null: false,
            kinds: ValueKinds::default(),
        };
        for item in items {
            match item {
                Value::Null => set.holds_null = true,
                Value::Int64(_) | Value::Float64(_) | Value::String(_) | Value::Bool(_) => {
                    set.kinds = set.kinds.union(ValueKinds::of(item));
                    let key = HashKey::for_equality(item)?;
                    set.items.entry(key).or_default().push((*item).clone());
                }
                _ => return None,
            }
        }
        Some(set)
    }

    /// The kinds of the items that are not null (see [`ValueKinds`]).
    #[must_use]
    pub fn kinds(&self) -> ValueKinds {
        self.kinds
    }

    /// Whether `needle`, which is not null, is in the list: true for an item of its key that `equal`
    /// finds equal, otherwise null when the list holds a null, and false.
    pub(super) fn answer(&self, needle: &Value, equal: impl Fn(&Value) -> bool) -> Value {
        let found = HashKey::for_equality(needle)
            .and_then(|key| self.items.get(&key))
            .is_some_and(|items| items.iter().any(equal));
        if found {
            Value::Bool(true)
        } else if self.holds_null {
            Value::Null
        } else {
            Value::Bool(false)
        }
    }
}
