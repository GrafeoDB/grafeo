//! How a batch search runs its queries: on rayon's pool with the `parallel`
//! feature, one after another without it. The feature picks the module
//! below; every batch search calls [`map`] alone, so no search has a body
//! of its own per feature.

#[cfg(feature = "parallel")]
mod run {
    use rayon::prelude::*;

    /// `search` of each of `queries`, in parallel, in the queries' order.
    pub(crate) fn map<Q, R, F>(queries: &[Q], search: F) -> Vec<R>
    where
        Q: Sync,
        R: Send,
        F: Fn(&Q) -> R + Sync + Send,
    {
        queries.par_iter().map(search).collect()
    }
}

#[cfg(not(feature = "parallel"))]
mod run {
    /// `search` of each of `queries`, one after another.
    pub(crate) fn map<Q, R, F>(queries: &[Q], search: F) -> Vec<R>
    where
        Q: Sync,
        R: Send,
        F: Fn(&Q) -> R + Sync + Send,
    {
        queries.iter().map(search).collect()
    }
}

pub(crate) use run::map;

#[cfg(test)]
mod tests {
    use super::map;

    #[test]
    fn the_results_come_in_the_queries_order() {
        let queries: Vec<u64> = (0..1_000).collect();
        assert_eq!(
            map(&queries, |query| query * 3),
            (0..1_000).map(|q| q * 3).collect::<Vec<_>>()
        );
    }
}
