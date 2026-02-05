//! Object pool for reusing frequently allocated types.
//!
//! If you're creating and destroying the same type of object repeatedly
//! (like temporary buffers during query execution), a pool avoids the
//! allocation overhead. Objects are reset and returned to the pool instead
//! of being freed.

use std::ops::{Deref, DerefMut};

use parking_lot::Mutex;

/// A thread-safe object pool for reusing allocations.
///
/// Use [`get()`](Self::get) to grab an object (created fresh if the pool is
/// empty). When you drop the returned [`Pooled`] wrapper, the object goes
/// back to the pool for reuse.
///
/// # Examples
///
/// ```
/// use grafeo_common::memory::ObjectPool;
///
/// // Pool of vectors that get cleared on return
/// let pool = ObjectPool::with_reset(Vec::<u8>::new, |v| v.clear());
///
/// let mut buf = pool.get();
/// buf.extend_from_slice(&[1, 2, 3]);
/// // buf is returned to pool when dropped, and cleared
/// ```
pub struct ObjectPool<T> {
    /// The pool of available objects.
    pool: Mutex<Vec<T>>,
    /// Factory function to create new objects.
    factory: Box<dyn Fn() -> T + Send + Sync>,
    /// Optional reset function called when returning objects to the pool.
    reset: Option<Box<dyn Fn(&mut T) + Send + Sync>>,
    /// Maximum pool size.
    max_size: usize,
}

impl<T> ObjectPool<T> {
    /// Creates a new object pool with the given factory function.
    pub fn new<F>(factory: F) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        Self {
            pool: Mutex::new(Vec::new()),
            factory: Box::new(factory),
            reset: None,
            max_size: 1024,
        }
    }

    /// Creates a new object pool with a factory and reset function.
    ///
    /// The reset function is called when an object is returned to the pool,
    /// allowing you to clear or reinitialize the object for reuse.
    pub fn with_reset<F, R>(factory: F, reset: R) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
        R: Fn(&mut T) + Send + Sync + 'static,
    {
        Self {
            pool: Mutex::new(Vec::new()),
            factory: Box::new(factory),
            reset: Some(Box::new(reset)),
            max_size: 1024,
        }
    }

    /// Sets the maximum pool size.
    ///
    /// Objects returned when the pool is at capacity will be dropped instead.
    #[must_use]
    pub fn with_max_size(mut self, max_size: usize) -> Self {
        self.max_size = max_size;
        self
    }

    /// Pre-populates the pool with `count` objects.
    pub fn prefill(&self, count: usize) {
        let mut pool = self.pool.lock();
        let to_add = count
            .saturating_sub(pool.len())
            .min(self.max_size - pool.len());
        for _ in 0..to_add {
            pool.push((self.factory)());
        }
    }

    /// Takes an object from the pool, creating a new one if necessary.
    ///
    /// Returns a `Pooled` wrapper that will return the object to the pool
    /// when dropped.
    pub fn get(&self) -> Pooled<'_, T> {
        let value = self.pool.lock().pop().unwrap_or_else(|| (self.factory)());
        Pooled {
            pool: self,
            value: Some(value),
        }
    }

    /// Takes an object from the pool without wrapping it.
    ///
    /// The caller is responsible for returning the object via `put()` if desired.
    pub fn take(&self) -> T {
        self.pool.lock().pop().unwrap_or_else(|| (self.factory)())
    }

    /// Returns an object to the pool.
    ///
    /// If the pool is at capacity, the object is dropped instead.
    pub fn put(&self, mut value: T) {
        if let Some(ref reset) = self.reset {
            reset(&mut value);
        }

        let mut pool = self.pool.lock();
        if pool.len() < self.max_size {
            pool.push(value);
        }
        // Otherwise, value is dropped
    }

    /// Returns the current number of objects in the pool.
    #[must_use]
    pub fn available(&self) -> usize {
        self.pool.lock().len()
    }

    /// Returns the maximum pool size.
    #[must_use]
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Clears all objects from the pool.
    pub fn clear(&self) {
        self.pool.lock().clear();
    }
}

/// A borrowed object from the pool - returns automatically when dropped.
///
/// Use [`take()`](Self::take) if you need to keep the object instead of
/// returning it to the pool.
pub struct Pooled<'a, T> {
    pool: &'a ObjectPool<T>,
    value: Option<T>,
}

impl<T> Pooled<'_, T> {
    /// Takes ownership of the inner value, preventing it from being returned to the pool.
    pub fn take(mut self) -> T {
        self.value.take().expect("Value already taken")
    }
}

impl<T> Deref for Pooled<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value.as_ref().expect("Value already taken")
    }
}

impl<T> DerefMut for Pooled<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value.as_mut().expect("Value already taken")
    }
}

impl<T> Drop for Pooled<'_, T> {
    fn drop(&mut self) {
        if let Some(value) = self.value.take() {
            self.pool.put(value);
        }
    }
}

/// A specialized pool for `Vec<T>` that clears vectors on return.
pub type VecPool<T> = ObjectPool<Vec<T>>;

impl<T: 'static> VecPool<T> {
    /// Creates a new vector pool.
    pub fn new_vec_pool() -> Self {
        ObjectPool::with_reset(Vec::new, |v| v.clear())
    }

    /// Creates a new vector pool with pre-allocated capacity.
    pub fn new_vec_pool_with_capacity(capacity: usize) -> Self {
        ObjectPool::with_reset(move || Vec::with_capacity(capacity), |v| v.clear())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_basic() {
        let pool: ObjectPool<Vec<u8>> = ObjectPool::new(Vec::new);

        // Get an object
        let mut obj = pool.get();
        obj.push(1);
        obj.push(2);
        assert_eq!(&*obj, &[1, 2]);

        // Object should be returned to pool when dropped
        drop(obj);
        assert_eq!(pool.available(), 1);

        // Get should return the pooled object
        let obj2 = pool.get();
        assert_eq!(pool.available(), 0);

        // The returned object still has data (no reset function)
        assert_eq!(&*obj2, &[1, 2]);
    }

    #[test]
    fn test_pool_with_reset() {
        let pool: ObjectPool<Vec<u8>> = ObjectPool::with_reset(Vec::new, Vec::clear);

        let mut obj = pool.get();
        obj.push(1);
        obj.push(2);

        drop(obj);

        // Get should return a cleared object
        let obj2 = pool.get();
        assert!(obj2.is_empty());
    }

    #[test]
    fn test_pool_prefill() {
        let pool: ObjectPool<String> = ObjectPool::new(String::new);

        pool.prefill(10);
        assert_eq!(pool.available(), 10);

        // Getting objects should reduce available count
        // Note: we must keep the Pooled handle alive, otherwise it returns the object on drop
        let _obj = pool.get();
        assert_eq!(pool.available(), 9);
    }

    #[test]
    fn test_pool_max_size() {
        let pool: ObjectPool<u64> = ObjectPool::new(|| 0).with_max_size(3);

        pool.prefill(10);
        // Should only have 3 objects
        assert_eq!(pool.available(), 3);

        // Return more than max - extras should be dropped
        let o1 = pool.take();
        let o2 = pool.take();
        let o3 = pool.take();

        assert_eq!(pool.available(), 0);

        pool.put(o1);
        pool.put(o2);
        pool.put(o3);
        pool.put(99); // This one should be dropped

        assert_eq!(pool.available(), 3);
    }

    #[test]
    fn test_pool_take_ownership() {
        let pool: ObjectPool<String> = ObjectPool::new(String::new);

        let mut obj = pool.get();
        obj.push_str("hello");

        // Take ownership - should NOT return to pool
        let owned = obj.take();
        assert_eq!(owned, "hello");
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn test_pool_clear() {
        let pool: ObjectPool<u64> = ObjectPool::new(|| 0);

        pool.prefill(10);
        assert_eq!(pool.available(), 10);

        pool.clear();
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn test_vec_pool() {
        let pool: VecPool<u8> = VecPool::new_vec_pool();

        let mut v = pool.get();
        v.extend_from_slice(&[1, 2, 3]);

        drop(v);

        let v2 = pool.get();
        assert!(v2.is_empty()); // Should be cleared
    }

    #[test]
    fn test_vec_pool_with_capacity() {
        let pool: VecPool<u8> = VecPool::new_vec_pool_with_capacity(100);

        let v = pool.get();
        assert!(v.capacity() >= 100);
    }

    #[test]
    fn test_pool_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let pool: Arc<ObjectPool<Vec<u8>>> = Arc::new(ObjectPool::with_reset(Vec::new, Vec::clear));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let pool = Arc::clone(&pool);
                thread::spawn(move || {
                    for _ in 0..100 {
                        let mut v = pool.get();
                        v.push(42);
                        // v is automatically returned on drop
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Pool should have some objects
        assert!(pool.available() > 0);
    }
}
