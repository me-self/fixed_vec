use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::cmp::max;
use std::fmt::{self, Debug, Formatter};
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut};
use std::ptr::{NonNull, drop_in_place, slice_from_raw_parts_mut};
use std::slice;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};

mod iter;
pub use iter::IntoIter;

/// A thread safe [`Vec`]-like structure that never implicitly reallocates.
///
/// Because it uses atomics and does not reallocate, [`Self::push`] does not
/// require locks or a mutable reference to self.
pub struct FixedVec<T> {
    ptr: NonNull<T>,
    next_idx: AtomicUsize,
    len: AtomicUsize,
    cap: usize,
}

// SAFETY: operations on the same value are atomic.
unsafe impl<T: Send> Send for FixedVec<T> {}

// SAFETY: the addresses are all based on the atomic length and unmodified
// pointer. They cannot overlap.
unsafe impl<T: Sync> Sync for FixedVec<T> {}

impl<T> FixedVec<T> {
    /// Creates a new, empty [`FixedVec<T>`] with the specified `capacity`.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` exceeds [`isize::MAX`] bytes.
    #[inline]
    pub fn new(capacity: usize) -> Self {
        let ptr;
        let layout = Layout::array::<T>(capacity).expect("Layout overflow");
        if layout.size() == 0 {
            ptr = NonNull::dangling();
        } else {
            // SAFETY: we check for a zero-sized type or capacity above.
            let raw_ptr = unsafe { alloc(layout) } as *mut T;

            if raw_ptr.is_null() {
                handle_alloc_error(layout);
            }

            // SAFETY: we check for a null pointer above.
            ptr = unsafe { NonNull::new_unchecked(raw_ptr) };
        }

        Self {
            ptr,
            next_idx: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
            cap: capacity,
        }
    }

    /// Reallocates this collection with the given `new_capacity`.
    ///
    /// # Panics
    ///
    /// Panics if `new_capacity` exceeds [`isize::MAX`] bytes.
    #[inline]
    pub fn realloc(&mut self, new_capacity: usize) {
        let len = self.len.load(Relaxed);
        if new_capacity <= len {
            panic!("new capacity must be greater than the current length");
        }

        let new_vec = Self::new(new_capacity);
        unsafe { new_vec.ptr.copy_from_nonoverlapping(self.ptr, len) };

        new_vec.next_idx.store(len, Relaxed);
        new_vec.len.store(len, Relaxed);

        // We move new_vec into self and get the old self, so we can drop the old one.
        let old_vec = std::mem::replace(self, new_vec);
        old_vec.len.store(0, Relaxed);
        // old_vec will be dropped at the end of this scope, deallocating its
        // memory.
    }

    /// Returns the length of this collection.
    ///
    /// This is a [`Relaxed`] load. It may include incompletely written elements,
    /// thus should not be used as a bound check when indexing except via a
    /// unique reference. See [`Self::len`] for a length that only includes written
    /// elements.
    #[inline]
    pub fn reserved_len(&self) -> usize {
        self.len.load(Relaxed)
    }

    /// Returns the length of this collection.
    ///
    /// This is an [`Acquire`] load. All elements up to this length are
    /// guaranteed to be initialized. See [`Self::reserved_len`] for a length
    /// that may include incompletely written elements.
    #[inline]
    pub fn len(&self) -> usize {
        // Acquire to ensure writes up to this length have actually completed.
        self.len.load(Acquire)
    }

    /// Returns the capacity of this collection.
    ///
    /// This value is guaranteed not to exceed [`isize::MAX`] (as this is the
    /// maximum layout size).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Attempts to push an element to this collection.
    ///
    /// Returns [`Ok<usize>`] (the index) on success and [`Err<T>`] (containing
    /// the passed value) on failure. The returned index should be used to
    /// access this element since, in the case of concurrent pushes, there is no
    /// guarantee that the length before pushing will be the index of this
    /// element.
    #[inline]
    pub fn push(&self, value: T) -> Result<usize, T> {
        // Using `Relaxed` since we don't care what goes on at previous indices when
        // pushing.
        let idx = self.next_idx.fetch_add(1, Relaxed);

        // For idx to wrap, we would need `usize::MAX - isize::MAX` concurrent pushes
        // since cap can't exceed `isize::MAX`. We can't possibly have that many
        // threads.
        if idx >= self.cap {
            self.next_idx.fetch_sub(1, Relaxed);
            return Err(value);
        }

        unsafe {
            let ptr = self.ptr.add(idx);
            ptr.write(value);
        }
        while self
            .len
            .compare_exchange_weak(idx, idx + 1, Release, Relaxed)
            .is_err()
        {
            std::hint::spin_loop();
        }
        Ok(idx)
    }

    /// Extracts a slice of the entire collection.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: all elements up to `len` have been initialized and are of type `T`.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len()) }
    }

    /// Extracts a mutable slice of the entire collection.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: all elements up to `len` have been initialized and are of type `T`.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len()) }
    }
}

impl<T> Deref for FixedVec<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for FixedVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> Default for FixedVec<T> {
    fn default() -> Self {
        // The default capacity is 1 since a capacity of 0 would be pretty useless.
        Self::new(1)
    }
}

impl<T: Debug> Debug for FixedVec<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&**self, f)
    }
}

impl<T> FromIterator<T> for FixedVec<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let cap = upper.unwrap_or(lower);
        let mut vec = Self::new(cap);

        for item in iter {
            // Check for an error since we can't rely on `size_hint` for safety.
            if let Err(item) = vec.push(item) {
                // We have an exclusive reference, so relaxed operations are fine.
                let len = vec.reserved_len();
                let new_cap = max(len.next_power_of_two(), len + 1);
                vec.realloc(new_cap);
                let _ = vec.push(item);
            }
        }
        vec
    }
}

impl<T> Extend<T> for FixedVec<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let iter_size = upper.unwrap_or(lower);
        let new_cap = self.reserved_len() + iter_size;
        if new_cap > self.cap {
            self.realloc(new_cap);
        }

        for item in iter {
            // Check for an error since we can't rely on `size_hint` for safety.
            if let Err(item) = self.push(item) {
                // We have an exclusive reference, so relaxed operations are fine.
                let len = self.reserved_len();
                let new_cap = max(len.next_power_of_two(), len + 1);
                self.realloc(new_cap);
                let _ = self.push(item);
            }
        }
    }
}

impl<T: Clone> Clone for FixedVec<T> {
    fn clone(&self) -> Self {
        let len = self.len();
        let new_vec = Self::new(len);

        for i in 0..len {
            if let Some(item) = self.get(i) {
                let _ = new_vec.push(item.clone());
            }
        }

        new_vec
    }
}

impl<T> Drop for FixedVec<T> {
    fn drop(&mut self) {
        struct DropGuard<T> {
            ptr: NonNull<T>,
            cap: usize,
        }

        impl<T> Drop for DropGuard<T> {
            fn drop(&mut self) {
                dealloc_vec(self.ptr, self.cap);
            }
        }

        let _guard = DropGuard {
            ptr: self.ptr,
            cap: self.cap,
        };

        // Drop elements. We have an exclusive reference, so relaxed operations are
        // fine.
        let elems = slice_from_raw_parts_mut(self.ptr.as_ptr(), self.reserved_len());
        unsafe {
            drop_in_place(elems);
        }

        // Deallocation occurs in DropGuard. This is called even if dropping
        // elements causes a panic.
    }
}

fn dealloc_vec<T>(ptr: NonNull<T>, capacity: usize) {
    // This should not return an error since this is the same layout as was used for
    // allocation.
    let layout = Layout::array::<T>(capacity).unwrap();
    unsafe {
        // We can't deallocate if it's zero-sized.
        if layout.size() > 0 {
            // SAFETY: the same layout was used to allocate.
            dealloc(ptr.as_ptr() as *mut u8, layout);
        }
    }
}
