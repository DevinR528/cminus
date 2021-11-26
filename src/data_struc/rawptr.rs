use std::{
    alloc::{alloc, dealloc, realloc, Allocator, Global, Layout},
    cell::{RefCell, UnsafeCell},
    fmt,
    mem::{self, ManuallyDrop},
    ptr::{self, NonNull},
    slice::{self, Iter, IterMut},
    sync::atomic::{AtomicBool, Ordering},
    vec::IntoIter,
};

#[macro_export]
macro_rules! rawptr {
    ($x:expr) => {
        $crate::data_struc::rawptr::RawPtr::new($x)
    };
}

pub struct RawPtr<T> {
    ptr: UnsafeCell<T>,
    is_used: AtomicBool,
}

impl<T> RawPtr<T> {
    /// Make a `RawPtr` with the given value.
    #[inline]
    pub fn new(it: T) -> Self {
        Self { ptr: UnsafeCell::new(it), is_used: AtomicBool::default() }
    }

    /// Return a reference to the value.
    #[inline]
    pub fn get(&self) -> &T {
        self.is_used.store(true, Ordering::Release);
        let x = unsafe { &*self.ptr.get() };
        self.is_used.store(false, Ordering::Release);
        x
    }

    /// Return a reference to the value.
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        unsafe { &mut *self.ptr.get() }
    }

    /// Replaces the previous value with `it` and returns the old one.
    #[inline]
    pub fn set(&self, it: T) -> T {
        debug_assert!(!self.is_used.load(Ordering::Acquire));
        unsafe {
            let ptr = self.ptr.get();
            mem::replace(&mut *ptr, it)
        }
    }
}

#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl<T: Send> Send for RawPtr<T> {}
unsafe impl<T: Sync> Sync for RawPtr<T> {}

impl<T: Clone> Clone for RawPtr<T> {
    fn clone(&self) -> Self {
        let it = self.get();
        RawPtr::new(it.clone())
    }
}

impl<T: fmt::Debug> fmt::Debug for RawPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().fmt(f)
    }
}

impl<T: std::hash::Hash> std::hash::Hash for RawPtr<T> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(self.get(), state)
    }
}

impl<T: PartialEq> PartialEq for RawPtr<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(self.get(), other.get())
    }
}

impl<T: Eq> Eq for RawPtr<T> {}

impl<T> Drop for RawPtr<T> {
    fn drop(&mut self) {
        debug_assert!(!self.is_used.load(Ordering::Acquire));
    }
}

#[test]
fn multiple_sets() {
    let x = crate::rawptr!(0_usize);

    {
        x.set(10);
    }
    assert_eq!(x.get(), &10);

    {
        x.set(1);
    }
    assert_eq!(x.get(), &1);
}

#[test]
fn multiple_get_muts() {
    let mut x = crate::rawptr!(0_usize);

    {
        *x.get_mut() = 10;
    }
    assert_eq!(x.get(), &10);

    {
        *x.get_mut() = 1;
    }
    assert_eq!(x.get(), &1);
}

#[test]
fn multiple_sets_string() {
    let x = crate::rawptr!(String::new());

    {
        x.set("ender".to_string());
    }
    assert_eq!(x.get(), "ender");

    {
        x.set("wiggen".to_string());
    }
    assert_eq!(x.get(), "wiggen");
}

#[test]
fn multiple_get_muts_string() {
    let mut x = crate::rawptr!(String::new());

    {
        *x.get_mut() = "hello".to_string();
    }
    assert_eq!(x.get(), "hello");

    {
        *x.get_mut() = "world".to_string();
    }
    assert_eq!(x.get(), "world");
}

#[test]
fn test_drop_leaks() {
    static mut DROP_COUNTER: usize = 0;

    #[derive(Debug)]
    enum Droppable {
        DroppedTwice(Box<i32>),
        PanicOnDrop,
    }

    impl Drop for Droppable {
        fn drop(&mut self) {
            match self {
                Droppable::DroppedTwice(_) => unsafe {
                    DROP_COUNTER += 1;
                },
                Droppable::PanicOnDrop => {
                    if !std::thread::panicking() {
                        panic!();
                    }
                }
            }
        }
    }

    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let v = crate::rawptr!(Droppable::DroppedTwice(Box::new(123)));
        let x = crate::rawptr!(Droppable::PanicOnDrop);
    }));

    assert_eq!(unsafe { DROP_COUNTER }, 1);
}
