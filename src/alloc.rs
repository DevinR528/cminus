use std::{
	alloc::{GlobalAlloc, Layout, System},
	fmt,
	sync::atomic::{AtomicUsize, Ordering},
};

/// An instrumented instance of the system allocator.
pub static INSTRUMENTED_SYSTEM: StatsAlloc<System> = StatsAlloc {
	allocations: AtomicUsize::new(0),
	bytes_allocated: AtomicUsize::new(0),
	inner: System,
};

/// Allocator statistics
#[derive(Clone, Copy, Default, Debug, Hash, PartialEq, Eq)]
pub struct Stats {
	/// Count of allocation operations
	pub allocations: usize,
	/// Total bytes requested by allocations
	pub bytes_allocated: usize,
}

impl fmt::Display for Stats {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "allocations: {} bytes: {}", self.allocations, self.bytes_allocated)
	}
}

/// An instrumenting middleware which keeps track of allocation, deallocation,
/// and reallocation requests to the underlying global allocator.
#[derive(Default, Debug)]
pub struct StatsAlloc<T: GlobalAlloc> {
	allocations: AtomicUsize,
	bytes_allocated: AtomicUsize,
	inner: T,
}

// A snapshot of the allocation statistics, which can be used to determine
/// allocation changes while the `Region` is alive.
#[derive(Debug)]
pub struct Region<'a, T: GlobalAlloc + 'a> {
	alloc: &'a StatsAlloc<T>,
	initial_stats: Stats,
}

impl<T: GlobalAlloc> StatsAlloc<T> {
	/// Provides access to an instrumented instance of the given global
	/// allocator.
	pub const fn new(inner: T) -> Self {
		StatsAlloc { allocations: AtomicUsize::new(0), bytes_allocated: AtomicUsize::new(0), inner }
	}

	/// Takes a snapshot of the current view of the allocator statistics.
	pub fn stats(&self) -> Stats {
		Stats {
			allocations: self.allocations.load(Ordering::SeqCst),
			bytes_allocated: self.bytes_allocated.load(Ordering::SeqCst),
		}
	}
}

impl<'a, T: GlobalAlloc + 'a> Region<'a, T> {
	/// Creates a new region using statistics from the given instrumented
	/// allocator.
	#[inline]
	pub fn new(alloc: &'a StatsAlloc<T>) -> Self {
		Region { alloc, initial_stats: alloc.stats() }
	}

	/// Returns the difference between the currently reported statistics and
	/// those provided by `initial()`, resetting initial to the latest
	/// reported statistics.
	#[inline]
	pub fn change_and_reset(&mut self) -> Stats {
		let latest = self.alloc.stats();

		let mut diff = latest;
		diff.allocations -= self.initial_stats.allocations;
		diff.bytes_allocated -= self.initial_stats.bytes_allocated;

		self.initial_stats = latest;

		diff
	}
}

unsafe impl<'a, T: GlobalAlloc + 'a> GlobalAlloc for &'a StatsAlloc<T> {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		(*self).alloc(layout)
	}

	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		(*self).dealloc(ptr, layout)
	}

	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		(*self).alloc_zeroed(layout)
	}

	unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
		(*self).realloc(ptr, layout, new_size)
	}
}

unsafe impl<T: GlobalAlloc> GlobalAlloc for StatsAlloc<T> {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		self.allocations.fetch_add(1, Ordering::SeqCst);
		self.bytes_allocated.fetch_add(layout.size(), Ordering::SeqCst);
		self.inner.alloc(layout)
	}

	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		self.inner.dealloc(ptr, layout)
	}

	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		self.allocations.fetch_add(1, Ordering::SeqCst);
		self.bytes_allocated.fetch_add(layout.size(), Ordering::SeqCst);
		self.inner.alloc_zeroed(layout)
	}

	unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
		self.inner.realloc(ptr, layout, new_size)
	}
}
