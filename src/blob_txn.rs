//! [`BlobStorage::transact`]: reads and one write batch of a blob store with every other writer kept out,
//! the other processes of a shared store included.
//!
//! While the closure runs this thread is marked as inside a transaction of that store. Any call on the store
//! itself from inside the closure (a write would wait on the gate the transaction holds) is refused with
//! [`NestedTransaction`]; the closure reads through the [`Txn`] it is given. The mark is a drop guard, so a
//! panic in the closure takes it down with the gate and the lease, and nothing of the batch is written.

use std::cell::{Cell, RefCell};
use std::io;
use std::ops::Bound;

use crate::blob::{BlobStorage, ScanOrder};

thread_local! {
    /// The stores this thread is inside a transaction of, by address (a store is borrowed for the whole
    /// transaction, so its address does not change under it).
    static INSIDE: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    /// Set while a [`Txn`] read runs, so the store lets that one call through.
    static TXN_READ: Cell<bool> = const { Cell::new(false) };
}

fn address(store: &BlobStorage) -> usize {
    std::ptr::from_ref(store) as usize
}

/// A call on a blob store from inside a [`BlobStorage::transact`] closure of the same store: carried inside an
/// `io::Error` of kind `Deadlock`. Waiting there would never end (the transaction holds what the call waits
/// for), so it is refused at once and nothing is written by it. Read through the closure's [`Txn`] instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct NestedTransaction;

impl std::fmt::Display for NestedTransaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            "a call on the blob store from inside its own transact closure: read through the Txn",
        )
    }
}

impl std::error::Error for NestedTransaction {}

/// `Err(Deadlock / NestedTransaction)` when this thread is inside a transaction of `store` and the call is not
/// a [`Txn`] read.
pub(crate) fn refuse_nested(store: &BlobStorage) -> io::Result<()> {
    if TXN_READ.with(Cell::get) {
        return Ok(());
    }
    let inside = INSIDE.with(|v| v.borrow().contains(&address(store)));
    if inside {
        return Err(io::Error::new(io::ErrorKind::Deadlock, NestedTransaction));
    }
    Ok(())
}

/// Marks this thread as inside a transaction of one store until dropped (a panic drops it too).
pub(crate) struct Inside(usize);

impl Inside {
    pub(crate) fn enter(store: &BlobStorage) -> Self {
        let a = address(store);
        INSIDE.with(|v| v.borrow_mut().push(a));
        Self(a)
    }
}

impl Drop for Inside {
    fn drop(&mut self) {
        INSIDE.with(|v| {
            let mut v = v.borrow_mut();
            if let Some(i) = v.iter().rposition(|&a| a == self.0) {
                v.remove(i);
            }
        });
    }
}

/// Lets one store call through while a [`Txn`] read runs.
struct ReadPass(bool);

impl ReadPass {
    fn enter() -> Self {
        Self(TXN_READ.with(|r| r.replace(true)))
    }
}

impl Drop for ReadPass {
    fn drop(&mut self) {
        TXN_READ.with(|r| r.set(self.0));
    }
}

/// The reads of a [`BlobStorage::transact`] closure: the store as it is with the transaction's lease held, so
/// what they return still holds when the batch is written (another process's writes up to the lease are seen,
/// later ones wait).
pub struct Txn<'a> {
    pub(crate) store: &'a BlobStorage,
}

impl Txn<'_> {
    /// [`BlobStorage::get`].
    ///
    /// # Errors
    /// As [`BlobStorage::get`].
    pub fn get(&self, key: &[u8]) -> io::Result<Option<Vec<u8>>> {
        let _pass = ReadPass::enter();
        self.store.get(key)
    }

    /// [`BlobStorage::scan_prefix`].
    ///
    /// # Errors
    /// As [`BlobStorage::scan_prefix`].
    pub fn scan_prefix(&self, prefix: &[u8]) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let _pass = ReadPass::enter();
        self.store.scan_prefix(prefix)
    }

    /// [`BlobStorage::scan_range`].
    ///
    /// # Errors
    /// As [`BlobStorage::scan_range`].
    pub fn scan_range(
        &self,
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
        order: ScanOrder,
        limit: usize,
    ) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let _pass = ReadPass::enter();
        self.store.scan_range(start, end, order, limit)
    }

    /// [`BlobStorage::count_prefix`].
    ///
    /// # Errors
    /// As [`BlobStorage::count_prefix`].
    pub fn count_prefix(&self, prefix: &[u8]) -> io::Result<usize> {
        let _pass = ReadPass::enter();
        self.store.count_prefix(prefix)
    }
}
