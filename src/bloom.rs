//! Bloom filter used by v0.2.0-alpha.6+ `SSTables` to accelerate
//! negative point lookups.
//!
//! A Bloom filter can answer "is `key` *definitely not* in this set?"
//! with no false negatives and a bounded false-positive rate. In our
//! case each `LoadedSstable` carries one, so `get_blob` can skip
//! probing the record map when the Bloom rejects the key.
//!
//! # Sizing
//!
//! The filter is sized for the target false-positive rate `p` at the
//! expected element count `n`:
//!
//! - `m` (number of bits) = `⌈-n · ln(p) / (ln 2)²⌉`
//! - `k` (number of hashes) = `⌈m/n · ln 2⌉`
//!
//! At `p = 0.01` the classic result is `m ≈ 9.585 · n`, `k = 7`, i.e.
//! about 1.2 bytes per key.
//!
//! # Hashing
//!
//! We derive the `k` hash positions from a single [`std::hash::SipHasher`]-based
//! computation via double hashing (Kirsch-Mitzenmacher 2006):
//!
//! ```text
//! h_i(key) = (h1(key) + i · h2(key)) mod m
//! ```
//!
//! where `h1` and `h2` come from a single 128-bit SipHash-2-4 output,
//! split into two `u64` halves. This gives us `k` independent-looking
//! positions with just one hash call per key, and keeps the crate
//! dependency-free (`std::hash::DefaultHasher` is the `SipHasher`
//! implementation shipped with the standard library).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A Bloom filter over byte-string keys.
///
/// Cheap to clone (its bit vector is stored in a `Vec<u8>` which shares
/// nothing between clones — deliberate: `SSTable` loads produce one owned
/// `BloomFilter` per file, and threads that want to share should wrap
/// in `Arc`).
#[derive(Debug, Clone)]
pub struct BloomFilter {
    /// Underlying bit set. Bit at position `i` lives at
    /// `bits[i / 8] & (1 << (i % 8))`.
    bits: Vec<u8>,
    /// Total number of bits (`bits.len() * 8` unless the last byte is
    /// only partially used, which the constructor prevents).
    num_bits: u64,
    /// Number of hash positions probed on `insert` and `contains`.
    num_hashes: u32,
}

impl BloomFilter {
    /// Construct a Bloom filter sized for `expected_elements` at a
    /// false-positive rate of `false_positive_rate`.
    ///
    /// The two arguments together determine `num_bits` and `num_hashes`
    /// per the classic formulas. Panics are avoided by clamping both
    /// derived values to at least 1.
    ///
    /// # Panics
    /// Panics if `false_positive_rate` is not in the exclusive
    /// interval `(0.0, 1.0)`. Values outside that range would make the
    /// derivation degenerate (`log(1.0) = 0`, `log(0.0) = -∞`).
    #[must_use]
    pub fn with_capacity(expected_elements: usize, false_positive_rate: f64) -> Self {
        assert!(
            false_positive_rate > 0.0 && false_positive_rate < 1.0,
            "Bloom false positive rate must be in (0, 1); got {false_positive_rate}",
        );
        let n = expected_elements.max(1) as f64;
        let ln2 = std::f64::consts::LN_2;
        let m_ideal = -n * false_positive_rate.ln() / (ln2 * ln2);
        let num_bits = m_ideal.ceil().max(1.0) as u64;
        let k_ideal = (m_ideal / n) * ln2;
        let num_hashes = k_ideal.ceil().max(1.0) as u32;

        let num_bytes = usize::try_from(num_bits.div_ceil(8)).unwrap_or(usize::MAX);
        Self {
            bits: vec![0_u8; num_bytes],
            num_bits,
            num_hashes,
        }
    }

    /// Rebuild a filter from its serialized components. Used by the
    /// `SSTable` reader to restore an on-disk filter without going
    /// through `with_capacity` (which would re-derive the parameters
    /// and might land in a different sizing regime).
    #[must_use]
    pub fn from_raw(bits: Vec<u8>, num_bits: u64, num_hashes: u32) -> Self {
        Self {
            bits,
            num_bits,
            num_hashes,
        }
    }

    /// Return the raw bit vector for serialization.
    #[must_use]
    pub fn as_bits(&self) -> &[u8] {
        &self.bits
    }

    /// Return the configured number of bits.
    #[must_use]
    pub fn num_bits(&self) -> u64 {
        self.num_bits
    }

    /// Return the configured number of hash positions.
    #[must_use]
    pub fn num_hashes(&self) -> u32 {
        self.num_hashes
    }

    /// Add `key` to the filter.
    pub fn insert(&mut self, key: &[u8]) {
        let (h1, h2) = hash_pair(key);
        let m = self.num_bits;
        for i in 0..u64::from(self.num_hashes) {
            let bit = h1.wrapping_add(i.wrapping_mul(h2)) % m;
            let byte = usize::try_from(bit / 8).unwrap_or(0);
            let mask = 1_u8 << (bit % 8);
            self.bits[byte] |= mask;
        }
    }

    /// `true` if `key` *may* be present in the set. `false` if `key`
    /// is *definitely not* present.
    #[must_use]
    pub fn contains(&self, key: &[u8]) -> bool {
        let (h1, h2) = hash_pair(key);
        let m = self.num_bits;
        for i in 0..u64::from(self.num_hashes) {
            let bit = h1.wrapping_add(i.wrapping_mul(h2)) % m;
            let byte = usize::try_from(bit / 8).unwrap_or(0);
            let mask = 1_u8 << (bit % 8);
            if self.bits[byte] & mask == 0 {
                return false;
            }
        }
        true
    }
}

/// Compute two `u64` hashes for `key` from a single SipHash-2-4 call
/// per hash. Feeding a differentiating suffix into the second hash is
/// enough to make the two independent for Kirsch-Mitzenmacher's
/// derivation to hold in practice.
fn hash_pair(key: &[u8]) -> (u64, u64) {
    let mut h1 = DefaultHasher::new();
    key.hash(&mut h1);
    let a = h1.finish();

    let mut h2 = DefaultHasher::new();
    key.hash(&mut h2);
    0x000C_0FFE_EBAD_D005_u64.hash(&mut h2);
    let b = h2.finish();

    // Guarantee `h2` is non-zero so `h1 + i · h2` visits distinct
    // positions across `i`.
    (a, if b == 0 { 1 } else { b })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_filter_rejects_every_key() {
        let bf = BloomFilter::with_capacity(1024, 0.01);
        for key in [b"a", b"b", b"c"] {
            assert!(
                !bf.contains(key.as_slice()),
                "empty filter must reject {key:?}"
            );
        }
    }

    #[test]
    fn inserted_keys_are_never_rejected_no_false_negatives() {
        let mut bf = BloomFilter::with_capacity(1024, 0.01);
        let keys: Vec<Vec<u8>> = (0..1024)
            .map(|i| format!("key-{i:04}").into_bytes())
            .collect();
        for k in &keys {
            bf.insert(k);
        }
        for k in &keys {
            assert!(bf.contains(k), "false negative on {k:?}");
        }
    }

    #[test]
    fn false_positive_rate_stays_within_2x_of_target() {
        // Target 1%; measure over unseen keys. Allow up to 2× the
        // configured rate to absorb variance; α-3.3b-2 can tighten.
        let n = 4_096_usize;
        let mut bf = BloomFilter::with_capacity(n, 0.01);
        for i in 0..n {
            bf.insert(format!("insert-{i}").as_bytes());
        }
        let probes = 4_096;
        let mut fps = 0;
        for i in 0..probes {
            if bf.contains(format!("probe-{i}").as_bytes()) {
                fps += 1;
            }
        }
        let observed = f64::from(fps) / f64::from(probes);
        assert!(
            observed < 0.02,
            "observed false positive rate {observed:.4} exceeds 2× target"
        );
    }

    #[test]
    fn roundtrip_through_raw_bits_preserves_membership() {
        let mut bf = BloomFilter::with_capacity(256, 0.01);
        for k in [b"one".as_slice(), b"two".as_slice(), b"three".as_slice()] {
            bf.insert(k);
        }
        let restored = BloomFilter::from_raw(bf.as_bits().to_vec(), bf.num_bits(), bf.num_hashes());
        for k in [b"one".as_slice(), b"two".as_slice(), b"three".as_slice()] {
            assert!(restored.contains(k), "roundtrip lost membership for {k:?}");
        }
        assert!(!restored.contains(b"never-inserted"));
    }

    #[test]
    fn sizing_scales_with_element_count() {
        let small = BloomFilter::with_capacity(100, 0.01);
        let large = BloomFilter::with_capacity(10_000, 0.01);
        assert!(
            large.num_bits() > 50 * small.num_bits(),
            "10 000× the elements should demand roughly 100× the bits; \
             got small={} large={}",
            small.num_bits(),
            large.num_bits()
        );
    }
}
