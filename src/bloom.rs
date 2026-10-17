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
//! The positions are part of the stored `SSTable`, so the hash is this crate's own and fixed:
//! SipHash-2-4 (Aumasson and Bernstein, 64-bit output) implemented below, checked against the
//! reference test vectors. `h1` and `h2` are two SipHash-2-4 values of the key bytes under two
//! fixed keys, and the `k` positions come from double hashing (Kirsch-Mitzenmacher 2006):
//!
//! ```text
//! h_i(key) = (h1(key) + i · h2(key)) mod m      (wrapping u64, h2 = 1 when it is 0)
//! ```
//!
//! [`BLOOM_HASH_ID`] names exactly this derivation (the SHA-256 of
//! [`BLOOM_HASH_DESCRIPTION`]); the `SSTable` stores it next to the filter, and a reader uses
//! a stored filter only when the identifier is this one. Filters written before 0.6.0 used
//! `std::hash::DefaultHasher`, whose algorithm the standard library does not fix between
//! releases: those are not probed but rebuilt from the file's keys (see
//! `crate::blob_sstable::BloomSource`).

/// What [`BLOOM_HASH_ID`] identifies, in words
pub const BLOOM_HASH_DESCRIPTION: &str = "alice-db bloom positions v1: SipHash-2-4 (64-bit output) of the key bytes alone, h1 with key k0=0x0706050403020100 k1=0x0f0e0d0c0b0a0908, h2 with key k0=0x0f0e0d0c0b0a0908 k1=0x0706050403020100 (0 becomes 1), position i = (h1 + i*h2) mod num_bits, wrapping u64 arithmetic";

/// SHA-256 of [`BLOOM_HASH_DESCRIPTION`]: the identifier of the position derivation, stored
/// with every filter (checked against the description by `tests::the_hash_id_is_the_description`)
pub const BLOOM_HASH_ID: [u8; 32] =
    hex32("75e603a497f7aab2eaea4b500f71269ed0102497b184b1192410a749bf435bdf");

const fn hex32(s: &str) -> [u8; 32] {
    const fn nibble(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => panic!("not a lower-case hex digit"),
        }
    }
    let b = s.as_bytes();
    assert!(b.len() == 64, "an identifier is 64 hex digits");
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = nibble(b[2 * i]) * 16 + nibble(b[2 * i + 1]);
        i += 1;
    }
    out
}

const K_A: (u64, u64) = (0x0706_0504_0302_0100, 0x0f0e_0d0c_0b0a_0908);
const K_B: (u64, u64) = (0x0f0e_0d0c_0b0a_0908, 0x0706_0504_0302_0100);

/// SipHash-2-4 with a 64-bit output (the reference algorithm: 2 compression rounds per
/// 8-byte word, 4 finalisation rounds, the message length in the top byte of the last word)
#[must_use]
pub fn siphash24(k0: u64, k1: u64, data: &[u8]) -> u64 {
    #[inline(always)]
    fn round(v: &mut [u64; 4]) {
        v[0] = v[0].wrapping_add(v[1]);
        v[1] = v[1].rotate_left(13);
        v[1] ^= v[0];
        v[0] = v[0].rotate_left(32);
        v[2] = v[2].wrapping_add(v[3]);
        v[3] = v[3].rotate_left(16);
        v[3] ^= v[2];
        v[0] = v[0].wrapping_add(v[3]);
        v[3] = v[3].rotate_left(21);
        v[3] ^= v[0];
        v[2] = v[2].wrapping_add(v[1]);
        v[1] = v[1].rotate_left(17);
        v[1] ^= v[2];
        v[2] = v[2].rotate_left(32);
    }
    let mut v = [
        k0 ^ 0x736f_6d65_7073_6575,
        k1 ^ 0x646f_7261_6e64_6f6d,
        k0 ^ 0x6c79_6765_6e65_7261,
        k1 ^ 0x7465_6462_7974_6573,
    ];
    let mut chunks = data.chunks_exact(8);
    for c in &mut chunks {
        let m = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
        v[3] ^= m;
        round(&mut v);
        round(&mut v);
        v[0] ^= m;
    }
    let rest = chunks.remainder();
    // the length modulo 256 in the top byte, the remaining bytes little-endian below it
    let mut last = (data.len() as u64 & 0xff) << 56;
    for (i, &b) in rest.iter().enumerate() {
        last |= u64::from(b) << (8 * i);
    }
    v[3] ^= last;
    round(&mut v);
    round(&mut v);
    v[0] ^= last;
    v[2] ^= 0xff;
    for _ in 0..4 {
        round(&mut v);
    }
    v[0] ^ v[1] ^ v[2] ^ v[3]
}

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
        // The filter's size is part of the SSTable: the serialised filter
        // occupies `num_bits / 8` bytes and the reader probes `num_hashes`
        // positions. A platform whose `ln` differs in the last place can round
        // `ceil()` to a different bit count, so the same entries would produce
        // a different file on a different target. The logarithm therefore goes
        // through `alice-det-math`, which returns the same bits everywhere.
        //
        // Measured on aarch64-apple-darwin when this moved off the platform
        // `ln`: `ln64` and `f64::ln` disagree in the last place on 1,220 of
        // 74,693 inputs on a logarithmic grid over (1e-12, 1), and on 71 of
        // 1,010 sampled false positive rates, but the resulting
        // `(num_bits, num_hashes)` agreed on all 21,299,890 (n, rate) pairs
        // tried, and `ln(0.01)` (the rate the SSTable writer uses) is the
        // same bits in both. Filters already on disk are read back with their
        // stored size and are unaffected either way.
        let m_ideal = -n * alice_det_math::ln64(false_positive_rate) / (ln2 * ln2);
        let num_bits = alice_det_math::max64(m_ideal.ceil(), 1.0) as u64;
        let k_ideal = (m_ideal / n) * ln2;
        let num_hashes = alice_det_math::max64(k_ideal.ceil(), 1.0) as u32;

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

/// The two `u64` hashes of `key` (module docs, "Hashing"): SipHash-2-4 under two fixed keys;
/// `h2` is made non-zero so that `h1 + i · h2` visits distinct positions across `i`
fn hash_pair(key: &[u8]) -> (u64, u64) {
    let a = siphash24(K_A.0, K_A.1, key);
    let b = siphash24(K_B.0, K_B.1, key);
    (a, if b == 0 { 1 } else { b })
}

/// The bit positions `key` sets in a filter of `num_bits` bits with `num_hashes` probes, in
/// probe order (what [`BloomFilter::insert`] sets and [`BloomFilter::contains`] reads)
#[must_use]
pub fn positions(key: &[u8], num_bits: u64, num_hashes: u32) -> Vec<u64> {
    let (h1, h2) = hash_pair(key);
    (0..u64::from(num_hashes))
        .map(|i| h1.wrapping_add(i.wrapping_mul(h2)) % num_bits)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference vectors of SipHash-2-4 (64-bit output, key 00 01 .. 0f, message
    /// 00 01 .. n-1), the first three of the published table
    #[test]
    fn siphash24_matches_the_reference_vectors() {
        let msg: Vec<u8> = (0..3u8).collect();
        let want = [
            0x726f_db47_dd0e_0e31_u64,
            0x74f8_39c5_93dc_67fd,
            0x0d6c_8009_d9a9_4f5a,
        ];
        for (n, w) in want.iter().enumerate() {
            assert_eq!(siphash24(K_A.0, K_A.1, &msg[..n]), *w, "length {n}");
        }
    }

    #[test]
    fn the_hash_id_is_the_description() {
        use sha2::{Digest, Sha256};
        let digest: [u8; 32] = Sha256::digest(BLOOM_HASH_DESCRIPTION.as_bytes()).into();
        assert_eq!(digest, BLOOM_HASH_ID);
    }

    /// ALICE-Law G10: the positions a key sets are stored on disk, so they are pinned. The
    /// empty key's row is h1 = 0x726fdb47dd0e0e31 (the reference vector) and h2 from the
    /// standard library's `SipHasher` under the second key, reduced independently of this code
    #[test]
    fn positions_are_pinned() {
        let cases: [(&[u8], [u64; 7]); 4] = [
            (b"", [553, 3130, 5707, 8284, 5693, 8270, 1261]),
            (b"a", [8276, 7710, 7144, 6578, 6012, 5446, 4880]),
            (b"key-0000", [7351, 4873, 2395, 4335, 1857, 8965, 6487]),
            (
                b"\x00\xff binary",
                [9340, 8587, 7834, 7081, 1160, 407, 9240],
            ),
        ];
        for (key, want) in cases {
            assert_eq!(positions(key, 9586, 7), want, "{key:?}");
        }
    }

    /// The same algorithm as the standard library's (deprecated, still SipHash-2-4)
    /// `SipHasher::write` on the raw bytes, over every length up to 64 and both keys
    #[test]
    #[allow(deprecated)]
    fn siphash24_agrees_with_std_siphasher() {
        use std::hash::Hasher;
        let data: Vec<u8> = (0..64u8)
            .map(|i| i.wrapping_mul(37).wrapping_add(11))
            .collect();
        let mut compared = 0;
        for k in [K_A, K_B] {
            for n in 0..=data.len() {
                let mut h = std::hash::SipHasher::new_with_keys(k.0, k.1);
                h.write(&data[..n]);
                assert_eq!(siphash24(k.0, k.1, &data[..n]), h.finish(), "length {n}");
                compared += 1;
            }
        }
        assert_eq!(compared, 130);
    }

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
