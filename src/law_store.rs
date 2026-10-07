//! Laws stored next to the data: [`SignalLaw`] records, their versions and
//! the verdicts of the evidence judged against them
//!
//! A [`SignalLaw`] (from ALICE-Zip, re-exported here) is `y = f(x)` together
//! with the evidence it was fitted to, the residual measured over that
//! evidence, the `x` range the evidence covers, its provenance and reference
//! values. This module keeps such laws in an [`AliceDB`] under a name:
//!
//! | operation | effect |
//! |-----------|--------|
//! | [`AliceDB::put_law`] | stores the law as the next version of `name` |
//! | [`AliceDB::get_law`] / [`AliceDB::get_law_version`] | restores the latest / a given version through [`SignalLaw::from_parts`], which measures the residual again from the stored evidence |
//! | [`AliceDB::evaluate_law`] | `f(x)` of the latest version; an `x` outside its valid range is refused, never extrapolated |
//! | [`AliceDB::ingest_evidence`] | judges new points with [`SignalLaw::ingest`], records the verdict, and stores a [`Verdict::ParameterUpdate`] as a new version (older versions stay readable) |
//! | [`AliceDB::law_history`] / [`AliceDB::law_versions`] / [`AliceDB::law_names`] | the recorded verdicts in order, the stored versions, the stored names |
//!
//! Both storage backends are supported: records live in the blob key-value
//! store, so they are written to the blob WAL / `SSTable`s by the file
//! backend, kept in memory by [`AliceDB::in_memory`], and carried by
//! [`AliceDB::to_bytes`] / [`AliceDB::from_bytes`].
//!
//! # Layout
//!
//! Names are non-empty UTF-8 strings without a NUL byte. Each record is one
//! blob under a key in the reserved prefix [`LAW_KEY_PREFIX`] (`"\0alice-law\0"`):
//!
//! ```text
//! version  "\0alice-law\0" name "\0v" version (u64 big-endian)  → law record
//! verdict  "\0alice-law\0" name "\0h" seq     (u64 big-endian)  → verdict record
//! law id   "\0alice-law\0" "\0i" law_id (32 bytes)
//!                           name length (u32 big-endian) name
//!                           version (u64 big-endian)     → empty
//! ```
//!
//! The third family addresses a law by its content instead of by the name a
//! person gave it: `law_id` is [`SignalLaw::law_id`], a hash over exactly the
//! inputs the evaluation reads plus an identifier for the numeric semantics it
//! is evaluated under. [`AliceDB::law_pointers_by_id`] returns every
//! `(name, version)` under one identifier and [`AliceDB::evaluate_by_id`]
//! evaluates it — well defined even for several pointers, because equal
//! identifier implies the evaluation returns the same bits. The converse does
//! not hold, so an identifier is not a deduplication key.
//!
//! The pointer lives in the key, not the value, so one identifier can address
//! several laws without the record format holding a list, and storing the same
//! law twice is idempotent. A name is non-empty and holds no NUL byte, so the
//! byte after the shared prefix is never NUL for the first two families and
//! always NUL for this one: the families cannot be a prefix of one another. The
//! name carries its length so `("ab", v)` and `("a", …)` cannot collide.
//!
//! Big-endian numbers make the byte order of the keys the numeric order, so a
//! prefix scan returns versions and verdicts in order; since a name holds no
//! NUL byte, the keys of one name are never a prefix of another name's keys.
//! Versions and verdict sequence numbers start at 1.
//!
//! Law record (integers and `f64` bit patterns little-endian):
//!
//! ```text
//! magic          8 bytes  b"ALAWREC\x01"
//! format         u32      LAW_RECORD_FORMAT (= 1)
//! name           u32 length + UTF-8
//! version        u64
//! coefficients   u32 count + f64 × count   (ascending, in u = (x - lo) / (hi - lo))
//! domain         f64 lo, f64 hi
//! evidence       u64 count + (f64 x, f64 y) × count
//! residual       u64 n, f64 rms, f64 max_abs   (as stored; not trusted on read)
//! provenance     u32 length + UTF-8 source, u32 length + UTF-8 method
//! oracles        u32 count + (f64 x, f64 expected, f64 tolerance, u32 length + UTF-8 source) × count
//! crc32          u32      CRC-32 (IEEE) of every preceding byte
//! ```
//!
//! Verdict record:
//!
//! ```text
//! magic          8 bytes  b"ALAWVRD\x01"
//! format         u32      LAW_RECORD_FORMAT (= 1)
//! seq            u64
//! law_version    u64      version the evidence was judged against
//! evidence_count u64
//! outcome        u8       VerdictKind code
//! outside        u64      points outside the range (OutOfRange), else 0
//! rms            u8 flag + f64   RMS of the new points about the judged law
//! new_version    u8 flag + u64   version stored by a ParameterUpdate
//! crc32          u32
//! ```
//!
//! A record is rejected on read ([`LawStoreError::Corrupt`]) when the magic,
//! format or checksum is wrong, it is truncated or has trailing bytes, or the
//! name / version / sequence number inside it differs from its key (a record
//! copied under another key). A record that decodes but describes an invalid
//! law is rejected by [`SignalLaw::from_parts`] ([`LawStoreError::Law`]), for
//! example evidence outside the stored domain ([`LawError::OutOfRange`]) or a
//! non-finite coefficient ([`LawError::NonFinite`]).
//!
//! # Write order
//!
//! [`AliceDB::ingest_evidence`] writes the new version (if any) before the
//! verdict, so a recorded [`VerdictRecord::new_version`] always refers to a
//! stored version. Writes are serialised per database by an internal lock.
//!
//! ```
//! use alice_db::law_store::{IngestPolicy, Provenance, SignalLaw, Verdict};
//! use alice_db::{AliceDB, StorageConfig};
//!
//! // Identifies the arithmetic the law is evaluated with. It goes into the
//! // law's content identifier, so a stored result names both the law and the
//! // numeric semantics it was computed under.
//! const SEMANTICS: [u8; 32] = [0x11; 32];
//!
//! let db = AliceDB::in_memory(StorageConfig::default())?;
//! // y = 1 + 2x measured at x = 0..=4
//! let pts: Vec<(f64, f64)> = (0..5).map(|i| (f64::from(i), 1.0 + 2.0 * f64::from(i))).collect();
//! let law = SignalLaw::fit_polynomial(&pts, 1, Provenance::new("run 1", "least squares"))?;
//! db.put_law("line", &law, &SEMANTICS)?;
//!
//! assert!((db.evaluate_law("line", 2.5)? - 6.0).abs() < 1e-12);
//! assert!(db.evaluate_law("line", 9.0).is_err()); // outside [0, 4]
//!
//! // The stored law is reachable by its content identifier as well as by name.
//! let id = law.law_id(&SEMANTICS);
//! assert_eq!(db.law_pointers_by_id(&id)?, vec![("line".to_string(), 1_u64)]);
//!
//! let policy = IngestPolicy { abs_tolerance: 0.01, break_factor: 4.0 };
//! let v = db.ingest_evidence("line", &[(0.5, 2.0), (3.5, 8.0)], &policy, &SEMANTICS)?;
//! assert!(matches!(v, Verdict::Supports { .. }));
//! assert_eq!(db.law_history("line")?.len(), 1);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::fmt;
use std::io;

pub use alice_core::law::{
    IngestPolicy, LawError, OracleCase, OracleOutcome, Provenance, ResidualStats, SignalLaw,
    SignalLawParts, ValidRange, Verdict,
};

use crate::AliceDB;

/// Leading bytes of every key written by this module.
pub const LAW_KEY_PREFIX: &[u8] = b"\0alice-law\0";

/// Leading bytes of a law record.
pub const LAW_RECORD_MAGIC: [u8; 8] = *b"ALAWREC\x01";

/// Leading bytes of a verdict record.
pub const VERDICT_RECORD_MAGIC: [u8; 8] = *b"ALAWVRD\x01";

/// Layout version of both record kinds written by this build.
pub const LAW_RECORD_FORMAT: u32 = 1;

const VERSION_TAG: &[u8] = b"\0v";
const HISTORY_TAG: &[u8] = b"\0h";
const ID_TAG: &[u8] = b"\0i";

/// Why a law store operation failed
#[derive(Debug)]
#[non_exhaustive]
pub enum LawStoreError {
    /// The underlying storage failed
    Io(io::Error),
    /// The name is empty or contains a NUL byte
    InvalidName,
    /// The policy has a non-finite or negative `abs_tolerance`, or a
    /// `break_factor` that is not finite or below 1
    InvalidPolicy,
    /// No law is stored under the name
    NotFound,
    /// A stored record is damaged or does not belong under its key
    Corrupt(&'static str),
    /// A stored record decodes but [`SignalLaw::from_parts`] refuses it,
    /// or a law operation refused its input
    Law(LawError),
}

impl fmt::Display for LawStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "law store I/O: {e}"),
            Self::InvalidName => f.write_str("law name is empty or contains a NUL byte"),
            Self::InvalidPolicy => f.write_str(
                "ingest policy needs a finite abs_tolerance >= 0 and a finite break_factor >= 1",
            ),
            Self::NotFound => f.write_str("no law stored under this name"),
            Self::Corrupt(why) => write!(f, "stored law record is corrupt: {why}"),
            Self::Law(e) => write!(f, "law: {e}"),
        }
    }
}

impl std::error::Error for LawStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Law(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for LawStoreError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<LawError> for LawStoreError {
    fn from(e: LawError) -> Self {
        Self::Law(e)
    }
}

/// The kind of a recorded [`Verdict`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VerdictKind {
    /// [`Verdict::NoEvidence`]
    NoEvidence,
    /// [`Verdict::OutOfRange`]
    OutOfRange,
    /// [`Verdict::Supports`]
    Supports,
    /// [`Verdict::ParameterUpdate`]
    ParameterUpdate,
    /// [`Verdict::ResidualGrew`]
    ResidualGrew,
    /// [`Verdict::Breaks`]
    Breaks,
}

impl VerdictKind {
    const fn code(self) -> u8 {
        match self {
            Self::NoEvidence => 0,
            Self::OutOfRange => 1,
            Self::Supports => 2,
            Self::ParameterUpdate => 3,
            Self::ResidualGrew => 4,
            Self::Breaks => 5,
        }
    }

    const fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0 => Self::NoEvidence,
            1 => Self::OutOfRange,
            2 => Self::Supports,
            3 => Self::ParameterUpdate,
            4 => Self::ResidualGrew,
            5 => Self::Breaks,
            _ => return None,
        })
    }
}

/// One entry of a law's verdict history ([`AliceDB::law_history`])
#[derive(Debug, Clone, PartialEq)]
pub struct VerdictRecord {
    /// Position in the history, from 1, in the order the evidence was ingested
    pub seq: u64,
    /// Version of the law the evidence was judged against
    pub law_version: u64,
    /// Number of points in the evidence
    pub evidence_count: u64,
    /// What the evidence did to the law
    pub outcome: VerdictKind,
    /// Points outside the valid range (or not finite); 0 unless `OutOfRange`
    pub outside: u64,
    /// RMS of the new points about the judged law (`None` for `NoEvidence` /
    /// `OutOfRange`; for `ParameterUpdate` the RMS before the update)
    pub rms: Option<f64>,
    /// Version stored by a `ParameterUpdate`
    pub new_version: Option<u64>,
}

// ---------------------------------------------------------------- keys

fn check_name(name: &str) -> Result<(), LawStoreError> {
    if name.is_empty() || name.as_bytes().contains(&0) {
        return Err(LawStoreError::InvalidName);
    }
    Ok(())
}

fn name_prefix(name: &str, tag: &[u8]) -> Vec<u8> {
    let mut k = Vec::with_capacity(LAW_KEY_PREFIX.len() + name.len() + tag.len() + 8);
    k.extend_from_slice(LAW_KEY_PREFIX);
    k.extend_from_slice(name.as_bytes());
    k.extend_from_slice(tag);
    k
}

fn numbered_key(name: &str, tag: &[u8], n: u64) -> Vec<u8> {
    let mut k = name_prefix(name, tag);
    k.extend_from_slice(&n.to_be_bytes());
    k
}

/// Blob key of version `version` of the law `name` (see the module docs)
///
/// # Errors
/// [`LawStoreError::InvalidName`] for an empty name or one with a NUL byte.
pub fn law_version_key(name: &str, version: u64) -> Result<Vec<u8>, LawStoreError> {
    check_name(name)?;
    Ok(numbered_key(name, VERSION_TAG, version))
}

/// Blob key of verdict `seq` of the law `name` (see the module docs)
///
/// # Errors
/// [`LawStoreError::InvalidName`] for an empty name or one with a NUL byte.
pub fn law_history_key(name: &str, seq: u64) -> Result<Vec<u8>, LawStoreError> {
    check_name(name)?;
    Ok(numbered_key(name, HISTORY_TAG, seq))
}

/// Blob key that addresses the law `name` version `version` by the content
/// identifier `law_id` (see the module docs)
///
/// The pointer lives in the key, not the value: one identifier can legitimately
/// address several `(name, version)` pairs, and a prefix scan then returns all
/// of them without the record format having to hold a list.
///
/// A name is non-empty and holds no NUL byte, so the byte after
/// [`LAW_KEY_PREFIX`] is never NUL for a name key and always NUL here: the two
/// key families can never be a prefix of one another. The name carries its
/// length so that `("ab", v)` and `("a", …)` cannot encode to the same bytes.
///
/// # Errors
/// [`LawStoreError::InvalidName`] for an empty name or one with a NUL byte.
pub fn law_id_key(law_id: &[u8; 32], name: &str, version: u64) -> Result<Vec<u8>, LawStoreError> {
    check_name(name)?;
    let name_len = u32::try_from(name.len()).map_err(|_| LawStoreError::InvalidName)?;
    let mut k = Vec::with_capacity(LAW_KEY_PREFIX.len() + ID_TAG.len() + 32 + 4 + name.len() + 8);
    k.extend_from_slice(LAW_KEY_PREFIX);
    k.extend_from_slice(ID_TAG);
    k.extend_from_slice(law_id);
    k.extend_from_slice(&name_len.to_be_bytes());
    k.extend_from_slice(name.as_bytes());
    k.extend_from_slice(&version.to_be_bytes());
    Ok(k)
}

/// Leading bytes shared by every key of one content identifier
fn law_id_prefix(law_id: &[u8; 32]) -> Vec<u8> {
    let mut k = Vec::with_capacity(LAW_KEY_PREFIX.len() + ID_TAG.len() + 32);
    k.extend_from_slice(LAW_KEY_PREFIX);
    k.extend_from_slice(ID_TAG);
    k.extend_from_slice(law_id);
    k
}

/// The `(name, version)` a key from [`law_id_key`] points at
fn law_id_pointer(key: &[u8], prefix: &[u8]) -> Result<(String, u64), LawStoreError> {
    let rest = key
        .strip_prefix(prefix)
        .ok_or(LawStoreError::Corrupt("identifier key outside its prefix"))?;
    let (len_bytes, rest) = rest
        .split_at_checked(4)
        .ok_or(LawStoreError::Corrupt("identifier key has no name length"))?;
    let name_len = u32::from_be_bytes(
        len_bytes
            .try_into()
            .map_err(|_| LawStoreError::Corrupt("identifier key name length"))?,
    ) as usize;
    let (name_bytes, version_bytes) = rest
        .split_at_checked(name_len)
        .ok_or(LawStoreError::Corrupt("identifier key name is truncated"))?;
    let name = core::str::from_utf8(name_bytes)
        .map_err(|_| LawStoreError::Corrupt("identifier key name is not UTF-8"))?;
    let version = u64::from_be_bytes(
        version_bytes
            .try_into()
            .map_err(|_| LawStoreError::Corrupt("identifier key does not end in a u64"))?,
    );
    Ok((String::from(name), version))
}

/// The number at the end of a key from [`numbered_key`] with this prefix
fn key_number(key: &[u8], prefix: &[u8]) -> Result<u64, LawStoreError> {
    let rest = key
        .strip_prefix(prefix)
        .ok_or(LawStoreError::Corrupt("key outside its prefix"))?;
    let bytes: [u8; 8] = rest
        .try_into()
        .map_err(|_| LawStoreError::Corrupt("key does not end in a u64"))?;
    Ok(u64::from_be_bytes(bytes))
}

// ---------------------------------------------------------------- encoding

struct Writer(Vec<u8>);

impl Writer {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.u64(v.to_bits());
    }
    fn len32(&mut self, len: usize) -> Result<(), LawStoreError> {
        let n = u32::try_from(len).map_err(|_| {
            LawStoreError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "law field longer than u32::MAX",
            ))
        })?;
        self.u32(n);
        Ok(())
    }
    fn str(&mut self, s: &str) -> Result<(), LawStoreError> {
        self.len32(s.len())?;
        self.0.extend_from_slice(s.as_bytes());
        Ok(())
    }
    fn finish(mut self) -> Vec<u8> {
        let crc = crc32fast::hash(&self.0);
        self.u32(crc);
        self.0
    }
}

struct Reader<'a>(&'a [u8]);

const TRUNCATED: LawStoreError = LawStoreError::Corrupt("truncated record");

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], LawStoreError> {
        if self.0.len() < n {
            return Err(TRUNCATED);
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(head)
    }
    fn u8(&mut self) -> Result<u8, LawStoreError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, LawStoreError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64, LawStoreError> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }
    fn f64(&mut self) -> Result<f64, LawStoreError> {
        Ok(f64::from_bits(self.u64()?))
    }
    /// A count whose items take at least `item_bytes` each: refuses counts the
    /// remaining bytes cannot hold, before `Vec::with_capacity` allocates them
    fn count(&self, n: u64, item_bytes: usize) -> Result<usize, LawStoreError> {
        let n = usize::try_from(n).map_err(|_| TRUNCATED)?;
        if n.checked_mul(item_bytes)
            .is_none_or(|need| need > self.0.len())
        {
            return Err(TRUNCATED);
        }
        Ok(n)
    }
    fn str(&mut self) -> Result<String, LawStoreError> {
        let n = self.u32()?;
        let n = self.count(u64::from(n), 1)?;
        let b = self.take(n)?;
        String::from_utf8(b.to_vec()).map_err(|_| LawStoreError::Corrupt("text is not UTF-8"))
    }
    fn flag(&mut self) -> Result<bool, LawStoreError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(LawStoreError::Corrupt("invalid presence flag")),
        }
    }
}

/// Checks magic, format and checksum and returns the body between the header
/// and the checksum
fn open_record(bytes: &[u8], magic: [u8; 8]) -> Result<&[u8], LawStoreError> {
    if bytes.len() < magic.len() + 4 + 4 {
        return Err(TRUNCATED);
    }
    let (data, crc) = bytes.split_at(bytes.len() - 4);
    if !data.starts_with(&magic) {
        return Err(LawStoreError::Corrupt("wrong magic"));
    }
    let stored = u32::from_le_bytes([crc[0], crc[1], crc[2], crc[3]]);
    if crc32fast::hash(data) != stored {
        return Err(LawStoreError::Corrupt("checksum mismatch"));
    }
    let mut r = Reader(&data[magic.len()..]);
    if r.u32()? != LAW_RECORD_FORMAT {
        return Err(LawStoreError::Corrupt("unknown record format"));
    }
    Ok(r.0)
}

/// Encodes a law record (layout in the module docs)
///
/// # Errors
/// [`LawStoreError::InvalidName`] for an invalid name; [`LawStoreError::Io`]
/// (`InvalidInput`) when a count or text does not fit its length field.
pub fn encode_law_record(
    name: &str,
    version: u64,
    parts: &SignalLawParts,
) -> Result<Vec<u8>, LawStoreError> {
    check_name(name)?;
    let mut w = Writer(Vec::new());
    w.0.extend_from_slice(&LAW_RECORD_MAGIC);
    w.u32(LAW_RECORD_FORMAT);
    w.str(name)?;
    w.u64(version);
    w.len32(parts.coefficients.len())?;
    for &c in &parts.coefficients {
        w.f64(c);
    }
    w.f64(parts.domain.lo);
    w.f64(parts.domain.hi);
    w.u64(parts.evidence.len() as u64);
    for &(x, y) in &parts.evidence {
        w.f64(x);
        w.f64(y);
    }
    w.u64(parts.residual.n as u64);
    w.f64(parts.residual.rms);
    w.f64(parts.residual.max_abs);
    w.str(&parts.provenance.source)?;
    w.str(&parts.provenance.method)?;
    w.len32(parts.oracles.len())?;
    for o in &parts.oracles {
        w.f64(o.x);
        w.f64(o.expected);
        w.f64(o.tolerance);
        w.str(&o.source)?;
    }
    Ok(w.finish())
}

/// Decodes a law record into `(name, version, parts)` without validating the
/// law (that is [`SignalLaw::from_parts`])
///
/// # Errors
/// [`LawStoreError::Corrupt`] for a wrong magic / format / checksum, a
/// truncated record, trailing bytes, invalid UTF-8 or an invalid name.
pub fn decode_law_record(bytes: &[u8]) -> Result<(String, u64, SignalLawParts), LawStoreError> {
    let mut r = Reader(open_record(bytes, LAW_RECORD_MAGIC)?);
    let name = r.str()?;
    if check_name(&name).is_err() {
        return Err(LawStoreError::Corrupt("invalid name in record"));
    }
    let version = r.u64()?;
    let nc = r.u32()?;
    let nc = r.count(u64::from(nc), 8)?;
    let mut coefficients = Vec::with_capacity(nc);
    for _ in 0..nc {
        coefficients.push(r.f64()?);
    }
    let domain = ValidRange {
        lo: r.f64()?,
        hi: r.f64()?,
    };
    let ne = r.u64()?;
    let ne = r.count(ne, 16)?;
    let mut evidence = Vec::with_capacity(ne);
    for _ in 0..ne {
        evidence.push((r.f64()?, r.f64()?));
    }
    let n = r.u64()?;
    let residual = ResidualStats {
        n: usize::try_from(n).map_err(|_| LawStoreError::Corrupt("residual count overflows"))?,
        rms: r.f64()?,
        max_abs: r.f64()?,
    };
    let source = r.str()?;
    let method = r.str()?;
    let no = r.u32()?;
    let no = r.count(u64::from(no), 28)?;
    let mut oracles = Vec::with_capacity(no);
    for _ in 0..no {
        let x = r.f64()?;
        let expected = r.f64()?;
        let tolerance = r.f64()?;
        let source = r.str()?;
        oracles.push(OracleCase {
            x,
            expected,
            tolerance,
            source,
        });
    }
    if !r.0.is_empty() {
        return Err(LawStoreError::Corrupt("trailing bytes"));
    }
    let parts = SignalLawParts {
        coefficients,
        domain,
        evidence,
        residual,
        provenance: Provenance { source, method },
        oracles,
    };
    Ok((name, version, parts))
}

fn encode_verdict_record(name: &str, r: &VerdictRecord) -> Result<Vec<u8>, LawStoreError> {
    let mut w = Writer(Vec::new());
    w.0.extend_from_slice(&VERDICT_RECORD_MAGIC);
    w.u32(LAW_RECORD_FORMAT);
    w.str(name)?;
    w.u64(r.seq);
    w.u64(r.law_version);
    w.u64(r.evidence_count);
    w.u8(r.outcome.code());
    w.u64(r.outside);
    w.u8(u8::from(r.rms.is_some()));
    w.f64(r.rms.unwrap_or(0.0));
    w.u8(u8::from(r.new_version.is_some()));
    w.u64(r.new_version.unwrap_or(0));
    Ok(w.finish())
}

fn decode_verdict_record(bytes: &[u8]) -> Result<(String, VerdictRecord), LawStoreError> {
    let mut r = Reader(open_record(bytes, VERDICT_RECORD_MAGIC)?);
    let name = r.str()?;
    let seq = r.u64()?;
    let law_version = r.u64()?;
    let evidence_count = r.u64()?;
    let outcome =
        VerdictKind::from_code(r.u8()?).ok_or(LawStoreError::Corrupt("unknown verdict code"))?;
    let outside = r.u64()?;
    let has_rms = r.flag()?;
    let rms = r.f64()?;
    let has_new = r.flag()?;
    let new_version = r.u64()?;
    if !r.0.is_empty() {
        return Err(LawStoreError::Corrupt("trailing bytes"));
    }
    Ok((
        name,
        VerdictRecord {
            seq,
            law_version,
            evidence_count,
            outcome,
            outside,
            rms: has_rms.then_some(rms),
            new_version: has_new.then_some(new_version),
        },
    ))
}

fn check_policy(p: &IngestPolicy) -> Result<(), LawStoreError> {
    let tol_ok = p.abs_tolerance.is_finite() && p.abs_tolerance >= 0.0;
    let bf_ok = p.break_factor.is_finite() && p.break_factor >= 1.0;
    if tol_ok && bf_ok {
        Ok(())
    } else {
        Err(LawStoreError::InvalidPolicy)
    }
}

fn record_for(verdict: &Verdict, law_version: u64, count: usize) -> VerdictRecord {
    let (outcome, outside, rms) = match verdict {
        Verdict::NoEvidence => (VerdictKind::NoEvidence, 0, None),
        Verdict::OutOfRange { outside } => (VerdictKind::OutOfRange, *outside as u64, None),
        Verdict::Supports { rms } => (VerdictKind::Supports, 0, Some(*rms)),
        Verdict::ParameterUpdate { previous_rms, .. } => {
            (VerdictKind::ParameterUpdate, 0, Some(*previous_rms))
        }
        Verdict::ResidualGrew { rms } => (VerdictKind::ResidualGrew, 0, Some(*rms)),
        Verdict::Breaks { rms } => (VerdictKind::Breaks, 0, Some(*rms)),
    };
    VerdictRecord {
        seq: 0,
        law_version,
        evidence_count: count as u64,
        outcome,
        outside,
        rms,
        new_version: None,
    }
}

// ---------------------------------------------------------------- AliceDB API

impl AliceDB {
    /// Stored version numbers of `name`, ascending
    fn version_numbers(&self, name: &str) -> Result<Vec<u64>, LawStoreError> {
        let prefix = name_prefix(name, VERSION_TAG);
        self.blob
            .scan_prefix(&prefix)?
            .iter()
            .map(|(k, _)| key_number(k, &prefix))
            .collect()
    }

    fn last_number(&self, name: &str, tag: &[u8]) -> Result<u64, LawStoreError> {
        let prefix = name_prefix(name, tag);
        match self.blob.scan_prefix(&prefix)?.last() {
            Some((k, _)) => key_number(k, &prefix),
            None => Ok(0),
        }
    }

    fn load_version(&self, name: &str, version: u64) -> Result<Option<SignalLaw>, LawStoreError> {
        let Some(bytes) = self.blob.get(&numbered_key(name, VERSION_TAG, version))? else {
            return Ok(None);
        };
        let (stored_name, stored_version, parts) = decode_law_record(&bytes)?;
        if stored_name != name || stored_version != version {
            return Err(LawStoreError::Corrupt("record stored under another key"));
        }
        Ok(Some(SignalLaw::from_parts(parts)?))
    }

    /// Writes a version and the key that addresses it by content
    ///
    /// Every path that stores a version goes through here, so the identifier
    /// index cannot be complete for one entry point and missing for another.
    fn store_version(
        &self,
        name: &str,
        version: u64,
        law: &SignalLaw,
        semantics_id: &[u8; 32],
    ) -> Result<(), LawStoreError> {
        let rec = encode_law_record(name, version, &law.to_parts())?;
        self.blob
            .put(&numbered_key(name, VERSION_TAG, version), &rec)?;
        // The pointer is entirely in the key, so the value carries nothing.
        self.blob
            .put(&law_id_key(&law.law_id(semantics_id), name, version)?, &[])?;
        Ok(())
    }

    /// Stores `law` as the next version of `name` (1 for a new name) and
    /// returns that version. Earlier versions are kept.
    ///
    /// # Errors
    /// [`LawStoreError::InvalidName`], [`LawStoreError::Io`], or
    /// [`LawStoreError::Corrupt`] if the existing keys of `name` are damaged.
    pub fn put_law(
        &self,
        name: &str,
        law: &SignalLaw,
        semantics_id: &[u8; 32],
    ) -> Result<u64, LawStoreError> {
        check_name(name)?;
        let _guard = self.law_lock.lock();
        let version = self.last_number(name, VERSION_TAG)? + 1;
        self.store_version(name, version, law, semantics_id)?;
        Ok(version)
    }

    /// The latest version of `name`, restored through [`SignalLaw::from_parts`]
    /// (`Ok(None)` if no law is stored under it)
    ///
    /// # Errors
    /// [`LawStoreError::InvalidName`], [`LawStoreError::Io`],
    /// [`LawStoreError::Corrupt`] for a damaged record, and
    /// [`LawStoreError::Law`] for a record [`SignalLaw::from_parts`] refuses.
    pub fn get_law(&self, name: &str) -> Result<Option<SignalLaw>, LawStoreError> {
        check_name(name)?;
        match self.last_number(name, VERSION_TAG)? {
            0 => Ok(None),
            v => self.load_version(name, v),
        }
    }

    /// Version `version` of `name` (`Ok(None)` if it is not stored)
    ///
    /// # Errors
    /// As [`Self::get_law`].
    pub fn get_law_version(
        &self,
        name: &str,
        version: u64,
    ) -> Result<Option<SignalLaw>, LawStoreError> {
        check_name(name)?;
        self.load_version(name, version)
    }

    /// `f(x)` of the latest version of `name`
    ///
    /// # Errors
    /// [`LawStoreError::Law`] carrying [`LawError::OutOfRange`] for an `x` outside
    /// the law's valid range or not finite (no extrapolation);
    /// [`LawStoreError::NotFound`] if no law is stored; otherwise as
    /// [`Self::get_law`].
    pub fn evaluate_law(&self, name: &str, x: f64) -> Result<f64, LawStoreError> {
        let law = self.get_law(name)?.ok_or(LawStoreError::NotFound)?;
        Ok(law.evaluate(x)?)
    }

    /// Every `(name, version)` stored under the content identifier `law_id`
    ///
    /// Empty when nothing was stored under it: an identifier that addresses no
    /// law is absence, not damage. An identifier is computed from the law
    /// together with the numeric semantics it is evaluated under, so a law
    /// stored under one `semantics_id` does not answer for another.
    ///
    /// The order is the byte order of the keys, which is the name length, then
    /// the name, then the version.
    ///
    /// # Errors
    /// [`LawStoreError::Io`], and [`LawStoreError::Corrupt`] if a key under the
    /// identifier prefix is damaged.
    pub fn law_pointers_by_id(
        &self,
        law_id: &[u8; 32],
    ) -> Result<Vec<(String, u64)>, LawStoreError> {
        let prefix = law_id_prefix(law_id);
        self.blob
            .scan_prefix(&prefix)?
            .into_iter()
            .map(|(k, _)| law_id_pointer(&k, &prefix))
            .collect()
    }

    /// `f(x)` of the law addressed by `law_id`
    ///
    /// Well defined even when the identifier addresses several
    /// `(name, version)` pairs: an identifier is a hash over exactly the inputs
    /// the evaluation reads, so every law it addresses returns the same bits for
    /// the same `x`. The first pointer in key order is used.
    ///
    /// An `x` outside the law's valid range is refused, never extrapolated.
    ///
    /// # Errors
    /// [`LawStoreError::NotFound`] when no law is stored under the identifier
    /// (an identifier cannot be turned into a value out of nothing),
    /// [`LawStoreError::Io`], [`LawStoreError::Corrupt`], and
    /// [`LawStoreError::Law`] for an `x` outside the valid range.
    pub fn evaluate_by_id(&self, law_id: &[u8; 32], x: f64) -> Result<f64, LawStoreError> {
        let (name, version) = self
            .law_pointers_by_id(law_id)?
            .into_iter()
            .next()
            .ok_or(LawStoreError::NotFound)?;
        let law = self
            .load_version(&name, version)?
            .ok_or(LawStoreError::NotFound)?;
        Ok(law.evaluate(x)?)
    }

    /// Judges `points` against the latest version of `name` and records the
    /// verdict in the law's history
    ///
    /// On [`Verdict::ParameterUpdate`] the refitted law is stored as the next
    /// version (before the verdict is recorded); the judged version stays
    /// readable through [`Self::get_law_version`]. The rules are those of
    /// [`SignalLaw::ingest`].
    ///
    /// # Errors
    /// [`LawStoreError::InvalidPolicy`] (nothing is recorded),
    /// [`LawStoreError::NotFound`], otherwise as [`Self::get_law`].
    pub fn ingest_evidence(
        &self,
        name: &str,
        points: &[(f64, f64)],
        policy: &IngestPolicy,
        semantics_id: &[u8; 32],
    ) -> Result<Verdict, LawStoreError> {
        check_name(name)?;
        check_policy(policy)?;
        let _guard = self.law_lock.lock();
        let version = self.last_number(name, VERSION_TAG)?;
        if version == 0 {
            return Err(LawStoreError::NotFound);
        }
        let law = self
            .load_version(name, version)?
            .ok_or(LawStoreError::NotFound)?;
        let verdict = law.ingest(points, policy);
        let mut record = record_for(&verdict, version, points.len());
        if let Verdict::ParameterUpdate { updated, .. } = &verdict {
            self.store_version(name, version + 1, updated, semantics_id)?;
            record.new_version = Some(version + 1);
        }
        record.seq = self.last_number(name, HISTORY_TAG)? + 1;
        let bytes = encode_verdict_record(name, &record)?;
        self.blob
            .put(&numbered_key(name, HISTORY_TAG, record.seq), &bytes)?;
        Ok(verdict)
    }

    /// Recorded verdicts of `name` in ingestion order (empty if none)
    ///
    /// # Errors
    /// [`LawStoreError::InvalidName`], [`LawStoreError::Io`], or
    /// [`LawStoreError::Corrupt`] for a damaged record or one stored under
    /// another key.
    pub fn law_history(&self, name: &str) -> Result<Vec<VerdictRecord>, LawStoreError> {
        check_name(name)?;
        let prefix = name_prefix(name, HISTORY_TAG);
        self.blob
            .scan_prefix(&prefix)?
            .iter()
            .map(|(k, v)| {
                let seq = key_number(k, &prefix)?;
                let (stored_name, rec) = decode_verdict_record(v)?;
                if stored_name != name || rec.seq != seq {
                    return Err(LawStoreError::Corrupt("record stored under another key"));
                }
                Ok(rec)
            })
            .collect()
    }

    /// Stored version numbers of `name`, ascending (empty if none)
    ///
    /// # Errors
    /// [`LawStoreError::InvalidName`], [`LawStoreError::Io`], or
    /// [`LawStoreError::Corrupt`] for a damaged key.
    pub fn law_versions(&self, name: &str) -> Result<Vec<u64>, LawStoreError> {
        check_name(name)?;
        self.version_numbers(name)
    }

    /// Names with at least one stored version, in byte order
    ///
    /// # Errors
    /// [`LawStoreError::Io`], or [`LawStoreError::Corrupt`] for a key in the
    /// reserved prefix that this module did not write.
    pub fn law_names(&self) -> Result<Vec<String>, LawStoreError> {
        let mut names: Vec<String> = Vec::new();
        for (k, _) in self.blob.scan_prefix(LAW_KEY_PREFIX)? {
            let rest = &k[LAW_KEY_PREFIX.len()..];
            let end = rest
                .iter()
                .position(|&b| b == 0)
                .ok_or(LawStoreError::Corrupt("key without a tag"))?;
            if !rest[end..].starts_with(VERSION_TAG) {
                continue;
            }
            let name = std::str::from_utf8(&rest[..end])
                .map_err(|_| LawStoreError::Corrupt("name is not UTF-8"))?;
            if names.last().map(String::as_str) != Some(name) {
                names.push(name.to_string());
            }
        }
        Ok(names)
    }
}
