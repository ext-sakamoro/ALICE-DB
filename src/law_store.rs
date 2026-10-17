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
//! format         u32      LAW_RECORD_FORMAT (= 2)
//! name           u32 length + UTF-8
//! version        u64
//! semantics      32 bytes the identifier the law was written under (format 2 only)
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
//! format         u32      VERDICT_RECORD_FORMAT (= 1)
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
//! # Semantics identifier
//!
//! A law record (format 2) stores the identifier it was written under (the
//! `semantics_id` given to [`AliceDB::put_law`] / [`AliceDB::ingest_evidence`],
//! normally [`crate::SEMANTICS_ID`]). Every read compares it with the identifier
//! this build evaluates with: the same one, or an earlier one
//! `alice_det_math::check_semantics` accepts, is read and reported as
//! [`RecordSemantics::Current`] / [`RecordSemantics::VerifiedPrevious`]; any
//! other is refused with [`LawStoreError::LawRecordMismatch`], which names
//! both. A format 1 record carries none: it is read as
//! [`RecordSemantics::Unpinned`] under [`LawReadPolicy::AcceptUnpinned`] (the
//! default of [`AliceDB::get_law`] and the other reads) and refused with
//! [`LawStoreError::Unpinned`] under [`LawReadPolicy::RequirePinned`]. The
//! status comes back with every law and value ([`StoredLaw`], [`LawValue`]),
//! so a caller always sees which of the three it got. Bytes carried by
//! [`AliceDB::to_bytes`] / [`AliceDB::from_bytes`] are checked the same way
//! when a law is read from the restored store.
//!
//! # Write order
//!
//! [`AliceDB::ingest_evidence`] writes the new version (if any) before the
//! verdict, so a recorded [`VerdictRecord::new_version`] always refers to a
//! stored version. Writes are serialised per database by an internal lock.
//!
//! ```
//! use alice_db::law_store::{IngestPolicy, Provenance, RecordSemantics, SignalLaw, Verdict};
//! use alice_db::{AliceDB, StorageConfig};
//!
//! // Identifies the arithmetic the law is evaluated with. It goes into the
//! // law's content identifier and into the stored record, so a stored result
//! // names both the law and the numeric semantics it was computed under, and
//! // a read under other semantics is refused.
//! const SEMANTICS: [u8; 32] = alice_db::SEMANTICS_ID;
//!
//! let db = AliceDB::in_memory(StorageConfig::default())?;
//! // y = 1 + 2x measured at x = 0..=4
//! let pts: Vec<(f64, f64)> = (0..5).map(|i| (f64::from(i), 1.0 + 2.0 * f64::from(i))).collect();
//! let law = SignalLaw::fit_polynomial(&pts, 1, Provenance::new("run 1", "least squares"))?;
//! db.put_law("line", &law, &SEMANTICS)?;
//!
//! assert!((db.evaluate_law("line", 2.5)?.value - 6.0).abs() < 1e-12);
//! assert!(db.evaluate_law("line", 9.0).is_err()); // outside [0, 4]
//! // the record carries the identifier it was written under
//! assert_eq!(db.get_law("line")?.expect("stored").semantics, RecordSemantics::Current);
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

/// Layout version of the law records written by this build. Format 2 adds
/// the semantics identifier the law was written under; format 1 records
/// (without it) are still read, as [`RecordSemantics::Unpinned`].
pub const LAW_RECORD_FORMAT: u32 = 2;

/// Layout version of the verdict records written by this build.
pub const VERDICT_RECORD_FORMAT: u32 = 1;

/// The law record format that carries no semantics identifier.
const LAW_RECORD_FORMAT_UNPINNED: u32 = 1;

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
    /// The law record was written under a semantics identifier this build
    /// does not reproduce: neither the identifier it evaluates with
    /// ([`crate::SEMANTICS_ID`]) nor one `alice_det_math::check_semantics`
    /// accepts as an earlier identifier with the same bits
    LawRecordMismatch {
        /// The identifier the record carries
        stored: [u8; 32],
        /// The identifier this build evaluates with
        running: [u8; 32],
    },
    /// The law record carries no semantics identifier (layout format 1) and
    /// the read asked for [`LawReadPolicy::RequirePinned`]
    Unpinned,
}

fn hex32(id: &[u8; 32]) -> String {
    use fmt::Write as _;
    id.iter().fold(String::with_capacity(64), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// What a read knows about the arithmetic a stored law was written under
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RecordSemantics {
    /// The record carries the identifier this build evaluates with
    /// ([`crate::SEMANTICS_ID`])
    Current,
    /// The record carries an earlier identifier that
    /// `alice_det_math::check_semantics` accepts as computing the same bits
    VerifiedPrevious([u8; 32]),
    /// The record carries no identifier (layout format 1, written before the
    /// identifier was stored): nothing says which arithmetic it was written
    /// under. Accepted only by [`LawReadPolicy::AcceptUnpinned`]
    Unpinned,
}

/// Whether a read accepts a law record that carries no semantics identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LawReadPolicy {
    /// Read it and report [`RecordSemantics::Unpinned`] with the law
    #[default]
    AcceptUnpinned,
    /// Refuse it with [`LawStoreError::Unpinned`]
    RequirePinned,
}

/// A law read back from the store, with what is known about the arithmetic
/// it was written under
#[derive(Debug, Clone)]
pub struct StoredLaw {
    /// The law, restored through [`SignalLaw::from_parts`]
    pub law: SignalLaw,
    /// The semantics identifier check of its record
    pub semantics: RecordSemantics,
}

/// `f(x)` of a stored law, with what is known about the arithmetic the law
/// was written under
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LawValue {
    /// `f(x)`
    pub value: f64,
    /// The semantics identifier check of the law's record
    pub semantics: RecordSemantics,
}

/// Checks the identifier a record carries against the one this build
/// evaluates with
fn check_record_semantics(
    stored: Option<[u8; 32]>,
    policy: LawReadPolicy,
) -> Result<RecordSemantics, LawStoreError> {
    // the identifier `alice_det_math::check_semantics` below compares with (the
    // stamp is `crate::SEMANTICS_ID`, which lib.rs asserts is the same one)
    let running = alice_det_math::SEMANTICS_ID;
    let Some(stored) = stored else {
        return match policy {
            LawReadPolicy::AcceptUnpinned => Ok(RecordSemantics::Unpinned),
            LawReadPolicy::RequirePinned => Err(LawStoreError::Unpinned),
        };
    };
    if stored == running {
        return Ok(RecordSemantics::Current);
    }
    match alice_det_math::check_semantics(&stored) {
        alice_det_math::SemanticsCheck::Verified => Ok(RecordSemantics::Current),
        alice_det_math::SemanticsCheck::VerifiedPrevious(id) => {
            Ok(RecordSemantics::VerifiedPrevious(id))
        }
        alice_det_math::SemanticsCheck::Mismatch => {
            Err(LawStoreError::LawRecordMismatch { stored, running })
        }
    }
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
            Self::LawRecordMismatch { stored, running } => write!(
                f,
                "law record written under semantics {} which this build (semantics {}) does not reproduce",
                hex32(stored),
                hex32(running)
            ),
            Self::Unpinned => f.write_str(
                "law record carries no semantics identifier and the read requires one",
            ),
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

/// Which semantics identifier [`AliceDB::law_pointers_for`] found a law
/// stored under
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LawPointersFoundUnder {
    /// Found under the identifier the lookup was given as current.
    Current,
    /// Found under an older identifier [`alice_det_math::check_semantics`]
    /// still accepts as reproducing the same bits as the current one.
    Previous([u8; 32]),
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

/// Checks magic, format and checksum and returns the format and the body
/// between the header and the checksum
fn open_record<'a>(
    bytes: &'a [u8],
    magic: [u8; 8],
    formats: &[u32],
) -> Result<(u32, &'a [u8]), LawStoreError> {
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
    let format = r.u32()?;
    if !formats.contains(&format) {
        return Err(LawStoreError::Corrupt("unknown record format"));
    }
    Ok((format, r.0))
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
    semantics_id: &[u8; 32],
) -> Result<Vec<u8>, LawStoreError> {
    check_name(name)?;
    let mut w = Writer(Vec::new());
    w.0.extend_from_slice(&LAW_RECORD_MAGIC);
    w.u32(LAW_RECORD_FORMAT);
    w.str(name)?;
    w.u64(version);
    w.0.extend_from_slice(semantics_id);
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

/// A decoded law record: `(name, version, parts, semantics identifier)`; the
/// identifier is `None` for a format 1 record, which does not carry one
pub type DecodedLawRecord = (String, u64, SignalLawParts, Option<[u8; 32]>);

/// A restored law and the identifier its record carries, before the check
type UncheckedLaw = (SignalLaw, Option<[u8; 32]>);

/// Decodes a law record (format 2, or format 1 without the semantics
/// identifier) without validating the law (that is
/// [`SignalLaw::from_parts`]) or the identifier (that is the read paths of
/// [`AliceDB`])
///
/// # Errors
/// [`LawStoreError::Corrupt`] for a wrong magic / format / checksum, a
/// truncated record, trailing bytes, invalid UTF-8 or an invalid name.
pub fn decode_law_record(bytes: &[u8]) -> Result<DecodedLawRecord, LawStoreError> {
    let (format, body) = open_record(
        bytes,
        LAW_RECORD_MAGIC,
        &[LAW_RECORD_FORMAT, LAW_RECORD_FORMAT_UNPINNED],
    )?;
    let mut r = Reader(body);
    let name = r.str()?;
    if check_name(&name).is_err() {
        return Err(LawStoreError::Corrupt("invalid name in record"));
    }
    let version = r.u64()?;
    let semantics = if format == LAW_RECORD_FORMAT_UNPINNED {
        None
    } else {
        let mut id = [0u8; 32];
        id.copy_from_slice(r.take(32)?);
        Some(id)
    };
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
    Ok((name, version, parts, semantics))
}

fn encode_verdict_record(name: &str, r: &VerdictRecord) -> Result<Vec<u8>, LawStoreError> {
    let mut w = Writer(Vec::new());
    w.0.extend_from_slice(&VERDICT_RECORD_MAGIC);
    w.u32(VERDICT_RECORD_FORMAT);
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
    let (_, body) = open_record(bytes, VERDICT_RECORD_MAGIC, &[VERDICT_RECORD_FORMAT])?;
    let mut r = Reader(body);
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

    /// The law of a stored version and the identifier its record carries,
    /// without checking the identifier (for the index, which only computes
    /// keys from it)
    fn load_version_unchecked(
        &self,
        name: &str,
        version: u64,
    ) -> Result<Option<UncheckedLaw>, LawStoreError> {
        let Some(bytes) = self.blob.get(&numbered_key(name, VERSION_TAG, version))? else {
            return Ok(None);
        };
        let (stored_name, stored_version, parts, semantics) = decode_law_record(&bytes)?;
        if stored_name != name || stored_version != version {
            return Err(LawStoreError::Corrupt("record stored under another key"));
        }
        Ok(Some((SignalLaw::from_parts(parts)?, semantics)))
    }

    /// The law of a stored version, its record's identifier checked against
    /// this build's under `policy`
    fn load_version(
        &self,
        name: &str,
        version: u64,
        policy: LawReadPolicy,
    ) -> Result<Option<StoredLaw>, LawStoreError> {
        let Some((law, stored)) = self.load_version_unchecked(name, version)? else {
            return Ok(None);
        };
        let semantics = check_record_semantics(stored, policy)?;
        Ok(Some(StoredLaw { law, semantics }))
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
        let rec = encode_law_record(name, version, &law.to_parts(), semantics_id)?;
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
    /// (`Ok(None)` if no law is stored under it), with the check of the
    /// semantics identifier its record carries ([module docs](self)); a
    /// record without one is read as [`RecordSemantics::Unpinned`]
    ///
    /// # Errors
    /// [`LawStoreError::InvalidName`], [`LawStoreError::Io`],
    /// [`LawStoreError::Corrupt`] for a damaged record,
    /// [`LawStoreError::Law`] for a record [`SignalLaw::from_parts`] refuses,
    /// and [`LawStoreError::LawRecordMismatch`] for a record written under an
    /// identifier this build does not reproduce.
    pub fn get_law(&self, name: &str) -> Result<Option<StoredLaw>, LawStoreError> {
        self.get_law_with(name, LawReadPolicy::AcceptUnpinned)
    }

    /// [`Self::get_law`] under `policy`
    ///
    /// # Errors
    /// As [`Self::get_law`], and [`LawStoreError::Unpinned`] for a record
    /// without an identifier under [`LawReadPolicy::RequirePinned`].
    pub fn get_law_with(
        &self,
        name: &str,
        policy: LawReadPolicy,
    ) -> Result<Option<StoredLaw>, LawStoreError> {
        check_name(name)?;
        match self.last_number(name, VERSION_TAG)? {
            0 => Ok(None),
            v => self.load_version(name, v, policy),
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
    ) -> Result<Option<StoredLaw>, LawStoreError> {
        self.get_law_version_with(name, version, LawReadPolicy::AcceptUnpinned)
    }

    /// [`Self::get_law_version`] under `policy`
    ///
    /// # Errors
    /// As [`Self::get_law_with`].
    pub fn get_law_version_with(
        &self,
        name: &str,
        version: u64,
        policy: LawReadPolicy,
    ) -> Result<Option<StoredLaw>, LawStoreError> {
        check_name(name)?;
        self.load_version(name, version, policy)
    }

    /// `f(x)` of the latest version of `name`
    ///
    /// # Errors
    /// [`LawStoreError::Law`] carrying [`LawError::OutOfRange`] for an `x` outside
    /// the law's valid range or not finite (no extrapolation);
    /// [`LawStoreError::NotFound`] if no law is stored; otherwise as
    /// [`Self::get_law`].
    pub fn evaluate_law(&self, name: &str, x: f64) -> Result<LawValue, LawStoreError> {
        self.evaluate_law_with(name, x, LawReadPolicy::AcceptUnpinned)
    }

    /// [`Self::evaluate_law`] under `policy`
    ///
    /// # Errors
    /// As [`Self::evaluate_law`], and [`LawStoreError::Unpinned`] for a
    /// record without an identifier under [`LawReadPolicy::RequirePinned`].
    pub fn evaluate_law_with(
        &self,
        name: &str,
        x: f64,
        policy: LawReadPolicy,
    ) -> Result<LawValue, LawStoreError> {
        let stored = self
            .get_law_with(name, policy)?
            .ok_or(LawStoreError::NotFound)?;
        Ok(LawValue {
            value: stored.law.evaluate(x)?,
            semantics: stored.semantics,
        })
    }

    /// Every `(name, version)` stored under the content identifier `law_id`
    ///
    /// Empty when nothing was stored under it: an identifier that addresses no
    /// law is absence, not damage. An identifier is computed from the law
    /// together with the numeric semantics it is evaluated under.
    ///
    /// Writing always addresses a law under the identifier of the day it was
    /// written; a stored pointer never moves on its own. So a law written
    /// before the arithmetic's identifier changed (a dependency upgrade that
    /// [`alice_det_math::check_semantics`] still judges bit-reproducing) stays
    /// addressed only under the old identifier here — this method does not
    /// try any other one. To look a law up without knowing which identifier
    /// it was last touched under, use [`Self::law_pointers_for`], which tries
    /// the current identifier first and then every one `check_semantics`
    /// still accepts, and reports which one it found the law under.
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

    /// Every `(name, version)` stored under `law`'s content identifier,
    /// trying `semantics_id` first and, if that finds nothing, every
    /// identifier in [`alice_det_math::PREVIOUS_SEMANTICS_IDS`] in order
    ///
    /// A write never moves a law to a new identifier on its own
    /// ([`Self::law_pointers_by_id`]'s doc), so a law written under an
    /// identifier a later `alice-det-math` release superseded is otherwise
    /// unreachable without knowing that old identifier by name. This method
    /// is the read-side counterpart: [`alice_det_math::check_semantics`]
    /// already decides which identifiers are accepted as reproducing the same
    /// bits as the current one ([`alice_det_math::SemanticsCheck::VerifiedPrevious`]),
    /// and this tries exactly that set, in the same order, after the current
    /// one. The second element of the result says which identifier the first
    /// non-empty answer came from, so a caller that cares can tell a fresh
    /// write from an inherited one; **no pointer is written here** — a
    /// successful previous-identifier match does not make the law reachable
    /// under the current identifier until [`Self::reindex_laws_under_current_semantics`]
    /// is called.
    ///
    /// Returns `(vec![], `[`LawPointersFoundUnder::Current`]`)` when nothing is
    /// found under any identifier (the marker is meaningless in that case;
    /// only the empty list is).
    ///
    /// # Errors
    /// As [`Self::law_pointers_by_id`].
    pub fn law_pointers_for(
        &self,
        law: &SignalLaw,
        semantics_id: &[u8; 32],
    ) -> Result<(Vec<(String, u64)>, LawPointersFoundUnder), LawStoreError> {
        let current = self.law_pointers_by_id(&law.law_id(semantics_id))?;
        if !current.is_empty() {
            return Ok((current, LawPointersFoundUnder::Current));
        }
        for previous in alice_det_math::PREVIOUS_SEMANTICS_IDS {
            let pointers = self.law_pointers_by_id(&law.law_id(previous))?;
            if !pointers.is_empty() {
                return Ok((pointers, LawPointersFoundUnder::Previous(*previous)));
            }
        }
        Ok((Vec::new(), LawPointersFoundUnder::Current))
    }

    /// Writes a pointer under `semantics_id` for every stored law version this
    /// store still only addresses under an identifier
    /// [`alice_det_math::PREVIOUS_SEMANTICS_IDS`] accepts as reproducing the
    /// same bits, so later lookups find it under the current identifier
    /// directly instead of falling through [`Self::law_pointers_for`] every
    /// time
    ///
    /// Idempotent: a `(name, version)` already pointed at by `semantics_id`
    /// writes nothing on a repeat call, so running this on a schedule or
    /// after every upgrade is safe. It does not remove the previous pointer
    /// — nothing is ever deleted here, so [`Self::law_pointers_by_id`] keeps
    /// answering to the old identifier too — this only adds the new one.
    ///
    /// A `(name, version)` written under an identifier neither `semantics_id`
    /// nor any entry of [`alice_det_math::PREVIOUS_SEMANTICS_IDS`] accounts for (one
    /// [`alice_det_math::SemanticsCheck::Mismatch`] would reject) is left
    /// alone: this method only ever repeats a pointer the fallback in
    /// [`Self::law_pointers_for`] would already have accepted, it never
    /// manufactures a new one for a law whose original identifier carries no
    /// such guarantee.
    ///
    /// Returns the number of `(name, version)` pairs a pointer was newly
    /// written for.
    ///
    /// # Errors
    /// [`LawStoreError::Io`], and [`LawStoreError::Corrupt`] for a damaged
    /// stored version or identifier key.
    pub fn reindex_laws_under_current_semantics(
        &self,
        semantics_id: &[u8; 32],
    ) -> Result<usize, LawStoreError> {
        self.reindex_with(semantics_id, alice_det_math::PREVIOUS_SEMANTICS_IDS)
    }

    /// [`Self::reindex_laws_under_current_semantics`] against an explicit
    /// `accepted` list instead of reading
    /// [`alice_det_math::PREVIOUS_SEMANTICS_IDS`] directly
    ///
    /// Split out so a test can exercise the "found under a previous
    /// identifier, add a pointer" path with an injected `accepted` entry: the
    /// crate's own `PREVIOUS_SEMANTICS_IDS` is empty at the currently pinned
    /// `alice-det-math` release, so a committed test that only goes through
    /// the public method cannot tell a correct restriction from a stub that
    /// always returns 0 — both report nothing added, because nothing in
    /// `PREVIOUS_SEMANTICS_IDS` is there to be found.
    pub(crate) fn reindex_with(
        &self,
        semantics_id: &[u8; 32],
        accepted: &[[u8; 32]],
    ) -> Result<usize, LawStoreError> {
        let mut added = 0usize;
        for name in self.law_names()? {
            for version in self.law_versions(&name)? {
                let Some((law, _)) = self.load_version_unchecked(&name, version)? else {
                    continue;
                };
                let current_key = law_id_key(&law.law_id(semantics_id), &name, version)?;
                if self.blob.get(&current_key)?.is_some() {
                    continue;
                }
                let mut found_under_previous = false;
                for previous in accepted {
                    let key = law_id_key(&law.law_id(previous), &name, version)?;
                    if self.blob.get(&key)?.is_some() {
                        found_under_previous = true;
                        break;
                    }
                }
                if !found_under_previous {
                    continue;
                }
                self.blob.put(&current_key, &[])?;
                added += 1;
            }
        }
        Ok(added)
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
    /// [`LawStoreError::Law`] for an `x` outside the valid range, and
    /// [`LawStoreError::LawRecordMismatch`] as [`Self::get_law`].
    pub fn evaluate_by_id(&self, law_id: &[u8; 32], x: f64) -> Result<LawValue, LawStoreError> {
        self.evaluate_by_id_with(law_id, x, LawReadPolicy::AcceptUnpinned)
    }

    /// [`Self::evaluate_by_id`] under `policy`
    ///
    /// # Errors
    /// As [`Self::evaluate_by_id`], and [`LawStoreError::Unpinned`] for a
    /// record without an identifier under [`LawReadPolicy::RequirePinned`].
    pub fn evaluate_by_id_with(
        &self,
        law_id: &[u8; 32],
        x: f64,
        policy: LawReadPolicy,
    ) -> Result<LawValue, LawStoreError> {
        let (name, version) = self
            .law_pointers_by_id(law_id)?
            .into_iter()
            .next()
            .ok_or(LawStoreError::NotFound)?;
        let stored = self
            .load_version(&name, version, policy)?
            .ok_or(LawStoreError::NotFound)?;
        Ok(LawValue {
            value: stored.law.evaluate(x)?,
            semantics: stored.semantics,
        })
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
            .load_version(name, version, LawReadPolicy::AcceptUnpinned)?
            .ok_or(LawStoreError::NotFound)?
            .law;
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

#[cfg(test)]
mod tests {
    use super::{LawStoreError, Provenance, ResidualStats, SignalLaw, SignalLawParts, ValidRange};
    use crate::{AliceDB, StorageConfig};

    const SEMANTICS_CURRENT: [u8; 32] = [0xAA; 32];
    const SEMANTICS_ACCEPTED: [u8; 32] = [0xBB; 32];
    const SEMANTICS_NOT_ACCEPTED: [u8; 32] = [0xCC; 32];

    fn memory_db() -> AliceDB {
        AliceDB::in_memory(StorageConfig::default()).expect("in-memory database")
    }

    fn law_with(coefficients: Vec<f64>, source: &str) -> SignalLaw {
        SignalLaw::from_parts(SignalLawParts {
            coefficients,
            domain: ValidRange { lo: 0.0, hi: 4.0 },
            evidence: vec![(0.0, 1.0), (4.0, 2.0)],
            residual: ResidualStats {
                n: 0,
                rms: 0.0,
                max_abs: 0.0,
            },
            provenance: Provenance::new(source, "hand-written"),
            oracles: Vec::new(),
        })
        .expect("fixture satisfies from_parts")
    }

    // `reindex_with` takes `accepted` as an explicit argument instead of
    // reading `alice_det_math::PREVIOUS_SEMANTICS_IDS`, so these tests
    // exercise the "found under a previous identifier, add a pointer" path
    // with an injected entry -- the real PREVIOUS_SEMANTICS_IDS is empty at
    // the currently pinned alice-det-math release, so the public
    // `reindex_laws_under_current_semantics` cannot be used to distinguish a
    // correct restriction from a stub that always returns 0: both report
    // nothing added when nothing in the (empty) list is there to be found.
    #[test]
    fn reindex_with_adds_a_pointer_for_a_law_found_under_an_accepted_identifier() {
        let db = memory_db();
        let law = law_with(vec![1.0, 2.0], "run 1");
        let version = db
            .put_law("line", &law, &SEMANTICS_ACCEPTED)
            .expect("stored under the accepted id");

        let added = db
            .reindex_with(&SEMANTICS_CURRENT, &[SEMANTICS_ACCEPTED])
            .expect("reindex");
        assert_eq!(added, 1, "the accepted-identifier law must be reindexed");
        assert_eq!(
            db.law_pointers_by_id(&law.law_id(&SEMANTICS_CURRENT))
                .expect("after"),
            vec![("line".to_string(), version)],
            "now reachable under the current identifier"
        );
        assert_eq!(
            db.law_pointers_by_id(&law.law_id(&SEMANTICS_ACCEPTED))
                .expect("old pointer"),
            vec![("line".to_string(), version)],
            "the previous pointer is not removed"
        );
    }

    #[test]
    fn reindex_with_leaves_a_law_under_an_unaccepted_identifier_alone() {
        let db = memory_db();
        let law = law_with(vec![1.0, 2.0], "run 1");
        db.put_law("line", &law, &SEMANTICS_NOT_ACCEPTED)
            .expect("stored under an id the accepted list does not name");

        let added = db
            .reindex_with(&SEMANTICS_CURRENT, &[SEMANTICS_ACCEPTED])
            .expect("reindex");
        assert_eq!(
            added, 0,
            "SEMANTICS_NOT_ACCEPTED is not in the accepted list"
        );
        assert!(db
            .law_pointers_by_id(&law.law_id(&SEMANTICS_CURRENT))
            .expect("after")
            .is_empty());
    }

    #[test]
    fn reindex_with_is_idempotent_on_the_accepted_path() -> Result<(), LawStoreError> {
        let db = memory_db();
        db.put_law(
            "line",
            &law_with(vec![1.0, 2.0], "run 1"),
            &SEMANTICS_ACCEPTED,
        )?;

        let first = db.reindex_with(&SEMANTICS_CURRENT, &[SEMANTICS_ACCEPTED])?;
        assert_eq!(first, 1);
        let second = db.reindex_with(&SEMANTICS_CURRENT, &[SEMANTICS_ACCEPTED])?;
        assert_eq!(second, 0, "already reindexed, nothing new to add");
        Ok(())
    }
}
