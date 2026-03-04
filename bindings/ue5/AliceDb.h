/**
 * ALICE-DB C++ Bindings for Unreal Engine 5
 *
 * Usage:
 * 1. Copy this file to your UE5 project's Source/ThirdParty/AliceDb/include/
 * 2. Copy libalice_db.dylib (macOS), alice_db.dll (Windows), or libalice_db.so (Linux)
 *    to Source/ThirdParty/AliceDb/lib/
 * 3. Add library path to your Build.cs
 * 4. Include this header and use the AliceDb namespace
 *
 * Example:
 *   AliceDb::FDatabase Db("/path/to/data");
 *   Db.Put(100, 42.0f);
 *   auto Point = alice_db_get(Db.Get(), 100);
 *   if (Point.Found) UE_LOG(LogTemp, Log, TEXT("Value: %f"), Point.Value);
 *   Db.Flush();
 *
 * Author: Moroya Sakamoto
 */

#pragma once

#include <cstdint>

// ============================================================================
// Types
// ============================================================================

/// Opaque handle to an ALICE-DB instance
typedef void* DbHandle;

/// Null handle constant
#define DB_HANDLE_NULL nullptr

/// Result code for FFI operations
enum class EDbResult : int32_t
{
    Ok              = 0,
    InvalidHandle   = 1,
    NullPointer     = 2,
    InvalidParameter = 3,
    IoError         = 4,
    Closed          = 5,
    Unknown         = 99,
};

/// Aggregation type
enum class EAggregationType : int32_t
{
    Sum      = 0,
    Avg      = 1,
    Min      = 2,
    Max      = 3,
    Count    = 4,
    First    = 5,
    Last     = 6,
    StdDev   = 7,
    Variance = 8,
};

/// Database statistics
struct FDbStats
{
    uint64_t TotalSegments;
    uint64_t MemtableSize;
    uint64_t TotalDiskSize;
    double AverageCompressionRatio;
};

/// Point query result
struct FPointResult
{
    int64_t Timestamp;
    float Value;
    bool Found;
};

/// Version information
struct FVersionInfo
{
    uint16_t Major;
    uint16_t Minor;
    uint16_t Patch;
};

// ============================================================================
// C API Declarations
// ============================================================================

extern "C"
{

// --- Info ---
FVersionInfo alice_db_version();

// --- Lifecycle ---
DbHandle alice_db_open(const char* Path);
DbHandle alice_db_open_with_config(const char* Path, uint32_t MemtableCapacity, bool EnableWal, bool SyncWrites);
EDbResult alice_db_close(DbHandle Handle);
bool alice_db_is_valid(DbHandle Handle);

// --- Write ---
EDbResult alice_db_put(DbHandle Handle, int64_t Timestamp, float Value);
EDbResult alice_db_put_batch(DbHandle Handle, const int64_t* Timestamps, const float* Values, uint32_t Count);

// --- Read ---
FPointResult alice_db_get(DbHandle Handle, int64_t Timestamp);
int32_t alice_db_scan(DbHandle Handle, int64_t Start, int64_t End, int64_t* OutTimestamps, float* OutValues, uint32_t MaxCount);

// --- Aggregation ---
EDbResult alice_db_aggregate(DbHandle Handle, int64_t Start, int64_t End, EAggregationType Agg, double* OutValue);
int32_t alice_db_downsample(DbHandle Handle, int64_t Start, int64_t End, int64_t Interval, EAggregationType Agg, int64_t* OutTimestamps, double* OutValues, uint32_t MaxCount);

// --- Management ---
EDbResult alice_db_flush(DbHandle Handle);
FDbStats alice_db_stats(DbHandle Handle);

// --- Memory ---
void alice_db_free_string(char* Str);

} // extern "C"

// ============================================================================
// RAII Wrapper (C++ convenience)
// ============================================================================

namespace AliceDb
{

/// RAII wrapper for DbHandle - automatically closes on destruction
class FDatabase
{
public:
    FDatabase() : Handle(DB_HANDLE_NULL) {}

    explicit FDatabase(const char* Path)
        : Handle(alice_db_open(Path)) {}

    FDatabase(const char* Path, uint32_t MemtableCapacity, bool EnableWal = true, bool SyncWrites = false)
        : Handle(alice_db_open_with_config(Path, MemtableCapacity, EnableWal, SyncWrites)) {}

    ~FDatabase()
    {
        if (Handle) alice_db_close(Handle);
    }

    // Move only
    FDatabase(FDatabase&& Other) noexcept : Handle(Other.Handle) { Other.Handle = DB_HANDLE_NULL; }
    FDatabase& operator=(FDatabase&& Other) noexcept
    {
        if (this != &Other)
        {
            if (Handle) alice_db_close(Handle);
            Handle = Other.Handle;
            Other.Handle = DB_HANDLE_NULL;
        }
        return *this;
    }
    FDatabase(const FDatabase&) = delete;
    FDatabase& operator=(const FDatabase&) = delete;

    DbHandle Get() const { return Handle; }
    bool IsValid() const { return Handle && alice_db_is_valid(Handle); }
    explicit operator bool() const { return IsValid(); }

    EDbResult Put(int64_t Timestamp, float Value) const
    {
        return alice_db_put(Handle, Timestamp, Value);
    }

    EDbResult PutBatch(const int64_t* Timestamps, const float* Values, uint32_t Count) const
    {
        return alice_db_put_batch(Handle, Timestamps, Values, Count);
    }

    FPointResult GetPoint(int64_t Timestamp) const
    {
        return alice_db_get(Handle, Timestamp);
    }

    int32_t Scan(int64_t Start, int64_t End, int64_t* OutTs, float* OutVals, uint32_t MaxCount) const
    {
        return alice_db_scan(Handle, Start, End, OutTs, OutVals, MaxCount);
    }

    EDbResult Aggregate(int64_t Start, int64_t End, EAggregationType Agg, double* OutValue) const
    {
        return alice_db_aggregate(Handle, Start, End, Agg, OutValue);
    }

    EDbResult Flush() const
    {
        return alice_db_flush(Handle);
    }

    FDbStats Stats() const
    {
        return alice_db_stats(Handle);
    }

private:
    DbHandle Handle;
};

} // namespace AliceDb
