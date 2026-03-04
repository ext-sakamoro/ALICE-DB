/**
 * ALICE-DB C# Bindings for Unity
 *
 * Usage:
 * 1. Copy this file to your Unity project's Assets folder
 * 2. Copy libalice_db.dylib (macOS), alice_db.dll (Windows), or libalice_db.so (Linux)
 *    to Assets/Plugins/
 * 3. Use AliceDb class to insert and query time-series data
 *
 * Example:
 *   using (var db = new AliceDb("/path/to/data"))
 *   {
 *       db.Put(100, 42.0f);
 *       var point = db.Get(100);
 *       if (point.Found) Debug.Log($"Value: {point.Value}");
 *       db.Flush();
 *   }
 *
 * Author: Moroya Sakamoto
 */

using System;
using System.Runtime.InteropServices;

namespace AliceDbUnity
{
    /// <summary>
    /// Result codes from FFI operations
    /// </summary>
    public enum DbResult : int
    {
        Ok = 0,
        InvalidHandle = 1,
        NullPointer = 2,
        InvalidParameter = 3,
        IoError = 4,
        Closed = 5,
        Unknown = 99
    }

    /// <summary>
    /// Aggregation type
    /// </summary>
    public enum AggregationType : int
    {
        Sum = 0,
        Avg = 1,
        Min = 2,
        Max = 3,
        Count = 4,
        First = 5,
        Last = 6,
        StdDev = 7,
        Variance = 8
    }

    /// <summary>
    /// Database statistics
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct DbStats
    {
        public ulong TotalSegments;
        public ulong MemtableSize;
        public ulong TotalDiskSize;
        public double AverageCompressionRatio;
    }

    /// <summary>
    /// Single point query result
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PointResult
    {
        public long Timestamp;
        public float Value;
        [MarshalAs(UnmanagedType.U1)]
        public bool Found;
    }

    /// <summary>
    /// Version information
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct VersionInfo
    {
        public ushort Major;
        public ushort Minor;
        public ushort Patch;
    }

    /// <summary>
    /// ALICE-DB high-level wrapper with IDisposable support
    /// </summary>
    public class AliceDb : IDisposable
    {
        private const string DllName = "alice_db";
        private IntPtr _handle;
        private bool _disposed;

        // --- Native imports ---

        [DllImport(DllName)] private static extern VersionInfo alice_db_version();
        [DllImport(DllName)] private static extern IntPtr alice_db_open([MarshalAs(UnmanagedType.LPUTF8Str)] string path);
        [DllImport(DllName)] private static extern IntPtr alice_db_open_with_config([MarshalAs(UnmanagedType.LPUTF8Str)] string path, uint memtableCapacity, [MarshalAs(UnmanagedType.U1)] bool enableWal, [MarshalAs(UnmanagedType.U1)] bool syncWrites);
        [DllImport(DllName)] private static extern DbResult alice_db_close(IntPtr handle);
        [DllImport(DllName)] private static extern DbResult alice_db_put(IntPtr handle, long timestamp, float value);
        [DllImport(DllName)] private static extern unsafe DbResult alice_db_put_batch(IntPtr handle, long* timestamps, float* values, uint count);
        [DllImport(DllName)] private static extern PointResult alice_db_get(IntPtr handle, long timestamp);
        [DllImport(DllName)] private static extern unsafe int alice_db_scan(IntPtr handle, long start, long end, long* outTimestamps, float* outValues, uint maxCount);
        [DllImport(DllName)] private static extern unsafe DbResult alice_db_aggregate(IntPtr handle, long start, long end, AggregationType agg, double* outValue);
        [DllImport(DllName)] private static extern unsafe int alice_db_downsample(IntPtr handle, long start, long end, long interval, AggregationType agg, long* outTimestamps, double* outValues, uint maxCount);
        [DllImport(DllName)] private static extern DbResult alice_db_flush(IntPtr handle);
        [DllImport(DllName)] private static extern DbStats alice_db_stats(IntPtr handle);
        [DllImport(DllName)] [return: MarshalAs(UnmanagedType.U1)] private static extern bool alice_db_is_valid(IntPtr handle);

        // --- Public API ---

        public static VersionInfo Version => alice_db_version();

        public AliceDb(string path)
        {
            _handle = alice_db_open(path);
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to open ALICE-DB at: {path}");
        }

        public AliceDb(string path, uint memtableCapacity, bool enableWal = true, bool syncWrites = false)
        {
            _handle = alice_db_open_with_config(path, memtableCapacity, enableWal, syncWrites);
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to open ALICE-DB at: {path}");
        }

        public bool IsValid => _handle != IntPtr.Zero && alice_db_is_valid(_handle);

        public void Put(long timestamp, float value)
        {
            var result = alice_db_put(_handle, timestamp, value);
            if (result != DbResult.Ok) throw new InvalidOperationException($"Put failed: {result}");
        }

        public unsafe void PutBatch(long[] timestamps, float[] values)
        {
            if (timestamps.Length != values.Length)
                throw new ArgumentException("timestamps and values must have same length");
            fixed (long* ts = timestamps)
            fixed (float* vs = values)
            {
                var result = alice_db_put_batch(_handle, ts, vs, (uint)timestamps.Length);
                if (result != DbResult.Ok) throw new InvalidOperationException($"PutBatch failed: {result}");
            }
        }

        public PointResult Get(long timestamp) => alice_db_get(_handle, timestamp);

        public unsafe (long[] timestamps, float[] values) Scan(long start, long end, uint maxCount = 65536)
        {
            // First call to get count
            int count = alice_db_scan(_handle, start, end, null, null, 0);
            if (count <= 0) return (Array.Empty<long>(), Array.Empty<float>());

            uint n = (uint)Math.Min(count, (int)maxCount);
            var ts = new long[n];
            var vs = new float[n];
            fixed (long* tsPtr = ts)
            fixed (float* vsPtr = vs)
            {
                int written = alice_db_scan(_handle, start, end, tsPtr, vsPtr, n);
                if (written < 0) throw new InvalidOperationException("Scan failed");
                if (written < (int)n)
                {
                    Array.Resize(ref ts, written);
                    Array.Resize(ref vs, written);
                }
            }
            return (ts, vs);
        }

        public unsafe double Aggregate(long start, long end, AggregationType agg = AggregationType.Avg)
        {
            double val;
            var result = alice_db_aggregate(_handle, start, end, agg, &val);
            if (result != DbResult.Ok) throw new InvalidOperationException($"Aggregate failed: {result}");
            return val;
        }

        public unsafe (long[] timestamps, double[] values) Downsample(long start, long end, long interval, AggregationType agg = AggregationType.Avg, uint maxCount = 4096)
        {
            int count = alice_db_downsample(_handle, start, end, interval, agg, null, null, 0);
            if (count <= 0) return (Array.Empty<long>(), Array.Empty<double>());

            uint n = (uint)Math.Min(count, (int)maxCount);
            var ts = new long[n];
            var vs = new double[n];
            fixed (long* tsPtr = ts)
            fixed (double* vsPtr = vs)
            {
                int written = alice_db_downsample(_handle, start, end, interval, agg, tsPtr, vsPtr, n);
                if (written < 0) throw new InvalidOperationException("Downsample failed");
                if (written < (int)n)
                {
                    Array.Resize(ref ts, written);
                    Array.Resize(ref vs, written);
                }
            }
            return (ts, vs);
        }

        public void Flush()
        {
            var result = alice_db_flush(_handle);
            if (result != DbResult.Ok) throw new InvalidOperationException($"Flush failed: {result}");
        }

        public DbStats Stats => alice_db_stats(_handle);

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed && _handle != IntPtr.Zero)
            {
                alice_db_close(_handle);
                _handle = IntPtr.Zero;
                _disposed = true;
            }
        }

        ~AliceDb() => Dispose(false);
    }
}
