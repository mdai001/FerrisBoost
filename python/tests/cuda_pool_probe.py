"""Test-only CUDA default-pool allocation accounting.

The reported used-memory values describe allocations requested by this process
from CUDA's default memory pool. They are not total device usage and intentionally
exclude driver-internal pools and allocations owned by other processes. Reserved
pool memory is retained as diagnostic data only; it is not counted as a requested
allocation.
"""
import ctypes
import os

import pytest


_MIB = 1024 * 1024
_DEVICE = 0
_MEMORY_POOLS_SUPPORTED = 115
_POOL_ATTRIBUTES = {
    "reserved_current": 5,
    "reserved_high": 6,
    "used_current": 7,
    "used_high": 8,
}


def _check_cuda_result(name, result):
    if result != 0:
        raise RuntimeError(f"{name} failed with CUDA driver error {result}")


class _CudaDriver:
    """Lazily loaded CUDA driver API with explicit ctypes signatures."""

    def __init__(self):
        if os.name == "nt":
            self.lib = ctypes.WinDLL("nvcuda.dll")
        else:
            self.lib = ctypes.CDLL("libcuda.so.1")

        void_p = ctypes.c_void_p
        uint = ctypes.c_uint
        int_ = ctypes.c_int
        uint64 = ctypes.c_uint64
        size = ctypes.c_size_t
        signatures = {
            "cuInit": ([uint], int_),
            "cuCtxGetCurrent": ([ctypes.POINTER(void_p)], int_),
            "cuCtxSetCurrent": ([void_p], int_),
            "cuCtxSynchronize": ([], int_),
            "cuDevicePrimaryCtxRetain": ([ctypes.POINTER(void_p), int_], int_),
            "cuDevicePrimaryCtxRelease_v2": ([int_], int_),
            "cuDeviceGetAttribute": ([ctypes.POINTER(int_), int_, int_], int_),
            "cuDeviceGetDefaultMemPool": ([ctypes.POINTER(void_p), int_], int_),
            "cuMemPoolGetAttribute": ([void_p, int_, void_p], int_),
            "cuMemPoolSetAttribute": ([void_p, int_, void_p], int_),
            "cuMemAllocAsync": ([ctypes.POINTER(uint64), size, void_p], int_),
            "cuMemFreeAsync": ([uint64, void_p], int_),
        }
        for name, (argtypes, restype) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = argtypes
            function.restype = restype

    def call(self, name, *args):
        _check_cuda_result(name, getattr(self.lib, name)(*args))


class PoolProbe:
    """Own a temporary retain on device 0's primary context while querying its pool."""

    def __init__(self, _driver_factory=None):
        self._driver_factory = _driver_factory or _CudaDriver
        self._driver = None
        self._previous_context = None
        self._primary_context = None
        self._retained = False
        self._made_current = False
        self._pool = None
        self._closed = False

    def __enter__(self):
        try:
            self._driver = self._driver_factory()
            d = self._driver
            d.call("cuInit", 0)

            previous = ctypes.c_void_p()
            d.call("cuCtxGetCurrent", ctypes.byref(previous))
            self._previous_context = previous

            primary = ctypes.c_void_p()
            d.call("cuDevicePrimaryCtxRetain", ctypes.byref(primary), _DEVICE)
            self._primary_context = primary
            self._retained = True
            d.call("cuCtxSetCurrent", primary)
            self._made_current = True

            supported = ctypes.c_int()
            d.call("cuDeviceGetAttribute", ctypes.byref(supported),
                   _MEMORY_POOLS_SUPPORTED, _DEVICE)
            if supported.value == 0:
                pytest.skip("CUDA device 0 does not support memory pools")

            pool = ctypes.c_void_p()
            d.call("cuDeviceGetDefaultMemPool", ctypes.byref(pool), _DEVICE)
            self._pool = pool
            self.synchronize()
            self._set_attribute(_POOL_ATTRIBUTES["used_high"], 0)
            return self
        except BaseException:
            self.close()
            raise

    def synchronize(self):
        self._driver.call("cuCtxSynchronize")

    def _set_attribute(self, attribute, value):
        value_ptr = ctypes.c_uint64(value)
        self._driver.call("cuMemPoolSetAttribute", self._pool, attribute,
                          ctypes.byref(value_ptr))

    def stats(self):
        values = {}
        for name, attribute in _POOL_ATTRIBUTES.items():
            value = ctypes.c_uint64()
            self._driver.call("cuMemPoolGetAttribute", self._pool, attribute,
                              ctypes.byref(value))
            values[name] = value.value
        return values

    def allocate_async(self, size_bytes):
        address = ctypes.c_uint64()
        self._driver.call("cuMemAllocAsync", ctypes.byref(address), size_bytes,
                          ctypes.c_void_p())
        return address.value

    def free_async(self, address):
        self._driver.call("cuMemFreeAsync", address, ctypes.c_void_p())

    def close(self):
        if self._closed or self._driver is None:
            return
        self._closed = True
        d = self._driver
        cleanup_error = None
        if self._made_current:
            try:
                d.call("cuCtxSetCurrent", self._previous_context)
            except BaseException as error:  # still release the retained primary context
                cleanup_error = error
            self._made_current = False
        if self._retained:
            try:
                d.call("cuDevicePrimaryCtxRelease_v2", _DEVICE)
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
            self._retained = False
        if cleanup_error is not None:
            raise cleanup_error

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def _used_pool_mb():
    """Return this process's default-pool current usage in MiB; CUDA errors propagate."""
    with PoolProbe() as probe:
        return probe.stats()["used_current"] / _MIB


class _PeakPoolAllocation:
    """Capture the driver's resettable USED_MEM_HIGH watermark, without polling."""

    def __init__(self):
        self.peak = 0.0
        self.stats = None
        self._probe = PoolProbe()

    def __enter__(self):
        self._probe.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self._probe.synchronize()
            self.stats = self._probe.stats()
            self.peak = self.stats["used_high"] / _MIB
        finally:
            self._probe.__exit__(exc_type, exc, traceback)
        return False
