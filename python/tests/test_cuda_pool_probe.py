"""Contract tests for process-requested CUDA default-pool allocation accounting."""
import ctypes

import numpy as np
import pytest

import ferrisboost as fb

from cuda_pool_probe import PoolProbe, _MIB, _check_cuda_result


def _has_cuda():
    x = np.zeros((64, 2), dtype=np.float32)
    y = np.zeros(64, dtype=np.float32)
    try:
        fb.train({"device": "cuda", "max_depth": 2,
                  "objective": "reg:squarederror"}, x, y,
                 num_boost_round=1, quiet=True)
        return True
    except Exception as error:  # noqa: BLE001
        if "CUDA" in str(error) or "cuda" in str(error):
            return False
        raise


cuda = pytest.mark.skipif(not _has_cuda(), reason="no usable CUDA build or device")


@cuda
def test_async_allocation_highwater_survives_free_and_current_returns_to_baseline():
    allocation_size = 64 * _MIB
    with PoolProbe() as probe:
        baseline = probe.stats()["used_current"]
        address = probe.allocate_async(allocation_size)
        probe.synchronize()
        probe.free_async(address)
        probe.synchronize()
        result = probe.stats()

    assert result["used_high"] >= baseline + allocation_size
    assert result["used_current"] == baseline
    # Pool reservation is diagnostic, not an allocation-budget measurement.
    assert result["reserved_current"] >= result["used_current"]
    assert result["reserved_high"] >= result["reserved_current"]


@cuda
def test_reset_highwater_does_not_retain_a_previous_context_peak():
    with PoolProbe() as probe:
        baseline = probe.stats()["used_current"]
        address = probe.allocate_async(16 * _MIB)
        probe.synchronize()
        probe.free_async(address)
        probe.synchronize()
        previous_peak = probe.stats()["used_high"]
    assert previous_peak >= baseline + 16 * _MIB

    with PoolProbe() as probe:
        result = probe.stats()
    assert result["used_high"] == result["used_current"]
    assert result["used_high"] < previous_peak


@cuda
def test_small_budget_control_rejects_the_measured_allocation():
    with PoolProbe() as probe:
        baseline = probe.stats()["used_current"]
        address = probe.allocate_async(16 * _MIB)
        probe.synchronize()
        probe.free_async(address)
        probe.synchronize()
        observed = probe.stats()["used_high"] - baseline

    with pytest.raises(AssertionError):
        assert observed <= 1 * _MIB, (observed, 1 * _MIB)


def test_cuda_driver_errors_are_not_converted_to_missing_measurements():
    with pytest.raises(RuntimeError, match="cuCtxSynchronize.*error 719"):
        _check_cuda_result("cuCtxSynchronize", 719)


def test_exception_restores_previous_context_and_releases_primary_reference():
    class FakeDriver:
        def __init__(self):
            self.calls = []
            self.current = None

        def call(self, name, *args):
            self.calls.append(name)
            if name == "cuCtxGetCurrent":
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = 0x1234
            elif name == "cuDevicePrimaryCtxRetain":
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = 0x5678
            elif name == "cuCtxSetCurrent":
                self.current = args[0].value if args[0] else None
            elif name == "cuDeviceGetAttribute":
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_int))[0] = 1
            elif name == "cuDeviceGetDefaultMemPool":
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = 0x9ABC
            elif name == "cuMemPoolGetAttribute":
                ctypes.cast(args[2], ctypes.POINTER(ctypes.c_uint64))[0] = 4096

    driver = FakeDriver()
    probe = PoolProbe(_driver_factory=lambda: driver)
    with pytest.raises(ValueError, match="sentinel"):
        with probe:
            raise ValueError("sentinel")

    assert driver.current == 0x1234
    assert driver.calls.count("cuDevicePrimaryCtxRetain") == 1
    assert driver.calls.count("cuDevicePrimaryCtxRelease_v2") == 1
    assert driver.calls[-2:] == ["cuCtxSetCurrent", "cuDevicePrimaryCtxRelease_v2"]
