"""
TensorRT 10.x inference session for DA3 ONNX (inputs: image; outputs: depth, sky).
Uses cuda.bindings.runtime for allocations and async execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart


def _dims_to_tuple(dims: Any) -> tuple[int, ...]:
    return tuple(int(dims[i]) for i in range(len(dims)))


def _volume(shape: tuple[int, ...]) -> int:
    v = 1
    for s in shape:
        v *= int(s)
    return v


def _dtype_nbytes(trt_dtype: trt.DataType) -> int:
    if trt_dtype in (trt.DataType.FLOAT, trt.DataType.INT32):
        return 4
    if trt_dtype in (trt.DataType.HALF,):
        return 2
    if trt_dtype in (trt.DataType.INT8, trt.DataType.BOOL):
        return 1
    return 4


def _unwrap(err: Any) -> Any:
    if isinstance(err, tuple):
        return err[0]
    return err


class Da3TensorRTSession:
    """Load a serialized TensorRT engine and run inference."""

    def __init__(self, engine_path: Path, *, verbose: bool = False) -> None:
        engine_path = Path(engine_path)
        if not engine_path.is_file():
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        self.logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        err, self.stream = cudart.cudaStreamCreate()
        if _unwrap(err) != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaStreamCreate failed: {err}")

        self.input_name: str | None = None
        self.output_names: list[str] = []
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._d_ptrs: dict[str, int] = {}
        self._host_out: dict[str, np.ndarray] = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = _dims_to_tuple(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            self._shapes[name] = shape
            nbytes = _volume(shape) * _dtype_nbytes(dtype)

            err, d_ptr = cudart.cudaMalloc(nbytes)
            if _unwrap(err) != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMalloc failed for {name}: {err}")
            self._d_ptrs[name] = int(d_ptr)

            trt_np = trt.nptype(dtype)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)
                self._host_out[name] = np.empty(shape, dtype=trt_np)

        if self.input_name is None:
            raise RuntimeError("Engine has no input tensor")

    @property
    def input_shape(self) -> tuple[int, ...]:
        assert self.input_name is not None
        return self._shapes[self.input_name]

    def infer(self, input_nchw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference. input_nchw must match engine input shape, float32 contiguous.

        Returns:
            depth, sky arrays (same shapes as ONNX outputs).
        """
        assert self.input_name is not None
        inp = np.ascontiguousarray(input_nchw, dtype=np.float32)
        if inp.shape != self._shapes[self.input_name]:
            raise ValueError(
                f"Input shape {inp.shape} does not match engine {self._shapes[self.input_name]}"
            )

        in_ptr = self._d_ptrs[self.input_name]
        err = cudart.cudaMemcpyAsync(
            in_ptr,
            inp.ctypes.data,
            inp.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        )
        if _unwrap(err) != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMemcpyAsync H2D failed: {err}")

        self.context.set_tensor_address(self.input_name, in_ptr)
        for name in self.output_names:
            self.context.set_tensor_address(name, self._d_ptrs[name])

        ok = self.context.execute_async_v3(self.stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned False")

        err = cudart.cudaStreamSynchronize(self.stream)
        if _unwrap(err) != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaStreamSynchronize failed: {err}")

        for name in self.output_names:
            ho = self._host_out[name]
            err = cudart.cudaMemcpyAsync(
                ho.ctypes.data,
                self._d_ptrs[name],
                ho.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )
            if _unwrap(err) != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMemcpyAsync D2H failed for {name}: {err}")

        err = cudart.cudaStreamSynchronize(self.stream)
        if _unwrap(err) != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaStreamSynchronize after D2H failed: {err}")

        def _copy_named(key: str, idx: int) -> np.ndarray:
            if key in self._host_out:
                return np.copy(self._host_out[key])
            if idx < len(self.output_names):
                return np.copy(self._host_out[self.output_names[idx]])
            raise KeyError(key)

        depth = _copy_named("depth", 0).astype(np.float32, copy=False)
        sky = _copy_named("sky", 1).astype(np.float32, copy=False)
        return depth, sky

    def close(self) -> None:
        for ptr in self._d_ptrs.values():
            cudart.cudaFree(ptr)
        if self.stream is not None:
            cudart.cudaStreamDestroy(self.stream)
            self.stream = None  # type: ignore[assignment]

    def __enter__(self) -> Da3TensorRTSession:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
