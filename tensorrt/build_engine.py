#!/usr/bin/env python3
"""
Build a TensorRT engine from DA3 ONNX (FP16, fixed shapes).
Requires: pip install tensorrt (TensorRT 10.x).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import tensorrt as trt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ONNX → TensorRT engine for DA3METRIC-LARGE")
    p.add_argument("--onnx", type=str, required=True, help="Path to .onnx file")
    p.add_argument("--output", "-o", type=str, required=True, help="Output .engine path")
    p.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 (default: on)")
    p.add_argument("--no-fp16", action="store_true", help="Disable FP16 (FP32 only)")
    p.add_argument(
        "--workspace-gib",
        type=float,
        default=4.0,
        help="Workspace size in GiB (default: 4)",
    )
    p.add_argument("--verbose", action="store_true", help="TensorRT verbose logger")
    p.add_argument(
        "--trtexec",
        action="store_true",
        help="Use trtexec CLI instead of Python builder (must be on PATH)",
    )
    return p.parse_args()


def build_with_python_api(onnx_path: Path, engine_path: Path, *, fp16: bool, workspace_gib: float, verbose: bool) -> None:
    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    data = onnx_path.read_bytes()
    if not parser.parse(data):
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * (1 << 30)))
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif fp16 and not builder.platform_has_fast_fp16:
        print("Warning: FP16 requested but platform_has_fast_fp16 is False; building FP32.", file=sys.stderr)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    print(f"Wrote engine: {engine_path.resolve()} ({engine_path.stat().st_size / 1e6:.2f} MB)")


def build_with_trtexec(onnx_path: Path, engine_path: Path, *, fp16: bool, workspace_gib: float, verbose: bool) -> None:
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "trtexec",
        f"--onnx={onnx_path.resolve()}",
        f"--saveEngine={engine_path.resolve()}",
        f"--memPoolSize=workspace:{int(workspace_gib * (1 << 30))}",
    ]
    if fp16:
        cmd.append("--fp16")
    if verbose:
        cmd.append("--verbose")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.is_file():
        raise SystemExit(f"ONNX not found: {onnx_path}")
    engine_path = Path(args.output).resolve()
    fp16 = args.fp16 and not args.no_fp16

    if args.trtexec:
        build_with_trtexec(onnx_path, engine_path, fp16=fp16, workspace_gib=args.workspace_gib, verbose=args.verbose)
    else:
        build_with_python_api(onnx_path, engine_path, fp16=fp16, workspace_gib=args.workspace_gib, verbose=args.verbose)


if __name__ == "__main__":
    main()
