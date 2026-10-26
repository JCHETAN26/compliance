"""The exporter's int8 scheme survives activation outlier channels (needs the tool overlay).

DeBERTa-v3-base's FFN inputs have a few channels around 100 against a spread of ~0.25.
Dynamic quantization uses one uint8 scale per activation tensor, so every other channel
falls below one quantization step and the model collapsed (docs/INT8_QUANTIZATION.md).
The tiny graph here has the same structure: embedding Gather -> MatMul -> Relu with one
huge channel -> MatMul whose weights for that channel are small.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx  # type: ignore[import-not-found,unused-ignore]  # tool overlay
import onnxruntime as ort  # type: ignore[import-untyped]
from onnx import TensorProto, helper, numpy_helper

from ml.export.export import quantize_int8

VOCAB, HIDDEN, INNER = 256, 128, 512


def _outlier_graph(path: Path) -> None:
    rng = np.random.default_rng(0)
    emb = rng.normal(0, 1, (VOCAB, HIDDEN)).astype(np.float32)
    w1 = rng.normal(0, HIDDEN**-0.5, (HIDDEN, INNER)).astype(np.float32)
    w1[:, 0] *= 300  # one outlier channel after the activation
    w2 = rng.normal(0, INNER**-0.5, (INNER, HIDDEN)).astype(np.float32)
    w2[0, :] /= 300  # ... that contributes little downstream
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["emb", "input_ids"], ["h"]),
            helper.make_node("MatMul", ["h", "w1"], ["a"]),
            helper.make_node("Relu", ["a"], ["r"]),
            helper.make_node("MatMul", ["r", "w2"], ["y"]),
        ],
        "outliers",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", "seq", HIDDEN])],
        [numpy_helper.from_array(a, n) for a, n in ((emb, "emb"), (w1, "w1"), (w2, "w2"))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


def _run(path: Path, ids: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.asarray(session.run(["y"], {"input_ids": ids})[0])


def _relative_error(out: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(out - ref) / np.linalg.norm(ref))


def test_weight_only_int8_keeps_outputs_where_dynamic_quantization_collapses(
    tmp_path: Path,
) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore[import-untyped]

    fp32, int8, dynamic = tmp_path / "fp32.onnx", tmp_path / "int8.onnx", tmp_path / "dyn.onnx"
    _outlier_graph(fp32)
    ids = np.random.default_rng(1).integers(0, VOCAB, (4, 16), dtype=np.int64)
    ref = _run(fp32, ids)

    quantize_int8(fp32, int8)
    # measured 0.099, almost all from the 4-bit embedding of this random N(0, 1) table
    assert _relative_error(_run(int8, ids), ref) < 0.15
    ops = {n.op_type for n in onnx.load(str(int8)).graph.node}
    assert {"MatMulNBits", "GatherBlockQuantized"} <= ops  # weights and embedding quantized
    assert not ops & {"MatMul", "Gather", "DynamicQuantizeLinear"}
    assert int8.stat().st_size < fp32.stat().st_size / 2

    # The previous exporter configuration, and the documented non-VNNI remedy (MatMul
    # only, per-channel, 7-bit weights), on the same graph: measured 1.03 and 0.78.
    for options in (
        {},
        {"op_types_to_quantize": ["MatMul"], "per_channel": True, "reduce_range": True},
    ):
        quantize_dynamic(str(fp32), str(dynamic), weight_type=QuantType.QInt8, **options)
        assert _relative_error(_run(dynamic, ids), ref) > 0.5, options
