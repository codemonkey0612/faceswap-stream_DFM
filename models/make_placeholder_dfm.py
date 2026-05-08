"""Generate models/placeholder.dfm — a minimal valid ONNX identity model.

This model has the same interface as a real DeepFaceLive DFM:
  input  : 'input'  — (1, 3, size, size) float32 RGB [0, 1]
  output0: 'face'   — (1, 3, size, size) float32 (identity pass-through)
  output1: 'mask'   — (1, 1, size, size) float32 all-ones

Run once before launching the pipeline in non-identity mode:
    .venv\\Scripts\\python.exe models\\make_placeholder_dfm.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


SIZE = 256
OUT = Path(__file__).parent / "placeholder.dfm"


def build_placeholder(size: int = SIZE) -> onnx.ModelProto:
    """Return an ONNX ModelProto: identity face + all-ones mask."""

    # --- graph inputs ---
    input_node = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, size, size]
    )

    # --- graph outputs ---
    face_out = helper.make_tensor_value_info(
        "face", TensorProto.FLOAT, [1, 3, size, size]
    )
    mask_out = helper.make_tensor_value_info(
        "mask", TensorProto.FLOAT, [1, 1, size, size]
    )

    # --- nodes ---
    # face = Identity(input)
    identity_node = helper.make_node("Identity", inputs=["input"], outputs=["face"])

    # mask = Constant — all ones (1, 1, size, size)
    ones = np.ones((1, 1, size, size), dtype=np.float32)
    ones_tensor = numpy_helper.from_array(ones, name="ones_value")
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["mask"],
        value=ones_tensor,
    )

    # --- graph ---
    graph = helper.make_graph(
        [identity_node, const_node],
        "placeholder_dfm",
        [input_node],
        [face_out, mask_out],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main() -> None:
    model = build_placeholder(SIZE)
    onnx.save(model, str(OUT))
    kb = OUT.stat().st_size // 1024
    print(f"Written: {OUT}  ({kb} KB)  size={SIZE}x{SIZE}")
    print("Load test … ", end="", flush=True)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    inp = np.zeros((1, 3, SIZE, SIZE), dtype=np.float32)
    face, mask = sess.run(None, {"input": inp})
    assert face.shape == (1, 3, SIZE, SIZE), face.shape
    assert mask.shape == (1, 1, SIZE, SIZE), mask.shape
    print("OK")


if __name__ == "__main__":
    main()
