# Quick Start

<!-- md-trans-meta sourceCommit=c020bdb62acb53e4224a27aecdcce6019c4e8cd5 translatedAt=2026-11-11T08:58:34.403Z pushedAt=2026-11-12T01:07:09.484Z -->

This document is a quick start guide for Kunpeng ONNX Runtime. It is intended for developers who have completed the build and installation and want to quickly run inference.

## Installing Kunpeng ONNX Runtime

Before using ONNX Runtime for inference services, see [Installation Guide](./installation_guide.md) to complete the [compilation and installation](./installation_guide.md#compiling-and-installing-onnx-runtime) of Kunpeng ONNX Runtime.

## Running Inference Services

Before running inference services, ensure that the following prerequisites are met.

1. `Python 3.x` has been installed in the current environment.

    ```bash
    python --version  # Expected output: the corresponding Python version
    ```

2. `onnxruntime` has been built and installed in the current environment.

    ```bash
    pip show onnxruntime  # Expected output: the corresponding onnxruntime version
    ```

3. The model file to be inferred has been prepared. Assume that the model is stored in the path `/path/to/model.onnx`.
4. The dependency package `numpy` has been installed in the current environment.

    ```bash
    pip show numpy  # Expected output: the corresponding numpy version
    ```

After the preceding prerequisites are met, run the following script to perform inference on the target ONNX model.

```python
# ort_cpu_infer_demo.py

import numpy as np
import onnxruntime as ort
import os
import ctypes
import onnxruntime.capi.onnxruntime_pybind11_state as rtpy

# ---------- Utility functions ----------
def _ort_type_to_np(ort_type: str) -> np.dtype:
    """Map ONNX tensor type strings to numpy dtypes."""
    # ort_type is in the form of "tensor(float)" / "tensor(int64)" / "tensor(string)"
    mapping = {
        "float":    np.float32,
        "double":   np.float64,
        "half":     np.float16,
        "bfloat16": np.float32,   # numpy does not have bfloat16. Use float32 as a placeholder temporarily.
        "int8":     np.int8,
        "int16":    np.int16,
        "int32":    np.int32,
        "int64":    np.int64,
        "uint8":    np.uint8,
        "uint16":   np.uint16,
        "uint32":   np.uint32,
        "uint64":   np.uint64,
        "bool":     np.bool_,
    }
    # Extract "float" from "tensor(float)"
    key = ort_type.replace("tensor(", "").rstrip(")")
    if key == "string":
        return None  # String type is handled separately
    if key not in mapping:
        raise ValueError(f"[ERROR] Unsupported ONNX dtype: {ort_type}")
    return mapping[key]

def _resolve_shape(raw_shape, batch_size: int) -> tuple:
    """
    Resolve the raw ORT shape (a list, possibly containing string dynamic dimensions) into an int tuple.
    Dynamic dimensions (strings / None) are uniformly replaced with batch_size.
    """
    resolved = []
    for d in raw_shape:
        if isinstance(d, int):
            resolved.append(d)
        else:  # String (such as 'unk_7517') or None
            resolved.append(batch_size)
    return tuple(resolved)

def _build_dummy_input(input_meta, batch_size: int) -> np.ndarray:
    """Generate a dummy NumPy array that matches the dtype and shape according to an ORT input meta."""
    shape = _resolve_shape(input_meta.shape, batch_size)
    ort_type = input_meta.type
    np_dtype = _ort_type_to_np(ort_type)

    if np_dtype is None:  # tensor(string)
        # Use an empty string array as a placeholder; for real inference, replace it with actual text.
        return np.full(shape, "", dtype=object)

    if np.issubdtype(np_dtype, np.floating):
        return np.zeros(shape, dtype=np_dtype)

    if np.issubdtype(np_dtype, np.integer):
        return np.zeros(shape, dtype=np_dtype)

    if np_dtype == np.bool_:
        return np.zeros(shape, dtype=np_dtype)

    raise ValueError(f"[ERROR] Unhandled dtype: {np_dtype}")

# ---------- Main flow ----------
def main():
    # 1. Parse command-line arguments
    model_path = "/path/to/model.onnx"  # Path to the ONNX model to be inferred
    batch_size = 32       # Batch size. Configure as needed.

    ctypes.CDLL(
        os.path.join(os.path.dirname(ort.__file__), "capi", "libonnxruntime.so.1.26.0"),
        mode=ctypes.RTLD_GLOBAL)
    # /path/to/ANNC is the ANNC source code directory
    rtpy.register_execution_provider_library(
        "AnncExecutionProvider", "/path/to/ANNC/install/lib64/libannc_ep.so")

    # 2. Create a session (explicitly specify the CPU provider to make the intent clear)
    print(f"[INFO] Loading model: {model_path}")
    session = ort.InferenceSession(
        model_path,
        providers=[
            ("KDNNExecutionProvider",{          # Invoke the KDNN EP
                "enable_attention_fusion": "1",  
                "attention_algorithm": "auto",
            }),
            "CPUExecutionProvider",
            "AnncExecutionProvider"             # Invoke the ANNC EP
        ])

    print(f"[INFO] Execution providers: {session.get_providers()}")
    print(f"[INFO] Number of inputs: {len(session.get_inputs())}")
    print(f"[INFO] Number of outputs: {len(session.get_outputs())}")

    # 3. Automatically build all inputs (key change: no hardcoded input name/shape/dtype)
    input_feed = {}
    np.random.seed(1)
    for meta in session.get_inputs():
        arr = _build_dummy_input(meta, batch_size)
        input_feed[meta.name] = arr

    # Print a summary of the first 5 inputs
    preview = list(input_feed.items())[:5]
    print(f"[INFO] Built {len(input_feed)} dummy inputs. The first 5:")
    for name, arr in preview:
        print(f"      '{name}'  shape={arr.shape}  dtype={arr.dtype}")
    if len(input_feed) > 5:
        print(f"      ... (the remaining {len(input_feed) - 5} inputs are omitted)")

    # 4. Run inference
    print("[INFO] Starting inference (CPU)...")
    outputs = session.run(None, input_feed)

    # 5. Print an output summary
    print(f"[INFO] Inference completed. {len(outputs)} outputs in total:")
    for i, out in enumerate(outputs):
        print(f"      output[{i}]  shape={out.shape}  dtype={out.dtype}  "
              f"sample={out.flat[:5] if out.size > 0 else '<empty>'}")

    print("[OK] inference done ✓")

if __name__ == "__main__":
    main()
```

Save the preceding script as `ort_cpu_infer_demo.py`, and run the following commands to enable all optimization features for inference.

```bash
export ANNC_EP_FUSION_LAYER_NORM=1
export ANNC_EP_FUSION_TENSORDOT_MATMUL=1
export ANNC_EP_FUSION_MASKED_MUL=1
export ANNC_EP_FUSION_BATCHED_MATMUL_ADD=1
export ORT_ENABLE_PARALLEL_EXECUTOR=1
python ort_cpu_infer_demo.py
```

The expected output after the inference is complete is as follows.

```output
[INFO] Loading model: /path/to/model.onnx
[INFO] Execution providers: ['KDNNExecutionProvider', 'CPUExecutionProvider', 'AnncExecutionProvider']
[INFO] Number of inputs: **
[INFO] Number of outputs: **
[INFO] Built ** dummy inputs. The first 5:
      ...
      ... (the remaining ** inputs are omitted)
[INFO] Starting inference (CPU)...
[INFO] Inference completed. ** outputs in total:
      output[0]  ...
      ...
[OK] inference done ✓
```

## Change History

| Version | Date | Description |
| ---- | ---- | ---- |
| 01 | 2026-12-30 | This is the first official release. |
