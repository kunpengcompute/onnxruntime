# API Reference

<!-- md-trans-meta sourceCommit=c020bdb62acb53e4224a27aecdcce6019c4e8cd5 translatedAt=2026-11-11T08:58:34.403Z pushedAt=2026-11-12T01:07:09.484Z -->

This document describes the details and usage of the interfaces for the ONNX Runtime graph optimization feature and the ONNX Runtime parallel scheduling optimization feature. It is intended for application developers who need to tune inference performance.

## ONNX Runtime Graph Optimization Feature Interfaces

Graph optimization is enabled by environment variables, which are used for global enabling. For details, see [Table 1 Graph optimization feature interfaces](#graph-optimization-features).

**Table 1** Graph optimization feature interfaces<a id="graph-optimization-features"></a>

| Interface | Type | Value | Description |
| ---- | ---- | ---- | ---- |
| ANNC_EP_FUSION_LAYER_NORM | Environment variable | `1`: enabled  `0`: disabled (default) | Enables the graph fusion policy of ANNC LayerNorm. |
| ANNC_EP_FUSION_TENSORDOT_MATMUL | Environment variable | `1`: enabled  `0`: disabled (default) | Enables the graph fusion policy of ANNC TensordotMatmul.|
| ANNC_EP_FUSION_MASKED_MUL | Environment variable | `1`: enabled  `0`: disabled (default) | Enables the graph fusion policy of ANNC MaskedMul.|
| ANNC_EP_FUSION_BATCHED_MATMUL_ADD | Environment variable | `1`: enabled  `0`: disabled (default) | Enables the graph fusion policy of ANNC BatchedMatmulAdd.|
| ORT_KDNN_FUSE_ATTENTION | Environment variable | `1`: enabled  `0`: disabled (default) | Enables the graph fusion policy of KDNN FuseAttention.|
| ORT_KDNN_ATTENTION_ALGORITHM | Environment variable | `auto` (default)/`classic`/`classic_no_pack` | Controls whether to pack Q/K/V.|

### KDNN Graph Fusion Policy Interfaces

The graph fusion policy of KDNN FuseAttention and the Attention algorithm can be explicitly specified through APIs when SessionOptions is created, and explicitly specified values take precedence over environment variables.

- Commands for specifying the APIs in C++:

    ```cpp
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    options.AppendExecutionProvider("KDNN", {{"enable_attention_fusion", "1"},
                                            {"attention_algorithm", "auto"}});
    ```

- Commands for specifying the APIs in Python:

    ```python
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    session = ort.InferenceSession(
        model_path,   # model_path is the path to the ONNX model to be inferred
        options,
        providers=[
            ("KDNNExecutionProvider", {
                "enable_attention_fusion": "1",
                "attention_algorithm": "auto",
            }),
            "CPUExecutionProvider",
        ],
    )
    ```

### ANNC Graph Fusion Policy Interfaces

The ANNC graph fusion policy requires explicitly specifying the ANNC API when SessionOptions is created. The corresponding APIs are as follows:

- Commands for specifying the APIs in C++:

    ```cpp
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    options.AppendExecutionProvider("ANNC");
    ```

- Commands for specifying the APIs in Python:

    ```python
    import os
    import onnxruntime as ort
    import ctypes
    import onnxruntime.capi.onnxruntime_pybind11_state as rtpy

    ctypes.CDLL(
        os.path.join(os.path.dirname(ort.__file__), "capi", "libonnxruntime.so.1.26.0"),
        mode=ctypes.RTLD_GLOBAL)
    rtpy.register_execution_provider_library(
        "AnncExecutionProvider", "<ANNC_PREFIX>/lib64/libannc_ep.so")  # Absolute path; ANNC build output

    session = ort.InferenceSession(
        model_path,   # model_path is the path to the ONNX model to be inferred
        options,
        providers=["AnncExecutionProvider", "CPUExecutionProvider"],
    )
    ```

## ONNX Runtime Parallel Scheduling Optimization

Parallel scheduling optimization is enabled by setting the environment variable and the open-source ONNX Runtime interface. The environment variable is described in [Table 2 Parallel scheduling optimization feature interface](#parallel-executor-features). Parallel scheduling optimization takes effect only when both the environment variable and the open-source ONNX Runtime interface are enabled.

**Table 2** Parallel scheduling optimization feature interface<a id="parallel-executor-features"></a>

| Interface  | Type | Value | Description |
| ----- | ---- | ---- | ---- |
| ORT_ENABLE_PARALLEL_EXECUTOR | Environment variable | `1`: enabled  `0`: disabled (default) | Enables the parallel scheduling optimization policy. |

The open-source ONNX Runtime interface defaults to Sequential mode (each node is executed serially in topological order with no concurrency between operators). You can set `ExecutionMode` to ORT_PARALLEL mode (multi-threaded parallel scheduling, where nodes without data dependencies can execute simultaneously on different threads).

**C API**

```c
OrtSessionOptions* options;
OrtCreateSessionOptions(&options);
OrtSetSessionExecutionMode(options, ORT_PARALLEL);  // ORT_SEQUENTIAL(0) / ORT_PARALLEL(1)
```

**C++ API**

```cpp
Ort::SessionOptions session_options;
session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
// session_options.execution_mode = ExecutionMode::ORT_PARALLEL;
auto session = Ort::Session(env, model_path, session_options);
```

**Python  API**

```python
import onnxruntime as ort

sess_options = ort.SessionOptions()
sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
session = ort.InferenceSession("model.onnx", sess_options)
```

## Change History

| Version | Date | Description |
| ---- | ---- | ---- |
| 01 | 2026-12-30 | This is the first official release. |
