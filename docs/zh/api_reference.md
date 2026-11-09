# API参考

本文档介绍ONNX Runtime图优化特性接口与ONNX Runtime并行调度优化特性接口的具体信息与调用方式，面向需要调优推理性能的应用开发人员。

## ONNX Runtime图优化特性接口

图优化通过环境变量开启，环境变量用于全局开启。具体说明如[表1 图优化特性接口介绍](#graph-optimization-features)所示。

**表 1** 图优化特性接口介绍<a id="graph-optimization-features"></a>

| 接口名称 | 接口类型 | 接口取值 | 接口功能 |
| ---- | ---- | ---- | ---- |
| ANNC_EP_FUSION_LAYER_NORM | 环境变量 | 1：开启  0：关闭（默认） | 用于开启ANNC LayerNorm的图融合策略。 |
| ANNC_EP_FUSION_TENSORDOT_MATMUL | 环境变量 | 1：开启  0：关闭（默认） | 用于开启ANNC TensordotMatmul的图融合策略。|
| ANNC_EP_FUSION_MASKED_MUL | 环境变量 | 1：开启  0：关闭（默认） | 用于开启ANNC MaskedMul的图融合策略。|
| ANNC_EP_FUSION_BATCHED_MATMUL_ADD | 环境变量 | 1：开启  0：关闭（默认） | 用于开启ANNC BatchedMatmulAdd的图融合策略。|
| ORT_KDNN_FUSE_ATTENTION | 环境变量 | 1：开启  0：关闭（默认） | 用于开启KDNN FuseAttention的图融合策略。|
| ORT_KDNN_ATTENTION_ALGORITHM | 环境变量 | auto（默认）/classic/classic_no_pack | 用于控制是否对Q/K/V进行pack。|

### KDNN的图融合策略接口

KDNN FuseAttention的图融合策略以及Attention算法也可以在创建SessionOptions时通过API显式指定，且显式指定的优先级高于环境变量。

- 通过C++指定API的命令如下。

    ```cpp
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    options.AppendExecutionProvider("KDNN", {{"enable_attention_fusion", "1"},
                                            {"attention_algorithm", "auto"}});
    ```

- 通过Python指定API的命令如下。

    ```python
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    session = ort.InferenceSession(
        model_path,   # model_path为待推理的onnx模型路径
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

### ANNC的图融合策略接口

ANNC的图融合策略需要在创建SessionOptions时显式指定ANNC API，对应的API如下：

- 通过C++指定API的命令如下。

    ```cpp
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    options.AppendExecutionProvider("ANNC");
    ```

- 通过Python指定API的命令如下。

    ```python
    import os
    import onnxruntime as ort
    import ctypes
    import onnxruntime.capi.onnxruntime_pybind11_state as rtpy

    ctypes.CDLL(
        os.path.join(os.path.dirname(ort.__file__), "capi", "libonnxruntime.so.1.26.0"),
        mode=ctypes.RTLD_GLOBAL)
    rtpy.register_execution_provider_library(
        "AnncExecutionProvider", "<ANNC_PREFIX>/lib64/libannc_ep.so")  # 绝对路径；ANNC 构建产物

    session = ort.InferenceSession(
        model_path,   # model_path为待推理的onnx模型路径
        options,
        providers=["AnncExecutionProvider", "CPUExecutionProvider"],
    )
    ```

## ONNX Runtime并行调度优化

并行调度优化通过环境变量和ONNX Runtime开源接口设置开启，环境变量接口说明如[表2 并行调度优化特性接口介绍](#parallel-executor-features)所示，并行调度优化需要同时开启环境变量和设置ONNX Runtime开源接口才会生效。

**表 2** 并行调度优化特性接口介绍<a id="parallel-executor-features"></a>

| 接口名称  | 接口类型 | 接口取值 | 接口功能 |
| ----- | ---- | ---- | ---- |
| ORT_ENABLE_PARALLEL_EXECUTOR | 环境变量 | 1：开启  0：关闭（默认值） | 用于开启并行调度优化策略。 |

ONNX Runtime开源接口默认为Sequential模式（算子间按拓扑序串行执行每个节点，无并发），可通过设置`ExecutionMode`改为ORT_PARALLEL模式（多线程并行调度，无数据依赖的节点可同时在不同线程上执行）。

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

## 修订记录

| 文档版本 | 发布日期 | 修订记录 |
| ---------- | ---------- | -------- |
| 01 | 2026-12-30 | 第一次正式发布。 |
