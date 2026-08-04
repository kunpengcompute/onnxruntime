# API参考

## ONNX Runtime图优化

图优化通过环境变量开启，具体说明如[表1 图优化特性开关](#graph-optimization-features)所示。

**表 1**  图优化特性开关<a id="graph-optimization-features"></a>

| 接口名称                             | 接口类型   | 接口取值         | 接口功能                                  |
| --------------------------------- | ---- | ---------- | ----------------------------------- |
| ORT_ENABLE_RM_LN_FUSION           | 环境变量 | 1：开启  0：关闭 | 用于开启KPLayerNormalization的图融合策略。      |
| ORT_ENABLE_LAYER_NORM_NEON        | 环境变量 | 1：开启  0：关闭 | 用于开启KPLayerNormalization融合算子的Neon实现。 |
| ORT_ENABLE_FUSED_TENSORDOT_MATMUL | 环境变量 | 1：开启  0：关闭 | 用于开启KPFusedTensordotMatmul的图融合策略。    |

## ONNX Runtime并行调度优化

并行调度优化通过ONNX Runtime开源接口设置开启，默认为Sequential模式。

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

| 发布日期 | 修订记录 |
| ---- | ---- |
| 2026-09-30 | 第一次正式发布。 |
