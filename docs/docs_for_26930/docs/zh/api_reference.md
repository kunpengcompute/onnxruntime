# API参考

## ONNX Runtime图优化

图优化通过环境变量开启，具体说明如[表1 图优化特性开关](#graph-optimization-features)所示。

**表 1**  图优化特性开关<a id="graph-optimization-features"></a>

| 接口名称                                | 接口类型 | 接口取值                        | 接口功能                                                                              |
| ----------------------------------- | ---- | --------------------------- | --------------------------------------------------------------------------------- |
| ORT_ENABLE_RM_LN_FUSION             | 环境变量 | 1：开启  0：关闭                  | 用于开启LayerNormalization的图融合策略。                                                     |
| ORT_ENABLE_LAYER_NORM_NEON          | 环境变量 | 1：开启  0：关闭                  | 用于开启LayerNormalization融合算子的Neon实现。                                                |
| ORT_ENABLE_FUSED_TENSORDOT_MATMUL   | 环境变量 | 1：开启  0：关闭                  | 用于开启FusedTensordotMatmul的图融合策略。                                                   |
| ORT_KDNN_FUSE_ATTENTION             | 环境变量 | 1：开启  0：关闭                  | 用于开启FusedAttention的图融合策略。                                                         |
| ORT_KDNN_ATTENTION_ALGORITHM        | 环境变量 | classic（默认）/classic_no_pack | 用于控制是否对Q/K/V进行pack。                                                               |
| ORT_KDNN_ATTENTION_INTERLEAVED_GEMV | 环境变量 | all（默认）/qk/pv/off           | 用于控制QK 和 PV 阶段是否使用特化Kernel，qk只在只在 QK^T 阶段用，pv只在 probs·V 阶段用，all两个阶段都用，off两个阶段都不用。 |

**使用示例：**

```bash
# 开启全部优化
export ORT_ENABLE_RM_LN_FUSION=1
export ORT_ENABLE_LAYER_NORM_NEON=1
export ORT_ENABLE_FUSED_TENSORDOT_MATMUL=1
export ORT_KDNN_FUSE_ATTENTION=1
export ORT_KDNN_ATTENTION_ALGORITHM=classic_no_pack
export ORT_KDNN_ATTENTION_INTERLEAVED_GEMV=all
# 运行推理脚本
python onnxruntime_demo.py
```

## ONNX Runtime并行调度优化

并行调度优化通过ONNX Runtime开源接口设置开启，默认为Sequential模式（算子间按拓扑序串行执行每个节点，无并发），可通过设置`ExecutionMode`改为ORT_PARALLEL模式（多线程并行调度，无数据依赖的节点可同时在不同线程上执行）。

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

| 发布日期       | 修订记录     |
| ---------- | -------- |
| 2026-09-30 | 第一次正式发布。 |
