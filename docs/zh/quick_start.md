# 快速入门

本文档是鲲鹏ONNX Runtime的快速入门指南，面向已完成构建安装、希望尽快跑通推理的开发者。

## 安装鲲鹏ONNX Runtime

在使用ONNX Runtime进行推理服务前，请参见《[安装指南](./installation_guide.md)》完成鲲鹏ONNX Runtime的[构建安装](./installation_guide.md##编译安装Onnxruntime)。

## 运行推理服务

运行推理服务前，需要确保以下前置条件满足。

1. 当前环境已安装`Python 3.x`。

    ```bash
    python --version  # 预期输出对应Python版本
    ```

2. 当前环境已完成`onnxruntime`的编译安装。

    ```bash
    pip show onnxruntime  # 预期输出对应onnxruntime版本
    ```

3. 已准备待推理的模型文件，假设模型存放路径为`/path/to/model.onnx`。
4. 当前环境已安装相关依赖包`numpy`。

    ```bash
    pip show numpy  # 预期输出对应numpy版本
    ```

上述前置条件满足后，可以执行如下脚本对目标onnx模型进行推理。

```python
# ort_cpu_infer_demo.py

import numpy as np
import onnxruntime as ort
import os
import ctypes
import onnxruntime.capi.onnxruntime_pybind11_state as rtpy

# ---------- 工具函数 ----------
def _ort_type_to_np(ort_type: str) -> np.dtype:
    """把 ONNX 的 tensor 类型字符串映射到 numpy dtype。"""
    # ort_type 形如 "tensor(float)" / "tensor(int64)" / "tensor(string)"
    mapping = {
        "float":    np.float32,
        "double":   np.float64,
        "half":     np.float16,
        "bfloat16": np.float32,   # np 没有 bfloat16，临时用 float32 占位
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
    # 从 "tensor(float)" 里提取 "float"
    key = ort_type.replace("tensor(", "").rstrip(")")
    if key == "string":
        return None  # 字符串类型单独处理
    if key not in mapping:
        raise ValueError(f"[ERROR] 不支持的 ONNX dtype: {ort_type}")
    return mapping[key]

def _resolve_shape(raw_shape, batch_size: int) -> tuple:
    """
    把 ORT 原始 shape（list，可能含字符串动态维度）解析成 int tuple。
    动态维度（字符串 / None）统一替换成 batch_size。
    """
    resolved = []
    for d in raw_shape:
        if isinstance(d, int):
            resolved.append(d)
        else:  # 字符串（如 'unk_7517'）或 None
            resolved.append(batch_size)
    return tuple(resolved)

def _build_dummy_input(input_meta, batch_size: int) -> np.ndarray:
    """根据 ORT 的一个 input meta，生成符合 dtype 和 shape 的 dummy numpy 数组。"""
    shape = _resolve_shape(input_meta.shape, batch_size)
    ort_type = input_meta.type
    np_dtype = _ort_type_to_np(ort_type)

    if np_dtype is None:  # tensor(string)
        # 用空字符串数组占位；真实推理需替换成具体文本
        return np.full(shape, "", dtype=object)

    if np.issubdtype(np_dtype, np.floating):
        return np.zeros(shape, dtype=np_dtype)

    if np.issubdtype(np_dtype, np.integer):
        return np.zeros(shape, dtype=np_dtype)

    if np_dtype == np.bool_:
        return np.zeros(shape, dtype=np_dtype)

    raise ValueError(f"[ERROR] 未处理的 dtype: {np_dtype}")

# ---------- 主流程 ----------
def main():
    # 1. 解析命令行参数
    model_path = "/path/to/model.onnx"  # 待推理的onnx模型路径
    batch_size = 32       # batch_size大小，按照预期值配置

    ctypes.CDLL(
        os.path.join(os.path.dirname(ort.__file__), "capi", "libonnxruntime.so.1.26.0"),
        mode=ctypes.RTLD_GLOBAL)
    # /path/to/ANNC为ANNC源码目录
    rtpy.register_execution_provider_library(
        "AnncExecutionProvider", "/path/to/ANNC/install/lib64/libannc_ep.so")

    # 2. 创建会话（显式指定 CPU 提供者，让意图清晰）
    print(f"[INFO] 加载模型: {model_path}")
    session = ort.InferenceSession(
        model_path,
        providers=[
            ("KDNNExecutionProvider",{          # 调用KDNN EP
                "enable_attention_fusion": "1",  
                "attention_algorithm": "auto",
            }),
            "CPUExecutionProvider",
            "AnncExecutionProvider"             # 调用ANNC EP
        ])

    print(f"[INFO] 执行提供者: {session.get_providers()}")
    print(f"[INFO] 输入数量: {len(session.get_inputs())}")
    print(f"[INFO] 输出数量: {len(session.get_outputs())}")

    # 3. 自动构建所有输入（核心改动：不写死任何 input name/shape/dtype）
    input_feed = {}
    np.random.seed(1)
    for meta in session.get_inputs():
        arr = _build_dummy_input(meta, batch_size)
        input_feed[meta.name] = arr

    # 打印前 5 个输入的概要
    preview = list(input_feed.items())[:5]
    print(f"[INFO] 已构造 {len(input_feed)} 个 dummy 输入，前 5 个:")
    for name, arr in preview:
        print(f"      '{name}'  shape={arr.shape}  dtype={arr.dtype}")
    if len(input_feed) > 5:
        print(f"      ... （其余 {len(input_feed) - 5} 个略）")

    # 4. 执行推理
    print("[INFO] 开始推理（CPU）...")
    outputs = session.run(None, input_feed)

    # 5. 打印输出概要
    print(f"[INFO] 推理完成，共 {len(outputs)} 个输出:")
    for i, out in enumerate(outputs):
        print(f"      output[{i}]  shape={out.shape}  dtype={out.dtype}  "
              f"sample={out.flat[:5] if out.size > 0 else '<empty>'}")

    print("[OK] inference done ✓")

if __name__ == "__main__":
    main()
```

将上述脚本保存为`ort_cpu_infer_demo.py`，执行如下命令使能所有优化特性进行推理。

```bash
export ANNC_EP_FUSION_LAYER_NORM=1
export ANNC_EP_FUSION_TENSORDOT_MATMUL=1
export ANNC_EP_FUSION_MASKED_MUL=1
export ANNC_EP_FUSION_BATCHED_MATMUL_ADD=1
export ORT_ENABLE_PARALLEL_EXECUTOR=1
python ort_cpu_infer_demo.py
```

推理完成后预期输出如下。

```output
[INFO] 加载模型: /path/to/model.onnx
[INFO] 执行提供者: ['KDNNExecutionProvider', 'CPUExecutionProvider', 'AnncExecutionProvider']
[INFO] 输入数量: **
[INFO] 输出数量: **
[INFO] 已构造 ** 个 dummy 输入，前 5 个:
      ...
      ... （其余 ** 个略）
[INFO] 开始推理（CPU）...
[INFO] 推理完成，共 ** 个输出:
      output[0]  ...
      ...
[OK] inference done ✓
```

## 修订记录

| 文档版本 | 发布日期 | 修订记录 |
| ----- | ----- | ----- |
| 01 | 2026-12-30 | 第一次正式发布。 |
