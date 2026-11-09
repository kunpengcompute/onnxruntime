# 版本说明书

本文档介绍鲲鹏ONNX Runtime的版本配套信息与更新内容。

## 版本配套说明

### 产品版本信息

| | |
| ---- | ---- |
| **产品名称** | Kunpeng BoostKit |
| **产品版本** | 26.3.0 |
| **软件名称** | 鲲鹏ONNX Runtime推理优化 |
| **软件版本** | V1.0.0 |

### 操作系统和CPU配套说明

| 操作系统                | CPU类型             |
| ----------------------- | ------------------- |
| openEuler 24.03 LTS SP3 | 鲲鹏950 7592C处理器 |

## 版本使用注意事项

### 使用注意事项

请参见《[安装指南](./installation_guide.md)》。

## V1.0.0

### 更新说明

**新增特性**

| 特性名称 | 更新说明 |
| ----- | ----- |
| 并行调度优化 | 新增并行调度器（ParallelExecutor）实现算子间并行调度机制。  |
| 图优化 | 新增五个融合算子（LayerNormalization，FusedTensordotMatmul，FusedBatchMatmulAdd，FusedMaskedMul，KdnnFusedAttention）。 |

**修改特性**

无

**删除特性**

无

### 已解决的问题

无

### 遗留问题

无

## 版本配套文档

### V1.0.0版本配套文档

| 文档名称 | 内容简介 | 交付形式 |
| ---- | ---- | ---- |
| 《版本说明书》 | 提供鲲鹏ONNX Runtime每个发布版本的基础信息和特性更新信息。 | 开源仓 |
| 《安装指南》 | 提供鲲鹏ONNX Runtime编译安装指导。 | 开源仓 |
| 《特性介绍》 | 介绍鲲鹏ONNX Runtime基于开源ONNX Runtime的功能点和优化点。 | 开源仓 |
| 《快速入门》 | 提供鲲鹏ONNX Runtime快速入门指导。 | 开源仓 |
| 《API参考》 | 提供鲲鹏ONNX Runtime的接口使用说明。 | 开源仓 |

### 获取文档的方法

您可以通过访问[开源仓](https://gitcode.com/boostkit/onnxruntime)浏览和获取相关文档。
