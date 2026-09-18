# 版本说明书

## 版本配套说明

### 产品版本信息

<a name="table62675726"></a>

<table><tbody><tr id="row41561572"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.1.1"><p id="p15977216937"><a name="p11044137"></a><a name="p11044137"></a>产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.1.1 "><p id="p15977216913"><a name="p1597721693713"></a><a name="p1597721693713"></a>Kunpeng BoostKit</p>
</td>
</tr>
<tr id="row24726251"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.2.1"><p id="p159772169"><a name="p56669300"></a><a name="p56669300"></a>产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.2.1 "><p id="p15977216"><a name="p11923034"></a><a name="p11923034"></a><span id="text14311218114"><a name="text14311218114"></a><a name="text14311218114"></a>26.2.RC1</span></p>
</td>
</tr>
<tr id="row1930811171892"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.3.1"><p id="p1597726913"><a name="p2030912172097"></a><a name="p2030912172097"></a>软件名称</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.3.1 "><p id="p159216913"><a name="p1730912179911"></a><a name="p1730912179911"></a><span id="text17191017111119"><a name="text17191017111119"></a><a name="text17191017111119"></a>鲲鹏ONNX Runtime推理优化</span></p>
</td>
</tr>
<tr id="row19308111718"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.3.1"><p id="p977216913"><a name="p2030912172097"></a><a name="p2030912172097"></a>软件版本</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.3.1 "><p id="p5977216913"><a name="p1730912179911"></a><a name="p1730912179911"></a><span id="text17191017111119"><a name="text17191017111119"></a><a name="text17191017111119"></a>V1.0.0</span></p>
</td>
</tr>
</tbody>
</table>

### 与操作系统、编译器和CPU配套说明

| 操作系统                    | CPU类型          |
| ----------------------- | -------------- |
| openEuler 24.03 LTS SP3 | 鲲鹏950 7592C处理器 |

## 版本使用注意事项

### 使用注意事项

请参见《[安装指南](./installation_guide.md)》。

## V1.0.0

### 更新说明

**新增特性**

| 特性名称   | 更新说明                                                                    |
| ------ | ----------------------------------------------------------------------- |
| 并行调度优化 | 新增并行调度器（ParallelExecutor）实现算子间并行调度机制。                                   |
| 图优化    | 新增三个融合算子（KPLayerNormalization，KPFusedTensordotMatmul，KPFusedAttention）。 |

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

| 文档名称    | 内容简介                                      | 交付形式 |
| ------- | ----------------------------------------- | ---- |
| 《版本说明书》 | 提供鲲鹏ONNX Runtime每个发布版本的基础信息和特性更新信息。       | 开源仓  |
| 《安装指南》  | 提供鲲鹏ONNX Runtime编译安装指导。                   | 开源仓  |
| 《特性介绍》  | 介绍鲲鹏ONNX Runtime基于开源ONNX Runtime的功能点和优化点。 | 开源仓  |
| 《API参考》 | 提供鲲鹏ONNX Runtime的接口使用说明。                  | 开源仓  |

### 获取文档的方法

您可以通过访问[开源仓](https://gitcode.com/boostkit/onnxruntime)浏览和获取相关文档。
