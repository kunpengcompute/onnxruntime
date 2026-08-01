# Onnxruntime介绍

## 最新消息

- [2026.09.30]：Onnxruntime优化补丁发布于Gitcode平台，聚焦于搜推广场景下的模型推理。

## 项目介绍

鲲鹏Onnxruntime基于开源Onnxruntime的高性能推理加速扩展，聚焦于搜推广推理场景下的高效执行。通过在图融合，算子执行，算子调度等方面进行了深度性能优化，显著提升了模型推理的吞吐量和时延表现，为AI应用提供基于鲲鹏CPU的极致性能。

## 特性介绍

| 特性名称   | 特性简介                                             |
| ------ | ------------------------------------------------ |
| 并行调度优化 | 改进算子调度算法，引入并行调度器实现算子间并行调度机制，有效提升了高并发场景下的模型推理吞吐量。 |
| 图优化    | 聚焦于通过计算图优化，调用高性能融合算子，加速模型的推理性能。                  |

## 目录结构

```bash
onnxruntime
├── patches
│   ├── 001-boostsra-onnxruntime-1.0.0-graph-optimization.patch     // onnxruntime图优化补丁文件
│   └── 002-boostsra-onnxruntime-1.0.0-parallel-executor.patch      // onnxruntime算子调度优化补丁文件
├── LICENSE                                                         // License文件
├── README.md                                                       // 项目介绍
└── docs                                                            // 文档
```

## 版本说明

关于Kunpeng Onnxruntime版本更新情况请参见《[版本说明书](./docs/zh/release_notes.md)》。

## 学习文档

| 学习资源类别 | 学习资源名称                                    | 学习资源简介                                        |
| ------ | ----------------------------------------- | --------------------------------------------- |
| 文档     | [版本说明书](./docs/zh/release_notes.md)       | 提供Kunpeng Onnxruntime每个发布版本的基础信息和特性更新信息。      |
| 文档     | [特性介绍](./docs/zh/feature_introduction.md) | 介绍Kunpeng Onnxruntime基于开源Onnxruntime的功能点和优化点。 |
| 文档     | [安装指南](./docs/zh/installation_guide.md)   | 提供Kunpeng Onnxruntime编译安装指导。                  |
| 文档     | [API参考](./docs/zh/api_reference.md)       | 提供Kunpeng Onnxruntime的接口使用说明。                 |

## 免责声明

此代码仓计划参与Onnxruntime社区开源，编码风格遵照原生开源软件，继承原生开源软件安全设计，不破坏原生开源软件设计及编码风格和方式，软件的任何漏洞与安全问题，均由相应的上游社区根据其漏洞和安全响应机制解决。请密切关注上游社区发布的通知和版本更新。鲲鹏计算社区对软件的漏洞及安全问题不承担任何责任。

## License

- 本项目采用MIT许可证授权，详见[LICENSE](./LICENSE)文件。

- 本项目文档适用CC-BY 4.0许可证，具体请参见[LICENSE](./docs/LICENSE)文件。

## 贡献声明

欢迎大家为社区做贡献，如果使用过程中有任何问题/建议，或者需要反馈特性需求和bug报告，可以提交[Issues](https://gitcode.com/boostkit/community/blob/master/docs/contributor/issue-submit.md)联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。同时也欢迎大家在[讨论专区](https://gitcode.com/boostkit/community/discussions)展开讨论交流。感谢您的支持。

## 致谢

Onnxruntime由华为公司的下列部门联合贡献：

- 鲲鹏计算Boostkit开发部

感谢来自社区的每一个PR，欢迎贡献Onnxruntime！