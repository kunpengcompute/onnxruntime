# Introduction to Kunpeng ONNX Runtime

<!-- md-trans-meta sourceCommit=c020bdb62acb53e4224a27aecdcce6019c4e8cd5 translatedAt=2026-11-11T08:58:34.403Z pushedAt=2026-11-12T01:07:09.484Z -->

English|[简体中文](./README.md)

## Latest News

[2026-12-30]: Released the Kunpeng ONNX Runtime optimization patch version 1.0.0, which added parallel scheduling optimization and graph optimization features.

## Project Introduction

Kunpeng ONNX Runtime is a high-performance inference acceleration extension based on open-source ONNX Runtime, focusing on efficient execution in search, recommendation, and advertising inference scenarios. With in-depth performance optimization in graph fusion, operator execution, and operator scheduling, it significantly improves the throughput and latency of model inference, providing AI applications with ultimate performance based on Kunpeng processors.

## Feature Introduction

| Feature | Description |
| ------ | ------ |
| Parallel scheduling optimization | Improves the operator scheduling algorithm and introduces a parallel scheduler to implement the inter-operator parallel scheduling mechanism, effectively improving model inference throughput in high-concurrency scenarios. |
| Graph optimization | Focuses on optimizing the computational graph and invoking high-performance fused operators to accelerate model inference. |

For more details about features, see [Feature Introduction](./docs/en/feature_introduction.md).

## Directory Structure

```text
onnxruntime
├── patches
│   ├── 001-boostsra-onnxruntime-1.0.0-graph-optimization.patch     // ONNX Runtime graph optimization patch file
│   └── 002-boostsra-onnxruntime-1.0.0-parallel-executor.patch      // ONNX Runtime operator scheduling optimization patch file
├── LICENSE                                                         // License file
├── README.md                                                       // Project introduction
└── docs                                                            // Documents
    ├── en
       ├── api_reference.md                                         // API Reference
       ├── feature_introduction.md                                  // Feature Introduction
       ├── installation_guide.md                                    // Installation Guide
       ├── quick_start.md                                           // Quick Start
       └── release_notes.md                                         // Release Notes
```

## Release Notes

For details about version updates of Kunpeng ONNX Runtime, see [Release Notes](./docs/en/release_notes.md).

## Learning Documents

| Learning Resource | Description |
| ------------ | --------------------------------- |
| [Release Notes](./docs/en/release_notes.md) | Provides basic information and feature update information about each release of Kunpeng ONNX Runtime. |
| [Feature Introduction](./docs/en/feature_introduction.md) | Describes the features and optimizations of Kunpeng ONNX Runtime based on open-source ONNX Runtime. |
| [Installation Guide](./docs/en/installation_guide.md) | Provides instructions for building and installing Kunpeng ONNX Runtime. |
| [Quick Start](./docs/en/quick_start.md) | Provides a quick start guide for Kunpeng ONNX Runtime. |
| [API Reference](./docs/en/api_reference.md) | Provides API usage instructions for Kunpeng ONNX Runtime. |

## Disclaimer

This code repository is planned to contribute to the ONNX Runtime community open-source project. It strictly adheres to the coding style and methods, as well as the security design, of open-source software. Any vulnerabilities and security issues of the software shall be resolved by the corresponding upstream communities according to their response mechanisms. Please pay attention to the notifications and version updates released by the upstream communities. The Kunpeng computing community does not assume any responsibility for software vulnerabilities and security issues.

## License

- This project is licensed under the MIT License. For details, see [LICENSE](./LICENSE).
- The documents of this project are licensed under the CC-BY 4.0 License. For details, see [LICENSE](./docs/LICENSE).

## Contribution

Contributions to the community are welcome. If you have any questions or suggestions during use, or need to report feature requests or bugs, submit an [issue](https://gitcode.com/boostkit/community/blob/master/docs/contributor/issue-submit.md) to contact us. For details about how to contribute, see the [contribution guide](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md). You are also welcome to join the discussions in [Discussions](https://gitcode.com/boostkit/community/discussions). Thank you for your support.

## Acknowledgments

Thank you for every PR from the community. Contributions to Kunpeng ONNX Runtime are welcome!
