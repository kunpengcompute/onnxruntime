# Release Notes

<!-- md-trans-meta sourceCommit=c020bdb62acb53e4224a27aecdcce6019c4e8cd5 translatedAt=2026-11-11T08:58:34.403Z pushedAt=2026-11-12T01:07:09.484Z -->

This document describes the version mapping information and updates of Kunpeng ONNX Runtime.

## Version Mapping

### Product Version Information

| | |
| ---- | ---- |
| **Product Name** | Kunpeng BoostKit |
| **Product Version** | 26.3.0 |
| **Software Name** | Kunpeng ONNX Runtime inference optimization |
| **Software Version** | V1.0.0 |

### OS and CPU Compatibility

| OS | CPU |
| ----------------------- | ------------------- |
| openEuler 24.03 LTS SP3 | Kunpeng 950 processor |

## Important Notes

### Precautions

See [Installation Guide](./installation_guide.md).

## V1.0.0

### Change Description

**New Features**

| Feature | Description |
| ----- | ----- |
| Parallel scheduling optimization | Adds a parallel scheduler (ParallelExecutor) to implement the inter-operator parallel scheduling mechanism. |
| Graph optimization | Adds five fused operators (LayerNormalization, FusedTensordotMatmul, FusedBatchMatmulAdd, FusedMaskedMul, and KdnnFusedAttention). |

**Modified Features**

None

**Removed Features**

None

### Resolved Issues

None

### Known Issues

None

## Related Documentation

### V1.0.0 Documents

| Document Name | Description | Delivery Form |
| ---- | ---- | ---- |
| Release Notes | Provides basic information and feature update information about each release of Kunpeng ONNX Runtime. | Open-source repository |
| Installation Guide | Provides instructions for compiling and installing Kunpeng ONNX Runtime. | Open-source repository |
| Feature Introduction | Describes the features and optimizations of Kunpeng ONNX Runtime based on open-source ONNX Runtime. | Open-source repository |
| Quick Start | Provides a quick start guide for Kunpeng ONNX Runtime. | Open-source repository |
| API Reference | Provides API usage instructions for Kunpeng ONNX Runtime. | Open-source repository |

### Obtaining Documentation

You can visit the [open-source repository](https://gitcode.com/boostkit/onnxruntime) to browse and obtain the related documents.
