// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.
#include "core/providers/kdnn/kdnn_kernel_registry.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime::kdnn {
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kKdnnExecutionProvider, kKdnnDomain, 1, 1, KdnnFusedAttention);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kKdnnExecutionProvider, kKdnnDomain, 2, 2, KdnnFusedAttention);

std::shared_ptr<KernelRegistry> RegisterKernels() {
  auto registry = std::make_shared<KernelRegistry>();
  ORT_THROW_IF_ERROR(registry->Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
                                            kKdnnExecutionProvider, kKdnnDomain, 1, 1, KdnnFusedAttention)>()));
  ORT_THROW_IF_ERROR(registry->Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
                                            kKdnnExecutionProvider, kKdnnDomain, 2, 2, KdnnFusedAttention)>()));
  return registry;
}
}  // namespace onnxruntime::kdnn
