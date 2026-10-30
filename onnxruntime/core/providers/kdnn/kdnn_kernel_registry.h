// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.
#pragma once
#include <memory>
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime::kdnn {
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();
std::shared_ptr<KernelRegistry> RegisterKernels();
}  // namespace onnxruntime::kdnn
