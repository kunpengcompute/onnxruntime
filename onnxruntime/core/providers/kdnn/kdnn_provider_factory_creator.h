// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.
#pragma once
#include "core/framework/provider_options.h"
#include "core/providers/providers.h"

namespace onnxruntime {
struct KdnnProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(const ProviderOptions& options = {});
};
}  // namespace onnxruntime
