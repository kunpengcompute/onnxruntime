// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.
#pragma once

#include "core/framework/provider_options.h"

namespace onnxruntime::kdnn {
enum class AttentionAlgorithm { Auto,
                                Classic,
                                ClassicNoPack };

// The provider factory parses environment defaults once and stores these options
// as const. Each EP receives a copy of that configuration.
struct KdnnProviderOptions {
  bool enable_attention_fusion = true;
  AttentionAlgorithm attention_algorithm = AttentionAlgorithm::Auto;
  static KdnnProviderOptions Parse(const ProviderOptions& options);
  ProviderOptions ToProviderOptions() const;
};
}  // namespace onnxruntime::kdnn
