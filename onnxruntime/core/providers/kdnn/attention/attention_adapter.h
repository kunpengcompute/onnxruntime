// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.
#pragma once
#include <cstddef>
#include <optional>

#include "core/mlas/inc/mlas.h"
#include "core/providers/kdnn/kdnn_provider_options.h"

namespace onnxruntime::kdnn {
struct KeepMaskTransform {
  float offset;
  float multiplier;
};

struct AttentionParameters {
  size_t query_length = 0;
  size_t key_length = 0;
  size_t num_heads = 0;
  size_t head_size = 0;
  float scale = 1.0f;
  AttentionAlgorithm algorithm = AttentionAlgorithm::Auto;
  bool preserve_onnx_softmax = false;
  std::optional<KeepMaskTransform> keep_mask;
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG mlas_config;
};

// Invoked once per output batch by the ORT threadpool. Scratch is worker-local.
void RunAttentionBatch(const AttentionParameters& parameters, const float* query, const float* key, const float* value,
                       const float* mask, float* output);
}  // namespace onnxruntime::kdnn
