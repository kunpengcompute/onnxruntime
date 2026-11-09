// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.
#include "core/providers/kdnn/attention/attention_adapter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <arm_neon.h>
#include "core/common/safeint.h"
#include "kdnn.hpp"

namespace onnxruntime::kdnn {
namespace {
double MaxMagnitude(const float* data, size_t count) {
  // FMAX (not FMAXNM) propagates NaN; check once after the vector reduction.
  float32x4_t maximum4 = vdupq_n_f32(0);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    maximum4 = vmaxq_f32(maximum4, vabsq_f32(vld1q_f32(data + i)));
  }
  float maximum = vmaxvq_f32(maximum4);
  if (!std::isfinite(maximum)) return std::numeric_limits<double>::infinity();
  for (; i < count; ++i) {
    if (!std::isfinite(data[i])) return std::numeric_limits<double>::infinity();
    maximum = std::max(maximum, std::abs(data[i]));
  }
  return maximum;
}

// v2 uses ONNX Softmax semantics for nonfinite scores. Bound intermediate
// products, scaling and addition to detect possible overflow from finite inputs.
bool RequiresOnnxComputation(const AttentionParameters& parameters, const float* query, const float* key,
                             const float* mask) {
  const size_t hidden_size = SafeInt<size_t>(parameters.num_heads) * parameters.head_size;
  const double query_max = MaxMagnitude(query, SafeInt<size_t>(parameters.query_length) * hidden_size);
  const double key_max = MaxMagnitude(key, SafeInt<size_t>(parameters.key_length) * hidden_size);
  const double mask_max = mask ? MaxMagnitude(mask, parameters.key_length) : 0;
  const double dot_bound = query_max * key_max * static_cast<double>(parameters.head_size);
  const double scale = std::abs(static_cast<double>(parameters.scale));
  const double score_bound = dot_bound * scale + mask_max;
  const double limit = std::numeric_limits<float>::max() / 4.0;
  return !std::isfinite(score_bound) || dot_bound > limit || score_bound > limit || query_max * scale > limit;
}

// Use the CPU MatMul layouts and math routines: separate multiply/add can
// overflow where fused multiply-add remains finite. The caller parallelizes
// batches, so these routines run without an inner thread pool.
void RunOnnxAttention(const AttentionParameters& parameters, const float* query, const float* key, const float* value,
                      const float* mask, float* output) {
  const size_t query_length = parameters.query_length;
  const size_t key_length = parameters.key_length;
  const size_t head_size = parameters.head_size;
  const size_t hidden_size = SafeInt<size_t>(parameters.num_heads) * head_size;
  std::vector<float> scores(SafeInt<size_t>(query_length) * key_length);
  std::vector<float> transposed_key(SafeInt<size_t>(key_length) * head_size);
  std::vector<float> packed_value(transposed_key.size());

  for (size_t head = 0; head < parameters.num_heads; ++head) {
    const size_t head_offset = head * head_size;
    for (size_t token = 0; token < key_length; ++token) {
      for (size_t dimension = 0; dimension < head_size; ++dimension) {
        const size_t source = token * hidden_size + head_offset + dimension;
        transposed_key[dimension * key_length + token] = key[source];
        packed_value[token * head_size + dimension] = value[source];
      }
    }
    MlasGemm(CblasNoTrans, CblasNoTrans, query_length, key_length, head_size, 1.0f, query + head_offset, hidden_size,
             transposed_key.data(), key_length, 0.0f, scores.data(), key_length, nullptr, &parameters.mlas_config);
    for (size_t row = 0; row < query_length; ++row) {
      for (size_t token = 0; token < key_length; ++token) {
        float& score = scores[row * key_length + token];
        score *= parameters.scale;
        if (mask) score += mask[token];
      }
    }
    MlasComputeSoftmax(scores.data(), scores.data(), query_length, key_length, false, false, 0.0f, nullptr);
    MlasGemm(CblasNoTrans, CblasNoTrans, query_length, head_size, key_length, 1.0f, scores.data(), key_length,
             packed_value.data(), head_size, 0.0f, output + head_offset, hidden_size, nullptr, &parameters.mlas_config);
  }
}
}  // namespace

void RunAttentionBatch(const AttentionParameters& parameters, const float* query, const float* key, const float* value,
                       const float* mask, float* output) {
  thread_local std::vector<float> mask_scratch;
  if (parameters.keep_mask) {
    if (mask_scratch.size() < parameters.key_length) mask_scratch.resize(parameters.key_length);
    for (size_t i = 0; i < parameters.key_length; ++i) {
      const float difference = parameters.keep_mask->offset - mask[i];
      mask_scratch[i] = parameters.keep_mask->multiplier * difference;
    }
    mask = mask_scratch.data();
  }
  if (parameters.preserve_onnx_softmax && RequiresOnnxComputation(parameters, query, key, mask)) {
    RunOnnxAttention(parameters, query, key, value, mask, output);
    return;
  }
  KDNN::MultiHeadAttentionOptions options;
  if (parameters.algorithm == AttentionAlgorithm::Classic) options.algorithm = KDNN::AttentionAlgorithm::CLASSIC;
  if (parameters.algorithm == AttentionAlgorithm::ClassicNoPack)
    options.algorithm = KDNN::AttentionAlgorithm::CLASSIC_NO_PACK;
  KDNN::MultiHeadAttention op(1, parameters.query_length, parameters.key_length, parameters.num_heads,
                              parameters.head_size, parameters.scale, true, options);
  thread_local std::vector<float> workspace;
  const size_t bytes = op.GetWorkspaceSize();
  const size_t elements = bytes / sizeof(float) + (bytes % sizeof(float) != 0);
  if (workspace.size() < elements) workspace.resize(elements);
  op.Run(query, key, value, mask, output, workspace.empty() ? nullptr : workspace.data());
}
}  // namespace onnxruntime::kdnn
