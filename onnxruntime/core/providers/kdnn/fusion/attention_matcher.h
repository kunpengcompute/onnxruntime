// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.
#pragma once
#include <optional>
#include <string>
#include <vector>
#include "core/framework/compute_capability.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime::kdnn {
struct AttentionMatch {
  std::vector<NodeIndex> nodes;
  std::vector<std::string> inputs;
  std::string output;
  int64_t num_heads;
  float scale;
  bool keep_mask = false;
  float mask_offset = 0;
  float mask_multiplier = 0;
};

// Pure structural/semantic match. Every removed intermediate is private.
std::optional<AttentionMatch> MatchAttention(const GraphViewer& graph, const Node& softmax);
std::unique_ptr<ComputeCapability> BuildAttentionCapability(AttentionMatch match, int version);
}  // namespace onnxruntime::kdnn
