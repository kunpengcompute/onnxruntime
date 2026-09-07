// Copyright 2025 Huawei Technologies Co., Ltd.
// Licensed under the Apache License, Version 2.0.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

// Structurally fuses scaled-dot-product multi-head attention subgraphs into a
// com.kdnn.internal::KdnnFusedAttention node backed by KDNN. The transformer
// recognizes the head layouts used by the Alipay models, applies a profitability
// gate to multi-head shapes, and is gated by ORT_KDNN_FUSE_ATTENTION for A/B tests.
class KdnnAttentionFusion : public GraphTransformer {
 public:
  explicit KdnnAttentionFusion(
      const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("KdnnAttentionFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level,
                   const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
