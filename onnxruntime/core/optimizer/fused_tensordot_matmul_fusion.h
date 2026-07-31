// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class FusedTensordotMatMulFusion : public GraphTransformer {
 public:
  FusedTensordotMatMulFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("FusedTensordotMatMulFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  static int Fuse(Node& reshape, Graph& graph, const logging::Logger& logger);
};

}  // namespace onnxruntime
