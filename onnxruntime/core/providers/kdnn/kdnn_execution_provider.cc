// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.
#include "core/providers/kdnn/kdnn_execution_provider.h"

#include <asm/hwcap.h>
#include <sys/auxv.h>
#include "core/framework/compute_capability.h"
#include "core/framework/session_options.h"
#include "core/optimizer/graph_optimizer_registry.h"
#include "core/providers/kdnn/fusion/attention_matcher.h"
#include "core/providers/kdnn/kdnn_kernel_registry.h"

namespace onnxruntime {
KDNNExecutionProvider::KDNNExecutionProvider(kdnn::KdnnProviderOptions options)
    : IExecutionProvider{kKdnnExecutionProvider}, options_{std::move(options)} {
  ORT_ENFORCE((getauxval(AT_HWCAP) & HWCAP_SVE) != 0,
              "KDNN EP requires SVE CPU capabilities and a compatible THREADPOOL libkdnn");
}

std::shared_ptr<KernelRegistry> KDNNExecutionProvider::GetKernelRegistry() const {
  static auto registry = kdnn::RegisterKernels();
  return registry;
}

std::vector<std::unique_ptr<ComputeCapability>> KDNNExecutionProvider::GetCapability(
    const GraphViewer& graph, const IKernelLookup& kernel_lookup,
    const GraphOptimizerRegistry& optimizer_registry, IResourceAccountant*) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  InlinedHashSet<NodeIndex> occupied;
  for (const Node& node : graph.Nodes()) {
    if (!node.GetExecutionProviderType().empty() && node.GetExecutionProviderType() != Type()) continue;
    if (node.Domain() == kKdnnDomain && node.OpType() == "KdnnFusedAttention" &&
        node.SinceVersion() >= 1 && node.SinceVersion() <= 2 && kernel_lookup.LookUpKernel(node)) {
      auto subgraph = std::make_unique<IndexedSubGraph>();
      subgraph->nodes.push_back(node.Index());
      occupied.insert(node.Index());
      result.push_back(std::make_unique<ComputeCapability>(std::move(subgraph)));
    }
  }
  if (!options_.enable_attention_fusion ||
      optimizer_registry.GetSessionOptions().graph_optimization_level < TransformerLevel::Level2) return result;
  const auto domain = graph.DomainToVersionMap().find(kKdnnDomain);
  if (domain == graph.DomainToVersionMap().end() || domain->second != 2) return result;
  for (auto index : graph.GetNodesInTopologicalOrder()) {
    if (occupied.count(index)) continue;
    auto match = kdnn::MatchAttention(graph, *graph.GetNode(index));
    if (!match) continue;
    if (std::any_of(match->nodes.begin(), match->nodes.end(), [&](auto n) { return occupied.count(n); })) continue;
    occupied.insert(match->nodes.begin(), match->nodes.end());
    result.push_back(kdnn::BuildAttentionCapability(std::move(*match), domain->second));
  }
  return result;
}
}  // namespace onnxruntime
