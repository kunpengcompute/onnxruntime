// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.
#pragma once

#include "core/framework/execution_provider.h"
#include "core/providers/kdnn/kdnn_provider_options.h"

namespace onnxruntime {
// Uses the Session's CPU allocator without overriding CreatePreferredAllocators.
// A plain CPUAllocator registered here would take priority over the CPU EP arena
// and disable allocation reuse even for nodes that remain on CPU.
class KDNNExecutionProvider final : public IExecutionProvider {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(KDNNExecutionProvider);
  explicit KDNNExecutionProvider(kdnn::KdnnProviderOptions options);
  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const GraphViewer& graph, const IKernelLookup& kernel_lookup,
      const GraphOptimizerRegistry& optimizer_registry,
      IResourceAccountant* resource_accountant) const override;
  FusionStyle GetFusionStyle() const override { return FusionStyle::FilteredGraphViewer; }
  ProviderOptions GetProviderOptions() const override { return options_.ToProviderOptions(); }
  const kdnn::KdnnProviderOptions& Options() const { return options_; }

 private:
  const kdnn::KdnnProviderOptions options_;
};
}  // namespace onnxruntime
