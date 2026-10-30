// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.
#include "core/providers/kdnn/kdnn_provider_factory_creator.h"
#include "core/providers/kdnn/kdnn_execution_provider.h"

namespace onnxruntime {
namespace {
struct KdnnProviderFactory final : IExecutionProviderFactory {
  explicit KdnnProviderFactory(const ProviderOptions& options)
      : options_{kdnn::KdnnProviderOptions::Parse(options)} {}
  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<KDNNExecutionProvider>(options_);
  }
  const kdnn::KdnnProviderOptions options_;
};
}  // namespace
std::shared_ptr<IExecutionProviderFactory> KdnnProviderFactoryCreator::Create(const ProviderOptions& options) {
  return std::make_shared<KdnnProviderFactory>(options);
}
}  // namespace onnxruntime
