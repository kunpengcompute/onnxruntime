// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.
#include "core/providers/kdnn/kdnn_provider_options.h"

#include "core/common/common.h"
#include "core/platform/env.h"

namespace onnxruntime::kdnn {
namespace {
const char* AlgorithmName(AttentionAlgorithm algorithm) {
  switch (algorithm) {
    case AttentionAlgorithm::Auto:
      return "auto";
    case AttentionAlgorithm::Classic:
      return "classic";
    case AttentionAlgorithm::ClassicNoPack:
      return "classic_no_pack";
  }
  ORT_THROW("Unknown KDNN attention algorithm");
}
}  // namespace

KdnnProviderOptions KdnnProviderOptions::Parse(const ProviderOptions& options) {
  for (const auto& [key, value] : options) {
    ORT_ENFORCE(key == "enable_attention_fusion" || key == "attention_algorithm",
                "Unknown KDNN provider option: ", key);
  }
  auto get_option = [&](const char* key, const char* environment, const char* fallback) {
    auto it = options.find(key);
    if (it != options.end()) return it->second;
    auto value = Env::Default().GetEnvironmentVar(environment);
    return value.empty() ? std::string(fallback) : value;
  };
  KdnnProviderOptions result;
  const auto fusion = get_option("enable_attention_fusion", "ORT_KDNN_FUSE_ATTENTION", "1");
  ORT_ENFORCE(fusion == "0" || fusion == "1", "KDNN enable_attention_fusion must be 0 or 1");
  result.enable_attention_fusion = fusion == "1";
  const auto algorithm = get_option("attention_algorithm", "ORT_KDNN_ATTENTION_ALGORITHM", "auto");
  if (algorithm == "classic") {
    result.attention_algorithm = AttentionAlgorithm::Classic;
  } else if (algorithm == "classic_no_pack") {
    result.attention_algorithm = AttentionAlgorithm::ClassicNoPack;
  } else {
    ORT_ENFORCE(algorithm == "auto", "KDNN attention_algorithm must be auto, classic or classic_no_pack");
  }
  return result;
}

ProviderOptions KdnnProviderOptions::ToProviderOptions() const {
  return {{"enable_attention_fusion", enable_attention_fusion ? "1" : "0"},
          {"attention_algorithm", AlgorithmName(attention_algorithm)}};
}
}  // namespace onnxruntime::kdnn
