// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.
#include "core/providers/kdnn/fusion/attention_matcher.h"

namespace onnxruntime::kdnn {
std::unique_ptr<ComputeCapability> BuildAttentionCapability(AttentionMatch match, int version) {
  auto subgraph = std::make_unique<IndexedSubGraph>();
  subgraph->nodes = std::move(match.nodes);
  auto meta = std::make_unique<IndexedSubGraph::MetaDef>();
  meta->name = "KdnnFusedAttention";
  meta->domain = kKdnnDomain;
  meta->since_version = version;
  meta->status = ONNX_NAMESPACE::EXPERIMENTAL;
  meta->inputs = std::move(match.inputs);
  meta->outputs = {std::move(match.output)};
  auto add_float = [&](const char* name, float value) {
    auto& attr = meta->attributes[name];
    attr.set_name(name);
    attr.set_type(ONNX_NAMESPACE::AttributeProto::FLOAT);
    attr.set_f(value);
  };
  auto& heads = meta->attributes["num_heads"];
  heads.set_name("num_heads");
  heads.set_type(ONNX_NAMESPACE::AttributeProto::INT);
  heads.set_i(match.num_heads);
  // Always preserve the effective explicit scale, including bare QK scale=1.
  add_float("scale", match.scale);
  if (match.keep_mask) {
    auto& mode = meta->attributes["mask_mode"];
    mode.set_name("mask_mode");
    mode.set_type(ONNX_NAMESPACE::AttributeProto::STRING);
    mode.set_s("keep");
    add_float("mask_offset", match.mask_offset);
    add_float("mask_multiplier", match.mask_multiplier);
  }
  subgraph->schema_source = IndexedSubGraph::SourceOfSchema::EXISTING;
  subgraph->SetMetaDef(std::move(meta));
  return std::make_unique<ComputeCapability>(std::move(subgraph));
}
}  // namespace onnxruntime::kdnn
