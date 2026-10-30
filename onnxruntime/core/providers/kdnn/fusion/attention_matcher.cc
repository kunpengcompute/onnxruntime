// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.
#include "core/providers/kdnn/fusion/attention_matcher.h"

#include <array>
#include <cmath>
#include <limits>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime::kdnn {
namespace {
// Match only the operator schema versions covered by the fusion tests.
bool IsSupportedOnnxOp(const Node* node, const char* name) {
  return node && node->Domain() == kOnnxDomain && node->OpType() == name && node->SinceVersion() == 13;
}

bool IsFloat(const NodeArg& arg) {
  const auto* type = arg.TypeAsProto();
  return type && type->has_tensor_type() && type->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto::FLOAT;
}

bool TryGetConstantScalar(const Graph& graph, const NodeArg& arg, float& value) {
  const auto* tensor = graph.GetConstantInitializer(arg.Name(), true);
  if (!tensor || tensor->data_type() != ONNX_NAMESPACE::TensorProto::FLOAT || tensor->dims_size() > 4) return false;
  for (auto dim : tensor->dims()) {
    if (dim != 1) return false;
  }
  Initializer init(graph, *tensor, graph.ModelPath());
  if (init.size() != 1) return false;
  value = init.data<float>()[0];
  return std::isfinite(value);
}

bool HasPermutation(const Node* node, const std::array<int64_t, 4>& perm) {
  if (!IsSupportedOnnxOp(node, "Transpose")) return false;
  auto it = node->GetAttributes().find("perm");
  return it != node->GetAttributes().end() && it->second.ints_size() == 4 &&
         std::equal(perm.begin(), perm.end(), it->second.ints().begin());
}

struct Branch {
  const Node* transpose;
  const NodeArg* input;
  int64_t sequence, heads, dim;
};

std::optional<std::array<int64_t, 3>> GetStaticInnerShape(const Graph& graph, const NodeArg& input,
                                                          const Node* producer) {
  const auto* shape = input.Shape();
  if (shape && shape->dim_size() == 4 && shape->dim(1).has_dim_value() && shape->dim(2).has_dim_value() &&
      shape->dim(3).has_dim_value()) {
    return std::array<int64_t, 3>{shape->dim(1).dim_value(), shape->dim(2).dim_value(), shape->dim(3).dim_value()};
  }
  // A constant [-1/positive, S, H, d] Reshape supplies missing shape information.
  // Reshape remains outside the fusion to validate the input size at runtime;
  // only non-overridable initializers can establish the output dimensions.
  if (!IsSupportedOnnxOp(producer, "Reshape")) return {};
  const auto* tensor = graph.GetConstantInitializer(producer->InputDefs()[1]->Name(), true);
  if (!tensor || tensor->data_type() != ONNX_NAMESPACE::TensorProto::INT64 || tensor->dims_size() != 1 ||
      tensor->dims(0) != 4)
    return {};
  Initializer init(graph, *tensor, graph.ModelPath());
  const auto dims = init.DataAsSpan<int64_t>();
  if (dims.size() != 4 || (dims[0] != -1 && dims[0] <= 0) || dims[1] <= 0 || dims[2] <= 0 || dims[3] <= 0) return {};
  return std::array<int64_t, 3>{dims[1], dims[2], dims[3]};
}

std::optional<Branch> MatchBranch(const Graph& graph, const Node& node, int slot, bool key) {
  const auto* transpose = graph_utils::GetInputNode(node, slot);
  if (!HasPermutation(transpose, key ? std::array<int64_t, 4>{0, 2, 3, 1} : std::array<int64_t, 4>{0, 2, 1, 3}))
    return {};
  const auto* input = transpose->InputDefs()[0];
  auto dims = GetStaticInnerShape(graph, *input, graph_utils::GetInputNode(*transpose, 0));
  if (!IsFloat(*input) || !dims || std::any_of(dims->begin(), dims->end(), [](auto d) { return d <= 0; })) return {};
  // L1 may rewrite the projection; matching depends on layout, not its OpType.
  return Branch{transpose, input, (*dims)[0], (*dims)[1], (*dims)[2]};
}

struct ScoreMatch {
  const Node* matmul;
  const Node* scale_node;
  float scale;
};

std::optional<ScoreMatch> MatchScores(const Graph& graph, const Node* scaled) {
  if (IsSupportedOnnxOp(scaled, "MatMul")) {
    return ScoreMatch{scaled, nullptr, 1.0f};
  }
  if ((!IsSupportedOnnxOp(scaled, "Mul") && !IsSupportedOnnxOp(scaled, "Div")) || scaled->InputDefs().size() != 2)
    return {};
  float constant;
  const bool divide = IsSupportedOnnxOp(scaled, "Div");
  const int constant_slot = TryGetConstantScalar(graph, *scaled->InputDefs()[1], constant) ? 1 : 0;
  if (constant_slot == 0 && (divide || !TryGetConstantScalar(graph, *scaled->InputDefs()[0], constant))) return {};
  const auto* matmul = graph_utils::GetInputNode(*scaled, 1 - constant_slot);
  if (!IsSupportedOnnxOp(matmul, "MatMul") || (divide && constant == 0)) return {};
  const float scale = divide ? 1.0f / constant : constant;
  if (!std::isfinite(scale)) return {};
  return ScoreMatch{matmul, scaled, scale};
}

struct KeepMaskMatch {
  const NodeArg* input = nullptr;
  std::vector<NodeIndex> nodes;
  float offset = 0;
  float multiplier = 1;
};

std::optional<KeepMaskMatch> MatchKeepMask(const Graph& graph, const Node* mul, int64_t heads, int64_t sk) {
  KeepMaskMatch match;
  const Node* sub = mul;
  // L1 can remove Mul(x, 1), leaving Sub with the default multiplier of 1.
  if (IsSupportedOnnxOp(mul, "Mul")) {
    if (mul->InputDefs().size() != 2) return {};
    const int constant_slot = TryGetConstantScalar(graph, *mul->InputDefs()[1], match.multiplier) ? 1 : 0;
    if (constant_slot == 0 && !TryGetConstantScalar(graph, *mul->InputDefs()[0], match.multiplier)) return {};
    sub = graph_utils::GetInputNode(*mul, 1 - constant_slot);
    match.nodes.push_back(mul->Index());
  }
  if (!IsSupportedOnnxOp(sub, "Sub") || sub->InputDefs().size() != 2 ||
      !TryGetConstantScalar(graph, *sub->InputDefs()[0], match.offset))
    return {};
  const NodeArg* mask = sub->InputDefs()[1];
  const auto* tile = graph_utils::GetInputNode(*sub, 1);
  const Node* mask_producer = tile;
  if (IsSupportedOnnxOp(tile, "Tile")) {
    const auto* tensor = graph.GetConstantInitializer(tile->InputDefs()[1]->Name(), true);
    if (!tensor || tensor->data_type() != ONNX_NAMESPACE::TensorProto::INT64 || tensor->dims_size() != 1 ||
        tensor->dims(0) != 4)
      return {};
    Initializer init(graph, *tensor, graph.ModelPath());
    auto repeats = init.DataAsSpan<int64_t>();
    if (repeats.size() != 4 || repeats[0] != 1 || repeats[1] != heads || repeats[2] != 1 || repeats[3] != 1) return {};
    mask = tile->InputDefs()[0];
    mask_producer = graph_utils::GetInputNode(*tile, 0);
    match.nodes.push_back(tile->Index());
  }
  auto dims = GetStaticInnerShape(graph, *mask, mask_producer);
  if (!IsFloat(*mask) || !dims || *dims != std::array<int64_t, 3>{1, 1, sk}) return {};
  match.nodes.push_back(sub->Index());
  match.input = mask;
  return match;
}

bool IsZeroMask(const Graph& graph, const NodeArg& arg, int64_t heads, int64_t sk) {
  // A constant zero bias needs no mask adapter if it leaves batch and query
  // dimensions unchanged. This also covers masks folded by L1.
  const auto* tensor = graph.GetConstantInitializer(arg.Name(), true);
  if (!tensor || tensor->data_type() != ONNX_NAMESPACE::TensorProto::FLOAT || tensor->dims_size() != 4 ||
      tensor->dims(0) != 1 || tensor->dims(2) != 1 || (tensor->dims(1) != 1 && tensor->dims(1) != heads) ||
      tensor->dims(3) != sk)
    return false;
  Initializer init(graph, *tensor, graph.ModelPath());
  const auto values = init.DataAsSpan<float>();
  return !values.empty() && std::all_of(values.begin(), values.end(), [](float value) { return value == 0.0f; });
}

bool HasExclusiveIntermediateNodes(const GraphViewer& viewer, const AttentionMatch& match, const Node& terminal) {
  const Graph& graph = viewer.GetGraph();
  InlinedHashSet<NodeIndex> unique;
  for (auto index : match.nodes) {
    const Node* node = viewer.GetNode(index);
    if (!node || !unique.insert(index).second || !node->GetExecutionProviderType().empty()) return false;
    if (node != &terminal && (graph.NodeProducesGraphOutput(*node) || node->GetOutputEdgesCount() != 1)) return false;
  }
  // Only the terminal may feed outside consumers or be a graph output.
  for (auto index : match.nodes) {
    const Node* node = viewer.GetNode(index);
    if (node != &terminal && !unique.count(node->OutputNodesBegin()->Index())) return false;
  }
  return true;
}
}  // namespace

std::optional<AttentionMatch> MatchAttention(const GraphViewer& viewer, const Node& softmax) {
  const Graph& graph = viewer.GetGraph();
  if (!IsSupportedOnnxOp(&softmax, "Softmax") || softmax.GetOutputEdgesCount() != 1) return {};
  auto axis = softmax.GetAttributes().find("axis");
  if (axis != softmax.GetAttributes().end() && axis->second.i() != -1 && axis->second.i() != 3) return {};
  const auto* av = &*softmax.OutputNodesBegin();
  if (!IsSupportedOnnxOp(av, "MatMul") || av->InputDefs()[0]->Name() != softmax.OutputDefs()[0]->Name()) return {};
  const auto* scores = graph_utils::GetInputNode(softmax, 0);
  std::optional<ScoreMatch> score_match;
  const Node* mask_node = nullptr;
  const NodeArg* mask_arg = nullptr;
  AttentionMatch match{};
  if (IsSupportedOnnxOp(scores, "Add")) {
    for (int score_slot : {0, 1}) {
      score_match = MatchScores(graph, graph_utils::GetInputNode(*scores, score_slot));
      if (score_match) {
        mask_node = graph_utils::GetInputNode(*scores, 1 - score_slot);
        mask_arg = scores->InputDefs()[1 - score_slot];
        break;
      }
    }
    match.nodes.push_back(scores->Index());
  } else {
    score_match = MatchScores(graph, scores);
  }
  if (!score_match) return {};
  match.scale = score_match->scale;
  auto q = MatchBranch(graph, *score_match->matmul, 0, false);
  auto k = MatchBranch(graph, *score_match->matmul, 1, true);
  auto v = MatchBranch(graph, *av, 1, false);
  if (!q || !k || !v || q->sequence != 1 || k->sequence != v->sequence || q->heads != k->heads ||
      q->heads != v->heads || q->dim != k->dim || q->dim != v->dim)
    return {};
  // Profitability policy, independent of the direct kernel's support for H=1.
  if (q->heads < 2 || q->heads > std::numeric_limits<int>::max() ||
      q->dim > std::numeric_limits<int64_t>::max() / q->heads)
    return {};
  match.num_heads = q->heads;
  match.inputs = {q->input->Name(), k->input->Name(), v->input->Name()};
  match.output = av->OutputDefs()[0]->Name();
  if (mask_arg && !IsZeroMask(graph, *mask_arg, q->heads, k->sequence)) {
    auto mask = MatchKeepMask(graph, mask_node, q->heads, k->sequence);
    if (!mask) return {};
    match.nodes.insert(match.nodes.end(), mask->nodes.begin(), mask->nodes.end());
    match.inputs.push_back(mask->input->Name());
    match.keep_mask = true;
    match.mask_offset = mask->offset;
    match.mask_multiplier = mask->multiplier;
  }
  match.nodes.insert(match.nodes.end(), {q->transpose->Index(), k->transpose->Index(), v->transpose->Index(),
                                         score_match->matmul->Index(), softmax.Index(), av->Index()});
  if (score_match->scale_node) match.nodes.push_back(score_match->scale_node->Index());
  if (!HasExclusiveIntermediateNodes(viewer, match, *av)) return {};
  return match;
}
}  // namespace onnxruntime::kdnn
