// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.
#ifdef USE_KDNN_EP
#include <limits>

#include "core/graph/model.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/kdnn/fusion/attention_matcher.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime::test {
namespace {
// These graphs are resolved but are not run: some deliberately contain NaN
// scales, unknown shapes or already assigned nodes. Assert matcher decisions
// directly so CPU optimizer rewrites cannot hide the conditions under test.
class KdnnMatcherTest : public ::testing::TestWithParam<std::string> {};
TEST_P(KdnnMatcherTest, RequiresExclusiveAndProvableAttentionPattern) {
  const auto& scenario = GetParam();
  Model model("kdnn_matcher", false, {}, PathString{}, {}, {{kOnnxDomain, 13}}, {},
              DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);
  const int64_t heads = scenario == "OneHead" ? 1 : 2;
  InlinedVector<NodeArg*> reshaped;
  auto branch = [&](int64_t sequence, bool key) {
    const bool value = sequence == 3 && !key;
    const int64_t branch_heads =
        ((scenario == "BroadcastKeyHeads" && key) || (scenario == "BroadcastValueHeads" && value)) ? 1 : heads;
    const int64_t dimension = scenario == "DifferentValueDimension" && value ? 3 : 2;
    auto* input = builder.MakeInput<float>({1, sequence, branch_heads * dimension}, -1, 1);
    const std::vector<int64_t> dims{-1, sequence, branch_heads, dimension};
    auto* shape =
        scenario == "DynamicReshape" ? builder.MakeInput<int64_t>({4}, dims) : builder.Make1DInitializer<int64_t>(dims);
    auto* split = builder.MakeIntermediate();
    builder.AddNode("Reshape", {input, shape}, {split});
    reshaped.push_back(split);
    auto* transposed = builder.MakeIntermediate();
    auto permutation = key ? std::vector<int64_t>{0, 2, 3, 1} : std::vector<int64_t>{0, 2, 1, 3};
    if (scenario == "WrongPermutation" && sequence == 1) permutation = {0, 1, 2, 3};
    builder.AddNode("Transpose", {split}, {transposed}).AddAttribute("perm", permutation);
    return transposed;
  };
  auto* q = branch(1, false);
  auto* k = branch(3, true);
  auto* v = branch(3, false);
  auto* scores = builder.MakeIntermediate();
  Node& qk = builder.AddNode("MatMul", {q, k}, {scores});
  auto* scaled = builder.MakeIntermediate();
  float divisor = scenario == "ZeroDivisor" ? 0.0f : 2.0f;
  if (scenario == "NanScale") divisor = std::numeric_limits<float>::quiet_NaN();
  if (scenario == "ReciprocalOverflow") divisor = std::numeric_limits<float>::denorm_min();
  NodeArg* scale =
      scenario == "DynamicScale" ? builder.MakeInput<float>({}, {2.0f}) : builder.MakeScalarInitializer<float>(divisor);
  if (scenario == "NonScalarScale") scale = builder.MakeInitializer<float>({3}, {2, 2, 2});
  if (scenario == "SingletonTensorScale") scale = builder.MakeInitializer<float>({1, 1, 1, 1}, {2});
  builder.AddNode("Div", {scores, scale}, {scaled});
  if (scenario == "UnsupportedMask" || scenario == "DynamicTile" || scenario == "InvalidTileRepeats") {
    auto* mask = builder.MakeInput<float>({1, 1, 1, 3}, 0, 1);
    if (scenario != "UnsupportedMask") {
      auto* repeats = scenario == "DynamicTile" ? builder.MakeInput<int64_t>({4}, {1, heads, 1, 1})
                                                : builder.Make1DInitializer<int64_t>({1, 1, 1, 1});
      auto* tiled = builder.MakeIntermediate();
      builder.AddNode("Tile", {mask, repeats}, {tiled});
      mask = builder.MakeIntermediate();
      builder.AddNode("Sub", {builder.MakeScalarInitializer<float>(1), tiled}, {mask});
    }
    auto* masked = builder.MakeIntermediate();
    builder.AddNode("Add", {scaled, mask}, {masked});
    scaled = masked;
  }
  auto* probabilities = builder.MakeIntermediate();
  Node& softmax = builder.AddNode("Softmax", {scaled}, {probabilities});
  softmax.AddAttribute("axis", int64_t{-1});
  auto* output = builder.MakeOutput();
  builder.AddNode("MatMul", {probabilities, v}, {output});
  builder.SetGraphOutputs();
  ASSERT_TRUE(graph.Resolve().IsOK());
  if (scenario == "ProvenReshape" || scenario == "DynamicReshape" || scenario == "UnknownShape") {
    for (auto* split : reshaped) split->ClearShape();
  }
  if (scenario == "UnknownShape") {
    // Remove the Reshape proof by making its shape an overridable initializer.
    auto inputs = graph.GetInputs();
    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "Reshape") inputs.push_back(node.InputDefs()[1]);
    }
    graph.SetInputs(inputs);
    ASSERT_TRUE(graph.Resolve().IsOK());
    for (auto* split : reshaped) split->ClearShape();
  }
  if (scenario == "AssignedNode") qk.SetExecutionProviderType(kCpuExecutionProvider);
  if (scenario == "IntermediateOutput") graph.SetOutputs({output, scores});
  if (scenario == "SharedSoftmax") {
    auto* extra = builder.MakeOutput();
    builder.AddNode("Identity", {probabilities}, {extra});
    graph.SetOutputs({output, extra});
    ASSERT_TRUE(graph.Resolve().IsOK());
  }
  GraphViewer viewer(graph);
  auto match = kdnn::MatchAttention(viewer, softmax);
  const bool expected = scenario == "Baseline" || scenario == "ProvenReshape" || scenario == "SingletonTensorScale";
  ASSERT_EQ(match.has_value(), expected);
  if (match) {
    EXPECT_EQ(match->num_heads, 2);
    EXPECT_EQ(match->scale, 0.5f);
    EXPECT_EQ(match->inputs.size(), 3U);
    EXPECT_EQ(match->nodes.size(), 7U);
    EXPECT_FALSE(match->keep_mask);
  }
}

INSTANTIATE_TEST_SUITE_P(Safety, KdnnMatcherTest,
                         ::testing::Values("Baseline", "ProvenReshape", "DynamicReshape", "UnknownShape",
                                           "AssignedNode", "IntermediateOutput", "SharedSoftmax", "OneHead",
                                           "ZeroDivisor", "NanScale", "ReciprocalOverflow", "DynamicScale",
                                           "NonScalarScale", "SingletonTensorScale", "WrongPermutation",
                                           "UnsupportedMask", "DynamicTile", "InvalidTileRepeats", "BroadcastKeyHeads",
                                           "BroadcastValueHeads", "DifferentValueDimension"),
                         [](const auto& info) { return info.param; });
}  // namespace
}  // namespace onnxruntime::test
#endif
