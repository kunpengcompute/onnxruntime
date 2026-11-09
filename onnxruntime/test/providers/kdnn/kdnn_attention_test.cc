// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.

#if defined(USE_KDNN_EP)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "core/graph/constants.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/providers/kdnn/kdnn_provider_factory_creator.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime {
namespace test {
namespace {
void RunKdnnTest(OpTester& test, OpTester::ExpectResult expected = OpTester::ExpectResult::kExpectSuccess,
                 const std::string& message = "") {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultKdnnExecutionProvider());
  test.Run(expected, message, {}, nullptr, &providers);
}

size_t Offset(size_t batch, size_t sequence, size_t head, size_t element,
              size_t sequence_size, size_t num_heads, size_t head_dim) {
  return ((batch * sequence_size + sequence) * num_heads + head) * head_dim + element;
}

std::vector<float> ReferenceAttention(const std::vector<float>& query,
                                      const std::vector<float>& key,
                                      const std::vector<float>& value,
                                      const std::vector<float>& mask,
                                      size_t batch, size_t seq_q, size_t seq_k,
                                      size_t num_heads, size_t head_dim, float scale,
                                      size_t query_batch = 0, size_t key_batch = 0,
                                      size_t value_batch = 0, size_t mask_batch = 0) {
  query_batch = query_batch == 0 ? batch : query_batch;
  key_batch = key_batch == 0 ? batch : key_batch;
  value_batch = value_batch == 0 ? batch : value_batch;
  mask_batch = mask_batch == 0 ? batch : mask_batch;
  std::vector<float> output(batch * seq_q * num_heads * head_dim, 0.0f);
  std::vector<float> scores(seq_k);
  for (size_t b = 0; b < batch; ++b) {
    for (size_t q = 0; q < seq_q; ++q) {
      for (size_t h = 0; h < num_heads; ++h) {
        for (size_t k = 0; k < seq_k; ++k) {
          float dot = 0.0f;
          for (size_t d = 0; d < head_dim; ++d) {
            dot += query[Offset(query_batch == 1 ? 0 : b, q, h, d,
                                seq_q, num_heads, head_dim)] *
                   key[Offset(key_batch == 1 ? 0 : b, k, h, d,
                              seq_k, num_heads, head_dim)];
          }
          const float bias = mask.empty() ? 0.0f : mask[(mask_batch == 1 ? 0 : b) * seq_k + k];
          scores[k] = scale * dot + bias;
        }
        const float max_score = *std::max_element(scores.begin(), scores.end());
        float sum = 0.0f;
        const size_t positive_inf_count = static_cast<size_t>(std::count_if(
            scores.begin(), scores.end(), [](float score) { return std::isinf(score) && score > 0.0f; }));
        if (positive_inf_count != 0) {
          for (float& score : scores) {
            score = (std::isinf(score) && score > 0.0f) ? 1.0f : 0.0f;
          }
          sum = static_cast<float>(positive_inf_count);
        } else if (std::isfinite(max_score)) {
          for (float& score : scores) {
            score = std::exp(score - max_score);
            sum += score;
          }
        } else {
          std::fill(scores.begin(), scores.end(), 0.0f);
          sum = 1.0f;
        }
        for (size_t d = 0; d < head_dim; ++d) {
          float result = 0.0f;
          for (size_t k = 0; k < seq_k; ++k) {
            result += scores[k] / sum *
                      value[Offset(value_batch == 1 ? 0 : b, k, h, d,
                                   seq_k, num_heads, head_dim)];
          }
          output[Offset(b, q, h, d, seq_q, num_heads, head_dim)] = result;
        }
      }
    }
  }
  return output;
}

enum class MaskPattern {
  None,
  SupportedTiled,
  UnsupportedPerHead,
  Untiled,
  SubOnly,
  Zero,
};

void RunFusionMatcherTest(bool reverse_div, bool share_qk,
                          MaskPattern mask_pattern, bool expect_fusion,
                          TransformerLevel level = TransformerLevel::Level2,
                          bool enable_ep = true, bool enable_fusion = true,
                          const std::string& scale_op = "Div", int64_t axis = -1) {
  constexpr int64_t batch = 2;
  constexpr int64_t seq_q = 1;
  constexpr int64_t seq_k = 3;
  constexpr int64_t num_heads = 2;
  constexpr int64_t head_dim = 2;
  constexpr int64_t hidden = num_heads * head_dim;

  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{{"ORT_KDNN_FUSE_ATTENTION", "1"}}};

  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto make_projection = [&](const std::vector<int64_t>& input_shape,
                               const std::vector<int64_t>& reshape_shape,
                               const std::vector<int64_t>& permutation,
                               float minimum, float maximum) {
      NodeArg* input = builder.MakeInput<float>(input_shape, minimum, maximum);
      NodeArg* bias = builder.MakeInitializer<float>({hidden}, std::vector<float>(hidden, 0.0f));
      NodeArg* add_out = builder.MakeIntermediate();
      builder.AddNode("Add", {input, bias}, {add_out});
      NodeArg* reshape_out = builder.MakeIntermediate();
      builder.AddNode("Reshape", {add_out, builder.Make1DInitializer<int64_t>(reshape_shape)}, {reshape_out});
      NodeArg* transpose_out = builder.MakeIntermediate();
      Node& transpose = builder.AddNode("Transpose", {reshape_out}, {transpose_out});
      transpose.AddAttribute("perm", permutation);
      return transpose_out;
    };

    NodeArg* query = make_projection({batch, seq_q, hidden}, {-1, seq_q, num_heads, head_dim},
                                     {0, 2, 1, 3}, 0.5f, 1.0f);
    NodeArg* key = make_projection({batch, seq_k, hidden}, {-1, seq_k, num_heads, head_dim},
                                   {0, 2, 3, 1}, 0.5f, 1.0f);
    NodeArg* value = make_projection({batch, seq_k, hidden}, {-1, seq_k, num_heads, head_dim},
                                     {0, 2, 1, 3}, -1.0f, 1.0f);

    NodeArg* qk = builder.MakeIntermediate();
    builder.AddNode("MatMul", {query, key}, {qk});
    NodeArg* divisor = builder.MakeScalarInitializer<float>(std::sqrt(static_cast<float>(head_dim)));
    NodeArg* scaled = builder.MakeIntermediate();
    if (scale_op == "None") {
      scaled = qk;
    } else {
      builder.AddNode(scale_op, reverse_div ? std::vector<NodeArg*>{divisor, qk} : std::vector<NodeArg*>{qk, divisor},
                      {scaled});
    }
    NodeArg* softmax_input = scaled;
    if (mask_pattern != MaskPattern::None) {
      const bool supported_tiled = mask_pattern == MaskPattern::SupportedTiled;
      const bool singleton_mask = mask_pattern != MaskPattern::UnsupportedPerHead;
      const std::vector<int64_t> input_shape = singleton_mask
                                                   ? std::vector<int64_t>{batch, seq_k}
                                                   : std::vector<int64_t>{batch, num_heads, seq_q, seq_k};
      const std::vector<int64_t> reshape_shape = singleton_mask
                                                     ? std::vector<int64_t>{batch, 1, 1, seq_k}
                                                     : input_shape;
      NodeArg* mask_input = builder.MakeInput<float>(input_shape, 0.0f, 1.0f);
      NodeArg* mask_reshape = builder.MakeIntermediate();
      builder.AddNode(
          "Reshape",
          {mask_input, builder.Make1DInitializer<int64_t>(reshape_shape)},
          {mask_reshape});
      NodeArg* keep_mask = mask_reshape;
      if (supported_tiled) {
        keep_mask = builder.MakeIntermediate();
        builder.AddNode(
            "Tile",
            {mask_reshape, builder.Make1DInitializer<int64_t>({1, num_heads, 1, 1})},
            {keep_mask});
      }
      NodeArg* mask_sub = builder.MakeIntermediate();
      builder.AddNode("Sub",
                      {builder.MakeScalarInitializer<float>(1.0f), keep_mask},
                      {mask_sub});
      NodeArg* mask_bias = mask_sub;
      if (mask_pattern != MaskPattern::SubOnly) {
        mask_bias = builder.MakeIntermediate();
        builder.AddNode("Mul",
                        {builder.MakeScalarInitializer<float>(-1.0e9f), mask_sub},
                        {mask_bias});
      }
      if (mask_pattern == MaskPattern::Zero) {
        mask_bias = builder.MakeInitializer<float>({1, num_heads, 1, seq_k}, std::vector<float>(num_heads * seq_k, 0));
      }
      softmax_input = builder.MakeIntermediate();
      builder.AddNode("Add", {mask_bias, scaled}, {softmax_input});
    }

    NodeArg* probabilities = builder.MakeIntermediate();
    Node& softmax = builder.AddNode("Softmax", {softmax_input}, {probabilities});
    softmax.AddAttribute("axis", axis);
    NodeArg* context = builder.MakeIntermediate();
    builder.AddNode("MatMul", {probabilities, value}, {context});
    NodeArg* output = builder.MakeOutput<float>(std::vector<int64_t>{batch, seq_q, hidden});
    builder.AddNode("Reshape",
                    {context, builder.Make1DInitializer<int64_t>({batch, seq_q, hidden})},
                    {output});

    if (share_qk) {
      NodeArg* qk_output = builder.MakeOutput<float>(
          std::vector<int64_t>{batch, num_heads, seq_q, seq_k});
      builder.AddNode("Identity", {qk}, {qk_output});
    }
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    const auto op_count = CountOpsInGraph(session.GetGraph());
    const std::string fused_op = std::string(kKdnnDomain) + ".KdnnFusedAttention";
    const auto fused_it = op_count.find(fused_op);
    const int fused_count = fused_it == op_count.end() ? 0 : fused_it->second;
    EXPECT_EQ(fused_count, expect_fusion ? 1 : 0);
    for (const auto& node : session.GetGraph().Nodes()) {
      if (node.Domain() == kKdnnDomain) {
        EXPECT_EQ(node.GetExecutionProviderType(), kKdnnExecutionProvider);
      }
    }
  };

  auto configure_session = [](SessionOptions& options) {
    options.intra_op_param.thread_pool_size = 2;
  };
  auto provider = enable_ep
                      ? KdnnProviderFactoryCreator::Create(
                            {{"enable_attention_fusion", enable_fusion ? "1" : "0"}})
                            ->CreateProvider()
                      : nullptr;
  TransformerTester(build_test_case, check_graph, TransformerLevel::Default, level,
                    13, 1.0e-4, 1.0e-4, nullptr, configure_session, {}, std::move(provider));
}

TEST(KdnnFusedAttentionTest, ProducesReferenceOutput) {
  constexpr size_t batch = 2;
  constexpr size_t seq_q = 2;
  constexpr size_t seq_k = 3;
  constexpr size_t num_heads = 2;
  constexpr size_t head_dim = 2;
  constexpr float scale = 0.5f;
  constexpr size_t hidden = num_heads * head_dim;

  std::vector<float> query(batch * seq_q * hidden);
  std::vector<float> key(batch * seq_k * hidden);
  std::vector<float> value(batch * seq_k * hidden);
  for (size_t i = 0; i < query.size(); ++i) {
    query[i] = 0.1f * static_cast<float>(static_cast<int>(i % 7) - 3);
  }
  for (size_t i = 0; i < key.size(); ++i) {
    key[i] = 0.07f * static_cast<float>(static_cast<int>(i % 9) - 4);
    value[i] = 0.05f * static_cast<float>(static_cast<int>(i % 11) - 5);
  }
  const std::vector<float> mask{0.0f, -0.25f, -1000.0f, -0.5f, 0.0f, -1000.0f};
  const auto expected = ReferenceAttention(query, key, value, mask, batch, seq_q, seq_k,
                                           num_heads, head_dim, scale);

  OpTester test("KdnnFusedAttention", 1, kKdnnDomain);
  test.AddAttribute<int64_t>("num_heads", num_heads);
  test.AddAttribute<float>("scale", scale);
  test.AddInput<float>("query", {batch, seq_q, hidden}, query);
  test.AddInput<float>("key", {batch, seq_k, hidden}, key);
  test.AddInput<float>("value", {batch, seq_k, hidden}, value);
  test.AddInput<float>("mask", {batch, 1, 1, seq_k}, mask);
  test.AddOutput<float>("output", {batch, seq_q, hidden}, expected,
                        false, 1.0e-4f, 1.0e-4f);
  RunKdnnTest(test);
}

TEST(KdnnFusedAttentionTest, SupportsNoMaskDefaultScaleAndOddTails) {
  constexpr size_t batch = 2;
  constexpr size_t seq_q = 3;
  constexpr size_t seq_k = 17;
  constexpr size_t num_heads = 3;
  constexpr size_t head_dim = 7;
  constexpr size_t hidden = num_heads * head_dim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  std::vector<float> query(batch * seq_q * hidden);
  std::vector<float> key(batch * seq_k * hidden);
  std::vector<float> value(batch * seq_k * hidden);
  for (size_t i = 0; i < query.size(); ++i) {
    query[i] = 0.02f * static_cast<float>(static_cast<int>(i % 19) - 9);
  }
  for (size_t i = 0; i < key.size(); ++i) {
    key[i] = 0.015f * static_cast<float>(static_cast<int>(i % 23) - 11);
    value[i] = 0.01f * static_cast<float>(static_cast<int>(i % 29) - 14);
  }
  const auto expected = ReferenceAttention(query, key, value, {}, batch, seq_q, seq_k,
                                           num_heads, head_dim, scale);

  OpTester test("KdnnFusedAttention", 1, kKdnnDomain);
  test.AddAttribute<int64_t>("num_heads", num_heads);
  test.AddInput<float>("query", {batch, seq_q, hidden}, query);
  test.AddInput<float>("key", {batch, seq_k, hidden}, key);
  test.AddInput<float>("value", {batch, seq_k, hidden}, value);
  test.AddOptionalInputEdge<float>();
  test.AddOutput<float>("output", {batch, seq_q, hidden}, expected,
                        false, 5.0e-4f, 5.0e-4f);
  RunKdnnTest(test);
}

TEST(KdnnFusedAttentionTest, SupportsRank4ModelLayout) {
  constexpr size_t batch = 2;
  constexpr size_t seq_q = 1;
  constexpr size_t seq_k = 5;
  constexpr size_t num_heads = 2;
  constexpr size_t head_dim = 3;
  constexpr size_t hidden = num_heads * head_dim;
  constexpr float scale = 0.75f;

  std::vector<float> query(batch * seq_q * hidden);
  std::vector<float> key(batch * seq_k * hidden);
  std::vector<float> value(batch * seq_k * hidden);
  for (size_t i = 0; i < query.size(); ++i) query[i] = 0.03f * static_cast<float>(i + 1);
  for (size_t i = 0; i < key.size(); ++i) {
    key[i] = 0.02f * static_cast<float>(static_cast<int>(i % 13) - 6);
    value[i] = 0.04f * static_cast<float>(static_cast<int>(i % 11) - 5);
  }
  const std::vector<float> mask{0.0f, -0.25f, -1.0f, 0.0f, -1000.0f};
  const auto expected = ReferenceAttention(query, key, value, mask, batch, seq_q, seq_k,
                                           num_heads, head_dim, scale, batch, batch, batch, 1);

  OpTester test("KdnnFusedAttention", 1, kKdnnDomain);
  test.AddAttribute<int64_t>("num_heads", num_heads);
  test.AddAttribute<float>("scale", scale);
  test.AddInput<float>("query", {batch, seq_q, num_heads, head_dim}, query);
  test.AddInput<float>("key", {batch, seq_k, num_heads, head_dim}, key);
  test.AddInput<float>("value", {batch, seq_k, num_heads, head_dim}, value);
  test.AddInput<float>("mask", {1, 1, 1, seq_k}, mask);
  test.AddOutput<float>("output", {batch, num_heads, seq_q, head_dim}, expected,
                        false, 1.0e-4f, 1.0e-4f);
  RunKdnnTest(test);
}

TEST(KdnnFusedAttentionTest, SupportsSplitBatchNoPack) {
  constexpr size_t batch = 3;
  constexpr size_t seq_q = 1;
  constexpr size_t seq_k = 7;
  constexpr size_t num_heads = 2;
  constexpr size_t head_dim = 3;
  constexpr size_t hidden = num_heads * head_dim;
  constexpr float scale = 0.75f;

  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{{"ORT_KDNN_ATTENTION_ALGORITHM", "classic_no_pack"}}};

  std::vector<float> query(batch * seq_q * hidden);
  std::vector<float> key(batch * seq_k * hidden);
  std::vector<float> value(batch * seq_k * hidden);
  for (size_t i = 0; i < query.size(); ++i) {
    query[i] = 0.03f * static_cast<float>(i + 1);
  }
  for (size_t i = 0; i < key.size(); ++i) {
    key[i] = 0.02f * static_cast<float>(static_cast<int>(i % 13) - 6);
    value[i] = 0.04f * static_cast<float>(static_cast<int>(i % 11) - 5);
  }
  const std::vector<float> mask{0.0f, -0.25f, -1.0f, 0.0f, -0.5f, -2.0f, -1000.0f};
  const auto expected = ReferenceAttention(query, key, value, mask, batch, seq_q, seq_k,
                                           num_heads, head_dim, scale, batch, batch, batch, 1);

  OpTester test("KdnnFusedAttention", 1, kKdnnDomain);
  test.AddAttribute<int64_t>("num_heads", num_heads);
  test.AddAttribute<float>("scale", scale);
  test.AddInput<float>("query", {batch, seq_q, num_heads, head_dim}, query);
  test.AddInput<float>("key", {batch, seq_k, num_heads, head_dim}, key);
  test.AddInput<float>("value", {batch, seq_k, num_heads, head_dim}, value);
  test.AddInput<float>("mask", {1, 1, 1, seq_k}, mask);
  test.AddOutput<float>("output", {batch, num_heads, seq_q, head_dim}, expected,
                        false, 1.0e-4f, 1.0e-4f);
  RunKdnnTest(test);
}

TEST(KdnnFusedAttentionTest, SupportsInterleavedNoPackModel4Shape) {
  constexpr size_t batch = 3;
  constexpr size_t seq_q = 1;
  constexpr size_t seq_k = 50;
  constexpr size_t num_heads = 4;
  constexpr size_t head_dim = 32;
  constexpr size_t hidden = num_heads * head_dim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{{"ORT_KDNN_ATTENTION_ALGORITHM", "classic_no_pack"}}};

  std::vector<float> query(batch * seq_q * hidden);
  std::vector<float> key(batch * seq_k * hidden);
  std::vector<float> value(batch * seq_k * hidden);
  std::vector<float> mask(seq_k);
  for (size_t i = 0; i < query.size(); ++i) {
    query[i] = 0.007f * static_cast<float>(static_cast<int>(i % 23) - 11);
  }
  for (size_t i = 0; i < key.size(); ++i) {
    key[i] = 0.009f * static_cast<float>(static_cast<int>(i % 41) - 20);
    value[i] = 0.011f * static_cast<float>(static_cast<int>(i % 43) - 21);
  }
  for (size_t i = 0; i < mask.size(); ++i) {
    mask[i] = -0.0005f * static_cast<float>(i);
  }
  const auto expected = ReferenceAttention(query, key, value, mask, batch, seq_q, seq_k,
                                           num_heads, head_dim, scale, batch, batch, batch, 1);

  OpTester test("KdnnFusedAttention", 1, kKdnnDomain);
  test.AddAttribute<int64_t>("num_heads", num_heads);
  test.AddAttribute<float>("scale", scale);
  test.AddInput<float>("query", {batch, seq_q, hidden}, query);
  test.AddInput<float>("key", {batch, seq_k, hidden}, key);
  test.AddInput<float>("value", {batch, seq_k, hidden}, value);
  test.AddInput<float>("mask", {1, 1, 1, seq_k}, mask);
  test.AddOutput<float>("output", {batch, seq_q, hidden}, expected,
                        false, 1.0e-3f, 1.0e-3f);
  RunKdnnTest(test);
}

TEST(KdnnFusedAttentionTest, AllMaskedRowsReturnZeroWithoutNonFiniteValues) {
  constexpr size_t batch = 2;
  constexpr size_t seq_q = 2;
  constexpr size_t seq_k = 5;
  constexpr size_t num_heads = 2;
  constexpr size_t head_dim = 3;
  constexpr size_t hidden = num_heads * head_dim;
  constexpr float scale = 0.5f;

  std::vector<float> query(batch * seq_q * hidden, 0.25f);
  std::vector<float> key(batch * seq_k * hidden, -0.5f);
  std::vector<float> value(batch * seq_k * hidden, 2.0f);
  std::vector<float> mask(seq_k, -std::numeric_limits<float>::infinity());
  std::vector<float> expected(batch * seq_q * hidden, 0.0f);

  OpTester test("KdnnFusedAttention", 1, kKdnnDomain);
  test.AddAttribute<int64_t>("num_heads", num_heads);
  test.AddAttribute<float>("scale", scale);
  test.AddInput<float>("query", {batch, seq_q, hidden}, query);
  test.AddInput<float>("key", {batch, seq_k, hidden}, key);
  test.AddInput<float>("value", {batch, seq_k, hidden}, value);
  test.AddInput<float>("mask", {1, 1, 1, seq_k}, mask);
  test.AddOutput<float>("output", {batch, seq_q, hidden}, expected,
                        false, 0.0f, 0.0f);
  RunKdnnTest(test);
}

TEST(KdnnFusedAttentionTest, RejectsNonSingletonMaskMiddleDimension) {
  OpTester test("KdnnFusedAttention", 1, kKdnnDomain);
  test.AddAttribute<int64_t>("num_heads", 2);
  test.AddInput<float>("query", {2, 1, 4}, std::vector<float>(8, 0.1f));
  test.AddInput<float>("key", {2, 3, 4}, std::vector<float>(24, 0.1f));
  test.AddInput<float>("value", {2, 3, 4}, std::vector<float>(24, 0.1f));
  test.AddInput<float>("mask", {2, 2, 3}, std::vector<float>(12, 0.0f));
  test.AddOutput<float>("output", {2, 1, 4}, std::vector<float>(8, 0.0f));
  RunKdnnTest(test, OpTester::ExpectResult::kExpectFailure,
              "intermediate mask dimensions must be 1");
}

// Reuse shapes across Sessions with different scales and algorithms to detect
// stale parameters in KDNN's primitive cache or thread-local workspaces.
TEST(KdnnFusedAttentionTest, SameShapeDifferentScaleAndAlgorithmAcrossRuns) {
  constexpr size_t batch = 2;
  constexpr size_t seq_q = 1;
  constexpr size_t seq_k = 6;
  constexpr size_t num_heads = 4;
  constexpr size_t head_dim = 32;
  constexpr size_t hidden = num_heads * head_dim;

  std::vector<float> query(batch * seq_q * hidden);
  std::vector<float> key(batch * seq_k * hidden);
  std::vector<float> value(batch * seq_k * hidden);
  for (size_t i = 0; i < query.size(); ++i) {
    query[i] = 0.007f * static_cast<float>(static_cast<int>(i % 23) - 11);
  }
  for (size_t i = 0; i < key.size(); ++i) {
    key[i] = 0.009f * static_cast<float>(static_cast<int>(i % 41) - 20);
    value[i] = 0.011f * static_cast<float>(static_cast<int>(i % 43) - 21);
  }
  const std::vector<float> mask{0.0f, -0.25f, -1.0f, 0.0f, -0.5f, -1000.0f};

  auto run_case = [&](float scale, const char* algorithm) {
    // Shared [1,1,1,Sk] mask: pass mask_batch=1 so the reference broadcasts it.
    const auto expected = ReferenceAttention(query, key, value, mask, batch, seq_q, seq_k,
                                             num_heads, head_dim, scale,
                                             /*query_batch=*/batch, /*key_batch=*/batch,
                                             /*value_batch=*/batch, /*mask_batch=*/1);
    OpTester test("KdnnFusedAttention", 1, kKdnnDomain);
    test.AddAttribute<int64_t>("num_heads", num_heads);
    test.AddAttribute<float>("scale", scale);
    test.AddInput<float>("query", {batch, seq_q, hidden}, query);
    test.AddInput<float>("key", {batch, seq_k, hidden}, key);
    test.AddInput<float>("value", {batch, seq_k, hidden}, value);
    test.AddInput<float>("mask", {1, 1, 1, seq_k}, mask);
    test.AddOutput<float>("output", {batch, seq_q, hidden}, expected,
                          false, 1.0e-4f, 1.0e-4f);
    // Check repeated execution after the first run populates the cache.
    {
      ScopedEnvironmentVariables scoped_env_vars{
          EnvVarMap{{"ORT_KDNN_ATTENTION_ALGORITHM", algorithm}}};
      test.SetNumRunCalls(2);
      RunKdnnTest(test);
    }
  };

  // Same shape, different scale in the same process: a primitive cache keyed
  // only on shape would silently reuse the first scale.
  run_case(0.5f, "classic_no_pack");
  run_case(0.25f, "classic_no_pack");
  run_case(0.5f, "classic");
  // Switch back: must not reuse the classic primitive for no-pack execution.
  run_case(0.125f, "classic_no_pack");
}

TEST(KdnnFusedAttentionTest, BroadcastsBatchInputs) {
  constexpr size_t output_batch = 2;
  constexpr size_t seq_q = 1;
  constexpr size_t seq_k = 3;
  constexpr size_t num_heads = 2;
  constexpr size_t head_dim = 2;
  constexpr size_t hidden = num_heads * head_dim;
  constexpr float scale = 0.5f;

  auto run_case = [&](size_t query_batch, size_t key_batch,
                      size_t value_batch, size_t mask_batch) {
    std::vector<float> query(query_batch * seq_q * hidden);
    std::vector<float> key(key_batch * seq_k * hidden);
    std::vector<float> value(value_batch * seq_k * hidden);
    std::vector<float> mask(mask_batch * seq_k);
    for (size_t i = 0; i < query.size(); ++i) query[i] = 0.03f * static_cast<float>(i + 1);
    for (size_t i = 0; i < key.size(); ++i) key[i] = 0.02f * static_cast<float>(i + 1);
    for (size_t i = 0; i < value.size(); ++i) value[i] = 0.01f * static_cast<float>(i + 1);
    for (size_t i = 0; i < mask.size(); ++i) mask[i] = -0.25f * static_cast<float>(i % seq_k);
    const auto expected = ReferenceAttention(query, key, value, mask, output_batch, seq_q, seq_k,
                                             num_heads, head_dim, scale, query_batch, key_batch,
                                             value_batch, mask_batch);

    OpTester test("KdnnFusedAttention", 1, kKdnnDomain);
    test.AddAttribute<int64_t>("num_heads", num_heads);
    test.AddAttribute<float>("scale", scale);
    test.AddInput<float>("query", {static_cast<int64_t>(query_batch), seq_q, hidden}, query);
    test.AddInput<float>("key", {static_cast<int64_t>(key_batch), seq_k, hidden}, key);
    test.AddInput<float>("value", {static_cast<int64_t>(value_batch), seq_k, hidden}, value);
    test.AddInput<float>("mask", {static_cast<int64_t>(mask_batch), 1, 1, seq_k}, mask);
    test.AddOutput<float>("output", {output_batch, seq_q, hidden}, expected,
                          false, 1.0e-4f, 1.0e-4f);
    RunKdnnTest(test);
  };

  run_case(/*query_batch=*/1, /*key_batch=*/1, /*value_batch=*/2, /*mask_batch=*/2);
  run_case(/*query_batch=*/1, /*key_batch=*/2, /*value_batch=*/1, /*mask_batch=*/1);
  run_case(/*query_batch=*/2, /*key_batch=*/1, /*value_batch=*/1, /*mask_batch=*/1);
}

TEST(KdnnFusedAttentionTest, PreservesEmptyBatchAndEmptyQuery) {
  for (bool empty_batch : {true, false}) {
    const int64_t batch = empty_batch ? 0 : 1;
    const int64_t sq = empty_batch ? 2 : 0;
    OpTester test("KdnnFusedAttention", 1, kKdnnDomain);
    test.AddAttribute<int64_t>("num_heads", 2);
    test.AddInput<float>("query", {batch, sq, 4}, {});
    test.AddInput<float>("key", {batch, 3, 4}, std::vector<float>(batch * 12, 0));
    test.AddInput<float>("value", {batch, 3, 4}, std::vector<float>(batch * 12, 0));
    test.AddOutput<float>("output", {batch, sq, 4}, {});
    RunKdnnTest(test);
  }
}

TEST(KdnnFusedAttentionTest, RejectsHeadCountNarrowingOverflow) {
  OpTester test("KdnnFusedAttention", 1, kKdnnDomain);
  test.AddAttribute<int64_t>("num_heads", std::numeric_limits<int64_t>::max());
  test.AddInput<float>("query", {1, 1, 4}, {0, 0, 0, 0});
  test.AddInput<float>("key", {1, 1, 4}, {0, 0, 0, 0});
  test.AddInput<float>("value", {1, 1, 4}, {0, 0, 0, 0});
  test.AddOutput<float>("output", {1, 1, 4}, {0, 0, 0, 0});
  RunKdnnTest(test, OpTester::ExpectResult::kExpectFailure, "num_heads");
}

TEST(KdnnFusedAttentionTest, KeepMaskRequiresInput) {
  OpTester test("KdnnFusedAttention", 2, kKdnnDomain);
  test.AddAttribute<int64_t>("num_heads", 2);
  test.AddAttribute<std::string>("mask_mode", "keep");
  test.AddAttribute<float>("mask_offset", 1.0f);
  test.AddAttribute<float>("mask_multiplier", -10.0f);
  test.AddInput<float>("query", {1, 1, 4}, {0, 0, 0, 0});
  test.AddInput<float>("key", {1, 1, 4}, {0, 0, 0, 0});
  test.AddInput<float>("value", {1, 1, 4}, {0, 0, 0, 0});
  test.AddOutput<float>("output", {1, 1, 4}, {0, 0, 0, 0});
  RunKdnnTest(test, OpTester::ExpectResult::kExpectFailure, "keep mask requires rank-4 mask input");
}

TEST(KdnnAttentionFusionTest, CpuOnlyIgnoresLegacyEnvironmentGate) {
  RunFusionMatcherTest(false, false, MaskPattern::SupportedTiled, false, TransformerLevel::Level2, false);
}
TEST(KdnnAttentionFusionTest, BasicDoesNotFuse) {
  RunFusionMatcherTest(false, false, MaskPattern::SupportedTiled, false, TransformerLevel::Level1);
}
TEST(KdnnAttentionFusionTest, ProviderOptionDisablesNewFusion) {
  RunFusionMatcherTest(false, false, MaskPattern::SupportedTiled, false, TransformerLevel::Level2, true, false);
}
TEST(KdnnAttentionFusionTest, AllEnablesFusion) {
  RunFusionMatcherTest(false, false, MaskPattern::SupportedTiled, true, TransformerLevel::Level3);
}
TEST(KdnnAttentionFusionTest, FusesSupportedPattern) {
  RunFusionMatcherTest(false, false, MaskPattern::None, true);
}

TEST(KdnnAttentionFusionTest, FusesSupportedTiledMaskPattern) {
  RunFusionMatcherTest(false, false, MaskPattern::SupportedTiled, true);
}

TEST(KdnnAttentionFusionTest, RejectsReversedDiv) {
  RunFusionMatcherTest(true, false, MaskPattern::None, false);
}

TEST(KdnnAttentionFusionTest, RejectsSharedQkOutput) {
  RunFusionMatcherTest(false, true, MaskPattern::None, false);
}

TEST(KdnnAttentionFusionTest, RejectsPerHeadMask) {
  RunFusionMatcherTest(false, false, MaskPattern::UnsupportedPerHead, false);
}

TEST(KdnnAttentionFusionTest, FusesMultiplyOnEitherSideAndUnscaledScores) {
  for (bool reverse : {false, true}) {
    RunFusionMatcherTest(reverse, false, MaskPattern::None, true, TransformerLevel::Level2, true, true, "Mul");
  }
  RunFusionMatcherTest(false, false, MaskPattern::None, true, TransformerLevel::Level2, true, true, "None");
}

TEST(KdnnAttentionFusionTest, FusesUntiledSubOnlyAndZeroMasks) {
  for (auto mask : {MaskPattern::Untiled, MaskPattern::SubOnly, MaskPattern::Zero}) {
    RunFusionMatcherTest(false, false, mask, true);
  }
}

TEST(KdnnAttentionFusionTest, ChecksSoftmaxAxis) {
  RunFusionMatcherTest(false, false, MaskPattern::None, true, TransformerLevel::Level2, true, true, "Mul", 3);
  RunFusionMatcherTest(false, false, MaskPattern::None, false, TransformerLevel::Level2, true, true, "Mul", 2);
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif  // USE_KDNN_EP
