// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.

#if defined(USE_KDNN)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "core/graph/constants.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime {
namespace test {
namespace {

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
          scores[k] = scale * dot + mask[(mask_batch == 1 ? 0 : b) * seq_k + k];
        }
        const float max_score = *std::max_element(scores.begin(), scores.end());
        float sum = 0.0f;
        for (float& score : scores) {
          score = std::exp(score - max_score);
          sum += score;
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

void RunFusionMatcherTest(bool reverse_div, bool share_qk,
                          bool per_head_mask, bool expect_fusion) {
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
    builder.AddNode("Div", reverse_div ? std::vector<NodeArg*>{divisor, qk} : std::vector<NodeArg*>{qk, divisor},
                    {scaled});
    NodeArg* softmax_input = scaled;
    if (per_head_mask) {
      NodeArg* mask_input = builder.MakeInput<float>(
          {batch, num_heads, seq_q, seq_k}, 0.0f, 1.0f);
      NodeArg* mask_reshape = builder.MakeIntermediate();
      builder.AddNode(
          "Reshape",
          {mask_input, builder.Make1DInitializer<int64_t>(
                           {batch, num_heads, seq_q, seq_k})},
          {mask_reshape});
      NodeArg* mask_sub = builder.MakeIntermediate();
      builder.AddNode("Sub",
                      {builder.MakeScalarInitializer<float>(1.0f), mask_reshape},
                      {mask_sub});
      NodeArg* mask_bias = builder.MakeIntermediate();
      builder.AddNode("Mul",
                      {builder.MakeScalarInitializer<float>(-1.0e9f), mask_sub},
                      {mask_bias});
      softmax_input = builder.MakeIntermediate();
      builder.AddNode("Add", {scaled, mask_bias}, {softmax_input});
    }

    NodeArg* probabilities = builder.MakeIntermediate();
    Node& softmax = builder.AddNode("Softmax", {softmax_input}, {probabilities});
    softmax.AddAttribute("axis", int64_t{-1});
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
  };

  TransformerTester(build_test_case, check_graph,
                    TransformerLevel::Default, TransformerLevel::Level1,
                    13, 1.0e-5, 1.0e-5);
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
                        false, 1.0e-5f, 1.0e-5f);
  test.Run();
}

TEST(KdnnFusedAttentionTest, SupportsInterleavedNoPack4H32) {
  constexpr size_t batch = 3;
  constexpr size_t seq_q = 1;
  constexpr size_t seq_k = 50;
  constexpr size_t num_heads = 4;
  constexpr size_t head_dim = 32;
  constexpr size_t hidden = num_heads * head_dim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{{"ORT_KDNN_ATTENTION_ALGORITHM", "classic_no_pack"},
                {"ORT_KDNN_ATTENTION_INTERLEAVED_GEMV", "all"}}};

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
                                           num_heads, head_dim, scale,
                                           batch, batch, batch, 1);

  OpTester test("KdnnFusedAttention", 1, kKdnnDomain);
  test.AddAttribute<int64_t>("num_heads", num_heads);
  test.AddAttribute<float>("scale", scale);
  test.AddInput<float>("query", {batch, seq_q, hidden}, query);
  test.AddInput<float>("key", {batch, seq_k, hidden}, key);
  test.AddInput<float>("value", {batch, seq_k, hidden}, value);
  test.AddInput<float>("mask", {1, 1, 1, seq_k}, mask);
  test.AddOutput<float>("output", {batch, seq_q, hidden}, expected,
                        false, 1.0e-5f, 1.0e-5f);
  test.Run();
}

TEST(KdnnFusedAttentionTest, RejectsNonSingletonMaskMiddleDimension) {
  OpTester test("KdnnFusedAttention", 1, kKdnnDomain);
  test.AddAttribute<int64_t>("num_heads", 2);
  test.AddInput<float>("query", {2, 1, 4}, std::vector<float>(8, 0.1f));
  test.AddInput<float>("key", {2, 3, 4}, std::vector<float>(24, 0.1f));
  test.AddInput<float>("value", {2, 3, 4}, std::vector<float>(24, 0.1f));
  test.AddInput<float>("mask", {2, 2, 3}, std::vector<float>(12, 0.0f));
  test.AddOutput<float>("output", {2, 1, 4}, std::vector<float>(8, 0.0f));
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "intermediate mask dimensions must be 1");
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
                          false, 1.0e-5f, 1.0e-5f);
    test.Run();
  };

  run_case(/*query_batch=*/1, /*key_batch=*/1, /*value_batch=*/2, /*mask_batch=*/2);
  run_case(/*query_batch=*/1, /*key_batch=*/2, /*value_batch=*/1, /*mask_batch=*/1);
  run_case(/*query_batch=*/2, /*key_batch=*/1, /*value_batch=*/1, /*mask_batch=*/1);
}

TEST(KdnnAttentionFusionTest, FusesSupportedPattern) {
  RunFusionMatcherTest(false, false, false, true);
}

TEST(KdnnAttentionFusionTest, RejectsReversedDiv) {
  RunFusionMatcherTest(true, false, false, false);
}

TEST(KdnnAttentionFusionTest, RejectsSharedQkOutput) {
  RunFusionMatcherTest(false, true, false, false);
}

TEST(KdnnAttentionFusionTest, RejectsPerHeadMask) {
  RunFusionMatcherTest(false, false, true, false);
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif  // USE_KDNN
