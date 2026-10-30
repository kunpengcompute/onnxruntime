// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.
#ifdef USE_KDNN_EP
#include <limits>
#include <numeric>

#include "gtest/gtest.h"
#include "core/graph/constants.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime::test {
namespace {
struct InvalidShape {
  const char* name;
  std::vector<int64_t> query, key, value, mask;
  const char* error;
};

size_t Elements(const std::vector<int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>{});
}

void RunValidationCase(OpTester& test, const char* error = nullptr) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultKdnnExecutionProvider());
  test.Run(error ? OpTester::ExpectResult::kExpectFailure : OpTester::ExpectResult::kExpectSuccess, error ? error : "",
           {}, nullptr, &providers);
}

class KdnnInvalidShapeTest : public ::testing::TestWithParam<InvalidShape> {};
TEST_P(KdnnInvalidShapeTest, RejectsAtRuntimeWithoutStaticShapes) {
  const auto& c = GetParam();
  OpTester test("KdnnFusedAttention", 2, kKdnnDomain);
  // Unknown shapes must be checked by Compute, not just shape inference.
  test.AddShapeToTensorData(false);
  test.AddAttribute<int64_t>("num_heads", 2);
  test.AddInput<float>("q", c.query, std::vector<float>(Elements(c.query), 0.25f));
  test.AddInput<float>("k", c.key, std::vector<float>(Elements(c.key), 0.25f));
  test.AddInput<float>("v", c.value, std::vector<float>(Elements(c.value), 1.0f));
  if (!c.mask.empty()) test.AddInput<float>("mask", c.mask, std::vector<float>(Elements(c.mask), 0));
  test.AddOutput<float>("out", {1, 1, 4}, {0, 0, 0, 0});
  RunValidationCase(test, c.error);
}

INSTANTIATE_TEST_SUITE_P(
    Validation, KdnnInvalidShapeTest,
    ::testing::Values(InvalidShape{"Rank", {1, 4}, {1, 1, 4}, {1, 1, 4}, {}, "matching rank-3"},
                      InvalidShape{"MixedRank", {1, 1, 4}, {1, 1, 2, 2}, {1, 1, 4}, {}, "matching rank-3"},
                      InvalidShape{"Sequence", {1, 1, 4}, {1, 2, 4}, {1, 3, 4}, {}, "K/V sequence length mismatch"},
                      InvalidShape{"QueryKeyBatch", {2, 1, 4}, {3, 1, 4}, {1, 1, 4}, {}, "Q/K batch dimensions"},
                      InvalidShape{
                          "SplitQuery", {1, 2, 2, 2}, {1, 1, 2, 2}, {1, 1, 2, 2}, {}, "q_sequence_length == 1"},
                      InvalidShape{"Heads", {1, 1, 3, 2}, {1, 1, 3, 2}, {1, 1, 3, 2}, {}, "must match num_heads"},
                      InvalidShape{"HeadSize", {1, 1, 2, 2}, {1, 1, 2, 3}, {1, 1, 2, 2}, {}, "head size mismatch"},
                      InvalidShape{"Hidden", {1, 1, 4}, {1, 1, 6}, {1, 1, 4}, {}, "hidden mismatch"},
                      InvalidShape{"Divisibility", {1, 1, 3}, {1, 1, 3}, {1, 1, 3}, {}, "not divisible"},
                      InvalidShape{"MaskRank", {1, 1, 4}, {1, 3, 4}, {1, 3, 4}, {3}, "batch and sequence"},
                      InvalidShape{"MaskSequence", {1, 1, 4}, {1, 3, 4}, {1, 3, 4}, {1, 2}, "mask last dim"},
                      InvalidShape{"MaskBatch", {2, 1, 4}, {2, 3, 4}, {2, 3, 4}, {3, 3}, "score and mask batch"},
                      InvalidShape{"ValueBatch", {2, 1, 4}, {2, 3, 4}, {3, 3, 4}, {}, "attention and V batch"},
                      InvalidShape{"EmptyKey", {1, 1, 4}, {1, 0, 4}, {1, 0, 4}, {}, "ValidateInput rejected"},
                      InvalidShape{"EmptyHidden", {1, 1, 0}, {1, 1, 0}, {1, 1, 0}, {}, "ValidateInput rejected"}),
    [](const auto& info) { return info.param.name; });

TEST(KdnnValidationTest, RejectsInvalidAttributes) {
  for (int c = 0; c < 6; ++c) {
    SCOPED_TRACE(c);
    OpTester test("KdnnFusedAttention", 2, kKdnnDomain);
    test.AddAttribute<int64_t>("num_heads", c == 0 ? 0 : 2);
    const char* error = "num_heads";
    if (c == 1) {
      test.AddAttribute<float>("scale", std::numeric_limits<float>::infinity());
      error = "scale must be finite";
    } else if (c == 2) {
      test.AddAttribute<std::string>("mask_mode", "invalid");
      error = "Invalid KDNN mask_mode";
    } else if (c >= 3) {
      test.AddAttribute<std::string>("mask_mode", "keep");
      if (c >= 4) test.AddAttribute<float>("mask_offset", c == 4 ? std::numeric_limits<float>::quiet_NaN() : 1.0f);
      if (c == 5) test.AddAttribute<float>("mask_multiplier", std::numeric_limits<float>::infinity());
      error = "finite mask_offset and mask_multiplier";
    }
    for (const char* name : {"q", "k", "v"}) test.AddInput<float>(name, {1, 1, 4}, {0, 0, 0, 0});
    test.AddOutput<float>("out", {1, 1, 4}, {0, 0, 0, 0});
    RunValidationCase(test, error);
  }
}

// v2 must match ONNX Softmax for nonfinite scores; v1 intentionally has
// different all-masked semantics. Exercise both NEON blocks and scalar tails.
TEST(KdnnValidationTest, VersionTwoPreservesNonfiniteAndOverflowSemantics) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (int c = 0; c < 8; ++c) {
    SCOPED_TRACE(c);
    OpTester test("KdnnFusedAttention", 2, kKdnnDomain);
    test.AddAttribute<int64_t>("num_heads", 1);
    const float scale = c == 7 ? std::numeric_limits<float>::max() / 8 : (c == 6 ? 4.0f : 0.5f);
    test.AddAttribute<float>("scale", scale);
    std::vector<float> q(3, 0.25f), k(15, 0.5f), mask(5, 0);
    if (c == 0) q.back() = nan;
    if (c == 1) k.front() = inf;
    if (c == 2) mask.back() = nan;
    if (c == 3) std::fill(mask.begin(), mask.end(), -inf);
    if (c == 4) {
      std::fill(q.begin(), q.end(), 1.0e30f);
      std::fill(k.begin(), k.end(), 1.0e30f);
    }
    if (c == 5) mask.front() = inf;
    if (c == 6) {
      std::fill(q.begin(), q.end(), std::numeric_limits<float>::max() / 8);
      std::fill(k.begin(), k.end(), 0);
    }
    if (c == 7) {
      std::fill(q.begin(), q.end(), 1);
      std::fill(k.begin(), k.end(), 1);
    }
    test.AddInput<float>("q", {1, 1, 3}, q);
    test.AddInput<float>("k", {1, 5, 3}, k);
    test.AddInput<float>("v", {1, 5, 3}, std::vector<float>(15, 2));
    if (c != 4 && c != 6) test.AddInput<float>("mask", {1, 1, 1, 5}, mask);
    test.AddOutput<float>("out", {1, 1, 3}, std::vector<float>(3, c >= 6 ? 2 : nan));
    RunValidationCase(test);
  }
}

TEST(KdnnValidationTest, KeepMaskMatchesAdditiveMaskWithWorkspaceReuse) {
  for (const char* algorithm : {"auto", "classic", "classic_no_pack"}) {
    ScopedEnvironmentVariables env(EnvVarMap{{"ORT_KDNN_ATTENTION_ALGORITHM", algorithm}});
    for (int64_t sk : {3, 17, 5}) {
      SCOPED_TRACE(algorithm);
      SCOPED_TRACE(sk);
      OpTester test("KdnnFusedAttention", 2, kKdnnDomain);
      test.AddAttribute<int64_t>("num_heads", 2);
      test.AddAttribute<std::string>("mask_mode", "keep");
      test.AddAttribute<float>("mask_offset", 1.0f);
      test.AddAttribute<float>("mask_multiplier", -10.0f);
      test.AddInput<float>("q", {1, 1, 2, 3}, std::vector<float>(6, 0));
      test.AddInput<float>("k", {1, sk, 2, 3}, std::vector<float>(sk * 6, 0));
      std::vector<float> values(sk * 6, 0);
      std::fill_n(values.begin(), 6, 2.0f);
      test.AddInput<float>("v", {1, sk, 2, 3}, values);
      std::vector<float> mask(sk, 0);
      mask[0] = 1;
      test.AddInput<float>("mask", {1, 1, 1, sk}, mask);
      const float expected = 2.0f / (1.0f + static_cast<float>(sk - 1) * std::exp(-10.0f));
      test.AddOutput<float>("out", {1, 2, 1, 3}, std::vector<float>(6, expected), false, 1e-5f, 1e-5f);
      test.SetNumRunCalls(2);
      RunValidationCase(test);
    }
  }
}
}  // namespace
}  // namespace onnxruntime::test
#endif
