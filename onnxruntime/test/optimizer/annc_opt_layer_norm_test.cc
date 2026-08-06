// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstdlib>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "core/framework/config_options.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/graph/constants.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/cpu/nn/layer_norm_impl.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime {
namespace test {
namespace {

bool EnvVarEqualsOne(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && std::string_view(value) == "1";
}

bool IsAarch64Build() {
#if defined(__aarch64__)
  return true;
#else
  return false;
#endif
}

std::vector<float> MakeSequence(int64_t count, float scale = 0.1f, float offset = -1.0f) {
  std::vector<float> data(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    data[static_cast<size_t>(i)] = offset + scale * static_cast<float>(i % 17);
  }
  return data;
}

std::vector<float> ComputeLayerNormReference(const std::vector<float>& x,
                                             int64_t rows,
                                             int64_t norm_size,
                                             const std::vector<float>& scale,
                                             const std::vector<float>* bias,
                                             float epsilon,
                                             std::vector<float>& means,
                                             std::vector<float>& inv_std_devs) {
  std::vector<float> expected;
  expected.reserve(x.size());
  means.reserve(static_cast<size_t>(rows));
  inv_std_devs.reserve(static_cast<size_t>(rows));
  for (int64_t row = 0; row < rows; ++row) {
    const float* row_x = x.data() + row * norm_size;
    float mean = 0.0f;
    for (int64_t i = 0; i < norm_size; ++i) {
      mean += row_x[i];
    }
    mean /= static_cast<float>(norm_size);

    float variance = 0.0f;
    for (int64_t i = 0; i < norm_size; ++i) {
      const float diff = row_x[i] - mean;
      variance += diff * diff;
    }
    variance /= static_cast<float>(norm_size);

    const float inv_std = 1.0f / std::sqrt(variance + epsilon);
    means.push_back(mean);
    inv_std_devs.push_back(inv_std);
    for (int64_t i = 0; i < norm_size; ++i) {
      float value = (row_x[i] - mean) * inv_std * scale[static_cast<size_t>(i)];
      if (bias != nullptr) {
        value += (*bias)[static_cast<size_t>(i)];
      }
      expected.push_back(value);
    }
  }
  return expected;
}

void RunLayerNormSmokeTest(int64_t rows, int64_t norm_size, bool with_bias) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<int64_t>("axis", -1);
  test.AddAttribute<float>("epsilon", 1e-5f);

  const std::vector<int64_t> x_dims{rows, norm_size};
  const std::vector<float> x = MakeSequence(rows * norm_size, 0.125f, -2.0f);
  const std::vector<float> scale = MakeSequence(norm_size, 0.03125f, 0.5f);
  const std::vector<float> bias = MakeSequence(norm_size, -0.015625f, 0.25f);
  std::vector<float> means;
  std::vector<float> inv_std_devs;
  const std::vector<float> expected =
      ComputeLayerNormReference(x, rows, norm_size, scale, with_bias ? &bias : nullptr, 1e-5f,
                                means, inv_std_devs);

  test.AddInput<float>("X", x_dims, x);
  test.AddInput<float>("Scale", {norm_size}, scale);
  if (with_bias) {
    test.AddInput<float>("B", {norm_size}, bias);
  }
  test.AddOutput<float>("Y", x_dims, expected);
  test.AddOutput<float>("Mean", {rows, 1}, means);
  test.AddOutput<float>("InvStdDev", {rows, 1}, inv_std_devs);
  test.SetOutputAbsErr("Y", 1e-5f);
  test.SetOutputRelErr("Y", 1e-5f);
  test.SetOutputAbsErr("Mean", 1e-5f);
  test.SetOutputRelErr("Mean", 1e-5f);
  test.SetOutputAbsErr("InvStdDev", 1e-5f);
  test.SetOutputRelErr("InvStdDev", 1e-5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

void RunSimplifiedLayerNormSmokeTest(int64_t rows, int64_t norm_size) {
  OpTester test("SimplifiedLayerNormalization", 1, kOnnxDomain);
  test.AddAttribute<int64_t>("axis", -1);
  test.AddAttribute<float>("epsilon", 1e-5f);

  const std::vector<int64_t> x_dims{rows, norm_size};
  const std::vector<float> x = MakeSequence(rows * norm_size, 0.125f, -2.0f);
  const std::vector<float> scale = MakeSequence(norm_size, 0.03125f, 0.5f);
  std::vector<float> expected;
  std::vector<float> inv_std_devs;
  expected.reserve(x.size());
  inv_std_devs.reserve(static_cast<size_t>(rows));

  for (int64_t row = 0; row < rows; ++row) {
    const float* row_x = x.data() + row * norm_size;
    float mean_square = 0.0f;
    for (int64_t i = 0; i < norm_size; ++i) {
      mean_square += row_x[i] * row_x[i];
    }

    const float inv_std = 1.0f / std::sqrt(mean_square / static_cast<float>(norm_size) + 1e-5f);
    inv_std_devs.push_back(inv_std);
    for (int64_t i = 0; i < norm_size; ++i) {
      expected.push_back(row_x[i] * inv_std * scale[static_cast<size_t>(i)]);
    }
  }

  test.AddInput<float>("X", x_dims, x);
  test.AddInput<float>("Scale", {norm_size}, scale);
  test.AddOutput<float>("Y", x_dims, expected);
  test.AddOutput<float>("InvStdDev", {rows, 1}, inv_std_devs);
  test.SetOutputAbsErr("Y", 1e-5f);
  test.SetOutputRelErr("Y", 1e-5f);
  test.SetOutputAbsErr("InvStdDev", 1e-5f);
  test.SetOutputRelErr("InvStdDev", 1e-5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

std::vector<float> ComputeGenericBroadcastLayerNormReference() {
  constexpr float epsilon = 1e-5f;
  const std::vector<float> x{1.0f, 2.0f, 3.0f, 4.0f,
                             5.0f, 6.0f, 7.0f, 8.0f};
  const std::vector<float> scale{0.5f, 1.5f};
  const std::vector<float> bias{0.25f, -0.5f};
  std::vector<float> expected(x.size());

  for (int64_t row = 0; row < 2; ++row) {
    const float* row_x = x.data() + row * 4;
    float mean = 0.0f;
    float mean_square = 0.0f;
    for (int64_t i = 0; i < 4; ++i) {
      mean += row_x[i];
      mean_square += row_x[i] * row_x[i];
    }
    mean /= 4.0f;
    const float denom = std::sqrt(mean_square / 4.0f - mean * mean + epsilon);
    for (int64_t i = 0; i < 4; ++i) {
      const size_t parameter_index = static_cast<size_t>(i / 2);
      expected[static_cast<size_t>(row * 4 + i)] =
          (row_x[i] - mean) / denom * scale[parameter_index] + bias[parameter_index];
    }
  }

  return expected;
}

void RunDoubleGenericBroadcastLayerNormTest() {
  OpTester test("LayerNormalization", 17, kOnnxDomain);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<float>("epsilon", 1e-5f);

  const std::vector<float> expected_float = ComputeGenericBroadcastLayerNormReference();
  const std::vector<double> expected(expected_float.begin(), expected_float.end());
  test.AddInput<double>("X", {2, 2, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
  test.AddInput<double>("Scale", {1, 2, 1}, {0.5, 1.5});
  test.AddInput<double>("B", {1, 2, 1}, {0.25, -0.5});
  test.AddOutput<double>("Y", {2, 2, 2}, expected);
  test.SetOutputAbsErr("Y", 1e-5f);
  test.SetOutputRelErr("Y", 1e-5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

template <typename T>
std::vector<T> ConvertLayerNormValues(const std::vector<float>& values) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return ToFloat16(values);
  } else {
    return std::vector<T>(values.begin(), values.end());
  }
}

template <typename T>
float LayerNormValueToFloat(T value) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return static_cast<float>(value);
  } else {
    return static_cast<float>(value);
  }
}

template <typename T, typename U>
void RunLayerNormWithoutContextSpecialization(bool generic_broadcast) {
  constexpr int64_t axis = 1;
  constexpr float epsilon = 1e-5f;
  Node node;
  node.AddAttribute("axis", axis);
  node.AddAttribute("epsilon", epsilon);

  KernelDef kernel_def;
  auto execution_provider = DefaultCpuExecutionProvider();
  ASSERT_NE(execution_provider, nullptr);
  std::unordered_map<int, OrtValue> constant_initialized_tensors;
  OrtValueNameIdxMap ort_value_name_idx_map;
  DataTransferManager data_transfer_manager;
  AllocatorMap allocators;
  ConfigOptions config_options;
  OpKernelInfo op_kernel_info(node, kernel_def, *execution_provider,
                              constant_initialized_tensors, ort_value_name_idx_map,
                              data_transfer_manager, allocators, config_options);
  LayerNormImpl layer_norm_impl(op_kernel_info);

  const TensorShape x_shape({2, 2, 2});
  const TensorShape parameter_shape(generic_broadcast
                                        ? std::vector<int64_t>{1, 2, 1}
                                        : std::vector<int64_t>{2, 2});
  const std::vector<float> x_float{1.0f, 2.0f, 3.0f, 4.0f,
                                   5.0f, 6.0f, 7.0f, 8.0f};
  const std::vector<float> scale_float = generic_broadcast
                                             ? std::vector<float>{0.5f, 1.5f}
                                             : std::vector<float>{0.5f, 0.75f, 1.25f, 1.5f};
  const std::vector<float> bias_float = generic_broadcast
                                            ? std::vector<float>{0.25f, -0.5f}
                                            : std::vector<float>{0.25f, 0.1f, -0.2f, -0.5f};
  const std::vector<T> x = ConvertLayerNormValues<T>(x_float);
  const std::vector<T> scale = ConvertLayerNormValues<T>(scale_float);
  const std::vector<T> bias = ConvertLayerNormValues<T>(bias_float);
  std::vector<T> y(x.size());
  std::vector<U> mean(2);
  std::vector<U> inv_std_dev(2);

  OrtMemoryInfo memory_info(CPU, OrtAllocatorType::OrtArenaAllocator);
  AllocatorPtr allocator = std::make_shared<CPUAllocator>(memory_info);
  const Status status = layer_norm_impl.ComputeWithoutContext<T, U>(
      x.data(), x_shape, scale.data(), parameter_shape, bias.data(), parameter_shape,
      y.data(), mean.data(), inv_std_dev.data(), nullptr, axis, epsilon, false, allocator);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  const float denom = std::sqrt(1.25f + epsilon);
  for (int64_t row = 0; row < 2; ++row) {
    const float row_mean = row == 0 ? 2.5f : 6.5f;
    EXPECT_NEAR(LayerNormValueToFloat(mean[static_cast<size_t>(row)]), row_mean, 2e-3f);
    EXPECT_NEAR(LayerNormValueToFloat(inv_std_dev[static_cast<size_t>(row)]), 1.0f / denom, 2e-3f);
    for (int64_t i = 0; i < 4; ++i) {
      const size_t parameter_index = generic_broadcast
                                         ? static_cast<size_t>(i / 2)
                                         : static_cast<size_t>(i);
      const float expected =
          (x_float[static_cast<size_t>(row * 4 + i)] - row_mean) / denom *
              scale_float[parameter_index] +
          bias_float[parameter_index];
      EXPECT_NEAR(LayerNormValueToFloat(y[static_cast<size_t>(row * 4 + i)]), expected, 2e-3f);
    }
  }
}

#define ORT_ANNC_SKIP_IF_NOT_AARCH64()                                             \
  do {                                                                            \
    if (!IsAarch64Build()) {                                                      \
      GTEST_SKIP() << "The ANNC LayerNorm graph fusion is compiled only for AArch64."; \
    }                                                                             \
  } while (false)

#define ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED()                                      \
  do {                                                                                \
    ORT_ANNC_SKIP_IF_NOT_AARCH64();                                                   \
    if (!EnvVarEqualsOne("ORT_ENABLE_RM_LN_FUSION")) {                                \
      GTEST_SKIP() << "Set ORT_ENABLE_RM_LN_FUSION=1 before launching onnxruntime_test_all."; \
    }                                                                                 \
  } while (false)

enum class LayerNormPatternVariant {
  Good,
  MismatchedAxes,
  FirstReduceMeanKeepDimsZero,
  ScaleBiasRankMismatch,
  ExtraSubConsumer,
  SquareMulInputsDiffer,
  MissingSecondReduceMean,
  MissingAddEpsilon,
  MissingSqrt,
  MissingReciprocal,
  MissingScaleMul,
  ExtraScaleMulConsumer,
  MissingMeanMulConsumer,
  BiasSubInputsReversed,
  MissingBiasSub,
  MissingFinalAdd,
  DynamicEpsilon,
  MultiDimParameters,
  ScaleBiasDimensionMismatch,
  SquareMulExtraConsumer,
  SecondReduceMeanExtraConsumer,
  AddEpsilonExtraConsumer,
  SqrtExtraConsumer,
  ReciprocalExtraConsumer,
  MeanMulExtraConsumer,
  XMulExtraConsumer,
  BiasSubExtraConsumer,
  WrongFinalAddInput,
  MissingAxes,
  UnknownInputRankPositiveAxes,
  UnknownInputRankNegativeAxes,
  ScaleShapeMissing,
  BiasShapeMissing,
  MissingSquareMul,
  ImplicitKeepDims,
};

Node& AddReduceMean(ModelTestBuilder& builder,
                    NodeArg* input,
                    const std::vector<int64_t>& axes,
                    NodeArg* output,
                    int64_t keepdims = 1,
                    bool omit_keepdims = false) {
  Node& reduce_mean = builder.AddNode("ReduceMean", {input, builder.Make1DInitializer<int64_t>(axes)}, {output});
  if (!omit_keepdims) {
    reduce_mean.AddAttribute("keepdims", keepdims);
  }
  return reduce_mean;
}

void BuildReciprocalMulLayerNormPattern(ModelTestBuilder& builder,
                                        LayerNormPatternVariant variant = LayerNormPatternVariant::Good) {
  const bool keepdims_zero_variant = variant == LayerNormPatternVariant::FirstReduceMeanKeepDimsZero;
  const bool multi_dim_parameters = variant == LayerNormPatternVariant::MultiDimParameters;
  NodeArg* x = multi_dim_parameters
                   ? builder.MakeInput<float>({2, 2, 3}, MakeSequence(12))
                   : keepdims_zero_variant
                   ? builder.MakeInput<float>({2, 2}, {-1.0f, 0.0f, 1.0f, 2.0f})
                   : builder.MakeInput<float>({2, 3}, {-1.0f, 0.0f, 1.0f, 2.0f, 4.0f, 8.0f});
  NodeArg* scale = multi_dim_parameters
                       ? builder.MakeInput<float>({2, 3}, MakeSequence(6, 0.1f, 0.5f))
                       : variant == LayerNormPatternVariant::ScaleBiasRankMismatch
                       ? builder.MakeInput<float>({1, 3}, {1.0f, 0.5f, 2.0f})
                       : keepdims_zero_variant
                       ? builder.MakeInput<float>({2}, {1.0f, 0.5f})
                       : builder.MakeInput<float>({3}, {1.0f, 0.5f, 2.0f});
  NodeArg* bias = multi_dim_parameters
                      ? builder.MakeInput<float>({2, 3}, MakeSequence(6, 0.05f, -0.1f))
                      : variant == LayerNormPatternVariant::ScaleBiasDimensionMismatch
                      ? builder.MakeInput<float>({1}, {0.1f})
                      : keepdims_zero_variant
                      ? builder.MakeInput<float>({2}, {0.1f, -0.2f})
                      : builder.MakeInput<float>({3}, {0.1f, -0.2f, 0.3f});
  if (variant == LayerNormPatternVariant::UnknownInputRankPositiveAxes ||
      variant == LayerNormPatternVariant::UnknownInputRankNegativeAxes) {
    x->ClearShape();
  }
  if (variant == LayerNormPatternVariant::ScaleShapeMissing) {
    scale->ClearShape();
  }
  if (variant == LayerNormPatternVariant::BiasShapeMissing) {
    bias->ClearShape();
  }

  const std::vector<int64_t> axes =
      multi_dim_parameters
          ? std::vector<int64_t>{1, 2}
          : variant == LayerNormPatternVariant::UnknownInputRankNegativeAxes
                ? std::vector<int64_t>{-1}
                : std::vector<int64_t>{1};

  NodeArg* mean = builder.MakeIntermediate();
  if (variant == LayerNormPatternVariant::MissingAxes) {
    builder.AddNode("ReduceMean", {x, builder.MakeInput<int64_t>({1}, {1})}, {mean})
        .AddAttribute("keepdims", static_cast<int64_t>(1));
  } else {
    AddReduceMean(builder, x, axes, mean,
                  variant == LayerNormPatternVariant::FirstReduceMeanKeepDimsZero ? 0 : 1,
                  variant == LayerNormPatternVariant::ImplicitKeepDims);
  }

  NodeArg* centered = builder.MakeIntermediate();
  builder.AddNode("Sub", {x, mean}, {centered});

  if (variant == LayerNormPatternVariant::ExtraSubConsumer) {
    NodeArg* extra_output = builder.MakeOutput();
    builder.AddNode("Identity", {centered}, {extra_output});
  }

  NodeArg* squared = builder.MakeIntermediate();
  builder.AddNode(variant == LayerNormPatternVariant::MissingSquareMul ? "Add" : "Mul",
                  {centered,
                   variant == LayerNormPatternVariant::SquareMulInputsDiffer ? x : centered},
                  {squared});
  if (variant == LayerNormPatternVariant::SquareMulExtraConsumer) {
    NodeArg* extra_output = builder.MakeOutput();
    builder.AddNode("Identity", {squared}, {extra_output});
  }

  NodeArg* variance = builder.MakeIntermediate();
  if (variant == LayerNormPatternVariant::MissingSecondReduceMean) {
    builder.AddNode("Abs", {squared}, {variance});
  } else {
    AddReduceMean(builder, squared,
                  variant == LayerNormPatternVariant::MismatchedAxes ? std::vector<int64_t>{0} : axes,
                  variance, 1,
                  variant == LayerNormPatternVariant::ImplicitKeepDims);
  }
  if (variant == LayerNormPatternVariant::SecondReduceMeanExtraConsumer) {
    NodeArg* extra_output = builder.MakeOutput();
    builder.AddNode("Identity", {variance}, {extra_output});
  }

  NodeArg* variance_plus_eps = builder.MakeIntermediate();
  if (variant == LayerNormPatternVariant::MissingAddEpsilon) {
    builder.AddNode("Abs", {variance}, {variance_plus_eps});
  } else {
    NodeArg* epsilon = variant == LayerNormPatternVariant::DynamicEpsilon
                           ? builder.MakeInput<float>({}, {1e-5f})
                           : builder.MakeScalarInitializer<float>(1e-5f);
    builder.AddNode("Add", {variance, epsilon}, {variance_plus_eps});
  }
  if (variant == LayerNormPatternVariant::AddEpsilonExtraConsumer) {
    NodeArg* extra_output = builder.MakeOutput();
    builder.AddNode("Identity", {variance_plus_eps}, {extra_output});
  }

  NodeArg* stddev = builder.MakeIntermediate();
  builder.AddNode(variant == LayerNormPatternVariant::MissingSqrt ? "Abs" : "Sqrt",
                  {variance_plus_eps}, {stddev});
  if (variant == LayerNormPatternVariant::SqrtExtraConsumer) {
    NodeArg* extra_output = builder.MakeOutput();
    builder.AddNode("Identity", {stddev}, {extra_output});
  }

  NodeArg* inv_stddev = builder.MakeIntermediate();
  if (variant == LayerNormPatternVariant::MissingReciprocal) {
    builder.AddNode("Div", {builder.MakeScalarInitializer<float>(1.0f), stddev}, {inv_stddev});
  } else {
    builder.AddNode("Reciprocal", {stddev}, {inv_stddev});
  }
  if (variant == LayerNormPatternVariant::ReciprocalExtraConsumer) {
    NodeArg* extra_output = builder.MakeOutput();
    builder.AddNode("Identity", {inv_stddev}, {extra_output});
  }

  NodeArg* scaled_inv_stddev = builder.MakeIntermediate();
  builder.AddNode(variant == LayerNormPatternVariant::MissingScaleMul ? "Add" : "Mul",
                  {inv_stddev, scale}, {scaled_inv_stddev});
  if (variant == LayerNormPatternVariant::ExtraScaleMulConsumer) {
    NodeArg* extra_output = builder.MakeOutput();
    builder.AddNode("Relu", {scaled_inv_stddev}, {extra_output});
  }

  NodeArg* mean_mul = builder.MakeIntermediate();
  builder.AddNode("Mul",
                  {variant == LayerNormPatternVariant::MissingMeanMulConsumer ? x : mean,
                   scaled_inv_stddev},
                  {mean_mul});
  if (variant == LayerNormPatternVariant::MeanMulExtraConsumer) {
    NodeArg* extra_output = builder.MakeOutput();
    builder.AddNode("Identity", {mean_mul}, {extra_output});
  }

  NodeArg* x_mul = builder.MakeIntermediate();
  builder.AddNode("Mul", {x, scaled_inv_stddev}, {x_mul});
  if (variant == LayerNormPatternVariant::XMulExtraConsumer) {
    NodeArg* extra_output = builder.MakeOutput();
    builder.AddNode("Identity", {x_mul}, {extra_output});
  }

  NodeArg* bias_sub = builder.MakeIntermediate();
  if (variant == LayerNormPatternVariant::MissingBiasSub) {
    builder.AddNode("Add", {bias, mean_mul}, {bias_sub});
  } else {
    builder.AddNode("Sub",
                    variant == LayerNormPatternVariant::BiasSubInputsReversed
                        ? std::vector<NodeArg*>{mean_mul, bias}
                        : std::vector<NodeArg*>{bias, mean_mul},
                    {bias_sub});
  }
  if (variant == LayerNormPatternVariant::BiasSubExtraConsumer) {
    NodeArg* extra_output = builder.MakeOutput();
    builder.AddNode("Identity", {bias_sub}, {extra_output});
  }

  NodeArg* y = builder.MakeOutput();
  builder.AddNode(variant == LayerNormPatternVariant::MissingFinalAdd ? "Sub" : "Add",
                  variant == LayerNormPatternVariant::WrongFinalAddInput
                      ? std::vector<NodeArg*>{mean_mul, bias_sub}
                      : std::vector<NodeArg*>{x_mul, bias_sub},
                  {y});
}

int GetOpCount(const OpCountMap& op_count, const std::string& op_type) {
  return OpCount(op_count, op_type);
}

void TestReciprocalMulLayerNormVariant(LayerNormPatternVariant variant,
                                       bool expect_fusion = false,
                                       int opset_version = 18) {
  auto build_test_case = [variant](ModelTestBuilder& builder) {
    BuildReciprocalMulLayerNormPattern(builder, variant);
  };

  auto check_transformed_graph = [expect_fusion](InferenceSessionWrapper& session) {
    const auto op_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(GetOpCount(op_count, "LayerNormalization"), expect_fusion ? 1 : 0);
  };

  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    opset_version,
                    1e-5,
                    1e-5);
}

enum class StandardLayerNormPatternVariant {
  UnknownInputRank,
  NonConsecutiveAxes,
  ScaleShapeMissing,
};

void BuildStandardLayerNormPattern(ModelTestBuilder& builder,
                                   StandardLayerNormPatternVariant variant) {
  NodeArg* x = builder.MakeInput<float>({2, 2, 3}, MakeSequence(12));
  if (variant == StandardLayerNormPatternVariant::UnknownInputRank) {
    x->ClearShape();
  }

  NodeArg* scale = builder.MakeInput<float>({3}, {1.0f, 0.5f, 2.0f});
  if (variant == StandardLayerNormPatternVariant::ScaleShapeMissing) {
    scale->ClearShape();
  }
  NodeArg* bias = builder.MakeInput<float>({3}, {0.1f, -0.2f, 0.3f});
  const std::vector<int64_t> axes =
      variant == StandardLayerNormPatternVariant::NonConsecutiveAxes
          ? std::vector<int64_t>{0, 2}
          : std::vector<int64_t>{2};

  NodeArg* mean = builder.MakeIntermediate();
  AddReduceMean(builder, x, axes, mean);

  NodeArg* centered = builder.MakeIntermediate();
  builder.AddNode("Sub", {x, mean}, {centered});

  NodeArg* squared = builder.MakeIntermediate();
  builder.AddNode("Pow", {centered, builder.MakeScalarInitializer<float>(2.0f)}, {squared});

  NodeArg* variance = builder.MakeIntermediate();
  AddReduceMean(builder, squared, axes, variance);

  NodeArg* variance_plus_eps = builder.MakeIntermediate();
  builder.AddNode("Add", {variance, builder.MakeScalarInitializer<float>(1e-5f)}, {variance_plus_eps});

  NodeArg* stddev = builder.MakeIntermediate();
  builder.AddNode("Sqrt", {variance_plus_eps}, {stddev});

  NodeArg* normalized = builder.MakeIntermediate();
  builder.AddNode("Div", {centered, stddev}, {normalized});

  NodeArg* scaled = builder.MakeIntermediate();
  builder.AddNode("Mul", {normalized, scale}, {scaled});

  NodeArg* y = builder.MakeOutput();
  builder.AddNode("Add", {scaled, bias}, {y});
}

void TestStandardLayerNormVariant(StandardLayerNormPatternVariant variant) {
  auto build_test_case = [variant](ModelTestBuilder& builder) {
    BuildStandardLayerNormPattern(builder, variant);
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    const auto op_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(GetOpCount(op_count, "LayerNormalization"), 0);
  };

  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    18,
                    1e-5,
                    1e-5);
}

}  // namespace

TEST(AnncOptLayerNormDefaultKernelTest, FloatWithBias) {
  if (EnvVarEqualsOne("ORT_ENABLE_LAYER_NORM_NEON")) {
    GTEST_SKIP() << "Run this default-kernel control test without ORT_ENABLE_LAYER_NORM_NEON=1.";
  }
  RunLayerNormSmokeTest(2, 20, true);
}

TEST(AnncOptLayerNormDefaultKernelTest, FloatWithoutBias) {
  if (EnvVarEqualsOne("ORT_ENABLE_LAYER_NORM_NEON")) {
    GTEST_SKIP() << "Run this default-kernel control test without ORT_ENABLE_LAYER_NORM_NEON=1.";
  }
  RunLayerNormSmokeTest(2, 20, false);
}

TEST(AnncOptLayerNormDefaultKernelTest, OnnxDoubleGenericBroadcast) {
  RunDoubleGenericBroadcastLayerNormTest();
}

// The double statistics specialization was previously exercised by calling
// LayerNormImpl::ComputeWithoutContext<double, double> directly, which required
// an explicit template instantiation in the production source solely for the
// test. Drive the same path through the standard ONNX LayerNormalization op
// (registered for double) instead, and additionally verify the float Mean and
// InvStdDev statistics outputs.
TEST(AnncOptLayerNormDefaultKernelTest, DoubleExactMatchParameters) {
  constexpr float kEpsilon = 1e-5f;
  constexpr double kInvStdDev = 1.0 / std::sqrt(1.25 + static_cast<double>(kEpsilon));
  // scale & bias shape {2,2} exactly matches the normalized axes.
  // Row means: r0=2.5, r1=6.5; variance=1.25; inv_std_dev≈0.894427.
  const std::vector<double> kExpectedY{
      (-1.5 * kInvStdDev) * 0.5 + 0.25,    // r0,c0: scale=0.50, bias= 0.25
      (-0.5 * kInvStdDev) * 0.75 + 0.1,    // r0,c1: scale=0.75, bias= 0.10
      (0.5 * kInvStdDev) * 1.25 - 0.2,     // r0,c2: scale=1.25, bias=-0.20
      (1.5 * kInvStdDev) * 1.5 - 0.5,      // r0,c3: scale=1.50, bias=-0.50
      (-1.5 * kInvStdDev) * 0.5 + 0.25,    // r1,c0
      (-0.5 * kInvStdDev) * 0.75 + 0.1,    // r1,c1
      (0.5 * kInvStdDev) * 1.25 - 0.2,     // r1,c2
      (1.5 * kInvStdDev) * 1.5 - 0.5};     // r1,c3

  OpTester test("LayerNormalization", 17, kOnnxDomain);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<float>("epsilon", kEpsilon);
  test.AddInput<double>("X", {2, 2, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
  test.AddInput<double>("Scale", {2, 2}, {0.5, 0.75, 1.25, 1.5});
  test.AddInput<double>("B", {2, 2}, {0.25, 0.1, -0.2, -0.5});
  test.AddOutput<double>("Y", {2, 2, 2}, kExpectedY);
  // ONNX LayerNormalization always emits float statistics even for double inputs.
  test.AddOutput<float>("Mean", {2, 1, 1}, {2.5f, 6.5f});
  test.AddOutput<float>("InvStdDev", {2, 1, 1}, {static_cast<float>(kInvStdDev), static_cast<float>(kInvStdDev)});
  test.SetOutputAbsErr("Y", 1e-5f);
  test.SetOutputRelErr("Y", 1e-5f);
  test.SetOutputAbsErr("Mean", 1e-5f);
  test.SetOutputRelErr("Mean", 1e-5f);
  test.SetOutputAbsErr("InvStdDev", 1e-5f);
  test.SetOutputRelErr("InvStdDev", 1e-5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(AnncOptLayerNormDefaultKernelTest, DoubleGenericBroadcastParameters) {
  constexpr float kEpsilon = 1e-5f;
  constexpr double kInvStdDev = 1.0 / std::sqrt(1.25 + static_cast<double>(kEpsilon));
  // scale & bias shape {1,2,1} broadcasts over the normalized axes. The two
  // parameter values map to the two groups of the flattened normalized dim.
  const std::vector<double> kExpectedY{
      (-1.5 * kInvStdDev) * 0.5 + 0.25,    // r0,c0: scale=0.50, bias= 0.25
      (-0.5 * kInvStdDev) * 0.5 + 0.25,    // r0,c1: scale=0.50, bias= 0.25
      (0.5 * kInvStdDev) * 1.5 - 0.5,      // r0,c2: scale=1.50, bias=-0.50
      (1.5 * kInvStdDev) * 1.5 - 0.5,      // r0,c3: scale=1.50, bias=-0.50
      (-1.5 * kInvStdDev) * 0.5 + 0.25,    // r1,c0
      (-0.5 * kInvStdDev) * 0.5 + 0.25,    // r1,c1
      (0.5 * kInvStdDev) * 1.5 - 0.5,      // r1,c2
      (1.5 * kInvStdDev) * 1.5 - 0.5};     // r1,c3

  OpTester test("LayerNormalization", 17, kOnnxDomain);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<float>("epsilon", kEpsilon);
  test.AddInput<double>("X", {2, 2, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
  test.AddInput<double>("Scale", {1, 2, 1}, {0.5, 1.5});
  test.AddInput<double>("B", {1, 2, 1}, {0.25, -0.5});
  test.AddOutput<double>("Y", {2, 2, 2}, kExpectedY);
  test.AddOutput<float>("Mean", {2, 1, 1}, {2.5f, 6.5f});
  test.AddOutput<float>("InvStdDev", {2, 1, 1}, {static_cast<float>(kInvStdDev), static_cast<float>(kInvStdDev)});
  test.SetOutputAbsErr("Y", 1e-5f);
  test.SetOutputRelErr("Y", 1e-5f);
  test.SetOutputAbsErr("Mean", 1e-5f);
  test.SetOutputRelErr("Mean", 1e-5f);
  test.SetOutputAbsErr("InvStdDev", 1e-5f);
  test.SetOutputRelErr("InvStdDev", 1e-5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(AnncOptLayerNormDefaultKernelTest, DirectFloat16StatisticsSpecialization) {
  RunLayerNormWithoutContextSpecialization<MLFloat16, MLFloat16>(true);
  RunLayerNormWithoutContextSpecialization<MLFloat16, MLFloat16>(false);
}

TEST(AnncOptLayerNormNeonTest, FloatWithBias) {
  if (!EnvVarEqualsOne("ORT_ENABLE_LAYER_NORM_NEON")) {
    GTEST_SKIP() << "Set ORT_ENABLE_LAYER_NORM_NEON=1 to validate the ANNC LayerNorm NEON path.";
  }
  if (!IsAarch64Build()) {
    GTEST_SKIP() << "The ANNC LayerNorm NEON implementation is compiled only for AArch64.";
  }
  // The optimized implementation requires norm_size >= 256. Use a non-multiple
  // of both 16 and 4 so the vectorized loops and scalar tail are all exercised.
  RunLayerNormSmokeTest(2, 263, true);
}

TEST(AnncOptLayerNormNeonTest, FloatWithoutBias) {
  if (!EnvVarEqualsOne("ORT_ENABLE_LAYER_NORM_NEON")) {
    GTEST_SKIP() << "Set ORT_ENABLE_LAYER_NORM_NEON=1 to validate the ANNC LayerNorm NEON path.";
  }
  if (!IsAarch64Build()) {
    GTEST_SKIP() << "The ANNC LayerNorm NEON implementation is compiled only for AArch64.";
  }
  // This shape exercises the 16-wide and 4-wide vectorized loops in the
  // no-bias branch while satisfying the NEON dispatch threshold.
  RunLayerNormSmokeTest(2, 260, false);
}

TEST(AnncOptLayerNormNeonTest, SimplifiedFloat) {
  if (!EnvVarEqualsOne("ORT_ENABLE_LAYER_NORM_NEON")) {
    GTEST_SKIP() << "Set ORT_ENABLE_LAYER_NORM_NEON=1 to validate the ANNC LayerNorm NEON path.";
  }
  if (!IsAarch64Build()) {
    GTEST_SKIP() << "The ANNC LayerNorm NEON implementation is compiled only for AArch64.";
  }
  RunSimplifiedLayerNormSmokeTest(2, 263);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, DisabledByDefaultDoesNotFuse) {
  ORT_ANNC_SKIP_IF_NOT_AARCH64();
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{{"ORT_ENABLE_RM_LN_FUSION", nullopt}}};

  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildReciprocalMulLayerNormPattern(builder);
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    const auto op_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(GetOpCount(op_count, "LayerNormalization"), 0);
    EXPECT_EQ(GetOpCount(op_count, "ReduceMean"), 2);
    EXPECT_EQ(GetOpCount(op_count, "Reciprocal"), 1);
  };

  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    18,
                    1e-5,
                    1e-5);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, PositivePatternFusesAndKeepsOutputs) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();

  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildReciprocalMulLayerNormPattern(builder);
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    const auto op_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(GetOpCount(op_count, "LayerNormalization"), 1);
    EXPECT_EQ(GetOpCount(op_count, "ReduceMean"), 0);
    EXPECT_EQ(GetOpCount(op_count, "Reciprocal"), 0);
    EXPECT_EQ(GetOpCount(op_count, "Sqrt"), 0);
  };

  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    18,
                    1e-5,
                    1e-5);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeMismatchedAxesDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();

  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildReciprocalMulLayerNormPattern(builder, LayerNormPatternVariant::MismatchedAxes);
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    const auto op_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(GetOpCount(op_count, "LayerNormalization"), 0);
    EXPECT_EQ(GetOpCount(op_count, "ReduceMean"), 2);
    EXPECT_EQ(GetOpCount(op_count, "Reciprocal"), 1);
  };

  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    18,
                    1e-5,
                    1e-5);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeKeepDimsZeroDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();

  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildReciprocalMulLayerNormPattern(builder, LayerNormPatternVariant::FirstReduceMeanKeepDimsZero);
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    const auto op_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(GetOpCount(op_count, "LayerNormalization"), 0);
    EXPECT_EQ(GetOpCount(op_count, "ReduceMean"), 2);
  };

  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    18,
                    1e-5,
                    1e-5);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeScaleBiasRankMismatchDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();

  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildReciprocalMulLayerNormPattern(builder, LayerNormPatternVariant::ScaleBiasRankMismatch);
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    const auto op_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(GetOpCount(op_count, "LayerNormalization"), 0);
    EXPECT_EQ(GetOpCount(op_count, "ReduceMean"), 2);
  };

  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    18,
                    1e-5,
                    1e-5);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeExtraSubConsumerDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();

  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildReciprocalMulLayerNormPattern(builder, LayerNormPatternVariant::ExtraSubConsumer);
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    const auto op_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(GetOpCount(op_count, "LayerNormalization"), 0);
    EXPECT_EQ(GetOpCount(op_count, "ReduceMean"), 2);
    EXPECT_EQ(GetOpCount(op_count, "Identity"), 1);
  };

  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    18,
                    1e-5,
                    1e-5);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeSquareMulInputsDifferDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::SquareMulInputsDiffer);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeMissingSecondReduceMeanDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::MissingSecondReduceMean);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeMissingAddEpsilonDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::MissingAddEpsilon);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeMissingSqrtDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::MissingSqrt);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeMissingReciprocalDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::MissingReciprocal);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeMissingScaleMulDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::MissingScaleMul);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeExtraScaleMulConsumerDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::ExtraScaleMulConsumer);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeMissingMeanMulConsumerDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::MissingMeanMulConsumer);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeBiasSubInputsReversedDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::BiasSubInputsReversed);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeMissingBiasSubDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::MissingBiasSub);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeMissingFinalAddDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::MissingFinalAdd);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, DynamicEpsilonUsesDefaultAndFuses) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::DynamicEpsilon, true);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeMultiDimParametersDoNotFuseInInferenceBuild) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::MultiDimParameters);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeScaleBiasDimensionMismatchDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::ScaleBiasDimensionMismatch);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeSquareMulExtraConsumerDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::SquareMulExtraConsumer);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeSecondReduceMeanExtraConsumerDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::SecondReduceMeanExtraConsumer);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeAddEpsilonExtraConsumerDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::AddEpsilonExtraConsumer);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeSqrtExtraConsumerDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::SqrtExtraConsumer);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeReciprocalExtraConsumerDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::ReciprocalExtraConsumer);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeMeanMulExtraConsumerDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::MeanMulExtraConsumer);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeXMulExtraConsumerDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::XMulExtraConsumer);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeBiasSubExtraConsumerDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::BiasSubExtraConsumer);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeWrongFinalAddInputDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::WrongFinalAddInput);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeMissingAxesDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::MissingAxes);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, UnknownInputRankPositiveAxesFuses) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::UnknownInputRankPositiveAxes, true);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, UnknownInputRankNegativeAxesFuses) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::UnknownInputRankNegativeAxes, true);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeScaleShapeMissingDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::ScaleShapeMissing);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeBiasShapeMissingDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::BiasShapeMissing);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, NegativeMissingSquareMulDoesNotFuse) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::MissingSquareMul);
}

TEST(AnncOptReciprocalMulLayerNormFusionTest, ImplicitKeepDimsFuses) {
  ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED();
  TestReciprocalMulLayerNormVariant(LayerNormPatternVariant::ImplicitKeepDims, true);
}

TEST(AnncOptStandardLayerNormFusionTest, UnknownInputRankDoesNotFuse) {
  ORT_ANNC_SKIP_IF_NOT_AARCH64();
  TestStandardLayerNormVariant(StandardLayerNormPatternVariant::UnknownInputRank);
}

TEST(AnncOptStandardLayerNormFusionTest, NonConsecutiveAxesDoNotFuse) {
  ORT_ANNC_SKIP_IF_NOT_AARCH64();
  TestStandardLayerNormVariant(StandardLayerNormPatternVariant::NonConsecutiveAxes);
}

TEST(AnncOptStandardLayerNormFusionTest, ScaleShapeMissingDoesNotFuse) {
  ORT_ANNC_SKIP_IF_NOT_AARCH64();
  TestStandardLayerNormVariant(StandardLayerNormPatternVariant::ScaleShapeMissing);
}


#undef ORT_ANNC_SKIP_IF_RM_LN_FUSION_DISABLED
#undef ORT_ANNC_SKIP_IF_NOT_AARCH64

}  // namespace test
}  // namespace onnxruntime
