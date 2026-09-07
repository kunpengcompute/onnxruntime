// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/graph/constants.h"
#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"
#include "onnx/checker.h"
#include "onnx/shape_inference/implementation.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime {
namespace test {
namespace {

bool HasSchema(const char* op_type, const char* domain, int since_version = 1) {
  return ONNX_NAMESPACE::OpSchemaRegistry::Schema(op_type, since_version, domain) != nullptr;
}

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

std::vector<float> ComputeTensordotMatMulReference(const std::vector<float>& x,
                                                   const std::vector<int64_t>& x_shape,
                                                   const std::vector<float>& w,
                                                   const std::vector<int64_t>& w_shape,
                                                   const std::vector<int64_t>& free_axes) {
  int64_t m = 1;
  for (const auto axis : free_axes) {
    const int64_t normalized_axis = axis < 0 ? axis + static_cast<int64_t>(x_shape.size()) : axis;
    m *= x_shape[static_cast<size_t>(normalized_axis)];
  }

  const int64_t k = x_shape.back();
  const int64_t n = w_shape[1];

  std::vector<float> y(static_cast<size_t>(m * n));
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int64_t kk = 0; kk < k; ++kk) {
        const size_t x_index = static_cast<size_t>(row * k + kk);
        const size_t w_index = static_cast<size_t>(kk * n + col);
        sum += x[x_index] * w[w_index];
      }
      y[static_cast<size_t>(row * n + col)] = sum;
    }
  }

  return y;
}

std::vector<float> MakeSequence(int64_t count, float scale = 0.1f, float offset = -1.0f) {
  std::vector<float> data(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    data[static_cast<size_t>(i)] = offset + scale * static_cast<float>(i % 17);
  }
  return data;
}

#define ORT_ANNC_SKIP_IF_NOT_AARCH64()                                             \
  do {                                                                            \
    if (!IsAarch64Build()) {                                                      \
      GTEST_SKIP() << "The ANNC FusedTensordotMatMul tests require an AArch64 build."; \
    }                                                                             \
  } while (false)

#define ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED()                                    \
  do {                                                                                         \
    ORT_ANNC_SKIP_IF_NOT_AARCH64();                                                            \
    if (!EnvVarEqualsOne("ORT_ENABLE_FUSED_TENSORDOT_MATMUL")) {                               \
      GTEST_SKIP() << "Set ORT_ENABLE_FUSED_TENSORDOT_MATMUL=1 before launching onnxruntime_test_all."; \
    }                                                                                          \
    if (!HasSchema("FusedTensordotMatMul", kMSDomain)) {                                       \
      GTEST_SKIP() << "FusedTensordotMatMul schema is available only after applying the ANNC optimization patch."; \
    }                                                                                          \
  } while (false)

NodeArg* AddShapeGatherReduceProdUnsqueeze(ModelTestBuilder& builder,
                                           NodeArg* input,
                                           const std::vector<int64_t>& gathered_axes,
                                           bool cast_shape_input = false,
                                           bool shape_has_end = false,
                                           bool legacy_axes_attributes = false,
                                           bool wrong_reduce_axis = false,
                                           bool wrong_gather_axis = false,
                                           bool shape_input_without_producer = false,
                                           bool missing_legacy_reduce_axes = false,
                                           bool gather_wrong_op = false) {
  NodeArg* shape = builder.MakeIntermediate();
  if (shape_input_without_producer) {
    shape = builder.Make1DInitializer<int64_t>({2, 3, 4});
  } else {
    Node& shape_node = builder.AddNode("Shape", {input}, {shape});
    if (shape_has_end) {
      shape_node.AddAttribute("end", static_cast<int64_t>(3));
    }
  }

  NodeArg* gather_input = shape;
  if (cast_shape_input) {
    gather_input = builder.MakeIntermediate();
    builder.AddNode("Cast", {shape}, {gather_input})
        .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_INT64));
  }

  NodeArg* gathered_shape = builder.MakeIntermediate();
  if (gather_wrong_op) {
    std::vector<int64_t> gathered_values;
    const std::vector<int64_t> input_shape{2, 3, 4};
    for (const auto axis : gathered_axes) {
      gathered_values.push_back(input_shape[static_cast<size_t>(axis)]);
    }
    builder.AddNode("Identity", {builder.Make1DInitializer<int64_t>(gathered_values)}, {gathered_shape});
  } else {
    builder.AddNode("Gather", {gather_input, builder.Make1DInitializer<int64_t>(gathered_axes)}, {gathered_shape})
        .AddAttribute("axis", static_cast<int64_t>(wrong_gather_axis ? -1 : 0));
  }

  NodeArg* dim_product = builder.MakeIntermediate();
  if (legacy_axes_attributes) {
    Node& reduce_prod = builder.AddNode("ReduceProd", {gathered_shape}, {dim_product});
    if (!missing_legacy_reduce_axes) {
      reduce_prod.AddAttribute("axes", std::vector<int64_t>{0});
    }
    reduce_prod.AddAttribute("keepdims", static_cast<int64_t>(0));
  } else {
    builder.AddNode("ReduceProd",
                    {gathered_shape, builder.Make1DInitializer<int64_t>({wrong_reduce_axis ? -1 : 0})},
                    {dim_product})
        .AddAttribute("keepdims", static_cast<int64_t>(0));
  }

  NodeArg* dim_product_1d = builder.MakeIntermediate();
  if (legacy_axes_attributes) {
    builder.AddNode("Unsqueeze", {dim_product}, {dim_product_1d})
        .AddAttribute("axes", std::vector<int64_t>{0});
  } else {
    builder.AddNode("Unsqueeze", {dim_product, builder.Make1DInitializer<int64_t>({0})}, {dim_product_1d});
  }
  return dim_product_1d;
}

void SetTensorTypeAndShape(NodeArg& node_arg,
                           ONNX_NAMESPACE::TensorProto_DataType elem_type,
                           const std::vector<int64_t>& shape) {
  ONNX_NAMESPACE::TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(elem_type);
  auto* shape_proto = type_proto.mutable_tensor_type()->mutable_shape();
  for (const auto dim : shape) {
    auto* dim_proto = shape_proto->add_dim();
    if (dim >= 0) {
      dim_proto->set_dim_value(dim);
    }
  }
  node_arg.ClearShape();
  ORT_THROW_IF_ERROR(node_arg.UpdateTypeAndShape(type_proto, false, true, logging::LoggingManager::DefaultLogger()));
}

enum class TensordotPatternVariant {
  Good,
  GoodWithCastShape,
  GoodWithShapeInputCast,
  GoodWithNegativeFreeAxis,
  GoodWithLegacyAxesAttributes,
  NoTranspose,
  UnknownInputRank,
  ReduceProdWrongAxis,
  GatherWrongAxis,
  ShapeInputWithoutProducer,
  GatherShapeInputWithoutProducer,
  LegacyReduceProdMissingAxes,
  GatherWrongOp,
  FlattenedIsGraphOutput,
  FinalReshapeIsGraphOutput,
  ShapeEndAttribute,
  ConstantFlattenShape,
  ConcatNegativeAxis,
  UnknownContractDimension,
  NonIdentityTranspose,
  NonLastContractAxis,
  DynamicWeight,
  FinalReshapeAllowZero,
};

void BuildTensordotLoweringPattern(ModelTestBuilder& builder,
                                   TensordotPatternVariant variant = TensordotPatternVariant::Good) {
  NodeArg* x = builder.MakeInput<float>(
      {2, 3, 4},
      {
          0.0f, 1.0f, 2.0f, 3.0f,
          4.0f, 5.0f, 6.0f, 7.0f,
          8.0f, 9.0f, 10.0f, 11.0f,
          12.0f, 13.0f, 14.0f, 15.0f,
          16.0f, 17.0f, 18.0f, 19.0f,
          20.0f, 21.0f, 22.0f, 23.0f,
      });
  SetTensorTypeAndShape(*x, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {-1, 3, 4});
  if (variant == TensordotPatternVariant::UnknownInputRank) {
    x->ClearShape();
  } else if (variant == TensordotPatternVariant::UnknownContractDimension) {
    SetTensorTypeAndShape(*x, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {-1, 3, -1});
  }

  NodeArg* transposed = x;
  if (variant != TensordotPatternVariant::NoTranspose) {
    transposed = builder.MakeIntermediate<float>(
        variant == TensordotPatternVariant::NonIdentityTranspose ? std::optional<std::vector<int64_t>>{{-1, 4, 3}}
                                                                 : std::optional<std::vector<int64_t>>{{-1, 3, 4}});
    builder.AddNode("Transpose", {x}, {transposed})
        .AddAttribute("perm", variant == TensordotPatternVariant::NonIdentityTranspose
                                  ? std::vector<int64_t>{0, 2, 1}
                                  : std::vector<int64_t>{0, 1, 2});
  }

  NodeArg* flatten_shape = nullptr;
  if (variant == TensordotPatternVariant::ConstantFlattenShape) {
    flatten_shape = builder.Make1DInitializer<int64_t>({6, 4});
  } else {
    const bool cast_shape_input = variant == TensordotPatternVariant::GoodWithShapeInputCast;
    const bool shape_has_end = variant == TensordotPatternVariant::ShapeEndAttribute;
    const bool legacy_axes_attributes =
        variant == TensordotPatternVariant::GoodWithLegacyAxesAttributes ||
        variant == TensordotPatternVariant::LegacyReduceProdMissingAxes;
    const std::vector<int64_t> free_axes =
        variant == TensordotPatternVariant::NonLastContractAxis
            ? std::vector<int64_t>{0, 2}
            : variant == TensordotPatternVariant::GoodWithNegativeFreeAxis
                  ? std::vector<int64_t>{0, -2}
                  : std::vector<int64_t>{0, 1};
    NodeArg* free_dim_product = nullptr;
    if (variant == TensordotPatternVariant::ShapeInputWithoutProducer) {
      free_dim_product = builder.Make1DInitializer<int64_t>({6});
    } else {
      free_dim_product =
          AddShapeGatherReduceProdUnsqueeze(builder, x, free_axes, cast_shape_input,
                                            shape_has_end, legacy_axes_attributes,
                                            variant == TensordotPatternVariant::ReduceProdWrongAxis,
                                            variant == TensordotPatternVariant::GatherWrongAxis,
                                            variant == TensordotPatternVariant::GatherShapeInputWithoutProducer,
                                            variant == TensordotPatternVariant::LegacyReduceProdMissingAxes,
                                            variant == TensordotPatternVariant::GatherWrongOp);
    }
    NodeArg* contract_dim_product = AddShapeGatherReduceProdUnsqueeze(
        builder, x,
        variant == TensordotPatternVariant::NonLastContractAxis ? std::vector<int64_t>{1}
                                                                : std::vector<int64_t>{2},
        cast_shape_input, shape_has_end, legacy_axes_attributes);

    NodeArg* concat_output = builder.MakeIntermediate();
    builder.AddNode("Concat", {free_dim_product, contract_dim_product}, {concat_output})
        .AddAttribute("axis", variant == TensordotPatternVariant::ConcatNegativeAxis
                                  ? static_cast<int64_t>(-1)
                                  : static_cast<int64_t>(0));

    flatten_shape = concat_output;
  }

  NodeArg* reshape_shape = flatten_shape;
  if (variant == TensordotPatternVariant::GoodWithCastShape) {
    reshape_shape = builder.MakeIntermediate();
    builder.AddNode("Cast", {flatten_shape}, {reshape_shape})
        .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_INT64));
  }

  NodeArg* flattened =
      variant == TensordotPatternVariant::FlattenedIsGraphOutput
          ? builder.MakeOutput<float>(std::optional<std::vector<int64_t>>{{-1, 4}})
          : builder.MakeIntermediate<float>(
                variant == TensordotPatternVariant::NonLastContractAxis
                    ? std::optional<std::vector<int64_t>>{{-1, 3}}
                    : std::optional<std::vector<int64_t>>{{-1, 4}});
  builder.AddNode("Reshape", {transposed, reshape_shape}, {flattened});

  const std::vector<int64_t> weight_shape =
      variant == TensordotPatternVariant::NonLastContractAxis ? std::vector<int64_t>{3, 5}
                                                              : std::vector<int64_t>{4, 5};
  const std::vector<float> weight_data =
      variant == TensordotPatternVariant::NonLastContractAxis
          ? std::vector<float>{
                0.1f, -0.2f, 0.3f, -0.4f, 0.5f,
                0.6f, -0.7f, 0.8f, -0.9f, 1.0f,
                -1.1f, 1.2f, -1.3f, 1.4f, -1.5f}
          : std::vector<float>{
      0.1f, -0.2f, 0.3f, -0.4f, 0.5f,
      0.6f, -0.7f, 0.8f, -0.9f, 1.0f,
      -1.1f, 1.2f, -1.3f, 1.4f, -1.5f,
                1.6f, -1.7f, 1.8f, -1.9f, 2.0f};
  NodeArg* weight = variant == TensordotPatternVariant::DynamicWeight
                        ? builder.MakeInput<float>(weight_shape, weight_data)
                        : builder.MakeInitializer<float>(weight_shape, weight_data);
  if (variant != TensordotPatternVariant::DynamicWeight) {
    SetTensorTypeAndShape(*weight, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, weight_shape);
  }

  NodeArg* matmul_out = builder.MakeIntermediate<float>(
      variant == TensordotPatternVariant::NonLastContractAxis ? std::optional<std::vector<int64_t>>{{-1, 5}}
                                                              : std::optional<std::vector<int64_t>>{{-1, 5}});
  builder.AddNode("MatMul", {flattened, weight}, {matmul_out});

  NodeArg* final_reshape_out =
      variant == TensordotPatternVariant::FinalReshapeIsGraphOutput
          ? builder.MakeOutput<float>(std::optional<std::vector<int64_t>>{{-1, 3, 5}})
          : builder.MakeIntermediate<float>(
                variant == TensordotPatternVariant::NonLastContractAxis
                    ? std::optional<std::vector<int64_t>>{{-1, 4, 5}}
                    : std::optional<std::vector<int64_t>>{{-1, 3, 5}});
  Node& final_reshape = builder.AddNode("Reshape",
                                        {matmul_out,
                                         variant == TensordotPatternVariant::NonLastContractAxis
                                             ? builder.Make1DInitializer<int64_t>({2, 4, 5})
                                             : builder.Make1DInitializer<int64_t>({2, 3, 5})},
                                        {final_reshape_out});
  if (variant == TensordotPatternVariant::FinalReshapeAllowZero) {
    final_reshape.AddAttribute("allowzero", static_cast<int64_t>(1));
  }

  NodeArg* y = builder.MakeOutput<float>(
      variant == TensordotPatternVariant::NonLastContractAxis ? std::optional<std::vector<int64_t>>{{-1, 4, 5}}
                                                              : std::optional<std::vector<int64_t>>{{-1, 3, 5}});
  // The transformer intentionally skips the pattern if the final Reshape directly produces a graph output.
  // Use a simple downstream op instead of Identity because Identity can be removed by earlier optimizers,
  // making the final Reshape look like a graph output before FusedTensordotMatMulFusion sees it.
  builder.AddNode("Relu", {final_reshape_out}, {y});
}

int GetOpCount(const OpCountMap& op_count, const std::string& op_type) {
  return OpCount(op_count, op_type);
}

void TestTensordotPatternVariant(TensordotPatternVariant variant,
                                 bool expect_fusion,
                                 int opset_version = 18) {
  auto build_test_case = [variant](ModelTestBuilder& builder) {
    BuildTensordotLoweringPattern(builder, variant);
  };

  auto check_transformed_graph = [expect_fusion](InferenceSessionWrapper& session) {
    const auto op_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(GetOpCount(op_count, "com.microsoft.FusedTensordotMatMul"), expect_fusion ? 1 : 0);
    EXPECT_EQ(GetOpCount(op_count, "MatMul"), expect_fusion ? 0 : 1);
  };

  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    opset_version,
                    1e-5,
                    1e-5);
}

}  // namespace

TEST(AnncOptFusedTensordotMatMulTest, KernelRank3LastAxisFloat) {
  if (!HasSchema("FusedTensordotMatMul", kMSDomain)) {
    GTEST_SKIP() << "FusedTensordotMatMul schema is available only after applying the ANNC optimization patch.";
  }

#if !defined(__aarch64__)
  GTEST_SKIP() << "The ANNC FusedTensordotMatMul CPU kernel is registered only for AArch64.";
#else
  const std::vector<int64_t> x_shape{2, 3, 4};
  const std::vector<int64_t> w_shape{4, 5};
  const std::vector<float> x{
      0.0f, 1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f, 7.0f,
      8.0f, 9.0f, 10.0f, 11.0f,
      12.0f, 13.0f, 14.0f, 15.0f,
      16.0f, 17.0f, 18.0f, 19.0f,
      20.0f, 21.0f, 22.0f, 23.0f};
  const std::vector<float> w{
      0.1f, -0.2f, 0.3f, -0.4f, 0.5f,
      0.6f, -0.7f, 0.8f, -0.9f, 1.0f,
      -1.1f, 1.2f, -1.3f, 1.4f, -1.5f,
      1.6f, -1.7f, 1.8f, -1.9f, 2.0f};

  OpTester test("FusedTensordotMatMul", 1, kMSDomain);
  test.AddAttribute("free_axes", std::vector<int64_t>{0, 1});
  test.AddAttribute("contract_axes", std::vector<int64_t>{2});
  test.AddInput<float>("X", x_shape, x);
  test.AddInput<float>("W", w_shape, w);
  test.AddOutput<float>("Y", {2, 3, 5}, ComputeTensordotMatMulReference(x, x_shape, w, w_shape, {0, 1}));
  test.SetOutputAbsErr("Y", 1e-5f);
  test.SetOutputRelErr("Y", 1e-5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
#endif
}

TEST(AnncOptFusedTensordotMatMulTest, KernelRank2MatrixFloat) {
  if (!HasSchema("FusedTensordotMatMul", kMSDomain)) {
    GTEST_SKIP() << "FusedTensordotMatMul schema is available only after applying the ANNC optimization patch.";
  }

#if !defined(__aarch64__)
  GTEST_SKIP() << "The ANNC FusedTensordotMatMul CPU kernel is registered only for AArch64.";
#else
  const std::vector<int64_t> x_shape{3, 4};
  const std::vector<int64_t> w_shape{4, 2};
  const std::vector<float> x = MakeSequence(12, 0.25f, -1.0f);
  const std::vector<float> w = {
      0.1f, -0.2f,
      0.3f, -0.4f,
      0.5f, -0.6f,
      0.7f, -0.8f};

  OpTester test("FusedTensordotMatMul", 1, kMSDomain);
  test.AddAttribute("free_axes", std::vector<int64_t>{0});
  test.AddAttribute("contract_axes", std::vector<int64_t>{1});
  test.AddInput<float>("X", x_shape, x);
  test.AddInput<float>("W", w_shape, w);
  test.AddOutput<float>("Y", {3, 2}, ComputeTensordotMatMulReference(x, x_shape, w, w_shape, {0}));
  test.SetOutputAbsErr("Y", 1e-5f);
  test.SetOutputRelErr("Y", 1e-5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
#endif
}

TEST(AnncOptFusedTensordotMatMulTest, KernelSupportsNegativeAxes) {
  if (!HasSchema("FusedTensordotMatMul", kMSDomain)) {
    GTEST_SKIP() << "FusedTensordotMatMul schema is available only after applying the ANNC optimization patch.";
  }

#if !defined(__aarch64__)
  GTEST_SKIP() << "The ANNC FusedTensordotMatMul CPU kernel is registered only for AArch64.";
#else
  const std::vector<int64_t> x_shape{2, 3, 4};
  const std::vector<int64_t> w_shape{4, 3};
  const std::vector<float> x = MakeSequence(24, 0.125f, -1.5f);
  const std::vector<float> w = MakeSequence(12, -0.05f, 0.75f);

  OpTester test("FusedTensordotMatMul", 1, kMSDomain);
  test.AddAttribute("free_axes", std::vector<int64_t>{0, -2});
  test.AddAttribute("contract_axes", std::vector<int64_t>{-1});
  test.AddInput<float>("X", x_shape, x);
  test.AddInput<float>("W", w_shape, w);
  test.AddOutput<float>("Y", {2, 3, 3}, ComputeTensordotMatMulReference(x, x_shape, w, w_shape, {0, -2}));
  test.SetOutputAbsErr("Y", 1e-5f);
  test.SetOutputRelErr("Y", 1e-5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
#endif
}

TEST(AnncOptFusedTensordotMatMulTest, KernelHandlesEmptyOutput) {
  if (!HasSchema("FusedTensordotMatMul", kMSDomain)) {
    GTEST_SKIP() << "FusedTensordotMatMul schema is available only after applying the ANNC optimization patch.";
  }

#if !defined(__aarch64__)
  GTEST_SKIP() << "The ANNC FusedTensordotMatMul CPU kernel is registered only for AArch64.";
#else
  OpTester test("FusedTensordotMatMul", 1, kMSDomain);
  test.AddAttribute("free_axes", std::vector<int64_t>{0});
  test.AddAttribute("contract_axes", std::vector<int64_t>{1});
  test.AddInput<float>("X", {0, 4}, std::vector<float>{});
  test.AddInput<float>("W", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddOutput<float>("Y", {0, 3}, std::vector<float>{});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
#endif
}

TEST(AnncOptFusedTensordotMatMulTest, KernelZeroContractDimensionReturnsZeros) {
  if (!HasSchema("FusedTensordotMatMul", kMSDomain)) {
    GTEST_SKIP() << "FusedTensordotMatMul schema is available only after applying the ANNC optimization patch.";
  }

#if !defined(__aarch64__)
  GTEST_SKIP() << "The ANNC FusedTensordotMatMul CPU kernel is registered only for AArch64.";
#else
  OpTester test("FusedTensordotMatMul", 1, kMSDomain);
  test.AddAttribute("free_axes", std::vector<int64_t>{0});
  test.AddAttribute("contract_axes", std::vector<int64_t>{1});
  test.AddInput<float>("X", {2, 0}, std::vector<float>{});
  test.AddInput<float>("W", {0, 3}, std::vector<float>{});
  test.AddOutput<float>("Y", {2, 3}, std::vector<float>(6, 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
#endif
}

TEST(AnncOptFusedTensordotMatMulTest, RejectsNonLastContractAxis) {
  if (!HasSchema("FusedTensordotMatMul", kMSDomain)) {
    GTEST_SKIP() << "FusedTensordotMatMul schema is available only after applying the ANNC optimization patch.";
  }

#if !defined(__aarch64__)
  GTEST_SKIP() << "The ANNC FusedTensordotMatMul CPU kernel is registered only for AArch64.";
#else
  OpTester test("FusedTensordotMatMul", 1, kMSDomain);
  test.AddAttribute("free_axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("contract_axes", std::vector<int64_t>{1});
  test.AddInput<float>("X", {2, 3, 4}, std::vector<float>(24, 1.0f));
  test.AddInput<float>("W", {3, 5}, std::vector<float>(15, 1.0f));
  test.AddOutput<float>("Y", {2, 4, 5}, std::vector<float>(40, 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "FusedTensordotMatMul currently requires the contract axis to be the last input dimension.",
           {}, nullptr, &execution_providers);
#endif
}

TEST(AnncOptFusedTensordotMatMulTest, RejectsWeightRankNot2D) {
  if (!HasSchema("FusedTensordotMatMul", kMSDomain)) {
    GTEST_SKIP() << "FusedTensordotMatMul schema is available only after applying the ANNC optimization patch.";
  }

#if !defined(__aarch64__)
  GTEST_SKIP() << "The ANNC FusedTensordotMatMul CPU kernel is registered only for AArch64.";
#else
  OpTester test("FusedTensordotMatMul", 1, kMSDomain);
  test.AddAttribute("free_axes", std::vector<int64_t>{0, 1});
  test.AddAttribute("contract_axes", std::vector<int64_t>{2});
  test.AddInput<float>("X", {2, 3, 4}, std::vector<float>(24, 1.0f));
  test.AddInput<float>("W", {4, 5, 1}, std::vector<float>(20, 1.0f));
  test.AddOutput<float>("Y", {2, 3, 5}, std::vector<float>(30, 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "FusedTensordotMatMul only supports a 2D weight tensor.",
           {}, nullptr, &execution_providers);
#endif
}

TEST(AnncOptFusedTensordotMatMulTest, RejectsMultipleContractAxes) {
  if (!HasSchema("FusedTensordotMatMul", kMSDomain)) {
    GTEST_SKIP() << "FusedTensordotMatMul schema is available only after applying the ANNC optimization patch.";
  }

#if !defined(__aarch64__)
  GTEST_SKIP() << "The ANNC FusedTensordotMatMul CPU kernel is registered only for AArch64.";
#else
  OpTester test("FusedTensordotMatMul", 1, kMSDomain);
  test.AddAttribute("free_axes", std::vector<int64_t>{0});
  test.AddAttribute("contract_axes", std::vector<int64_t>{1, 2});
  test.AddInput<float>("X", {2, 3, 4}, std::vector<float>(24, 1.0f));
  test.AddInput<float>("W", {4, 5}, std::vector<float>(20, 1.0f));
  test.AddOutput<float>("Y", {2, 5}, std::vector<float>(10, 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "FusedTensordotMatMul currently only supports one contract axis.",
           {}, nullptr, &execution_providers);
#endif
}

TEST(AnncOptFusedTensordotMatMulTest, RejectsOutOfRangeFreeAxis) {
  if (!HasSchema("FusedTensordotMatMul", kMSDomain)) {
    GTEST_SKIP() << "FusedTensordotMatMul schema is available only after applying the ANNC optimization patch.";
  }

#if !defined(__aarch64__)
  GTEST_SKIP() << "The ANNC FusedTensordotMatMul CPU kernel is registered only for AArch64.";
#else
  OpTester test("FusedTensordotMatMul", 1, kMSDomain);
  test.AddAttribute("free_axes", std::vector<int64_t>{0, 3});
  test.AddAttribute("contract_axes", std::vector<int64_t>{2});
  test.AddInput<float>("X", {2, 3, 4}, std::vector<float>(24, 1.0f));
  test.AddInput<float>("W", {4, 5}, std::vector<float>(20, 1.0f));
  test.AddOutput<float>("Y", {2, 3, 5}, std::vector<float>(30, 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "free_axes contains an out-of-range axis.",
           {}, nullptr, &execution_providers);
#endif
}

TEST(AnncOptFusedTensordotMatMulTest, RejectsContractDimMismatch) {
  if (!HasSchema("FusedTensordotMatMul", kMSDomain)) {
    GTEST_SKIP() << "FusedTensordotMatMul schema is available only after applying the ANNC optimization patch.";
  }

#if !defined(__aarch64__)
  GTEST_SKIP() << "The ANNC FusedTensordotMatMul CPU kernel is registered only for AArch64.";
#else
  OpTester test("FusedTensordotMatMul", 1, kMSDomain);
  test.AddAttribute("free_axes", std::vector<int64_t>{0, 1});
  test.AddAttribute("contract_axes", std::vector<int64_t>{2});
  test.AddInput<float>("X", {2, 3, 4}, std::vector<float>(24, 1.0f));
  test.AddInput<float>("W", {5, 6}, std::vector<float>(30, 1.0f));
  test.AddOutput<float>("Y", {2, 3, 6}, std::vector<float>(36, 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Input contract dimension must match weight dimension 0.",
           {}, nullptr, &execution_providers);
#endif
}

TEST(AnncOptDynamicExpandTest, ShapeInferenceUsesShapeSourceDim0) {
  if (!HasSchema("DynamicExpand", kMSDomain)) {
    GTEST_SKIP() << "DynamicExpand schema is available only after applying the ANNC optimization patch.";
  }

  ONNX_NAMESPACE::ModelProto model;
  auto* opset = model.add_opset_import();
  opset->set_domain(kMSDomain);
  opset->set_version(1);
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  model.set_producer_name("annc_opt_test");

  auto* graph = model.mutable_graph();
  graph->set_name("dynamic_expand_shape_inference");

  auto* node = graph->add_node();
  node->set_op_type("DynamicExpand");
  node->set_domain(kMSDomain);
  node->add_input("X");
  node->add_input("shape_source");
  node->add_output("Y");

  auto add_input = [](ONNX_NAMESPACE::GraphProto* graph_proto,
                      const std::string& name,
                      ONNX_NAMESPACE::TensorProto_DataType elem_type,
                      const std::vector<int64_t>& shape) {
    auto* value_info = graph_proto->add_input();
    value_info->set_name(name);
    auto* tensor_type = value_info->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(elem_type);
    auto* shape_proto = tensor_type->mutable_shape();
    for (int64_t dim : shape) {
      shape_proto->add_dim()->set_dim_value(dim);
    }
  };

  add_input(graph, "X", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, 4, 8});
  add_input(graph, "shape_source", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {7, 16});

  auto* output = graph->add_output();
  output->set_name("Y");
  output->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  ONNX_NAMESPACE::shape_inference::InferShapes(model, ONNX_NAMESPACE::OpSchemaRegistry::Instance());
  ONNX_NAMESPACE::checker::check_model(model);

  const auto& inferred = model.graph().output(0);
  const auto& tensor_type = inferred.type().tensor_type();
  ASSERT_EQ(tensor_type.elem_type(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  ASSERT_EQ(tensor_type.shape().dim_size(), 3);
  EXPECT_EQ(tensor_type.shape().dim(0).dim_value(), 7);
  EXPECT_EQ(tensor_type.shape().dim(1).dim_value(), 4);
  EXPECT_EQ(tensor_type.shape().dim(2).dim_value(), 8);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, DisabledByDefaultDoesNotFuse) {
  ORT_ANNC_SKIP_IF_NOT_AARCH64();
  if (!HasSchema("FusedTensordotMatMul", kMSDomain)) {
    GTEST_SKIP() << "FusedTensordotMatMul schema is available only after applying the ANNC optimization patch.";
  }
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{{"ORT_ENABLE_FUSED_TENSORDOT_MATMUL", nullopt}}};

  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildTensordotLoweringPattern(builder);
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    const auto op_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(GetOpCount(op_count, "com.microsoft.FusedTensordotMatMul"), 0);
    EXPECT_EQ(GetOpCount(op_count, "MatMul"), 1);
    EXPECT_EQ(GetOpCount(op_count, "Reshape"), 2);
  };

  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    18,
                    1e-5,
                    1e-5);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, PositivePatternFusesAndKeepsOutputs) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();

  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildTensordotLoweringPattern(builder);
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    const auto op_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(GetOpCount(op_count, "com.microsoft.FusedTensordotMatMul"), 1);
    EXPECT_EQ(GetOpCount(op_count, "MatMul"), 0);
    EXPECT_EQ(GetOpCount(op_count, "Reshape"), 0);
  };

  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    18,
                    1e-5,
                    1e-5);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, PositivePatternWithCastShapeFusesAndKeepsOutputs) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();

  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildTensordotLoweringPattern(builder, TensordotPatternVariant::GoodWithCastShape);
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    const auto op_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(GetOpCount(op_count, "com.microsoft.FusedTensordotMatMul"), 1);
    EXPECT_EQ(GetOpCount(op_count, "MatMul"), 0);
    EXPECT_EQ(GetOpCount(op_count, "Reshape"), 0);
  };

  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    18,
                    1e-5,
                    1e-5);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, PositivePatternWithCastBeforeShapeFuses) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();
  TestTensordotPatternVariant(TensordotPatternVariant::GoodWithShapeInputCast, true);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, PositivePatternWithEquivalentNegativeFreeAxisFuses) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();
  TestTensordotPatternVariant(TensordotPatternVariant::GoodWithNegativeFreeAxis, true);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, PositiveLegacyAxesAttributesPatternFuses) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();
  TestTensordotPatternVariant(TensordotPatternVariant::GoodWithLegacyAxesAttributes, true, 11);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, PositiveNoTransposeFuses) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();
  TestTensordotPatternVariant(TensordotPatternVariant::NoTranspose, true);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, NegativeUnknownInputRankDoesNotFuse) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();
  TestTensordotPatternVariant(TensordotPatternVariant::UnknownInputRank, false);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, NegativeReduceProdWrongAxisDoesNotFuse) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();
  TestTensordotPatternVariant(TensordotPatternVariant::ReduceProdWrongAxis, false);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, NegativeGatherWrongAxisDoesNotFuse) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();
  TestTensordotPatternVariant(TensordotPatternVariant::GatherWrongAxis, false);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, NegativeShapeInputWithoutProducerDoesNotFuse) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();
  TestTensordotPatternVariant(TensordotPatternVariant::ShapeInputWithoutProducer, false);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, NegativeGatherShapeInputWithoutProducerDoesNotFuse) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();
  TestTensordotPatternVariant(TensordotPatternVariant::GatherShapeInputWithoutProducer, false);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, NegativeLegacyReduceProdMissingAxesDoesNotFuse) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();
  TestTensordotPatternVariant(TensordotPatternVariant::LegacyReduceProdMissingAxes, false, 11);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, NegativeGatherWrongOpDoesNotFuse) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();
  TestTensordotPatternVariant(TensordotPatternVariant::GatherWrongOp, false);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, NegativeFlattenedGraphOutputDoesNotFuse) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();
  TestTensordotPatternVariant(TensordotPatternVariant::FlattenedIsGraphOutput, false);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, NegativeFinalReshapeGraphOutputDoesNotFuse) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();
  TestTensordotPatternVariant(TensordotPatternVariant::FinalReshapeIsGraphOutput, false);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, NegativeNonIdentityTransposeDoesNotFuse) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();

  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildTensordotLoweringPattern(builder, TensordotPatternVariant::NonIdentityTranspose);
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    const auto op_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(GetOpCount(op_count, "com.microsoft.FusedTensordotMatMul"), 0);
    EXPECT_EQ(GetOpCount(op_count, "MatMul"), 1);
    EXPECT_EQ(GetOpCount(op_count, "Reshape"), 2);
  };

  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    18,
                    1e-5,
                    1e-5);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, NegativeNonLastContractAxisDoesNotFuse) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();

  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildTensordotLoweringPattern(builder, TensordotPatternVariant::NonLastContractAxis);
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    const auto op_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(GetOpCount(op_count, "com.microsoft.FusedTensordotMatMul"), 0);
    EXPECT_EQ(GetOpCount(op_count, "MatMul"), 1);
    EXPECT_EQ(GetOpCount(op_count, "Reshape"), 2);
  };

  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    18,
                    1e-5,
                    1e-5);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, NegativeDynamicWeightDoesNotFuse) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();

  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildTensordotLoweringPattern(builder, TensordotPatternVariant::DynamicWeight);
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    const auto op_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(GetOpCount(op_count, "com.microsoft.FusedTensordotMatMul"), 0);
    EXPECT_EQ(GetOpCount(op_count, "MatMul"), 1);
    EXPECT_EQ(GetOpCount(op_count, "Reshape"), 2);
  };

  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    18,
                    1e-5,
                    1e-5);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, NegativeFinalReshapeAllowZeroDoesNotFuse) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();

  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildTensordotLoweringPattern(builder, TensordotPatternVariant::FinalReshapeAllowZero);
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    const auto op_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(GetOpCount(op_count, "com.microsoft.FusedTensordotMatMul"), 0);
    EXPECT_EQ(GetOpCount(op_count, "MatMul"), 1);
    EXPECT_EQ(GetOpCount(op_count, "Reshape"), 2);
  };

  TransformerTester(build_test_case,
                    check_transformed_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    18,
                    1e-5,
                    1e-5);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, NegativeShapeEndAttributeDoesNotFuse) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();
  TestTensordotPatternVariant(TensordotPatternVariant::ShapeEndAttribute, false);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, NegativeConstantFlattenShapeDoesNotFuse) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();
  TestTensordotPatternVariant(TensordotPatternVariant::ConstantFlattenShape, false);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, NegativeConcatNegativeAxisDoesNotFuse) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();
  TestTensordotPatternVariant(TensordotPatternVariant::ConcatNegativeAxis, false);
}

TEST(AnncOptFusedTensordotMatMulFusionTest, NegativeUnknownContractDimensionDoesNotFuse) {
  ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED();
  TestTensordotPatternVariant(TensordotPatternVariant::UnknownContractDimension, false);
}

#undef ORT_ANNC_SKIP_IF_FUSED_TENSORDOT_MATMUL_DISABLED
#undef ORT_ANNC_SKIP_IF_NOT_AARCH64

}  // namespace test
}  // namespace onnxruntime
