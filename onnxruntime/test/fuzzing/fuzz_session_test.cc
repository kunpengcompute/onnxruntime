// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>

#include "gtest/gtest.h"
#include "OnnxPrediction.h"
#include "onnxruntime_session_options_config_keys.h"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime::test {
namespace {
namespace fs = std::filesystem;

void AddShape(ONNX_NAMESPACE::ValueInfoProto& value, const char* name,
              std::initializer_list<int64_t> shape) {
  value.set_name(name);
  auto* tensor = value.mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);
  auto* tensor_shape = tensor->mutable_shape();
  size_t index = 0;
  for (auto dim : shape) {
    auto* dimension = tensor_shape->add_dim();
    if (dim < 0) {
      dimension->set_dim_param("dynamic_" + std::to_string(index));
    } else {
      dimension->set_dim_value(dim);
    }
    ++index;
  }
}

ONNX_NAMESPACE::NodeProto& AddNode(ONNX_NAMESPACE::GraphProto& graph, const char* op,
                                   std::initializer_list<const char*> inputs, const char* output) {
  auto& node = *graph.add_node();
  node.set_op_type(op);
  for (auto input : inputs) node.add_input(input);
  node.add_output(output);
  return node;
}

ONNX_NAMESPACE::ModelProto AttentionModel(bool fused, int64_t batch = 1) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(8);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);
  auto& graph = *model.mutable_graph();
  graph.set_name("fuzz_attention");
  AddShape(*graph.add_input(), "q", {batch, 1, 2, 4});
  AddShape(*graph.add_input(), "k", {batch, 3, 2, 4});
  AddShape(*graph.add_input(), "v", {batch, 3, 2, 4});
  AddShape(*graph.add_output(), "output", {batch, 2, 1, 4});
  if (fused) {
    opset = model.add_opset_import();
    opset->set_domain("com.kdnn.internal");
    opset->set_version(2);
    auto& node = AddNode(graph, "KdnnFusedAttention", {"q", "k", "v"}, "attention");
    node.set_domain("com.kdnn.internal");
    auto* attr = node.add_attribute();
    attr->set_name("num_heads");
    attr->set_type(ONNX_NAMESPACE::AttributeProto::INT);
    attr->set_i(2);
  } else {
    auto transpose = [&](const char* input, const char* output, std::initializer_list<int64_t> perm) {
      auto* attr = AddNode(graph, "Transpose", {input}, output).add_attribute();
      attr->set_name("perm");
      attr->set_type(ONNX_NAMESPACE::AttributeProto::INTS);
      for (auto dim : perm) attr->add_ints(dim);
    };
    transpose("q", "qt", {0, 2, 1, 3});
    transpose("k", "kt", {0, 2, 3, 1});
    transpose("v", "vt", {0, 2, 1, 3});
    auto* scale = graph.add_initializer();
    scale->set_name("scale");
    scale->set_data_type(ONNX_NAMESPACE::TensorProto::FLOAT);
    scale->add_float_data(0.5f);
    AddNode(graph, "MatMul", {"qt", "kt"}, "scores");
    AddNode(graph, "Mul", {"scores", "scale"}, "scaled");
    auto* axis = AddNode(graph, "Softmax", {"scaled"}, "probabilities").add_attribute();
    axis->set_name("axis");
    axis->set_type(ONNX_NAMESPACE::AttributeProto::INT);
    axis->set_i(-1);
    AddNode(graph, "MatMul", {"probabilities", "vt"}, "attention");
  }
  // Relu must continue to run on CPU when KDNN takes the attention subgraph.
  AddNode(graph, "Relu", {"attention"}, "output");
  return model;
}

ONNX_NAMESPACE::ModelProto IdentityModel(std::initializer_list<int64_t> shape) {
  auto model = AttentionModel(false);
  auto& graph = *model.mutable_graph();
  graph.clear_node();
  graph.clear_initializer();
  graph.clear_input();
  graph.clear_output();
  AddShape(*graph.add_input(), "input", shape);
  AddShape(*graph.add_output(), "output", shape);
  AddNode(graph, "Identity", {"input"}, "output");
  return model;
}

class FuzzSessionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    directory_ = fs::temp_directory_path() /
                 ("ort_fuzz_session_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ASSERT_TRUE(fs::create_directory(directory_));
  }

  void TearDown() override {
    std::error_code error;
    fs::remove_all(directory_, error);
  }

  void CheckAttention(OnnxPrediction& prediction) {
    size_t inputs = 0;
    prediction.SetupInput([&](OnnxPrediction& target, size_t, const std::string& name,
                              ONNXTensorElementDataType type, size_t count, size_t) {
      EXPECT_EQ(type, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      // The driver takes ownership of a std::vector through its input API.
      target << std::vector<float>(count, name == "v" ? 2.0f : 0.0f);
      ++inputs;
    });
    ASSERT_EQ(inputs, 3U);
    prediction.RunInference();
    std::wostringstream output;
    output << prediction;
    EXPECT_EQ(output.str(), L"output = [2, 2, 2, 2, 2, 2, 2, 2]\n");
  }

  // Test the defaults independently of environment variables used by a fuzz campaign.
  ScopedEnvironmentVariables environment_{EnvVarMap{{"ORT_KDNN_FUSE_ATTENTION", std::nullopt},
                                                    {"ORT_KDNN_ATTENTION_ALGORITHM", std::nullopt}}};
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "fuzz_session_test"};
  fs::path directory_;
};

TEST_F(FuzzSessionTest, RunsThreeInputAttentionFromProtoAndFile) {
  for (bool fused : {false, true}) {
#ifndef USE_KDNN_EP
    if (fused) continue;
#endif
    for (int64_t batch : {1, -1}) {
      auto model = AttentionModel(fused, batch);
      OnnxPrediction from_proto(model, env_);
      CheckAttention(from_proto);
      const auto path = directory_ / "attention.onnx";
      {
        std::ofstream file(path, std::ios::binary);
        ASSERT_TRUE(model.SerializeToOstream(&file));
      }
      auto filename = path.wstring();
      OnnxPrediction from_file(filename, env_);
      CheckAttention(from_file);
    }
  }
}

TEST_F(FuzzSessionTest, RunsThreeInputAttentionFromOrtBuffer) {
  for (bool fused : {false, true}) {
#ifndef USE_KDNN_EP
    if (fused) continue;
#endif
    for (int64_t batch : {1, -1}) {
      auto model = AttentionModel(fused, batch).SerializeAsString();
      const auto path = directory_ / "attention.ort";
      auto options = fuzzing::CreateSessionOptions();
      options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
      options.AddConfigEntry(kOrtSessionOptionsConfigSaveModelFormat, "ORT");
      options.SetOptimizedModelFilePath(path.c_str());
      {
        Ort::Session converter(env_, model.data(), model.size(), options);
      }
      std::ifstream file(path, std::ios::binary);
      const std::vector<char> bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
      ASSERT_FALSE(bytes.empty());
      OnnxPrediction from_ort(bytes, env_);
      CheckAttention(from_ort);
    }
  }
}

TEST_F(FuzzSessionTest, HandlesMoreOutputsThanInputs) {
  auto model = AttentionModel(false);
  auto& graph = *model.mutable_graph();
  graph.mutable_input()->DeleteSubrange(1, 2);
  graph.clear_node();
  graph.clear_initializer();
  graph.clear_output();
  AddNode(graph, "Identity", {"q"}, "first");
  AddNode(graph, "Identity", {"q"}, "second");
  AddShape(*graph.add_output(), "first", {1, 1, 2, 4});
  AddShape(*graph.add_output(), "second", {1, 1, 2, 4});
  OnnxPrediction prediction(model, env_);
  prediction.SetupInput(GenerateDataForInputTypeTensor, 42);
  EXPECT_NO_THROW(prediction.RunInference());
}

TEST_F(FuzzSessionTest, PassesLargeInputElementCountToGenerator) {
  auto model = IdentityModel({int64_t{1} << 48});
  OnnxPrediction prediction(model, env_);
  size_t element_count = 0;
  prediction.SetupInput([&](OnnxPrediction&, size_t, const std::string&,
                            ONNXTensorElementDataType, size_t count, size_t) { element_count = count; },
                        42);
  EXPECT_EQ(element_count, size_t{1} << 48);
}

TEST_F(FuzzSessionTest, PreservesScalarEmptyAndConcreteInputShapes) {
  for (auto shape : {std::initializer_list<int64_t>{}, {0, 2}, {3, 4}}) {
    auto model = IdentityModel(shape);
    size_t expected_count = 1;
    for (int64_t dimension : shape) expected_count *= static_cast<size_t>(dimension);
    for (size_t seed : {size_t{42}, size_t{43}}) {
      OnnxPrediction prediction(model, env_);
      prediction.SetupInput([&](OnnxPrediction& target, size_t, const std::string&,
                                ONNXTensorElementDataType, size_t count, size_t data_seed) {
        EXPECT_EQ(count, expected_count);
        EXPECT_EQ(data_seed, seed);
        target << std::vector<float>(count, 2.0f);
      },
                            seed);
      ASSERT_NO_THROW(prediction.RunInference());
      std::wostringstream output;
      output << prediction;
      std::wstring expected = L"output = [";
      for (size_t i = 0; i < expected_count; ++i) expected += i == 0 ? L"2" : L", 2";
      expected += L"]\n";
      EXPECT_EQ(output.str(), expected);
    }
  }
}

TEST_F(FuzzSessionTest, ResolvesSymbolicAndAnonymousInputDimensions) {
  for (auto shape : {std::initializer_list<int64_t>{-1, 4}, {2, -1}, {-1, -1}, {2, -1, 3, -1}, {0, -1}}) {
    size_t expected_count = 1;
    for (int64_t dimension : shape) expected_count *= static_cast<size_t>(dimension < 0 ? 1 : dimension);
    for (bool symbolic : {true, false}) {
      auto model = IdentityModel(shape);
      if (!symbolic) {
        auto& graph = *model.mutable_graph();
        for (auto* value : {graph.mutable_input(0), graph.mutable_output(0)}) {
          for (auto& dimension : *value->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim()) {
            if (dimension.has_dim_param()) dimension.clear_dim_param();
          }
        }
      }
      for (auto element_type : {ONNX_NAMESPACE::TensorProto::FLOAT, ONNX_NAMESPACE::TensorProto::INT32}) {
        auto& graph = *model.mutable_graph();
        for (auto* value : {graph.mutable_input(0), graph.mutable_output(0)}) {
          value->mutable_type()->mutable_tensor_type()->set_elem_type(element_type);
        }
        for (size_t seed : {size_t{42}, size_t{43}}) {
          OnnxPrediction prediction(model, env_);
          size_t generated_inputs = 0;
          prediction.SetupInput([&](OnnxPrediction& target, size_t, const std::string&,
                                    ONNXTensorElementDataType type, size_t count, size_t data_seed) {
            EXPECT_EQ(count, expected_count);
            EXPECT_EQ(data_seed, seed);
            if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
              target << std::vector<float>(count, 2.0f);
            } else {
              target << std::vector<int32_t>(count, 2);
            }
            ++generated_inputs;
          },
                                seed);
          ASSERT_EQ(generated_inputs, 1U);
          ASSERT_NO_THROW(prediction.RunInference());
          std::wostringstream output;
          output << prediction;
          std::wstring expected = L"output = [";
          for (size_t i = 0; i < expected_count; ++i) expected += i == 0 ? L"2" : L", 2";
          expected += L"]\n";
          EXPECT_EQ(output.str(), expected);
        }
      }
    }
  }
}

TEST_F(FuzzSessionTest, RejectsUnknownRankBeforeGeneratingData) {
  auto model = IdentityModel({});
  auto& graph = *model.mutable_graph();
  for (auto* value : {graph.mutable_input(0), graph.mutable_output(0)}) {
    value->mutable_type()->mutable_tensor_type()->clear_shape();
  }
  OnnxPrediction prediction(model, env_);
  size_t generated_inputs = 0;
  EXPECT_THROW(prediction.SetupInput([&](OnnxPrediction&, size_t, const std::string&,
                                         ONNXTensorElementDataType, size_t, size_t) { ++generated_inputs; }),
               std::runtime_error);
  EXPECT_EQ(generated_inputs, 0U);
}

TEST_F(FuzzSessionTest, RejectsElementCountOverflowAfterResolvingDimensions) {
  auto model = IdentityModel({-1, int64_t{1} << 32, int64_t{1} << 32});
  OnnxPrediction prediction(model, env_);
  size_t generated_inputs = 0;
  EXPECT_THROW(prediction.SetupInput([&](OnnxPrediction&, size_t, const std::string&,
                                         ONNXTensorElementDataType, size_t, size_t) { ++generated_inputs; }),
               Ort::Exception);
  EXPECT_EQ(generated_inputs, 0U);
}

TEST_F(FuzzSessionTest, RunsAttentionWithModelBatchAndCpuFallback) {
  for (bool fused : {false, true}) {
#ifndef USE_KDNN_EP
    if (fused) continue;
#endif
    for (int64_t batch : {-1, 1, 2, 3}) {
      auto model = AttentionModel(fused, batch);
      const int64_t resolved_batch = batch < 0 ? 1 : batch;
      for (size_t seed : {size_t{42}, size_t{43}}) {
        OnnxPrediction prediction(model, env_);
        size_t inputs = 0;
        prediction.SetupInput([&](OnnxPrediction& target, size_t, const std::string& name,
                                  ONNXTensorElementDataType type, size_t count, size_t data_seed) {
          EXPECT_EQ(type, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
          EXPECT_EQ(count, static_cast<size_t>(resolved_batch) * (name == "q" ? 1U : 3U) * 2U * 4U);
          EXPECT_EQ(data_seed, seed + inputs);
          target << std::vector<float>(count, name == "v" ? 2.0f : 0.0f);
          ++inputs;
        },
                              seed);
        ASSERT_EQ(inputs, 3U);
        ASSERT_NO_THROW(prediction.RunInference());
        std::wostringstream output;
        output << prediction;
        std::wstring expected = L"output = [";
        for (int64_t i = 0; i < resolved_batch * 8; ++i) expected += i == 0 ? L"2" : L", 2";
        expected += L"]\n";
        EXPECT_EQ(output.str(), expected);
      }
    }
  }
}

TEST_F(FuzzSessionTest, AttentionFusionUsesProviderDefaultsAndEnvironment) {
  for (auto gate : {std::optional<std::string>{}, std::optional<std::string>{"0"}}) {
    ScopedEnvironmentVariables environment(EnvVarMap{{"ORT_KDNN_FUSE_ATTENTION", gate}});
    const auto path = directory_ / "optimized.onnx";
    auto options = fuzzing::CreateSessionOptions();
    options.SetOptimizedModelFilePath(path.c_str());
    const auto model = AttentionModel(false).SerializeAsString();
    {
      Ort::Session session(env_, model.data(), model.size(), options);
    }
    std::ifstream file(path, std::ios::binary);
    ONNX_NAMESPACE::ModelProto optimized;
    ASSERT_TRUE(optimized.ParseFromIstream(&file));
    int attention_nodes = 0;
    for (const auto& node : optimized.graph().node()) {
      if (node.domain() == "com.kdnn.internal" && node.op_type() == "KdnnFusedAttention") ++attention_nodes;
    }
#ifdef USE_KDNN_EP
    EXPECT_EQ(attention_nodes, gate.has_value() ? 0 : 1);
#else
    EXPECT_EQ(attention_nodes, 0);
#endif
  }
}
}  // namespace
}  // namespace onnxruntime::test
