// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.
#ifdef USE_KDNN_EP
#include "gtest/gtest.h"
#include "core/framework/execution_provider.h"
#include "core/framework/allocator_utils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/kdnn/kdnn_provider_options.h"
#include "core/providers/kdnn/kdnn_provider_factory_creator.h"
#include "test/util/include/scoped_env_vars.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime::test {
TEST(KdnnProviderTest, OptionsAreStrictAndExplicitValuesOverrideEnvironment) {
  ScopedEnvironmentVariables env(EnvVarMap{{"ORT_KDNN_FUSE_ATTENTION", "0"},
                                           {"ORT_KDNN_ATTENTION_ALGORITHM", "classic"}});
  auto options = kdnn::KdnnProviderOptions::Parse({{"enable_attention_fusion", "1"},
                                                   {"attention_algorithm", "classic_no_pack"}});
  EXPECT_TRUE(options.enable_attention_fusion);
  EXPECT_EQ(options.attention_algorithm, kdnn::AttentionAlgorithm::ClassicNoPack);
  EXPECT_FALSE(kdnn::KdnnProviderOptions::Parse({}).enable_attention_fusion);
  EXPECT_ANY_THROW(kdnn::KdnnProviderOptions::Parse({{"enable_attention_fusion", "yes"}}));
  EXPECT_ANY_THROW(kdnn::KdnnProviderOptions::Parse({{"attention_algorithm", "other"}}));
  EXPECT_ANY_THROW(kdnn::KdnnProviderOptions::Parse({{"unknown", "1"}}));
}

TEST(KdnnProviderTest, FactorySnapshotsEnvironment) {
  std::shared_ptr<IExecutionProviderFactory> factory;
  {
    ScopedEnvironmentVariables env(EnvVarMap{{"ORT_KDNN_ATTENTION_ALGORITHM", "classic"}});
    factory = KdnnProviderFactoryCreator::Create();
  }
  ScopedEnvironmentVariables env(EnvVarMap{{"ORT_KDNN_ATTENTION_ALGORITHM", "classic_no_pack"}});
  EXPECT_EQ(factory->CreateProvider()->GetProviderOptions().at("attention_algorithm"), "classic");
}

TEST(KdnnProviderTest, GenericCApiAcceptsBothNamesAndRejectsInvalidOptions) {
  for (const char* name : {"KDNN", "KDNNExecutionProvider"}) {
    Ort::SessionOptions options;
    EXPECT_NO_THROW(options.AppendExecutionProvider(name, {{"attention_algorithm", "auto"}}));
    EXPECT_THROW(options.AppendExecutionProvider(name, {{"enable_attention_fusion", "2"}}), Ort::Exception);
  }
}

TEST(KdnnProviderTest, PreservesSessionCpuArenaSetting) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(8);
  model.add_opset_import()->set_version(13);
  auto* graph = model.mutable_graph();
  graph->set_name("allocator_selection");
  for (auto* arg : {graph->add_input(), graph->add_output()}) {
    auto* type = arg->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);
    type->mutable_shape()->add_dim()->set_dim_value(1);
  }
  graph->mutable_input(0)->set_name("input");
  graph->mutable_output(0)->set_name("output");
  auto* node = graph->add_node();
  node->set_op_type("Identity");
  node->add_input("input");
  node->add_output("output");
  const auto serialized = model.SerializeAsString();
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "kdnn_allocator_test");
  for (bool use_arena : {true, false}) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.AppendExecutionProvider("KDNN", {});
    if (!use_arena) options.DisableCpuMemArena();
    // Allocator selection affects the whole graph, even when KDNN takes no
    // nodes. Check the actual Session allocator, not just EP metadata.
    Ort::Session session(env, serialized.data(), serialized.size(), options);
    auto info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Allocator allocator(session, info);
    const bool expect_arena = use_arena && DoesCpuAllocatorSupportArenaUsage();
    EXPECT_EQ(allocator.GetInfo().GetAllocatorType(), expect_arena ? OrtArenaAllocator : OrtDeviceAllocator);
  }
}
}  // namespace onnxruntime::test
#endif
