// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"
#include "core/graph/model.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/test_environment.h"
#include "core/session/inference_session.h"

#include "gtest/gtest.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace test {
  
static std::atomic<int> g_currently_running{0};
static std::atomic<int> g_peak_concurrency{0};

// Test kernel that will return success, or failure, or throw based on the input
struct TestOp {
  static constexpr const char* OpName = "TestOp";
  static constexpr const char* OpDomain = "testing";

  static ONNX_NAMESPACE::OpSchema OpSchema() {
    ONNX_NAMESPACE::OpSchema schema;
    schema.SetDoc("Return success, error, or throw based on the input.")
        .SetName(OpName)
        .SetDomain(OpDomain)
        .SinceVersion(10)
        .Input(0, "action", "Action to take.", "T", OpSchema::Single)
        .Output(0, "action_out", "Return input as is", "T", OpSchema::Single)
        .TypeConstraint("T", {"tensor(int64)"}, "Type of the action and values component");
    return schema;
  }

  class OpKernelImpl final : public OpKernel {
   public:
    OpKernelImpl(const OpKernelInfo& info) : OpKernel{info} {}

    Status Compute(OpKernelContext* ctx) const override {
      const Tensor& action_tensor = *ctx->Input<Tensor>(0);
      const int64_t* action = action_tensor.Data<int64_t>();

      Status status = Status::OK();

      switch (*action) {
        case 0: {
          // success
          Tensor* Y = ctx->Output(0, action_tensor.Shape());
          void* target = Y->MutableData<int64_t>();
          memcpy(target, action, action_tensor.SizeInBytes());
          break;
        }
        case 1: {
          // fail
          status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Action was ", *action);
          break;
        }
        case 3: {
          // delayed success: keeps the worker inside RunOneNode() for a while,
          // widening the window in which another worker may fail and trigger
          // the early-return path of ParallelExecutor::Execute().
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
          Tensor* Y = ctx->Output(0, action_tensor.Shape());
          void* target = Y->MutableData<int64_t>();
          memcpy(target, action, action_tensor.SizeInBytes());
          break;
        }
        case 4: {
          // success while tracking concurrency: used by the fan-out test to
          // measure how many sibling nodes execute at the same time.
          {
            int now = ++g_currently_running;
            int prev_peak = g_peak_concurrency.load();
            while (now > prev_peak &&
                   !g_peak_concurrency.compare_exchange_weak(prev_peak, now)) {
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            --g_currently_running;
          }
          Tensor* Y = ctx->Output(0, action_tensor.Shape());
          void* target = Y->MutableData<int64_t>();
          memcpy(target, action, action_tensor.SizeInBytes());
          break;
        }
        default: {
          ORT_THROW("Throwing as action was ", *action);
        }
      }

      return status;
    }
  };

  static KernelDefBuilder KernelDef() {
    KernelDefBuilder def;
    def.SetName(OpName)
        .SetDomain(OpDomain)
        .SinceVersion(10)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>())
        .Provider(onnxruntime::kCpuExecutionProvider);

    return def;
  }
};

// test that the status from TestOp is correctly returned from InferenceSession::Run
TEST(ParallelExecutor, TestStatusPropagation) {
  auto registry = std::make_shared<CustomRegistry>();
  std::vector<OpSchema> schemas{TestOp::OpSchema()};
  Status status;
  ASSERT_TRUE((status = registry->RegisterOpSet(schemas, TestOp::OpDomain, 10, 11)).IsOK()) << status;
  KernelCreateFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) { out = std::make_unique<typename TestOp::OpKernelImpl>(info); return Status::OK(); };
  auto kernel_def = TestOp::KernelDef();
  ASSERT_TRUE((status = registry->RegisterCustomKernel(kernel_def, kernel_create_fn)).IsOK()) << status;

  {  // test success
    OpTester tester{"TestOp", 10, TestOp::OpDomain};
    tester.AddCustomOpRegistry(registry);

    tester.AddInput<int64_t>("action", {1}, {/*success*/ 0});
    tester.AddOutput<int64_t>("action_out", {1}, {0});
    // TensorRT doesn't handle a custom op. Possibly it should, but that would be a separate PR
    tester.Run(OpTester::ExpectResult::kExpectSuccess, {}, {kTensorrtExecutionProvider}, nullptr, nullptr,
               ExecutionMode::ORT_PARALLEL);
  }

  {  // test success with profiler enabled
    onnxruntime::SessionOptions so;
    so.enable_profiling = true;
    so.execution_mode = ExecutionMode::ORT_PARALLEL;

    OpTester tester{"TestOp", 10, TestOp::OpDomain};
    tester.AddCustomOpRegistry(registry);

    tester.AddInput<int64_t>("action", {1}, {/*success*/ 0});
    tester.AddOutput<int64_t>("action_out", {1}, {0});
    tester.Run(so, OpTester::ExpectResult::kExpectSuccess, {}, {kTensorrtExecutionProvider}, nullptr, nullptr);
  }

  {  // test failure
    OpTester tester{"TestOp", 10, TestOp::OpDomain};
    tester.AddCustomOpRegistry(registry);

    tester.AddInput<int64_t>("action", {1}, {/*failure*/ 1});
    tester.AddOutput<int64_t>("action_out", {1}, {0});
    tester.Run(OpTester::ExpectResult::kExpectFailure, "Action was 1", {kTensorrtExecutionProvider}, nullptr, nullptr,
               ExecutionMode::ORT_PARALLEL);
  }

  {  // test exception
    OpTester tester{"TestOp", 10, TestOp::OpDomain};
    tester.AddCustomOpRegistry(registry);

    tester.AddInput<int64_t>("action", {1}, {/*exception*/ 2});
    tester.AddOutput<int64_t>("action_out", {1}, {0});
    tester.Run(OpTester::ExpectResult::kExpectFailure, "Throwing as action was 2", {kTensorrtExecutionProvider}, nullptr, nullptr, ExecutionMode::ORT_PARALLEL);
  }
}

// Model with two root TestOp nodes (action=1, both fail).
TEST(ParallelExecutor, TestMultiError) {
  auto registry = std::make_shared<CustomRegistry>();
  std::vector<OpSchema> schemas{TestOp::OpSchema()};
  Status status;
  ASSERT_TRUE((status = registry->RegisterOpSet(schemas, TestOp::OpDomain, 10, 11)).IsOK()) << status;
  KernelCreateFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) {
    out = std::make_unique<typename TestOp::OpKernelImpl>(info);
    return Status::OK();
  };
  auto kernel_def = TestOp::KernelDef();
  ASSERT_TRUE((status = registry->RegisterCustomKernel(kernel_def, kernel_create_fn)).IsOK()) << status;

  ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model_proto.add_opset_import();
  opset->set_domain("");
  opset->set_version(14);
  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("multi_error_test");

  auto make_io = [](auto* io_list, const std::string& name) {
    auto* io = io_list->Add();
    io->set_name(name);
    io->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  };
  make_io(graph_proto->mutable_input(), "A");
  make_io(graph_proto->mutable_input(), "B");
  make_io(graph_proto->mutable_output(), "A_out");
  make_io(graph_proto->mutable_output(), "B_out");

  auto make_testop_node = [&](const std::string& name, const std::string& input, const std::string& output) {
    auto* node = graph_proto->add_node();
    node->set_op_type("TestOp");
    node->set_domain(TestOp::OpDomain);
    node->set_name(name);
    node->add_input(input);
    node->add_output(output);
  };
  make_testop_node("test_op_A", "A", "A_out");
  make_testop_node("test_op_B", "B", "B_out");

  std::string model_str;
  ASSERT_TRUE(model_proto.SerializeToString(&model_str));

  const std::vector<int64_t> dims{1};
  std::vector<int64_t> fail_action{1};  // action=1 → failure, non-const for InitOrtValue
  OrtValue feed_a, feed_b;
  Tensor::InitOrtValue(DataTypeImpl::GetType<int64_t>(), TensorShape(dims),
                       fail_action.data(), OrtMemoryInfo(), feed_a);
  Tensor::InitOrtValue(DataTypeImpl::GetType<int64_t>(), TensorShape(dims),
                       fail_action.data(), OrtMemoryInfo(), feed_b);

  SessionOptions so;
  so.execution_mode = ExecutionMode::ORT_PARALLEL;
  InferenceSession session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.RegisterCustomRegistry(registry));
  ASSERT_STATUS_OK(session.Load(model_str.data(), static_cast<int>(model_str.size())));
  ASSERT_STATUS_OK(session.Initialize());

  std::vector<std::string> feed_names{"A", "B"};
  std::vector<OrtValue> feeds{feed_a, feed_b};
  std::vector<std::string> output_names{"A_out", "B_out"};
  std::vector<OrtValue> fetches;
  auto run_status = session.Run(RunOptions{}, feed_names, feeds, output_names, &fetches);
  ASSERT_FALSE(run_status.IsOK());
}

TEST(ParallelExecutor, TestEarlyReturnWaitsForInFlightWorkers) {
  auto registry = std::make_shared<CustomRegistry>();
  std::vector<OpSchema> schemas{TestOp::OpSchema()};
  Status status;
  ASSERT_TRUE((status = registry->RegisterOpSet(schemas, TestOp::OpDomain, 10, 11)).IsOK()) << status;
  KernelCreateFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) {
    out = std::make_unique<typename TestOp::OpKernelImpl>(info);
    return Status::OK();
  };
  auto kernel_def = TestOp::KernelDef();
  ASSERT_TRUE((status = registry->RegisterCustomKernel(kernel_def, kernel_create_fn)).IsOK()) << status;

  ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model_proto.add_opset_import();
  opset->set_domain("");
  opset->set_version(14);
  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("early_return_test");

  auto make_io = [](auto* io_list, const std::string& name) {
    auto* io = io_list->Add();
    io->set_name(name);
    io->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  };
  make_io(graph_proto->mutable_input(), "A");
  make_io(graph_proto->mutable_input(), "B");
  make_io(graph_proto->mutable_output(), "A_out");
  make_io(graph_proto->mutable_output(), "B_out");

  auto make_testop_node = [&](const std::string& name, const std::string& input, const std::string& output) {
    auto* node = graph_proto->add_node();
    node->set_op_type("TestOp");
    node->set_domain(TestOp::OpDomain);
    node->set_name(name);
    node->add_input(input);
    node->add_output(output);
  };
  make_testop_node("test_op_A_immediate_failure", "A", "A_out");
  make_testop_node("test_op_B_delayed_success", "B", "B_out");

  std::string model_str;
  ASSERT_TRUE(model_proto.SerializeToString(&model_str));

  const std::vector<int64_t> dims{1};
  std::vector<int64_t> fail_action{1};   // action=1 → immediate failure
  std::vector<int64_t> delay_action{3};  // action=3 → sleep 200ms, then success
  OrtValue feed_a, feed_b;
  Tensor::InitOrtValue(DataTypeImpl::GetType<int64_t>(), TensorShape(dims),
                       fail_action.data(), OrtMemoryInfo(), feed_a);
  Tensor::InitOrtValue(DataTypeImpl::GetType<int64_t>(), TensorShape(dims),
                       delay_action.data(), OrtMemoryInfo(), feed_b);

  SessionOptions so;
  so.execution_mode = ExecutionMode::ORT_PARALLEL;
  InferenceSession session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.RegisterCustomRegistry(registry));
  ASSERT_STATUS_OK(session.Load(model_str.data(), static_cast<int>(model_str.size())));
  ASSERT_STATUS_OK(session.Initialize());

  std::vector<std::string> feed_names{"A", "B"};
  std::vector<OrtValue> feeds{feed_a, feed_b};
  std::vector<std::string> output_names{"A_out", "B_out"};
  std::vector<OrtValue> fetches;
  auto run_status = session.Run(RunOptions{}, feed_names, feeds, output_names, &fetches);
  ASSERT_FALSE(run_status.IsOK());
  ASSERT_NE(run_status.ErrorMessage().find("Action was 1"), std::string::npos)
      << "Expected the failure from test_op_A to be propagated.";
}

// Graph with no nodes at all: graph input X -> graph output X. Regression
// test for the empty-graph early return: ParallelExecutor::Execute() used to
// return OK without calling root_frame_->GetOutputs(fetches), leaving the
// caller's fetches empty (the sequential executor path always fills them).
TEST(ParallelExecutor, TestEmptyGraphFillsFetches) {
  ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model_proto.add_opset_import();
  opset->set_domain("");
  opset->set_version(14);
  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("empty_graph_test");

  auto make_io = [](auto* io_list, const std::string& name) {
    auto* io = io_list->Add();
    io->set_name(name);
    io->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  };
  make_io(graph_proto->mutable_input(), "X");
  make_io(graph_proto->mutable_output(), "X");  // output aliases input, nodes == []

  std::string model_str;
  ASSERT_TRUE(model_proto.SerializeToString(&model_str));

  const std::vector<int64_t> dims{2};
  std::vector<int64_t> values{7, 9};
  OrtValue feed;
  Tensor::InitOrtValue(DataTypeImpl::GetType<int64_t>(), TensorShape(dims),
                       values.data(), OrtMemoryInfo(), feed);

  SessionOptions so;
  so.execution_mode = ExecutionMode::ORT_PARALLEL;
  InferenceSession session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.Load(model_str.data(), static_cast<int>(model_str.size())));
  ASSERT_STATUS_OK(session.Initialize());

  std::vector<std::string> feed_names{"X"};
  std::vector<OrtValue> feeds{feed};
  std::vector<std::string> output_names{"X"};
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(session.Run(RunOptions{}, feed_names, feeds, output_names, &fetches));

  ASSERT_EQ(fetches.size(), 1u);
  const auto& fetched_tensor = fetches[0].Get<Tensor>();
  const auto& fetched_dims = fetched_tensor.Shape().GetDims();
  ASSERT_EQ(fetched_dims.size(), dims.size());
  ASSERT_EQ(fetched_dims[0], dims[0]);
  const auto* fetched_data = fetched_tensor.Data<int64_t>();
  ASSERT_EQ(fetched_data[0], 7);
  ASSERT_EQ(fetched_data[1], 9);
}

// Graph with a single root node (success) whose output fans out into four
// sibling TestOp nodes (action=4, each sleeping 100ms while tracking
// concurrency). Regression test for the worker-count limitation: workers used
// to be capped by the number of root nodes, so this single-root graph ran all
// four siblings sequentially on one worker (peak concurrency 1, ~400ms).
// With workers sized by the pool's degree of parallelism, siblings execute
// concurrently: peak concurrency > 1 and total time well below 400ms.
TEST(ParallelExecutor, TestSingleRootFanOutRunsSiblingsConcurrently) {
  auto registry = std::make_shared<CustomRegistry>();
  std::vector<OpSchema> schemas{TestOp::OpSchema()};
  Status status;
  ASSERT_TRUE((status = registry->RegisterOpSet(schemas, TestOp::OpDomain, 10, 11)).IsOK()) << status;
  KernelCreateFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) {
    out = std::make_unique<typename TestOp::OpKernelImpl>(info);
    return Status::OK();
  };
  auto kernel_def = TestOp::KernelDef();
  ASSERT_TRUE((status = registry->RegisterCustomKernel(kernel_def, kernel_create_fn)).IsOK()) << status;

  ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model_proto.add_opset_import();
  opset->set_domain("");
  opset->set_version(14);
  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("single_root_fan_out_test");

  auto make_io = [](auto* io_list, const std::string& name) {
    auto* io = io_list->Add();
    io->set_name(name);
    io->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  };
  make_io(graph_proto->mutable_input(), "root_in");
  make_io(graph_proto->mutable_output(), "sib_0_out");
  make_io(graph_proto->mutable_output(), "sib_1_out");
  make_io(graph_proto->mutable_output(), "sib_2_out");
  make_io(graph_proto->mutable_output(), "sib_3_out");

  auto add_testop = [&](const std::string& name, const std::string& input, const std::string& output) {
    auto* node = graph_proto->add_node();
    node->set_op_type("TestOp");
    node->set_domain(TestOp::OpDomain);
    node->set_name(name);
    node->add_input(input);
    node->add_output(output);
  };
  // Single root whose output fans out into 4 siblings. The root gets
  // action=4 (100ms tracked success); it copies its action value to its
  // output, so all 4 siblings also receive action=4. Every kernel in the
  // graph thus participates in the concurrency measurement.
  add_testop("root_op", "root_in", "root_out");
  for (int i = 0; i < 4; ++i) {
    add_testop("sibling_op_" + std::to_string(i), "root_out",
               "sib_" + std::to_string(i) + "_out");
  }

  std::string model_str;
  ASSERT_TRUE(model_proto.SerializeToString(&model_str));

  const std::vector<int64_t> dims{1};
  std::vector<int64_t> root_action{4};
  OrtValue feed;
  Tensor::InitOrtValue(DataTypeImpl::GetType<int64_t>(), TensorShape(dims),
                       root_action.data(), OrtMemoryInfo(), feed);

  SessionOptions so;
  so.execution_mode = ExecutionMode::ORT_PARALLEL;
  so.inter_op_param.thread_pool_size = 4;
  InferenceSession session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.RegisterCustomRegistry(registry));
  ASSERT_STATUS_OK(session.Load(model_str.data(), static_cast<int>(model_str.size())));
  ASSERT_STATUS_OK(session.Initialize());

  g_currently_running.store(0);
  g_peak_concurrency.store(0);

  std::vector<std::string> feed_names{"root_in"};
  std::vector<OrtValue> feeds{feed};
  std::vector<std::string> output_names{"sib_0_out", "sib_1_out", "sib_2_out", "sib_3_out"};
  std::vector<OrtValue> fetches;
  auto begin = std::chrono::steady_clock::now();
  ASSERT_STATUS_OK(session.Run(RunOptions{}, feed_names, feeds, output_names, &fetches));
  auto end = std::chrono::steady_clock::now();
  auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

  // 5 tracked kernels (root + 4 siblings). If siblings run in parallel the
  // wall time is ~200ms (root 100ms + one sibling batch 100ms); if fully
  // sequential it would be ~500ms. Peak concurrency must exceed 1.
  const int peak = g_peak_concurrency.load();
  ASSERT_GT(peak, 1) << "Sibling nodes did not run concurrently (peak concurrency = "
                    << peak << ", elapsed = " << elapsed_ms << "ms)";
}

class ParallelExecutorThreadPoolTest : public testing::TestWithParam<int> {
};

TEST_P(ParallelExecutorThreadPoolTest, TestNullInterOpThreadPool) {
  auto registry = std::make_shared<CustomRegistry>();
  std::vector<OpSchema> schemas{TestOp::OpSchema()};
  Status status;
  ASSERT_TRUE((status = registry->RegisterOpSet(schemas, TestOp::OpDomain, 10, 11)).IsOK()) << status;
  KernelCreateFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) { out = std::make_unique<typename TestOp::OpKernelImpl>(info); return Status::OK(); };
  auto kernel_def = TestOp::KernelDef();
  ASSERT_TRUE((status = registry->RegisterCustomKernel(kernel_def, kernel_create_fn)).IsOK()) << status;

  OpTester tester{"TestOp", 10, TestOp::OpDomain};
  tester.AddCustomOpRegistry(registry);

  tester.AddInput<int64_t>("action", {1}, {/*success*/ 0});
  tester.AddOutput<int64_t>("action_out", {1}, {0});
  // TensorRT doesn't handle a custom op. Possibly it should, but that would be a separate PR
  onnxruntime::SessionOptions so;
  so.session_logid = "TestOp";
  so.session_log_verbosity_level = 1;
  so.execution_mode = ExecutionMode::ORT_PARALLEL;
  so.inter_op_param.thread_pool_size = GetParam();
  tester.Run(so, OpTester::ExpectResult::kExpectSuccess, {}, {kTensorrtExecutionProvider}, nullptr, nullptr);
}

INSTANTIATE_TEST_SUITE_P(ParallelExecutorThreadPoolTests, ParallelExecutorThreadPoolTest,
                         testing::Values(1, 0));
}  // namespace test
}  // namespace onnxruntime
