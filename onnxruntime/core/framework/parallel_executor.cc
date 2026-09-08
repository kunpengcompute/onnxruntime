// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.

#include <algorithm>
#include <memory>
#include <sstream>
#include <vector>
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/spin_pause.h"
#include "core/framework/allocation_planner.h"
#include "core/framework/execution_frame.h"
#include "core/framework/session_state.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/utils.h"
#include "core/platform/threadpool.h"
#include "core/framework/parallel_executor.h"

namespace onnxruntime {

ParallelExecutor::ParallelExecutor(const SessionState& session_state, const bool& terminate_flag)
    : terminate_flag_(terminate_flag),
      executor_pool_(session_state.GetInterOpThreadPool()),
      intra_op_pool_(session_state.GetThreadPool()) {
  const auto& graph_viewer = session_state.GetGraphViewer();
  node_refs_size_ = static_cast<size_t>(graph_viewer.MaxNodeIndex());
  node_refs_ = std::make_unique<std::atomic<size_t>[]>(node_refs_size_);
  for (auto& node : graph_viewer.Nodes()) {
    node_refs_[node.Index()].store(node.GetInputEdgesCount(), std::memory_order_relaxed);
  }
}

ParallelExecutor::~ParallelExecutor() {
  done_.store(true, std::memory_order_release);
  queue_cv_.notify_all();
  done_cv_.notify_one();
}

Status ParallelExecutor::Execute(const SessionState& session_state, gsl::span<const int> feed_mlvalue_idxs,
                                 gsl::span<const OrtValue> feeds, gsl::span<const int> fetch_mlvalue_idxs,
                                 std::vector<OrtValue>& fetches,
                                 const std::unordered_map<size_t, CustomAllocator>& fetch_allocators,
                                 const logging::Logger& logger) {
  TimePoint tp;
  const bool is_profiler_enabled = session_state.Profiler().IsEnabled();
  if (is_profiler_enabled) {
    tp = session_state.Profiler().Start();
  }

  root_frame_ = std::make_unique<ExecutionFrame>(feed_mlvalue_idxs, feeds,
                                                  fetch_mlvalue_idxs, fetches,
                                                  fetch_allocators,
#ifdef ORT_ENABLE_STREAM
                                                  nullptr,
#endif
                                                  session_state);

  outstanding_count_.store(0, std::memory_order_relaxed);
  terminate_seen_.store(false, std::memory_order_relaxed);
  done_.store(false, std::memory_order_relaxed);
  active_workers_.store(0, std::memory_order_relaxed);
  errors_.clear();
  ready_queue_.clear();

  const auto& graph_viewer = session_state.GetGraphViewer();
  const auto& root_nodes = graph_viewer.GetRootNodes();

  int num_root_nodes = 0;
  for (auto node_index : root_nodes) {
    auto p_op_kernel = session_state.GetKernel(node_index);
    if (!p_op_kernel)
      continue;
    num_root_nodes++;
    ready_queue_.push_back(node_index);
  }

  if (num_root_nodes == 0) {
    VLOGS(logger, 1) << "No root nodes to execute.";
    ORT_RETURN_IF_ERROR(root_frame_->GetOutputs(fetches));
    if (is_profiler_enabled) {
      session_state.Profiler().EndTimeAndRecordEvent(profiling::SESSION_EVENT, "ParallelExecutor::Execute", tp);
    }
    return Status::OK();
  }

  outstanding_count_.store(num_root_nodes, std::memory_order_relaxed);

  int num_workers = concurrency::ThreadPool::DegreeOfParallelism(executor_pool_);

  if (executor_pool_) {
    executor_pool_->DisableSpinning();
  }
  if (intra_op_pool_) {
    intra_op_pool_->DisableSpinning();
  }

  active_workers_.store(num_workers, std::memory_order_relaxed);

  for (int i = 0; i < num_workers; ++i) {
    onnxruntime::concurrency::ThreadPool::Schedule(executor_pool_,
                                                   [this, &session_state, &logger]() {
                                                     WorkerLoop(session_state, logger);
                                                   });
  }

  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.notify_all();
    while (!done_.load(std::memory_order_acquire) && !terminate_seen_.load(std::memory_order_acquire)) {
      done_cv_.wait(lock);
    }
    while (active_workers_.load(std::memory_order_acquire) != 0) {
      done_cv_.wait(lock);
    }
  }

  if (executor_pool_) {
    executor_pool_->EnableSpinning();
  }
  if (intra_op_pool_) {
    intra_op_pool_->EnableSpinning();
  }

  Status status = Status::OK();

  if (!errors_.empty()) {
    if (errors_.size() == 1)
      status = errors_.front();
    else {
      std::ostringstream ss;
      ss << "Multiple errors were found.";
      for (const auto& s : errors_) {
        ss << '\n'
           << s;
      }
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, ss.str());
    }

    LOGS(logger, ERROR) << status;
    return status;
  }

  VLOGS(logger, 1) << "Fetching output.";
  ORT_RETURN_IF_ERROR(root_frame_->GetOutputs(fetches));
  VLOGS(logger, 1) << "Done execution.";

  if (is_profiler_enabled) {
    session_state.Profiler().EndTimeAndRecordEvent(profiling::SESSION_EVENT, "ParallelExecutor::Execute", tp);
  }

  return Status::OK();
}

void ParallelExecutor::WorkerLoop(const SessionState& session_state, const logging::Logger& logger) {
  [&]() {
  const auto& graph_viewer = session_state.GetGraphViewer();
  constexpr int kSpinBeforeWait = 64;

  while (true) {
    size_t node_index;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);

      int spin_count = 0;
      while (ready_queue_.empty() &&
             !done_.load(std::memory_order_acquire) &&
             !terminate_seen_.load(std::memory_order_acquire)) {
        if (spin_count < kSpinBeforeWait) {
          lock.unlock();
          onnxruntime::concurrency::SpinPause();
          lock.lock();
          ++spin_count;
        } else {
          queue_cv_.wait_for(lock, std::chrono::microseconds(100), [this]() {
            return !ready_queue_.empty() ||
                   done_.load(std::memory_order_acquire) ||
                   terminate_seen_.load(std::memory_order_acquire);
          });
          break;
        }
      }

      if (done_.load(std::memory_order_acquire) || terminate_seen_.load(std::memory_order_acquire)) {
        return;
      }

      if (ready_queue_.empty()) {
        continue;
      }

      node_index = ready_queue_.back();
      ready_queue_.pop_back();
    }

    while (true) {
      Status status = RunOneNode(node_index, session_state, logger);

      if (!status.IsOK()) {
        {
          std::lock_guard<std::mutex> lock(errors_mutex_);
          errors_.push_back(status);
        }
        terminate_seen_.store(true, std::memory_order_release);
        done_cv_.notify_one();
        queue_cv_.notify_all();
        return;
      }

      InlinedVector<size_t, 4> newly_ready;
      {
        const auto& node = *graph_viewer.GetNode(node_index);
        auto begin = node.OutputEdgesBegin();
        auto end = node.OutputEdgesEnd();
        for (auto it = begin; it != end; ++it) {
          auto idx = (*it).GetNode().Index();
          if (node_refs_[idx].fetch_sub(1, std::memory_order_acq_rel) == 1) {
            newly_ready.push_back(idx);
          }
        }
      }

      if (!newly_ready.empty()) {
        int num_new = gsl::narrow<int>(newly_ready.size());
        outstanding_count_.fetch_add(num_new, std::memory_order_relaxed);
      }

      int prev = outstanding_count_.fetch_sub(1, std::memory_order_acq_rel);
      if (prev == 1) {
        done_.store(true, std::memory_order_release);
        done_cv_.notify_one();
        queue_cv_.notify_all();
        return;
      }

      if (terminate_flag_) {
        terminate_seen_.store(true, std::memory_order_release);
        done_cv_.notify_one();
        queue_cv_.notify_all();
        return;
      }

      if (terminate_seen_.load(std::memory_order_acquire)) {
        return;
      }

      if (newly_ready.empty()) {
        break;
      }

      if (newly_ready.size() == 1) {
        node_index = newly_ready[0];
      } else {
        size_t next_node = newly_ready[0];

        {
          std::lock_guard<std::mutex> lock(queue_mutex_);
          for (size_t i = 1; i < newly_ready.size(); ++i) {
            ready_queue_.push_back(newly_ready[i]);
          }
        }

        queue_cv_.notify_one();

        node_index = next_node;
      }
    }
  }
  }();

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    active_workers_.fetch_sub(1, std::memory_order_acq_rel);
    done_cv_.notify_all();
  }
}

Status ParallelExecutor::RunOneNode(size_t node_index, const SessionState& session_state,
                                    const logging::Logger& logger) {
  const auto* p_op_kernel = session_state.GetKernel(node_index);
  const auto& node = *session_state.GetGraphViewer().GetNode(node_index);

  if (p_op_kernel == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Got nullptr from GetKernel for node: ", node.Name());
  }

  OpKernelContextInternal op_kernel_context(session_state, *root_frame_, *p_op_kernel, logger, terminate_flag_,
                                            nullptr, nullptr);

  TimePoint kernel_begin_time;
  const bool f_profiler_enabled = session_state.Profiler().IsEnabled();

  if (f_profiler_enabled) {
    concurrency::ThreadPool::StartProfiling(session_state.GetThreadPool());
    kernel_begin_time = session_state.Profiler().Start();
  }

  VLOGS(logger, 1) << "Computing kernel: " << node.Name();

  Status status;
  ORT_TRY {
    status = p_op_kernel->Compute(&op_kernel_context);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, ex.what());
    });
  }

  if (!status.IsOK()) {
    std::ostringstream ss;
    ss << "Non-zero status code returned while running " << node.OpType() << " node. Name:'" << node.Name()
       << "' Status Message: " << status.ErrorMessage();
    return Status(status.Category(), status.Code(), ss.str());
  }

  if (f_profiler_enabled) {
    session_state.Profiler().EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                                   node.Name() + "_kernel_time",
                                                   kernel_begin_time,
                                                   {{"op_name", p_op_kernel->KernelDef().OpName()},
                                                    {"provider", p_op_kernel->KernelDef().Provider()},
                                                    {"thread_scheduling_stats", 
                                                      concurrency::ThreadPool::StopProfiling(session_state.GetThreadPool())}});
  }

  return Status::OK();
}
}  // namespace onnxruntime
