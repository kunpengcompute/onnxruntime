// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/iexecutor.h"
#include "core/framework/framework_common.h"
#include "core/framework/ort_value.h"
#include "core/framework/session_state.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {

class ExecutionFrame;

class ParallelExecutor : public IExecutor {
 public:
  ParallelExecutor(const SessionState& session_state, const bool& terminate_flag = false);
  ~ParallelExecutor();

  common::Status Execute(const SessionState& session_state, gsl::span<const int> feed_mlvalue_idxs,
                         gsl::span<const OrtValue> feeds, gsl::span<const int> fetch_mlvalue_idxs,
                         std::vector<OrtValue>& fetches,
                         const std::unordered_map<size_t, CustomAllocator>& fetch_allocators,
                         const logging::Logger& logger) override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ParallelExecutor);

  Status RunOneNode(size_t node_index, const SessionState& session_state, const logging::Logger& logger);

  void WorkerLoop(const SessionState& session_state, const logging::Logger& logger);

  std::unique_ptr<ExecutionFrame> root_frame_;
  std::unique_ptr<std::atomic<size_t>[]> node_refs_;
  size_t node_refs_size_;

  std::atomic<int> outstanding_count_{0};
  std::atomic<bool> terminate_seen_{false};
  std::atomic<bool> done_{false};

  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::condition_variable done_cv_;
  InlinedVector<size_t, 16> ready_queue_;

  std::mutex errors_mutex_;
  InlinedVector<Status> errors_;

  const bool& terminate_flag_;
  onnxruntime::concurrency::ThreadPool* const executor_pool_;
  onnxruntime::concurrency::ThreadPool* const intra_op_pool_;
};
}  // namespace onnxruntime
