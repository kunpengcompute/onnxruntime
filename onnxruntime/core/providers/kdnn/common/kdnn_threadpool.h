// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.

#pragma once

#if defined(USE_KDNN)

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>

#include "core/common/common.h"
#include "core/platform/threadpool.h"

#include "service/kdnn_threading.hpp"

namespace onnxruntime {
namespace kdnn {

// Adapts an ORT operator thread pool to KDNN's synchronous parallel-for API.
// KDNN propagates the active adapter to worker threads while a parallel region
// is running. The callback-local depth prevents nested KDNN work from trying to
// enter the same ORT pool recursively.
class ThreadPoolAdapter final : public KDNN::Threading::ThreadpoolIface {
public:
  explicit ThreadPoolAdapter(concurrency::ThreadPool* thread_pool) noexcept
      : thread_pool_(thread_pool) {}

  int GetNumThreads() const override {
    return concurrency::ThreadPool::DegreeOfParallelism(thread_pool_);
  }

  bool IsInParallel() const override {
    return ParallelDepth() != 0;
  }

  // Still required by ThreadpoolIface although current KDNN code never reads
  // it; report synchronous execution.
  uint64_t GetFlags() const override {
    return 0;
  }

  void ParallelFor(int count, int64_t cost_per_unit,
                   const std::function<void(int, int)>& fn) override {
    // 'count' is the total number of work units and this adapter owns the
    // range partitioning. Distribute the units directly across all pool
    // threads with unit granularity: fn(idx, idx + 1) covers [0, count) and
    // SimpleParallelFor load-balances dynamically, without ORT's CostModel,
    // which may underestimate the task cost and under-parallelize (observed:
    // 7.7/16 CPUs utilized vs 15.8 with OMP). The cost hint stays unused:
    // non-Eigen adapters get conservative KDNN work plans anyway.
    concurrency::ThreadPool::TrySimpleParallelFor(
        thread_pool_, count,
        [&](std::ptrdiff_t idx) {
          ParallelRegion region;
          fn(static_cast<int>(idx), static_cast<int>(idx) + 1);
        });
  }

private:
  static int& ParallelDepth() {
    static thread_local int depth = 0;
    return depth;
  }

  class ParallelRegion final {
  public:
    ParallelRegion() { ++ParallelDepth(); }
    ~ParallelRegion() { --ParallelDepth(); }

    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ParallelRegion);
  };

  concurrency::ThreadPool* thread_pool_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ThreadPoolAdapter);
};

// Activation must begin before primitive construction: KDNN uses the effective
// thread count as part of the primitive cache key and implementation selection.
class ScopedThreadPoolActivation final {
public:
  explicit ScopedThreadPoolActivation(ThreadPoolAdapter& adapter) {
    if (ActivationDepth()++ == 0) {
      KDNN::Threading::ActivateThreadpool(&adapter);
    }
  }

  ~ScopedThreadPoolActivation() {
    assert(ActivationDepth() > 0);
    if (--ActivationDepth() == 0) {
      KDNN::Threading::DeactivateThreadpool();
    }
  }

private:
  static int& ActivationDepth() {
    static thread_local int depth = 0;
    return depth;
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ScopedThreadPoolActivation);
};

}  // namespace kdnn
}  // namespace onnxruntime

#endif  // defined(USE_KDNN)
