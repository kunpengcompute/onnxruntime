/*
   Copyright 2025 Huawei Technologies Co., Ltd.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

#ifndef KDNN_THREADING_HPP
#define KDNN_THREADING_HPP

#include <cstdint>
#include <functional>

#include "service/kdnn_api.hpp"

namespace KDNN {
namespace Threading {

enum class EnvMode { KDNN_THREAD_USE_ENV, KDNN_THREAD_IGNORE_ENV };

enum class ThreadingControl { KDNN_DEFAULT, KDNN_MANUAL };

KDNN_API bool ShouldCalculateOptThreads() noexcept;
KDNN_API int SetMaxNumThreads(int nThreads) noexcept;
KDNN_API int SetFixedNumThreads(int nThreads) noexcept;
KDNN_API int GetMaxNumThreads() noexcept;
KDNN_API int SetNumThreadsLocal(int nThreads) noexcept;
KDNN_API EnvMode SetEnvMode(EnvMode mode) noexcept;
KDNN_API ThreadingControl GetThreadingControlStatus() noexcept;

struct KDNN_API ThreadpoolIface {
    virtual int GetNumThreads() const = 0;
    virtual bool IsInParallel() const = 0;
    virtual void ParallelFor(int n, int64_t cost_per_unit, const std::function<void(int, int)> &fn) = 0;
    virtual ~ThreadpoolIface() {};
};

KDNN_API void ActivateThreadpool(ThreadpoolIface *tp);
KDNN_API void DeactivateThreadpool();

} // namespace Threading
} // namespace KDNN

#endif // KDNN_THREADING_HPP
