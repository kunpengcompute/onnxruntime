/*
   Copyright 2026 Huawei Technologies Co., Ltd.

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

#ifndef KDNN_PRIMITIVE_CACHE_HPP
#define KDNN_PRIMITIVE_CACHE_HPP

#include <cstdint>

#include "service/kdnn_api.hpp"
#include "types/kdnn_types.hpp"

namespace KDNN {
namespace PrimitiveCache {

struct KDNN_API Statistics final {
    std::uint64_t l0Hits = 0;
    std::uint64_t l1Hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t waits = 0;
    std::uint64_t creations = 0;
    std::uint64_t creationFailures = 0;
    std::uint64_t evictions = 0;
    std::uint64_t bypasses = 0;
    SizeType size = 0;
    SizeType capacity = 0;
};

// The process-wide primitive cache is enabled by default. Its initial capacity
// is read once from KDNN_PRIMITIVE_CACHE_CAPACITY; zero disables the cache.
KDNN_API void SetCapacity(SizeType capacity) noexcept;
KDNN_API SizeType GetCapacity() noexcept;
KDNN_API SizeType GetSize() noexcept;
KDNN_API void Clear() noexcept;

// Statistics are disabled by default so cache hits do not update shared
// counters on latency-sensitive paths.
KDNN_API void SetStatisticsEnabled(bool enabled) noexcept;
KDNN_API bool IsStatisticsEnabled() noexcept;
KDNN_API Statistics GetStatistics() noexcept;
KDNN_API void ResetStatistics() noexcept;

} // namespace PrimitiveCache
} // namespace KDNN

#endif // KDNN_PRIMITIVE_CACHE_HPP
