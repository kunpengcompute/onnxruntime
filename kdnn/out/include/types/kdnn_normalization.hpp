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

#ifndef KDNN_NORMALIZATION_HPP
#define KDNN_NORMALIZATION_HPP

#include <cstdint>

#include "service/kdnn_api.hpp"

namespace KDNN {

enum class NormalizationFlags : std::uint32_t {
    NONE               = 0x0U,
    USE_GLOBAL_STATS   = 0x1U,
    USE_SCALE          = 0x2U,
    USE_SHIFT          = 0x4U,
    FUSE_NORM_RELU     = 0x8U,
    RMS_NORM           = 0x20U
};

KDNN_API NormalizationFlags operator & (const NormalizationFlags &lhs, const NormalizationFlags &rhs);
KDNN_API NormalizationFlags operator | (const NormalizationFlags &lhs, const NormalizationFlags &rhs);
KDNN_API NormalizationFlags &operator &= (NormalizationFlags &lhs, const NormalizationFlags &rhs);
KDNN_API NormalizationFlags &operator |= (NormalizationFlags &lhs, const NormalizationFlags &rhs);

} // namespace KDNN

#endif // KDNN_NORMALIZATION_HPP
