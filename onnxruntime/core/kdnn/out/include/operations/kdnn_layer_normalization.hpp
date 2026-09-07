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

#ifndef KDNN_LAYER_NORMALIZATION_HPP
#define KDNN_LAYER_NORMALIZATION_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

namespace Detail {

class NormalizationLayerFWDImpl;
class NormalizationLayerBWDImpl;

} // Detail

class KDNN_API NormalizationLayerFWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &statsInfo,
	                            const TensorInfo &scaleshiftInfo, const TensorInfo &dstInfo,
                                NormalizationFlags flags) noexcept;
    NormalizationLayerFWD(const TensorInfo &srcInfo, const TensorInfo &statsInfo, const TensorInfo &scaleshiftInfo,
                          const TensorInfo &dstInfo, NormalizationFlags flags) noexcept(false);
    NormalizationLayerFWD(const NormalizationLayerFWD &other) noexcept(false);
    NormalizationLayerFWD(NormalizationLayerFWD &&other) noexcept;
    NormalizationLayerFWD& operator=(const NormalizationLayerFWD &other) noexcept(false);
    NormalizationLayerFWD& operator=(NormalizationLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst, const void *scale, const void *shift, float *mean, float *variance,
             bool saveStats, const float eps) const noexcept(false);
    ~NormalizationLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::NormalizationLayerFWDImpl> pImpl;
};

class KDNN_API NormalizationLayerBWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &statInfo,
        const TensorInfo &diffSrcInfo, const TensorInfo &diffDstInfo,
        const TensorInfo &scaleShiftInfo, const TensorInfo &diffscaleShiftInfo,
        NormalizationFlags flags) noexcept;
    NormalizationLayerBWD(const TensorInfo &srcInfo, const TensorInfo &statInfo,
        const TensorInfo &diffSrcInfo, const TensorInfo &diffDstInfo,
        const TensorInfo &scaleShiftInfo, const TensorInfo &diffscaleShiftInfo,
        NormalizationFlags flags) noexcept(false);
    NormalizationLayerBWD(const NormalizationLayerBWD &other) noexcept(false);
    NormalizationLayerBWD(NormalizationLayerBWD &&other) noexcept;
    NormalizationLayerBWD& operator=(const NormalizationLayerBWD &other) noexcept(false);
    NormalizationLayerBWD& operator=(NormalizationLayerBWD &&other) noexcept;
    void Run(const void *src, const float *mean, const float *variance, const void *diffDst,
            const void *scale, void *diffSrc, void *diffScale,
            void *diffShift, float eps) const noexcept(false);
    ~NormalizationLayerBWD() noexcept;
private:
    std::unique_ptr<Detail::NormalizationLayerBWDImpl> pImpl;
};

} // KDNN

#endif // KDNN_LAYER_NORMALIZATION_HPP
