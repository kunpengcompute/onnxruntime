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

#ifndef KDNN_RESAMPLING_HPP
#define KDNN_RESAMPLING_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

enum class ResamplingAlg {
    UNIMPLEMENTED,
    NEAREST_NEIGHBOR,
    LINEAR,
};

namespace Detail {
namespace Resampling {
    class ResamplingLayerImplFWD;
    class ResamplingLayerImplBWD;
} // Resampling
} // Detail

class KDNN_API ResamplingLayerFWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
        ResamplingAlg algKind) noexcept;
    ResamplingLayerFWD(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
        ResamplingAlg algKind) noexcept(false);
    ResamplingLayerFWD(const ResamplingLayerFWD &other) noexcept(false);
    ResamplingLayerFWD(ResamplingLayerFWD &&other) noexcept;
    ResamplingLayerFWD& operator=(const ResamplingLayerFWD &other) noexcept(false);
    ResamplingLayerFWD& operator=(ResamplingLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst) const noexcept(false);
    ~ResamplingLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::Resampling::ResamplingLayerImplFWD> pImpl;
};

class KDNN_API ResamplingLayerBWD final {
public:
    using SizeType = ::KDNN::SizeType;
    ResamplingLayerBWD(const TensorInfo &diffDstInfo, const TensorInfo &diffSrcInfo,
        ResamplingAlg algKind) noexcept(false);

    ResamplingLayerBWD(const ResamplingLayerBWD &other) noexcept(false);
    ResamplingLayerBWD(ResamplingLayerBWD &&other) noexcept;
    ResamplingLayerBWD& operator=(const ResamplingLayerBWD &other) noexcept(false);
    ResamplingLayerBWD& operator=(ResamplingLayerBWD &&other) noexcept;

    ~ResamplingLayerBWD() noexcept;

    static Status ValidateInput(const TensorInfo &diffDstInfo, const TensorInfo &diffSrcInfo,
        ResamplingAlg algKind) noexcept;

    void Run(const void *diffDst, void *diffSrc) const noexcept(false);

private:
    std::unique_ptr<Detail::Resampling::ResamplingLayerImplBWD> pImpl;
};

} // KDNN

#endif
