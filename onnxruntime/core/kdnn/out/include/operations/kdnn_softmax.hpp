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

#ifndef KDNN_SOFTMAX_HPP
#define KDNN_SOFTMAX_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

enum class AlgorithmKind : std::uint32_t {
    SOFTMAX         = 0x0U,
    LOGSOFTMAX      = 0x1U,
};

namespace Detail {

class SoftmaxLayerFWDImpl;
class SoftmaxLayerBWDImpl;

} // Detail

class KDNN_API SoftmaxLayerFWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo, SizeType axis,
        AlgorithmKind algorithm) noexcept;
    SoftmaxLayerFWD(const TensorInfo &srcInfo, const TensorInfo &dstInfo, SizeType axis,
        AlgorithmKind algorithm) noexcept(false);
    SoftmaxLayerFWD(const SoftmaxLayerFWD &other) noexcept(false);
    SoftmaxLayerFWD(SoftmaxLayerFWD &&other) noexcept;
    SoftmaxLayerFWD& operator=(const SoftmaxLayerFWD &other) noexcept(false);
    SoftmaxLayerFWD& operator=(SoftmaxLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst) const noexcept(false);
    ~SoftmaxLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::SoftmaxLayerFWDImpl> pImpl;
};

class KDNN_API SoftmaxLayerBWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &dstInfo, const TensorInfo &dstDiffInfo,
                                const TensorInfo &srcDiffInfo, SizeType axis, AlgorithmKind algorithm) noexcept;
    SoftmaxLayerBWD(const TensorInfo &dstInfo, const TensorInfo &dstDiffInfo, const TensorInfo &srcDiffInfo,
	                SizeType axis, AlgorithmKind algorithm) noexcept(false);
    SoftmaxLayerBWD(const SoftmaxLayerBWD &other) noexcept(false);
    SoftmaxLayerBWD(SoftmaxLayerBWD &&other) noexcept;
    SoftmaxLayerBWD& operator=(const SoftmaxLayerBWD &other) noexcept(false);
    SoftmaxLayerBWD& operator=(SoftmaxLayerBWD &&other) noexcept;
    void Run(const void *dst, const void *dstDiff, void *srcDiff) const noexcept(false);
    ~SoftmaxLayerBWD() noexcept;
private:
    std::unique_ptr<Detail::SoftmaxLayerBWDImpl> pImpl;
};

} // KDNN

#endif // KDNN_SOFTMAX_HPP
