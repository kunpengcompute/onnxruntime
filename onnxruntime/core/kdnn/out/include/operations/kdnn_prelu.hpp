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

#ifndef KDNN_PRELU_HPP
#define KDNN_PRELU_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

namespace Detail {

class PReLULayerFWDImpl;
class PReLULayerBWDImpl;

} // Detail

class KDNN_API PReLULayerFWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &weightsInfo,
        const TensorInfo &dstInfo) noexcept;
    PReLULayerFWD(const TensorInfo &srcInfo, const TensorInfo &weightsInfo,
        const TensorInfo &dstInfo) noexcept(false);
    PReLULayerFWD(const PReLULayerFWD &other) noexcept(false);
    PReLULayerFWD(PReLULayerFWD &&other) noexcept;
    PReLULayerFWD& operator=(const PReLULayerFWD &other) noexcept(false);
    PReLULayerFWD& operator=(PReLULayerFWD &&other) noexcept;
    void Run(const void *src, const void *wei, void *dst) const noexcept(false);
    ~PReLULayerFWD() noexcept;
private:
    std::unique_ptr<Detail::PReLULayerFWDImpl> pImpl;
};

class KDNN_API PReLULayerBWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &diffSrcInfo,
        const TensorInfo &weightsInfo, const TensorInfo &diffWeightsInfo,
        const TensorInfo &diffDstInfo) noexcept;
    PReLULayerBWD(const TensorInfo &srcInfo, const TensorInfo &diffSrcInfo,
        const TensorInfo &weightsInfo, const TensorInfo &diffWeightsInfo,
        const TensorInfo &diffDstInfo) noexcept(false);
    PReLULayerBWD(const PReLULayerBWD &other) noexcept(false);
    PReLULayerBWD(PReLULayerBWD &&other) noexcept;
    PReLULayerBWD& operator=(const PReLULayerBWD &other) noexcept(false);
    PReLULayerBWD& operator=(PReLULayerBWD &&other) noexcept;
    void Run(const void *src, void *diffSrc, const void *wei,
        void *diffWeights, const void *diffDst) const noexcept(false);
    ~PReLULayerBWD() noexcept;
private:
    std::unique_ptr<Detail::PReLULayerBWDImpl> pImpl;
};

} // KDNN

#endif // KDNN_PRELU_HPP
