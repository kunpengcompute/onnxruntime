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

#ifndef KDNN_INNER_PRODUCT_HPP
#define KDNN_INNER_PRODUCT_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

namespace Detail {

class InnerProductLayerFWDImpl;
class InnerProductLayerBWDDataImpl;
class InnerProductLayerBWDWeightsImpl;

} // Detail

class KDNN_API InnerProductLayerFWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias) noexcept;
    static Status ValidateInput(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias, const Attributes &attributes) noexcept;
    InnerProductLayerFWD(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias) noexcept(false);
    InnerProductLayerFWD(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias, const Attributes &attributes) noexcept(false);
    InnerProductLayerFWD(const InnerProductLayerFWD &other) noexcept(false);
    InnerProductLayerFWD(InnerProductLayerFWD &&other) noexcept;
    InnerProductLayerFWD& operator=(const InnerProductLayerFWD &other) noexcept(false);
    InnerProductLayerFWD& operator=(InnerProductLayerFWD &&other) noexcept;
    void Run(const void *src, const void *wei, void *dst,  const void *bia = nullptr,
        float alpha = 1.0f, float beta = 0.0f) const noexcept(false);
    void Run(const void *src, const void *wei, void *dst, const void *bia, const PostOpsDataPtrs &postOpsPtrs,
        float alpha = 1.0f, float beta = 0.0f) const noexcept(false);
    ~InnerProductLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::InnerProductLayerFWDImpl> pImpl;
};

class KDNN_API InnerProductLayerBWDData final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc) noexcept;
    InnerProductLayerBWDData(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc) noexcept(false);
    InnerProductLayerBWDData(const InnerProductLayerBWDData &other) noexcept(false);
    InnerProductLayerBWDData(InnerProductLayerBWDData &&other) noexcept;
    InnerProductLayerBWDData& operator=(const InnerProductLayerBWDData &other) noexcept(false);
    InnerProductLayerBWDData& operator=(InnerProductLayerBWDData &&other) noexcept;
    void Run(const void *diffDst, const void *wei, void *diffSrc) const noexcept(false);
    ~InnerProductLayerBWDData() noexcept;
private:
    std::unique_ptr<Detail::InnerProductLayerBWDDataImpl> pImpl;
};

class KDNN_API InnerProductLayerBWDWeights final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &diffDst, const TensorInfo &src,
        const TensorInfo &diffWeights, const TensorInfo &diffBias) noexcept;
    InnerProductLayerBWDWeights(const TensorInfo &diffDst, const TensorInfo &src,
        const TensorInfo &diffWeights, const TensorInfo &diffBias) noexcept(false);
    InnerProductLayerBWDWeights(const InnerProductLayerBWDWeights &other) noexcept(false);
    InnerProductLayerBWDWeights(InnerProductLayerBWDWeights &&other) noexcept;
    InnerProductLayerBWDWeights& operator=(const InnerProductLayerBWDWeights &other) noexcept(false);
    InnerProductLayerBWDWeights& operator=(InnerProductLayerBWDWeights &&other) noexcept;
    void Run(const void *diffDst, const void *src, void *diffWeights, void *diffBias = nullptr) const noexcept(false);
    ~InnerProductLayerBWDWeights() noexcept;
private:
    std::unique_ptr<Detail::InnerProductLayerBWDWeightsImpl> pImpl;
};

} // KDNN

#endif // KDNN_INNER_PRODUCT_HPP
