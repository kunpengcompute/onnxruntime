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

#ifndef KDNN_GEMM_HPP
#define KDNN_GEMM_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

#include "operations/kdnn_attributes.hpp"
#include "operations/kdnn_post_ops.hpp"

namespace KDNN {

namespace Detail {

class GemmImpl;

} // Detail

class KDNN_API Gemm final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo,
        const TensorInfo &cInfo, const TensorInfo &biasInfo) noexcept;
    static Status ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo,
        const TensorInfo &cInfo) noexcept;
    static Status ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo,
        const TensorInfo &cInfo, const TensorInfo &biasInfo, const Attributes &attributes) noexcept;
    static Status ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo,
            const TensorInfo &cInfo, const Attributes &attributes) noexcept;
    // Returns the weight layout preferred by a prepack-capable GEMM implementation.
    static Layout GetDesiredWeiLayout(const Shape &weiDims, const Element::TypeT &srcDt,
        const Element::TypeT &weiDt, const Element::TypeT &dstDt,
        const Element::TypeT &biaDt = Element::TypeT::UNDEFINED) noexcept(false);
    Gemm(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo,
        const TensorInfo &biasInfo) noexcept(false);
    Gemm(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo) noexcept(false);
    Gemm(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo,
        const TensorInfo &biasInfo, const Attributes &attributes) noexcept(false);
    Gemm(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo,
        const Attributes &attributes) noexcept(false);
    Gemm(const Gemm &other) noexcept(false);
    Gemm(Gemm &&other) noexcept;
    Gemm& operator=(const Gemm &other) noexcept(false);
    Gemm& operator=(Gemm &&other) noexcept;
    void Run(const void *a, const void *b, void *c, const void *bias,
        float alpha = 1.0f, float beta = 0.0f) const noexcept(false);
    void Run(const void *a, const void *b, void *c, const void *bias, const PostOpsDataPtrs &postOpsPtrs,
        float alpha = 1.0f, float beta = 0.0f) const noexcept(false);
    void Run(const void *a, const void *b, void *c,
        float alpha = 1.0f, float beta = 0.0f) const noexcept(false);
    void Run(const void *a, const void *b, void *c, const PostOpsDataPtrs &postOpsPtrs,
        float alpha = 1.0f, float beta = 0.0f) const noexcept(false);
    ~Gemm() noexcept;

private:
    Detail::GemmImpl *pImpl;
};

static_assert(sizeof(Gemm) == sizeof(void *), "Gemm must remain a single-pointer ABI wrapper");

} // KDNN

#endif // KDNN_GEMM_HPP
