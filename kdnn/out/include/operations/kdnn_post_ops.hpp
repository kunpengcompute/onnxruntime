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

#ifndef KDNN_POST_OPS_HPP
#define KDNN_POST_OPS_HPP

#include <memory>
#include <vector>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"
#include "types/kdnn_eltwise_kinds.hpp"
#include "types/kdnn_binary_kinds.hpp"
#include "types/kdnn_primitive_kind.hpp"

namespace KDNN {

namespace Detail {

class PostOpsImpl;
struct PrimitiveCacheKeyBuilder;

} // Detail

using PostOpsDataPtrs = std::vector<const void *>;

enum class PReLUMask {
    COMMON     = 0,
    PER_DIM_0  = (1 << 0),
    PER_DIM_1  = (1 << 1),
    PER_DIM_01 = (1 << 0) + (1 << 1),
    PER_DIM_2  = (1 << 2),
    PER_DIM_3  = (1 << 3),
    PER_DIM_4  = (1 << 4),
    PER_TENSOR = (1 << MAX_DIMS) - 1,
    PER_OC     = PER_DIM_1,
    PER_OCIC   = PER_DIM_01
};

class KDNN_API PostOps final {
public:
    using SizeType = ::KDNN::SizeType;
    PostOps() noexcept(false);
    PostOps(const PostOps &other) noexcept(false);
    PostOps(PostOps &&other) noexcept;
    PostOps& operator=(const PostOps &other) noexcept(false);
    PostOps& operator=(PostOps &&other) noexcept;
    ~PostOps() noexcept;
    SizeType Len() const noexcept;
    PrimitiveKind Kind(SizeType index) const noexcept;
    Status AppendSum(float scale = 1.0f) noexcept(false);
    void GetParamsSum(SizeType index, float &scale) const noexcept(false);
    Status AppendEltwise(ActivationFunction kind, float alpha = 0.0f, float beta = 0.0f) noexcept(false);
    void GetParamsEltwise(SizeType index, ActivationFunction &kind, float &alpha, float &beta) const noexcept(false);
    Status AppendDW(Element::Type weiType, Element::Type biaType, Element::Type dstType,
            SizeType kernelSize, SizeType strides, SizeType paddingL) noexcept(false);
    void GetParamsDW(SizeType index, Element::Type &weiType, Element::Type &biaType, Element::Type &dstType,
            SizeType &kernelSize, SizeType &strides, SizeType &paddingL) const noexcept(false);
    Status AppendBinary(BinaryFunction op, const TensorInfo &src1) noexcept(false);
    void GetParamsBinary(SizeType index, BinaryFunction &op, TensorInfo &src1) const noexcept(false);
    Status AppendPReLU(PReLUMask mask) noexcept(false);
    void GetParamsPReLU(SizeType index, PReLUMask &mask) const noexcept(false);

private:
    std::unique_ptr<Detail::PostOpsImpl> pImpl;
    friend bool operator==(const PostOps &lhs, const PostOps &rhs) noexcept;
    friend struct Detail::PrimitiveCacheKeyBuilder;
};
KDNN_API bool operator==(const PostOps &lhs, const PostOps &rhs) noexcept;

} // KDNN

#endif // KDNN_POST_OPS_HPP
