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

#ifndef KDNN_BINARY_HPP
#define KDNN_BINARY_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"
#include "types/kdnn_binary_kinds.hpp"

namespace KDNN {

namespace Detail {

class BinaryLayerImpl;

} // Detail

class KDNN_API BinaryLayer final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &src0Info, const TensorInfo &src1Info,
        const TensorInfo &dstInfo, BinaryFunction op) noexcept;
    BinaryLayer(const TensorInfo &src0Info, const TensorInfo &src1Info,
        const TensorInfo &dstInfo, BinaryFunction op) noexcept(false);
    BinaryLayer(const BinaryLayer &other) noexcept(false);
    BinaryLayer(BinaryLayer &&other) noexcept;
    BinaryLayer& operator=(const BinaryLayer &other) noexcept(false);
    BinaryLayer& operator=(BinaryLayer &&other) noexcept;
    void Run(const void *src0, const void *src1, void *dst,
        const float scale0 = 1.0f, const float scale1 = 1.0f) const noexcept(false);
    ~BinaryLayer() noexcept;
private:
    std::unique_ptr<Detail::BinaryLayerImpl> pImpl;
};

} // KDNN

#endif // KDNN_BINARY_HPP
