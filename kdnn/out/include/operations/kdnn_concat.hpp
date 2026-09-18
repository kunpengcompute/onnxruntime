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

#ifndef KDNN_CONCAT_HPP
#define KDNN_CONCAT_HPP

#include <memory>
#include <vector>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

namespace Detail {

class ConcatLayerImpl;

} // Detail

class KDNN_API ConcatLayer final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const std::vector<TensorInfo> &srcInfo,
        const int concatDim, const TensorInfo &dstInfo) noexcept;
    ConcatLayer(const std::vector<TensorInfo> &srcInfo, const int concatDim,
        const TensorInfo &dstInfo) noexcept(false);
    ConcatLayer(const ConcatLayer &other) noexcept(false);
    ConcatLayer(ConcatLayer &&other) noexcept;
    ConcatLayer& operator=(const ConcatLayer &other) noexcept(false);
    ConcatLayer& operator=(ConcatLayer &&other) noexcept;
    void Run(const void **src, void *dst) const noexcept(false);
    ~ConcatLayer() noexcept;
private:
    std::unique_ptr<Detail::ConcatLayerImpl> pImpl;
};

} // KDNN

#endif // KDNN_CONCAT_HPP
