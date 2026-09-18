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

#ifndef KDNN_REORDER_HPP
#define KDNN_REORDER_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {
namespace Detail {

class ReorderLayerImpl;

} // Detail

class KDNN_API ReorderLayer final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo) noexcept;
    ReorderLayer(const TensorInfo &srcInfo, const TensorInfo &dstInfo) noexcept(false);
    ReorderLayer(const ReorderLayer &other) noexcept(false);
    ReorderLayer(ReorderLayer &&other) noexcept;
    ReorderLayer& operator=(const ReorderLayer &other) noexcept(false);
    ReorderLayer& operator=(ReorderLayer &&other) noexcept;
    void Run(const void *src, void *dst) const noexcept(false);
    void Run(const void *src, void *dst, float alpha) const noexcept(false);
    ~ReorderLayer() noexcept;
private:
    std::unique_ptr<Detail::ReorderLayerImpl> pImpl;
};

} // KDNN

#endif // KDNN_REORDER_HPP
