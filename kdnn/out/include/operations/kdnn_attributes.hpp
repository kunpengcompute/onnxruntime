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

#ifndef KDNN_ATTRIBUTES_HPP
#define KDNN_ATTRIBUTES_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "operations/kdnn_post_ops.hpp"
#include "service/kdnn_err_codes.hpp"
namespace KDNN {

namespace Detail {

class AttributesImpl;
struct PrimitiveCacheKeyBuilder;

} // Detail

class KDNN_API Attributes final {
public:
    using SizeType = ::KDNN::SizeType;
    Attributes() noexcept(false);
    Attributes(const Attributes &other) noexcept(false);
    Attributes(Attributes &&other) noexcept;
    Attributes& operator=(const Attributes &other) noexcept(false);
    Attributes& operator=(Attributes &&other) noexcept;
    ~Attributes() noexcept;
    PostOps GetPostOps() const noexcept(false);
    void SetPostOps(const PostOps &po) noexcept(false);

private:
    std::unique_ptr<Detail::AttributesImpl> pImpl;
    friend bool operator==(const Attributes &lhs, const Attributes &rhs) noexcept;
    friend struct Detail::PrimitiveCacheKeyBuilder;
};

KDNN_API bool operator==(const Attributes &lhs, const Attributes &rhs) noexcept;

} // KDNN

#endif // KDNN_ATTRIBUTES_HPP
