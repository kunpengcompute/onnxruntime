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

#ifndef KDNN_ELTWISE_HPP
#define KDNN_ELTWISE_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"
#include "types/kdnn_eltwise_kinds.hpp"

namespace KDNN {
#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

namespace Detail {

class ActivationLayerImplFWD;
class ActivationLayerImplBWD;

} // Detail

class KDNN_API ActivationLayerFWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
        ActivationFunction kind, float alpha = 0.0f, float beta = 0.0f) noexcept;
    ActivationLayerFWD(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
        ActivationFunction kind, float alpha = 0.0f, float beta = 0.0f) noexcept(false);
    ActivationLayerFWD(const ActivationLayerFWD &other) noexcept(false);
    ActivationLayerFWD(ActivationLayerFWD &&other) noexcept;
    ActivationLayerFWD& operator=(const ActivationLayerFWD &other) noexcept(false);
    ActivationLayerFWD& operator=(ActivationLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst) const noexcept(false);
    ~ActivationLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::ActivationLayerImplFWD> pImpl;
};

class KDNN_API ActivationLayerBWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &dsInfo, const TensorInfo &ddInfo,
        const TensorInfo &srcInfo, ActivationFunction kind, float alpha = 0.0f,
        float beta = 0.0f) noexcept;
    ActivationLayerBWD(const TensorInfo &dsInfo, const TensorInfo &ddInfo,
        const TensorInfo &srcInfo, ActivationFunction kind, float alpha = 0.0f,
        float beta = 0.0f) noexcept(false);
    ActivationLayerBWD(const ActivationLayerBWD &other) noexcept(false);
    ActivationLayerBWD(ActivationLayerBWD &&other) noexcept;
    ActivationLayerBWD& operator=(const ActivationLayerBWD &other) noexcept(false);
    ActivationLayerBWD& operator=(ActivationLayerBWD &&other) noexcept;
    void Run(void *ds, const void *dd, const void *src) const noexcept(false);
    ~ActivationLayerBWD() noexcept;
private:
    std::unique_ptr<Detail::ActivationLayerImplBWD> pImpl;
};

} // KDNN

#endif // KDNN_ELTWISE_HPP
