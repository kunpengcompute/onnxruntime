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

#ifndef KDNN_SPBLAS_H
#define KDNN_SPBLAS_H

#include <memory>

#include "service/kdnn_api.hpp"
#include "types/kdnn_tensor_info.hpp"

namespace KDNN {
namespace Detail {

class SparseGemmImpl;

} // Detail

class KDNN_API SparseGemm final {
public:
    using SizeType = ::KDNN::SizeType;
    SparseGemm(const CsrSparseTensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo) noexcept(false);
    void Run(const void *a, const void *b, void *c) const noexcept(false);
    void Run(const void *a, const void *b, void *c, float alpha, float beta) const noexcept(false);
    ~SparseGemm() noexcept;
private:
    std::unique_ptr<Detail::SparseGemmImpl> pImpl;
};

}
#endif // KDNN_SPBLAS_H
