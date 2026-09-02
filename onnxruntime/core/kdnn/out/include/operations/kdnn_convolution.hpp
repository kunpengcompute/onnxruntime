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

#ifndef KDNN_CONVOLUTION_HPP
#define KDNN_CONVOLUTION_HPP

#include "service/kdnn_err_codes.hpp"
#include "types/kdnn_tensor_info.hpp"
#include <memory>

namespace KDNN {

enum class ConvolutionAlgorithm { UNIMPLEMENTED, AUTO, DIRECT, WINOGRAD };

namespace Detail {
class ConvolutionFWDImplBase;
class ConvolutionLayerBWDDataImpl;
class ConvolutionLayerBWDWeightsImpl;
} // namespace Detail

class KDNN_API ConvolutionLayerFWD final {
public:
    static Status ValidateInput(const TensorInfo &src, const TensorInfo &weights, const TensorInfo &dst,
                                const TensorInfo &bias, const Shape &strides, const Shape &dilates,
                                const Shape &paddingL, int kCpuIsa, bool isLayoutBlock, bool &isJitConv,
                                bool bUseWino) noexcept;
    static Status ValidateInput(const TensorInfo &src, const TensorInfo &weights, const TensorInfo &dst,
                                const TensorInfo &bias, const Shape &strides, const Shape &dilates,
                                const Shape &paddingL, int kCpuIsa, bool isLayoutBlock, bool &isJitConv, bool bUseWino,
                                int &brgBlock, int nthreads) noexcept;
    explicit ConvolutionLayerFWD(int isa, const TensorInfo &src, const TensorInfo &weights, const TensorInfo &dst,
                                 const TensorInfo &bias, KDNN::Shape strides, KDNN::Shape paddingL, KDNN::Shape dilates,
                                 KDNN::Propagation prop, int nthreads, bool isLayoutBlock,
                                 bool bUseWino) noexcept(false);
    ConvolutionLayerFWD(const TensorInfo &src, const TensorInfo &weights, const TensorInfo &dst, const TensorInfo &bias,
                        const Shape &strides, const Shape &paddingL, const Shape &paddingR,
                        ConvolutionAlgorithm alg) noexcept(false);
    ConvolutionLayerFWD(const TensorInfo &src, const TensorInfo &weights, const TensorInfo &dst, const TensorInfo &bias,
                        const Shape &strides, const Shape &dilates, const Shape &paddingL, const Shape &paddingR,
                        ConvolutionAlgorithm alg) noexcept(false);
    void Run(const void *src, const void *wei, void *dst, const void *bias) noexcept(false);
    ConvolutionLayerFWD(const ConvolutionLayerFWD &other) noexcept(false);
    ConvolutionLayerFWD(ConvolutionLayerFWD &&other) noexcept;
    ConvolutionLayerFWD &operator=(const ConvolutionLayerFWD &other) noexcept(false);
    ConvolutionLayerFWD &operator=(ConvolutionLayerFWD &&other) noexcept;
    ~ConvolutionLayerFWD() noexcept;

private:
    std::unique_ptr<Detail::ConvolutionFWDImplBase> pImpl;
};

class KDNN_API ConvolutionLayerBWDData final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &diffDst, const TensorInfo &weights, const TensorInfo &diffSrc,
                                const Shape &strides, const Shape &dilates, const Shape &paddingL,
                                const Shape &paddingR, ConvolutionAlgorithm alg) noexcept;
    ConvolutionLayerBWDData(const TensorInfo &diffDst, const TensorInfo &weights, const TensorInfo &diffSrc,
                            const Shape &strides, const Shape &paddingL, const Shape &paddingR,
                            ConvolutionAlgorithm alg) noexcept(false);
    ConvolutionLayerBWDData(const TensorInfo &diffDst, const TensorInfo &weights, const TensorInfo &diffSrc,
                            const Shape &strides, const Shape &dilates, const Shape &paddingL, const Shape &paddingR,
                            ConvolutionAlgorithm alg) noexcept(false);
    ConvolutionLayerBWDData(const ConvolutionLayerBWDData &other) noexcept(false);
    ConvolutionLayerBWDData(ConvolutionLayerBWDData &&other) noexcept;
    ConvolutionLayerBWDData &operator=(const ConvolutionLayerBWDData &other) noexcept(false);
    ConvolutionLayerBWDData &operator=(ConvolutionLayerBWDData &&other) noexcept;
    void Run(const void *diffDst, const void *wei, void *diffSrc) const noexcept(false);
    ~ConvolutionLayerBWDData() noexcept;

private:
    std::unique_ptr<Detail::ConvolutionLayerBWDDataImpl> pImpl;
};

class KDNN_API ConvolutionLayerBWDWeights final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &diffDst, const TensorInfo &src, const TensorInfo &diffWeights,
                                const TensorInfo &diffBias, const Shape &strides, const Shape &dilates,
                                const Shape &paddingL, const Shape &paddingR, ConvolutionAlgorithm alg) noexcept;
    ConvolutionLayerBWDWeights(const TensorInfo &diffDst, const TensorInfo &src, const TensorInfo &diffWeights,
                               const TensorInfo &diffBias, const Shape &strides, const Shape &paddingL,
                               const Shape &paddingR, ConvolutionAlgorithm alg) noexcept(false);
    ConvolutionLayerBWDWeights(const TensorInfo &diffDst, const TensorInfo &src, const TensorInfo &diffWeights,
                               const TensorInfo &diffBias, const Shape &strides, const Shape &dilates,
                               const Shape &paddingL, const Shape &paddingR, ConvolutionAlgorithm alg) noexcept(false);
    ConvolutionLayerBWDWeights(const ConvolutionLayerBWDWeights &other) noexcept(false);
    ConvolutionLayerBWDWeights(ConvolutionLayerBWDWeights &&other) noexcept;
    ConvolutionLayerBWDWeights &operator=(const ConvolutionLayerBWDWeights &other) noexcept(false);
    ConvolutionLayerBWDWeights &operator=(ConvolutionLayerBWDWeights &&other) noexcept;
    void Run(const void *diffDst, const void *src, void *diffWei, void *diffBias) const noexcept(false);
    ~ConvolutionLayerBWDWeights() noexcept;

private:
    std::unique_ptr<Detail::ConvolutionLayerBWDWeightsImpl> pImpl;
};
} // namespace KDNN

#endif // KDNN_CONVOLUTION_HPP
