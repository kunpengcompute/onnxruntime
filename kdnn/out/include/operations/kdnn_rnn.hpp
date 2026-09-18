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

#ifndef KDNN_RNN_HPP
#define KDNN_RNN_HPP

#include <memory>
#include <vector>
#include "service/kdnn_err_codes.hpp"
#include "types/kdnn_tensor_info.hpp"

namespace KDNN {

enum class RnnAlgorithm {
    UNIMPLEMENTED,
    RNN,
    LSTM,
    GRU,
    LBR_GRU,
    AUGRU,
    LBR_AUGRU
};

enum class ActivateFunctionRNN {
    UNIMPLEMENTED,
    RELU,
    TANH,
    LOGISTIC
};

namespace Detail {

enum class ExecutionDirectionT {
    L2R,
    R2L,
    BI_CONCAT,
    BI_SUM,
};

enum class CellPositionT {
    MIDDLE_CELL = 0x0,
    FIRST_LAYER = 0x1,
    FIRST_ITER = 0x2,
    LAST_LAYER = 0x4,
    LAST_ITER = 0x8,
    C_STATE_FIRST_ITER = 0x10,
    C_STATE_LAST_ITER = 0x20,
    MERGED_ITER = 0x40,
    MERGED_LAYER = 0x80
};

enum class WeightsTypeT {
    LAYER,
    ITER,
    PROJECTION,
    PEEPHOLE,
};

inline CellPositionT &operator|=(CellPositionT &lhs, CellPositionT rhs)
{
    lhs = static_cast<CellPositionT>(
            static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
    return lhs;
}

inline CellPositionT operator|(CellPositionT lhs, CellPositionT rhs)
{
    return static_cast<CellPositionT>(
            static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}

inline bool operator&(CellPositionT lhs, CellPositionT rhs)
{
    return static_cast<bool>(
            static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs));
}

enum class BrgemmRnnExecuteLoopOrderT {
    // default for kernels w/o loop order choice
    UNDEFINED = 0x0,
    // mBlocking loop is outermost
    M_BLK_N_BLK = 0x1,
    // nBlocking loop is outermost
    N_BLK_M_BLK = 0x2
};

struct DiffSrcBrgemmConfT {
    SizeType m = 0, n = 0, k = 0;

    SizeType nBlock = 0, nBlocks = 0, nTail = 0;
    SizeType mBlock = 0, mBlocks = 0;

    SizeType kBlocks = 0, kBlock = 0, kTail = 0;
    SizeType kpadded = 0;

    SizeType nIter = 0, nLayer = 0;
    SizeType nLayerBlocks = 0, nLayerTail = 0;
    SizeType nIterBlocks = 0, nIterTail = 0;
    SizeType lda = 0, ldb = 0, ldc = 0;

    BrgemmRnnExecuteLoopOrderT loopOrder
            = BrgemmRnnExecuteLoopOrderT::UNDEFINED;
    int gatesBlock;
};

struct DiffWeiBrgemmConfT {
    SizeType m = 0, mLayer = 0, mIter = 0, n = 0, k = 0;

    SizeType nBlock = 0, nBlocks = 0, nTail = 0;
    SizeType mBlock = 0, mBlocks = 0;
    SizeType kBlocks = 0, kBlock = 0, kTail = 0;
    SizeType kpadded = 0;
    SizeType ldaLayer = 0, ldaIter = 0, ldb = 0, ldcIter = 0, ldcLayer = 0;

    bool globalTranspose = false;

    BrgemmRnnExecuteLoopOrderT loopOrder
            = BrgemmRnnExecuteLoopOrderT::UNDEFINED;
};

constexpr int KDNN_RNN_MAX_N_PARTS = 4;

namespace Rnn {
class RnnFWDImpl;
class RnnBWDImpl;
} // Rnn

} // Detail

struct KDNN_API RnnCfg {
    using SizeType = KDNN::SizeType;

    KDNN::Detail::ExecutionDirectionT execDir;

    IntType nLayer = 0, nIter = 0, nDir = 0, nGates = 0;
    IntType mb = 0;
    IntType slc = 0, sic = 0, dhc = 0, dic = 0, dlc = 0;

    IntType nBias = 0;

    SizeType weightsLayerLd = 0;
    SizeType diffWeightsLayerLd = 0;
    SizeType weightsIterLd = 0;
    SizeType diffWeightsIterLd = 0;
    SizeType weightsProjectionLd = 0;
    SizeType diffWeightsProjectionLd = 0, diffWeightsProjectionNld = 0;

    SizeType wsGatesLd = 0, wsGatesNld = 0;
    SizeType wsHtLd = 0, wsHtNld = 0;
    SizeType projHtLd = 0;
    SizeType wsStatesLayerLd = 0, wsStatesLayerNld = 0;
    SizeType wsStatesIterLd = 0;
    SizeType wsDiffStatesLayerLd = 0;
    SizeType wsDiffStatesIterLd = 0;
    SizeType wsDiffStatesIterCLd = 0;

    SizeType scratchGatesLd = 0, scratchGatesNld = 0;
    SizeType scratchDiffHtLd = 0;

    SizeType srcLayerLd = 0, srcLayerCellLd = 0;
    SizeType srcIterLd = 0;
    SizeType srcIterCLd = 0;
    SizeType dstLayerLd = 0, dstLayerCellLd = 0;
    SizeType dstIterLd = 0;
    SizeType dstIterCLd = 0;

    bool isTraining = false, isLbr = false;

    SizeType wsGatesSize = 0;
    SizeType wsHtSize = 0;
    SizeType wsStatesLayerSize = 0;
    SizeType wsStatesIterCSize = 0;
    SizeType wsDiffStatesLayerSize = 0;
    SizeType wsDiffStatesIterSize = 0;
    SizeType wsDiffStatesIterCSize = 0;
    SizeType scratchGatesSize = 0;
    SizeType scratchCellSize = 0;
    SizeType wsGridCompSize = 0;
    SizeType wsPerCell = 0;

    bool srcLayerIsTrivialStride = false;
    bool dstLayerIsTrivialStride = false;
    bool diffWeightsOverwrite = false;
    bool gatesFuncLinearMode = false;
    bool isScale = false;

    SizeType wsGatesOffset = 0;
    SizeType wsHtOffset = 0;
    SizeType wsStatesLayerOffset = 0;
    SizeType wsStatesIterOffset = 0;
    SizeType wsStatesIterCOffset = 0;
    SizeType wsBiasOffset = 0;
    SizeType wsDiffStatesLayerOffset = 0;
    SizeType wsDiffStatesIterOffset = 0;
    SizeType wsDiffStatesIterCOffset = 0;
    SizeType wsGridCompOffset = 0;
    SizeType scratchGatesOffset = 0;
    SizeType scratchHtOffset = 0;
    SizeType scratchDiffHtOffset = 0;
    SizeType scratchCellOffset = 0;

    float scales[4] = {0.0};
    float alpha = 0.0;
    float cscale = 0.0;
    inline bool SkipDstLayerCopy() const
    {
        return (execDir == KDNN::Detail::ExecutionDirectionT::L2R || execDir == KDNN::Detail::ExecutionDirectionT::R2L);
    }
    inline bool SkipDstIterCopy() const
    {
        return (execDir == KDNN::Detail::ExecutionDirectionT::L2R ||
            execDir == KDNN::Detail::ExecutionDirectionT::R2L) && (dstIterLd > 0);
    }

    inline SizeType SrcLayerLd(KDNN::Detail::CellPositionT cellPosition) const
    {
        return (cellPosition & KDNN::Detail::CellPositionT::FIRST_LAYER)
                ? srcLayerLd
                : (cellPosition & KDNN::Detail::CellPositionT::LAST_ITER)
                ? dstIterLd
                : wsStatesLayerLd;
    }

    inline SizeType SrcIterLd(KDNN::Detail::CellPositionT cellPosition) const
    {
        return (cellPosition & KDNN::Detail::CellPositionT::FIRST_ITER)
                ? srcIterLd
                : ((cellPosition & KDNN::Detail::CellPositionT::LAST_LAYER) && SkipDstLayerCopy()
                                        && !(cellPosition & KDNN::Detail::CellPositionT::FIRST_ITER)
                                ? dstLayerLd
                                : wsStatesIterLd);
    }

    inline SizeType SrcIterCLd(KDNN::Detail::CellPositionT cellPosition) const
    {
        return isTraining ?
            ((cellPosition & KDNN::Detail::CellPositionT::FIRST_ITER) ? srcIterCLd : wsDiffStatesIterCLd) :
            (cellPosition & KDNN::Detail::CellPositionT::FIRST_ITER) ? srcIterCLd : dstIterCLd;
    }
    inline SizeType DstIterCLd(KDNN::Detail::CellPositionT cellPosition) const
    {
        return isTraining ?
            ((cellPosition & KDNN::Detail::CellPositionT::LAST_ITER) ? dstIterCLd : wsDiffStatesIterCLd) :
            dstIterCLd;
    }
    inline SizeType DstLayerLd(KDNN::Detail::CellPositionT cellPosition) const
    {
        return (cellPosition & KDNN::Detail::CellPositionT::LAST_LAYER) && SkipDstLayerCopy()
                ? dstLayerLd
                : (cellPosition & KDNN::Detail::CellPositionT::LAST_ITER) && SkipDstIterCopy()
                ? dstIterLd
                : wsStatesLayerLd;
    }

    inline SizeType DstIterLd(KDNN::Detail::CellPositionT cellPosition) const
    {
        return (cellPosition & KDNN::Detail::CellPositionT::LAST_ITER) && SkipDstLayerCopy()
                ? dstIterLd
                : wsStatesLayerLd;
    }

    inline SizeType DstIterPart2Ld(KDNN::Detail::CellPositionT cellPosition) const
    {
        return (cellPosition & KDNN::Detail::CellPositionT::LAST_LAYER) ? DstLayerLd(cellPosition)
                                            : DstIterLd(cellPosition);
    }

    // get DiffWeightsBeta based on cell position
    inline float DiffWeightsBeta(KDNN::Detail::CellPositionT cellPosition) const
    {
        if (diffWeightsOverwrite && (cellPosition & KDNN::Detail::CellPositionT::LAST_ITER)) {
            // Initialize diff weights if needed
            return 0.0f;
        }
        return 1.0f;
    }
};

class KDNN_API RnnFWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const std::vector<TensorInfo> &srcInfos, const std::vector<TensorInfo> &weightInfos,
    const std::vector<TensorInfo> &dstInfos, TensorInfo &biasInfo, RnnCfg &rnn,
    RnnAlgorithm alg, const ActivateFunctionRNN actf) noexcept;
    RnnFWD(const std::vector<TensorInfo> &srcInfos, const std::vector<TensorInfo> &weightInfos,
        const std::vector<TensorInfo> &dstInfos, TensorInfo &biasInfo, RnnCfg &rnn, RnnAlgorithm alg,
        const ActivateFunctionRNN actf) noexcept(false);
    RnnFWD(const RnnFWD &other) noexcept(false);
    RnnFWD(RnnFWD &&other) noexcept;
    RnnFWD& operator=(const RnnFWD &other) noexcept(false);
    RnnFWD& operator=(RnnFWD &&other) noexcept;
    void Run(const void **src, const void **weight, void **dst, const void *bias, void *wsBase,
        const void *augruAttentions) const noexcept(false);
    ~RnnFWD() noexcept;
private:
    std::unique_ptr<Detail::Rnn::RnnFWDImpl> pImpl;
};

class KDNN_API RnnBWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const std::vector<TensorInfo> &srcInfos, const std::vector<TensorInfo> &weightInfos,
    const std::vector<TensorInfo> &dstInfos, TensorInfo &biasInfo,
    const std::vector<TensorInfo> &diffSrcInfos, const std::vector<TensorInfo> &diffWeightInfos,
    const std::vector<TensorInfo> &diffDstInfos, TensorInfo &diffBiasInfo, RnnAlgorithm alg,
    const ActivateFunctionRNN actf) noexcept;
    RnnBWD(const std::vector<TensorInfo> &srcInfos, const std::vector<TensorInfo> &weightInfos,
           const std::vector<TensorInfo> &dstInfos, TensorInfo &biasInfo,
           const std::vector<TensorInfo> &diffSrcInfos, const std::vector<TensorInfo> &diffWeightInfos,
           const std::vector<TensorInfo> &diffDstInfos, TensorInfo &diffBiasInfo, RnnCfg &rnn, RnnAlgorithm alg,
           const ActivateFunctionRNN actf) noexcept(false);
    RnnBWD(const RnnBWD &other) noexcept(false);
    RnnBWD(RnnBWD &&other) noexcept;
    RnnBWD& operator=(const RnnBWD &other) noexcept(false);
    RnnBWD& operator=(RnnBWD &&other) noexcept;
    void Run(const void **src, const void **weight, const void **dst, const void *bias, void *wsBase,
        const void **diffDstVec, void **diffSrcVec, void **diffWeightVec, void *diffBias,
        const void *augruAttention, void *diffAugruAttention) const noexcept(false);
    ~RnnBWD() noexcept;
private:
    std::unique_ptr<Detail::Rnn::RnnBWDImpl> pImpl;
};

} // KDNN

#endif // KDNN_RNN_HPP
