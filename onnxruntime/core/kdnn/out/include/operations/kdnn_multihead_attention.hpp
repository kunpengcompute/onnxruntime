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

#ifndef KDNN_MULTIHEAD_ATTENTION_HPP
#define KDNN_MULTIHEAD_ATTENTION_HPP

#include <memory>

#include "service/kdnn_err_codes.hpp"
#include "types/kdnn_tensor_info.hpp"

namespace KDNN {

namespace Detail {

class MultiHeadAttentionImpl;

} // namespace Detail

// Execution algorithm selection for MultiHeadAttention.
//
// CLASSIC         Full materialization path (current default): K/V (and Q/out
//                 when Sq>1) are PackHeads-normalised into head-major scratch,
//                 scores [B,H,Sq,Sk] are fully materialised, softmax is a full
//                 axis reduce.  Behaviour is unchanged from the original
//                 implementation.
// CLASSIC_NO_PACK Direct decode path: K/V are read from the native
//                 [B,S,H*d] layout without PackHeads. Scores are materialised
//                 per (batch, query-row), followed by a full softmax and PV.
enum class AttentionAlgorithm {
    CLASSIC = 0,
    CLASSIC_NO_PACK,
};

// Decode-GEMV traversal selector. The interleaved kernels consume the native
// [B,S,H*d] projection layout jointly across heads. Unsupported shapes fall
// back to generic per-head GEMV; OFF is available for controlled A/B tests.
enum class AttentionGemvMode {
    OFF = 0,
    QK,
    PV,
    ALL,
};

// Execution configuration. The algorithm defaults to classic; the interleaved
// selector is ignored by classic and defaults to the optimized no-pack path.
// The legacy constructor (without options) remains the compatibility entry point.
struct KDNN_API MultiHeadAttentionOptions {
    AttentionAlgorithm algorithm = AttentionAlgorithm::CLASSIC;
    // Outer worker budget for CLASSIC_NO_PACK. 0 means one worker. GEMVs inside
    // each worker always stay single-threaded to avoid nested parallelism.
    int num_threads = 0;
    AttentionGemvMode interleaved_gemv = AttentionGemvMode::ALL;
    // K/V passed as a single interleaved buffer [B, Sk, 2*hidden] (K in the first
    // hidden columns, V in the last hidden). When true the key/value row stride
    // used by the no-pack kernels and PackHeads is 2*hidden instead of hidden.
    // The caller still passes two pointers (key = kv_base, value = kv_base + hidden);
    // this flag only tells the implementation the actual row stride.
    bool kv_merged = false;
};

// Fused scaled-dot-product multi-head (cross) attention.
//
// Computes, for batch b and head h:
//   scores = scale * Q . K^T          [B, numHeads, Sq, Sk]
//   scores = scores + mask            (mask broadcast over heads / query rows)
//   probs  = softmax(scores, axis=Sk) [B, numHeads, Sq, Sk]
//   out    = probs . V                [B, numHeads, Sq, headDim]
// then merges the heads back into the last dimension giving [B, Sq, hidden]
// where hidden = numHeads * headDim.
//
// Tensors are plain row-major float buffers laid out exactly as they come out
// of the per-projection BiasAdd in a transformer block, i.e.:
//   Q     : [B, Sq, hidden]   (hidden == numHeads * headDim)
//   K, V  : [B, Sk, hidden]
//   mask  : additive pre-softmax bias, broadcast over heads and query rows.
//           Its batch dimension may be either B (per-batch bias, layout
//           [B, 1, 1, Sk]) or 1 (a single bias shared by every batch element,
//           layout [1, 1, 1, Sk]); which one is selected at construction via
//           maskBatchBroadcast.  mask == nullptr disables masking.
//   out   : [B, Sq, hidden]
//
// CLASSIC normalizes K/V into head-major scratch before the two batched GEMMs;
// Q and output use the same normalization when Sq is greater than one.
// CLASSIC_NO_PACK directly consumes the native layout. The selected path's
// scratch requirement is returned by GetWorkspaceSize().
class KDNN_API MultiHeadAttention final {
public:
    using SizeType = ::KDNN::SizeType;

    static Status ValidateInput(SizeType batch, SizeType seqQ, SizeType seqK, SizeType numHeads,
                                SizeType headDim) noexcept;

    // maskBatchBroadcast == true  -> mask laid out [1, 1, 1, Sk] (shared across
    //                                the batch); false -> mask laid out
    //                                [B, 1, 1, Sk] (one bias row per batch).
    MultiHeadAttention(SizeType batch, SizeType seqQ, SizeType seqK, SizeType numHeads, SizeType headDim, float scale,
                       bool maskBatchBroadcast = false) noexcept(false);
    // Options overload: selects the execution algorithm (default classic) and
    // no-pack worker budget / interleaved-GEMV mode. Preserves the legacy
    // constructor as the exact-compatibility entry point.
    MultiHeadAttention(SizeType batch, SizeType seqQ, SizeType seqK, SizeType numHeads, SizeType headDim, float scale,
                       bool maskBatchBroadcast, const MultiHeadAttentionOptions &options) noexcept(false);
    MultiHeadAttention(const MultiHeadAttention &other) noexcept(false);
    MultiHeadAttention(MultiHeadAttention &&other) noexcept;
    MultiHeadAttention &operator=(const MultiHeadAttention &other) noexcept(false);
    MultiHeadAttention &operator=(MultiHeadAttention &&other) noexcept;
    ~MultiHeadAttention() noexcept;

    // Bytes of caller-provided scratch the Run() overload taking a workspace
    // needs.  Pass a buffer of at least this size; if you call the Run()
    // overload without a workspace the operator allocates internally.
    SizeType GetWorkspaceSize() const noexcept;

    void Run(const void *q, const void *k, const void *v, const void *mask, void *out, void *workspace) const
        noexcept(false);
    void Run(const void *q, const void *k, const void *v, const void *mask, void *out) const noexcept(false);

private:
    Detail::MultiHeadAttentionImpl *pImpl;
};

static_assert(sizeof(MultiHeadAttention) == sizeof(void *),
              "MultiHeadAttention must remain a single-pointer ABI wrapper");

} // namespace KDNN

#endif // KDNN_MULTIHEAD_ATTENTION_HPP
