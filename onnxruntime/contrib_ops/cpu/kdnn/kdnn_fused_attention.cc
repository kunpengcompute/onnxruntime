// Copyright 2025 Huawei Technologies Co., Ltd.
// Licensed under the Apache License, Version 2.0.

#if defined(USE_KDNN)

#include "contrib_ops/cpu/kdnn/kdnn_fused_attention.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/platform/threadpool.h"

#include "kdnn.hpp"

namespace onnxruntime {
namespace contrib {

using onnxruntime::concurrency::ThreadPool;

namespace {

KDNN::MultiHeadAttentionOptions ReadAttentionOptions() {
  KDNN::MultiHeadAttentionOptions options;

  const char* algorithm = std::getenv("ORT_KDNN_ATTENTION_ALGORITHM");
  if (algorithm != nullptr && algorithm[0] != '\0') {
    const std::string value(algorithm);
    if (value == "classic_no_pack") {
      options.algorithm = KDNN::AttentionAlgorithm::CLASSIC_NO_PACK;
    } else if (value == "classic") {
      options.algorithm = KDNN::AttentionAlgorithm::CLASSIC;
    }
  }

  const char* interleaved_gemv = std::getenv("ORT_KDNN_ATTENTION_INTERLEAVED_GEMV");
  if (interleaved_gemv != nullptr && interleaved_gemv[0] != '\0') {
    const std::string value(interleaved_gemv);
    if (value == "qk") {
      options.interleaved_gemv = KDNN::AttentionGemvMode::QK;
    } else if (value == "pv") {
      options.interleaved_gemv = KDNN::AttentionGemvMode::PV;
    } else if (value == "all") {
      options.interleaved_gemv = KDNN::AttentionGemvMode::ALL;
    } else if (value == "off") {
      options.interleaved_gemv = KDNN::AttentionGemvMode::OFF;
    }
  }

  return options;
}

}  // namespace

// Internal op: registered outside the onnx domain.
ONNX_OPERATOR_KERNEL_EX(
    KdnnFusedAttention,
    kKdnnDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    KdnnFusedAttention);

KdnnFusedAttention::KdnnFusedAttention(const OpKernelInfo& info) : OpKernel(info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0,
              "KdnnFusedAttention requires a positive num_heads attribute");
  num_heads_ = static_cast<int>(num_heads);

  float scale = 0.0f;
  has_scale_ = info.GetAttr<float>("scale", &scale).IsOK();
  scale_ = has_scale_ ? scale : -1.0f;
}

Status KdnnFusedAttention::Compute(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* mask = context->Input<Tensor>(3);  // optional

  ORT_RETURN_IF_NOT(query != nullptr && key != nullptr && value != nullptr,
                    "KdnnFusedAttention requires query, key and value inputs");

  const auto& q_dims = query->Shape().GetDims();
  const auto& k_dims = key->Shape().GetDims();
  const auto& v_dims = value->Shape().GetDims();
  const bool split_heads = q_dims.size() == 4;
  ORT_RETURN_IF_NOT((q_dims.size() == 3 || split_heads) &&
                        k_dims.size() == q_dims.size() && v_dims.size() == q_dims.size(),
                    "KdnnFusedAttention expects matching rank-3 [B,S,hidden] or "
                    "rank-4 [B,S,H,d] query/key/value");

  const int64_t query_batch = q_dims[0];
  const int64_t key_batch = k_dims[0];
  const int64_t value_batch = v_dims[0];
  const int64_t seq_q = q_dims[1];
  const int64_t seq_k = k_dims[1];
  int64_t hidden = 0;
  int64_t head_dim = 0;

  ORT_RETURN_IF_NOT(v_dims[1] == seq_k,
                    "KdnnFusedAttention: K/V sequence length mismatch");

  auto broadcast_batch = [](int64_t lhs, int64_t rhs, int64_t& result) {
    if (lhs == rhs) {
      result = lhs;
      return true;
    }
    if (lhs == 1) {
      result = rhs;
      return true;
    }
    if (rhs == 1) {
      result = lhs;
      return true;
    }
    return false;
  };

  int64_t score_batch = 0;
  ORT_RETURN_IF_NOT(broadcast_batch(query_batch, key_batch, score_batch),
                    "KdnnFusedAttention: Q/K batch dimensions are not broadcastable");

  if (split_heads) {
    ORT_RETURN_IF_NOT(seq_q == 1,
                      "KdnnFusedAttention: rank-4 input currently requires q_sequence_length == 1");
    ORT_RETURN_IF_NOT(q_dims[2] == num_heads_ && k_dims[2] == num_heads_ && v_dims[2] == num_heads_,
                      "KdnnFusedAttention: rank-4 head dimension must match num_heads");
    ORT_RETURN_IF_NOT(k_dims[3] == q_dims[3] && v_dims[3] == q_dims[3],
                      "KdnnFusedAttention: rank-4 head size mismatch across Q/K/V");
    head_dim = q_dims[3];
    hidden = static_cast<int64_t>(num_heads_) * head_dim;
  } else {
    hidden = q_dims[2];
    ORT_RETURN_IF_NOT(k_dims[2] == hidden && v_dims[2] == hidden,
                      "KdnnFusedAttention: hidden mismatch across Q/K/V");
    ORT_RETURN_IF_NOT(hidden % num_heads_ == 0,
                      "KdnnFusedAttention: hidden_size not divisible by num_heads");
    head_dim = hidden / num_heads_;
  }
  const float scale = has_scale_
                          ? scale_
                          : 1.0f / std::sqrt(static_cast<float>(head_dim));

  // Mask handling: optional additive pre-softmax bias of shape [B or 1, 1, 1, Sk].
  // Batch dimensions follow ONNX multidirectional broadcasting. Each output batch
  // row is processed as an independent batch=1 KDNN problem.
  const float* mask_base = nullptr;
  int64_t mask_batch_stride = 0;  // elements to advance per batch (0 == shared)
  int64_t attention_batch = score_batch;
  if (mask != nullptr) {
    const auto& m_dims = mask->Shape().GetDims();
    ORT_RETURN_IF_NOT(m_dims.size() >= 2,
                      "KdnnFusedAttention: mask must have batch and sequence dimensions");
    ORT_RETURN_IF_NOT(m_dims.back() == seq_k,
                      "KdnnFusedAttention: mask last dim must equal K sequence length");
    const int64_t mask_batch = m_dims.front();
    int64_t masked_batch = 0;
    ORT_RETURN_IF_NOT(broadcast_batch(attention_batch, mask_batch, masked_batch),
                      "KdnnFusedAttention: score and mask batch dimensions are not broadcastable");
    attention_batch = masked_batch;
    for (size_t i = 1; i + 1 < m_dims.size(); ++i) {
      ORT_RETURN_IF_NOT(m_dims[i] == 1,
                        "KdnnFusedAttention: intermediate mask dimensions must be 1");
    }
    mask_base = mask->Data<float>();
    mask_batch_stride = (mask_batch == 1) ? 0 : seq_k;
  }

  int64_t output_batch = 0;
  ORT_RETURN_IF_NOT(broadcast_batch(attention_batch, value_batch, output_batch),
                    "KdnnFusedAttention: attention and V batch dimensions are not broadcastable");

  Tensor* output = split_heads
                       ? context->Output(0, TensorShape({output_batch, num_heads_, seq_q, head_dim}))
                       : context->Output(0, TensorShape({output_batch, seq_q, hidden}));
  if (output_batch == 0 || seq_q == 0) {
    return Status::OK();  // nothing to compute
  }

  const auto Sq = static_cast<KDNN::SizeType>(seq_q);
  const auto Sk = static_cast<KDNN::SizeType>(seq_k);
  const auto NH = static_cast<KDNN::SizeType>(num_heads_);
  const auto HD = static_cast<KDNN::SizeType>(head_dim);

  KDNN::Status vstatus = KDNN::MultiHeadAttention::ValidateInput(1, Sq, Sk, NH, HD);
  ORT_RETURN_IF_NOT(vstatus == KDNN::Status::SUCCESS,
                    "KdnnFusedAttention: KDNN ValidateInput rejected the shapes");

  const float* q_data = query->Data<float>();
  const float* k_data = key->Data<float>();
  const float* v_data = value->Data<float>();
  float* out_data = output->MutableData<float>();
  const int64_t output_batch_stride = seq_q * hidden;
  const int64_t query_batch_stride = query_batch == 1 ? 0 : output_batch_stride;
  const int64_t key_batch_stride = key_batch == 1 ? 0 : seq_k * hidden;
  const int64_t value_batch_stride = value_batch == 1 ? 0 : seq_k * hidden;
  const KDNN::MultiHeadAttentionOptions attention_options = ReadAttentionOptions();

  // Parallelize independent batch rows with ORT. Each task owns both its KDNN
  // primitive and workspace.
  ThreadPool* tp = context->GetOperatorThreadPool();

  std::atomic<bool> have_exception{false};
  std::exception_ptr pending_exception;

  ThreadPool::TryBatchParallelFor(
      tp, static_cast<std::ptrdiff_t>(output_batch),
      [&](std::ptrdiff_t b) {
        if (have_exception.load(std::memory_order_relaxed)) {
          return;
        }
        try {
          KDNN::MultiHeadAttention op(1, Sq, Sk, NH, HD, scale,
                                      /*maskBatchBroadcast=*/true,
                                      attention_options);
          const size_t workspace_bytes = op.GetWorkspaceSize();
          std::vector<char> workspace(workspace_bytes);
          const float* mask_row =
              mask_base ? mask_base + static_cast<int64_t>(b) * mask_batch_stride : nullptr;
          op.Run(q_data + static_cast<int64_t>(b) * query_batch_stride,
                 k_data + static_cast<int64_t>(b) * key_batch_stride,
                 v_data + static_cast<int64_t>(b) * value_batch_stride,
                 mask_row,
                 out_data + static_cast<int64_t>(b) * output_batch_stride,
                 workspace.empty() ? nullptr : workspace.data());
        } catch (...) {
          if (!have_exception.exchange(true)) {
            pending_exception = std::current_exception();
          }
        }
      },
      0);  // num_batches == 0 -> let ORT choose the shard count

  if (pending_exception) {
    std::rethrow_exception(pending_exception);
  }
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

#endif  // USE_KDNN
