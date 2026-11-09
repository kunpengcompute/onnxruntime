// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.

#include "core/providers/kdnn/attention/kdnn_fused_attention.h"

#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <string>

#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/platform/threadpool.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "kdnn.hpp"
#include "core/common/safeint.h"
#include "core/providers/kdnn/kdnn_execution_provider.h"
#include "core/providers/kdnn/kdnn_kernel_registry.h"
#include "core/providers/kdnn/attention/attention_adapter.h"

namespace onnxruntime {
namespace kdnn {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(KdnnFusedAttention, kKdnnDomain, 1, 1, kKdnnExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                  KdnnFusedAttention);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(KdnnFusedAttention, kKdnnDomain, 2, 2, kKdnnExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                  KdnnFusedAttention);

KdnnFusedAttention::KdnnFusedAttention(const OpKernelInfo& info) : OpKernel(info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(
      info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0 && num_heads <= std::numeric_limits<int>::max(),
      "KdnnFusedAttention requires a positive num_heads attribute");
  num_heads_ = static_cast<int>(num_heads);

  float scale = 0.0f;
  if (info.GetAttr<float>("scale", &scale).IsOK()) {
    ORT_ENFORCE(std::isfinite(scale), "KdnnFusedAttention scale must be finite");
    scale_ = scale;
  }
  algorithm_ = static_cast<const KDNNExecutionProvider*>(info.GetExecutionProvider())->Options().attention_algorithm;
  onnx_softmax_ = info.node().SinceVersion() == 2;
  mlas_config_.use_kleidiai =
      info.GetConfigOptions().GetConfigOrDefault(kOrtSessionOptionsMlasDisableKleidiAi, "0") != "1";
  std::string mask_mode = "additive";
  if (onnx_softmax_) {
    mask_mode = info.GetAttrOrDefault<std::string>("mask_mode", "additive");
  }
  ORT_ENFORCE(mask_mode == "additive" || mask_mode == "keep", "Invalid KDNN mask_mode");
  if (mask_mode == "keep") {
    KeepMaskTransform transform{};
    ORT_ENFORCE(info.GetAttr("mask_offset", &transform.offset).IsOK() && std::isfinite(transform.offset) &&
                    info.GetAttr("mask_multiplier", &transform.multiplier).IsOK() &&
                    std::isfinite(transform.multiplier),
                "KDNN keep mask requires finite mask_offset and mask_multiplier");
    keep_mask_ = transform;
  }
}

Status KdnnFusedAttention::Compute(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* mask = context->Input<Tensor>(3);  // optional

  ORT_RETURN_IF_NOT(query != nullptr && key != nullptr && value != nullptr,
                    "KdnnFusedAttention requires query, key and value inputs");

  ORT_RETURN_IF_NOT(!keep_mask_ || (mask && mask->Shape().NumDimensions() == 4),
                    "KDNN keep mask requires rank-4 mask input");

  const auto& q_dims = query->Shape().GetDims();
  const auto& k_dims = key->Shape().GetDims();
  const auto& v_dims = value->Shape().GetDims();
  const bool split_heads = q_dims.size() == 4;
  ORT_RETURN_IF_NOT(
      (q_dims.size() == 3 || split_heads) && k_dims.size() == q_dims.size() && v_dims.size() == q_dims.size(),
      "KdnnFusedAttention expects matching rank-3 [B,S,hidden] or "
      "rank-4 [B,S,H,d] query/key/value");

  const int64_t query_batch = q_dims[0];
  const int64_t key_batch = k_dims[0];
  const int64_t value_batch = v_dims[0];
  const int64_t seq_q = q_dims[1];
  const int64_t seq_k = k_dims[1];
  int64_t hidden = 0;
  int64_t head_dim = 0;

  ORT_RETURN_IF_NOT(v_dims[1] == seq_k, "KdnnFusedAttention: K/V sequence length mismatch");

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
    ORT_RETURN_IF_NOT(seq_q == 1, "KdnnFusedAttention: rank-4 input currently requires q_sequence_length == 1");
    ORT_RETURN_IF_NOT(q_dims[2] == num_heads_ && k_dims[2] == num_heads_ && v_dims[2] == num_heads_,
                      "KdnnFusedAttention: rank-4 head dimension must match num_heads");
    ORT_RETURN_IF_NOT(k_dims[3] == q_dims[3] && v_dims[3] == q_dims[3],
                      "KdnnFusedAttention: rank-4 head size mismatch across Q/K/V");
    head_dim = q_dims[3];
    hidden = SafeInt<int64_t>(num_heads_) * head_dim;
  } else {
    hidden = q_dims[2];
    ORT_RETURN_IF_NOT(k_dims[2] == hidden && v_dims[2] == hidden, "KdnnFusedAttention: hidden mismatch across Q/K/V");
    ORT_RETURN_IF_NOT(hidden % num_heads_ == 0, "KdnnFusedAttention: hidden_size not divisible by num_heads");
    head_dim = hidden / num_heads_;
  }
  const float scale = scale_ ? *scale_ : 1.0f / std::sqrt(static_cast<float>(head_dim));

  // Mask handling: optional additive pre-softmax bias of shape [B or 1, 1, 1, Sk].
  // Batch dimensions follow ONNX multidirectional broadcasting. Each output batch
  // row is processed as an independent batch=1 KDNN problem.
  const float* mask_base = nullptr;
  int64_t mask_batch_stride = 0;  // elements to advance per batch (0 == shared)
  int64_t attention_batch = score_batch;
  if (mask != nullptr) {
    const auto& m_dims = mask->Shape().GetDims();
    ORT_RETURN_IF_NOT(m_dims.size() >= 2, "KdnnFusedAttention: mask must have batch and sequence dimensions");
    ORT_RETURN_IF_NOT(m_dims.back() == seq_k, "KdnnFusedAttention: mask last dim must equal K sequence length");
    const int64_t mask_batch = m_dims.front();
    int64_t masked_batch = 0;
    ORT_RETURN_IF_NOT(broadcast_batch(attention_batch, mask_batch, masked_batch),
                      "KdnnFusedAttention: score and mask batch dimensions are not broadcastable");
    attention_batch = masked_batch;
    for (size_t i = 1; i + 1 < m_dims.size(); ++i) {
      ORT_RETURN_IF_NOT(m_dims[i] == 1, "KdnnFusedAttention: intermediate mask dimensions must be 1");
    }
    mask_base = mask->Data<float>();
    mask_batch_stride = (mask_batch == 1) ? 0 : seq_k;
  }

  int64_t output_batch = 0;
  ORT_RETURN_IF_NOT(broadcast_batch(attention_batch, value_batch, output_batch),
                    "KdnnFusedAttention: attention and V batch dimensions are not broadcastable");

  Tensor* output = split_heads ? context->Output(0, TensorShape({output_batch, num_heads_, seq_q, head_dim}))
                               : context->Output(0, TensorShape({output_batch, seq_q, hidden}));
  if (output_batch == 0 || seq_q == 0) {
    return Status::OK();  // nothing to compute
  }

  const auto Sq = static_cast<KDNN::SizeType>(seq_q);
  const auto Sk = static_cast<KDNN::SizeType>(seq_k);
  const auto NH = static_cast<KDNN::SizeType>(num_heads_);
  const auto HD = static_cast<KDNN::SizeType>(head_dim);

  KDNN::Status vstatus = KDNN::MultiHeadAttention::ValidateInput(1, Sq, Sk, NH, HD);
  ORT_RETURN_IF_NOT(vstatus == KDNN::Status::SUCCESS, "KdnnFusedAttention: KDNN ValidateInput rejected the shapes");

  AttentionParameters parameters;
  parameters.query_length = Sq;
  parameters.key_length = Sk;
  parameters.num_heads = NH;
  parameters.head_size = HD;
  parameters.scale = scale;
  parameters.algorithm = algorithm_;
  parameters.preserve_onnx_softmax = onnx_softmax_;
  parameters.keep_mask = keep_mask_;
  parameters.mlas_config = mlas_config_;

  const float* q_data = query->Data<float>();
  const float* k_data = key->Data<float>();
  const float* v_data = value->Data<float>();
  float* out_data = output->MutableData<float>();
  const int64_t output_batch_stride = SafeInt<int64_t>(seq_q) * hidden;
  const int64_t query_batch_stride = query_batch == 1 ? 0 : output_batch_stride;
  const int64_t key_batch_stride = key_batch == 1 ? 0 : static_cast<int64_t>(SafeInt<int64_t>(seq_k) * hidden);
  const int64_t value_batch_stride = value_batch == 1 ? 0 : static_cast<int64_t>(SafeInt<int64_t>(seq_k) * hidden);

  auto* tp = context->GetOperatorThreadPool();

  // KDNN owns primitive reuse. ORT keeps only a grow-only workspace per worker;
  // a worker runs one callback at a time, so the scratch buffer is exclusive.
  std::atomic<bool> have_exception{false};
  std::exception_ptr pending_exception;
  concurrency::ThreadPool::TryBatchParallelFor(
      tp, static_cast<std::ptrdiff_t>(output_batch),
      [&](std::ptrdiff_t b) {
        if (have_exception.load(std::memory_order_relaxed)) {
          return;
        }
        try {
          const int64_t batch = static_cast<int64_t>(b);
          const float* mask_row = mask_base ? mask_base + batch * mask_batch_stride : nullptr;
          RunAttentionBatch(parameters, q_data + batch * query_batch_stride, k_data + batch * key_batch_stride,
                            v_data + batch * value_batch_stride, mask_row, out_data + batch * output_batch_stride);
        } catch (...) {
          if (!have_exception.exchange(true)) {
            pending_exception = std::current_exception();
          }
        }
      },
      0);  // num_batches == 0 -> let ORT choose the shard count

  if (pending_exception) {
    try {
      std::rethrow_exception(pending_exception);
    } catch (const std::exception& e) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "KDNN MHA ", Node().Name(), ": ", e.what());
    } catch (...) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "KDNN MHA ", Node().Name(), ": unknown exception");
    }
  }
  return Status::OK();
}

}  // namespace kdnn
}  // namespace onnxruntime
