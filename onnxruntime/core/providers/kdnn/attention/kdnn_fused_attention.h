// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/kdnn/attention/attention_adapter.h"

namespace onnxruntime {
namespace kdnn {

// KDNN EP kernel for the internal com.kdnn.internal::KdnnFusedAttention op. Dispatches the
// fused scaled-dot-product multi-head attention to the KDNN MultiHeadAttention
// kernel. Produced by the KDNN EP capability matcher.
class KdnnFusedAttention final : public OpKernel {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(KdnnFusedAttention);
  explicit KdnnFusedAttention(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int num_heads_;
  std::optional<float> scale_;  // Absent means derive 1/sqrt(head_size) at run time.
  AttentionAlgorithm algorithm_;
  bool onnx_softmax_ = false;
  std::optional<KeepMaskTransform> keep_mask_;
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG mlas_config_;
};

}  // namespace kdnn
}  // namespace onnxruntime
