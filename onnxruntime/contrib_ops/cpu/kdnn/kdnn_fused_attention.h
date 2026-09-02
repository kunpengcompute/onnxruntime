// Copyright 2025 Huawei Technologies Co., Ltd.
// Licensed under the Apache License, Version 2.0.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

// CPU kernel for the internal com.kdnn.internal::KdnnFusedAttention op. Dispatches the
// fused scaled-dot-product multi-head attention to the KDNN MultiHeadAttention
// kernel. Produced by the KdnnAttentionFusion graph transformer.
class KdnnFusedAttention final : public OpKernel {
 public:
  explicit KdnnFusedAttention(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int num_heads_;
  float scale_;  // < 0 means "derive 1/sqrt(head_size) at run time"
  bool has_scale_;
};

}  // namespace contrib
}  // namespace onnxruntime
