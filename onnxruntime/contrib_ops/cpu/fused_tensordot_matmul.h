// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/mlas_backend_kernel_selector_config_utils.h"

namespace onnxruntime {
namespace contrib {

class FusedTensordotMatMul final : public OpKernel {
 public:
  explicit FusedTensordotMatMul(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  InlinedVector<int64_t> free_axes_;
  InlinedVector<int64_t> contract_axes_;
  InlinedVector<int64_t> final_shape_;
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG mlas_backend_kernel_selector_config_;
};

}  // namespace contrib
}  // namespace onnxruntime
