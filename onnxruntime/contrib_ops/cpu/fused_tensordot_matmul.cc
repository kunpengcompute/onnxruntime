// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(__aarch64__)

#include "contrib_ops/cpu/fused_tensordot_matmul.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/math.h"

#include <algorithm>

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    FusedTensordotMatMul,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FusedTensordotMatMul);

FusedTensordotMatMul::FusedTensordotMatMul(const OpKernelInfo& info) : OpKernel(info) {
  ORT_THROW_IF_ERROR(info.GetAttrs("free_axes", free_axes_));
  ORT_THROW_IF_ERROR(info.GetAttrs("contract_axes", contract_axes_));
  SetupMlasBackendKernelSelectorFromConfigOptions(mlas_backend_kernel_selector_config_, info.GetConfigOptions());
}

Status FusedTensordotMatMul::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weight = context->Input<Tensor>(1);
  const TensorShape& input_shape = input->Shape();
  const TensorShape& weight_shape = weight->Shape();

  ORT_RETURN_IF_NOT(weight_shape.NumDimensions() == 2,
                    "FusedTensordotMatMul only supports a 2D weight tensor.");
  ORT_RETURN_IF_NOT(contract_axes_.size() == 1,
                    "FusedTensordotMatMul currently only supports one contract axis.");

  const int64_t rank = input_shape.NumDimensions();
  InlinedVector<int64_t> normalized_free_axes = free_axes_;
  InlinedVector<int64_t> normalized_contract_axes = contract_axes_;
  auto normalize_axis = [rank](int64_t& axis) {
    if (axis < 0) {
      axis += rank;
    }
  };

  for (auto& axis : normalized_free_axes) {
    normalize_axis(axis);
    ORT_RETURN_IF_NOT(axis >= 0 && axis < rank, "free_axes contains an out-of-range axis.");
  }

  for (auto& axis : normalized_contract_axes) {
    normalize_axis(axis);
    ORT_RETURN_IF_NOT(axis >= 0 && axis < rank, "contract_axes contains an out-of-range axis.");
  }

  const int64_t contract_axis = normalized_contract_axes[0];
  ORT_RETURN_IF_NOT(contract_axis == rank - 1,
                    "FusedTensordotMatMul currently requires the contract axis to be the last input dimension.");
  ORT_RETURN_IF_NOT(input_shape[contract_axis] == weight_shape[0],
                    "Input contract dimension must match weight dimension 0.");

  TensorShapeVector output_dims;
  output_dims.reserve(normalized_free_axes.size() + 1);
  int64_t m = 1;
  for (const auto axis : normalized_free_axes) {
    const int64_t dim = input_shape[axis];
    output_dims.push_back(dim);
    m *= dim;
  }

  const int64_t k = weight_shape[0];
  const int64_t n = weight_shape[1];
  output_dims.push_back(n);

  Tensor* output = context->Output(0, TensorShape(output_dims));
  if (output->Shape().Size() == 0) {
    return Status::OK();
  }

  if (k == 0) {
    std::fill_n(output->MutableData<float>(), static_cast<size_t>(output->Shape().Size()), 0.0f);
    return Status::OK();
  }

  math::MatMul<float>(static_cast<int>(m),
                      static_cast<int>(n),
                      static_cast<int>(k),
                      input->Data<float>(),
                      weight->Data<float>(),
                      output->MutableData<float>(),
                      context->GetOperatorThreadPool(),
                      &mlas_backend_kernel_selector_config_);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

#endif  // defined(__aarch64__)
