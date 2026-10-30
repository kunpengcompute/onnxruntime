// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/kdnn/schema/kdnn_defs.h"
#include "core/graph/constants.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

namespace onnxruntime::kdnn {
using namespace ONNX_NAMESPACE;
namespace {
void MergeBatchDimension(TensorShapeProto::Dimension& batch, const TensorShapeProto::Dimension& next) {
  TensorShapeProto left, right, result;
  *left.add_dim() = batch;
  *right.add_dim() = next;
  bidirectionalBroadcastShapeInference(left, right, result);
  batch = result.dim(0);
}

void InferAttentionShape(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasInputShape(ctx, 0)) return;
  const auto& query_shape = getInputShape(ctx, 0);
  if (query_shape.dim_size() != 3 && query_shape.dim_size() != 4) return;

  auto batch = query_shape.dim(0);
  // Follow the QK -> mask -> V broadcast order used by the runtime kernel.
  for (size_t input_index : {1, 3, 2}) {
    if (input_index >= ctx.getNumInputs() || ctx.getInputType(input_index) == nullptr) continue;
    TensorShapeProto::Dimension input_batch;
    if (hasInputShape(ctx, input_index) && getInputShape(ctx, input_index).dim_size() > 0) {
      input_batch = getInputShape(ctx, input_index).dim(0);
    }
    // An input with missing shape can expand batch=1 at runtime. Treat its
    // batch as unknown so a downstream Shape cannot be folded to a false 1.
    MergeBatchDimension(batch, input_batch);
  }
  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  *output_shape->add_dim() = batch;
  if (query_shape.dim_size() == 4) {
    *output_shape->add_dim() = query_shape.dim(2);
    *output_shape->add_dim() = query_shape.dim(1);
    *output_shape->add_dim() = query_shape.dim(3);
  } else {
    *output_shape->add_dim() = query_shape.dim(1);
    *output_shape->add_dim() = query_shape.dim(2);
  }
}

void RegisterVersion(int version) {
  static const char* KdnnFusedAttention_doc = R"DOC(
Internal fused scaled-dot-product multi-head attention backed by KDNN. This
operator is produced by the KDNN execution provider and is not intended to be authored
directly in user models.

For each batch and head it computes:
  scores = scale * Q . K^T
  scores = scores + mask
  probs  = softmax(scores, axis=Sk)
  output = probs . V

Q/K/V may use flattened [B, S, hidden] tensors or head-split [B, S, H, d]
tensors. Their batch dimensions and the optional mask batch dimension follow
ONNX multidirectional broadcasting. For head-split inputs, Sq must be 1 and
output is [B, H, Sq, d]. The additive mask's intermediate singleton dimensions
are broadcast over heads and query rows.

In version 2, mask_mode="keep" first computes
  mask = mask_multiplier * (mask_offset - mask_input)
as separate FP32 Sub and Mul operations. This mode requires a rank-4 mask.
The default additive mode uses the mask input directly in both versions.
)DOC";
  auto schema =
      OpSchema("KdnnFusedAttention", __FILE__, __LINE__)
          .SetDomain(kKdnnDomain)
          .SinceVersion(version)
          .SetDoc(KdnnFusedAttention_doc)
          .Attr("num_heads", "Number of attention heads.", AttributeProto::INT)
          .Attr("scale", "Scaling factor applied to Q.K^T before softmax. Default is 1/sqrt(head_size).",
                AttributeProto::FLOAT, OPTIONAL_VALUE)
          .Input(0, "query", "Query tensor with shape (B, Sq, hidden) or (B, Sq, H, d).", "T")
          .Input(1, "key", "Key tensor with shape (B, Sk, hidden) or (B, Sk, H, d).", "T")
          .Input(2, "value", "Value tensor with shape (B, Sk, hidden) or (B, Sk, H, d).", "T")
          .Input(3, "mask", "Optional pre-softmax mask ending in kv_sequence_length; v2 also supports keep mode.",
                 "T", OpSchema::Optional)
          .Output(0, "output", "Output is (broadcast_B, Sq, hidden) or (broadcast_B, H, Sq, d).", "T")
          .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output to float tensors.")
          .TypeAndShapeInferenceFunction(InferAttentionShape);
  if (version == 2) {
    schema
        .Attr("mask_mode", "additive bias or keep-mask Sub/Mul adapter.", AttributeProto::STRING,
              std::string("additive"))
        .Attr("mask_offset", "Sub offset for keep mode.", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("mask_multiplier", "Mul multiplier for keep mode.", AttributeProto::FLOAT, OPTIONAL_VALUE);
  }
  ONNX_NAMESPACE::RegisterSchema(std::move(schema));
}
}  // namespace
void RegisterKdnnSchemas() {
  RegisterVersion(1);
  RegisterVersion(2);
}
}  // namespace onnxruntime::kdnn
