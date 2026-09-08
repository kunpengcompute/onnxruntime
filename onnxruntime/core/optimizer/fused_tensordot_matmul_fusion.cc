// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.

#include "core/optimizer/fused_tensordot_matmul_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "core/providers/common.h"

#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {

static bool IsFusedTensordotMatMulEnabled() {
  const char* enable_fusion = std::getenv("ORT_ENABLE_FUSED_TENSORDOT_MATMUL");
  return enable_fusion != nullptr && std::string_view(enable_fusion) == "1";
}

static bool GetAxesFromUnsqueezeNode(const Graph& graph, const Node& unsqueeze, InlinedVector<int64_t>& axes) {
  if (graph_utils::MatchesOpSinceVersion(unsqueeze, {1, 11})) {
    return graph_utils::GetRepeatedNodeAttributeValues(unsqueeze, "axes", axes);
  } else if (graph_utils::MatchesOpSinceVersion(unsqueeze, {13})) {
    const NodeArg* axes_node_arg = unsqueeze.InputDefs()[1];
    return optimizer_utils::AppendTensorFromInitializer(graph, *axes_node_arg, axes, true);
  }

  return false;
}

static bool IsDefaultShapeNode(const Node& shape) {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(shape, "Shape", {1, 13, 15, 19, 21, 23, 24, 25})) {
    return false;
  }

  if (graph_utils::MatchesOpSinceVersion(shape, {15})) {
    const auto* start_attr = graph_utils::GetNodeAttribute(shape, "start");
    const auto* end_attr = graph_utils::GetNodeAttribute(shape, "end");
    if ((start_attr && start_attr->has_i() && start_attr->i() != 0) || end_attr) {
      return false;
    }
  }

  return true;
}

static const Node* GetOptionalIntCastInputNode(const Node& node, int input_index) {
  const Node* input_node = graph_utils::GetInputNode(node, input_index);
  if (input_node == nullptr) {
    return nullptr;
  }

  if (graph_utils::IsSupportedOptypeVersionAndDomain(*input_node, "Cast", {1, 6, 9, 13, 19, 21, 23, 24, 25})) {
    const auto* to_attr = graph_utils::GetNodeAttribute(*input_node, "to");
    if (to_attr == nullptr || !to_attr->has_i() ||
        (to_attr->i() != ONNX_NAMESPACE::TensorProto_DataType_INT32 &&
         to_attr->i() != ONNX_NAMESPACE::TensorProto_DataType_INT64)) {
      return nullptr;
    }

    input_node = graph_utils::GetInputNode(*input_node, 0);
  }

  return input_node;
}

struct TensordotFlattenShapeInput {
  const Node* shape = nullptr;
  InlinedVector<int64_t> axes;
};

static bool MatchTensordotFlattenShapeInput(Graph& graph, const Node& concat, int input_index,
                                            TensordotFlattenShapeInput& info) {
  const Node* unsqueeze = graph_utils::GetInputNode(concat, input_index);
  if (unsqueeze == nullptr ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*unsqueeze, "Unsqueeze", {1, 11, 13, 21, 23, 24, 25})) {
    return false;
  }

  InlinedVector<int64_t> unsqueeze_axes;
  if (!(GetAxesFromUnsqueezeNode(graph, *unsqueeze, unsqueeze_axes) &&
        unsqueeze_axes.size() == 1 && unsqueeze_axes[0] == 0)) {
    return false;
  }

  const Node* reduce_prod = graph_utils::GetInputNode(*unsqueeze, 0);
  if (reduce_prod == nullptr ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*reduce_prod, "ReduceProd", {1, 11, 13, 18})) {
    return false;
  }

  const auto* keepdims_attr = graph_utils::GetNodeAttribute(*reduce_prod, "keepdims");
  if (keepdims_attr == nullptr || !keepdims_attr->has_i() || keepdims_attr->i() != 0) {
    return false;
  }

  InlinedVector<int64_t> reduce_axes;
  if (reduce_prod->InputDefs().size() >= 2 &&
      optimizer_utils::AppendTensorFromInitializer(graph, *reduce_prod->InputDefs()[1], reduce_axes, true)) {
    // axes input was added in later opsets.
  } else if (!graph_utils::GetRepeatedNodeAttributeValues(*reduce_prod, "axes", reduce_axes)) {
    return false;
  }

  if (reduce_axes.size() != 1 || reduce_axes[0] != 0) {
    return false;
  }

  const Node* gather = graph_utils::GetInputNode(*reduce_prod, 0);
  if (gather == nullptr ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*gather, "Gather", {1, 11, 13})) {
    return false;
  }

  const auto* axis_attr = graph_utils::GetNodeAttribute(*gather, "axis");
  if (axis_attr != nullptr && axis_attr->has_i() && axis_attr->i() != 0) {
    return false;
  }

  InlinedVector<int64_t> axes;
  if (gather->InputDefs().size() < 2 ||
      !optimizer_utils::AppendTensorFromInitializer(graph, *gather->InputDefs()[1], axes, true) ||
      axes.empty()) {
    return false;
  }

  const Node* shape = GetOptionalIntCastInputNode(*gather, 0);
  if (shape == nullptr || !IsDefaultShapeNode(*shape)) {
    return false;
  }

  info.shape = shape;
  info.axes = std::move(axes);
  return true;
}

static bool NormalizeAxes(InlinedVector<int64_t>& axes, int64_t rank) {
  for (auto& axis : axes) {
    if (!IsAxisInRange(axis, rank)) {
      return false;
    }

    if (axis < 0) {
      axis += rank;
    }
  }

  return true;
}

static bool ShapeDimsKnownForAxes(const NodeArg& input, gsl::span<const int64_t> axes, int64_t& dim_product) {
  const auto* input_shape = input.Shape();
  if (input_shape == nullptr) {
    return false;
  }

  dim_product = 1;
  for (const auto axis : axes) {
    const auto& dim = input_shape->dim(gsl::narrow<int>(axis));
    if (!dim.has_dim_value() || dim.dim_value() <= 0) {
      return false;
    }
    dim_product *= dim.dim_value();
  }

  return dim_product > 0;
}

static bool IsIdentityTranspose(const Node& transpose, int64_t rank) {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(transpose, "Transpose", {1, 13, 21, 23, 24, 25})) {
    return false;
  }

  InlinedVector<int64_t> perm;
  if (!graph_utils::GetRepeatedNodeAttributeValues(transpose, "perm", perm)) {
    return true;
  }

  if (perm.size() != gsl::narrow<size_t>(rank)) {
    return false;
  }

  for (int64_t i = 0; i < rank; ++i) {
    if (perm[gsl::narrow<size_t>(i)] != i) {
      return false;
    }
  }

  return true;
}

static bool HasFloatTensorType(const NodeArg& node_arg) {
  const auto* type_proto = node_arg.TypeAsProto();
  return type_proto != nullptr &&
         type_proto->has_tensor_type() &&
         type_proto->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
}

Status FusedTensordotMatMulFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                             const logging::Logger& logger) const {
  if (!IsFusedTensordotMatMulEnabled()) {
    return Status::OK();
  }

  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  int fused_count = 0;
  for (auto node_index : node_topology_list) {
    auto* p_reshape = graph.GetNode(node_index);
    if (p_reshape == nullptr) {
      continue;
    }

    Node& reshape = *p_reshape;
    ORT_RETURN_IF_ERROR(Recurse(reshape, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(reshape, "Reshape", {5, 13, 14, 19, 21, 23, 24, 25}) ||
        reshape.InputDefs().size() < 2 ||
        !graph_utils::IsSupportedProvider(reshape, GetCompatibleExecutionProviders())) {
      continue;
    }

    const auto* attr_proto = graph_utils::GetNodeAttribute(reshape, "allowzero");
    if ((nullptr != attr_proto) && attr_proto->has_i() && attr_proto->i() != 0) {
      continue;
    }

    const int newly_fused = FusedTensordotMatMulFusion::Fuse(reshape, graph, logger);
    if (newly_fused > 0) {
      fused_count += newly_fused;
      modified = true;
    }
  }

  if (fused_count > 0) {
    LOGS(logger, INFO) << "Total FusedTensordotMatMul node count: " << fused_count;
  }

  return Status::OK();
}

int FusedTensordotMatMulFusion::Fuse(Node& reshape, Graph& graph, const logging::Logger& logger) {
  if (reshape.InputDefs().size() < 2 ||
      reshape.GetOutputEdgesCount() == 0 ||
      graph.NodeProducesGraphOutput(reshape) ||
      !HasFloatTensorType(*reshape.InputDefs()[0])) {
    return 0;
  }

  // Accept the TensorFlow lowering both before and after a no-op Transpose
  // elimination pass. If a Transpose is present, only an identity permutation
  // may be bypassed; a real layout change must never be folded into this op.
  const Node* transpose = nullptr;
  const Node* reshape_input_node = graph_utils::GetInputNode(reshape, 0);
  const NodeArg* original_input = reshape.InputDefs()[0];
  if (reshape_input_node != nullptr &&
      graph_utils::IsSupportedOptypeVersionAndDomain(*reshape_input_node, "Transpose", {1, 13, 21, 23, 24, 25})) {
    if (reshape_input_node->InputDefs().empty() || reshape_input_node->InputDefs()[0] == nullptr) {
      return 0;
    }

    const NodeArg* transpose_input = reshape_input_node->InputDefs()[0];
    const auto* transpose_input_shape = transpose_input->Shape();
    if (transpose_input_shape == nullptr ||
        !IsIdentityTranspose(*reshape_input_node, transpose_input_shape->dim_size())) {
      return 0;
    }

    transpose = reshape_input_node;
    original_input = transpose_input;
  }

  const auto* input_shape = original_input->Shape();
  if (input_shape == nullptr) {
    return 0;
  }

  const int64_t rank = input_shape->dim_size();

  const Node* shape_input_node = graph_utils::GetInputNode(reshape, 1);
  if (shape_input_node == nullptr) {
    return 0;
  }

  const Node* concat = shape_input_node;
  if (graph_utils::IsSupportedOptypeVersionAndDomain(*shape_input_node, "Cast", {1, 6, 9, 13, 19, 21, 23, 24, 25})) {
    concat = graph_utils::GetInputNode(*shape_input_node, 0);
    if (concat == nullptr) {
      return 0;
    }
  }

  if (!graph_utils::IsSupportedOptypeVersionAndDomain(*concat, "Concat", {1, 4, 11, 13})) {
    return 0;
  }

  const auto* axis_attr = graph_utils::GetNodeAttribute(*concat, "axis");
  if (axis_attr == nullptr || !axis_attr->has_i() || axis_attr->i() != 0 ||
      concat->InputArgCount().front() != 2) {
    return 0;
  }

  TensordotFlattenShapeInput free_info;
  TensordotFlattenShapeInput contract_info;
  if (!MatchTensordotFlattenShapeInput(graph, *concat, 0, free_info) ||
      !MatchTensordotFlattenShapeInput(graph, *concat, 1, contract_info)) {
    return 0;
  }

  if (free_info.shape->InputDefs()[0]->Name() != original_input->Name() ||
      contract_info.shape->InputDefs()[0]->Name() != original_input->Name() ||
      !NormalizeAxes(free_info.axes, rank) ||
      !NormalizeAxes(contract_info.axes, rank) ||
      contract_info.axes.size() != 1 ||
      contract_info.axes[0] != rank - 1) {
    return 0;
  }

  int64_t contract_dim_product = 0;
  if (!ShapeDimsKnownForAxes(*original_input, contract_info.axes, contract_dim_product)) {
    return 0;
  }
  // A zero contract product is safe here. Per-branch validation below requires
  // positive weight dimensions and an exact K-dimension match, so zero-sized
  // contraction branches are skipped instead of fused.

  InlinedVector<NodeIndex> matmul_indices;
  for (auto edge_it = reshape.OutputEdgesBegin(); edge_it != reshape.OutputEdgesEnd(); ++edge_it) {
    if (edge_it->GetDstArgIndex() == 0) {
      matmul_indices.push_back(edge_it->GetNode().Index());
    }
  }

  int fused_count = 0;
  const std::string reshape_name = reshape.Name();
  const NodeIndex reshape_index = reshape.Index();
  const NodeIndex shape_input_node_index = shape_input_node->Index();

  for (const NodeIndex matmul_index : matmul_indices) {
    Node* matmul = graph.GetNode(matmul_index);
    if (matmul == nullptr ||
        !graph_utils::IsSupportedOptypeVersionAndDomain(*matmul, "MatMul", {1, 9, 13}) ||
        matmul->GetOutputEdgesCount() != 1 ||
        graph.NodeProducesGraphOutput(*matmul) ||
        matmul->InputDefs().size() < 2 ||
        !HasFloatTensorType(*matmul->InputDefs()[1])) {
      continue;
    }

    const auto* weight = graph_utils::GetConstantInitializer(graph, matmul->InputDefs()[1]->Name(), true);
    if (weight == nullptr || weight->dims_size() != 2 || weight->dims(0) <= 0 || weight->dims(1) <= 0 ||
        contract_dim_product != weight->dims(0)) {
      continue;
    }

    const auto final_reshape_edge = matmul->OutputEdgesBegin();
    const Node& final_reshape_const = final_reshape_edge->GetNode();
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(final_reshape_const, "Reshape", {5, 13, 14, 19, 21, 23, 24, 25}) ||
        final_reshape_edge->GetDstArgIndex() != 0 ||
        graph.NodeProducesGraphOutput(final_reshape_const)) {
      continue;
    }

    const auto* final_reshape_allowzero = graph_utils::GetNodeAttribute(final_reshape_const, "allowzero");
    if (final_reshape_allowzero != nullptr &&
        final_reshape_allowzero->has_i() &&
        final_reshape_allowzero->i() != 0) {
      continue;
    }

    InlinedVector<int64_t> final_shape;
    if (final_reshape_const.InputDefs().size() < 2 ||
        !optimizer_utils::AppendTensorFromInitializer(graph, *final_reshape_const.InputDefs()[1],
                                                      final_shape, true)) {
      continue;
    }

    InlinedVector<int64_t> expected_shape;
    expected_shape.reserve(free_info.axes.size() + 1);
    for (const auto axis : free_info.axes) {
      const auto& dim = input_shape->dim(gsl::narrow<int>(axis));
      expected_shape.push_back(dim.has_dim_value() ? dim.dim_value() : -1);
    }
    expected_shape.push_back(weight->dims(1));

    if (final_shape.size() != expected_shape.size()) {
      continue;
    }

    bool shape_matches = true;
    for (size_t i = 0; i < expected_shape.size(); ++i) {
      const int64_t expected = expected_shape[i];
      const int64_t actual = final_shape[i];
      if (expected >= 0) {
        if (actual != expected) {
          shape_matches = false;
          break;
        }
      } else if (actual != -1) {
        shape_matches = false;
        break;
      }
    }

    if (!shape_matches) {
      continue;
    }

    Node* final_reshape = graph.GetNode(final_reshape_const.Index());
    if (final_reshape == nullptr) {
      continue;
    }

    NodeAttributes attrs;
    ONNX_NAMESPACE::AttributeProto free_axes_attr;
    free_axes_attr.set_name("free_axes");
    free_axes_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
    for (const auto axis : free_info.axes) {
      free_axes_attr.add_ints(axis);
    }
    attrs[free_axes_attr.name()] = std::move(free_axes_attr);

    ONNX_NAMESPACE::AttributeProto contract_axes_attr;
    contract_axes_attr.set_name("contract_axes");
    contract_axes_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
    for (const auto axis : contract_info.axes) {
      contract_axes_attr.add_ints(axis);
    }
    attrs[contract_axes_attr.name()] = std::move(contract_axes_attr);

    InlinedVector<NodeArg*> fused_inputs{const_cast<NodeArg*>(original_input), matmul->MutableInputDefs()[1]};
    InlinedVector<NodeArg*> fused_outputs{final_reshape->MutableOutputDefs()[0]};
    Node& fused_node = graph.AddNode(graph.GenerateNodeName(reshape_name + "_FusedTensordotMatMul"),
                                     "FusedTensordotMatMul",
                                     "fused TensorFlow Tensordot flatten Reshape + MatMul + final Reshape",
                                     fused_inputs,
                                     fused_outputs,
                                     &attrs,
                                     kMSDomain);
    fused_node.SetExecutionProviderType(matmul->GetExecutionProviderType());

    graph_utils::ReplaceDownstreamNodeInput(graph, *final_reshape, 0, fused_node, 0);
    graph_utils::RemoveNodeOutputEdges(graph, *matmul);
    graph_utils::RemoveNodeOutputEdges(graph, *final_reshape);
    graph.RemoveNode(final_reshape->Index());
    graph.RemoveNode(matmul_index);

    ++fused_count;
  }

  if (fused_count > 0) {
    LOGS(logger, INFO) << "Fused " << fused_count
                       << " Tensordot MatMul branches from shared flatten Reshape: " << reshape_name;
  }

  Node* remaining_reshape = graph.GetNode(reshape_index);
  if (fused_count > 0 && remaining_reshape != nullptr && remaining_reshape->GetOutputEdgesCount() == 0) {
    graph.RemoveNode(reshape_index);

    if (Node* old_shape_input_node = graph.GetNode(shape_input_node_index);
        old_shape_input_node != nullptr && old_shape_input_node->GetOutputEdgesCount() == 0) {
      graph_utils::RemoveNodesWithOneOutputBottomUp(graph, *old_shape_input_node);
    }

    if (transpose != nullptr) {
      if (Node* transpose_node = graph.GetNode(transpose->Index());
          transpose_node != nullptr && transpose_node->GetOutputEdgesCount() == 0) {
        graph_utils::RemoveNodesWithOneOutputBottomUp(graph, *transpose_node);
      }
    }
  }

  return fused_count;
}

}  // namespace onnxruntime
