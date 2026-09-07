// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/initializer.h"
#include "core/optimizer/layer_norm_fusion.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "float.h"
#include <algorithm>
#include <cstdlib>
#include <deque>
#include <string_view>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

// LayerNorm supports limited data types.
static constexpr std::array<std::string_view, 4> supported_data_types{"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"};
// Default epsilon
static constexpr float DEFAULT_LAYERNORM_EPSILON = 1e-5f;

static bool IsSupportedDataType(const Node& node, int first_n_inputs = -1) {
  int input_index = 0;
  for (const auto& input_arg : node.InputDefs()) {
    if (first_n_inputs != -1 && input_index >= first_n_inputs) {
      return true;
    }
    if (std::find(supported_data_types.begin(), supported_data_types.end(),
                  *(input_arg->Type())) == supported_data_types.end()) {
      return false;
    }
    ++input_index;
  }
  return true;
}

static bool CheckAxesOnReduceMean(std::vector<int64_t>& axes_values, int64_t rank) {
  // axes has be to be consecutive and constains the last dim.
  std::sort(axes_values.begin(), axes_values.end());
  if (axes_values.back() > 0) {
    // if reduce_mean node has input shape [N, C1, C2, C3] and  axes_values = [1, 2], it's invalid.
    // handle axes_values with both positive and negative values.
    if (rank == -1) {
      return false;
    }
    std::transform(axes_values.begin(), axes_values.end(), axes_values.begin(),
                   [rank](int64_t v) { return v >= 0 ? v - rank : v; });
    std::sort(axes_values.begin(), axes_values.end());
  }
  // check if axes are consecutive
  for (size_t i = 1; i < axes_values.size(); i++) {
    if (axes_values[i] != axes_values[i - 1] + 1) {
      axes_values.clear();
      break;
    }
  }

  if (axes_values.empty() || axes_values.back() != -1) {
    // axes_values should contain the last dim.
    return false;
  }
  return true;
}

#if defined(__aarch64__)
static bool IsReciprocalMulLayerNormFusionEnabled() {
  const char* enable_fusion = std::getenv("ORT_ENABLE_RM_LN_FUSION");
  return enable_fusion != nullptr && std::string_view(enable_fusion) == "1";
}

static bool CheckAxesOnReduceMeanWithParamRank(std::vector<int64_t>& axes_values,
                                               int64_t input_rank,
                                               int64_t param_rank) {
  if (input_rank != -1) {
    return CheckAxesOnReduceMean(axes_values, input_rank);
  }

  if (param_rank <= 0 || axes_values.size() != static_cast<size_t>(param_rank)) {
    return false;
  }

  std::sort(axes_values.begin(), axes_values.end());
  for (size_t i = 1; i < axes_values.size(); i++) {
    if (axes_values[i] != axes_values[i - 1] + 1) {
      return false;
    }
  }

  if (axes_values.back() < 0) {
    return CheckAxesOnReduceMean(axes_values, input_rank);
  }

  const int64_t inferred_input_rank = axes_values.front() + param_rank;
  return CheckAxesOnReduceMean(axes_values, inferred_input_rank);
}

static std::vector<int64_t> GetAxesFromReduceMeanNode(Node& reduce_mean_node, const Graph& graph) {
  const onnxruntime::NodeAttributes& attributes = reduce_mean_node.GetAttributes();
  std::vector<int64_t> axes_values;
  // TODO: modify this codes when opset >= 18 (axes is an input).
  if (attributes.find("axes") != attributes.end()) {
    axes_values = RetrieveValues<int64_t>(attributes.at("axes"));
  } else if (reduce_mean_node.InputDefs().size() == 2) {
    const auto* axes = reduce_mean_node.InputDefs()[1];
    const auto* axes_const = graph.GetConstantInitializer(axes->Name(), true);
    if (axes_const != nullptr) {
      Initializer initializer{graph, *axes_const, graph.ModelPath()};
      auto span_axes = initializer.DataAsSpan<int64_t>();
      axes_values.insert(axes_values.end(), span_axes.begin(), span_axes.end());
    }
  }
  return axes_values;
};

static bool HasKeepDimsOne(const Node& reduce_mean_node) {
  const auto& attributes = reduce_mean_node.GetAttributes();
  const auto keepdims_it = attributes.find("keepdims");
  if (keepdims_it == attributes.end()) {
    return true;
  }

  return keepdims_it->second.i() == 1;
}

static const NodeArg* GetOtherInput(const Node& node, const NodeArg* input) {
  if (node.InputDefs().size() != 2) {
    return nullptr;
  }

  if (node.InputDefs()[0] == input) {
    return node.InputDefs()[1];
  }

  if (node.InputDefs()[1] == input) {
    return node.InputDefs()[0];
  }

  return nullptr;
}

static const Node* FirstChildByTypeWithInput(const Node& node, const std::string& op_type, const NodeArg* input) {
  for (auto it = node.OutputNodesBegin(); it != node.OutputNodesEnd(); ++it) {
    const Node& child = *it;
    if (child.OpType() == op_type &&
        std::find(child.InputDefs().begin(), child.InputDefs().end(), input) != child.InputDefs().end()) {
      return &child;
    }
  }

  return nullptr;
}

static bool IsSupportedLayerNormOp(const Node& node, const std::string& op_type,
                                   std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> versions,
                                   const Node& provider_source, int first_n_inputs = -1) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, op_type, versions) &&
         node.GetExecutionProviderType() == provider_source.GetExecutionProviderType() &&
         IsSupportedDataType(node, first_n_inputs);
}

static Status TryFuseReciprocalMulLayerNorm(Graph& graph,
                                            Node& reduce_mean_node,
                                            Node& sub_node,
                                            const logging::Logger& logger,
                                            bool& modified) {
  bool fused = false;

#define RETURN_RECIPROCAL_MUL_LN_SKIP(message)                                                     \
  do {                                                                                              \
    LOGS(logger, VERBOSE) << "ReciprocalMulLayerNormFusion skip. ReduceMean='"                     \
                          << reduce_mean_node.Name() << "', Sub='" << sub_node.Name() << "': "      \
                          << message;                                                              \
    modified = fused;                                                                               \
    return Status::OK();                                                                            \
  } while (false)

  const NodeArg* x_input = reduce_mean_node.MutableInputDefs()[0];
  const NodeArg* mean = reduce_mean_node.MutableOutputDefs()[0];
  if (sub_node.MutableInputDefs().size() != 2 ||
      sub_node.MutableInputDefs()[0] != x_input ||
      sub_node.MutableInputDefs()[1] != mean ||
      !optimizer_utils::CheckOutputEdges(graph, sub_node, 2)) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Sub is not Sub(x, mean) with exactly two output edges. "
                                  "sub_input_count="
                                  << sub_node.MutableInputDefs().size()
                                  << ", sub_output_edges=" << sub_node.GetOutputEdgesCount()
                                  << ", expected x='" << x_input->Name()
                                  << "', expected mean='" << mean->Name() << "'");
  }

  const Node* p_square_mul = graph_utils::FirstChildByType(sub_node, "Mul");
  if (p_square_mul == nullptr) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Cannot find Mul child for square after Sub.");
  }

  Node& square_mul = *graph.GetNode(p_square_mul->Index());
  const NodeArg* sub_output = sub_node.MutableOutputDefs()[0];
  if (!IsSupportedLayerNormOp(square_mul, "Mul", {7, 13, 14}, reduce_mean_node) ||
      !optimizer_utils::CheckOutputEdges(graph, square_mul, 1) ||
      square_mul.MutableInputDefs().size() != 2 ||
      square_mul.MutableInputDefs()[0] != sub_output ||
      square_mul.MutableInputDefs()[1] != sub_output) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Square Mul does not match Mul(sub_out, sub_out) or has unsupported properties. "
                                  "mul='" << square_mul.Name()
                                          << "', input_count=" << square_mul.MutableInputDefs().size()
                                          << ", output_edges=" << square_mul.GetOutputEdgesCount());
  }

  const Node* p_reduce_mean2 = graph_utils::FirstChildByType(square_mul, "ReduceMean");
  if (p_reduce_mean2 == nullptr) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Cannot find second ReduceMean child after square Mul '" << square_mul.Name() << "'.");
  }

  Node& reduce_mean2_node = *graph.GetNode(p_reduce_mean2->Index());
  if (!IsSupportedLayerNormOp(reduce_mean2_node, "ReduceMean", {1, 11, 13, 18}, reduce_mean_node, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, reduce_mean2_node, 1) ||
      !HasKeepDimsOne(reduce_mean_node) ||
      !HasKeepDimsOne(reduce_mean2_node)) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Second ReduceMean is unsupported, does not keep dims, or has unexpected consumers. "
                                  "reduce_mean2='" << reduce_mean2_node.Name()
                                                   << "', output_edges=" << reduce_mean2_node.GetOutputEdgesCount()
                                                   << ", first_keepdims_one=" << HasKeepDimsOne(reduce_mean_node)
                                                   << ", second_keepdims_one=" << HasKeepDimsOne(reduce_mean2_node));
  }

  const Node* p_add_epsilon = graph_utils::FirstChildByType(reduce_mean2_node, "Add");
  if (p_add_epsilon == nullptr) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Cannot find Add(epsilon) child after second ReduceMean '"
                                  << reduce_mean2_node.Name() << "'.");
  }

  Node& add_epsilon = *graph.GetNode(p_add_epsilon->Index());
  if (!IsSupportedLayerNormOp(add_epsilon, "Add", {7, 13, 14}, reduce_mean_node) ||
      !optimizer_utils::CheckOutputEdges(graph, add_epsilon, 1)) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Add(epsilon) is unsupported or has unexpected consumers. add='"
                                  << add_epsilon.Name() << "', output_edges=" << add_epsilon.GetOutputEdgesCount());
  }

  const Node* p_sqrt = graph_utils::FirstChildByType(add_epsilon, "Sqrt");
  if (p_sqrt == nullptr) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Cannot find Sqrt child after Add(epsilon) '" << add_epsilon.Name() << "'.");
  }

  Node& sqrt_node = *graph.GetNode(p_sqrt->Index());
  if (!IsSupportedLayerNormOp(sqrt_node, "Sqrt", {6, 13}, reduce_mean_node) ||
      !optimizer_utils::CheckOutputEdges(graph, sqrt_node, 1)) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Sqrt is unsupported or has unexpected consumers. sqrt='"
                                  << sqrt_node.Name() << "', output_edges=" << sqrt_node.GetOutputEdgesCount());
  }

  const Node* p_reciprocal = graph_utils::FirstChildByType(sqrt_node, "Reciprocal");
  if (p_reciprocal == nullptr) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Cannot find Reciprocal child after Sqrt '" << sqrt_node.Name() << "'.");
  }

  Node& reciprocal_node = *graph.GetNode(p_reciprocal->Index());
  if (!IsSupportedLayerNormOp(reciprocal_node, "Reciprocal", {6, 13}, reduce_mean_node) ||
      !optimizer_utils::CheckOutputEdges(graph, reciprocal_node, 1)) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Reciprocal is unsupported or has unexpected consumers. reciprocal='"
                                  << reciprocal_node.Name()
                                  << "', output_edges=" << reciprocal_node.GetOutputEdgesCount());
  }

  const Node* p_scale_mul = graph_utils::FirstChildByType(reciprocal_node, "Mul");
  if (p_scale_mul == nullptr) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Cannot find Mul(scale) child after Reciprocal '" << reciprocal_node.Name() << "'.");
  }

  Node& scale_mul = *graph.GetNode(p_scale_mul->Index());
  if (!IsSupportedLayerNormOp(scale_mul, "Mul", {7, 13, 14}, reduce_mean_node) ||
      !optimizer_utils::CheckOutputEdges(graph, scale_mul, 2)) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Mul(scale) is unsupported or does not have exactly two consumers. scale_mul='"
                                  << scale_mul.Name() << "', output_edges=" << scale_mul.GetOutputEdgesCount());
  }

  const NodeArg* scale = GetOtherInput(scale_mul, reciprocal_node.MutableOutputDefs()[0]);
  if (scale == nullptr || scale->Shape() == nullptr) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Cannot identify scale input or scale shape is missing. scale_mul='"
                                  << scale_mul.Name() << "'");
  }

  const Node* p_mean_mul = FirstChildByTypeWithInput(scale_mul, "Mul", mean);
  const Node* p_x_mul = FirstChildByTypeWithInput(scale_mul, "Mul", x_input);
  if (p_mean_mul == nullptr || p_x_mul == nullptr || p_mean_mul == p_x_mul) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Cannot find separate mean*scale_inv_std and x*scale_inv_std Mul consumers. "
                                  "mean='" << mean->Name() << "', x='" << x_input->Name() << "'");
  }

  Node& mean_mul = *graph.GetNode(p_mean_mul->Index());
  Node& x_mul = *graph.GetNode(p_x_mul->Index());
  if (!IsSupportedLayerNormOp(mean_mul, "Mul", {7, 13, 14}, reduce_mean_node) ||
      !IsSupportedLayerNormOp(x_mul, "Mul", {7, 13, 14}, reduce_mean_node) ||
      !optimizer_utils::CheckOutputEdges(graph, mean_mul, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, x_mul, 1) ||
      GetOtherInput(mean_mul, mean) != scale_mul.MutableOutputDefs()[0] ||
      GetOtherInput(x_mul, x_input) != scale_mul.MutableOutputDefs()[0]) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Mean/X Mul consumers are unsupported or do not consume scale_inv_std correctly. "
                                  "mean_mul='" << mean_mul.Name() << "', mean_mul_edges="
                                               << mean_mul.GetOutputEdgesCount()
                                               << ", x_mul='" << x_mul.Name() << "', x_mul_edges="
                                               << x_mul.GetOutputEdgesCount());
  }

  const Node* p_bias_sub = graph_utils::FirstChildByType(mean_mul, "Sub");
  if (p_bias_sub == nullptr) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Cannot find bias Sub child after mean Mul '" << mean_mul.Name() << "'.");
  }

  Node& bias_sub = *graph.GetNode(p_bias_sub->Index());
  if (!IsSupportedLayerNormOp(bias_sub, "Sub", {7, 13, 14}, reduce_mean_node) ||
      !optimizer_utils::CheckOutputEdges(graph, bias_sub, 1) ||
      bias_sub.MutableInputDefs().size() != 2 ||
      bias_sub.MutableInputDefs()[1] != mean_mul.MutableOutputDefs()[0]) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Bias Sub does not match Sub(bias, mean_mul_out) or has unsupported properties. "
                                  "bias_sub='" << bias_sub.Name()
                                               << "', input_count=" << bias_sub.MutableInputDefs().size()
                                               << ", output_edges=" << bias_sub.GetOutputEdgesCount());
  }

  const NodeArg* bias = bias_sub.MutableInputDefs()[0];
  if (bias == nullptr || bias->Shape() == nullptr) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Cannot identify bias input or bias shape is missing. bias_sub='"
                                  << bias_sub.Name() << "'");
  }

  const Node* p_last_add = graph_utils::FirstChildByType(bias_sub, "Add");
  if (p_last_add == nullptr) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Cannot find final Add child after bias Sub '" << bias_sub.Name() << "'.");
  }

  Node& last_add = *graph.GetNode(p_last_add->Index());
  if (!IsSupportedLayerNormOp(last_add, "Add", {7, 13, 14}, reduce_mean_node) ||
      GetOtherInput(last_add, bias_sub.MutableOutputDefs()[0]) != x_mul.MutableOutputDefs()[0]) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Final Add does not combine x_mul_out and bias_sub_out. last_add='"
                                  << last_add.Name() << "'");
  }

  auto axes_values = GetAxesFromReduceMeanNode(reduce_mean_node, graph);
  auto axes2_values = GetAxesFromReduceMeanNode(reduce_mean2_node, graph);
  if (axes_values.empty() || axes2_values.empty()) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("ReduceMean axes are missing or non-constant. axes1_size="
                                  << axes_values.size() << ", axes2_size=" << axes2_values.size());
  }

  auto input_shape = reduce_mean_node.MutableInputDefs()[0]->Shape();
  auto rank = input_shape ? input_shape->dim().size() : -1;
  const int64_t param_rank = scale->Shape()->dim_size();
  if (param_rank != bias->Shape()->dim_size()) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Scale and bias ranks differ. scale_rank="
                                  << param_rank << ", bias_rank=" << bias->Shape()->dim_size());
  }

  if (!CheckAxesOnReduceMeanWithParamRank(axes_values, rank, param_rank) ||
      !CheckAxesOnReduceMeanWithParamRank(axes2_values, rank, param_rank) ||
      axes_values != axes2_values) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("ReduceMean axes are not valid LayerNorm axes or axes differ. rank="
                                  << rank << ", param_rank=" << param_rank);
  }

#ifndef ENABLE_TRAINING_CORE
  if (axes_values.size() != 1 || scale->Shape()->dim_size() != 1 || bias->Shape()->dim_size() != 1) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("Scale/bias are not 1D in inference build. axes_size="
                                  << axes_values.size() << ", scale_rank=" << scale->Shape()->dim_size()
                                  << ", bias_rank=" << bias->Shape()->dim_size());
  }
#endif

  for (int i = 0; i < scale->Shape()->dim_size(); i++) {
    if (scale->Shape()->dim(i).dim_value() != bias->Shape()->dim(i).dim_value()) {
      RETURN_RECIPROCAL_MUL_LN_SKIP("Scale and bias dimensions differ at dim " << i << ". scale_dim="
                                                                            << scale->Shape()->dim(i).dim_value()
                                                                            << ", bias_dim="
                                                                            << bias->Shape()->dim(i).dim_value());
    }
  }

  if (reduce_mean_node.GetExecutionProviderType() == kCpuExecutionProvider &&
      x_input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    RETURN_RECIPROCAL_MUL_LN_SKIP("CPU float16 LayerNormalization is not supported for this fusion.");
  }

  InlinedVector<NodeArg*> layer_norm_input_defs{const_cast<NodeArg*>(x_input),
                                                const_cast<NodeArg*>(scale),
                                                const_cast<NodeArg*>(bias)};
  Node& layer_norm_node = graph.AddNode(graph.GenerateNodeName(last_add.Name() + "/ReciprocalMulLayerNormFusion/"),
                                        "LayerNormalization",
                                        "fused reciprocal-mul LayerNorm subgraph",
                                        layer_norm_input_defs,
                                        {}, last_add, nullptr, kOnnxDomain);

  const NodeArg* epsilon_arg = GetOtherInput(add_epsilon, reduce_mean2_node.MutableOutputDefs()[0]);
  const ONNX_NAMESPACE::TensorProto* tensor_proto =
      epsilon_arg ? graph_utils::GetConstantInitializer(graph, epsilon_arg->Name()) : nullptr;
  if (tensor_proto != nullptr && tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    Initializer initializer{graph, *tensor_proto, graph.ModelPath()};
    layer_norm_node.AddAttribute("epsilon", initializer.data<float>()[0]);
  } else {
    layer_norm_node.AddAttribute("epsilon", DEFAULT_LAYERNORM_EPSILON);
  }

  layer_norm_node.AddAttribute("axis", static_cast<int64_t>(axes_values[0]));
  if (x_input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE ||
      scale->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
    layer_norm_node.AddAttribute("stash_type", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE));
  }

  layer_norm_node.SetExecutionProviderType(reduce_mean_node.GetExecutionProviderType());

  InlinedVector<std::reference_wrapper<Node>> nodes_to_remove{reduce_mean_node,
                                                              sub_node,
                                                              square_mul,
                                                              reduce_mean2_node,
                                                              add_epsilon,
                                                              sqrt_node,
                                                              reciprocal_node,
                                                              scale_mul,
                                                              mean_mul,
                                                              x_mul,
                                                              bias_sub,
                                                              last_add};
  const std::string reduce_mean_name = reduce_mean_node.Name();
  const std::string last_add_name = last_add.Name();
  const std::string layer_norm_name = layer_norm_node.Name();

  graph_utils::FinalizeNodeFusion(graph, nodes_to_remove, layer_norm_node);

  LOGS(logger, VERBOSE) << "ReciprocalMulLayerNormFusion fused pattern. ReduceMean='"
                        << reduce_mean_name << "', final Add='" << last_add_name
                        << "', new LayerNormalization='" << layer_norm_name << "'";

#ifdef ENABLE_TRAINING_CORE
  layer_norm_node.MutableOutputDefs().push_back(&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("saved_mean"), nullptr));
  layer_norm_node.MutableOutputDefs().push_back(&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("saved_inv_std_var"), nullptr));
#endif

  fused = true;
  modified = fused;
#undef RETURN_RECIPROCAL_MUL_LN_SKIP
  return Status::OK();
}
#endif  // defined(__aarch64__)

/**
Layer Normalization will fuse LayerNormalization into one node :
+---------------------+
|                     |
|                     v
X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
                      |                                               ^
                      |                                               |
                      +-----------------------------------------------+
It also handles cases of duplicated sub nodes exported from older version of PyTorch :
+---------------------+
|                     v
|          +-------> Sub ---------------------------------------------+
|          |                                                          |
|          |                                                          v
X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
|                     ^
|                     |
+---------------------+

In recent pytorch, Cast nodes may be inserted before Pow to ensure that both inputs 'base' and 'power' are the same type
due to restriction in older opsets. Therefore, Layer Normalization will also handle the case below :
+---------------------+
|                     |
|                     v
X --> ReduceMean --> Sub --> Cast --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
                              |                                                ^
                              |                                                |
                              +------------------------------------------------+
+---------------------+       Cast
|                     |        |
|                     v        v
X --> ReduceMean --> Sub -->  Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
                      |                                                ^
                      |                                                |
                      +------------------------------------------------+

When using Apex O2, a Cast node may be inserted between Div and Mul, Layer Normalization will also handle the case below:
+---------------------+
|                     |
|                     v
X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Cast --> Mul --> Add
                      |                                               ^
                      |                                               |
                      +-----------------------------------------------+

OR

         +---------------------+
         |                     |
         |                     v
X --> Cast --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Cast --> Mul --> Add
                               |                                               ^
                               |                                               |
                               +-----------------------------------------------+

Logically since LayerNormalization supports input and scale/bias in different data types, and during the kernel execution,
data are casted to float/double to calculate for precision, so if there is any Cast Ops in the sub-graph, we can remove it.
Such Cast Op can be the input of the sub-graph, or an Cast Op between the Div and Mul nodes.
*/
Status LayerNormFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  const auto& version_map = graph.DomainToVersionMap();
  const auto& onnx_version = version_map.find(kOnnxDomain);
  // LayerNorm is an official ONNX operator as of opset 17, so we can fuse in level 1 if it is available
  const bool onnx_layernorm_available = (onnx_version != version_map.end() && onnx_version->second >= 17);
  const bool fuse_in_level_1 = onnx_layernorm_available || allow_contrib_op_in_level_1_;

  if ((optimization_level_ == TransformerLevel::Level1 && !fuse_in_level_1) ||
      // The following check assumes that there is a LayerNormFusion instance registered in Level1 that may have
      // already done this fusion, in which case we don't need to do it again.
      (optimization_level_ == TransformerLevel::Level2 && fuse_in_level_1)) {
    return Status::OK();
  }

  const auto compatible_providers = GetCompatibleExecutionProviders();

  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  InlinedVector<std::reference_wrapper<Node>> nodes_to_remove;
#if defined(__aarch64__)
  const bool enable_reciprocal_mul_layer_norm_fusion = IsReciprocalMulLayerNormFusionEnabled();
#endif
  for (auto node_index : node_topology_list) {
    nodes_to_remove.clear();
    auto* p_reduce_mean = graph.GetNode(node_index);
    if (p_reduce_mean == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& reduce_mean_node = *p_reduce_mean;
    ORT_RETURN_IF_ERROR(Recurse(reduce_mean_node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(reduce_mean_node, "ReduceMean", {1, 11, 13, 18}) ||
        !graph_utils::IsSupportedProvider(reduce_mean_node, GetCompatibleExecutionProviders()) ||
        (reduce_mean_node.GetOutputEdgesCount() != 1 && reduce_mean_node.GetOutputEdgesCount() != 2) ||
        graph.NodeProducesGraphOutput(reduce_mean_node) ||
        !IsSupportedDataType(reduce_mean_node, 1)) {
      continue;
    }
    nodes_to_remove.push_back(reduce_mean_node);

#if defined(__aarch64__)
    if (enable_reciprocal_mul_layer_norm_fusion) {
      for (auto iter = reduce_mean_node.OutputNodesBegin(); iter != reduce_mean_node.OutputNodesEnd(); ++iter) {
        if ((*iter).OpType() != "Sub") {
          continue;
        }

        Node& candidate_sub_node = *graph.GetNode(iter->Index());
        if (!graph_utils::IsSupportedOptypeVersionAndDomain(candidate_sub_node, "Sub", {7, 13, 14}) ||
            candidate_sub_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
            !IsSupportedDataType(candidate_sub_node)) {
          continue;
        }

        bool reciprocal_mul_layer_norm_fused = false;
        ORT_RETURN_IF_ERROR(TryFuseReciprocalMulLayerNorm(graph, reduce_mean_node, candidate_sub_node, logger,
                                                          reciprocal_mul_layer_norm_fused));
        if (reciprocal_mul_layer_norm_fused) {
          modified = true;
          break;
        }
      }

      if (modified && graph.GetNode(node_index) == nullptr) {
        continue;
      }
    }
#endif

    // Loop through the children of current "ReduceMean" node. See if they match ["Sub"] or ["Sub", "Sub"]
    int subCnt = 0;
    const Node* p_sub_node = nullptr;
    const Node* p_sub_node_dup = nullptr;
    for (auto iter = reduce_mean_node.OutputNodesBegin(); iter != reduce_mean_node.OutputNodesEnd(); ++iter) {
      if ((*iter).OpType().compare("Sub") == 0) {
        if (subCnt == 0) {
          p_sub_node = &(*iter);
        } else {
          p_sub_node_dup = &(*iter);
        }
        subCnt++;
      } else {
        // doesn't match layer norm pattern. break.
        subCnt = -1;
        break;
      }
    }

    if (subCnt != 1 && subCnt != 2) {
      continue;
    }

    Node& sub_node = *graph.GetNode(p_sub_node->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(sub_node, "Sub", {7, 13, 14}) ||
        sub_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !IsSupportedDataType(sub_node)) {
      continue;
    }
    nodes_to_remove.push_back(sub_node);

    // Apex O2 pattern specific match starts...
    // Logically since we support input and scale/bias in different data types, those Cast Ops in sub-graph
    // can be removed. This is one possible place a Cast Op can exist, that is the input of the sub-graph.
    // Make sure it consumes by the sub-graph only.
    const NodeArg* p_reduce_mean_input = reduce_mean_node.MutableInputDefs()[0];
    const NodeArg* p_sub_input = nullptr;
    for (NodeArg* node_arg : sub_node.MutableInputDefs()) {
      if (node_arg != reduce_mean_node.MutableOutputDefs()[0]) {
        p_sub_input = node_arg;
        break;
      }
    }

    if (!p_reduce_mean_input || !p_sub_input || p_reduce_mean_input != p_sub_input) {
      continue;
    }

    if (p_sub_node_dup) {
      const NodeArg* p_sub_dup_input = nullptr;
      for (NodeArg* node_arg : graph.GetNode(p_sub_node_dup->Index())->MutableInputDefs()) {
        if (node_arg != reduce_mean_node.MutableOutputDefs()[0]) {
          p_sub_dup_input = node_arg;
          break;
        }
      }
      if (!p_sub_dup_input || p_reduce_mean_input != p_sub_dup_input) {
        continue;
      }
    }

    const Node* p_reduce_mean_input_node = graph_utils::GetInputNode(reduce_mean_node, 0);
    bool has_leading_cast = false;
    if (p_reduce_mean_input_node) {
      Node& reduce_mean_input_node = *graph.GetNode(p_reduce_mean_input_node->Index());
      // If input to the 1st ReduceMean is a Cast, and the Cast has same consumer count as subCnt + 1
      if (graph_utils::IsSupportedOptypeVersionAndDomain(reduce_mean_input_node, "Cast", {9, 13, 19, 21, 23, 24, 25}) &&
          reduce_mean_input_node.GetExecutionProviderType() == reduce_mean_node.GetExecutionProviderType() &&
          optimizer_utils::CheckOutputEdges(graph, reduce_mean_input_node, static_cast<size_t>(subCnt) + 1)) {
        nodes_to_remove.insert(nodes_to_remove.begin(), reduce_mean_input_node);
        has_leading_cast = true;
      }
    }
    // Apex O2 pattern specific match ends...

    // Find the "Div" node after "Sub". It's possible that there is "Cast" node after "Sub" node.
    const Node* p_cast1 = nullptr;
    if (!p_sub_node_dup && sub_node.GetOutputEdgesCount() == 1) {
      Node& cast_node = *graph.GetNode(sub_node.OutputNodesBegin()->Index());
      if (graph_utils::IsSupportedOptypeVersionAndDomain(cast_node, "Cast", {9, 13, 19, 21, 23, 24, 25}) &&
          cast_node.GetExecutionProviderType() == reduce_mean_node.GetExecutionProviderType() &&
          optimizer_utils::CheckOutputEdges(graph, cast_node, 2u) && IsSupportedDataType(cast_node)) {
        p_cast1 = &cast_node;
        nodes_to_remove.push_back(cast_node);
      }
    }

    if (!optimizer_utils::CheckOutputEdges(graph, sub_node, subCnt == 1 && !p_cast1 ? 2u : 1u)) {
      continue;
    }

    const Node* p_div = nullptr;
    p_div = graph_utils::FirstChildByType(p_cast1 ? *p_cast1 : sub_node, "Div");

    // Find the sub_dup node if exist
    if (p_sub_node_dup != nullptr) {
      Node& sub_node_dup = *graph.GetNode(p_sub_node_dup->Index());
      if (!graph_utils::IsSupportedOptypeVersionAndDomain(sub_node_dup, "Sub", {7, 13, 14}) ||
          sub_node_dup.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
          !optimizer_utils::CheckOutputEdges(graph, sub_node, 1) ||
          !IsSupportedDataType(sub_node_dup)) {
        continue;
      }
      nodes_to_remove.push_back(sub_node_dup);
      // Find Div node after the duplicated sub node if it's not found after the first sub node.
      if (p_div == nullptr) {
        p_div = graph_utils::FirstChildByType(sub_node_dup, "Div");
      }
    }

    if (p_div == nullptr) {
      continue;
    }
    Node& div_node = *graph.GetNode(p_div->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(div_node, "Div", {7, 13, 14}) ||
        div_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, div_node, 1) ||
        !IsSupportedDataType(div_node)) {
      continue;
    }
    nodes_to_remove.push_back(div_node);

    // Traceback the div node to find sqrt --> div
    const Node* p_sqrt = graph_utils::FirstParentByType(div_node, "Sqrt");
    if (p_sqrt == nullptr) {
      continue;
    }
    Node& sqrt_node = *graph.GetNode(p_sqrt->Index());

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(sqrt_node, "Sqrt", {6, 13}) ||
        sqrt_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, sqrt_node, 1) ||
        !IsSupportedDataType(sqrt_node) ||
        sqrt_node.GetInputEdgesCount() == 0) {
      continue;
    }
    nodes_to_remove.push_back(sqrt_node);

    // Traceback the sqrt node to find add --> sqrt
    Node& add2_node = *graph.GetNode(sqrt_node.InputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(add2_node, "Add", {7, 13, 14}) ||
        add2_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, add2_node, 1) ||
        !IsSupportedDataType(add2_node)) {
      continue;
    }
    nodes_to_remove.push_back(add2_node);
    // Traceback the add node to find reduceMean --> add
    const Node* p_reduce_mean2 = nullptr;

    p_reduce_mean2 = graph_utils::FirstParentByType(add2_node, "ReduceMean");
    if (p_reduce_mean2 == nullptr) {
      continue;
    }
    Node& reduce_mean2_node = *graph.GetNode(p_reduce_mean2->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(reduce_mean2_node, "ReduceMean", {1, 11, 13, 18}) ||
        reduce_mean2_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, reduce_mean2_node, 1) ||
        !IsSupportedDataType(reduce_mean2_node, 1) ||
        reduce_mean2_node.GetInputEdgesCount() == 0) {
      continue;
    }
    nodes_to_remove.push_back(reduce_mean2_node);

    // Traceback the reduceMean node to find pow --> reduceMean
    Node& pow_node = *graph.GetNode(reduce_mean2_node.InputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(pow_node, "Pow", {7, 12, 13, 15}) ||
        pow_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, pow_node, 1) ||
        !IsSupportedDataType(pow_node)) {
      continue;
    }
    nodes_to_remove.push_back(pow_node);

    // check if Cast node exists: either between sub and pow, or as second input to pow
    const Node* p_cast2 = graph_utils::FirstParentByType(pow_node, "Cast");
    if (p_cast2 != nullptr && p_cast2 != p_cast1) {
      Node& cast_node = *graph.GetNode(p_cast2->Index());
      if (!graph_utils::IsSupportedOptypeVersionAndDomain(cast_node, "Cast", {9, 13, 19, 21, 23, 24, 25}) ||
          cast_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
          !optimizer_utils::CheckOutputEdges(graph, cast_node, 1)) {
        continue;
      }
      nodes_to_remove.push_back(cast_node);
    } else if (!p_cast2) {
      const Node* p_sub2_node = graph_utils::FirstParentByType(pow_node, "Sub");
      if (!p_sub2_node || (p_sub2_node != p_sub_node && p_sub2_node != p_sub_node_dup)) {
        continue;
      }
    }

    // Apex O2 pattern specific match starts...
    // Logically since we support input and scale/bias in different data types, those Cast Ops in sub-graph
    // can be removed. This is one possible place a Cast Op can exist, that is between Div and Mul nodes.
    // div --> mul or div --> cast --> mul
    Node* next_node = graph.GetNode(div_node.OutputNodesBegin()->Index());
    if (graph_utils::IsSupportedOptypeVersionAndDomain(*next_node, "Cast", {9, 13, 19, 21, 23, 24, 25}) &&
        optimizer_utils::CheckOutputEdges(graph, *next_node, 1)) {
      nodes_to_remove.push_back(*next_node);
      next_node = graph.GetNode(next_node->OutputNodesBegin()->Index());
    }
    // Apex O2 pattern specific match ends...

    Node& mul_node = *next_node;
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul_node, "Mul", {7, 13, 14}) ||
        mul_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, mul_node, 1) ||
        !IsSupportedDataType(mul_node)) {
      continue;
    }
    nodes_to_remove.push_back(mul_node);

    // mul --> add
    // Need not check output edges of last node since they will be moved to fused node.
    Node& last_add_node = *graph.GetNode(mul_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(last_add_node, "Add", {7, 13, 14}) ||
        last_add_node.GetExecutionProviderType() != reduce_mean_node.GetExecutionProviderType() ||
        !IsSupportedDataType(last_add_node)) {
      continue;
    }
    nodes_to_remove.push_back(last_add_node);

    // get axes attributes

    auto axes_values = GetAxesFromReduceMeanNode(reduce_mean_node, graph);
    auto axes2_values = GetAxesFromReduceMeanNode(reduce_mean2_node, graph);

    // empty axes means reduce over all axes, which is not supported on layer-norm
    if (axes_values.empty() || axes2_values.empty()) {
      continue;
    }

    auto input_shape = reduce_mean_node.MutableInputDefs()[0]->Shape();
    auto rank = input_shape ? input_shape->dim().size() : -1;
    if (!CheckAxesOnReduceMean(axes_values, rank) ||
        !CheckAxesOnReduceMean(axes2_values, rank) ||
        axes_values != axes2_values) {
      continue;
    }

#ifdef ENABLE_TRAINING_CORE
#else
    // scale as 1D
    if (axes_values.size() != 1) {
      continue;
    }
#endif

    // Get the inputs for the new LayerNormalization node.
    // scale and bias could be multi-dims; we only support it for training at the moment
    // because SkipLayerNorm kernel, for example, has dependency on single dim size
    NodeArg* scale = nullptr;
    NodeArg* bias = nullptr;
    for (size_t i = 0; i < mul_node.MutableInputDefs().size(); i++) {
      if (mul_node.MutableInputDefs()[i]->Shape() == nullptr) {
        continue;
      }
      if (mul_node.MutableInputDefs()[i]->Shape()->dim_size() == static_cast<int>(axes_values.size())) {
        scale = mul_node.MutableInputDefs()[i];
      }
    }

    for (size_t i = 0; i < last_add_node.MutableInputDefs().size(); i++) {
      if (last_add_node.MutableInputDefs()[i]->Shape() == nullptr) {
        continue;
      }
      if (last_add_node.MutableInputDefs()[i]->Shape()->dim_size() == static_cast<int>(axes_values.size())) {
        bias = last_add_node.MutableInputDefs()[i];
      }
    }
    if (scale == nullptr || bias == nullptr) {
      continue;
    }

    // Scale and bias must have the same shape.
    bool same_dim = true;
    for (int i = 0; i < scale->Shape()->dim_size(); i++) {
      if (scale->Shape()->dim(i).dim_value() != bias->Shape()->dim(i).dim_value()) {
        same_dim = false;
        break;
      }
    }
    if (!same_dim)
      continue;

    NodeArg* x_input = has_leading_cast ? graph.GetNode(p_reduce_mean_input_node->Index())->MutableInputDefs()[0]
                                        : reduce_mean_node.MutableInputDefs()[0];

    // CPU doesn't support fp16
    if (reduce_mean_node.GetExecutionProviderType() == kCpuExecutionProvider &&
        x_input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      continue;
    }

    InlinedVector<NodeArg*> layer_norm_input_defs{x_input, scale, bias};
    Node& layer_norm_node = graph.AddNode(graph.GenerateNodeName(mul_node.Name() + "/LayerNormFusion/"),
                                          "LayerNormalization",
                                          "fused LayerNorm subgraphs ",
                                          layer_norm_input_defs,
                                          {}, mul_node, nullptr, kOnnxDomain);

    // Get constant "epsilon" from "Add2" node if available. Else, default value will be used.
    const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, add2_node.MutableInputDefs()[1]->Name());
    if (tensor_proto != nullptr &&
        tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      Initializer initializer{graph, *tensor_proto, graph.ModelPath()};
      layer_norm_node.AddAttribute("epsilon", initializer.data<float>()[0]);
    } else {
      layer_norm_node.AddAttribute("epsilon", DEFAULT_LAYERNORM_EPSILON);
    }

    // The axis definition of layer_norm is ranging from axis to the last dim
    layer_norm_node.AddAttribute("axis", static_cast<int64_t>(axes_values[0]));

    // Set stash_type to double if any input is double, default value if float.
    if (x_input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE ||
        scale->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
      layer_norm_node.AddAttribute("stash_type", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE));
    }

    // Assign provider to this new node. Provider should be same as the provider for old node.
    layer_norm_node.SetExecutionProviderType(reduce_mean_node.GetExecutionProviderType());

    // move input edges to add (first in list) across to the layer_norm_node.
    // move output definitions and output edges from mul_node (last in list) to layer_norm_node.
    // remove all the other nodes.
    graph_utils::FinalizeNodeFusion(graph, nodes_to_remove, layer_norm_node);

#ifdef ENABLE_TRAINING_CORE
    // add two extra output defs, so we have 3 output defs that match what gradient builder expected
    layer_norm_node.MutableOutputDefs().push_back(&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("saved_mean"), nullptr));
    layer_norm_node.MutableOutputDefs().push_back(&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("saved_inv_std_var"), nullptr));
#endif

    modified = true;
  }
  return Status::OK();
}

/**
Layer Normalization will fuse LayerNormalization into one node :

X --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul
|                                              ^
|                                              |
+----------------------------------------------+
Additional FP16 patterns supported:

X --> Cast1 --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Cast2 --> Mul
        |                                               ^                  ^
        |                                               |                  |
        +-----------------------------------------------+                Scale

Since SimplifiedLayerNormalization supports input and scale in different data types,
and during the kernel execution, data are casted to float/double to calculate for precision,
so we can fuse it to a single SimplifiedLayerNormalization, the output type is same as Scale.
This results in the graph:

X ------> SimplifiedLayerNormalization
              ^
Scale --------|
*/
Status SimplifiedLayerNormFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                            const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  InlinedVector<std::reference_wrapper<Node>> nodes_to_remove;
  for (auto node_index : node_topology_list) {
    nodes_to_remove.clear();
    auto* p_pow = graph.GetNode(node_index);
    if (p_pow == nullptr) continue;  // we removed the node as part of an earlier fusion

    Node& pow_node = *p_pow;
    ORT_RETURN_IF_ERROR(Recurse(pow_node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(pow_node, "Pow", {7, 12, 13, 15}) ||
        !graph_utils::IsSupportedProvider(pow_node, GetCompatibleExecutionProviders()) ||
        !optimizer_utils::CheckOutputEdges(graph, pow_node, 1) || graph.NodeProducesGraphOutput(pow_node) ||
        !IsSupportedDataType(pow_node)) {
      continue;
    }
    nodes_to_remove.push_back(pow_node);

    const Node* p_reduce_mean = nullptr;

    p_reduce_mean = graph_utils::FirstChildByType(pow_node, "ReduceMean");
    if (p_reduce_mean == nullptr) {
      continue;
    }
    Node& reduce_mean_node = *graph.GetNode(p_reduce_mean->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(reduce_mean_node, "ReduceMean", {1, 11, 13, 18}) ||
        reduce_mean_node.GetExecutionProviderType() != pow_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, reduce_mean_node, 1) || !IsSupportedDataType(reduce_mean_node, 1) ||
        reduce_mean_node.GetInputEdgesCount() == 0) {
      continue;
    }
    nodes_to_remove.push_back(reduce_mean_node);

    const Node* p_add = graph_utils::FirstChildByType(reduce_mean_node, "Add");
    if (p_add == nullptr) {
      continue;
    }
    Node& add_node = *graph.GetNode(p_add->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(add_node, "Add", {7, 13, 14}) ||
        add_node.GetExecutionProviderType() != pow_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, add_node, 1) || !IsSupportedDataType(add_node)) {
      continue;
    }
    nodes_to_remove.push_back(add_node);

    const Node* p_sqrt = graph_utils::FirstChildByType(add_node, "Sqrt");
    if (p_sqrt == nullptr) {
      continue;
    }
    Node& sqrt_node = *graph.GetNode(p_sqrt->Index());

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(sqrt_node, "Sqrt", {6, 13}) ||
        sqrt_node.GetExecutionProviderType() != pow_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, sqrt_node, 1) || !IsSupportedDataType(sqrt_node) ||
        sqrt_node.GetInputEdgesCount() == 0) {
      continue;
    }
    nodes_to_remove.push_back(sqrt_node);

    const Node* p_div = graph_utils::FirstChildByType(sqrt_node, "Div");
    if (p_div == nullptr) {
      continue;
    }
    Node& div_node = *graph.GetNode(p_div->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(div_node, "Div", {7, 13, 14}) ||
        div_node.GetExecutionProviderType() != pow_node.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, div_node, 1) || !IsSupportedDataType(div_node)) {
      continue;
    }
    nodes_to_remove.push_back(div_node);

    // Check Div and Pow has same input, and if this input is a Cast, we can also remove it.
    const NodeArg* p_div_input = div_node.MutableInputDefs()[0];
    const NodeArg* p_pow_input = pow_node.MutableInputDefs()[0];
    if (!p_pow_input || !p_div_input || p_div_input != p_pow_input) {
      continue;
    }

    // There are only 4 possible cases (x=Pow->ReduceMean->Add->Sqrt->Div and cannot on fp16 type, y=Mul):
    // 1. Cast(to:float)->x->Cast(to:fp16)->y : SimplifiedLayerNorm(T:fp16,V:fp16)
    // 2. Cast(to:float)->x->y : SimplifiedLayerNorm(T:fp16,V:float)
    // 3. x->Cast(to:fp16)->y : SimplifiedLayerNorm(T:float,V:fp16)
    // 4. x->y : SimplifiedLayerNorm(T:float,V:float)
    // They all work for GPU EP.
    // For CPU EP, we have only SimplifiedlayerNorm(T:float,V:float) implementation, so only #4 works. We made an
    // exception here, since pre-training optimization happens without device assignment. skip_device_check_ is the
    // flag to disable device check intent only for pre-training optimization.
    // For #1 and #2, if we treat the entry Cast as a normal node, meaning has_leading_cast is false, then for #2,
    // we can still fuse it to "Cast(to:float)->SimplifiedlayerNorm(T:float,V:float)" (same as applying #4 to the x->y
    // after Cast), so the condition for CPU EP to fuse or not is always setting has_leading_cast to false and checking
    // if there is a Cast between x and y. Having Cast between means cannot fuse.
    const Node* p_pow_input_node = graph_utils::GetInputNode(pow_node, 0);
    bool has_leading_cast = false;
    bool is_gpu_ep = pow_node.GetExecutionProviderType() == kCudaExecutionProvider || skip_device_check_;
    if (is_gpu_ep && p_pow_input_node) {
      Node& pow_input_node = *graph.GetNode(p_pow_input_node->Index());
      // If input to Pow is a Cast, and the Cast has 2 consumers only (Pow, Div)
      if (graph_utils::IsSupportedOptypeVersionAndDomain(pow_input_node, "Cast", {9, 13, 19, 21, 23, 24, 25}) &&
          pow_input_node.GetExecutionProviderType() == pow_node.GetExecutionProviderType() &&
          optimizer_utils::CheckOutputEdges(graph, pow_input_node, 2)) {
        nodes_to_remove.insert(nodes_to_remove.begin(), pow_input_node);
        has_leading_cast = true;
      }
    }

    // div --> mul or div --> cast --> mul
    Node* next_node = graph.GetNode(div_node.OutputNodesBegin()->Index());
    if (graph_utils::IsSupportedOptypeVersionAndDomain(*next_node, "Cast", {9, 13, 19, 21, 23, 24, 25}) &&
        optimizer_utils::CheckOutputEdges(graph, *next_node, 1)) {
      if (!is_gpu_ep) continue;
      nodes_to_remove.push_back(*next_node);
      next_node = graph.GetNode(next_node->OutputNodesBegin()->Index());
    }

    Node& mul_node = *next_node;
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul_node, "Mul", {7, 13, 14}) ||
        mul_node.GetExecutionProviderType() != pow_node.GetExecutionProviderType() || !IsSupportedDataType(mul_node)) {
      continue;
    }
    nodes_to_remove.push_back(mul_node);

    // get axes attributes
    std::vector<int64_t> axes_values = GetAxesFromReduceMeanNode(reduce_mean_node, graph);

    if (axes_values.empty()) {
      continue;
    }

    auto rmean_input_shape = reduce_mean_node.MutableInputDefs()[0]->Shape();
    auto rank = rmean_input_shape ? rmean_input_shape->dim().size() : -1;
    if (!CheckAxesOnReduceMean(axes_values, rank)) {
      continue;
    }

#ifdef ENABLE_TRAINING_CORE
#else
    // scale as 1D
    if (axes_values.size() != 1) {
      continue;
    }
#endif

    // Get the inputs for the new LayerNormalization node.
    // scale and bias could be multi-dims; we only support it for training at the moment
    // because SkipLayerNorm kernel, for example, has dependency on single dim size
    NodeArg* scale = nullptr;
    for (size_t i = 0; i < mul_node.MutableInputDefs().size(); i++) {
      if (mul_node.MutableInputDefs()[i]->Shape() == nullptr) {
        continue;
      }
#ifdef ENABLE_TRAINING_CORE
      if (axes_values.empty() ||
          mul_node.MutableInputDefs()[i]->Shape()->dim_size() == static_cast<int>(axes_values.size())) {
        scale = mul_node.MutableInputDefs()[i];
      }
#else
      // Scale must be 1d.
      if (mul_node.MutableInputDefs()[i]->Shape()->dim_size() == 1) {
        scale = mul_node.MutableInputDefs()[i];
      }
#endif
    }

    if (scale == nullptr) {
      continue;
    }

    NodeArg* x_input = has_leading_cast ? graph.GetNode(p_pow_input_node->Index())->MutableInputDefs()[0]
                                        : pow_node.MutableInputDefs()[0];

    // CPU doesn't support fp16
    if (reduce_mean_node.GetExecutionProviderType() == kCpuExecutionProvider &&
        x_input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      continue;
    }

    InlinedVector<NodeArg*> layer_norm_input_defs{x_input, scale};
    Node& layer_norm_node =
        graph.AddNode(graph.GenerateNodeName(mul_node.Name() + "/SimplifiedLayerNormFusion/"), "SimplifiedLayerNormalization",
                      "fused LayerNorm subgraphs ", layer_norm_input_defs, {}, mul_node, nullptr, kOnnxDomain);

    // Get constant "epsilon" from "Add" node if available. Else, default value will be used.
    const ONNX_NAMESPACE::TensorProto* tensor_proto =
        graph_utils::GetConstantInitializer(graph, add_node.MutableInputDefs()[1]->Name());
    if (tensor_proto != nullptr && tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      Initializer initializer{graph, *tensor_proto, graph.ModelPath()};
      layer_norm_node.AddAttribute("epsilon", initializer.data<float>()[0]);
    } else {
      layer_norm_node.AddAttribute("epsilon", DEFAULT_LAYERNORM_EPSILON);
    }

    // Set stash_type to double if any input is double, default value if float.
    if (x_input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE ||
        scale->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
      layer_norm_node.AddAttribute("stash_type", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE));
    }

    layer_norm_node.AddAttribute("axis", static_cast<int64_t>(axes_values[0]));

    // Assign provider to this new node. Provider should be same as the provider for old node.
    layer_norm_node.SetExecutionProviderType(reduce_mean_node.GetExecutionProviderType());

    // move input edges to add (first in list) across to the layer_norm_node.
    // move output definitions and output edges from mul_node (last in list) to layer_norm_node.
    // remove all the other nodes.
    graph_utils::FinalizeNodeFusion(graph, nodes_to_remove, layer_norm_node);

#ifdef ENABLE_TRAINING_CORE
    // add one extra output def, so we have 2 output defs that match what gradient builder expected
    layer_norm_node.MutableOutputDefs().push_back(
        &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("saved_inv_std_var"), nullptr));
#endif

    modified = true;
  }
  return Status::OK();
}

}  // namespace onnxruntime
