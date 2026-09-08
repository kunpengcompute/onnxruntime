// Copyright (c) Huawei Technologies Co., Ltd. 2026.
// Licensed under the MIT License.

#include "core/optimizer/kdnn_attention_fusion.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/common/inlined_containers.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {

namespace {

bool FuseEnabled() {
  const char* e = std::getenv("ORT_KDNN_FUSE_ATTENTION");
  return e != nullptr && e[0] != '\0' && e[0] != '0';
}

bool IsConst(const Graph& graph, const std::string& name) {
  return graph.GetConstantInitializer(name, /*check_outer_scope=*/true) != nullptr;
}

// Read a float scalar constant initializer; false if it isn't one.
bool ScalarFloat(const Graph& graph, const std::string& name, float& out) {
  const auto* tp = graph.GetConstantInitializer(name, true);
  if (tp == nullptr || tp->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) return false;
  Initializer init{graph, *tp, graph.ModelPath()};
  if (init.size() != 1) return false;
  out = init.data<float>()[0];
  return true;
}

// Read an int64 1-D constant initializer's values; false if not int64.
bool Int64Vec(const Graph& graph, const std::string& name, std::vector<int64_t>& out) {
  const auto* tp = graph.GetConstantInitializer(name, true);
  if (tp == nullptr || tp->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64) return false;
  Initializer init{graph, *tp, graph.ModelPath()};
  auto span = init.DataAsSpan<int64_t>();
  out.assign(span.begin(), span.end());
  return true;
}

bool IsOnnxOp(const Node* node, const char* op_type) {
  return node != nullptr && node->Domain() == kOnnxDomain && node->OpType() == op_type;
}

bool IsFloatTensor(const NodeArg& node_arg) {
  const auto* type = node_arg.TypeAsProto();
  return type != nullptr && type->has_tensor_type() &&
         type->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
}

bool HasLastAxisSoftmax(const Node& node) {
  const auto& attributes = node.GetAttributes();
  const auto axis_it = attributes.find("axis");
  const int64_t axis = axis_it == attributes.end()
                           ? (node.SinceVersion() >= 13 ? -1 : 1)
                           : axis_it->second.i();
  return axis == -1 || axis == 3;
}

template <size_t N>
bool HasTransposePermutation(const Node* node, const std::array<int64_t, N>& expected) {
  if (!IsOnnxOp(node, "Transpose")) return false;
  const auto& attributes = node->GetAttributes();
  const auto perm_it = attributes.find("perm");
  if (perm_it == attributes.end() || perm_it->second.ints_size() != static_cast<int>(N)) {
    return false;
  }
  for (size_t i = 0; i < N; ++i) {
    if (perm_it->second.ints(static_cast<int>(i)) != expected[i]) return false;
  }
  return true;
}

bool CanRemoveSingleConsumerNode(const Graph& graph, const Node* node) {
  return node != nullptr && !graph.NodeProducesGraphOutput(*node) &&
         optimizer_utils::CheckOutputEdges(graph, *node, 1);
}

// Walk a Q/K/V branch (an input of the QK or AV matmul) back to the dense
// projection's BiasAdd and recover the head split. The tf2onnx attention form is:
//
//   BiasAdd -> Reshape(head-split) -> [Transpose] -> matmul
//
// Two head-split layouts occur:
//   * multi-head form  [-1, S, H, d]
//   * single-head form [1, B, S, d] : single head; the BiasAdd is flattened
//     [1, B*S, d], so it must be RESHAPED to [B, S, d] before feeding the op.
//
// We pass through one optional Transpose and preserve the head-split Reshape as
// `feed`. The fused op consumes the multi-head [B,S,H,d] layout directly; the
// single-head form requires canonicalization to [B, S, d].
struct Branch {
  std::string feed;  // 4-D head-split Reshape output
  int64_t num_heads = 0;
  int64_t head_dim = 0;
  int64_t seq = -1;                      // static S from the head-split
  bool needs_reshape = false;            // single-head input/output/mask layout needs canonicalization
  const Node* skip_transpose = nullptr;  // a head-major/QK^T Transpose we passed through;
                                         // its sole consumer is the matmul we subsume, so it
                                         // must be dropped or it executes (dead) at runtime.
  bool ok = false;
};

Branch TraceProjection(const Graph& graph, const Node* matmul, int input_slot) {
  Branch b;
  const Node* cur = graph_utils::GetInputNode(*matmul, input_slot);
  // optional transpose (head-major reorder for V, or the [..,3,2] QK^T transpose on K)
  // before the reshape. We pass through it but MUST record it: once the matmul that
  // consumes it is fused away the Transpose is orphaned, and ORT still schedules it
  // (a 4D physical reorder that can cost ~5.7ms/call) unless we delete it here.
  if (IsOnnxOp(cur, "Transpose")) {
    b.skip_transpose = cur;
    cur = graph_utils::GetInputNode(*cur, 0);
  }
  if (!IsOnnxOp(cur, "Reshape") || cur->InputDefs().size() < 2) return b;
  const Node* reshape = cur;

  std::vector<int64_t> tgt;
  if (!Int64Vec(graph, reshape->InputDefs()[1]->Name(), tgt)) return b;
  if (tgt.size() != 4) return b;  // unrecognised head split

  if (tgt[0] == 1 && tgt[1] == -1 && tgt[2] > 0 && tgt[3] > 0) {
    // Single-head form [1, B, S, d]: feed the head-split output and let
    // the caller reshape [1,B,S,d] -> [B, S, d].
    b.num_heads = 1;
    b.head_dim = tgt[3];
    b.seq = tgt[2];
    b.feed = reshape->OutputDefs()[0]->Name();
    b.needs_reshape = true;
    b.ok = true;
    return b;
  }
  if (tgt[0] != 1 && tgt[1] > 0 && tgt[2] > 0 && tgt[3] > 0) {
    // Keep the proven [B,S,H,d] head-split output for the fused op.
    b.num_heads = tgt[2];
    b.head_dim = tgt[3];
    b.seq = tgt[1];
    const Node* proj = graph_utils::GetInputNode(*reshape, 0);
    if (!IsOnnxOp(proj, "Add")) return b;
    b.feed = reshape->OutputDefs()[0]->Name();
    b.needs_reshape = false;
    b.ok = true;
    return b;
  }
  return b;
}

// Make an INT64 1-D shape initializer and return its NodeArg.
NodeArg& MakeShapeInitializer(Graph& graph, const std::string& base,
                              const std::vector<int64_t>& dims) {
  ONNX_NAMESPACE::TensorProto t;
  t.set_name(graph.GenerateNodeArgName(base));
  t.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  t.add_dims(static_cast<int64_t>(dims.size()));
  for (int64_t d : dims) t.add_int64_data(d);
  return graph_utils::AddInitializer(graph, t);
}

// Insert Reshape(src, dims) -> new arg; returns the reshaped output NodeArg.
NodeArg& InsertReshape(Graph& graph, NodeArg& src, const std::vector<int64_t>& dims,
                       const std::string& ep, const std::string& base) {
  NodeArg& shape_arg = MakeShapeInitializer(graph, base + "_shape", dims);
  NodeArg& out = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(base), nullptr);
  std::array<NodeArg*, 2> in{&src, &shape_arg};
  std::array<NodeArg*, 1> outs{&out};
  Node& r = graph.AddNode(graph.GenerateNodeName(base), "Reshape", "KDNN attn layout canon",
                          in, outs, nullptr, kOnnxDomain);
  r.SetExecutionProviderType(ep);
  return out;
}

}  // namespace

Status KdnnAttentionFusion::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/,
                                      const logging::Logger& logger) const {
  if (!FuseEnabled()) {
    return Status::OK();  // gated off -> original graph (A/B baseline)
  }

  const auto compatible_providers = GetCompatibleExecutionProviders();
  GraphViewer graph_viewer(graph);
  const auto& topo = graph_viewer.GetNodesInTopologicalOrder();

  int fused = 0;

  for (auto node_index : topo) {
    Node* p_softmax = graph.GetNode(node_index);
    if (p_softmax == nullptr) continue;  // removed earlier
    if (!IsOnnxOp(p_softmax, "Softmax") || !HasLastAxisSoftmax(*p_softmax)) continue;
    if (!graph_utils::IsSupportedProvider(*p_softmax, compatible_providers)) continue;
    Node& softmax = *p_softmax;
    if (softmax.InputDefs().empty() || softmax.OutputDefs().empty()) continue;

    const std::string ep = softmax.GetExecutionProviderType();
    auto same_ep = [&](const Node& n) { return n.GetExecutionProviderType() == ep; };

    // --- forward: Softmax must feed exactly one MatMul(probs, V). This is the AV
    //     product; its presence (probs . V) is what separates real SDPA from a
    //     plain logit softmax (whose child is Slice/ReduceMean/etc). ---
    if (!optimizer_utils::CheckOutputEdges(graph, softmax, 1)) continue;
    if (graph.NodeProducesGraphOutput(softmax)) continue;
    const Node* av = &(*softmax.OutputNodesBegin());
    if (!IsOnnxOp(av, "MatMul") || !same_ep(*av) || av->InputDefs().size() != 2) continue;
    // SDPA terminates in probabilities . V, never V . probabilities.
    if (av->InputDefs()[0]->Name() != softmax.OutputDefs()[0]->Name()) continue;
    constexpr int v_slot = 1;

    // --- backward: scores = Softmax.input, optionally Add(scaled, maskbias). ---
    const Node* scores = graph_utils::GetInputNode(softmax, 0);
    if (scores == nullptr || !same_ep(*scores)) continue;

    // Resolve a candidate "scaled scores" node to its QK matmul + scale. The
    // scaled scores is Mul(qk,c) | Div(qk,c) | a bare MatMul (scale folded). qk
    // must be a MatMul whose BOTH inputs are computed (non-initializer) -- that is
    // the QK^T product and is what distinguishes real SDPA from a logit softmax.
    auto resolve_scores = [&](const Node* cand, const Node*& qk_out,
                              float& scale_out, bool& has_scale_out) -> bool {
      if (cand == nullptr || !same_ep(*cand)) return false;
      const Node* qk = nullptr;
      float sc = 0.0f;
      bool hs = false;
      if (IsOnnxOp(cand, "MatMul")) {
        qk = cand;
        sc = 1.0f;
        hs = true;
      } else if ((IsOnnxOp(cand, "Mul") || IsOnnxOp(cand, "Div")) &&
                 cand->InputDefs().size() == 2) {
        const std::string i0 = cand->InputDefs()[0]->Name();
        const std::string i1 = cand->InputDefs()[1]->Name();
        const bool c0 = IsConst(graph, i0), c1 = IsConst(graph, i1);
        if (c0 == c1) return false;
        // Division is only equivalent to a scalar scale for MatMul / constant.
        if (cand->OpType() == "Div" && c0) return false;
        float c = 0.0f;
        if (!ScalarFloat(graph, c0 ? i0 : i1, c)) return false;
        if (!std::isfinite(c) || (cand->OpType() == "Div" && c == 0.0f)) return false;
        sc = (cand->OpType() == "Mul") ? c : (1.0f / c);
        hs = true;
        qk = graph_utils::GetInputNode(*cand, c0 ? 1 : 0);
      } else {
        return false;
      }
      if (!IsOnnxOp(qk, "MatMul") || !same_ep(*qk) ||
          qk->InputDefs().size() != 2) return false;
      if (IsConst(graph, qk->InputDefs()[0]->Name()) ||
          IsConst(graph, qk->InputDefs()[1]->Name())) return false;
      qk_out = qk;
      scale_out = sc;
      has_scale_out = hs;
      return true;
    };

    const Node* mask_mul = nullptr;  // Mul(neg, Sub(1, mask...)) additive-bias producer
    const Node* qk = nullptr;
    const Node* scaled = nullptr;  // the scaled-scores producer (Mul/Div/MatMul)
    float scale = 0.0f;
    bool has_scale = false;
    if (IsOnnxOp(scores, "Add") && scores->InputDefs().size() == 2) {
      // Try each Add branch as the scores; the other branch is the mask bias.
      const Node* a0 = graph_utils::GetInputNode(*scores, 0);
      const Node* a1 = graph_utils::GetInputNode(*scores, 1);
      if (resolve_scores(a0, qk, scale, has_scale)) {
        scaled = a0;
        mask_mul = a1;
      } else if (resolve_scores(a1, qk, scale, has_scale)) {
        scaled = a1;
        mask_mul = a0;
      } else
        continue;
      if (mask_mul == nullptr) continue;  // constant/additional mask forms are not supported
    } else if (resolve_scores(scores, qk, scale, has_scale)) {
      scaled = scores;  // no additive mask
    } else {
      continue;
    }

    // --- recover Q, K, V projections + head split. ---
    Branch qB = TraceProjection(graph, qk, 0);
    Branch kB = TraceProjection(graph, qk, 1);
    Branch vB = TraceProjection(graph, av, v_slot);
    if (!qB.ok || !kB.ok || !vB.ok) continue;
    if (qB.num_heads != kB.num_heads || qB.num_heads != vB.num_heads ||
        qB.head_dim != kB.head_dim || qB.head_dim != vB.head_dim) continue;
    // Keep single-head attention on the native ORT path. Without head-level work
    // to amortize packing, workspace, and dispatch costs, the fused kernel is slower.
    if (qB.num_heads < 2) continue;
    // The current rewrite preserves the AV output byte layout only for a
    // single query row. Multi-row attention needs an explicit output transpose.
    if (!qB.needs_reshape && qB.seq != 1) continue;
    if (!HasTransposePermutation(qB.skip_transpose, std::array<int64_t, 4>{0, 2, 1, 3}) ||
        !HasTransposePermutation(kB.skip_transpose, std::array<int64_t, 4>{0, 2, 3, 1}) ||
        !HasTransposePermutation(vB.skip_transpose, std::array<int64_t, 4>{0, 2, 1, 3})) {
      continue;
    }

    NodeArg* q_arg = graph.GetNodeArg(qB.feed);
    NodeArg* k_arg = graph.GetNodeArg(kB.feed);
    NodeArg* v_arg = graph.GetNodeArg(vB.feed);
    if (q_arg == nullptr || k_arg == nullptr || v_arg == nullptr) continue;
    if (!IsFloatTensor(*q_arg) || !IsFloatTensor(*k_arg) || !IsFloatTensor(*v_arg)) continue;

    const int64_t hidden = qB.num_heads * qB.head_dim;
    // --- recover the additive mask bias, if any. The tf2onnx form is
    //     Mul(neg, Sub(one, keepmask[ -> Tile])), keepmask = Reshape(mask) of shape
    //     [B|1, 1, 1, Sk]. KdnnFusedAttention wants a clean additive [B|1, 1, Sk]
    //     bias and broadcasts it over heads internally, so we REBUILD the bias from
    //     the keep-mask (skipping any Tile, which only replicates over heads) with
    //     fresh Sub/Mul nodes -- exactly the proven original recipe. This is what
    //     keeps the result bit-identical even though the live graph's mask Mul may
    //     be post-Tile ([B, H, 1, Sk]).
    NodeArg* mask_arg = nullptr;
    const Node* mask_sub = nullptr;  // the Sub(one, keepmask) we will subsume
    const Node* mask_tile = nullptr;
    const Node* mask_reshape = nullptr;
    NodeArg* keepmask_arg = nullptr;
    NodeArg* one_arg = nullptr;
    NodeArg* neg_arg = nullptr;
    if (mask_mul != nullptr) {
      if (!IsOnnxOp(mask_mul, "Mul") || mask_mul->InputDefs().size() != 2) continue;
      // neg constant + Sub branch
      const std::string m0 = mask_mul->InputDefs()[0]->Name();
      const std::string m1 = mask_mul->InputDefs()[1]->Name();
      const bool mc0 = IsConst(graph, m0), mc1 = IsConst(graph, m1);
      if (mc0 == mc1) continue;
      const std::string neg_name = mc0 ? m0 : m1;
      mask_sub = graph_utils::GetInputNode(*mask_mul, mc0 ? 1 : 0);
      if (!IsOnnxOp(mask_sub, "Sub") || mask_sub->InputDefs().size() != 2) continue;
      // Only Sub(constant, keepmask) has the semantics rebuilt below.
      const std::string s0 = mask_sub->InputDefs()[0]->Name();
      const std::string s1 = mask_sub->InputDefs()[1]->Name();
      const bool sc0 = IsConst(graph, s0), sc1 = IsConst(graph, s1);
      if (!sc0 || sc1) continue;
      float one_value = 0.0f;
      float neg_value = 0.0f;
      if (!ScalarFloat(graph, s0, one_value) || !ScalarFloat(graph, neg_name, neg_value) ||
          !std::isfinite(one_value) || !std::isfinite(neg_value)) {
        continue;
      }
      const Node* km = graph_utils::GetInputNode(*mask_sub, 1);
      // keepmask may be Tile(Reshape(mask)) or Reshape(mask) directly.
      if (IsOnnxOp(km, "Tile")) {
        if (km->InputDefs().size() != 2) continue;
        std::vector<int64_t> repeats;
        if (!Int64Vec(graph, km->InputDefs()[1]->Name(), repeats) ||
            repeats != std::vector<int64_t>({1, qB.num_heads, 1, 1})) {
          continue;
        }
        mask_tile = km;
        mask_reshape = graph_utils::GetInputNode(*km, 0);
        keepmask_arg = graph.GetNode(km->Index())->MutableInputDefs()[0];  // pre-Tile [B|1,1,1,Sk]
      } else if (IsOnnxOp(km, "Reshape")) {
        // use the Sub's computed input directly (no Tile)
        mask_reshape = km;
        keepmask_arg = graph.GetNode(mask_sub->Index())->MutableInputDefs()[1];
      } else {
        continue;
      }
      if (keepmask_arg == nullptr) continue;
      if (!IsOnnxOp(mask_reshape, "Reshape") || mask_reshape->InputDefs().size() < 2) continue;
      std::vector<int64_t> mask_shape;
      if (!Int64Vec(graph, mask_reshape->InputDefs()[1]->Name(), mask_shape)) continue;
      const bool valid_mask_shape =
          qB.needs_reshape
              ? mask_shape == std::vector<int64_t>({1, -1, 1, kB.seq})
              : mask_shape.size() == 4 && mask_shape[0] >= -1 && mask_shape[1] == 1 &&
                    mask_shape[2] == 1 && mask_shape[3] == kB.seq;
      if (!valid_mask_shape) continue;
      one_arg = graph.GetNodeArg(s0);
      neg_arg = graph.GetNodeArg(neg_name);
      if (one_arg == nullptr || neg_arg == nullptr) continue;
    }

    // Every removed interior node must be private to this pattern. Otherwise
    // deleting it would silently disconnect another consumer or graph output.
    Node& av_node = *graph.GetNode(av->Index());
    if (graph.NodeProducesGraphOutput(av_node) ||
        !CanRemoveSingleConsumerNode(graph, qk) ||
        (scaled != qk && !CanRemoveSingleConsumerNode(graph, scaled))) {
      continue;
    }
    if (mask_mul != nullptr &&
        (!CanRemoveSingleConsumerNode(graph, scores) ||
         !CanRemoveSingleConsumerNode(graph, mask_mul) ||
         !CanRemoveSingleConsumerNode(graph, mask_sub) ||
         (mask_tile != nullptr && !CanRemoveSingleConsumerNode(graph, mask_tile)))) {
      continue;
    }

    // The multi-head [B,S,H,d] form is consumed without a copy. Only the
    // single-head [1,B,S,d] layout needs canonicalization.
    if (qB.needs_reshape) {
      q_arg = &InsertReshape(graph, *q_arg, {-1, qB.seq, hidden}, ep, softmax.Name() + "/kdnn_q_canon");
      k_arg = &InsertReshape(graph, *k_arg, {-1, kB.seq, hidden}, ep, softmax.Name() + "/kdnn_k_canon");
      v_arg = &InsertReshape(graph, *v_arg, {-1, vB.seq, hidden}, ep, softmax.Name() + "/kdnn_v_canon");
    }

    if (mask_mul != nullptr) {
      // Rebuild: sub_out = Sub(one, keepmask); mask = Mul(neg, sub_out).
      NodeArg& sub_out = graph.GetOrCreateNodeArg(
          graph.GenerateNodeArgName(softmax.Name() + "/kdnn_mask_sub"), nullptr);
      NodeArg& bias_out = graph.GetOrCreateNodeArg(
          graph.GenerateNodeArgName(softmax.Name() + "/kdnn_mask_bias"), nullptr);
      std::array<NodeArg*, 2> sub_in{one_arg, keepmask_arg};
      std::array<NodeArg*, 1> sub_outs{&sub_out};
      Node& sub_n = graph.AddNode(graph.GenerateNodeName(softmax.Name() + "/kdnn_mask_sub"),
                                  "Sub", "KDNN attention mask: 1 - keep_mask", sub_in, sub_outs,
                                  nullptr, kOnnxDomain);
      sub_n.SetExecutionProviderType(ep);
      std::array<NodeArg*, 2> mul_in{neg_arg, &sub_out};
      std::array<NodeArg*, 1> mul_outs{&bias_out};
      Node& mul_n = graph.AddNode(graph.GenerateNodeName(softmax.Name() + "/kdnn_mask_mul"),
                                  "Mul", "KDNN attention mask: * neg", mul_in, mul_outs,
                                  nullptr, kOnnxDomain);
      mul_n.SetExecutionProviderType(ep);
      mask_arg = &bias_out;
      // Single-head form: the bias is [1, B, 1, Sk]; the op needs leading dim == B
      // (or 1 for a shared bias). Reshape [1,B,1,Sk] -> [B,1,Sk] so each batch row
      // gets its own mask. (Sk is static from the head split.)
      if (qB.needs_reshape) {
        mask_arg = &InsertReshape(graph, *mask_arg, {-1, 1, kB.seq}, ep,
                                  softmax.Name() + "/kdnn_mask_canon");
      }
    }

    // The AV matmul's output is the fusion's terminal: its consumers are preserved
    // and re-pointed at the fused op. Build inputs {Q, K, V[, mask]} and the fused node.
    InlinedVector<NodeArg*> attn_in{q_arg, k_arg, v_arg};
    if (mask_arg != nullptr) attn_in.push_back(mask_arg);
    NodeArg& attn_out = graph.GetOrCreateNodeArg(
        graph.GenerateNodeArgName(av_node.Name() + "/kdnn_attn_out"), nullptr);
    InlinedVector<NodeArg*> attn_outs{&attn_out};
    Node& attn = graph.AddNode(graph.GenerateNodeName(av_node.Name() + "/KdnnFusedAttention"),
                               "KdnnFusedAttention", "Fused multi-head attention (KDNN, structural)",
                               attn_in, attn_outs, nullptr, kKdnnDomain);
    attn.AddAttribute("num_heads", qB.num_heads);
    // Only emit an explicit scale when it actually differs from the kernel's
    // default 1/sqrt(head_dim). When the graph's scale IS 1/sqrt(head_dim) (the
    // usual case) we omit the attribute so the kernel derives the identical alpha
    // it always has -- passing 1.0f/sqrtf(<rounded sqrt>) instead would perturb
    // the last bit and break bit-exactness with the unfused graph.
    if (has_scale) {
      const float kernel_default = 1.0f / std::sqrt(static_cast<float>(qB.head_dim));
      if (std::fabs(scale - kernel_default) > 1e-6f * kernel_default) {
        attn.AddAttribute("scale", scale);
      }
    }
    attn.SetExecutionProviderType(ep);

    // The rank-4 output already matches the original AV shape [B,H,Sq,d]. For the
    // single-head form the rank-3 op output is [B,Sq,d], while the original AV
    // matmul produced [1,B,Sq,d], so reshape it back before feeding downstream
    // consumers.
    NodeArg* downstream_arg = &attn_out;
    if (qB.needs_reshape) {
      downstream_arg = &InsertReshape(graph, attn_out, {1, -1, qB.seq, hidden}, ep,
                                      av_node.Name() + "/kdnn_out_canon");
    }

    // Re-point every consumer of the AV matmul output at the (possibly reshaped) op output.
    std::vector<std::pair<Node*, int>> rewire;
    for (auto it = av_node.OutputEdgesBegin(); it != av_node.OutputEdgesEnd(); ++it) {
      rewire.emplace_back(&const_cast<Node&>(it->GetNode()), it->GetDstArgIndex());
    }
    // (collect first, then mutate: ReplaceNodeInput invalidates the edge iterator)
    graph_utils::RemoveNodeOutputEdges(graph, av_node);
    for (auto& [dst, idx] : rewire) {
      graph_utils::ReplaceNodeInput(*dst, idx, *downstream_arg);
    }

    // Remove the interior nodes that the fused op subsumes: the AV matmul, the
    // softmax, the (optional) mask Add, the scale layer (if separate), the QK
    // matmul and the OLD mask chain (Mul/Sub/Tile) which we replaced with fresh
    // clean-bias nodes above. The Q/K/V head-split Reshapes feed the inserted canon
    // Reshapes (single-head form) or the op directly (multi-head form) so they stay live; but the
    // Q/K/V-branch Transposes we passed through fed ONLY the now-removed QK/AV matmul
    // and would otherwise be scheduled as dead 4D reorders -- drop them explicitly
    // (ORT's Level1 DCE does not reliably collect them after this pass).
    auto drop = [&](const Node* n) {
      if (n == nullptr) return;
      Node* m = graph.GetNode(n->Index());
      if (m == nullptr) return;
      graph_utils::RemoveNodeOutputEdges(graph, *m);
      graph.RemoveNode(m->Index());
    };
    // Drop a passed-through Transpose only if it is now genuinely dead (its single
    // consumer was the matmul we just subsumed); guards against a shared Transpose.
    auto drop_dead_transpose = [&](const Node* n) {
      if (n == nullptr) return;
      Node* m = graph.GetNode(n->Index());
      if (m == nullptr) return;
      if (m->GetOutputEdgesCount() != 0) return;  // still consumed elsewhere -> keep
      if (graph.NodeProducesGraphOutput(*m)) return;
      graph.RemoveNode(m->Index());
    };
    drop(av);
    drop(&softmax);
    if (mask_mul != nullptr) {
      drop(scores);     // the Add(scaled, maskbias)
      drop(mask_mul);   // old Mul(neg, Sub)
      drop(mask_sub);   // old Sub(one, keepmask)
      drop(mask_tile);  // old Tile (may be null)
    }
    if (scaled != qk) drop(scaled);  // the separate Mul/Div scale
    drop(qk);
    // qk/av are gone now -> the branch Transposes they fed are dead; collect them.
    drop_dead_transpose(qB.skip_transpose);
    drop_dead_transpose(kB.skip_transpose);
    drop_dead_transpose(vB.skip_transpose);

    ++fused;
  }

  if (fused > 0) {
    modified = true;
    LOGS(logger, INFO) << "KdnnAttentionFusion fused " << fused << " attention block(s).";
  }
  return Status::OK();
}

}  // namespace onnxruntime
