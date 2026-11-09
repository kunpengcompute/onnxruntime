# Copyright (c) Huawei Technologies Co., Ltd. 2026.
# Licensed under the MIT License.
"""KDNN EP graph optimization, partitioning and ONNX serialization tests."""

import concurrent.futures
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.capi.onnxruntime_pybind11_state import NotImplemented as OrtNotImplemented

import onnxruntime as ort

DOMAIN = "com.kdnn.internal"
EP = "KDNNExecutionProvider"


def attention_model(
    *,
    mask=True,
    tile=True,
    reverse_div=False,
    axis=-1,
    shared=False,
    terminal=True,
    domain_version=None,
    dynamic=False,
    heads=4,
    dim=5,
    sk=7,
    offset=1.0,
    multiplier=-10.0,
    scale=0.25,
    overridable=False,
):
    """Build an attention graph with BSHd inputs and no QKV projection nodes."""
    batch = "batch" if dynamic else 2
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, [batch, seq, heads, dim])
        for name, seq in [("q", 1), ("k", sk), ("v", sk)]
    ]
    nodes = [
        helper.make_node("Transpose", [name], [name + "t"], perm=perm)
        for name, perm in [("q", [0, 2, 1, 3]), ("k", [0, 2, 3, 1]), ("v", [0, 2, 1, 3])]
    ]
    initializers = []

    def constant(name, value, dtype=np.float32):
        initializers.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    nodes.append(helper.make_node("MatMul", ["qt", "kt"], ["qk"]))
    if reverse_div:
        nodes.append(helper.make_node("Div", [constant("scale", scale), "qk"], ["scaled"]))
    else:
        nodes.append(helper.make_node("Mul", ["qk", constant("scale", scale)], ["scaled"]))
    if overridable:
        inputs.append(helper.make_tensor_value_info("scale", TensorProto.FLOAT, []))
    scores = "scaled"
    if mask:
        inputs.append(helper.make_tensor_value_info("mask", TensorProto.FLOAT, [batch, 1, 1, sk]))
        keep = "mask"
        if tile:
            keep = "tiled"
            nodes.append(helper.make_node("Tile", ["mask", constant("repeats", [1, heads, 1, 1], np.int64)], [keep]))
        nodes.extend(
            [
                helper.make_node("Sub", [constant("offset", offset), keep], ["sub"]),
                helper.make_node("Mul", [constant("multiplier", multiplier), "sub"], ["bias"]),
                helper.make_node("Add", ["scaled", "bias"], ["scores"]),
            ]
        )
        scores = "scores"
    nodes.extend(
        [
            helper.make_node("Softmax", [scores], ["probs"], axis=axis),
            helper.make_node("MatMul", ["probs", "vt"], ["out" if terminal else "av"]),
        ]
    )
    if not terminal:
        nodes.append(helper.make_node("Reshape", ["av", constant("shape", [-1, heads * dim], np.int64)], ["out"]))
    outputs = [
        helper.make_tensor_value_info(
            "out", TensorProto.FLOAT, [batch, heads, 1, dim] if terminal else [batch, heads * dim]
        )
    ]
    if shared:
        outputs.append(helper.make_tensor_value_info("qt", TensorProto.FLOAT, [batch, heads, 1, dim]))
    imports = [helper.make_opsetid("", 13)]
    if domain_version is not None:
        imports.append(helper.make_opsetid(DOMAIN, domain_version))
    return helper.make_model(
        helper.make_graph(nodes, "attention", inputs, outputs, initializers), opset_imports=imports, ir_version=8
    )


def feeds_for(model, batch=2):
    rng = np.random.default_rng(42)
    feeds = {}
    initializers = {x.name for x in model.graph.initializer}
    for arg in model.graph.input:
        if arg.name in initializers:
            continue
        shape = [d.dim_value if d.HasField("dim_value") else batch for d in arg.type.tensor_type.shape.dim]
        feeds[arg.name] = rng.uniform(-0.5, 0.5, shape).astype(np.float32)
    return feeds


def fused_attention_with_shape_output(shapes, version):
    """Return attention and Shape outputs to check schema inference after optimization."""
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in shapes.items()]
    nodes = [
        helper.make_node("KdnnFusedAttention", list(shapes), ["out"], domain=DOMAIN, num_heads=2),
        helper.make_node("Shape", ["out"], ["output_shape"]),
    ]
    outputs = [
        helper.make_tensor_value_info("out", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("output_shape", TensorProto.INT64, [len(shapes["q"])]),
    ]
    return helper.make_model(
        helper.make_graph(nodes, "attention_shape", inputs, outputs),
        opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DOMAIN, version)],
        ir_version=8,
    )


def make_session(
    model,
    *,
    kdnn=True,
    level=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    options=None,
    save=None,
    profile=False,
    threads=2,
):
    so = ort.SessionOptions()
    so.graph_optimization_level = level
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.enable_profiling = profile
    if save:
        so.optimized_model_filepath = str(save)
        if profile:
            so.profile_file_prefix = str(Path(save).parent / "profile")
    providers = [(EP, options or {}), "CPUExecutionProvider"] if kdnn else ["CPUExecutionProvider"]
    payload = model.SerializeToString() if isinstance(model, onnx.ModelProto) else str(model)
    session = ort.InferenceSession(payload, so, providers=providers)
    session.disable_fallback()
    if kdnn and EP not in session.get_providers():
        raise AssertionError("KDNN EP creation silently fell back to CPU")
    return session


def fused_nodes(graph):
    result = [n for n in graph.node if n.domain == DOMAIN and n.op_type == "KdnnFusedAttention"]
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                result.extend(fused_nodes(attr.g))
    return result


@unittest.skipUnless(EP in ort.get_available_providers(), "requires a KDNN EP build")
class TestKdnnExecutionProvider(unittest.TestCase):
    def compare(
        self,
        model,
        *,
        expected=1,
        options=None,
        level=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        feeds=None,
        kdnn=True,
        executed=None,
    ):
        feeds = feeds_for(model) if feeds is None else feeds
        reference = make_session(model, kdnn=False).run(None, feeds)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimized.onnx"
            session = make_session(model, kdnn=kdnn, options=options, level=level, save=path, profile=True)
            actual = session.run(None, feeds)
            saved = onnx.load(path)
            self.assertEqual(len(fused_nodes(saved.graph)), expected)
            for actual_output, reference_output in zip(actual, reference, strict=True):
                np.testing.assert_allclose(actual_output, reference_output, atol=1e-4, rtol=1e-4, equal_nan=True)
            profile = json.loads(Path(session.end_profiling()).read_text())
            kernel_events = [e for e in profile if "provider" in e.get("args", {})]
            self.assertEqual(
                sum(e["args"]["provider"] == EP for e in kernel_events), expected if executed is None else executed
            )
            self.assertFalse(any("Memcpy" in e["args"].get("op_name", "") for e in kernel_events))
            if expected:
                self.assertEqual(next(i.version for i in saved.opset_import if i.domain == DOMAIN), 2)
                reloaded = make_session(
                    path, level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL, options={"enable_attention_fusion": "0"}
                )
                for result, original in zip(reloaded.run(None, feeds), actual, strict=True):
                    np.testing.assert_allclose(result, original, atol=1e-6, rtol=1e-6, equal_nan=True)

    def test_supported_shapes_mask_and_terminal_output(self):
        for mask, tile in [(False, False), (True, False), (True, True)]:
            for terminal in [False, True]:
                with self.subTest(mask=mask, tile=tile, terminal=terminal):
                    self.compare(attention_model(mask=mask, tile=tile, terminal=terminal, offset=0.75, multiplier=3.25))

    def test_provider_and_optimization_gates(self):
        model = attention_model()
        self.compare(model, kdnn=False, expected=0)
        self.compare(model, level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL, expected=0)
        self.compare(model, level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, expected=0)
        self.compare(model, options={"enable_attention_fusion": "0"}, expected=0)
        self.compare(model, level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL)

    def test_cpu_tensordot_prepacked_weight(self):
        weight = ((np.arange(32 * 64).reshape(32, 64) % 17 - 8) / 32).astype(np.float32)
        node = helper.make_node(
            "FusedTensordotMatMul", ["x", "w"], ["y"], domain="com.microsoft", free_axes=[0, 1], contract_axes=[2]
        )
        model = helper.make_model(
            helper.make_graph(
                [node],
                "tensordot_prepack",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3, 32])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3, 64])],
                [numpy_helper.from_array(weight, "w")],
            ),
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("com.microsoft", 1)],
            ir_version=8,
        )
        for prepack in ["0", "1"]:
            with mock.patch.dict(
                os.environ, {"ORT_KDNN_FUSED_TENSORDOT_MATMUL": "1", "ORT_KDNN_FUSED_TENSORDOT_PREPACK": prepack}
            ):
                session = make_session(model, kdnn=False, level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
                for batch in [1, 4, 2]:
                    with self.subTest(prepack=prepack, batch=batch):
                        value = ((np.arange(batch * 3 * 32).reshape(batch, 3, 32) % 13 - 6) / 16).astype(np.float32)
                        np.testing.assert_array_equal(session.run(None, {"x": value})[0], value @ weight)

    def test_negative_patterns_and_domain_v1(self):
        for kwargs in [
            {"reverse_div": True},
            {"axis": 1},
            {"shared": True},
            {"heads": 1},
            {"overridable": True},
            {"domain_version": 1},
        ]:
            with self.subTest(**kwargs):
                self.compare(attention_model(**kwargs), expected=0)

    def test_scale_near_default(self):
        scale = np.float32(1.0 / np.sqrt(5.0))
        scale = float(np.nextafter(scale, np.float32(np.inf)))
        self.compare(attention_model(mask=False, scale=scale))

    def test_constant_reshape_without_upstream_shape_inference(self):
        model = attention_model()
        feeds = feeds_for(model)
        reshapes = []
        for arg in model.graph.input:
            name = arg.name
            shape = [d.dim_value for d in arg.type.tensor_type.shape.dim]
            arg.type.tensor_type.ClearField("shape")
            arg.name = name + "raw"
            feeds[arg.name] = feeds.pop(name)
            shape[0] = -1
            model.graph.initializer.append(numpy_helper.from_array(np.asarray(shape, dtype=np.int64), name + "shape"))
            reshapes.append(helper.make_node("Reshape", [arg.name, name + "shape"], [name]))
        original_nodes = list(model.graph.node)
        del model.graph.node[:]
        model.graph.node.extend(reshapes + original_nodes)
        self.compare(model, feeds=feeds)

    def test_l1_constant_folded_mask(self):
        for value in [1.0, 0.0, float("nan")]:
            with self.subTest(value=value):
                model = attention_model()
                mask = model.graph.input.pop()
                model.graph.initializer.append(
                    numpy_helper.from_array(np.full([1, 1, 1, 7], value, dtype=np.float32), mask.name)
                )
                self.compare(model, expected=1 if value == 1.0 else 0)

    def test_nonfinite_and_extreme_finite_inputs(self):
        model = attention_model(tile=False, multiplier=1.0, offset=0.0)
        for name, value in [
            ("mask", np.inf),
            ("mask", -np.inf),
            ("q", np.nan),
            ("q", np.inf),
            ("k", 1e38),
            ("q", 1e38),
            ("v", np.nan),
        ]:
            with self.subTest(name=name, value=value):
                feeds = feeds_for(model)
                feeds[name].fill(value)
                self.compare(model, feeds=feeds)

    def test_dynamic_batch_broadcast_and_concurrent_sessions(self):
        model = attention_model(dynamic=True)
        sessions = [
            make_session(model, options={"attention_algorithm": algorithm})
            for algorithm in ["classic", "classic_no_pack"]
        ]
        reference = make_session(model, kdnn=False)
        cases = []
        for batch in [1, 3, 2]:
            for broadcast in ["q", "k", "v", "mask"]:
                feeds = feeds_for(model, batch)
                feeds[broadcast] = feeds[broadcast][:1]
                cases.append((feeds, reference.run(None, feeds)))

        def run(index):
            feeds, expected = cases[index % len(cases)]
            actual = sessions[index % 2].run(None, feeds)
            np.testing.assert_allclose(actual[0], expected[0], atol=1e-4, rtol=1e-4)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(run, range(48)))

    def test_finite_products_do_not_overflow_before_accumulation(self):
        # The second product exceeds FLT_MAX, but the fused multiply-add used
        # by CPU MatMul cancels it against the first, negative product. A
        # separate multiply/add in the exceptional path incorrectly yields Inf
        # and then NaN in Softmax even though every input is finite.
        for dim in [2, 5, 32]:
            model = attention_model(mask=False, heads=2, dim=dim, sk=2, scale=1.0)
            feeds = feeds_for(model)
            feeds["q"].fill(1e20)
            feeds["k"].fill(0)
            feeds["k"][:, 0, :, 0] = -3e18
            feeds["k"][:, 0, :, 1] = 6e18
            feeds["v"].fill(1)
            reference = make_session(model, kdnn=False).run(None, feeds)[0]
            np.testing.assert_array_equal(reference, np.ones_like(reference))
            for algorithm in ["auto", "classic", "classic_no_pack"]:
                with self.subTest(dim=dim, algorithm=algorithm):
                    self.compare(model, feeds=feeds, options={"attention_algorithm": algorithm})

    def test_v1_remains_executable_without_new_fusion(self):
        model = attention_model(mask=False)
        del model.graph.node[:]
        model.opset_import.append(helper.make_opsetid(DOMAIN, 1))
        model.graph.node.append(
            helper.make_node("KdnnFusedAttention", ["q", "k", "v"], ["out"], domain=DOMAIN, num_heads=4, scale=0.25)
        )
        feeds = feeds_for(model)
        reference = make_session(attention_model(mask=False), kdnn=False).run(None, feeds)
        for level in [ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_BASIC]:
            result = make_session(model, level=level, options={"enable_attention_fusion": "0"}).run(None, feeds)
            np.testing.assert_allclose(result[0], reference[0], atol=1e-4, rtol=1e-4)
        with self.assertRaises(OrtNotImplemented):
            make_session(model, kdnn=False)

    def test_unknown_input_shapes_do_not_freeze_output_batch(self):
        for version in [1, 2]:
            for rank in [3, 4]:
                shape_suffixes = {"q": [1, 2, 5], "k": [3, 2, 5], "v": [3, 2, 5], "mask": [1, 1, 3]}
                if rank == 3:
                    shape_suffixes.update(q=[1, 10], k=[3, 10], v=[3, 10])
                for unknown in ["k", "v", "mask"]:
                    with self.subTest(version=version, rank=rank, unknown=unknown):
                        shapes = {name: [1, *suffix] for name, suffix in shape_suffixes.items()}
                        shapes[unknown] = None
                        model = fused_attention_with_shape_output(shapes, version)
                        with tempfile.TemporaryDirectory() as directory:
                            path = Path(directory) / "optimized.onnx"
                            session = make_session(model, save=path)
                            self.assertNotIsInstance(session.get_outputs()[0].shape[0], int)
                            reloaded = make_session(path, level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
                            for batch in [2, 3, 1]:
                                feeds = {
                                    name: np.ones([batch if name == unknown else 1, *suffix], dtype=np.float32)
                                    for name, suffix in shape_suffixes.items()
                                }
                                expected_shape = [batch, 2, 1, 5] if rank == 4 else [batch, 1, 10]
                                for current in [session, reloaded]:
                                    output, reported_shape = current.run(None, feeds)
                                    self.assertEqual(list(output.shape), expected_shape)
                                    np.testing.assert_array_equal(reported_shape, expected_shape)
                                    np.testing.assert_allclose(output, np.ones(expected_shape), rtol=1e-6, atol=1e-6)

    def test_batch_shape_inference_preserves_known_nonunit_dimensions(self):
        for version in [1, 2]:
            for batch in [0, 1, 3]:
                with self.subTest(version=version, batch=batch):
                    shapes = {"q": [batch, 1, 2, 5], "k": [1, 3, 2, 5], "v": [1, 3, 2, 5]}
                    # Omitting the optional mask must not introduce an unknown batch.
                    session = make_session(fused_attention_with_shape_output(shapes, version))
                    self.assertEqual(session.get_outputs()[0].shape, [batch, 2, 1, 5])
                    if batch != 1:
                        for key_shape in [None, ["key_batch", 3, 2, 5]]:
                            shapes["k"] = key_shape
                            session = make_session(fused_attention_with_shape_output(shapes, version))
                            self.assertEqual(session.get_outputs()[0].shape, [batch, 2, 1, 5])

    def test_two_independent_candidates(self):
        model = attention_model()
        other = attention_model()
        for item in [*other.graph.input, *other.graph.output, *other.graph.initializer]:
            item.name += "2"
        for node in other.graph.node:
            for i in range(len(node.input)):
                node.input[i] += "2"
            for i in range(len(node.output)):
                node.output[i] += "2"
        model.graph.node.extend(other.graph.node)
        model.graph.input.extend(other.graph.input)
        model.graph.output.extend(other.graph.output)
        model.graph.initializer.extend(other.graph.initializer)
        self.compare(model, expected=2)

    def test_loop_captures(self):
        model = attention_model()
        body = model.graph
        inputs, constants = list(body.input), list(body.initializer)
        del body.input[:]
        del body.initializer[:]
        body.input.extend(
            [
                helper.make_tensor_value_info("iteration", TensorProto.INT64, []),
                helper.make_tensor_value_info("condition_in", TensorProto.BOOL, []),
                helper.make_tensor_value_info("state", TensorProto.FLOAT, [2, 4, 1, 5]),
            ]
        )
        body.node.append(helper.make_node("Identity", ["condition_in"], ["condition_out"]))
        body.output.insert(0, helper.make_tensor_value_info("condition_out", TensorProto.BOOL, []))
        loop = helper.make_node("Loop", ["trip", "condition", "initial"], ["result"], body=body)
        constants.extend(
            [
                numpy_helper.from_array(np.array(2, dtype=np.int64), "trip"),
                numpy_helper.from_array(np.array(True), "condition"),
                numpy_helper.from_array(np.zeros((2, 4, 1, 5), dtype=np.float32), "initial"),
            ]
        )
        model.graph.CopyFrom(
            helper.make_graph(
                [loop],
                "loop_attention",
                inputs,
                [helper.make_tensor_value_info("result", TensorProto.FLOAT, [2, 4, 1, 5])],
                constants,
            )
        )
        self.compare(model, expected=1, executed=2)

    def test_if_captures_and_multiple_candidates(self):
        model = attention_model()
        branch = model.graph
        # Both branches capture Q/K/V/mask and use outer-scope scalar constants.
        branch_inputs = list(branch.input)
        constants = list(branch.initializer)
        del branch.input[:]
        del branch.initializer[:]
        condition = numpy_helper.from_array(np.array(True), "condition")
        if_node = helper.make_node("If", ["condition"], ["result"], then_branch=branch, else_branch=branch)
        output = helper.make_tensor_value_info("result", TensorProto.FLOAT, [2, 4, 1, 5])
        model.graph.CopyFrom(
            helper.make_graph([if_node], "if_attention", branch_inputs, [output], [*constants, condition])
        )
        # A runtime condition prevents L1 from inlining the selected branch.
        model.graph.input.append(helper.make_tensor_value_info("condition", TensorProto.BOOL, []))
        feeds = feeds_for(model)
        feeds["condition"] = np.array(True)
        self.compare(model, expected=2, executed=1, feeds=feeds)


if __name__ == "__main__":
    if "--require-kdnn" in sys.argv:
        sys.argv.remove("--require-kdnn")
        if EP not in ort.get_available_providers():
            raise SystemExit("KDNN EP was requested for this test run but is missing from the loaded ORT build")
    unittest.main()
