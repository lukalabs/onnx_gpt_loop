import tempfile
from os import path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper
from onnxruntime import InferenceSession
from onnxruntime.transformers import optimizer

from onnx_gpt_loop.models.one_step_torch import OneStepTorchModel
from onnx_gpt_loop.utils import get_dummy_pasts

# onnxruntime can't fuse attention layer of opset versions > 12
_OPSET_VERSION = 12


def export_as_loop_model(model: OneStepTorchModel, out_file_path):
    """Converts `OneStepTorchModel` to the loop ONNX model and
    exports it to the file.

    It also optimizes a model with onnxruntime gpt optimizer and converts
    it to fp16.

    :param model: `OneStepTorchModel` object.
    :param out_file_path: File path to save output ONNX model file.
    """
    with TemporaryDirectory() as tmp_dir:
        one_step_onnx_file_path = path.join(tmp_dir, 'one_step.onnx')
        export_one_step_model(model, one_step_onnx_file_path)
        convert_one_step_to_loop_onnx(one_step_onnx_file_path, out_file_path)

    InferenceSession(out_file_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])


def export_one_step_model(model: OneStepTorchModel, out_file_path):
    """Saves `OneStepTorchModel` to the optimized one step ONNX model.

    It also optimizes a model with onnxruntime gpt optimizer and converts
    it to fp16.

    :param model: `OneStepTorchModel` object.
    :param out_file_path: File path to save output ONNX model file.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file_path = path.join(tmp_dir, 'model.onnx')

        batch_size = 4
        seq_len = 4
        temperature = 0.7
        top_k = 50

        input_ids = torch.ones((batch_size, seq_len), dtype=torch.long, device=model.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.float64)
        position_ids = torch.ones_like(input_ids, dtype=torch.int64)
        past_key_values = get_dummy_pasts(
            batch_size=batch_size,
            seq_len=0,
            hidden_size=model.hidden_size,
            num_attention_heads=model.num_attention_heads,
            num_hidden_layers=model.num_hidden_layers,
            device=model.device,
        )
        dynamic_axes = {
            'input_ids': {
                0: 'batch_size',
                1: 'input_seq_len'
            },
            'next_input_ids': {
                0: 'batch_size'
            },
            'attention_mask': {
                0: 'batch_size',
                1: 'attention_mask_seq_len'
            },
            'next_attention_mask': {
                0: 'batch_size',
                1: 'next_attention_mask_seq_len'
            },
            'position_ids': {
                0: 'batch_size',
                1: 'position_ids_seq_len'
            },
            'next_position_ids': {
                0: 'batch_size'
            },
        }

        input_past_key_values_names = []
        output_past_key_values_names = []
        for i in range(model.num_hidden_layers):
            input_name = f'input_past_key_values_{i}'
            output_name = f'output_past_key_values_{i}'
            input_past_key_values_names.append(input_name)
            output_past_key_values_names.append(output_name)
            dynamic_axes[input_name] = {1: 'batch_size', 3: f'{input_name}_seq_len'}
            dynamic_axes[output_name] = {1: 'batch_size', 3: f'{output_name}_seq_len'}

        input_names = ['input_ids', 'attention_mask', 'position_ids', 'temperature', 'top_k']
        input_names += input_past_key_values_names
        output_names = ['next_input_ids', 'next_attention_mask', 'next_position_ids']
        output_names += output_past_key_values_names
        torch.onnx.export(
            model=model,
            args=tuple([input_ids, attention_mask, position_ids, temperature, top_k] + past_key_values),
            f=onnx_file_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=_OPSET_VERSION,
            use_external_data_format=True,
        )

        optimized = optimizer.optimize_model(
            input=onnx_file_path,
            model_type='gpt2',
            num_heads=model.num_attention_heads,
            hidden_size=model.hidden_size,
            use_gpu=True,
            opt_level=1,
        )

        optimized.convert_float_to_float16(keep_io_types=False, force_fp16_initializers=True)
        if not optimized.is_fully_optimized():
            raise ValueError("Can't optimize model!")

        optimized.save_model_to_file(out_file_path, use_external_data_format=False)

    InferenceSession(out_file_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])


def convert_one_step_to_loop_onnx(inp_file_path, out_file_path):
    """Converts ONNX model, saved by `export_one_step_model` to the looped version.

    This looped model encapsulates the whole inference loop with past-key-values
    caching in the graph.

    :param inp_file_path: Path to the one step model, saved by `export_one_step_model`.
    :param out_file_path: File path to save output ONNX model file.
    """
    loop_body, graph_inputs = _extract_loop_body_and_graph_inputs(inp_file_path)
    loop_node = _make_loop_node(loop_body)
    graph = _make_graph(loop_node, graph_inputs)

    op = onnx.OperatorSetIdProto()
    op.version = _OPSET_VERSION
    model = onnx.helper.make_model(graph, opset_imports=[op])

    onnx.save(model, out_file_path)


def _extract_loop_body_and_graph_inputs(file_path):
    loop_body = onnx.load(file_path).graph

    temperature = loop_body.input.pop(3)
    top_k = loop_body.input.pop(3)
    input_ids = loop_body.input[0]
    attention_mask = loop_body.input[1]
    position_ids = loop_body.input[2]
    past_key_values = loop_body.input[3:]
    # ['input_ids', 'attention_mask', 'position_ids', *'past_key_values_{i}']

    i_step = helper.make_tensor_value_info('i_step', TensorProto.INT64, [1])
    cond = helper.make_tensor_value_info('cond', onnx.TensorProto.BOOL, [])

    # Loop and its graph:
    loop_body.input.insert(0, cond)
    loop_body.input.insert(0, i_step)
    # ['i_step', 'cond', 'input_ids', 'attention_mask', 'position_ids', *'past_key_values_{i}']

    next_input_ids = loop_body.output[0]
    loop_body.output.insert(0, cond)
    loop_body.output.append(next_input_ids)
    # ['cond', 'next_input_ids', 'next_attention_mask', 'next_position_ids', *'output_past_key_values_{i}',
    # next_input_ids]

    graph_inputs = [temperature, top_k, input_ids, attention_mask, position_ids, *past_key_values]
    return loop_body, graph_inputs


def _make_loop_node(loop_body):
    loop_node_input_names = [x.name for x in loop_body.input[2:]]
    loop_node_output_names = [x + '_step' for x in loop_node_input_names]
    loop_node = helper.make_node(
        op_type='Loop',
        inputs=['n_steps', 'cond'] + loop_node_input_names,
        outputs=loop_node_output_names + ['all_output_ids_3d'],
        body=loop_body,
    )

    return loop_node


def _make_graph(loop_node, graph_inputs):
    all_output_ids = helper.make_tensor_value_info('all_output_ids', TensorProto.INT64, [None, None])
    n_steps = helper.make_tensor_value_info('n_steps', TensorProto.INT64, [1])

    cond_const_node = helper.make_node(
        op_type='Constant',
        inputs=[],
        outputs=['cond'],
        value=helper.make_tensor(
            name='cond',
            data_type=TensorProto.BOOL,
            dims=[],
            vals=np.array([True], dtype=np.bool),
        ),
    )

    squeeze_all_output_ids_3d_node = helper.make_node(
        op_type='Squeeze',
        inputs=['all_output_ids_3d'],
        outputs=['all_output_ids'],
    )

    graph = helper.make_graph(
        nodes=[
            cond_const_node,
            loop_node,
            squeeze_all_output_ids_3d_node,
        ],
        name='graph',
        # ['n_steps', 'input_ids', 'attention_mask', 'position_ids', 'temperature', 'top_k', *'past_key_values_{i}']:
        inputs=[n_steps] + graph_inputs,
        outputs=[all_output_ids],
    )

    return graph
