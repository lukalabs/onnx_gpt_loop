import datetime
import time
import logging
import tempfile
from argparse import ArgumentParser
from functools import partial
from os import path

import numpy as np
import torch
import tqdm

from onnx_gpt_loop.models import HasGenerationLoop
from onnx_gpt_loop.models.loop_onnx import LoopOnnxModel
from onnx_gpt_loop.models.one_step_torch import OneStepTorchModel
from onnx_gpt_loop.onnx_export import export_as_loop_model

_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def _parse_args():
    parser = ArgumentParser(description='Benchmarks generation loop execution '
                            'time for pure torch model and ONNX loop model.')
    parser.add_argument(
        '--model-name',
        type=str,
        required=False,
        default='gpt2',
        help='Huggingface pretrained gpt2 model name',
        choices=['gpt2', 'gpt2-medium', 'gpt2-large'],
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        required=False,
        default=64,
        help='Number of samples in benchmark batch',
    )
    parser.add_argument(
        '--prefix-seq-len',
        type=int,
        required=False,
        default=36,
        help='Number of prefix tokens',
    )
    parser.add_argument(
        '--n-generation-steps',
        type=int,
        required=False,
        default=36,
        help='Number of tokens to be generated',
    )
    parser.add_argument(
        '--n-benchmark-steps',
        type=int,
        required=False,
        default=20,
        help='Number of sequences to be generated for each benchmark model',
    )
    return parser.parse_args()


def _benchmark_generation_loop(
        model: HasGenerationLoop,
        batch_size,
        prefix_seq_len,
        n_generation_steps,
        n_benchmark_steps,
):

    prefix_ids = np.ones((batch_size, prefix_seq_len), dtype=np.int64)
    attention_mask = np.ones_like(prefix_ids, dtype=np.float64)
    position_ids = np.ones_like(prefix_ids)
    step_seconds = []
    for _ in tqdm.trange(n_benchmark_steps, desc=model.__class__.__name__):
        start_time = datetime.datetime.now()
        model.generate(
            n_steps=n_generation_steps,
            prefix_ids=prefix_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            temperature=0.7,
            top_k=50,
        )
        step_seconds.append((datetime.datetime.now() - start_time).total_seconds())

    return min(step_seconds)


def main(model_name, batch_size, prefix_seq_len, n_generation_steps, n_benchmark_steps):
    benchmark_generation_loop = partial(
        _benchmark_generation_loop,
        batch_size=batch_size,
        prefix_seq_len=prefix_seq_len,
        n_generation_steps=n_generation_steps,
        n_benchmark_steps=n_benchmark_steps,
    )
    _logger.info('Creating torch model')
    one_step_torch_model = OneStepTorchModel.from_pretrained(model_name).cuda()

    with tempfile.TemporaryDirectory() as tmp_dir:
        loop_onnx_file_path = path.join(tmp_dir, 'loop.onnx')

        _logger.info('Creating loop ONNX model')
        export_as_loop_model(one_step_torch_model, loop_onnx_file_path)
        loop_onnx_model = LoopOnnxModel(loop_onnx_file_path)

        one_step_torch_model = one_step_torch_model.eval().half()
        onnx_model_time = benchmark_generation_loop(loop_onnx_model)

        del loop_onnx_model._session
        time.sleep(3)
        torch.cuda.empty_cache()

        torch_model_time = benchmark_generation_loop(one_step_torch_model)

        print(f'Torch: {torch_model_time:.4f}s')
        print(f'ONNX: {onnx_model_time:.4f}s')


if __name__ == '__main__':
    args = _parse_args()
    main(**vars(args))
