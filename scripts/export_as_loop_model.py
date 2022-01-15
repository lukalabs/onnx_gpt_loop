from argparse import ArgumentParser

from onnx_gpt_loop.models.one_step_torch import OneStepTorchModel
from onnx_gpt_loop.onnx_export import export_as_loop_model


def _parse_args():
    parser = ArgumentParser(description='Exports pytorch gpt model to the optimized loop ONNX model.')
    parser.add_argument(
        '--model-name-or-path',
        '-m',
        type=str,
        required=True,
        help='Huggingface pretrained gpt2 model name or path. '
        'It could be a path to your pretrained model.',
    )
    parser.add_argument(
        '--out-file-path',
        '-f',
        type=str,
        required=True,
        help='Path to the saved ONNX model.',
    )
    return parser.parse_args()


def main(model_name_or_path, out_file_path):
    one_step_torch_model = OneStepTorchModel.from_pretrained(model_name_or_path)
    export_as_loop_model(one_step_torch_model, out_file_path)


if __name__ == '__main__':
    args = _parse_args()
    main(**vars(args))
