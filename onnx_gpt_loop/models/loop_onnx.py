import numpy as np
from onnxruntime import InferenceSession
from onnx_gpt_loop.models import HasGenerationLoop


class LoopOnnxModel(HasGenerationLoop):
    """Onnxruntime InferenceSession wrapper for the GPT loop ONNX model.

    To construct ONNX file for the model use the following steps::
        
        from onnx_gpt_loop.models.one_step_torch import OneStepTorchModel
        from onnx_gpt_loop.onnx_export import convert_one_step_to_loop_onnx, export_one_step_model

        one_step_torch_model = OneStepTorchModel.from_pretrained('gpt2').cuda()
        export_one_step_model(one_step_torch_model, '/tmp/one_step.onnx')
        convert_one_step_to_loop('/tmp/one_step.onnx', '/tmp/loop.onnx')
    
    At this point you'll have '/tmp/loop.onnx' file path which represents an ONNX
    model with full generation loop blended inside.
    """

    def __init__(self, file_path):
        """
        :param file_path: Path to the ONNX loop model.
        """
        self._session = InferenceSession(
            path_or_bytes=str(file_path),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )

    def generate(self, n_steps, prefix_ids, temperature, top_k):
        """Runs full GPT inference loop cycle.

        :param n_step: Number of tokens to be generated.
        :param prefix_ids: Prefix token ids.
        :param temperature: Temperature of the tokens sampling distribution.
        :param top_k: Top-k sampling number of tokens.

        :return: Numpy array of generated tokens with shape (batch_size, n_steps).
        """
        pasts_input_feed = self._get_pasts_input_feed(batch_size=prefix_ids.shape[0])
        input_feed = {
            'n_steps': np.array([n_steps], dtype=np.int64),
            'input_ids': prefix_ids,
            'temperature': np.array(temperature, dtype=np.float64),
            'top_k': np.array(top_k, dtype=np.int64),
            **pasts_input_feed
        }

        outputs = self._session.run(['all_output_ids'], input_feed)
        output_ids = outputs[0].T
        return output_ids

    def _get_pasts_input_feed(self, batch_size):
        pasts_shape = self._session.get_inputs()[-1].shape[:]
        pasts_shape[1] = batch_size
        pasts_shape[3] = 0

        pasts_input_feed = {}
        for inp in self._session.get_inputs():
            if inp.name.startswith('input_past_key_values_'):
                pasts_input_feed[inp.name] = np.zeros(pasts_shape, dtype=np.float16)

        return pasts_input_feed
