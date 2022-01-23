import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxruntime.transformers.gpt2_helper import MyGPT2LMHeadModel
from transformers import GPT2Config

from onnx_gpt_loop.models import HasGenerationLoop
from onnx_gpt_loop.utils import get_dummy_pasts


class OneStepTorchModel(nn.Module, HasGenerationLoop):
    """Provides inference loop for the pure pytorch GPT model.

    It is more optimized than a naive inference loop with past-key-value
    recomputing on each iteration. In this implementation past-key-values
    from previous steps are reused on each iteration. Such an optimization
    has zero cost in terms of memory consumption and decreases generation
    complexity from O(n^3) to O(n^2). Huggingface transformer models have
    this capability via passing `past_key_values` argument.

    Past-key-values caching is a strong baseline in the generative
    transformers optimization and should be considered as a starting point
    for the generation speed improvements.
    """

    def __init__(self, gpt2: MyGPT2LMHeadModel):
        """
        :param gpt2: Pretrained MyGPT2LMHeadModel object.
            Convert the gpt2 model to half and eval beforehand::

                one_step_torch_model = OneStepTorchModel(gpt2.half().eval())
        """
        super().__init__()
        self._gpt2 = gpt2

    @property
    def hidden_size(self):
        return self._gpt2.config.hidden_size

    @property
    def num_attention_heads(self):
        return self._gpt2.config.num_attention_heads

    @property
    def num_hidden_layers(self):
        return self._gpt2.config.num_hidden_layers

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def make_toy(cls):
        config = GPT2Config(
            vocab_size=1000,
            n_positions=512,
            n_embd=128,
            n_layer=5,
            n_head=8,
            use_cache=True,
        )
        return cls(MyGPT2LMHeadModel(config))

    @classmethod
    def from_pretrained(cls, model_name_or_path):
        gpt2 = MyGPT2LMHeadModel.from_pretrained(model_name_or_path, use_cache=True)
        return cls(gpt2)

    @torch.no_grad()
    def forward(self, input_ids, position_ids, attention_mask, temperature, top_k, *past_key_values):
        out = self._gpt2(
            input_ids,
            position_ids,
            attention_mask,
            *past_key_values,
        )

        next_token_logits = out[0][:, -1, :]

        next_token_logits = next_token_logits / temperature
        top_k_logits, top_k_inds = torch.topk(next_token_logits, top_k, sorted=False)
        top_k_probas = F.softmax(top_k_logits, dim=-1)
        next_input_ids = torch.multinomial(top_k_probas.type(torch.float32), num_samples=1)
        next_input_ids = top_k_inds.gather(-1, next_input_ids)

        next_position_ids = position_ids[:, -1:] + 1
        next_attention_mask = torch.ones((input_ids.size()[0], input_ids.size()[1] + 1))

        return next_input_ids, next_position_ids, next_attention_mask, *out[1]

    @torch.no_grad()
    def generate(self, n_steps, prefix_ids, temperature, top_k):
        """Runs full GPT inference loop cycle.

        :param n_step: Number of tokens to be generated.
        :param prefix_ids: Prefix token ids.
        :param temperature: Temperature of the tokens sampling distribution.
        :param top_k: Top-k sampling number of tokens.

        :return: Numpy array of generated tokens with shape (batch_size, n_steps).
        """
        prefix_ids = torch.tensor(prefix_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(prefix_ids)
        # TODO: make correct position ids:
        position_ids = torch.ones_like(prefix_ids)
        batch_size = prefix_ids.size()[0]
        pasts = get_dummy_pasts(
            batch_size=batch_size,
            seq_len=0,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            device=self.device,
            dtype=torch.float16,
        )

        output_ids = torch.zeros((batch_size, n_steps), dtype=torch.long, device=self.device)
        for i_step in range(n_steps):
            out = self.forward(
                prefix_ids,
                position_ids,
                attention_mask,
                temperature,
                top_k,
                *pasts,
            )

            prefix_ids = out[0]
            attention_mask = out[1]
            output_ids[:, i_step] = prefix_ids.squeeze()
            pasts = out[2:]

        return output_ids.cpu().numpy()
