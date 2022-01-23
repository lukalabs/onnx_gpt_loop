import abc


class HasGenerationLoop(abc.ABC):
    @abc.abstractmethod
    def generate(self, n_steps, temperature, top_k, prefix_ids, attention_mask, position_ids):
        """Runs full GPT inference loop cycle.

        :param n_step: Number of tokens to be generated.
        :param temperature: Temperature of the tokens sampling distribution.
        :param top_k: Top-k sampling number of tokens.
        :param prefix_ids: Prefix token ids.
        :param attention_mask: Initial attention mask. It has the same shape as `prefix_ids`.
        :param position_ids: Initial position ids. It has the same shape as `prefix_ids`.

        :return: Numpy array of generated tokens with shape (batch_size, n_steps).
        """
        pass
