import abc


class HasGenerationLoop(abc.ABC):
    @abc.abstractmethod
    def generate(self, n_steps, prefix_ids, temperature, top_k):
        pass
