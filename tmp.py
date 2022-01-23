from transformers import GPT2LMHeadModel
from onnxruntime.transformers.gpt2_helper import MyGPT2LMHeadModel


def post_process(result, num_layer):
    if isinstance(result[1][0], tuple) or isinstance(result[1][0], list):
        assert len(result[1]) == num_layer and len(result[1][0]) == 2
        #assert len(result[1][0][0].shape) == 4 and result[1][0][0].shape == result[1][0][1].shape
        present = []
        for i in range(num_layer):
            # Since transformers v4.*, past key and values are separated outputs.
            # Here we concate them into one tensor to be compatible with Attention operator.
            present.append(torch.cat((result[1][i][0].unsqueeze(0), result[1][i][1].unsqueeze(0)), dim=0))
        return (result[0], tuple(present))

    return result

def forward(self, input_ids, position_ids, attention_mask, *past):
    result = super().forward(input_ids,
                                position_ids=position_ids,
                                attention_mask=attention_mask,
                                past_key_values=past,
                                return_dict=False)
    return MyGPT2Model.post_process(result, self.config.n_layer)


MyGPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
