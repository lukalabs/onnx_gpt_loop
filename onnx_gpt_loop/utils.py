import torch


def get_dummy_pasts(
        batch_size,
        seq_len,
        hidden_size,
        num_attention_heads,
        num_hidden_layers,
        device,
        dtype=torch.float32
):
    head_size = int(hidden_size / num_attention_heads)
    shape = (batch_size, num_attention_heads, seq_len, head_size)

    past_key_values = []
    for _ in range(num_hidden_layers):
        _past_key_values = []
        for _ in range(2):
            _past_key_values.append(torch.zeros(
                shape,
                device=device,
                dtype=dtype,
            ))
        past_key_values.append(torch.stack(_past_key_values))

    return past_key_values
