# ONNX GPT Loop
Export GPT2 full inference loop to the single ONNX graph.

Such conversion increases inference speed by 30% with compared with pure PyTorch
model with cached past-key-value. It allows you to generate a full sequence in
one call via your favorite runtime or inference engine: onnxruntime, triton, etc. 

[Quick Start](#quick-start)

[Benchmark](#benchmark)

[How it Works](#how-it-works)

## Quick Start
Start a container:
```
git clone https://github.com/alexeykarnachev/onnx_gpt_loop.git && \
cd onnx_gpt_loop && \
docker build -t onnx_gpt_loop . && \
docker run --name onnx_gpt_loop -d -it --rm --gpus all --network host onnx_gpt_loop
```

Enter the container:
```
docker exec -it onnx_gpt_loop bash
```

Export pre-trained gpt model to the optimized ONNX loop model
(for large models it will take 3-5 minutes):
```
python scripts/export_as_loop_model.py -m gpt2 -f ./loop.onnx
```

Now you can generate text with this model:
```python
import numpy as np
from transformers import GPT2TokenizerFast

from onnx_gpt_loop.models.loop_onnx import LoopOnnxModel

model = LoopOnnxModel('./loop.onnx')

n_samples = 5
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
prefix_text = "We are consumers. We're the by-products of a lifestyle obsession."
prefix_ids = [tokenizer.encode(prefix_text) for _ in range(n_samples)]
prefix_ids = np.array(prefix_ids, dtype=np.int64)

output_ids = model.generate(n_steps=10, prefix_ids=prefix_ids, temperature=0.7, top_k=50)

print(prefix_text)
for i, ids in enumerate(output_ids):
    text = tokenizer.decode(ids)
    print(f' [{i}] {text}')
```

```
We are consumers. We're the by-products of a lifestyle obsession.
 [0]  We're the products that are really on offer to
 [1]  And we're in the first place.
 [2]  For many of us, that means getting a new
 [3]  We're the result of a series of brainwashing
 [4]  We're not just a commodity or a food.
```

## Benchmark
Models were benchmarked on RTX3090.
Here we measure time for full inference generation loop, i.e generation of 36 tokens given 36 prefix
tokens for 64 candidates.

### gpt2:
```
python scripts/benchmark.py --model-name gpt2 --batch-size 64 --prefix-seq-len 36 --n-generation-steps 36
```
```
Torch: 0.2250s
ONNX: 0.1265s
```

### gpt2-medium:
```
python scripts/benchmark.py --model-name gpt2-medium --batch-size 64 --prefix-seq-len 36 --n-generation-steps 36
```
```
Torch: 0.4399s
ONNX: 0.2616s
```

### gpt2-large:
```
python scripts/benchmark.py --model-name gpt2-large --batch-size 64 --prefix-seq-len 36 --n-generation-steps 36
```
```
Torch: 0.6604s
ONNX: 0.4434s
```

## How it Works
### Difficulties with standard approaches
The main difficulty of the exporting GPT loop to the ONNX graph is the fact that ONNX is an acyclic graph.
So, there is no easy way to naively trace a loop with past-key-values caching, since the reuse of cached
values on the next iteration will form a graph loop. 

Instead of model tracing, ONNX can utilize a model scripting procedure. It is a more robust way to convert
Pytorch model with non-linear control flow to the ONNX graph. But scripting requires detailed
code hinting and typing to correctly transform python representation to the intermediate torch-script
representation. The huggingface transformers implementation is not suitable for the correct scripting.

Another way is to trace (or script) an inference loop is firstly to trace the GPT backbone (single-step model)
and then wrap this traced model to the python scriptable loop. Pytorch has official examples for this
approach: [Mixing Tracing and Scripting](https://pytorch.org/docs/stable/jit.html#mixing-tracing-and-scripting).
For some reasons, the resulting model became not compatible with onnxruntime GPT optimizer. So, you can perform
inference of this hybrid traced-scripted model, but without onnxruntime optimizations (including fp16 conversion)
the performance will be very poor.

### What can we do else?
First, let's export the GPT backbone (single-step model) to the ONNX and optimize it with onnxruntime GPT optimizer.
It can be done by function: `onnx_gpt_loop.onnx_export.export_one_step_model`.
At this point, we'll have fast and optimized GPT backbone.

At the next step, we can wrap this optimized graph into the ONNX Loop Operator.
Accordingly to the official ONNX Operators [specification](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Loop),
the Loop node has the body attribute:

```
body : graph (required)
The graph run each iteration. It has 2+N inputs: (iteration_num, condition, loop carried dependencies...). It has 1+N+K outputs: (condition, loop carried dependencies..., scan_outputs...). Each scan_output is created by concatenating the value of the specified output value at the end of each iteration of the loop. It is an error if the dimensions or data type of these scan_outputs change across loop iterations.
```

The body itself is a graph. And we already have one: an optimized GPT, exported from the previous step. But our GPT does not quite
meet the requirements. As they say, there are 2+N inputs to this graph: `[iteration_num, condition, loop carried dependencies...]`.
In our case `loop carried dependencies` are already present: `[input_ids, temperature, top_k, past_key_values]` and we just need to add
two more inputs: `[iteration_num, condition]`:
```python
# Loop and its graph:
loop_body.input.insert(0, cond)
loop_body.input.insert(0, i_step)
```
Part of `onnx_gpt_loop.onnx_export._extract_loop_body_and_graph_inputs` function.

Besides a graph attribute, the Loop node has inputs:
```
Inputs (2 - ∞)
M (optional) : I
    A maximum trip-count for the loop specified at runtime. Optional. Pass empty string to skip.
cond (optional) : B
    A boolean termination condition. Optional. Pass empty string to skip.
v_initial (variadic, heterogeneous) : V
    The initial values of any loop-carried dependencies (values that change across loop iterations)
```
We already have `v_initial`: there are our initial loop body values: `[input_ids, temperature, top_k, past_key_values]`
and we need to add two more: `M` is a total number of iterations (in our case it's `n_steps`) and `cond` which is just
a constant node, which contains `True` value:

```python
loop_node = helper.make_node(
    op_type='Loop',
    inputs=['n_steps', 'cond'] + loop_node_input_names,
    outputs=loop_node_output_names + ['all_output_ids_3d'],
    body=loop_body,
)
```
Part of `onnx_gpt_loop.onnx_export._make_loop_node` function.

As you can see, the Loop node also has outputs: `loop_node_output_names + ['all_output_ids_3d']`. To understand this
let's check an official description of the Loop node output values:

```
Outputs (1 - ∞)
v_final_and_scan_outputs (variadic, heterogeneous) : V
    Final N loop carried dependency values then K scan_outputs. Scan outputs must be Tensors.
```

So, here we must return from the Loop node all carried dependencies, which are just values of `[input_ids, temperature, top_k, past_key_values]` from the last iteration. And also, `scan_outputs`. These are accumulated generated token ids. It's worth noting that
the Loop node doesn't squeeze the `scan_output` at the end, so I named this output with `_3d` postfix for more clarity: `all_output_ids_3d`.

And the final graph which combines the optimized GPT graph (our loop body) and the loop node itself will look like this:

```python
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
    inputs=[n_steps] + graph_inputs,
    outputs=[all_output_ids],
)
```
Part of `onnx_gpt_loop.onnx_export._make_graph` function.

Where `cond_const_node` is just a constant `True` condition value which is
required by the ONNX Loop Node specification.

`squeeze_all_output_ids_3d_node` is a node which transforms generated token ids
from 3d to 2d shape (batch_size, n_steps).

That's it. The final Loop Node looks like [this](images/loop_node.png),
and the Loop Graph for 5 layers GPT [here](images/loop_graph.png). 











