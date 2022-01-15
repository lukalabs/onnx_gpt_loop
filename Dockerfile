FROM nvcr.io/nvidia/pytorch:21.12-py3

RUN pip install --no-cache-dir nvidia-pyindex==1.0.9

RUN mkdir onnx_gpt_loop
COPY requirements.txt setup.py onnx_gpt_loop/
COPY onnx_gpt_loop onnx_gpt_loop/onnx_gpt_loop
COPY scripts onnx_gpt_loop/scripts
RUN pip install -r onnx_gpt_loop/requirements.txt
RUN pip install -U -e onnx_gpt_loop

WORKDIR onnx_gpt_loop
CMD ["/bin/bash"]
