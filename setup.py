import pathlib

import onnx_gpt_loop
from setuptools import setup

_THIS_DIR = pathlib.Path(__file__).parent


def _get_requirements():
    with (_THIS_DIR / 'requirements.txt').open() as fp:
        return fp.read()


setup(
    name='onnx_gpt_loop',
    version=onnx_gpt_loop.__version__,
    install_requires=_get_requirements(),
    package_dir={'onnx_gpt_loop': 'onnx_gpt_loop'},
)
