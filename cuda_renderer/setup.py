# -*- coding: utf-8 -*-
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

module_name = 'cuda_renderer'
root = Path(__file__).resolve().parent

setup(
    name=module_name,
    version='1.0.0',
    description='CUDA-accelerated Point Cloud Splatting Renderer',
    packages=[module_name], 
    ext_modules=[
        CUDAExtension(
            name='{}.{}'.format(module_name, '_splat_cuda'),
            sources=[
                str(root / "splat_binding.cpp"),
                str(root / "splat_cuda_kernel.cu"),
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
