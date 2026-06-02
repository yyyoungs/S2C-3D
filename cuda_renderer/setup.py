# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

module_name = 'cuda_renderer'

setup(
    name=module_name,
    version='1.0.0',
    description='CUDA-accelerated Point Cloud Splatting Renderer',
    packages=[module_name], 
    ext_modules=[
        CUDAExtension(
            name='{}.{}'.format(module_name, '_splat_cuda'),
            sources=[
                "splat_binding.cpp",
                "splat_cuda_kernel.cu",
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)