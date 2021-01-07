from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
setup(name='adaptive_sigmoid', ext_modules=[CUDAExtension('adaptive_sigmoid_gpu',['adaptive_sigmoid.cpp', 'adaptive_sigmoid_cuda.cu']),], cmdclass={'build_ext': BuildExtension})