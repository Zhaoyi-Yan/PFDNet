from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
setup(name='pad_conv2d',
      ext_modules=[CUDAExtension('pad_conv2d_gpu', ['pad_conv2d.cpp', 'pad_conv2d_cuda.cu']),],
      cmdclass={'build_ext': BuildExtension})
