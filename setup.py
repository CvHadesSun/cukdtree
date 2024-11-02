import os
import glob

if os.name == "nt":
    os.environ["PATH"] += r';D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.32.31326\bin\Hostx64\x64'
# os.environ["NVCC_APPEND_FLAGS"] = "--allow-unsupported-compiler"
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_src_path = os.path.dirname(os.path.abspath(__file__))

nvcc_flags = [
    '-O3', '-use_fast_math', '-std=c++17'
]

c_flags = ['-O3', '-std=c++17']
dir_sources = glob.glob(os.path.join(_src_path, 'cukd', '**', '*'), recursive=True)

# Filter to include only `.cpp` and `.cu` files
dir_sources = [f for f in dir_sources if f.endswith(('.cpp', '.cu'))]
# dir_sources=[]

# Include additional files explicitly
sources = dir_sources + [
    os.path.join(_src_path, 'warpper', 'bind.cu'),
    os.path.join(_src_path, 'warpper', 'cukd_tree.cu'),
]


setup(
    name='cuda_kdtree',  # package name, import this to use python API
    ext_modules=[
        CUDAExtension(
            name='cuda_kdtree',  # extension name, import this to use CUDA API
            sources=sources,
            include_dirs=[os.path.join(_src_path, '')], 
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    }
    
)
