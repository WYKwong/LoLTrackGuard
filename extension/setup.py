import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Note: You must adjust these paths to match your local OpenCV installation
# Example paths (user must verify):
opencv_include = r"C:\opencv\build\include"
opencv_lib_dir = r"C:\opencv\build\x64\vc16\lib"
opencv_libs = ["opencv_world480"] # Adjust version number, e.g. 480 = 4.8.0

setup(
    name='lol_accelerator',
    ext_modules=[
        CUDAExtension(
            name='lol_accelerator',
            sources=[
                'src/bindings.cpp',
                'src/video_loader.cpp',
                'src/kernels.cu',
            ],
            include_dirs=[opencv_include],
            library_dirs=[opencv_lib_dir],
            libraries=opencv_libs,
            extra_compile_args={
                'cxx': ['/O2'], 
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

