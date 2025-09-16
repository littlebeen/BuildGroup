from setuptools import setup
import torch
torch.cuda.device_count()
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
print(torch.cuda.is_available()) 
print(os.environ['CUDA_HOME'])
if __name__ == '__main__':
    setup(
        name='softgroup',
        version='1.0',
        description='SoftGroup: SoftGroup for 3D Instance Segmentation [CVPR 2022]',
        author='Thang Vu',
        author_email='thangvubk@kaist.ac.kr',
        packages=['softgroup'],
        package_data={'softgroup.ops': ['*/*.so']},
        ext_modules=[
            CUDAExtension(
                name='softgroup.ops.ops',
                sources=[
                    'softgroup/ops/src/softgroup_api.cpp', 'softgroup/ops/src/softgroup_ops.cpp',
                    'softgroup/ops/src/cuda.cu'
                ],
                extra_compile_args={
                    'cxx': ['-g'],
                    'nvcc': ['-O2']
                })
        ],
        cmdclass={'build_ext': BuildExtension})


# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# if __name__ == "__main__":
#     setup(
#         name="spherical_mask",
#         version="1.0",
#         description="spherical_mask",
#         author="sangyun shin",
#         packages=["spherical_mask"],
#         package_data={"spherical_mask.ops": ["*/*.so"]},
#         ext_modules=[
#             CUDAExtension(
#                 name="spherical_mask.ops.ops",
#                 sources=[
#                     "spherical_mask/ops/src/isbnet_api.cpp",
#                     "spherical_mask/ops/src/isbnet_ops.cpp",
#                     "spherical_mask/ops/src/cuda.cu",
#                 ],
#                 extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
#             )
#         ],
#         cmdclass={"build_ext": BuildExtension},
#     )