from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ncompass_ext",
    packages=["ncompass_ext"],
    ext_modules=[CUDAExtension("ncompass_ext._C", ["csrc/scale.cu"])],
    cmdclass={"build_ext": BuildExtension},
)
