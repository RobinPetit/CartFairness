# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import pandas as pd

np_include_path = "C:/Users/Admin/.conda/envs/PhD_TM/lib/site-packages/numpy/core/include" # from numpy.get_include()
pandas_include_path = "C:/Users/Admin/.conda/envs/PhD_TM/lib/site-packages/pandas/_libs/tslibs"#/package/include"

# Define the extension module
extensions = [
    Extension(
        name="version_simple_cython21_adj_categorical_clean_epsilon",
        sources=["version_simple_cython21_adj_categorical_clean_epsilon.pyx"],
        #name="version_simple_cython21_adj_categorical_clean_epsilon",
        #sources=["version_simple_cython21_adj_categorical_clean_epsilon.pyx"],
        include_dirs=[numpy.get_include(), pandas_include_path],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"]
        # Specify NumPy include path
    )
]

setup(
    ext_modules = cythonize(extensions)
)