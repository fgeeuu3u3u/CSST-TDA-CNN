'''
Author: xin zhangxinbjfu@gmail.com
Date: 2022-04-29 16:13:01
LastEditors: xin zhangxinbjfu@gmail.com
LastEditTime: 2022-07-05 15:33:36
FilePath: /undefined/Users/zhangxin/Work/SlitlessSim/sls_lit_demo/setup.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from setuptools import setup, find_packages

from setuptools.extension import Extension
from setuptools.config import read_configuration

from Cython.Build import cythonize

import numpy

extensions = [
    Extension("SpecDisperser.disperse_c.interp", ["SpecDisperser/disperse_c/interp.pyx"],
        include_dirs = [numpy.get_include()],
        libraries=["m"]),

    Extension("SpecDisperser.disperse_c.disperse", ["SpecDisperser/disperse_c/disperse.pyx"],
        include_dirs = [numpy.get_include()],
        libraries=["m"]),
]


version='0.5'

setup(
    name = "SpecDisperser",
    version = version,
    description = "Slitless spec disperse",

    packages=['SpecDisperser', 'SpecDisperser/disperse_c'],
    install_requires=[
        'galsim',
        'numpy',
        'scipy',
        'astropy',
    ],
    ext_modules = cythonize(extensions),

)


