# This file is part of SNPknock.

#     Copyright (C) 2017-2019 Matteo Sesia

#     SNPknock is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     SNPknock is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with SNPknock.  If not, see <http://www.gnu.org/licenses/>.

#!/usr/bin/env python3
# distutils: language = c++

CLASSIFIERS = ["Development Status :: 4 - Beta",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

DESCRIPTION = 'Generates knockoffs of HMMs and genetic data.'

NUMPY_MIN_VERSION = '1.13.1'
CYTHON_MIN_VERSION = '0.26'

REQUIRES            = ["numpy (>=%s)" % NUMPY_MIN_VERSION,
                       "Cython (>=%s)" % CYTHON_MIN_VERSION]

from setuptools import Extension
from setuptools import setup
import sys

# we'd better have Cython installed, or it's a no-go
try: 
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except:
    print("You don't seem to have Cython installed. Please get a")
    print("copy from www.cython.org and install it")
    sys.exit(1)

# Compiler flags
EXTRA_COMPILE_ARGS = ["-O3", "-std=c++11"]
EXTRA_LINK_ARGS = ["-static-libstdc++", "-larmadillo"]
UNDEF_MACROS = [ "NDEBUG" ]

# Define extensions
ext_modules=[
    Extension("SNPknock.knockoffs", ["SNPknock/knockoffs.pyx"], \
              undef_macros=UNDEF_MACROS,extra_compile_args=EXTRA_COMPILE_ARGS,extra_link_args=EXTRA_LINK_ARGS)
]

# By setting this compiler directive, cython will embed signature information 
# in docstrings. Sphinx then knows how to extract and use those signatures.
for e in ext_modules:
    e.cython_directives = {"embedsignature": True}

# Speficy dependencies
DEPENDENCIES = ['Cython>='+CYTHON_MIN_VERSION,
                'numpy>='+NUMPY_MIN_VERSION]

################ COMPILE

def main(**extra_args):
    setup(name='SNPknock',
          maintainer="Matteo Sesia",
          maintainer_email="msesia@stanford.edu",
          description=DESCRIPTION,
          url="https://github.com/msesia/snpknock-python",
          license="GPL-v3 license",
          classifiers=CLASSIFIERS,
          author="Matteo Sesia",
          author_email="msesia@stanford.edu",
          platforms="OS Independent",
          version='0.8.3',
          requires=REQUIRES,
          provides=["SNPknock"],
          packages     = ['SNPknock',
                         ],
          ext_modules = cythonize(ext_modules),
          package_data = {},
          data_files=[],
          scripts= [],
          long_description = open('README.rst', 'rt').read(),
          install_requires = DEPENDENCIES,
          setup_requires=["pytest-runner"],
          tests_require=["pytest"],
    )

#simple way to test what setup will do
#python setup.py install --prefix=/tmp
if __name__ == "__main__":
    main()
