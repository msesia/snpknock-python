# distutils: language=c++
# distutils: sources=src/knockoffsMF/dmc_knock.cpp

# Cython interface file for wrapping the object
#
#
from libcpp.vector cimport vector

# c++ interface to cython
cdef extern from "dmc_knock.h" namespace "knockoffs":
    cdef cppclass KnockoffDMC:
        KnockoffDMC(vector[double] initP, vector[vector[vector[double]]] Q) except +
        vector[int] sample(vector[int] X)
