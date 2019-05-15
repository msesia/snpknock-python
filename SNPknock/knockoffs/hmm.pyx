#  This file is part of SNPknock.
#
#    Copyright (C) 2017-2019 Matteo Sesia
#
#    SNPknock is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    SNPknock is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with SNPknock.  If not, see <http://www.gnu.org/licenses/>.

# distutils: language=c++
# distutils: sources=src/knockoffs/hmm.cpp

# Cython interface file for wrapping the object
#
#
from libcpp.vector cimport vector
from src.knockoffs.dmc cimport KnockoffDMC
import numpy as np

# c++ interface to cython
cdef extern from "hmm.h" namespace "knockoffs":
    cdef cppclass KnockoffHMM:
        KnockoffHMM(vector[double] initP, vector[vector[vector[double]]] Q, \
                    vector[vector[vector[double]]] emissionP, vector[int] groups, seed) except +
        vector[int] sample(vector[int] X)

# creating a cython wrapper class
cdef class knockoffHMM:
    cdef KnockoffHMM *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, initP, Q, emissionP, groups, seed):
        self.thisptr = new KnockoffHMM(initP, Q, emissionP, groups, seed)
    def __dealloc__(self):
        del self.thisptr
    def sample(self, X):
        return self.thisptr.sample(X)
