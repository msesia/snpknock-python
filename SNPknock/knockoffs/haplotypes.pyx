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
# distutils: sources=src/knockoffs/haplotypes.cpp

# Cython interface file for wrapping the object
#
#

import numpy as np

# creating a cython wrapper class
cdef class knockoffHaplotypes:
    cdef knockoffHaplotypes *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, r, alpha, theta, groups, seed):
        self.thisptr = new knockoffHaplotypes(r, alpha, theta, groups, seed)
    def __dealloc__(self):
        del self.thisptr
    def sample(self, X):
        return self.thisptr.sample(X)
