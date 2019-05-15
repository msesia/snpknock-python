# This file is part of SNPknock.
#
#     Copyright (C) 2017-2019 Matteo Sesia
#
#     SNPknock is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     SNPknock is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with SNPknock.  If not, see <http://www.gnu.org/licenses/>.

# distutils: language = c++
# distutils: sources = SNPknock/knockoffs/dmc.cpp SNPknock/knockoffs/hmm.cpp SNPknock/knockoffs/genotypes.cpp SNPknock/knockoffs/haplotypes.cpp SNPknock/knockoffs/utils.cpp

# Cython interface file for wrapping the object
#
#
from libcpp.vector cimport vector
import numpy as np
import cython

# c++ interface to cython
cdef extern from "knockoffs/dmc.h":
    cdef cppclass KnockoffDMC:
        KnockoffDMC(vector[double] pInit, vector[vector[vector[double]]] Q, \
                    vector[int] groups, int seed) except +
        vector[int] sample(vector[int] X)
        vector[vector[int]] sample(vector[vector[int]] X)

# c++ interface to cython
cdef extern from "knockoffs/hmm.h":
    cdef cppclass KnockoffHMM:
        KnockoffHMM(vector[double] pInit, vector[vector[vector[double]]] Q, vector[vector[vector[double]]] pEmit, \
                    vector[int] groups, int seed) except +
        vector[int] sample(vector[int] X)
        vector[vector[int]] sample(vector[vector[int]] X)

# c++ interface to cython
cdef extern from "knockoffs/genotypes.h":
    cdef cppclass KnockoffGenotypes:
        KnockoffGenotypes(vector[double] r, vector[vector[double]] alpha, vector[vector[double]] theta, \
                          vector[int] groups, int seed) except +
        vector[int] sample(vector[int] X)
        vector[vector[int]] sample(vector[vector[int]] X)

# c++ interface to cython
cdef extern from "knockoffs/haplotypes.h":
    cdef cppclass KnockoffHaplotypes:
        KnockoffHaplotypes(vector[double] r, vector[vector[double]] alpha, vector[vector[double]] theta, \
                           vector[int] groups, int seed) except +
        vector[int] sample(vector[int] X)
        vector[vector[int]] sample(vector[vector[int]] X)

# creating a cython wrapper class
@cython.embedsignature(True)
cdef class knockoffDMC:
    """
    Class for knockoffs of a discrete Markov chain.

    :param pInit: a numpy array of length K, containing the marginal distribution of the
                  states for the first variable.
    :param Q: a numpy array of size (p-1,K,K), containing a list of p-1 transition matrices
              between the K states of the Markov chain.
    :param groups: a numpy array of length p, describing the group membership of each variable (default: [1,2,...,p]).
    :param seed: an integer random seed (default: 123).

    """
    cdef KnockoffDMC *thisptr      # hold a C++ instance which we're wrapping
    cdef vector[double] pInit
    cdef vector[vector[vector[double]]] Q
    cdef vector[int] groups
    cdef int seed
    def __cinit__(self, pInit, Q, groups=None, seed=123):
        if(groups is None):
            p = Q.shape[0]+1
            groups = np.arange(p)
        self.thisptr = new KnockoffDMC(pInit, Q, groups, seed)
        self.pInit = pInit
        self.Q = Q

    def sample(self, X):
        """
        Sample a knockoff copy of each row of X.

        :param X: a numpy array of size (n,p), where n is the number of individuals and p
                  is the number of variables, containing the original Markov chain variables.
                  The entries of X must be integers ranging from 0 to K-1, where K is the
                  number of possible states of the Markov chain.

        :returns: a numpy array of size (n,p), containing a knockoff copy of X.

        """
        if X.ndim == 1:
            return np.array(self.thisptr.sample(<vector[int]> X))
        if X.ndim == 2:
            return np.array(self.thisptr.sample(<vector[vector[int]]> X))

    def __dealloc__(self):
        del self.thisptr

    def __reduce__(self):
        # a tuple as specified in the pickle docs - (class_or_constructor, (tuple, of, args, to, constructor))
        return (self.__class__, (self.pInit, self.Q))

# creating a cython wrapper class
@cython.embedsignature(True)
cdef class knockoffHMM:
    """
    Class for knockoffs of a hidden Markov model.

    :param pInit: a numpy array of length K, containing the marginal distribution of the hidden
                  states for the first variable.
    :param Q: a numpy array of size (p-1,K,K), containing a list of p-1 transition matrices
              between the K latent states of the HMM.
    :param pEmit: a numpy array of size (p,M,K), containing the emission probabilities for
                  each of the M possible emission states, from each of the K hidden states
                  and the p variables.
    :param groups: a numpy array of length p, describing the group membership of each variable (default: [1,2,...,p]).
    :param seed: an integer random seed (default: 123).

    """
    cdef KnockoffHMM *thisptr      # hold a C++ instance which we're wrapping
    cdef vector[double] pInit
    cdef vector[vector[vector[double]]] Q
    cdef vector[vector[vector[double]]] pEmit
    cdef vector[int] groups
    cdef int seed
    def __cinit__(self, pInit, Q, pEmit, groups=None, seed=123):
        if(groups is None):
            p = Q.shape[0]+1
            groups = np.arange(p)
        self.thisptr = new KnockoffHMM(pInit, Q, pEmit, groups, seed)
        self.pInit = pInit
        self.Q = Q
        self.pEmit = pEmit

    def sample(self, X):
        """
        Samples a knockoff copy of each row of X.

        :param X: a numpy array of size (n,p), where n is the number of individuals and p
                  is the number of variables, containing the original HMM variables
                  The entries of X must be integers ranging from 0 to M-1, where M is the
                  number of possible emission states of the HMM.

        :returns: a numpy array of size (n,p), containing a knockoff copy of X.

        """
        if X.ndim == 1:
            return np.array(self.thisptr.sample(<vector[int]> X))
        if X.ndim == 2:
            return np.array(self.thisptr.sample(<vector[vector[int]]> X))

    def __dealloc__(self):
        del self.thisptr

    def __reduce__(self):
        # a tuple as specified in the pickle docs - (class_or_constructor, (tuple, of, args, to, constructor))
        return (self.__class__, (self.pInit, self.Q, self.pEmit))

# creating a cython wrapper class
@cython.embedsignature(True)
cdef class knockoffGenotypes:
    """
    Class for knockoffs of a hidden Markov model.

    :param pInit: a numpy array of length K, containing the marginal distribution of the hidden
                  states for the first variable.
    :param Q: a numpy array of size (p-1,K,K), containing a list of p-1 transition matrices
              between the K latent states of the Genotypes.
    :param pEmit: a numpy array of size (p,M,K), containing the emission probabilities for
                  each of the M possible emission states, from each of the K hidden states
                  and the p variables.
    :param groups: a numpy array of length p, describing the group membership of each variable (default: [1,2,...,p]).
    :param seed: an integer random seed (default: 123).

    """
    cdef KnockoffGenotypes *thisptr      # hold a C++ instance which we're wrapping
    cdef vector[double] r
    cdef vector[vector[double]] alpha
    cdef vector[vector[double]] theta
    cdef vector[int] groups
    cdef int seed
    def __cinit__(self, r, alpha, theta, groups=None, seed=123):
        if(groups is None):
            p = alpha.shape[0]
            groups = np.arange(p)
        self.thisptr = new KnockoffGenotypes(r, alpha, theta, groups, seed)
        self.r = r
        self.alpha = alpha
        self.theta = theta

    def sample(self, X):
        """
        Samples a knockoff copy of each row of X.

        :param X: a numpy array of size (n,p), where n is the number of individuals and p
                  is the number of variables, containing the original Genotypes variables
                  The entries of X must be integers ranging from 0 to M-1, where M is the
                  number of possible emission states of the Genotypes.

        :returns: a numpy array of size (n,p), containing a knockoff copy of X.

        """
        if X.ndim == 1:
            return np.array(self.thisptr.sample(<vector[int]> X))
        if X.ndim == 2:
            return np.array(self.thisptr.sample(<vector[vector[int]]> X))

    def __dealloc__(self):
        del self.thisptr

    def __reduce__(self):
        # a tuple as specified in the pickle docs - (class_or_constructor, (tuple, of, args, to, constructor))
        return (self.__class__, (self.r, self.alpha, self.theta))

# creating a cython wrapper class
@cython.embedsignature(True)
cdef class knockoffHaplotypes:
    """
    Class for knockoffs of a hidden Markov model.

    :param pInit: a numpy array of length K, containing the marginal distribution of the hidden
                  states for the first variable.
    :param Q: a numpy array of size (p-1,K,K), containing a list of p-1 transition matrices
              between the K latent states of the Haplotypes.
    :param pEmit: a numpy array of size (p,M,K), containing the emission probabilities for
                  each of the M possible emission states, from each of the K hidden states
                  and the p variables.
    :param groups: a numpy array of length p, describing the group membership of each variable (default: [1,2,...,p]).
    :param seed: an integer random seed (default: 123).

    """
    cdef KnockoffHaplotypes *thisptr      # hold a C++ instance which we're wrapping
    cdef vector[double] r
    cdef vector[vector[double]] alpha
    cdef vector[vector[double]] theta
    cdef vector[int] groups
    cdef int seed
    def __cinit__(self, r, alpha, theta, groups=None, seed=123):
        if(groups is None):
            p = alpha.shape[0]
            groups = np.arange(p)
        self.thisptr = new KnockoffHaplotypes(r, alpha, theta, groups, seed)
        self.r = r
        self.alpha = alpha
        self.theta = theta

    def sample(self, X):
        """
        Samples a knockoff copy of each row of X.

        :param X: a numpy array of size (n,p), where n is the number of individuals and p
                  is the number of variables, containing the original Haplotypes variables
                  The entries of X must be integers ranging from 0 to M-1, where M is the
                  number of possible emission states of the Haplotypes.

        :returns: a numpy array of size (n,p), containing a knockoff copy of X.

        """
        if X.ndim == 1:
            return np.array(self.thisptr.sample(<vector[int]> X))
        if X.ndim == 2:
            return np.array(self.thisptr.sample(<vector[vector[int]]> X))

    def __dealloc__(self):
        del self.thisptr

    def __reduce__(self):
        # a tuple as specified in the pickle docs - (class_or_constructor, (tuple, of, args, to, constructor))
        return (self.__class__, (self.r, self.alpha, self.theta))
