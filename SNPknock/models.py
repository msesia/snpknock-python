# This file is part of SNPknock.
#
#     Copyright (C) 2017-2019 Matteo Sesia
#
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

import numpy as np

def _random_weighted_samples(W):
    """
    Draw independent random samples from a discrete distibution.

    :param W: a numpy matrix of size (n,K). Each row of W contains the probabilities
              for each of the K possible states of the random variable.
    """

    W_rowSums = np.sum(W,axis=1)
    W_cdf = np.cumsum(W/W_rowSums[:,None],1)
    R = np.random.random_sample((W.shape[0],1))
    return np.sum(W_cdf<R,1)

class HMM:
    """
    Hidden Markov model with a discrete emission distribution.

   :param pInit: a numpy array of length K, containing the marginal distribution of the hidden
                 states for the first variable.
   :param Q: a numpy array of size (p,K,K), containing a list of p-1 transition matrices
             between the K latent states of the hidden Markov model.
   :param pEmit: a numpy array of size (p,M,K), containing the emission probabilities for
                 each of the M possible emission states, from each of the K hidden states
                 and the p variables.

    """

    def __init__(self, pInit, Q, pEmit):
        self.hiddenChain = DMC(pInit,Q)
        self.p = self.hiddenChain.p
        self.pEmit = pEmit
        self.nStates = pEmit.shape[1]
        self.name = "HMM"

    def sample(self, n=1):
        """
        Sample the n observations of the hidden Markov model.

        :param n: the number of observations to be sampled (default: 1).
        :return: a numpy matrix of size (n,p).

        """
        H = self.hiddenChain.sample(n)
        self.X = np.ones((n,self.p))*(-1)
        for j in range(self.p):
            pEmit = self.pEmit[j][:,H[:,j]]
            self.X[:,j] = _random_weighted_samples(pEmit.T)
        return self.X.astype(int)

class DMC:
    """
    Discrete Markov chain model.

    :param pInit: a numpy array of length K, containing the marginal distribution of the
                  first variable in the chain.
    :param Q: a numpy array of size (p,K,K), containing a list of p-1 transition matrices
              between the K states of the Markov chain.

    """

    def __init__(self, pInit, Q):
        assert Q.shape[0]>0, "The list of transition matrices must have non-zero length"
        assert Q.shape[2]==len(pInit), "The dimensions of Q and pInit are not compatible"
        self.Q = Q
        self.nStates = Q.shape[1]
        self.pInit = pInit
        self.p = Q.shape[0]+1
        self.name = "DMC"

    def sample(self, n=1):
        """
        Sample the observations from their marginal distribution.

        :param n: the number of observations to be sampled (default: 1).
        :return: a numpy matrix of size (n,p).

        """
        chain = np.zeros((n, self.p)).astype(int)
        chain[:,0] = _random_weighted_samples(np.tile(self.pInit, (n,1)))

        for t in range(1,self.p):
            W = self.Q[t-1,chain[:,t-1],:]
            chain[:,t] = _random_weighted_samples(W)
        return chain.astype(int)
