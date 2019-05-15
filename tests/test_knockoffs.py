import numpy as np
from SNPknock import models, knockoffDMC, knockoffHMM
import unittest
from unittest import TestCase

def generate_DMC(p,K):
    Q = np.zeros((p-1,K,K)) # Transition matrices
    for j in range(p-1):
        Q[j,:,:] = np.resize(np.random.uniform(size=K*K),(K,K))
        Q[j,:,:] /= np.sum(Q[j,:,:],1)[:,None]
    pInit = np.array([1.0/K]*K) # Initial distribution
    return pInit, Q

def generate_HMM(p,K,M):
    Q = np.zeros((p-1,K,K)) # Transition matrices
    for j in range(p-1):
        Q[j,:,:] = np.resize(np.random.uniform(size=K*K),(K,K))
        Q[j,:,:] += np.diag([10]*K)
        Q[j,:,:] /= np.sum(Q[j,:,:],1)[:,None]
    pEmit = np.zeros((p,M,K)) # Emission probabilities
    for j in range(p):
        pEmit[j,:,:] = np.resize(np.random.uniform(size=M*K),(M,K))
        pEmit[j,:,:] /= np.sum(pEmit[j,:,:],0)
    pInit = np.array([1.0/K]*K) # Initial distribution
    return pInit, Q, pEmit

class TestDMC(TestCase):
    def test_DMC(self):
        p = 50
        K = 4
        n = 100
        pInit, Q = generate_DMC(p,K)
        modelX = models.DMC(pInit, Q)
        X = modelX.sample(n)
        knockoffs = knockoffDMC(pInit, Q, seed=123)
        Xk = knockoffs.sample(X)
        groups = np.arange(p)
        knockoffs_g = knockoffDMC(pInit, Q, groups=groups, seed=123)
        Xk_g = knockoffs_g.sample(X)
        self.assertTrue(np.array_equal(Xk, Xk_g))
        self.assertTrue(np.isfinite(Xk).all())

class TestHMM(TestCase):
    def test_HMM(self):
        p = 50
        K = 4
        M = 5
        n = 100
        pInit, Q, pEmit = generate_HMM(p,K,M)
        modelX = models.HMM(pInit, Q, pEmit)
        X = modelX.sample(n)
        knockoffs = knockoffHMM(pInit, Q, pEmit, seed=123)
        Xk = knockoffs.sample(X)
        groups = np.arange(p)
        knockoffs_g = knockoffHMM(pInit, Q, pEmit, groups=groups, seed=123)
        Xk_g = knockoffs_g.sample(X)
        self.assertTrue(np.array_equal(Xk, Xk_g))
        self.assertTrue(np.isfinite(Xk).all())

if __name__ == '__main__':
    unittest.main()
