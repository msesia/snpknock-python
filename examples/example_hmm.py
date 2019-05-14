import numpy as np
from matplotlib import pyplot as plt
from SNPknock import knockoffHMM
from SNPknock import models
import util, pdb

# Initialize a random HMM
p = 100 # Number of variables
K = 5   # Number of hidden states
M = 5   # Number of emission states
Q = np.zeros((p-1,K,K))
pEmit = np.zeros((p,M,K))
gamma = np.random.uniform(low=0, high=10, size=p)
for j in range(p-1):    
    Q[j,:,:] = np.resize(np.random.uniform(size=K*K),(K,K))
    Q[j,:,:] += np.diag([gamma[j]]*K) 
    Q[j,:,:] /= np.sum(Q[j,:,:],1)[:,None]
for j in range(p):
    pEmit[j,:,:] = np.resize(np.random.uniform(size=M*K),(M,K))    
    pEmit[j,:,:] += np.diag([gamma[j]]*K) 
    pEmit[j,:,:] /= np.sum(pEmit[j,:,:],0)
pInit = np.zeros((K,))
pInit[0] = 1

# Sample X
n=10000
modelX = models.HMM(pInit, Q, pEmit)
X = modelX.sample(n)

# Generate the knockoffs
knockoffs = knockoffHMM(pInit, Q, pEmit)
Xk = knockoffs.sample(X)

# Plot paths
util.plotPaths(X,Xk)

# Compare original variables and knockoffs 
util.compare_marginals(X,Xk)
util.compare_cons_corr(X,Xk)
util.compare_cross_corr(X,Xk)
util.compare_cross_corr(X,Xk,dist=0)
