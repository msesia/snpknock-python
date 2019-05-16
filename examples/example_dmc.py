import numpy as np
from matplotlib import pyplot as plt
from SNPknock import knockoffDMC
from SNPknock import models
import util, pdb

# Initialize a random DMC
p = 50 # Number of variables
K = 4  # Number of states
Q = np.zeros((p-1,K,K))
for j in range(p-1):
    Q[j,:,:] = np.resize(np.random.uniform(size=K*K),(K,K))
    Q[j,:,:] += np.diag([10]*K)
    Q[j,:,:] /= np.sum(Q[j,:,:],1)[:,None]
pInit = np.zeros((K,))
pInit[0] = 1

# Sample X
n=10000
modelX = models.DMC(pInit, Q)
X = modelX.sample(n)

# Generate the knockoffs
knockoffs = knockoffDMC(pInit, Q)
Xk = knockoffs.sample(X)

# Plot paths
util.plotPaths(X,Xk)

# Compare original variables and knockoffs 
util.compare_marginals(X,Xk)
util.compare_cons_corr(X,Xk)
util.compare_cross_corr(X,Xk)
util.compare_cross_corr(X,Xk,dist=0)
