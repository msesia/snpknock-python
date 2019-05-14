import numpy as np
from matplotlib import pyplot as plt
from SNPknock.fastphase import loadHMM
from SNPknock import models
from SNPknock import knockoffHMM, knockoffHaplotypes, knockoffGenotypes
import util, pdb

# Load HMM
r_file = "data/haplotypes_rhat.txt"
alpha_file = "data/haplotypes_alphahat.txt"
theta_file = "data/haplotypes_thetahat.txt"
char_file = "data/haplotypes_origchars"
hmm = loadHMM(r_file, alpha_file, theta_file, char_file, compact=True)
hmm_full = loadHMM(r_file, alpha_file, theta_file, char_file, compact=False)

# Sample X
n=10
modelX = models.HMM(hmm_full['pInit'], hmm_full['Q'], hmm_full['pEmit'])
X = modelX.sample(n)

# Generate the knockoffs
#knockoffs = knockoffHMM(hmm_full['pInit'], hmm_full['Q'], hmm_full['pEmit'])
#Xk = knockoffs.sample(X)
#print("Generated knockoffs")

# Generate the knockoffs (genotypes)
knockoffs_gen = knockoffGenotypes(hmm['r'], hmm['alpha'], hmm['theta'])
Xk = knockoffs_gen.sample(X)
print("Generated knockoffs (genotypes)")

# Generate the knockoffs (haplotypes)
H = X
H[X==2] = 1
knockoffs_hap = knockoffHaplotypes(hmm['r'], hmm['alpha'], hmm['theta'])
Hk = knockoffs_hap.sample(H)
print("Generated knockoffs (haplotypes)")

pdb.set_trace()
