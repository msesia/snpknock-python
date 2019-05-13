# This file is part of SNPknock.
#
#     Copyright (C) 2017 Matteo Sesia
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

import numpy as np
import os

def runFastPhase(fp_path, X_file, out_path, K=12, numit=25, seed=1):
    """
    This function calls fastPhase to fit an HMM to the genotype data. FastPhase will fit the HMM
    to the genotype data and write the corresponding parameter estimates in three separate
    files named:
       * out_path + "_rhat.txt"
       * out_path + "_alphahat.txt"
       * out_path + "_thetahat.txt"

    The HMM for the genotype data can then be loaded from this files using
    :meth:`SNPknock.fastphase.loadFit`.

    :param fp_path: a string with the path to the directory with the fastPhase executable.
    :param X_file: a string with the path of the genotype input file containing X in fastPhase
                   format (as created by :meth:`SNPknock.fastphase.writeX`).
    :param out_path: a string with the path of the directory in which the parameter estimates
                     will be saved.
    :param K: the number of hidden states for each haplotype sequence (default: 12).
    :param numit: the number of EM iterations (default: 25).
    :param seed: the random seed for the EM algorithm (default: 1).

    """

    # Verify that the fastPhase executable can be found
    fp_path = os.path.abspath(fp_path)
    assert os.path.isfile(fp_path), "Could not find fastPhase executable.\nFile %s does not exist" %fp_path
    assert os.access(fp_path, os.X_OK), "Could not call fastPhase.\nFile %s is not executable" %fp_path

    # Prepare arguments for fastPhase
    command = fp_path
    command += " -Pp -T1 -K" + str(K)
    command += " -g -H-4 -C" + str(numit)
    command += " -S" + str(seed)
    command += " -o'" + out_path + "' " + X_file

    # Call fastPhase
    os.system(command)

def loadFit(r_file, theta_file, alpha_file, x):
    """
    Load the parameter estimates obtained by fastPhase and assembles the HMM model
    for the genotype data. For more information about fastPhase format see:
    http://scheet.org/software.html

    :param r_file: a string with the path of the "_rhat.txt" file produced by fastPhase.
    :param theta_file: a string with the path of the "_thetahat.txt" file produced by fastPhase.
    :param alpha_file: a string with the path of the "_alphahat.txt" file produced by fastPhase.
    :param x: a numpy array of length p, where p is the number of SNPs, containing the genotype
              sequence of the first individual in the dataset (the first individual is intended
              in the same order as provided to fastPhase). This is needed in order to correctly
              intepret the emission parameters estimated by fastPhase.
    :returns: a dictionary {'Q','pInit','pEmit'} where

        - Q is a numpy array of size (p-1,K,K), containing a list of p-1 transition matrices
          between the K latent states of the HMM.
        - pInit is a numpy array of length K, containing the marginal distribution of the hidden
          states for the first SNP.
        - pEmit is a numpy array of size (p,K,3), containing the emission probabilities of
          the hidden states for each of the p SNPs.

    """
    # Convert x to int
    x = x.astype(int)

    # Load parameter estimates from fastPhase output files
    r = _loadEMParameters(r_file)
    theta = np.matrix(_loadEMParameters(theta_file)).T
    alpha = np.matrix(_loadEMParameters(alpha_file)).T

    # Swap theta according to the correct definition based on the minor allele
    X_chr_flip = np.array(x>0).flatten()
    theta[:,X_chr_flip] = 1-theta[:,X_chr_flip]

    # Assemble the HMM variables according to the model assumed by fastPhase
    K,_ = theta.shape
    Q1 = _computeQ1(r, alpha)
    Q = _assembleQ(Q1)
    pInit = _assemblePInit(alpha)
    pEmit = _assembleEmissionP(theta)

    return {'pInit': pInit, 'Q':Q, 'pEmit':pEmit}

def _loadEMParameters(filepath):
    """
    .. document private functions
    """
    # Read parameter estimate file produced by fastPhase
    # Keep only estimates from the first EM start, if multiple starts are present
    num_EM = sum(line[0]=='>' for line in open(filepath))      # Count number of EM starts
    paramHat = np.loadtxt(filepath, comments='>')              # Load parameter estimates
    keep_rows = range(int(paramHat.shape[0]/num_EM))
    if len(paramHat.shape)==1:
        paramHat = paramHat[keep_rows]
    else:
        paramHat = paramHat[keep_rows,:]
    return paramHat

def _computeQ1(r, alpha):
    """
    .. document private functions
    """
    # Compute the transition matrix for a single haplotype
    K,p = alpha.shape
    Q = np.zeros((p-1,K,K))
    rExp = np.exp(-r)
    for j in range(1,p):
        Q1 = np.repeat( (1-rExp[j])*alpha[:,j].T, K, 0)
        Q1[np.diag_indices(K)] += rExp[j]
        Q[j-1,:,:] = Q1
    return Q

def _assembleQ(Q1):
    """
    .. document private functions
    """
    # Compute the transition matrix for both haplotypes, given the one for a single haplotype
    p = len(Q1)
    K,_ = Q1[0].shape
    Keff = int(K*(K+1)/2)
    Q = np.zeros((p,Keff,Keff))
    for k1 in range(K):
        for k2 in range(k1+1):
            i = int(((k1+1)*k1)/2+k2)
            for k1p in range(K):
                for k2p in range(k1p+1):
                    j = int(((k1p+1)*k1p)/2+k2p)
                    Q[:,i,j] = np.multiply(Q1[:,k1,k1p],Q1[:,k2,k2p])
                    if (k1p != k2p) : # Normalization seems correct, but different from the paper
                        Q[:,i,j] += np.multiply(Q1[:,k1,k2p],Q1[:,k2,k1p])
    for m in range(p): # Correct for numerical errors
        row_sums = np.sum(Q[m],1)
#       if any(abs(row_sums-1)>1e-6):
#           raise ValueError('Warning: incorrect normalization of transition matrix! Perhaps a numerical error in fastPhase?')
        Q[m] /= row_sums[:,None]
    return Q

def _assembleEmissionP(theta):
    """
    .. document private functions
    """
    # Compute the emission probabilities
    K,p = theta.shape
    Keff = int(K*(K+1)/2)
    pEmit = np.zeros((p,3,Keff))
    for m in range(p):
        pEmit1 = np.zeros((3,Keff))
        for k1 in range(K):
            for k2 in range(k1+1):
                i = int(((k1+1)*k1)/2+k2)
                pEmit1[0,i] = (1-theta[k1,m])*(1-theta[k2,m])
                pEmit1[1,i] = theta[k1,m]*(1-theta[k2,m]) + theta[k2,m]*(1-theta[k1,m])
                pEmit1[2,i] = theta[k1,m]*theta[k2,m]
#               if abs(np.sum(pEmit1[:,i])-1)>1e-6:
#                   raise ValueError('Warning: incorrect normalization of emission probabilities! Perhaps a numerical error in fastPhase?')
                pEmit1[:,i] /= np.sum(pEmit1[:,i]) # Normalize to compensate for numerical errors
        pEmit[m,:,:] = np.matrix(pEmit1)
    return pEmit

def _assemblePInit(alpha):
    """
    .. document private functions
    """
    # Compute the marginal distribution for the first SNP
    K,p = alpha.shape
    pInit = np.zeros((int(K*(K+1)/2),))
    for k1 in range(K):
        for k2 in range(k1+1):
            i = int(((k1+1)*k1)/2+k2)
            if (k1==k2):
                pInit[i] = alpha[k1,0]*alpha[k1,0]
            else:
                pInit[i] = 2*alpha[k1,0]*alpha[k2,0]
#    if abs(np.sum(pInit)-1)>1e-4:
#       raise ValueError('Warning: incorrect normalization of initial probabilities! Perhaps a numerical error in fastPhase?')
    pInit /= np.sum(pInit)  # Normalize to compensate for numerical errors
    return pInit

def writeX(X, out_file):
    '''
    Convert the genotype data matrix X (consisting of 0,1 and 2's) into the fastPhase
    input format and saves it to a text file. This script assumes that there are no
    missing values in X.
    For more information about the fastPhase format see: http://scheet.org/software.html

    :param X: a numpy array of size (n,p), where n is the number of individuals and p
              is the number of SNPs.
    :param out_file: a string containing the path of the output file onto which X
                     will be written.

    '''
    # Transpose X and convert it to int
    X = X.T.astype(int)

    # Create the two matrices of covariates for fastPhase (i.e. random phasing)
    v1 = np.array([0,1,1])
    v2 = np.array([0,0,1])
    Xp1 = v1[X]
    Xp2 = v2[X]

    # Create output file
    out_f = open(out_file, 'w')

    # Write initial lines with number of individuals and number of sites
    p,n = X.shape
    out_f.write('{0:1d}\n'.format(n))
    out_f.write('{0:1d}\n'.format(p))

    # Write the genotype of each individual
    for i in range(n):
        out_f.write('#id{0:1d}\n'.format(i))
        out_f.write(''.join(map(str, Xp1[:,i]))+'\n')
        out_f.write(''.join(map(str, Xp2[:,i]))+'\n')

    # Close output file
    out_f.close()
