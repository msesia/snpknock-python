import pytest
import numpy as np
from SNPknock import models, knockoffDMC, knockoffHMM, knockoffGenotypes, knockoffHaplotypes
import SNPknock.fastphase as fp
import tempfile

def generate_DMC(p,K):
    Q = np.zeros((p-1,K,K)) # Transition matrices
    for j in range(p-1):
        Q[j,:,:] = np.resize(np.random.uniform(size=K*K),(K,K))
        Q[j,:,:] += np.diag([10]*K)
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
    if(M>K):
        pEmit_diag = np.pad(np.diag([10]*K), ((0,M-K),(0,0)), 'constant')
    else:
        pEmit_diag = np.pad(np.diag([10]*M), ((0,0),(0,K-M)), 'constant')
    for j in range(p):
        pEmit[j,:,:] = np.resize(np.random.uniform(size=M*K),(M,K))
        pEmit[j,:,:] += pEmit_diag
        pEmit[j,:,:] /= np.sum(pEmit[j,:,:],0)
    pInit = np.array([1.0/K]*K) # Initial distribution
    return pInit, Q, pEmit

def verify_exchangeability(X,Xk,groups=None,tolerance=1e-3):
    assert X.shape==Xk.shape, "Knockoffs do not have the correct dimensions"
    n,p = X.shape
    if(groups is None):
        groups = np.arange(p)
    # Check first moments
    means_X = np.mean(X,0)
    means_Xk = np.mean(Xk,0)
    means_diff = (np.linalg.norm(means_X-means_Xk)**2) / (np.linalg.norm(means_X)**2)
    assert means_diff < tolerance, "Knockoffs do not have exchangeable first moments"
    # Check second moments (corr(X_i,X_j) vs corr(Xk_i,Xk_j))
    XX = np.corrcoef(X.T)
    XkXk = np.corrcoef(Xk.T)
    corr_diff = (np.linalg.norm(XX-XkXk)**2) / (np.linalg.norm(XX)**2)
    assert corr_diff < tolerance, "Knockoffs do not have exchangeable second moments"
    # Check second moments (corr(X_i,X_j) vs corr(X_i,Xk_j))
    G = np.corrcoef(X.T, Xk.T)
    XXk = G[:p,p:(2*p)]
    Mask = np.ones((p,p))
    for j in range(p):
        Mask[j,:] = groups!=groups[j]
    corr_diff = (np.linalg.norm(Mask*(XX-XXk))**2) / (np.linalg.norm(XX)**2)
    assert corr_diff < tolerance, "Knockoffs do not have exchangeable second moments"

def test_DMC_basic():
    """
    Test whether the DMC knockoff generation function does not crash
    """
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
    assert np.array_equal(Xk, Xk_g), "Knockoffs with trivial groups do not match"
    assert np.isfinite(Xk).all(), "Knockoffs are not finite"

def test_HMM_basic():
    """
    Test whether the HMM knockoff generation function does not crash
    """
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
    assert np.array_equal(Xk, Xk_g), "Knockoffs with trivial groups do not match"
    assert np.isfinite(Xk).all(), "Knockoffs are not finite"

def test_DMC():
    """
    Test whether the DMC knockoff generation is correct
    """
    p = 10
    K = 4
    n = 100000
    pInit, Q = generate_DMC(p,K)
    modelX = models.DMC(pInit, Q)
    X = modelX.sample(n)
    knockoffs = knockoffDMC(pInit, Q, seed=123)
    Xk = knockoffs.sample(X)
    verify_exchangeability(X,Xk)

def test_DMC_groups():
    """
    Test whether the DMC knockoff generation is correct
    """
    p = 10
    K = 4
    n = 100000
    pInit, Q = generate_DMC(p,K)
    modelX = models.DMC(pInit, Q)
    X = modelX.sample(n)
    groups = np.repeat(np.arange(p),3)[:p]
    knockoffs = knockoffDMC(pInit, Q, groups=groups, seed=123)
    Xk = knockoffs.sample(X)
    verify_exchangeability(X,Xk,groups=groups)

def test_HMM():
    """
    Test whether the HMM knockoff generation is correct
    """
    p = 10
    K = 4
    M = 5
    n = 100000
    pInit, Q, pEmit = generate_HMM(p,K,M)
    modelX = models.HMM(pInit, Q, pEmit)
    X = modelX.sample(n)
    knockoffs = knockoffHMM(pInit, Q, pEmit, seed=123)
    Xk = knockoffs.sample(X)
    verify_exchangeability(X,Xk)

def test_HMM_groups():
    """
    Test whether the HMM knockoff generation is correct
    """
    p = 10
    K = 4
    M = 5
    n = 100000
    pInit, Q, pEmit = generate_HMM(p,K,M)
    modelX = models.HMM(pInit, Q, pEmit)
    X = modelX.sample(n)
    groups = np.repeat(np.arange(p),3)[:p]
    knockoffs = knockoffHMM(pInit, Q, pEmit, groups=groups, seed=123)
    Xk = knockoffs.sample(X)
    verify_exchangeability(X, Xk, groups=groups)

def test_genotypes_fastphase():
    """
    Test whether genotype knockoffs with HMM fitted by fastPHASE are accurate
    """
    p = 10
    K = 3
    M = 3
    n = 1000
    pInit, Q, pEmit = generate_HMM(p,K,M)
    modelX = models.HMM(pInit, Q, pEmit)
    X = modelX.sample(n)
    _, Xfp_file = tempfile.mkstemp()
    fp.writeXtoInp(X, Xfp_file)
    fastphase = "fastphase" # Name of fastPhase executable
    _, out_path = tempfile.mkstemp()
    fp.runFastPhase(Xfp_file, out_path, fastphase=fastphase, K=5, numit=25)
    r_file     = out_path + "_rhat.txt"
    alpha_file = out_path + "_alphahat.txt"
    theta_file = out_path + "_thetahat.txt"
    char_file = out_path + "_origchars"
    hmm = fp.loadHMM(r_file, alpha_file, theta_file, char_file)
    knockoffs = knockoffGenotypes(hmm["r"], hmm["alpha"], hmm["theta"], seed=123)
    Xk = knockoffs.sample(X)
    verify_exchangeability(X, Xk, tolerance=1e-1)

def test_genotypes_exact():
    """
    Test whether genotype knockoffs with true HMM are accurate
    """
    p = 10
    K = 3
    M = 3
    n_train = 1000
    n_test = 100000
    pInit, Q, pEmit = generate_HMM(p,K,M)
    modelX = models.HMM(pInit, Q, pEmit)
    X = modelX.sample(n_train)
    _, Xfp_file = tempfile.mkstemp()
    fp.writeXtoInp(X, Xfp_file)
    fastphase = "fastphase" # Name of fastPhase executable
    _, out_path = tempfile.mkstemp()
    fp.runFastPhase(Xfp_file, out_path, fastphase=fastphase, K=5, numit=25)
    r_file     = out_path + "_rhat.txt"
    alpha_file = out_path + "_alphahat.txt"
    theta_file = out_path + "_thetahat.txt"
    char_file = out_path + "_origchars"
    hmm = fp.loadHMM(r_file, alpha_file, theta_file, char_file)
    knockoffs = knockoffGenotypes(hmm["r"], hmm["alpha"], hmm["theta"], seed=123)
    hmm_hat = fp.loadHMM(r_file, alpha_file, theta_file, char_file, compact=False)
    modelX_hat = models.HMM(hmm_hat["pInit"], hmm_hat["Q"], hmm_hat["pEmit"])
    X_new = modelX_hat.sample(n_test)
    Xk_new = knockoffs.sample(X_new)
    verify_exchangeability(X_new, Xk_new, tolerance=1e-3)

def test_haplotypes_fastphase():
    """
    Test whether haplotype knockoffs with HMM fitted by fastPHASE are accurate
    """
    p = 10
    K = 3
    M = 2
    n = 1000
    pInit, Q, pEmit = generate_HMM(p,K,M)
    modelX = models.HMM(pInit, Q, pEmit)
    X = modelX.sample(n)
    _, Xfp_file = tempfile.mkstemp()
    fp.writeXtoInp(X, Xfp_file, phased=True)
    fastphase = "fastphase" # Name of fastPhase executable
    _, out_path = tempfile.mkstemp()
    fp.runFastPhase(Xfp_file, out_path, fastphase=fastphase, phased=True, K=5, numit=25)
    r_file     = out_path + "_rhat.txt"
    alpha_file = out_path + "_alphahat.txt"
    theta_file = out_path + "_thetahat.txt"
    char_file = out_path + "_origchars"
    hmm = fp.loadHMM(r_file, alpha_file, theta_file, char_file, phased=True)
    knockoffs = knockoffHaplotypes(hmm["r"], hmm["alpha"], hmm["theta"], seed=123)
    Xk = knockoffs.sample(X)
    verify_exchangeability(X, Xk, tolerance=1e-1)

def test_haplotypes_exact():
    """
    Test whether haplotype knockoffs with true HMM are accurate
    """
    p = 10
    K = 3
    M = 2
    n_train = 1000
    n_test = 100000
    pInit, Q, pEmit = generate_HMM(p,K,M)
    modelX = models.HMM(pInit, Q, pEmit)
    X = modelX.sample(n_train)
    _, Xfp_file = tempfile.mkstemp()
    fp.writeXtoInp(X, Xfp_file, phased=True)
    fastphase = "fastphase" # Name of fastPhase executable
    _, out_path = tempfile.mkstemp()
    fp.runFastPhase(Xfp_file, out_path, fastphase=fastphase, phased=True, K=5, numit=25)
    r_file     = out_path + "_rhat.txt"
    alpha_file = out_path + "_alphahat.txt"
    theta_file = out_path + "_thetahat.txt"
    char_file = out_path + "_origchars"
    hmm = fp.loadHMM(r_file, alpha_file, theta_file, char_file)
    knockoffs = knockoffHaplotypes(hmm["r"], hmm["alpha"], hmm["theta"], seed=123)
    hmm_hat = fp.loadHMM(r_file, alpha_file, theta_file, char_file, compact=False, phased=True)
    modelX_hat = models.HMM(hmm_hat["pInit"], hmm_hat["Q"], hmm_hat["pEmit"])
    X_new = modelX_hat.sample(n_test)
    Xk_new = knockoffs.sample(X_new)
    verify_exchangeability(X_new, Xk_new, tolerance=1e-3)

def test_haplotypes_hmm():
    """
    Test whether specialized haplotype knockoff algorithm agrees with special case
    """
    p = 10
    K = 5
    M = 2
    n_train = 1000
    n_test = 100000
    pInit, Q, pEmit = generate_HMM(p,K,M)
    modelX = models.HMM(pInit, Q, pEmit)
    X = modelX.sample(n_train)
    _, Xfp_file = tempfile.mkstemp()
    fp.writeXtoInp(X, Xfp_file, phased=True)
    fastphase = "fastphase" # Name of fastPhase executable
    _, out_path = tempfile.mkstemp()
    fp.runFastPhase(Xfp_file, out_path, fastphase=fastphase, phased=True, K=5, numit=25)
    r_file     = out_path + "_rhat.txt"
    alpha_file = out_path + "_alphahat.txt"
    theta_file = out_path + "_thetahat.txt"
    char_file = out_path + "_origchars"
    groups = np.repeat(np.arange(p),3)[:p]
    hmm_compact = fp.loadHMM(r_file, alpha_file, theta_file, char_file)
    hmm = fp.loadHMM(r_file, alpha_file, theta_file, char_file, compact=False, phased=True)
    knockoffs = knockoffHMM(hmm["pInit"], hmm["Q"], hmm["pEmit"], groups=groups, seed=123)
    knockoffs_hap = knockoffHaplotypes(hmm_compact["r"], hmm_compact["alpha"], hmm_compact["theta"], \
                                       groups=groups, seed=123)
    hmm_hat = fp.loadHMM(r_file, alpha_file, theta_file, char_file, compact=False, phased=True)
    Xk = knockoffs.sample(X)
    Xk_compact = knockoffs_hap.sample(X)
    assert np.array_equal(Xk, Xk_compact), "Knockoffs with trivial groups do not match"

def test_genotypes_hmm():
    """
    Test whether specialized genotype knockoff algorithm agrees with special case
    """
    p = 10
    K = 5
    M = 3
    n_train = 1000
    n_test = 100000
    pInit, Q, pEmit = generate_HMM(p,K,M)
    modelX = models.HMM(pInit, Q, pEmit)
    X = modelX.sample(n_train)
    _, Xfp_file = tempfile.mkstemp()
    fp.writeXtoInp(X, Xfp_file)
    fastphase = "fastphase" # Name of fastPhase executable
    _, out_path = tempfile.mkstemp()
    fp.runFastPhase(Xfp_file, out_path, fastphase=fastphase, K=5, numit=25)
    r_file     = out_path + "_rhat.txt"
    alpha_file = out_path + "_alphahat.txt"
    theta_file = out_path + "_thetahat.txt"
    char_file = out_path + "_origchars"
    groups = np.repeat(np.arange(p),3)[:p]
    hmm_compact = fp.loadHMM(r_file, alpha_file, theta_file, char_file)
    hmm = fp.loadHMM(r_file, alpha_file, theta_file, char_file, compact=False)
    knockoffs = knockoffHMM(hmm["pInit"], hmm["Q"], hmm["pEmit"], groups=groups, seed=123)
    knockoffs_gen = knockoffGenotypes(hmm_compact["r"], hmm_compact["alpha"], hmm_compact["theta"], \
                                      groups=groups, seed=123)
    hmm_hat = fp.loadHMM(r_file, alpha_file, theta_file, char_file, compact=False, phased=True)
    Xk = knockoffs.sample(X)
    Xk_compact = knockoffs_gen.sample(X)
    assert np.array_equal(Xk, Xk_compact), "Knockoffs with trivial groups do not match"
