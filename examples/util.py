import numpy as np
from matplotlib import pyplot as plt
import pdb

def empCov(x, y):
    return np.mean(np.multiply(x-np.mean(x), y-np.mean(y)))

def empCorr(x, y):
    num = np.mean(np.multiply(x-np.mean(x), y-np.mean(y)))
    den = np.sqrt(np.cov(x)*np.cov(y))
    return num/den

def compare_marginals(X,Xk):
    """
    Compare expected values
    """
    X_means = np.mean(X,0)
    Xk_means = np.mean(Xk,0)

    fig, ax = plt.subplots()
    ax.scatter(X_means, Xk_means, alpha=0.5)
    range_low = np.min((np.min(X_means), np.min(Xk_means)))
    range_high = np.max((np.max(X_means), np.max(Xk_means)))
    ax.plot([range_low, range_high], [range_low, range_high], ls="--", c=".3")
    plt.xlabel('Originals variables')
    plt.ylabel('Knockoff variables')
    plt.title('Comparison of mean values')
    plt.show()

def compare_cons_corr(X,Xk,dist=1):
    """
    Compare correlations among consecutive variables
    """
    _,p = X.shape
    X_corr = [empCorr(X[:,i],X[:,i+dist]) for i in range(p-dist)]
    Xk_corr = [empCorr(Xk[:,i],Xk[:,i+dist]) for i in range(p-dist)]
    fig, ax = plt.subplots()
    ax.scatter(X_corr, Xk_corr, alpha=0.5)
    ax.plot([-1, 1], [-1, 1], ls="--", c=".3")
    plt.xlabel('Original variables')
    plt.ylabel('Knockoff variables')
    plt.title('Comparison of consecutive correlations (dist: '+str(int(dist))+')')
    plt.show()

def compare_cross_corr(X,Xk,dist=1):
    """
    Compare cross-correlations among consecutive variables
    """
    _,p = X.shape
    X_corr = [empCorr(X[:,i],X[:,i+dist]) for i in range(p-dist)]
    Xk_corr = [empCorr(X[:,i],Xk[:,i+dist]) for i in range(p-dist)]
    fig, ax = plt.subplots()
    ax.scatter(X_corr, Xk_corr, alpha=0.5)
    ax.plot([-1, 1], [-1, 1], ls="--", c=".3")
    plt.xlabel('Original variables')
    plt.ylabel('Knockoff variables')
    plt.title('Comparison of cross-correlations (dist: '+str(int(dist))+')')
    plt.show()

def plotPaths(X,Xk):
    """
    Plot path of one observation in X and Xk
    """
    _,p = X.shape
    plt.plot(range(p),X[0,:], label='Original variables')
    plt.plot(range(p),Xk[0,:], label='Knockoff variables')
    plt.legend()
    plt.show()
