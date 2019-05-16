Tutorial
--------------
This page contains examples of how to use SNPknock.


Verify installation
^^^^^^^^^^^^^^^^^^^

To get started, we verify that SNPknock was installed correctly.

.. runblock:: pycon

  >>> import SNPknock
  >>> print SNPknock.__version__

Knockoffs of a discrete Markov chain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, let us define a discrete Markov chain (DMC) using the `models` subpackage from SNPknock

.. runblock:: pycon

  >>> import numpy as np
  >>> np.random.seed(123) # Make this examples replicable
  >>> p = 50 # Number of variables
  >>> K = 4  # Number of states
  >>> Q = np.zeros((p-1,K,K)) # Transition matrices
  >>> for j in range(p-1):
  ...    Q[j,:,:] = np.resize(np.random.uniform(size=K*K),(K,K))
  ...    Q[j,:,:] += np.diag([10]*K)
  ...    Q[j,:,:] /= np.sum(Q[j,:,:],1)[:,None]
  >>> pInit = np.array([1.0/K]*K) # Initial distribution

We create a synthetic data matrix X by drawing from the DMC defined above.

.. runblock:: pycon

  >>> from SNPknock import models
  >>> n=100 # Number of samples
  >>> modelX = models.DMC(pInit, Q)
  >>> X = modelX.sample(n)
  >>> print(X)

Knockoffs can be sampled as follows.

.. runblock:: pycon

  >>> from SNPknock import knockoffDMC
  >>> knockoffs = knockoffDMC(pInit, Q, seed=123)
  >>> Xk = knockoffs.sample(X)
  >>> print(Xk)

Group-knockoffs can be sampled similarly.

.. runblock:: pycon

  >>> groups = np.repeat(np.arange(p),3)[:p] # Group assignments (groups of size 3)
  >>> knockoffs = knockoffDMC(pInit, Q, groups=groups, seed=123)
  >>> Xk = knockoffs.sample(X)
  >>> print(Xk)

Knockoffs of a hidden Markov model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can define a hidden Markov model (HMM) using the `models` subpackage from SNPknock

.. runblock:: pycon

  >>> import numpy as np
  >>> p = 50 # Number of variables
  >>> K = 5  # Number of latent states
  >>> M = 3  # Number of emission states
  >>> Q = np.zeros((p-1,K,K)) # Transition matrices
  >>> for j in range(p-1):
  ...    Q[j,:,:] = np.resize(np.random.uniform(size=K*K),(K,K))
  ...    Q[j,:,:] += np.diag([10]*K)
  ...    Q[j,:,:] /= np.sum(Q[j,:,:],1)[:,None]
  >>> pEmit = np.zeros((p,M,K)) # Emission probabilities
  >>> for j in range(p):
  ...   pEmit[j,:,:] = np.resize(np.random.uniform(size=M*K),(M,K))
  ...   pEmit[j,:,:] += np.pad(np.diag([10]*M), ((0,0),(0,K-M)), 'constant')
  ...   pEmit[j,:,:] /= np.sum(pEmit[j,:,:],0)
  >>> pInit = np.array([1.0/K]*K) # Initial distribution

Now, we create a synthetic data matrix X by drawing from the DMC defined above.

.. runblock:: pycon

  >>> from SNPknock import models
  >>> n=100 # Number of samples
  >>> modelX = models.HMM(pInit, Q, pEmit)
  >>> X = modelX.sample(n)
  >>> print(X)

Finally, we sample the knockoffs.

.. runblock:: pycon

  >>> from SNPknock import knockoffHMM
  >>> groups = np.repeat(np.arange(p),3)[:p] # Group assignments (groups of size 3)
  >>> knockoffs = knockoffHMM(pInit, Q, pEmit, groups=groups, seed=123)
  >>> Xk = knockoffs.sample(X)
  >>> print(Xk)

Working with genotype data
^^^^^^^^^^^^^^^^^^^^^^^^^^
In this section we show how to use SNPknock to create knockoffs
of genotype data, using an HMM fitted with the imputation software
`fastPhase <http://stephenslab.uchicago.edu/software.html#fastphase>`_.
The submodule `fastphase` of SNPknock contains a simple interface to
the relevant features of the imputation software.

We assume that our genotype data consists of a matrix `X` in the
same format as in the HMM example above. Each row of `X` is a sequence
of 0,1 and 2's representing the genotype of an individual.
In order to fit an HMM to this data using fastPhase, we first need to
convert the matrix `X` into the appropriate input format. This can be
easily done as follows.

.. runblock:: pycon

  >>> import SNPknock.fastphase as fp
  >>> Xfp_file = 'tmp/X.inp' # Temporary file that will be used as input for fastPhase
  >>> fp.writeXtoInp(X, Xfp_file)

Once we have created the input file for fastPhase, we can call
execute the imputation software to fit the HMM.

.. runblock:: pycon

  >>> fastphase = "fastphase" # Name of fastPhase executable
  >>> out_path = "tmp/example" # Prefix to temporary output files produced by fastPhase
  >>> fp.runFastPhase(Xfp_file, out_path, fastphase=fastphase, K=12, numit=25)

We collect the parameter estimates obtained by fastPhase and use
use them to define an HMM for knockoffs as follows.

.. runblock:: pycon

  >>> r_file     = out_path + "_rhat.txt"
  >>> alpha_file = out_path + "_alphahat.txt"
  >>> theta_file = out_path + "_thetahat.txt"
  >>> origchars_file = out_path + "_origchars"
  >>> hmm = fp.loadHMM(r_file, alpha_file, theta_file, origchars_file)

Finally, we sample the knockoffs.

.. runblock:: pycon

  >>> from SNPknock import knockoffGenotypes
  >>> knockoffs = knockoffGenotypes(hmm["r"], hmm["alpha"], hmm["theta"], seed=123)
  >>> Xk = knockoffs.sample(X)
  >>> print(Xk)

Alternatively, we can use the general knockoff construction for HMMs.

.. runblock:: pycon

  >>> r_file     = out_path + "_rhat.txt"
  >>> alpha_file = out_path + "_alphahat.txt"
  >>> theta_file = out_path + "_thetahat.txt"
  >>> origchars_file = out_path + "_origchars"
  >>> hmm = fp.loadHMM(r_file, alpha_file, theta_file, origchars_file, compact=False)
  >>> knockoffs = knockoffHMM(hmm["pInit"], hmm["Q"], hmm["pEmit"], seed=123)
  >>> Xk = knockoffs.sample(X)
  >>> print(Xk)

Working with phased haplotypes
^^^^^^^^^^^^^^^^^^^^^^^^^^

If phased haplotypes are available, we can efficiently construct knockoffs as follows.

First, let's generate some fake haplotypes

.. runblock:: pycon

  >>> import numpy as np
  >>> p = 50 # Number of variables
  >>> K = 5  # Number of latent states
  >>> M = 2  # Number of emission states
  >>> Q = np.zeros((p-1,K,K)) # Transition matrices
  >>> for j in range(p-1):
  ...    Q[j,:,:] = np.resize(np.random.uniform(size=K*K),(K,K))
  ...    Q[j,:,:] += np.diag([10]*K)
  ...    Q[j,:,:] /= np.sum(Q[j,:,:],1)[:,None]
  >>> pEmit = np.zeros((p,M,K)) # Emission probabilities
  >>> for j in range(p):
  ...   pEmit[j,:,:] = np.resize(np.random.uniform(size=M*K),(M,K))
  ...   pEmit[j,:,:] /= np.sum(pEmit[j,:,:],0)
  >>> pInit = np.array([1.0/K]*K) # Initial distribution

.. runblock:: pycon

  >>> from SNPknock import models
  >>> n=100 # Number of samples
  >>> modelX = models.HMM(pInit, Q, pEmit)
  >>> H = modelX.sample(n)
  >>> print(H)


Then, we fit fastPhase.

.. runblock:: pycon

  >>> # Write binary data as phased haplotypes
  >>> Hfp_file = 'tmp/H.inp' # Temporary file that will be used as input for fastPhase
  >>> fp.writeXtoInp(H, Hfp_file, phased=True)
  >>> # Fit fastPhase to phased haplotypes
  >>> fp.runFastPhase(Hfp_file, out_path, fastphase=fastphase, phased=True, K=12, numit=25)  
  >>> r_file     = out_path + "_rhat.txt"
  >>> alpha_file = out_path + "_alphahat.txt"
  >>> theta_file = out_path + "_thetahat.txt"
  >>> origchars_file = out_path + "_origchars"
  >>> hmm = fp.loadHMM(r_file, alpha_file, theta_file, origchars_file)

Finally, we generate the knockoffs

.. runblock:: pycon

  >>> from SNPknock import knockoffHaplotypes
  >>> knockoffs = knockoffHaplotypes(hmm["r"], hmm["alpha"], hmm["theta"], seed=123)
  >>> Hk = knockoffs.sample(H)
  >>> print(Hk)
