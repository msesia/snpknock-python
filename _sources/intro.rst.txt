Introduction
--------------------------

SNPknock is a Python library for generating knockoff variables from discrete 
Markov chains and hidden Markov models, with specific support for genomic data.
See also the corresponding `R package <https://msesia.github.io/snpknock/>`_.

This package implements the algorithms for knockoff generation described in:

   * "Gene hunting with hidden Markov model knockoffs", Sesia et al., Biometrika, 2019, (`<doi:10.1093/biomet/asy033>`_).
   * "Multi-resolution localization of causal variants across the genome", Sesia et al., bioRxiv, 2019, (`<doi:10.1101/631390>`_).

Feature highlights:

   * Generate knockoffs for discrete Markov chains (DMC).
   * Generate knockoffs for hidden Markov models (HMM).
   * Generate knockoffs for genotype and haplotype data.
   * Provides a user-friendly interface for fitting an HMM to genetic data 
     using the software `fastPhase <http://stephenslab.uchicago.edu/software.html#fastphase>`_.

If you want to learn about applying SNPknock to analyze data from large genome-wide association studies, see *KnockoffZoom*: https://msesia.github.io/knockoffzoom.

SNPknock is licensed under the GPL-v3 :doc:`license`.
