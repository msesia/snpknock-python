setwd("~/Workspace/hmm_knockoffs/tmp")

# library(devtools)
# install_bitbucket("msesia/hmm_knockoffs",subdir="SNPknock_R/SNPknock/",auth_user="msesia@stanford.edu",password="Tq7ySUGyXGAQu2e1XLTV")

library(SNPknock)
library(readr)

# Control parameters
n_keep = 1000
  
# Load X data
Xinp_file = "../tmp/fastphase_small/ukb_hap_chr22.inp"
n = as.numeric(read_lines(Xinp_file, n_max=1))
p = as.numeric(read_lines(Xinp_file, n_max=1, skip=1))
X = data.matrix(read_fwf(Xinp_file, skip=3, n_max=n_keep*2, fwf_widths(rep(1,p)),comment="#", col_types=cols(.default = col_integer())))

# Load fastPHASE fitted model
fp_outPath = "../tmp/fastphase_small/ukb_hap_chr22"
r_file = paste(fp_outPath, "_rhat.txt", sep="")
theta_file = paste(fp_outPath, "_thetahat.txt", sep="")
alpha_file = paste(fp_outPath, "_alphahat.txt", sep="")
char_file = paste(fp_outPath, "_origchars", sep="")
hmm = SNPknock.fp.loadFit_phased(r_file, theta_file, alpha_file, char_file)

# Make knockoff copies of haplotypes
Xk = SNPknock.knockoffHMM(X, hmm$pInit, hmm$Q, hmm$pEmit)

# Compare haplotype column means
plot(colMeans(X),colMeans(Xk),col=rgb(0,0,0,alpha=0.1), pch=16,cex=1); abline(a=0, b=1, col='red', lty=2)
# Compare correlations between consecutive haplotypes
corrX = sapply(2:dim(X)[2], function(j) cor(X[,j-1],X[,j]))
corrXk = sapply(2:dim(X)[2], function(j) cor(Xk[,j-1],Xk[,j]))
plot(corrX,corrXk,col=rgb(0,0,0,alpha=0.1), pch=16,cex=1); abline(a=0, b=1, col='red', lty=2)
# Compare correlations between original SNPs and their successive knockoff
corrXXk = sapply(2:dim(X)[2], function(j) cor(X[,j-1],Xk[,j]))
plot(corrX,corrXXk,col=rgb(0,0,0,alpha=0.1), pch=16,cex=1); abline(a=0, b=1, col='red', lty=2)

# Combine haplotypes into genotypes
X_gen = t(sapply(1:n_keep, function(i) X[2*i-1,]+X[2*i,]))
Xk_gen = t(sapply(1:n_keep, function(i) Xk[2*i-1,]+Xk[2*i,]))

# Compare genotype column means
plot(colMeans(X_gen),colMeans(Xk_gen),col=rgb(0,0,0,alpha=0.1), pch=16,cex=1); abline(a=0, b=1, col='red', lty=2)
