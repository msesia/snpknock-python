library(SNPknockG)

n = 100

# Parameters
group.size = 10

# Load data
hap_file <- system.file("extdata", "haplotypes.RData", package = "SNPknockG")
load(hap_file)

# Load HMM
r_file <- system.file("extdata", "haplotypes_rhat.txt", package = "SNPknockG")
alpha_file <- system.file("extdata", "haplotypes_alphahat.txt", package = "SNPknockG")
theta_file <- system.file("extdata", "haplotypes_thetahat.txt", package = "SNPknockG")
char_file <- system.file("extdata", "haplotypes_origchars", package = "SNPknockG")
hmm <- SNPknock.fp.loadFit(r_file, alpha_file, theta_file, char_file)

# Make problem bigger
M = 3
H = do.call(cbind, replicate(M, H, simplify=FALSE))
hmm$r = do.call(rbind, replicate(M, hmm$r, simplify=FALSE))
hmm$alpha = do.call(rbind, replicate(M, hmm$alpha, simplify=FALSE))
hmm$theta = do.call(rbind, replicate(M, hmm$theta, simplify=FALSE))
  
# Define groups
p = ncol(H)
groups = sample(p/group.size,p,replace=T)

# Generate group knockoffs with specialized algorithm
#Hk <- SNPknock.knockoffHaplotypes_group(H, hmm$r, hmm$alpha, hmm$theta, groups, display_progress=T)
Hk <- SNPknock.knockoffHaplotypes(H, hmm$r, hmm$alpha, hmm$theta, display_progress=T)

plot(colMeans(H), colMeans(Hk))

knock.quality = sapply(1:dim(H)[2], function(j) cor(H[,j],Hk[,j]))
plot(knock.quality)
