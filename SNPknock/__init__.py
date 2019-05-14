from pkg_resources import get_distribution
__version__ = get_distribution('SNPknock').version

from SNPknock.knockoffs import knockoffHMM
from SNPknock.knockoffs import knockoffDMC
from SNPknock.knockoffs import knockoffGenotypes
from SNPknock.knockoffs import knockoffHaplotypes
