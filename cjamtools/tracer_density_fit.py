import numpy as np
from math import *
import matplotlib.pyplot as plt

import emcee
import corner

def surface_number_density(radial_position, n_bins = 0, n_in_bins = 0,\
                   binned_log = False, plot= False):
    """
        This function creates a surface number density profile given a set of 1 dimensional positions. Either a number of bins can be set or number of points in the bins. It is also possible to have fixed number of bins in log10 space or linear. If the data has large dynamical range log10 binning can give better results. The area is that of a circular anulus: pi*(r[i+1]**2- r[i]**2)
        
        Input:
            radial_position     ... Array of 1 dimensional positions
            n_bins = 0          ... Number of bins to make, if 0 then the code will make bins such that there are equal number of points in the bin
            n_in_bins = 0       ... Number of points in all of the bins, aside from the last. If 0 then n_bins has to be different than 0 and r will we linearly or log equidistantly spaced
            binned_log = False  ... If you want logaritmically spaced bins
            plot= False         ... if True plot will be produced
    
        
        Output: 
            r_median            ... median r in each respective bin
            num_density         ... number of points in each bin devided by the area of the anulus
            err_num_density     ... uncertainty assuming only Poission error on the number counts in each bin
            bin_loc             ... array of bin edges
            
    
    """
    # lets sort the radial positions first
    r = np.sort(radial_position)
    
    # string for plots
    string = ''
    
    if n_bins != 0:
        
        # the code will make equally spaced bins in log or linear regime
        if binned_log:
            print('Making number denstiy profile in {} equidistantly spaced logaritm bins'.format(n_bins))
            N_bin, bla = np.histogram(np.log10(r), n_bins)
            bin_loc = 10**bla    
            
            # string for plots
            string +='{} eq spaced logaritm bins'.format(n_bins)
        
        else:
            print('Making number denstiy profile in {} equidistantly spaced linear bins'.format(n_bins))
            N_bin, bin_loc = np.histogram(r, n_bins)
            
            # string for plots
            string +='{} eq spaced linear bins'.format(n_bins)
    if n_in_bins != 0:
        
        # the code will make equally spaced bins in log or linear regime
        bin_loc = r[::n_in_bins]
        N_bin = np.ones(len(bin_loc)-1) * n_in_bins
        
        if bin_loc[-1] != r[-1]:
            
            # if the length of r is not a multiple of n_in_bin
            # and the last bin does not correspond to max(r)
            bin_loc = np.append(bin_loc, r[-1])
            leftover = len(r) - (len(bin_loc)-2)*n_in_bins
            assert leftover > 0, "Something went wrong, leftover is smaller than 0!"
            N_bin = np.append(N_bin, leftover)
            # string for plots
            string +='{} number of points in bins'.format(n_in_bins)
            
    if np.any(N_bin==0):
        
        # if any bin is empty than those bins will be masked and 
        # a warning raised. If bin 3 is empty than the 2nd bin will be extended
        print('WARNING: There are some empty bins.\nI will proceede with merging neighboring bins.\nIf that is not desirable then reduce the number of bins or do sampling in log: binned_log = True')
        bin_loc = np.delete(bin_loc, np.argwhere(N_bin==0))
        N_bin = N_bin[~(N_bin ==0 )]
    
    # calculation of area array ... with circular annulus
    # first lets increase the last bin a bit, just so that
    # we can set a mask r_min<=r<r_max and be sure to include all the 
    # objects in each bin
    bin_loc[-1] *= 1.00001 
    area = np.pi * np.diff(bin_loc**2)
    num_density = N_bin/area
    #assuming a poission counting error
    err_num_density = np.sqrt(N_bin)/area
    
    # computing the median or the positions in each bin:
    r_median = np.zeros(len(N_bin))
    for idx in range(len(r_median)):
        mask = (r >= bin_loc[idx]) & (r < bin_loc[idx+1])
        r_median[idx] = np.median(r[mask])
        
    if plot:
        plt.errorbar(r_median, num_density, yerr=err_num_density, fmt = 'o', label = string)
        plt.loglog()
        plt.xlabel(r"$log_{10}(R_{ell}[arcsec])$")
        plt.ylabel(r"$log_{10}(N_{GC}[arcsec^{-2}])$")
        plt.legend()
    return(r_median, num_density, err_num_density, bin_loc)
    
    
def sersic_profile(Re,n, R, length_unbinned_parent_sample, deltaR = np.ones(1)):
    """
        This generates a sersic profile at the location R, normalized to fit a bined profile. Or whicever profile. If you are using a binned profile then do be sure to input the size of your original dataset otherwise the normalisation will be off! Meaning your mcmc will not be happy.
        All parameters are expected to be be in linear scale!
        
        Input:
            Re ... effective radius
            n ... the power of the exponential function
            R ... array of positions where Sersic should be evaluated 
            length_unbinned_parent_sample ... the size of the original dataset that was used to make the binned profile
            deltaR = np.ones(1) ... array of dR, if R is not equally spaced. Iff R are taken from a linearly equidistant bins then deltaR can be omited. If used in combination with surface_number_density, then deltaR = np.diff(surface_number_density(...)[3])
            
        Output: 
            Analytic sersic profile
        
    """
    # exponent of the sersic profile
    bn = 2*n - 1/3 + 4/405 * 1/n + 46/25515 * 1/n**2
    f = np.exp( - bn * ((R/Re)**(1/n) - 1))

    # if array of bin sizes is not provided then assuming linearly equidistant bins
    if len(deltaR) ==1:
        deltaR = R[1]-R[0]
    # normalising the profile
    norm = len(R) / (2*np.pi * np.sum(f * R * deltaR)) 
    
    # renormalisation to fit the initial sample size
    renorm = length_unbinned_parent_sample / len(R)
    
    return(f * norm *  renorm)