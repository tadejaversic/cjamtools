import numpy as np
from math import *
from scipy import stats
import matplotlib.pyplot as plt

import emcee
import corner

"""
    Module for modeling the velocity dispersion profile. 
    It can create a mock dataset for testing, MCMC and Pryor and Meylan 1993
    
    v1.0 2.6.2021: Mock data and MCMC

"""

def vel_dispersion_profile_mock_data(size = 300, plot = False):
    """
        This function just demonstrates the functionality of the mcmc
        It creates a mock dataset of size size and fits mean velocity and velocity dispersion
        assuming a gausian velocity distribution.
        
        Input:
            size    ... size of the dataset
            plot    ... if True than the plot of the dataset is shown
    """
    
    np.random.seed(33)
    # distribution of the positions. Just some function
    r = stats.beta.rvs(.8, 3, loc = 1 ,scale=100, size=size)
    sorted_r = np.sort(r)

    # just some made up function for the velocity dispersion profile
    std = lambda x: (x/100)**-0.3 *(1+x/100)**0.1
    # simulated uncertainty of the measuemrnts
    vel_err = stats.lognorm.rvs(1, size = size)

    # velocities of particles are already sorted by the radius
    vel = np.array([ np.random.normal(0, std(sorted_r[i]), size =1) for i in range(0, size) ])
    # observed velocities perturbed by the measurment uncertainties
    vel_obs = np.array([ np.random.normal(vel[i], vel_err[i]/10, size =1) for i in range(0, size) ])
    
    if plot:
        plt.scatter(sorted_r, vel, label = 'Unpertubed velocity', s = 5)
        plt.scatter(sorted_r, vel_obs, label = 'Mock sample', s = 5)
        for idx, i in enumerate([1,3]):
            plt.plot(sorted_r, -i*std(sorted_r), c = 'C'+str(idx), label = str(i)+r'$\sigma$')
            plt.plot(sorted_r, i*std(sorted_r), c = 'C'+str(idx))
        plt.legend()
        plt.xlabel('r')
        plt.ylabel('v')
        plt.show()
    
    return(sorted_r, vel_obs, vel_err)

################# Velocity dispersion modeling with MCMC ##################
###########################################################################

def sigma_prior(sigma, sigma_min, sigma_max):
    """
        Uniform prior to the intrinsitc velocity dispersion of the system.
        If sigma_min < sigma <sigma_max then the function returns 0,
        otherwise -inf!
        Sigma should be in linear scale, or adjust sigma min and max
        
        Input: 
            sigma           ... tested parameter in linear scale!
            sigma_min = 0   ... minimal value of sigma for the uniform prior
            sigma_max = 20  ... maximal value of sigma for the uniform prior
        
        Return:
            log prior: 0 or -inf
    """
    if (sigma<sigma_max) & (sigma>sigma_min):
        return(np.log(1))
    else:
        return(-np.inf)

def mean_v_prior(mean_v, mean_v_min, mean_v_max):    
    """
        Uniform prior to the intrinsitc mean velocity of the system.
        If mean_v_min < mean_v <mean_v_max then the function returns 0,
        otherwise -inf!
        
        Input: 
            sigma           ... tested parameter in linear scale!
            mean_v_min = 0   ... minimal value of mean_v for the uniform prior
            mean_v_max = 20  ... maximal value of mean_v for the uniform prior

        Return:
            log prior: 0 or -inf
    """
    if (mean_v<mean_v_max) & (mean_v>mean_v_min):
        return(np.log(1))
    else:
        return(-np.inf)

def log_likelihood(velocity, velocity_uncertainty, mean_velocity, velocity_dispersion):
    """
        Gaussian likelihood for the velocity distribution.
        
        Input: 
            velocity               ... data: array of velocities
            velocity_uncertainty   ... data: array of velocity uncertainties
            mean_velocity          ... modeled parameters: value of proposed mean velocity
            velocity_dispersion    ... modeled parameters: value of proposed intrinsic velocity dispersion

        Return:
            log(likelihood)
    """   
    # combining the velocity dispersion and the uncertainty
    sigy = velocity_uncertainty**2 + velocity_dispersion**2
    
    a = -0.5 * np.log(2*np.pi * sigy) - (velocity-mean_velocity)**2/(2*sigy)
    
    return(np.sum(a))
    
def log_posterior(parameters, velocity, velocity_uncertainty,\
                  mean_v_min, mean_v_max,\
                sigma_min, sigma_max):
    """
        Logarithm of posterior. Combining the likelihood and 
        
        Input: 
            parameters       ... tested parameters:
                mean_velocity          ... modeled parameters: value of proposed mean velocity
                velocity_dispersion    ... modeled parameters: value of proposed intrinsic velocity 
            velocity               ... data: array of velocities
            velocity_uncertainty   ... data: array of velocity uncertainties
            sigma           ... tested parameter in linear scale!
            mean_v_min = 0   ... minimal value of mean_v for the uniform prior
            mean_v_max = 20  ... maximal value of mean_v for the uniform prior
            sigma_min = 0   ... minimal value of sigma for the uniform prior
            sigma_max = 20  ... maximal value of sigma for the uniform prior

        Return:
            log prior: 0 or -inf
    """
    # unpacking parameters
    mean_v, sigma = parameters
    
    # computing the prior
    prior = mean_v_prior(mean_v, mean_v_min, mean_v_max) +\
                    sigma_prior(sigma, sigma_min, sigma_max)
    
    # if either of the parameters are outside of the priror region then
    # the log posteriror is -inf
    if prior == -np.inf:
        return(prior)
    else:
        bla = log_likelihood(velocity, velocity_uncertainty, mean_v, sigma)
        return(bla + prior)   
    
    
    
def vel_dispersion_profile_mcmc(r, vel_obs, vel_err, mean_v_min = -20, mean_v_max = 20, sigma_min = 0, sigma_max = 20, num_of_bins = 20, percentile = [16,50,84], mcmc_plot = True, test_mode = False, save_fig ='no'):
     
    """
        Logarithm of posterior. 
        The code first bins the data acording to logarithmically spaced bins. Number is defined with num_of_bins. Currently there is no possibility to make bins with equal number of points inside. The location of the bin is computed by evaluating the median of the positions in each bin.
        
        The mean and dispersion will only be computed for bins that have 4 points! If any bin is empty the code returns nans for that bin and raises a warning.
        
        MCMC:
            50 walkers and 2000 steps. First 500 steps are rejected as a burn in phase.
            Initial positions for mean _v and sigma are taken as the arithemtic mean of the prior interval
        
        Input: 
            r                ... data: 1d positions .. do not have to be sorted
            vel_obs          ... data: 1d velocities .. have to be in the same order as r
            vel_err          ... data: uncertainties of 1d velocities .. have to be in the same order as r
            mean_v_min = 0   ... minimal value of mean_v for the uniform prior
            mean_v_max = 20  ... maximal value of mean_v for the uniform prior
            sigma_min = 0    ... minimal value of sigma for the uniform prior
            sigma_max = 20   ... maximal value of sigma for the uniform prior
            num_of_bins = 20 ... 
            percentile = [16,50,84] ... percentiles of the chain to be returned by the code. The values should be in acending order with 50 in the center
            mcmc_plot = True        ... keyword to toggle mcmc plots for each bin
            test_mode = False       ... if True than the above mock data set will be used
            save_fig ='no'          ... location and name of saved figure. If left at default nothing will happen 
        
        Return:
            centers        ... median of the location of the 
            vel_moments    ... percentiles of the modeled velocity from the posterior in each bin
            sig_moments    ... percentiles of the sigma form the posterior at each bin
            
        Calling sequence:
        
        If you want to use the mock data then first generate it:
            vel_dispersion_profile_mcmc(*vel_dispersion_profile_mock_data(size = 300, plot = True))
            
        ADD: MCMC analysis!
    """
    
    # preparation of the data: r has to be sorted and 
    # velocity and velocity uncertainty also has to be sorted in the same 
    sorted_r = np.sort(r)
    velocity, velocity_uncertainty = vel_obs[np.argsort(r)], vel_err[np.argsort(r)]
    
    # array of bin edges created logspaced
    bin_r = np.logspace(np.log10(sorted_r[0]*0.99), np.log10(sorted_r[-1]*1.001), num = num_of_bins+1)
    
    # declaring the array of bin locations, number of objects in bins
    # normal mean and standard deviation in each bin
    centers = np.zeros(num_of_bins)
    N_in_bins = np.zeros(num_of_bins)
    vel_bins = np.zeros(num_of_bins)
    vel_sigma = np.zeros(num_of_bins)      
    
    # computing the median of the bins and number of objects in each bin
    for i in range(0, num_of_bins):
        r_mask = (sorted_r>=bin_r[i]) &  (sorted_r<bin_r[i+1])
        centers[i] += np.median(sorted_r[r_mask])
        N_in_bins[i] += np.sum(r_mask)
    print('\nCOMMENT: {} bins out of {} have 2 or less points. MCMC will not be run for those and nans will be returned\n'.format(np.sum(N_in_bins < 3), num_of_bins))
    
    # asserting nothing went wrong with the binning
    assert np.sum(N_in_bins) == len(sorted_r), 'Some points were not be counted'
    if np.any(N_in_bins == 0):
        print('WARNING: {} bins are empty, sugest you reduce num_of_bins'.format(np.sum(N_in_bins == 0)))

    # declaring an array of velocity moments and dispersion moments
    # in each bin from the posterior 
    vel_moments = np.zeros((num_of_bins, len(percentile)))
    sig_moments = np.zeros((num_of_bins, len(percentile)))
  
    
    # innitialising parameters for the mcmc
    init = np.array([(mean_v_max-mean_v_min)/2,(sigma_max-sigma_min)/2])
    n_walkers = 50
    nsteps = 2000
    initial_pos = [init * np.random.uniform(size = 2) for i in range(n_walkers)]
    
    # starting the mcmc
    for i in range(0, num_of_bins):
        r_mask = (sorted_r>=bin_r[i]) &  (sorted_r<bin_r[i+1])
    
        if np.sum(r_mask)>2:
            # Only bins with 3 or more points will be computed
            vel_bins[i] = np.mean(velocity[r_mask])
            vel_sigma[i] = np.std(velocity[r_mask])
            sampler = emcee.EnsembleSampler(n_walkers, len(init),\
                                            log_posterior, args=(velocity[r_mask], velocity_uncertainty[r_mask], mean_v_min, mean_v_max, sigma_min, sigma_max))
            sampler.run_mcmc(initial_pos, nsteps, progress=True)
            chains = sampler.get_chain(discard=500, flat=True)
            if mcmc_plot:
                # making corner plot for each bin
                title = 'Posteriror for the\nmean velocity and\nvelocity dispersion\nin bin {}out of {}\n{:.2f}<=r<{:.2f} with\n{} points'.format(i+1, num_of_bins, bin_r[i], bin_r[i+1], N_in_bins[i])
                figure = corner.corner(chains, labels=[r'$\overline{v}$', r'$\sigma$'], title = title,\
                             quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
                figure.text(0.6,0.7, title, fontsize = 12)
                plt.pause(5)       

            # writting the percentiles for the      
            vel_moments[i] = np.percentile(chains[:,0], percentile)
            sig_moments[i] = np.percentile(chains[:,1], percentile)

        else: 
            # if the bin has less than 3 points writting nans
            vel_moments[i] = np.ones(len(percentile)) * np.nan
            sig_moments[i] = np.ones(len(percentile)) * np.nan
            vel_bins[i] = np.nan
            vel_sigma[i] = np.nan
    
    # making final plots for inspection
    pos_median = int(len(percentile)/2)
    # mean velocity
    plt.title('Best fit mean velocity as a function of radius')
    plt.plot(centers, vel_moments[:, pos_median], label = 'median', c = 'C1')
    plt.fill_between(centers, vel_moments[:, pos_median-1], vel_moments[:, pos_median+1],\
                     alpha = 0.4, color = 'C1')

    plt.errorbar(centers, vel_bins, yerr= vel_sigma, fmt = 'ko', label='ordinary bin mean')
    plt.plot(centers, vel_moments[:, pos_median], label = 'best fit median v', c = 'C0', lw = 3)
    plt.fill_between(centers, vel_moments[:, pos_median-1], vel_moments[:, pos_median+1],\
                     alpha = 0.4, color = 'C0', lw = 3)

    plt.plot(centers, sig_moments[:, pos_median], label = r'$best\ fit\ \sigma$', color = 'C1', lw = 3)
    plt.fill_between(centers, sig_moments[:, pos_median-1],\
                     sig_moments[:, pos_median+1], alpha = 0.4, color = 'C1', lw = 3)

    plt.plot(centers, -sig_moments[:, pos_median], color = 'C1', lw = 3)
    plt.fill_between(centers, -sig_moments[:, pos_median-1],\
                     -sig_moments[:, pos_median+1], alpha = 0.4, color = 'C1', lw = 3)
    plt.scatter(r, vel_obs, c = "k", s = 5)
    plt.legend()
    plt.xlabel('r')
    plt.ylabel(r"$\bar{v}$")
    if save_fig !='no':
        plt.savefig(save_fig)
    plt.pause(20)
    
    return(centers, vel_moments, sig_moments)


################## Pryor & Meylan 1993 ###############
################ Fill me #############################

def equation_1(vel, vel_err , mean_v, sigma):
    """
        Equation 2 in Pryor and Maylen
        vel        .. array of velocities in the bin 
        vel_err   ... array of observed velocity uncertainties
        mean_v    ... mean velocity in the bin .. true
        sigma     ... true velocity dispersion
    """
    a = 1/(sigma **2 + vel_err**2)
    return(np.sum(vel*a) - mean_v * np.sum(a))

def equation_2(vel, vel_err , mean_v, sigma):
    """
        Equation 3 in Pryor and Maylen
        vel        .. array of velocities in the bin 
        vel_err   ... array of observed velocity uncertainties
        mean_v    ... mean velocity in the bin .. true
        sigma     ... true velocity dispersion
    """
    a = 1/(sigma **2 + vel_err**2)
    return(np.sum((vel - mean_v)**2 * a**2) - np.sum(a))




def vel_dispersion_profile_PM93(r, vel_obs, vel_err, mean_v_min = -20, mean_v_max = 20, sigma_min = 0, sigma_max = 20, num_of_bins = 20, percentile = [16,50,84], mcmc_plot = True, test_mode = False):
     
    """
        Algorithm of Pryor an Meylen 1993
        
        MCMC:
            50 walkers and 2000 steps. First 500 steps are rejected as a burn in phase.
            Initial positions for mean _v and sigma are taken as the arithemtic mean of the prior interval
        
        Input: 
            r                ... data: 1d positions .. do not have to be sorted
            vel_obs          ... data: 1d velocities .. have to be in the same order as r
            vel_err          ... data: uncertainties of 1d velocities .. have to be in the same order as r
            mean_v_min = 0   ... minimal value of mean_v for the uniform prior
            mean_v_max = 20  ... maximal value of mean_v for the uniform prior
            sigma_min = 0    ... minimal value of sigma for the uniform prior
            sigma_max = 20   ... maximal value of sigma for the uniform prior
            num_of_bins = 20 ... 
            percentile = [16,50,84] ... percentiles of the chain to be returned by the code. The values should be in acending order with 50 in the center
            mcmc_plot = True        ... keyword to toggle mcmc plots for each bin
            test_mode = False       ... if True than the above mock data set will be used
        
        Return:
            centers        ... median of the location of the 
            vel_moments    ... percentiles of the modeled velocity from the posterior in each bin
            sig_moments    ... percentiles of the sigma form the posterior at each bin
            
        Calling sequence:
        
        If you want to use the mock data then first generate it:
            vel_dispersion_profile_mcmc(*vel_dispersion_profile_mock_data(size = 300, plot = True, test_mode = False))

    """
    

    print('FIll me please')
    
    return()