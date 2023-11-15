import numpy as np
from math import *

"""

        This module includes costume functions to do bining and other statistics related to this

            FlexBins -   computes bins based on the percentiles of data and number of bins in a given percentile
            RuningMean - computes array of running mean and dispersion in each bin

"""

def doc():
    bla1 = "        This module includes costume functions to do bining and other statistics related to this:\n"
    bla2 = "            flex_bins -   computes bins based on the percentiles of data and number of bins in a given percentile\n"
    bla3 = "            runing_mean - computes array of running mean and dispersion in each bin"
    print(bla1, bla2, bla3)

def flex_bins(array, percentiles, bins):

    """

        This piece of code computes bin edges for (n+1) different regimes based on the input percentiles.
        The required input is array to be binned, the array of percentiles (len = n)
        and array of number of bins in each binning region (len = n+1) !!!this has to be sorted in the same
        way as the percentile!!!!

        The function returns array of binned data, edges and number of bins in the array

    """


    min, max = np.min(array), np.max(array)
    loc_percentiles = np.percentile(array, percentiles)

    edges = np.array([])
    for index, location in enumerate(loc_percentiles):
        if index == 0 :
            bla = np.linspace(min-min/10, location, num = bins[index])
            edges = np.concatenate((edges, bla), axis=0)
        elif index == (len(loc_percentiles)-1):
            bla = np.linspace(loc_percentiles[index-1], location, num = bins[index])
            edges = np.concatenate((edges, bla[1:]), axis = 0)
            bla = np.linspace(location, max+max/10, num = bins[index+1])
            edges = np.concatenate((edges, bla[1:]), axis=0)
        else:
            bla = np.linspace(loc_percentiles[index-1], location, num = bins[index])
            edges = np.concatenate((edges, bla[1:]), axis = 0)

    return(np.histogram(array, bins = edges), len(edges)+1)

def runing_mean(sorting_array, sorted_array, bin_size, step_size):
    """
        The function returns running mean of the sorting_array as well as
        running mean and standard deviation of the sorted_array
    """
    sort_r = np.sort(sorting_array)
    arg_sort = np.argsort(sorting_array)
    sorted = np.zeros(len(sorted_array))
    for index, i in enumerate(arg_sort):
        sorted[index] += sorted_array[i] # sort the velocities acording to the sorting sorted_array

    mean = np.array([])
    std = np.array([])
    mean_r = np.array([])
    for i in np.arange(0, (len(sorted_array)-int(bin_size/2)), step_size):
        mean = np.append(mean, np.mean( sorted[i:i+bin_size] ) )
        std = np.append(std, np.std( sorted[i:i+bin_size] ) )
        mean_r = np.append(mean_r, np.mean(sort_r[i:i+bin_size]))
    return(mean_r, mean, std)

def runing_weighted_mean(sorting_array, sorted_array, Gaussian_error, bin_size, step_size):
    """
        The function returns running weighted mean of the sorting_array as well as
        running mean and the error on the weighted mean of the sorted_array.
        Weight is considered to be 1 sigma gaussian error and will be converted:

        weight = 1/Gaussian_error**2
        weigh_mean = sum ( measurments * weight) / sum( weight )
        err_weight_mean = sqrt(1 / sum( weight  ))


        Calling sequence:


        mean_r, master_weighted_mean, master_weighted_error =
                runing_weighted_mean(R, pos["vlos"].value, pos["err_vlos"].value, bin_size, step_size)

    """
    sort_r = np.sort(sorting_array) # array sorted by projected radius
    arg_sort = np.argsort(sorting_array) # array of sorted indices
    sorted_values = np.zeros(len(sorted_array))
    sorted_errors = np.zeros(len(sorted_array))
    for index, i in enumerate(arg_sort):
        sorted_values[index] += sorted_array[i] # sort the values acording to the sorting sorted_array
        sorted_errors[index] += Gaussian_error[i] # sort the errors acording to the sorting sorted_array

    master_weighted_mean, master_weighted_error = np.array([]), np.array([])
    mean_r = np.array([]) # array of radius
    for i in np.arange(0, (len(sorted_array)-int(bin_size/2)), step_size):
        weights = 1 / sorted_errors[i:i+bin_size]**2 # weights are computed
        weighted_mean = np.sum(sorted_values[i:i+bin_size] * weights) / np.sum(weights)

        master_weighted_mean = np.append(master_weighted_mean, weighted_mean)
        master_weighted_error = np.append(master_weighted_error, np.sqrt(1/ np.sum(weights)))

        mean_r = np.append(mean_r, np.mean(sort_r[i:i+bin_size]))
        
    return(mean_r, master_weighted_mean, master_weighted_error)



