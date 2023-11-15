import numpy as np
from math import *
import datetime
from astropy import table, units as u
from astropy.table import QTable
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import ascii

import mgetools
import genhernquist

"""

    Custom package that includes various coordinate transformations
    as well as transformation to CJAM input from 1D fit either to surface
    or volume density.
"""

def SkyCoord_to_Rad(coor, raise_warning = False, string = ""):
    """
        converts sky coordinates (RA, DEC) in any astropy format to radians or
        as numpy array in degrees.
    """
    try:
        # if coordinates are given as SkyCoord object then they can be given in
        # any format and be converted to radians to evaluate the  trigonometric
        # functions in numpy
        ra = (coor.ra).to( u.rad).value
        dec = (coor.dec).to( u.rad).value

    except:
        # in the case when no units are given to the coordinates the program
        # assumes the following form: coordinates = [ra[deg], dec[deg]]
        # and converts to radians
        if raise_warning:
            print("WARNING: "+ string +" coordinates are not in SkyCoord format."+\
            "\nAssuming format structure (ra[deg], dec[deg])")
        ra = coor[0]*np.pi/180
        dec = coor[1]*np.pi/180

    return(ra,dec)

def projected_cartesian(coordinates, center_coordinates = np.array([0,0]),\
    raise_warning = False, output_unit = u.arcsec):
    """
        Converting from arbitrary right accension and declination
        to projected cartesian coordinate system.
        van de Ven+06 equation (1).

        !! Where x is the direction of West and y of North.

        Delta RA = RA - RA_center
        Delta DEC = DEC - DEC_center

        x = - 180/pi cos(DEC) sin(Delta RA) [deg]
        y = 180/pi {(sin(DEC) cos(DEC_center)) -
                    (cos(DEC) sin(DEC_center) sin(Delta RA))} [deg]

        Input:
            coordinates             Can be either as SkyCoord objects or numpy array in degrees
            center_coordinates      Coordinates of the center. By defauls 0,0.
                                    Can be either as SkyCoord objects or numpy array in degrees
            raise_warning           Raises warning if coordinates and/or center_coordinates
                                    do not have units, are not SkyCoord objects.
                                    False by default
            output_unit             the units of output. Accepted: u.arcsec (default),
                                    u.arcmin, u.deg.

        Output:
            x, y                    projected cartesian coordinates in units corresponding
                                    to output_unit.

        Changelog:
        v1: 24.5.2021 created. Implemented ra and dec in SkyCoordformat and numpy array
    """

    ra, dec = SkyCoord_to_Rad(coordinates, raise_warning, string = "object")
    ra_center, dec_center = SkyCoord_to_Rad(center_coordinates, raise_warning,\
                                            string = "center")

    Dra = ra - ra_center
    Ddec = dec - dec_center

    r0 = 180/np.pi # conversion from radians to degrees

    x = - r0 * np.cos(dec) * np.sin(Dra)
    y = r0 * (np.sin(dec ) * np.cos(dec_center) -\
                np.cos(dec) * np.sin(dec_center) *np.cos(Dra))

    if output_unit == u.arcsec:
        return(x*3600, y*3600)

    elif output_unit == u.arcmin:
        return(x*60, y*60)

    elif output_unit == u.deg:
        return(x, y)

    else:
        print("WARNING: Please restate the output unit\n"+\
                str(output_unit) + " not identified as an angular unit.\n"+
                "Returning calues in degrees.")
        return(x, y)


def angular_to_physical_units(x, distance, output_unit = u.pc,\
                                raise_warning = False):
    """
        Conversion from arcsec to parsec. If units not given then arcsec and kpc
        are assumed for x and distance respectively.

        input:
            x               angular size on the plane of the sky to be converted to physical [arcsec]
            distance        distance to the object in physical units [kpc]
            output_unit     the units of output [pc] by default
            raise_warning   raises warning if x and distance have no unit. False by default

        return:
            x               in physical units

        Changelog:
        v1: 24.5.2021 created. Conversion from angular to physical units
    """

    try:
        x_rad = x.to(u.rad).value
    except:
        if raise_warning:
            print("WARNING: x without units. Assuming arcsec.")
        x_rad = x/3600 * np.pi/180  # converting to radians

    try:
        distance_unit = distance.to(output_unit).value
    except:
        if raise_warning:
            print("WARNING: distance without units. Assuming kpc.")
        distance_unit = (distance*u.kpc).to(output_unit).value

    x_physical = distance_unit * np.tan(x_rad)
    #print(x_physical)

    return(x_physical)

def physical_to_angular_units(x, distance, output_unit = u.arcsec,\
                                raise_warning = False):
    """
        Conversion from parsec to arcsec. If units not given then pc and kpc
        are assumed for x and distance respectively.

        input:
            x               physical size on the plane of the sky to be converted to physical [pc]
            distance        distance to the object in physical units [kpc]
            output_unit     the units of output [arcsec] by default
            raise_warning   raises warning if x and distance have no unit. False by default

        return:
            x               in angular units

        Changelog:
        v1: 24.5.2021 created. Conversion from physical to angular units
    """

    try:
        x_kpc = x.to(u.kpc).value
    except:
        if raise_warning:
            print("WARNING: x without units. Assuming pc.")
        x_kpc = x/1000   # converting from pc to kpc

    try:
        distance_unit = distance.to(u.kpc).value
    except:
        if raise_warning:
            print("WARNING: distance without units. Assuming kpc.")
        distance_unit = distance

    x_angular = (np.arctan(x_kpc/distance_unit)*u.rad).to(output_unit).value

    return(x_angular)

def density_1Dmge_to_CJAMmge(solution, distance, projected_flattening = 1,\
                fitted_profile_units = u.Lsun/u.arcsec**2, n_gauss_start = 1,\
                ML = 1, save = "", print_out = True, additional_comment = ""):

    """
        Converting the output of mge_fit_1d to CJAM input. The routine is flexible to
        account for fiting of number or volume density in physical or angular units.

        solution                Output of mge_fit_1d
        distance                Must be in units
        projected_flattening    Projected flattening if circularised radii are
                                used set to !=1
        fitted_profile_units    Units of the original denisty profile fitted
                                with the mge_fit_1d. By default: u.Lsun/u.arcsec**2
                                Options: u.Lsun/u.arcsec**2 and u.Msun/pc**3
        n_gauss_start           If for some reason you are concating two mge
                                tables and need one to start with a different number
        ML                      set mass to light ratio and add units if applicible
                                u.Msun/u.Lsun
        save                    name of the saved file. Do not include format type in the name
                                the code will add it automatically.
        print_out               True/False whether you want the table to be returned or not
        additional_comment      additional comments to be printed in the saved file

    """

    try:
        a = distance.unit
    except:
        print('ERROR: Distance has no unit!\nNothing to do here: QUITTING')
        return

    if fitted_profile_units.bases[1] == u.arcsec:
        # checking the power of fitted_profile_units is what we expect for
        # surface density fit
        assert fitted_profile_units.powers[1] == -2, "The Surface density is not in the right units!"

        sigma_pc = angular_to_physical_units(solution[1,:]*u.arcsec, distance,\
                                    output_unit = u.pc, raise_warning = True)

        # converting u.Lsun/u.arcsec to u.Lsun/u.pc
        i = solution[0,:]/angular_to_physical_units(1*u.arcsec, distance,\
                                    output_unit = u.pc)
        # converting u.Lsun/u.pc to u.Lsun/u.pc**2
        # if using circularised radii then the height of gaussians
        # has to be corrected for projected flattening q'
        i /= (np.sqrt(2*np.pi)*sigma_pc * projected_flattening)
        sigma_arcsec = solution[1,:]

    elif fitted_profile_units.bases[1] == u.pc:
        # checking the power of fitted_profile_units is what we expect for
        # volume density fit
        assert fitted_profile_units.powers[1] == -3, "The volume density is not in the right units!"

        sigma_arcsec = physical_to_angular_units(solution[1,:]*u.pc, distance,\
                                    output_unit = u.arcsec, raise_warning = True)
        i = solution[0,:]

    else:
        print("WARNING: the units of fitted_profile_units not compatible with" +\
        " the routine! Add new functions or restate the units!")
        return

    # array of indexes of

    n = np.arange(n_gauss_start, n_gauss_start + len(sigma_arcsec) , 1)
    q = np.ones_like(n) * projected_flattening
    units = ['', fitted_profile_units.bases[0] /u.pc**2 , u.arcsec, '']

    if len(save)>1:
        save_format = 'ecsv'
        save += '.'+save_format
        # comments to include in the saved file
        comment = "File created on: " + str(datetime.datetime.now()) + additional_comment +\
            " Read with this command: ascii.read(\"" + save + "\")"

        tabl = QTable([n, i, sigma_arcsec, q], names=('n', 'i', 's', 'q'),\
                    meta={'comment': comment})
        # adding units
        for jdx, name in enumerate(tabl.colnames):
            tabl[name].unit = units[jdx]

        tabl['i'] *=ML

        ascii.write(tabl, save, format = save_format, formats={'i': '%4.4e', 's': '%4.4e'},\
                     overwrite=True)
    else:

        tabl = QTable([n, i, sigma_arcsec, q], names=('n', 'i', 's', 'q'))
        # adding units
        for jdx, name in enumerate(tabl.colnames):
            tabl[name].unit = units[jdx]
        tabl['i'] *=ML
    if print_out:
        return(tabl)

    else:
        return

def derot(x, y, angle):
    """
       Returns 2 vectors rotated by -angle
       angle should be in degrees
    """
    
    v = np.column_stack((x, y)).T
    
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    bla = np.array(((c, s), (-s, c)))
        
    x2, y2 = bla.dot(v)    
    return (x2, y2)

def projected_ellipse_parameter(x,y):
    """
       The routine computes the position angle, counter-clockwise from the x-axis, projected ellipticity and semi major axis.
       The semimajor axis however will not be correct for a distribution that is elliptical!!! So be caureful when using it.
       
       The following snippets are taken from Capellari find_galaxy class
       
       q' = 1 - eps
    """    

    i = len(x)
    x2 = np.sum(x**2)/i
    y2 = np.sum(y**2)/i
    xy = np.sum(x*y)/i
    theta = np.degrees(np.arctan2(2*xy, x2 - y2)/2.) #+ 90.
    a2 = (x2 + y2)/2.
    b2 = np.sqrt(((x2 - y2)/2.)**2 + xy**2)
    eps = 1. - np.sqrt((a2 - b2)/(a2 + b2))
    majoraxis = np.sqrt(np.max(x**2 + y**2))
    
    return(theta, eps, majoraxis)