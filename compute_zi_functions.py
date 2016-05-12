# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:11:31 2016

@author: adam
"""


import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.signal import argrelextrema

from sklearn import neighbors as nb
from sklearn.gaussian_process import GaussianProcess

from scipy.interpolate import SmoothBivariateSpline 
from scipy.interpolate import LSQBivariateSpline


def build_tree(pointcloud, leafsize):
    return nb.KDTree(pointcloud, leaf_size=leafsize)
    
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def compute_zs(t_f, params, t_f_uncert):
    """
    Take in the total freeboard (tf, float array), slope and intercept for an empirical model
    of snow depth from elevation (s_i, tuple) and the total freeboard uncertainty
    (tf_uncert, float array)

    Return a snow depth, with an associated uncertainty.
    zs, uncert = compute_zs(tf, ([slope, intercept]), tf_uncert)
    
    """
    nz_ = np.where(t_f > 0)
    #nz_ = nz_[0]
    lz_ = np.where(t_f <= 0)
    #lz_ = nz_[0] 
    z_s = np.zeros(len(t_f))
    z_s[nz_] = params[0] * t_f[nz_] + params[1]
    z_s[lz_] = 0
    z_s[z_s < 0] = 0
    z_s_uncert = params[0] * t_f_uncert
    
    return z_s, z_s_uncert

def compute_zi(tf, zs, d_ice, d_water, d_snow, sd_tf, sd_zs, sd_dsnow, \
               sd_dice, sd_dwater):
    '''
    sea ice thickness from elevation error propagation
    after Kwok, 2010; kwok and cunningham, 2008
    equations:
    4: sea ice thickness from elevation
    6: taylor series expansion of variance/covariance propagation using
        partial derivatives
    8, 9 and 10: partial derivatives used in the taylor series
    '''
    zi = (d_water / (d_water-d_ice)) * tf - ((d_water-d_snow) / \
         (d_water - d_ice)) * zs

    zi_uncert = sd_tf**2 * (d_water / (d_water - d_ice))**2 + \
           sd_zs**2 * ((d_snow - d_water) / (d_water - d_ice))**2 + \
           sd_dsnow**2 * (zs / (d_water - d_ice))**2 + \
           sd_dice**2 * (tf /  (d_water - d_ice))**2 + \
           sd_dwater**2 * (((-d_ice * tf) + ((d_ice-d_snow) * zs)) / \
           (d_water - d_ice)**2)**2

    return zi, zi_uncert

def zero_mean(data):
    """
    take in any array of data
    retuun an array modified such that mean(data) == 0
    """

    oldmean = np.mean(data)
    return data - oldmean

def find_lowpoints(xyz, leafsize, nhood):
    """
    Pass in an array of elevation vlues and an integer  number of points to use
    as a neighbourhood.
    Get back the indices of elevation values closest to the neighbourhooed median
    for neighbourhoods with more than 40 points
    """    
    
    low_tree = build_tree(xyz, leafsize)
    nhoods = low_tree.query_radius(xyz, r=nhood)
    point_filt = []   
  
    for nh in nhoods:
        #print(nhood)
        #print(xyz[nhood,2])
        # we want indices in the input cloud
        # of the minimum value in each nhood?
        if len(nh) > 5*nhood:
            point_filt.append(nh[find_nearest(xyz[nh[:],2], np.median(xyz[nh[:],2]))])
            #print(xyz[nhood,:])
        #print(new_z[i])
    
    return point_filt

def find_low_intens(intens,pntile):
    """
    Take in some return intensity data (array) and a percentile (float)
    Return the indices of chosen percentil of intensity values
    """
    li_idx =  np.where(intens <= np.percentile(intens, pntile))
    return list(li_idx[0])

def find_water_keys(xyzi, e_p, i_p):
    """
    Pass in:
    - a set of X,Y,Z and Intensity points
    - the number of points to use as a neighbourhood for choosing low elevation
    - a percentile level for intensity thresholding
    Get back:
    - a set of XYZI points corresponding to 'local sea level'
    """
    #find lowest intensity values
    low_intens_inds = find_low_intens(xyzi[:, 3], i_p)
    li_xyzi = xyzi[low_intens_inds, :]

    #find low elevations
    low_elev_inds = find_lowpoints(li_xyzi[:,0:3], 60, e_p)    
    #low_elev_inds = find_lowpoints(li_xyzi[:, 2], e_p)
    low_xyzi = li_xyzi[low_elev_inds, :]
    #low_xyzi = np.squeeze(low_xyzi)
    #return an array of xyz coordinates of points which pass the low intensity
    #and nhood filter
    return low_xyzi

def n_filter(pointcloud, tree, radius):
    '''
    takes in a point cloud (xyzi), a kDtree (generated in spatialmedian),
    and a neighbourhood.
    returns the standard deviation of points within (radius) metres
    of each point as a new point list.
    '''
    nhoods = tree.query_radius(pointcloud[:,0:3], r=radius)
    n_stats = []
    i = 0
    for nhood in nhoods:
        #print(nhood)
        n_stats.append([np.mean(pointcloud[nhood[:],2]),\
                      np.median(pointcloud[nhood[:],2]),\
                      np.std(pointcloud[nhood[:],2])])
        #print(pointcloud[i,:])
        #print(new_z[i])
        i += 1
    return n_stats

def fit_points(in_, params, knnf):
    '''
    takes in:
    - input xyzi points
    - parameters to find water points [points in nhood, intenseity percentile]    
    - KDtree to query for nhoods
    
    gives back:
    - modified xyzi points
    - a set of points used to create the regression model    
    '''
    # find open water points using find_water_keys
    xyz_fp = np.squeeze(find_water_keys(in_, params[0], params[1]))
    
    #fit a neighbourhood regression model object to those points
    this = knnf.fit(np.array([xyz_fp[:, 0], xyz_fp[:, 1]]).reshape(len(xyz_fp[:, 1]), 2), xyz_fp[:, 2])
    
    #genrate a set of corrections for fit points
    fitpoint_adjust = this.predict(np.array([xyz_fp[:,0], xyz_fp[:, 1]]).reshape(len(xyz_fp[:, 0]), 2))
    #apply them   
    fitpoints_mod = np.column_stack([xyz_fp[:,0], xyz_fp[:,1], \
                                    xyz_fp[:,2] - fitpoint_adjust])
    print('mean fitpoint adjustment (Z): {}'.format(np.mean(fitpoint_adjust)))
    
    #using the knn model, predict Z values for all lIDAR points                     
    z_fit = knnf.predict(np.array([in_[:,0], in_[:, 1]]).reshape(len(in_[:, 0]), 2))  
    
    #remove predicted values from elevations                            
    xyzi_mod = np.column_stack([in_[:,0], in_[:,1], \
                            in_[:,2] - z_fit, in_[:,3]])
                            
    return xyzi_mod, fitpoints_mod


def f_points(in_, params, dims, smth):
    '''
    find open water points using find_water_keys
    fit a surface to central tendency of keypoints
    adjust the rest of a LiDAR swath by water keypoint values
    In this method, coordinates are normalised so that 
    interpolation occurs in an almost-gridded environment which data pretty
    much fill.
    
    Best applied with a lot of smoothing
    '''
    #normalise input data to a unit block
    #http://stackoverflow.com/questions/3864899/resampling-irregularly-spaced-data-to-a-regular-grid-in-python
    # first get min/max of points
    xmin = np.min(in_[:,0])
    ymin = np.min(in_[:,1])    
    
    #translate to starting at [0,0]
    t_x = in_[:,0]-xmin
    t_y = in_[:,1]-ymin

    #normalise coordinates in each direction    
    xmax = np.max(t_x)
    ymax = np.max(t_y)
    norm_x = t_x/xmax
    norm_y = t_y/ymax
    
    #set up a translated world-space array to send to water point finding
    to_waterkeys = np.column_stack([t_x, t_y, in_[:,2], in_[:,3]])
    #print(to_waterkeys)
    
    #find water points   
    xyz_fp = np.squeeze(find_water_keys(to_waterkeys, params[0], params[1]))
    
    #translate the results to the same normalised coordinate system
    norm_fit_x = xyz_fp[:,0]/xmax
    norm_fit_y = xyz_fp[:,1]/ymax
    
    #check extents
    print('min. norm fit Y: {}, max. norm fit Y: {}'.format(min(norm_fit_y), max(norm_fit_y)))    
    
    #another 2D spline, pretty much the same as SmoothBivariateSpline
    #xcoord = np.linspace(0, 1, 0.1)
    #ycoord = np.linspace(0, 1, 0.1)
    #this_f = LSQBivariateSpline(norm_fit_x, norm_fit_y,
    #                            xyz_fp[:, 2], xcoord, ycoord)
    
    #this one is the winner right now.
    #fit a spline using normalised XY coordinates
    this_f = SmoothBivariateSpline(norm_fit_x, norm_fit_y,
                                   xyz_fp[:, 2], kx=dims[0], ky=dims[1],
                                   bbox=[0,1,0,1], s = smth*len(norm_fit_x))
                                   
    #evaluate the function at real-space coordinates
    #fpoints_mod = this_f.ev(xyz_fp[:, 0], xyz_fp[:, 1])
    #or normalised?
    fpoints_mod = this_f.ev(norm_fit_x, norm_fit_y)
    
    #one more filter - kill extreme values!
    
    adjusted_points =  xyz_fp[:, 2] - fpoints_mod
    
    e_f = np.where((adjusted_points >= 0-3*np.std(adjusted_points)) & (adjusted_points<= 0+3*np.std(adjusted_points)))

    print('mean fitpoint adjustment (Z): {}'.format(np.mean(fpoints_mod[e_f[0]])))
    
    #translate fit points back!
    fitpoints_mod = np.column_stack([xyz_fp[e_f[0], 0]+xmin, xyz_fp[e_f[0], 1]+ymin,
                                    adjusted_points[e_f[0]]])
    #evaluate the surface at 
    #z_mod = this_f.ev(in_[:, 0], in_[:, 1])
    #normalised coordinatesfrom sklearn.gaussian_process import GaussianProcess

    z_mod = this_f.ev(norm_x, norm_y)
    
    coeffs = this_f.get_coeffs()

    resid = this_f.get_residual()    
    
    #remove predicted values from elevations
    xyzi_mod = np.column_stack([in_[:, 0], in_[:, 1],
                               in_[:, 2] - z_mod, in_[:, 3]])
                               

    return xyzi_mod, fitpoints_mod, resid, coeffs



def gp_f_points(in_, params):
    '''
    This is a replicate of the function above, using an ordinary Kriging
    approach.
    
    not currently used, it's unhappy with input data being irregular
    points AFAICS, and I don't really want to grid it.
    
    The input cloud could be gridded just for the interpolation, but that's
    pretty hungry code when a smooth spline is fast and seems to do OK.
      
    '''
    #normalise input data to a unit block
    #http://stackoverflow.com/questions/3864899/resampling-irregularly-spaced-data-to-a-regular-grid-in-python
    # first get min/max of points
    
    xmin = np.min(in_[:,0])
    ymin = np.min(in_[:,1])    
    
    #translate to starting at [0,0]
    t_x = in_[:,0]-xmin
    t_y = in_[:,1]-ymin

    
    #normalise coordinates in each direction    
    #xmax = np.max(t_x)
    #ymax = np.max(t_y)
    #norm_x = t_x/xmax
    #norm_y = t_y/ymax

    to_waterkeys = np.column_stack([t_x, t_y, in_[:,2], in_[:,3]])
    print(to_waterkeys)
    #find water points   
    xyz_fp = np.squeeze(find_water_keys(to_waterkeys, params[0], params[1]))
    
    #translate these to the same normalised coordinate system
    #norm_fit_x = xyz_fp[:,0]/xmax
    #norm_fit_y = xyz_fp[:,1]/ymax
    
    #fit using a gaussian process (kriging):
    # http://stackoverflow.com/questions/24978052/interpolation-over-regular-grid-in-python
    # http://scikit-learn.org/stable/modules/gaussian_process.html
    
    gp = GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1., nugget=0.5)
    gp.fit(np.column_stack((xyz_fp[:,0], xyz_fp[:,1])), xyz_fp[:,2])
    
    fpoints_mod = gp.predict(xyz_fp[:,0], nxyz_fp[:,1])
    #fit a spline using normalised XY coordinates
    #this_f = SmoothBivariateSpline(norm_fit_x, norm_fit_y,
    #                               xyz_fp[:, 2], kx=5, ky=5)
                                   
    #evaluate the function at real-space coordinates
    #fpoints_mod = this_f.ev(xyz_fp[:, 0], xyz_fp[:, 1])
    #or normalised?
    #fpoints_mod = this_f.ev(norm_fit_x, norm_fit_y)
    
    #one more filter - kill extreme values!
    
    adjusted_points =  xyz_fp[:, 2] - fpoints_mod
    
    e_f = np.where((adjusted_points >= 0-3*np.std(adjusted_points)) & (adjusted_points<= 0+3*np.std(adjusted_points)))

    print('mean fitpoint adjustment (Z): {}'.format(np.mean(fpoints_mod[e_f[0]])))
    
    #translate fit points back!
    fitpoints_mod = np.column_stack([xyz_fp[e_f[0], 0]+xmin, xyz_fp[e_f[0], 1]+ymin,
                                    adjusted_points[e_f[0]]])
    #evaluate the surface at 
    #z_mod = this_f.ev(in_[:, 0], in_[:, 1])
    #normalised coordinates
    #z_mod = this_f.ev(norm_x, norm_y)    
    z_mod = gp.predict(t_x, t_y)
    
    #remove predicted values from elevations
    xyzi_mod = np.column_stack([in_[:, 0], in_[:, 1],
                               in_[:, 2] - z_mod, in_[:, 3]])

    return xyzi_mod, fitpoints_mod
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------