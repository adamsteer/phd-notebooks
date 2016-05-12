# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 22:43:48 2015

@author: adam
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.signal import argrelextrema
from sklearn import neighbors as nb
from shapely import geometry
#from scipy.signal import medfilt

infile_dir = '../lidar.matlab/swaths/'
infile = 'f22_1milao102506_102714_c.xyz'
outfilename = infile[0:-4]+'_zi.xyz'


#density parameters, should add as a data dictionary
d_snow = 305.67 #mean of all EA obs
sd_dsnow = 10
d_ice = 891 #empirically derived from matching with AUV draft
sd_dice = 10
d_water = 1028 #Hutchings2015
sd_dwater = 1

#sipex2 snow model
s_i = ([0.701, -0.0012])

started = datetime.now()
print(started)

def compute_zs(tf, s_i, tf_uncert):
    """
    Take in the total freeboard (tf, float array), slope and intercept for an empirical model
    of snow depth from elevation (s_i, tuple) and the total freeboard uncertainty
    (tf_uncert, float array)

    Return a snow depth, with an associated uncertainty.
    zs, uncert = compute_zs(tf, ([slope, intercept]), tf_uncert)
    """
    
    zs = (s_i[0] * tf) + s_i[1]

    zs_uncert = 0.72 * tf_uncert

    return zs, zs_uncert

def compute_zi(tf, zs, d_ice, d_water, d_snow, sd_tf, sd_zs, sd_dsnow, \
               sd_dice, sd_dwater):
#sea ice thickness from elevation error propagation
#after Kwok, 2010; kwok and cunningham, 2008
#equations:
# 4: sea ice thickness from elevation
# 6: taylor series expansion of variance/covariance propagation using
#    partial derivatives
# 8, 9 and 10: partial derivatives used in the taylor series

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

def find_lowpoints(elev_data, nhood):
    """
    Pass in an array of elevation vlues and an integer  number of points to use
    as a neighbourhood.
    Get back the indices of local miminum elevation values for the chosen
    neighbourhoods
    """
#http://docs.scipy.org/doc/scipy-0.16.1/reference/generated/
# scipy.signal.argrelextrema.html#scipy.signal.argrelextrema
    #as a tuple...

    le_idx = argrelextrema(np.array(elev_data), np.less, order=nhood)
    return list(le_idx[0])

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
    low_elev_inds = find_lowpoints(li_xyzi[:, 2], e_p)
    low_xyzi = li_xyzi[low_elev_inds, :]
    #low_xyzi = np.squeeze(low_xyzi)

    return low_xyzi

def query_tree(pointcloud, tree, radius):
    nhoods = tree.query_radius(pointcloud[:,0:3], r=radius)
    #new_z = np.median(xyzi[nhoods[0],2])

    new_z = []
    i = 0
    for nhood in nhoods:
        #print(nhood)
        new_z.append(np.nanmean(pointcloud[nhood[:],2]))

        i += 1
    return new_z

def spatialmedian(pointcloud,radius):
    """
    Using a KDTree and scikit-learn's query_radius method,
    ingest an n-d point cloud with the XYZ coordinates occupying
    the first three columns respectively (pointcloud[:,0:3]) and a radius in
    metres (or whatever units the pointcloud uses).
    Returns a new set of Z values which contain the median Z value of points within
    radius r of each point.
    """

    from sklearn import neighbors as nb

    from datetime import datetime
    startTime = datetime.now()

    print(startTime)

    tree = nb.KDTree(pointcloud[:,0:3], leaf_size=60)

    print(datetime.now() - startTime)

    nhoods = tree.query_radius(pointcloud[:,0:3], r=radius)
    #new_z = np.median(xyzi[nhoods[0],2])

    new_z = []
    i = 0
    for nhood in nhoods:
        #print(nhood)
        new_z.append(np.median(pointcloud[nhood[:],2]))
        #print(pointcloud[i,:])
        #print(new_z[i])

        i += 1

    print(datetime.now() - startTime)

    # we really want to retugn the tree - only generate it once per operation!
    return new_z, tree

input_points = np.genfromtxt(infile_dir + infile)

#xyz = float32(input_points[:, 1:4])
xyzi = input_points[:, 1:5]




#adjust lidar to a reference level
#find the lowest point
lowestpoint = np.min(input_points[:, 3])

#find the sd of intensity
sd_intens = np.std(input_points[:, 4])

#compute a reference point set

xyz_fitpoints = np.squeeze(find_water_keys(xyzi, 200, 1.2))


knnf = nb.KNeighborsRegressor(10, algorithm='kd_tree', n_jobs=-1)
#knn = nb.RadiusNeighborsRegressor(10, algorithm='kd_tree', weights = 'distance', n_jobs=-1)
#knnf = nb.RadiusNeighborsRegressor(10, algorithm='kd_tree', n_jobs=-1)

knnf.fit(np.array([xyz_fitpoints[:, 0], xyz_fitpoints[:, 1]]).reshape(len(xyz_fitpoints[:, 1]), 2), xyz_fitpoints[:, 2])

#z_mod = clf.predict(np.array([xyzi[100000::500,0],xyzi[100000::500,1]]).reshape(len(xyzi[100000::500,0]),2))

fitpoint_adjust = knnf.predict(np.array([xyz_fitpoints[:,0], xyz_fitpoints[:, 1]]).reshape(len(xyz_fitpoints[:, 0]), 2))

z_fit = knnf.predict(np.array([xyzi[:,0], xyzi[:, 1]]).reshape(len(xyzi[:, 0]), 2))



xyz_fitpoints_adj = np.column_stack([xyz_fitpoints[:,1], xyz_fitpoints[:,2], \
                            xyz_fitpoints[:,3] - fitpoint_adjust])

'''
elevations are set ab0ut the Y = 0 plane here!!
'''
xyzi_mod = np.column_stack([input_points[:,1], input_points[:,2], \
                            input_points[:,3] - z_fit, input_points[:,4]])


median_z, points_kdtree = spatialmedian(xyzi_mod, 2)

#median_z = query_tree(xyzi, points_kdtree, 3)


#replace xyzi Z with median Z

xyzi2 = np.column_stack([input_points[:,1], input_points[:,2], \
                            median_z, input_points[:,4]])

plt.plot(xyzi2[:,2] - xyzi_mod[:,2])


#np.savetxt('xyz_fitpoints_pass1_nr2_100_1.2.txt', xyz_fitpoints, fmt='%.5f')

###
#and make snow/ice thicknesses

tf = np.array(median_z)
#tf = input_points[:,3]-z_mod

tf[np.where(tf < 0)] = 0

sd_tf = input_points[:, 8]

zs, zs_uncert = compute_zs(tf, s_i, sd_tf)

#should really be passing a dictionary here...
zi, zi_uncert = compute_zi(tf, zs, d_ice, d_water, d_snow, sd_tf, \
                           zs_uncert, sd_dsnow, sd_dice, sd_dwater)

np.mean(zs)
np.std(zs)

np.mean(zi)
np.median(zi)
np.std(zi)


'''
need draft as well. We compute ZI, ZS and TF, so FB =  TF-ZS,
and draft = (TF-ZS) - ZI


'''
outfile = np.column_stack((input_points[:, 0], input_points[:, 1], \
                           input_points[:, 2], tf.T, sd_tf.T, zi.T, \
                           zi_uncert.T, \
                           (tf.T-zs.T) - zi.T, \
                           np.sqrt(sd_tf**2 + zi_uncert**2 + zs_uncert**2).T, \
                           tf.T-zs.T, np.sqrt(sd_tf**2 + zs_uncert**2), \
                           zs.T, zs_uncert.T))

with open(outfilename, 'wb') as f:
    f.write(b'GPS_secs X Y Ft sd_Ft Zi sd_Zi Df sd_Df Fi sd_Fi Zs sd_Zs\n')
    np.savetxt(f, outfile, fmt='%.5f')

np.savetxt( infile[0:-4]+'mf_4m_xyzi.xyz', xyzi2)
np.savetxt( infile[0:-4]+'uf_xyzi.xyz', xyzi_mod)
np.savetxt( infile[0:-4]+'keypoints.xyz', xyz_fitpoints_adj)
