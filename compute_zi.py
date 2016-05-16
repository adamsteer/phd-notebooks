#fist run compute_zi_functions

from compute_zi_functions import *


infile_dir = '../lidar.matlab/swaths/'
infile = 'is6_f11_pass1_aa522816_523019_c.xyz'

print(infile)
outfilename = infile[0:-4]+'_zi.xyz'

#density parameters, should add as a data dictionary
d_snow = 326.31 #SIPEX2 snow
#d_snow = 305.67 #mean of all EA obs
sd_dsnow = 10
d_ice = 915.6 #empirically derived from matching with AUV draft
sd_dice = 10
d_water = 1028 #Hutchings2015
sd_dwater = 1

#All EA snow model
s_i = ([0.701, -0.0012])

started = datetime.now()
print(started)

#import some points
input_points = np.genfromtxt(infile_dir + infile)

# crop any that are likely to be air returns
input_points = input_points[input_points[:,3] < -5,:]

#input_points = input_points[1:2000000,:]

#early on, let's find water points and trim the data such that 
# it starts and ends with water.
print('finding water keys to define data limits (swath must start and end with water)')

#build this tree once
#low_tree = build_tree(input_points[:,1:4], 60)

water_points = np.squeeze(find_water_keys(input_points[:,1:5], 20, 1.5, 40))

print('x limits: {} - {}'.format(np.min(water_points[:,0]), np.max(water_points[:,0])))
print('y limits: {} - {}'.format(np.min(water_points[:,1]), np.max(water_points[:,1])))

p1 = len(input_points[:,0])

input_points = input_points[(input_points[:,2] >= np.min(water_points[:,1]))&\
                            (input_points[:,2] <= np.max(water_points[:,1]))]
p2 = len(input_points[:,0])

print('trimming data: {} points trimmed from {}'.format(p1-p2, p1))
del water_points, p1, p2

#grab an xyzi subset to work with
xyzi_ = input_points[:, 1:5]

print('x limits: {} - {}'.format(np.min(xyzi_[:,0]), np.max(xyzi_[:,0])))
print('y limits: {} - {}'.format(np.min(xyzi_[:,1]), np.max(xyzi_[:,1])))
print('total points: {}'.format(len(xyzi_[:,1])))

#testingregression methods
# build a KD tree for neighbourhood regression
#knnf = nb.KNeighborsRegressor(500, algorithm='ball_tree', n_jobs=-1)
#experimental!!!!
#knn = nb.RadiusNeighborsRegressor(10, algorithm='kd_tree', weights = 'distance', n_jobs=-1)
#knnf = nb.RadiusNeighborsRegressor(10, algorithm='ball_tree', n_jobs=-1)

#using a loop, recursively iterate fit points toward 'level'

i = 0
while i < 1:
    #xyzi_, keypoints = fit_points(xyzi_, [10, 2.0], knnf)
    xyzi_, keypoints, resid, coeff = f_points(xyzi_, [10, 1.5], [1 ,3], 20, 40 )
    print('keypoints: {}'.format(len(keypoints)))
    print('keypoints mean elev: {}'.format(np.mean(keypoints[:,2])))
    print('mean elev: {}'.format(np.mean(xyzi_[:,2])))
    
    print(i)
    i += 1


#xyzi_, keypoints = fit_points(xyzi_, [10, 2.0], knnf)
#print('keypoints: {}'.format(len(keypoints)))
#print('mean elev: {}'.format(np.mean(xyzi_[:,2])))

##assuming elev > 5m = iceberg!
berg_h = 10
nobergs = np.where(xyzi_[:,2] < berg_h)
xyzi_ = xyzi_[nobergs[0],:]
input_points = input_points[nobergs[0],:]

# because water is still too high!
#adjust = np.percentile(keypoints[:,2],1)
#print('adjustment for heights after model fitting: {}'.format(adjust))
#xyzi_[:,2] = xyzi_[:,2] + adjust


print('building KDtree')
#one last tree build
startTime = datetime.now()
points_kdtree = build_tree(xyzi_[:,0:2], 60)
print('time to build tree: {}'.format(datetime.now() - startTime))

#n_stats returns mean, median, sd in order for a 1.5m radius nhood
startTime = datetime.now()
nhood_stats = np.array(n_filter(xyzi_, points_kdtree, 1))
print('time to generate n stats: {}'.format(datetime.now() - startTime))

#median_z = query_tree(xyzi, points_kdtree, 3)
#replace xyzi Z with median Z [:,1]
# or mean_z [:,0]
xyzi2 = np.column_stack([input_points[:,1], input_points[:,2], \
                            nhood_stats[:,1], input_points[:,4]])

#rproxy uses a wider nhood. Here, 10m radius!
startTime = datetime.now()
r_proxy = np.array(n_filter(xyzi_, points_kdtree, 5.5 ))
print('time to generate n stats: {}'.format(datetime.now() - startTime))

###
#and make snow/ice thicknesses

#smooth things just a little
tf = nhood_stats[:,1]

#or keep them raw
#tf = xyzi_[:,2]

print('first few TF points: {}'.format(tf[0:3]))
#tf = input_points[:,3]-z_mod

sub_z = np.where(tf<0)
print('total points where total freeboard < 0 (reassigned to 0): {}'.format(len(sub_z[0])))
tf[sub_z] = 0

sd_tf = input_points[:, 8]

zs, zs_uncert = compute_zs(tf, s_i, sd_tf)

sub_z = np.where(zs<0)
print('total points where snow < 0 (reasigned to 0): {}'.format(len(sub_z[0])))
zs[sub_z] = 0

#oversnowed = np.where(zs > tf)
#print('total oversnowed points (zs > tf, zs reassigned to 0): {}'.format(len(oversnowed[0])))
#zs[oversnowed] = 0

#should really be passing a dictionary here...
zi, zi_uncert = compute_zi(tf, zs, d_ice, d_water, d_snow, sd_tf, \
                           zs_uncert, sd_dsnow, sd_dice, sd_dwater)

#probably want to try and kill icebergs...

print('assuming tF > {} m is a berg...'.format(berg_h))

print('min snow depth: {}'.format(np.min(zs)))
print('max snow depth: {}'.format(np.max(zs)))
print('mean snow depth: {}'.format(np.mean(zs)))
print('snow depth sd: {}'.format(np.std(zs)))

print('min total freeboard: {}'.format(np.min(tf)))
print('max total freeboard: {}'.format(np.max(tf)))
print('mean total freeboard: {}'.format(np.mean(tf)))
print('total freeboard sd: {}'.format(np.std(tf)))

print('min ice thickness: {}'.format(np.min(zi)))
print('max ice thickness: {}'.format(np.max(zi)))
print('mean ice thickness: {}'.format(np.mean(zi)))
print('median ice thickness: {}'.format(np.median(zi)))
print('ice thickness sd: {}'.format(np.std(zi)))

print('x limits: {} - {}'.format(np.min(xyzi2[:,0]), np.max(xyzi2[:,0])))
print('y limits: {} - {}'.format(np.min(xyzi2[:,1]), np.max(xyzi2[:,1])))


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
    
#keep bergs in these data
with open(infile[0:-4]+'nstats.xyz', 'wb') as f:
    f.write(b'GPS_secs X Y mean median sd\n')
    np.savetxt(f, np.column_stack((input_points[:, 0], input_points[:, 1], \
                                   input_points[:, 2], nhood_stats[:,0],\
                                   nhood_stats[:,1], nhood_stats[:,2])),\
                                   fmt='%.5f')

with open(infile[0:-4]+'rproxy.xyz', 'wb') as f:
    f.write(b'GPS_secs X Y mean median sd\n')
    np.savetxt(f, np.column_stack((input_points[:, 0], input_points[:, 1], \
                                   input_points[:, 2], r_proxy[:,0],\
                                   r_proxy[:,1], r_proxy[:,2])),\
                                   fmt='%.5f')

#np.savetxt( infile[0:-4]+'mf_4m_xyzi.xyz', xyzi2)
#np.savetxt( infile[0:-4]+'uf_xyzi.xyz', xyzi_)
np.savetxt( infile[0:-4]+'keypoints.xyz', keypoints)
