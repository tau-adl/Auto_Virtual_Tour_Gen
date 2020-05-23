import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import pandas
import pandas as pd
from sklearn.cluster import KMeans

#%%
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

#%% input data
    
# cloud point map file
# filename = r'C:\Users\shirang\Documents\myStuff\Project\3d_recon\Data\gerrard-hall\sparse\point_cloud.ply'
# filename = r'C:\Users\shirang\Documents\myStuff\Project\fwdorb_slam_map_points\map_point_orb.txt'
#filename = r'/home/yoni/Documents/vtour_project/point_clouds/map_point1.txt'
#filename = r'/home/yoni/Documents/vtour_project/point_clouds/map_point2.txt'
# filename = r'/home/yoni/Documents/vtour_project/point_clouds/map_point3.txt'
# filename = r'point_clouds/map_point3.txt'
filename = r'point_clouds/map_point.txt'

# location of z axis
x_axis = 0
y_axis = 1
z_axis = 2 # the z coordinates location in the point cloud

# visualization flag
vis = True

# params of Hough
rho_res = 1 # [pixels]
theta_res_deg = 1 # [degrees]

threshold_first_line = 50 # hough param
search_range_deg = 5
threshold_walls_lines = 20 # hough param
n_lines_to_use = 50 #to find the 2 required walls

values_scale = 100 # when conevrting to image

#%%
# change delimeter from ['] to [ ]
# convert comma delimiter into dataframe
dataframe = pandas.read_csv(filename, delimiter=",")
# write dataframe into CSV
dataframe.to_csv("map_point_edited.txt", sep=" ", index=False)

# load point cloud from edited file
print("Load a point cloud and print it")
pcd = o3d.io.read_point_cloud("map_point_edited.txt", format='xyz')
print(pcd)
print(np.asarray(pcd.points))

if vis:
    print("Visualize the original point cloud")
    o3d.visualization.draw_geometries([pcd])

#%% shrink z axis and visualize
if vis:  
    pcd_xy = pcd.__copy__()
    for i in range(0, len(pcd_xy.points)):
        pcd_xy.points[i][z_axis] = 0
    print("Visualize point cloud in 2D")
    o3d.visualization.draw_geometries([pcd_xy])

#%% plot a histogram of axes values of all points
if vis:
    print("Histograms of axes values of all points")
    df = pd.DataFrame(pcd.points)
    
    df[x_axis].hist()
    plt.title('x')
    plt.show()
    
    df[y_axis].hist()
    plt.title('y')
    plt.show()  
    
    df[z_axis].hist()
    plt.title('z')
    plt.show()



#%% outlier removal
print("Statistical outlier removal")
# nb_neighbors allows to specify how many neighbors are taken into account in order to calculate the average distance for a given point.
# std_ratio allows to set the threshold level based on the standard deviation of the average distances across the point cloud.
# The lower this number the more aggressive the filter will be.
pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=3.0)

if vis:
    print("Visualize point cloud after outlier removal")
    
    # a plot where outlier points are in red
    display_inlier_outlier(pcd, ind)
    
    # a regular plot without outlier points
    o3d.visualization.draw_geometries([pcd_clean])
    
#%% shrink z axis after outlier removal and visualize
if vis:
    print("Visualize point cloud in 2D after outlier removal")
    pcd_clean_xy = pcd_clean.__copy__()
    for i in range(0, len(pcd_clean_xy.points)):
        pcd_clean_xy.points[i][z_axis] = 0
    o3d.visualization.draw_geometries([pcd_clean_xy])

#%% filter points by z value - from a certain height
# idea: remove points close to the floor, as they might represent objects rather than walls

# convert Open3D.o3d.geometry.PointCloud to numpy array
arr = np.asarray(pcd_clean.points)
print("before z filtering: ", arr.shape[0] , " points")

percentile = 0


## opt1 - different thresh to different areas of the point cloud
#def filter_partial(arr, percentile):
#    # inds - indices of points taking part in the filtering
#    # (the rest are ignored)
#    thresh = np.percentile(arr[:, z_axis], percentile)
#    arr_filt = arr[arr[:, z_axis]>thresh, :]
#    return arr_filt
#
## calculate a threshold base on data distribution
#y_med = np.median(arr[:, y_axis])
#x_med = np.median(arr[:, x_axis])

#inds = (arr[:, x_axis]>x_med) & (arr[:, y_axis]>y_med)
#arr1 = filter_partial(arr[inds,:], percentile)
#
#inds = (arr[:, x_axis]<x_med) & (arr[:, y_axis]>y_med)
#arr2 = filter_partial(arr[inds,:], percentile)
#
#inds = (arr[:, x_axis]>x_med) & (arr[:, y_axis]<y_med)
#arr3 = filter_partial(arr[inds,:], percentile)
#
#inds = (arr[:, x_axis]<x_med) & (arr[:, y_axis]<y_med)
#arr4 = filter_partial(arr[inds,:], percentile)
#
#arr_filt_z = np.concatenate((arr1, arr2, arr3, arr4), axis=0)
#print("after z filtering: ", arr_filt_z.shape[0] , " points")


## opt2 - single thresh to whole point cloud
thresh = np.percentile(arr[:, z_axis], percentile)
print("z filtering by thresh: ", thresh)

arr_filt_z = arr[arr[:, z_axis]>thresh, :]
print("after z filtering: ", arr_filt_z.shape[0] , " points")


# back from numpy array to point cloud object
pcd_clean_zfilt = o3d.geometry.PointCloud()
pcd_clean_zfilt.points = o3d.utility.Vector3dVector(arr_filt_z)

# visualize z-filtered point cloud
if vis:
    o3d.visualization.draw_geometries([pcd_clean_zfilt])
    
    print("Histograms of z axis after z filtering")
    df = pd.DataFrame(pcd_clean_zfilt.points)
    df[z_axis].hist()
    plt.title('z')
    plt.show()
    

#%% shrink z axis and plot
if vis:
    pcd_clean_zfilt_xy = pcd_clean_zfilt.__copy__()
    for i in range(0, len(pcd_clean_zfilt_xy.points)):
        pcd_clean_zfilt_xy.points[i][z_axis] = 0
    o3d.visualization.draw_geometries([pcd_clean_zfilt_xy])


#%% create an image from list of points in order to use Hough transform

# convert Open3D.o3d.geometry.PointCloud to numpy array
pcd_clean_zfilt_xy = pcd_clean_zfilt.__copy__()
for i in range(0, len(pcd_clean_zfilt_xy.points)):
    pcd_clean_zfilt_xy.points[i][z_axis] = 0
arr = np.asarray(pcd_clean_zfilt_xy.points)[:, [x_axis,y_axis]] * values_scale
x = arr[:,x_axis]
y = arr[:,y_axis]

min_x = np.min(x)
min_y = np.min(y)
x_norm = (x - min_x).astype(int)
y_norm = (y - min_y).astype(int)

# x_shape = int(np.max(x) - np.min(x))
# y_shape = int(np.max(y) - np.min(y))

im = np.zeros((np.max(x_norm)+1, np.max(y_norm)+1))
indices = np.stack([x_norm, y_norm], axis=1).astype(int)
im[indices[:, 0], indices[:, 1]] = 1

plt.figure(figsize=[12,12])
plt.imshow(im)
plt.show()

#%%
# # align point cloud and 2D image - for visualization in report
# im = np.flipud(np.transpose(im))
# plt.imshow(im)
# # plt.imshow(im,cmap='Greys',interpolation='nearest')
# plt.show()


im_uint8 = np.uint8(im)
#plt.figure(figsize=[12,12])
#plt.imshow(im_uint8)
#plt.show()

#%% lines visualization function - plots all lines in array created by cv2.HoughLines
def visualize_lines(img, lines_array, plot_each_line=False):
    for line_params in lines_array:
    
        # get rho and theta
        line_params = line_params[0]
        rho = line_params[0]
        theta = line_params[1]
    
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
    
        #  cv2.line(image, start_point, end_point, color, thickness)
        cv2.line(im_clean,(x1,y1),(x2,y2),color=1,thickness=2)
        
        if plot_each_line:
            cv2.line(img,(x1,y1),(x2,y2),color=1,thickness=2)
            plt.imshow(img)
            plt.show()

#%% find lines in image
# function: cv2.HoughLines(image, rho_res, theta_res, threshold[, lines[, srn[, stn]]]) → lines
# Input:
# rho_res – Distance resolution of the accumulator in pixels.
# theta_res – Angle resolution of the accumulator in radians.
# threshold – Accumulator threshold parameter. Only those lines are returned that get enough votes (>threshold).
# Output:
# Output: Each line is represented by a two-element vector (\rho, \theta):
# rho is the distance from the coordinate origin (0,0) (top-left corner of the image).
# theta is the line rotation angle in radians ( 0 ~{vertical line}, pi/2 ~{horizontal line} ).


theta_res_rad = np.pi/180 * theta_res_deg # convert to radians
all_lines = cv2.HoughLines(im_uint8, rho_res, theta_res_rad, threshold=threshold_first_line)
print("found", all_lines.shape[0], "lines")


#%% plot lines found by Hough
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
im_clean = im_uint8.copy()
visualize_lines(im_clean, all_lines[:4], plot_each_line=True)


#%% create a table lines params with columns ['rho', 'theta']
rho_list = []
theta_list = []
for line_params in all_lines:
    
    line_params = line_params[0]
    rho = line_params[0]
    theta = line_params[1]

    rho_list.append(rho)
    theta_list.append(theta)

df_lines = pd.DataFrame(list(zip(rho_list, theta_list)), columns=['rho', 'theta'])

#%% find 2 lines by angle of best single line found by hough

lines_2walls_axis1 = cv2.HoughLines(im_uint8, rho_res, theta_res_rad, threshold=threshold_walls_lines, 
                       min_theta=df_lines.theta[0] - search_range_deg * np.pi/180, 
                       max_theta=df_lines.theta[0] + search_range_deg * np.pi/180 )
print("found", lines_2walls_axis1.shape[0], "lines for first axis walls")


## plot some lines for example
#im_clean = im_uint8.copy()
#visualize_lines(im_clean, lines_2walls_axis1[0:10])
#plt.imshow(im_clean)
#plt.show()

# find 2 walls from the lines found by hough transform
rho_list = []
theta_list = []
for line_params in lines_2walls_axis1[0:n_lines_to_use]:

    # get rho and theta
    line_params = line_params[0]
    rho = line_params[0]
    theta = line_params[1]

    rho_list.append(rho)
    theta_list.append(theta)
    
df_lines_2walls_axis1 = pd.DataFrame(list(zip(rho_list, theta_list)), columns =['rho', 'theta']) 

## get 2 center line derived by kmeans clustering
## create a kmeans model on our data, using k clusters
#kmeans_model = KMeans(n_clusters=2, random_state=1).fit(df_lines_2walls_axis1)
#labels = kmeans_model.labels_
#interia = kmeans_model.inertia_
#lines = kmeans_model.cluster_centers_

# get 2 most distant lines from root(closest and furthest) by rho values
ind_min = df_lines_2walls_axis1['rho'].idxmin()
ind_max = df_lines_2walls_axis1['rho'].idxmax()
lines = df_lines_2walls_axis1.loc[[ind_min, ind_max],:].values

# plot 2 line detected as walls
im_clean = im_uint8.copy()
l1 = np.array([lines[0]])
l2 = np.array([lines[1]])
walls_axis1 = np.array([l1,l2])
visualize_lines(im_clean, walls_axis1)
plt.imshow(im_clean)
plt.show()

#%% find 2 walls by angle perpendicular to the best wall found by hough (assuming angle* = angle+90)

lines_2walls_axis2 = cv2.HoughLines(im_uint8, rho_res, theta_res_rad, threshold=threshold_walls_lines, 
                       min_theta=df_lines.theta[0] + (90 - search_range_deg) * np.pi/180, 
                       max_theta=df_lines.theta[0] + (90 + search_range_deg) * np.pi/180 )
print("found", lines_2walls_axis2.shape[0], "lines for second axis")

## plot some lines for example
#im_clean = im_uint8.copy()
#visualize_lines(im_clean, lines_2walls_axis2[0:10])
#plt.imshow(im_clean)
#plt.show()

# find 2 walls from lines found by hough transform
rho_list = []
theta_list = []
for line_params in lines_2walls_axis2[0:n_lines_to_use]:

    # get rho and theta
    line_params = line_params[0]
    rho = line_params[0]
    theta = line_params[1]

    rho_list.append(rho)
    theta_list.append(theta)
    
df_lines_2walls_axis2 = pd.DataFrame(list(zip(rho_list, theta_list)), columns =['rho', 'theta']) 

## get 2 center line derived by kmeans clustering
## create a kmeans model on our data, using k clusters
#kmeans_model = KMeans(n_clusters=2, random_state=1).fit(df_lines_2walls_axis2)
#labels = kmeans_model.labels_
#interia = kmeans_model.inertia_
#lines = kmeans_model.cluster_centers_

# get 2 most distant lines from root(closest and furthest) by rho values
ind_min = df_lines_2walls_axis2['rho'].idxmin()
ind_max = df_lines_2walls_axis2['rho'].idxmax()
lines = df_lines_2walls_axis2.loc[[ind_min, ind_max],:].values

# plot 2 line detected as walls
im_clean = im_uint8.copy()
l1 = np.array([lines[0]])
l2 = np.array([lines[1]])
walls_axis2 = np.array([l1,l2])
visualize_lines(im_clean, walls_axis2)
plt.imshow(im_clean)
plt.show()
    

#%% visualize 4 walls that were found
plt.figure(figsize=[12,12])
walls = np.concatenate((walls_axis1, walls_axis2), axis=0)
visualize_lines(im_clean, walls)
plt.imshow(im_clean)
plt.show()


#%% find walls intersections to define corners of the room
def polar2cartesian(rho: float, theta_rad: float, rotate90: bool = False):
    """
    Converts line equation from polar to cartesian coordinates

    Args:
        rho: input line rho
        theta_rad: input line theta
        rotate90: output line perpendicular to the input line

    Returns:
        m: slope of the line
           For horizontal line: m = 0
           For vertical line: m = np.nan
        b: intercept when x=0
    """
    x = np.cos(theta_rad) * rho
    y = np.sin(theta_rad) * rho
    m = np.nan
    if not np.isclose(x, 0.0):
        m = y / x
    if rotate90:
        if m is np.nan:
            m = 0.0
        elif np.isclose(m, 0.0):
            m = np.nan
        else:
            m = -1.0 / m
    b = 0.0
    if m is not np.nan:
        b = y - m * x

    return m, b

def intersection(m1: float, b1: float, m2: float, b2: float):
    # Consider y to be equal and solve for x
    # Solve:
    #   m1 * x + b1 = m2 * x + b2
    x = (b2 - b1) / (m1 - m2)
    # Use the value of x to calculate y
    y = m1 * x + b1

    return int(round(x)), int(round(y))


walls_arr = walls
m_list = []
b_list = []

for i in range(0,len(walls)):
    print(walls[i][0][0])
    print(walls[i][0][1])
    m,b = polar2cartesian(walls[i][0][0], walls[i][0][1], True)
    m_list.append(m)
    b_list.append(b)

inter_points = []
inter_points.append(intersection(m_list[0],b_list[0],m_list[2],b_list[2]))
inter_points.append(intersection(m_list[0],b_list[0],m_list[3],b_list[3]))
inter_points.append(intersection(m_list[1],b_list[1],m_list[2],b_list[2]))
inter_points.append(intersection(m_list[1],b_list[1],m_list[3],b_list[3]))

# plot corners by showing a circle around each corner
im_new = im_uint8.copy()
plt.figure(figsize=[12,12])
color = 1
radius = 5
for point in inter_points:
    x = point[0]
    y = point[1]
    center_coordinates = (x,y)
    cv2.circle(im_new, center_coordinates, radius, color=1, thickness=2)
# visualize_lines(im_clean, walls)
plt.imshow(im_new)
plt.show()


#%% find the center of the room
# sort points by x value to get 2 adjacent (non-opposite) point as 2 first points, and 2 last points is the list
corners = pd.DataFrame(inter_points, dtype=np.dtype('int'), columns=['x','y']).sort_values(by='x').reset_index(drop=True)
x_mid, y_mid = corners.mean()
mid_point = np.array((int(x_mid), int(y_mid)))

# # axis 1
# mid_wall1 = ( np.mean((corners.x[0], corners.x[1])), np.mean((corners.y[0], corners.y[1])))
# mid_wall2 = ( np.mean((corners.x[2], corners.x[3])), np.mean((corners.y[2], corners.y[3])))

# axis 2
mid_wall1 = ( np.mean((corners.x[0], corners.x[2])) , np.mean((corners.y[0], corners.y[2])))
mid_wall2 = ( np.mean((corners.x[1], corners.x[3])) , np.mean((corners.y[1], corners.y[3])))

#%% choose 2 points near the center of the room as point of interest

w = 0.6 # how close to the middle point should the other points be?
mid_point1 = np.average((mid_point, mid_wall1), weights=[w, 1-w], axis=0).astype('int')
mid_point2 = np.average((mid_point, mid_wall2), weights=[w, 1-w], axis=0).astype('int')

# plot middle points by showing a circle around each point
im_new = im_uint8.copy()
plt.figure(figsize=[12,12])
cv2.circle(im_new, tuple(mid_point), radius, color=1, thickness=2)
cv2.circle(im_new, tuple(mid_point1), radius, color=1, thickness=2)
cv2.circle(im_new, tuple(mid_point2), radius, color=1, thickness=2)
plt.imshow(im_new)
plt.show()

#%% convert middle points back to point cloud format

def convert_xy_to_pcd(point):
    point_shifted = point + (min_x, min_y)
    point_shifted_scaled = point_shifted / values_scale
    point_shifted_scaled_z = np.insert(point_shifted_scaled, z_axis, [0])
    return point_shifted_scaled_z

mid_point_xyz = convert_xy_to_pcd(np.flip(mid_point))
mid_point_xyz1 = convert_xy_to_pcd(np.flip(mid_point1))
mid_point_xyz2 = convert_xy_to_pcd(np.flip(mid_point2))

#%% add middle points to point cloud and visualize

arr_filt_z_edited = arr_filt_z
arr_filt_z_edited = np.append(arr_filt_z, np.reshape(mid_point_xyz, (1,3)), axis=0)
arr_filt_z_edited = np.append(arr_filt_z_edited, np.reshape(mid_point_xyz1, (1,3)), axis=0)
arr_filt_z_edited = np.append(arr_filt_z_edited, np.reshape(mid_point_xyz2, (1,3)), axis=0)

pcd_clean_zfilt = o3d.geometry.PointCloud()
pcd_clean_zfilt.points = o3d.utility.Vector3dVector(arr_filt_z_edited)

# visuzlie final results over point cloud
o3d.visualization.draw_geometries([pcd_clean_zfilt])

pcd_clean_zfilt_xy = pcd_clean_zfilt.__copy__()
for i in range(0, len(pcd_clean_zfilt_xy.points)):
    pcd_clean_zfilt_xy.points[i][z_axis] = 0
o3d.visualization.draw_geometries([pcd_clean_zfilt_xy])





#%% Additional capabilities that were tested but not used:

#%% find best k
    
#import pandas as pd
#from sklearn.cluster import KMeans
#
#costs = []
#for k in range (1, 11):
# 
#	# Create a kmeans model on our data, using k clusters.  random_state helps ensure that the algorithm returns the same results each time.
#	kmeans_model = KMeans(n_clusters=k, random_state=1).fit(df_lines)
#	
#	# These are our fitted labels for clusters -- the first cluster has label 0, and the second has label 1.
#	labels = kmeans_model.labels_
# 
#	# Sum of distances of samples to their closest cluster center
#	interia = kmeans_model.inertia_
#	costs.append(interia)
#     
#d = [(costs[i]-costs[i+1])/costs[i] for i in range(len(costs)-1)]
#K = np.argmax(d) + 2 #  +1 to count from 1 and not zero, +1 to find cluster number from diff

#%% remove outlier points manually by IQR
# df = pd.DataFrame(pcd.points)
# df.boxplot()
# plt.show()
#
# Q1 = np.percentile(df, 25, axis=0)
# Q3 = np.percentile(df, 75, axis=0)
# IQR = Q3 - Q1
# outlier_step = 1.5 * IQR
# outlier_list = df[
#                   ( df[0] < (Q1[0] - outlier_step[0]) ) |
#                   ( df[1] < (Q1[1] - outlier_step[1]) ) |
#                   ( df[2] < (Q1[2] - outlier_step[2]) )
#                   ]
#
# n_outliers = len(outlier_list)
# ind_outliers = outlier_list.index.tolist()
#
# print("Found " + str(n_outliers) + " outliers.")
# print("Index of outliers: ", ind_outliers)
#
# df_clean = df.drop(ind_outliers, axis=0)
# df_clean.boxplot()
# plt.show()

#%% downsample
# print("Downsample the point cloud with a voxel of 0.05")
# downpcd = pcd.voxel_down_sample(voxel_size=0.05)
# o3d.visualization.draw_geometries([downpcd])


#%% estimate normals
# print("Recompute the normal of the downsampled point cloud")
# downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
#     radius=0.1, max_nn=30))
# o3d.visualization.draw_geometries([downpcd])
#
# print("Print a normal vector of the 0th point")
# print(downpcd.normals[0])
# print("Print the normal vectors of the first 10 points")
# print(np.asarray(downpcd.normals)[:10, :])
# print("")


