import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
import scipy
import math
# import open3d 
# from open3d.geometry import uniform_down_sample

def sample_mesh(mesh, num_points):
    # Load the mesh using trimesh
    # mesh = trimesh.load(mesh_path)

    # Sample points on the mesh
    points, face_indices = trimesh.sample.sample_surface_even(mesh, num_points)

    # Calculate normals at sampled points
    normals = mesh.face_normals[face_indices]

    return points, normals

def plot_3d_points_with_normals(points, normals):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', label='Points')

    # Plot the normals as lines
    # for i in range(len(points)):
    #     ax.quiver(points[i, 0], points[i, 1], points[i, 2],
    #               normals[i, 0], normals[i, 1], normals[i, 2],
    #               length=10, color='r', normalize=True, label='Normals')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()

def downsample_pointcloud(points, k):
    # Use k-means clustering to downsample the point cloud
    kmeans = KMeans(n_clusters=k, random_state=0).fit(points)
    centroids = kmeans.cluster_centers_

    # Calculate the distances from each original point to its assigned centroid
    distances = np.linalg.norm(points - centroids[kmeans.labels_], axis=1)

    # Get the index of the point closest to each centroid
    closest_points_idx = np.argmin(distances, axis=0)

    # Return the downsampled points and the corresponding indices in the original point cloud
    return centroids, closest_points_idx


def sift_3d(pts, normals, n_horiz_bins = 6, n_vert_bins = 3):
    horiz_bins = np.linspace(-math.pi, math.pi, n_horiz_bins + 1)
    vert_bins = np.linspace(-1, 1, n_vert_bins + 1)

    #GRAVITY ALIGNED
    phis = np.arctan2(normals[:, 1], normals[:, 0])
    total_horiz_histogram = np.histogram(phis, bins=horiz_bins)[0]
    print(total_horiz_histogram)

    principial_horiz_bin_idx = np.argmax(total_horiz_histogram )
    print("PRINCIP BIN IDX: " + str(principial_horiz_bin_idx))
    total_horiz_histogram = np.roll(total_horiz_histogram, -principial_horiz_bin_idx)
    print(total_horiz_histogram)

    # thetas = np.arctan2(normals[:, 1], normals[:, 0])
    # vert_bins = np.histogram(normals[:, 2], bins = n_vert_bins, range=(-1,1))

    vert_bin_ids = (((normals[:, 2] + 1) / 2.0) * n_vert_bins).astype(int)

    result = np.zeros((n_horiz_bins * n_vert_bins))
    for i in range(n_vert_bins):
        hist = np.zeros(n_horiz_bins)
        # print(vert_bins[i])
        # print(vert_bins[i+1])

        normals_mask = np.logical_and(normals[:, 2] >= vert_bins[i] , normals[:, 2] < vert_bins[i+1])
        hist_notrolled = np.histogram(phis[normals_mask], bins=horiz_bins)[0]
        # print(hist_notrolled)

        result[i*n_horiz_bins: (i+1)*n_horiz_bins] = np.roll(hist_notrolled, -principial_horiz_bin_idx)
    # return result / np.sum(result)
    return result / np.linalg.norm(result)
    

num_points = 1000
# mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
#                        faces=[[0, 1, 2]])
mesh = trimesh.load('Bunny.stl')


points, normals = sample_mesh(mesh, num_points)
print("sampled")

import time 
cs = time.time()

# print("downsampled")
# tree.query(points)

# kmeans = KMeans(n_clusters=100, random_state=0).fit(points)

n_clusters = 20

codebook, distortion = scipy.cluster.vq.kmeans(points, n_clusters, iter=2)
kdtree = KDTree(codebook)
pts_cluster_idxs = kdtree.query(points)[1]
print(pts_cluster_idxs)

# print(sift_3d(points, normals))

# Plot whitened data and cluster centers in red

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')


n_horiz_bins = 20
n_vert_bins = 5
desc_size = n_vert_bins * n_horiz_bins

descriptors = []

for i in range(n_clusters):
    pts_mask = pts_cluster_idxs == i
    if not np.any(pts_mask):
        print("EMPTTY CLUSTER!")
        descriptors.append(np.ones((desc_size)) * 10)
        continue
    descriptors.append(sift_3d(None, normals[pts_mask]))

descriptors = np.array(descriptors)
dif = scipy.spatial.distance_matrix(descriptors, descriptors)
maxdist = np.max(dif.flatten())
max_linewidth = 5

for i in range(n_clusters):
    print("CLUSTER " + str(i))
    pts_mask = pts_cluster_idxs == i
    if not np.any(pts_mask):
        print("EMPTTY CLUSTER!")
        continue
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[pts_mask, 0], points[pts_mask, 1], points[pts_mask, 2], marker='x')
    # ax.scatter(codebook[i, 0], codebook[i, 1], codebook[i,2], c='r')

    ax.scatter(codebook[:, 0], codebook[:, 1], codebook[:,2], c='r')

    print(np.sort(dif[i,:]))
    for j in range(n_clusters):
        if j == i:
            continue

        relative_maxdist = np.max(dif[i, :])
        # lw = max_linewidth * (1 - (dif[i,j] / maxdist))
        lw = max_linewidth * (1 - (dif[i,j] / relative_maxdist))
        point1 = codebook[i, :]
        point2 = codebook[j, :]
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], c='r', linewidth = lw)

    plt.show()

ctime = (time.time() - cs)
print("COMP TIME: " + str(ctime * 1000) + " ms")

# ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='x')
# ax.scatter(codebook[:, 0], codebook[:, 1], codebook[:,2], c='r')
plt.show()

# downsampled_points, original_indices = downsample_pointcloud(points, num_points // downsample_factor)
# print(original_indices)
# # Now, 'downsampled_points' contains the downsampled points, and 'original_indices'
# # contains the indices of the original points corresponding to each downsampled point

# # Visualize the downsampled points along with their correspondences in the original point cloud
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(downsampled_points[:, 0], downsampled_points[:, 1], downsampled_points[:, 2],
#            c='r', marker='o', label='Downsampled Points')

# ax.scatter(points[original_indices, 0], points[original_indices, 1], points[original_indices, 2],
#            c='b', marker='x', label='Original Points (Correspondence)')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()

# plt.show()

