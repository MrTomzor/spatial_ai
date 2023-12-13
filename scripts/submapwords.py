import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
import scipy
import math
import rospkg

from spatial_ai.common_spatial import *
# import open3d 
# from open3d.geometry import uniform_down_sample

def sample_mesh(mesh, num_points):# # #{
    # Load the mesh using trimesh
    # mesh = trimesh.load(mesh_path)

    # Sample points on the mesh
    points, face_indices = trimesh.sample.sample_surface_even(mesh, num_points)

    # Calculate normals at sampled points
    normals = mesh.face_normals[face_indices]

    return points, normals# # #}

def plot_3d_points_with_normals(points, normals):# # #{
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

    plt.show()# # #}

def downsample_pointcloud(points, k):# # #{
    # Use k-means clustering to downsample the point cloud
    kmeans = KMeans(n_clusters=k, random_state=0).fit(points)
    centroids = kmeans.cluster_centers_

    # Calculate the distances from each original point to its assigned centroid
    distances = np.linalg.norm(points - centroids[kmeans.labels_], axis=1)

    # Get the index of the point closest to each centroid
    closest_points_idx = np.argmin(distances, axis=0)

    # Return the downsampled points and the corresponding indices in the original point cloud
    return centroids, closest_points_idx# # #}

def sift_3d(pts, normals, n_horiz_bins , n_vert_bins ):# # #{
    horiz_bins = np.linspace(-math.pi, math.pi, n_horiz_bins + 1)
    vert_bins = np.linspace(-1, 1, n_vert_bins + 1)

    #GRAVITY ALIGNED
    phis = np.arctan2(normals[:, 1], normals[:, 0])
    total_horiz_histogram = np.histogram(phis, bins=horiz_bins)[0]
    # print(total_horiz_histogram)

    principial_horiz_bin_idx = np.argmax(total_horiz_histogram )
    # print("PRINCIP BIN IDX: " + str(principial_horiz_bin_idx))
    total_horiz_histogram = np.roll(total_horiz_histogram, -principial_horiz_bin_idx)
    # print(total_horiz_histogram)

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
    return result / np.linalg.norm(result)# # #}
    
def detect_and_describe_3d_key_areas(submap, n_horiz_bins, n_vert_bins):# # #{
    ''' Select interesting areas, stores centroids of the areas and their descriptors '''
    points = submap.surfel_points
    normals = submap.surfel_normals
    
    print("sampled")
    # print(np.argwhere(np.isnan(points)))
    # print("normals")
    # print(np.argwhere(np.isnan(normals)))
    # print("normals2")
    # print(normals)
    
    nan_norms = np.any(np.isnan(normals), axis = 1)
    print("N with nans: " + str(np.sum(nan_norms)) + " / " + str(normals.shape[0]))
    nan_normal_pts = points[nan_norms]
    
    points = points[np.logical_not(nan_norms)]
    normals = normals[np.logical_not(nan_norms)]
    
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2],
    #            c='b', marker='x', label='Original Points (Correspondence)')
    # ax.scatter(nan_normal_pts[:, 0], nan_normal_pts[:, 1], nan_normal_pts[:, 2],
    #            c='r', marker='x', label='NAN Points (Correspondence)')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()
    # plt.show()
    
    import time 
    cs = time.time()
    
    
    # n_clusters = 100
    n_clusters = int(points.shape[0] / 10)
    if n_clusters == 0:
        n_clusters = 1
    cluster_centroids, distortion = scipy.cluster.vq.kmeans(points, n_clusters, iter=2)
    kdtree = KDTree(cluster_centroids)
    pts_cluster_idxs = kdtree.query(points)[1]
    # print(pts_cluster_idxs)
    
    # COMPUTE DESCRIPTORS
    # n_horiz_bins = 8
    # n_vert_bins = 3
    
    desc_size = n_vert_bins * n_horiz_bins
    descriptors = []
    for i in range(n_clusters):
        pts_mask = pts_cluster_idxs == i
        if not np.any(pts_mask):
            print("EMPTY CLUSTER!")
            descriptors.append(np.ones((desc_size)) * 10)
            continue
        descriptors.append(sift_3d(None, normals[pts_mask], n_horiz_bins, n_vert_bins))
    print("N DESCRIPTORS: " + str(len(descriptors)))
    # print(descriptors)
    
    if len(descriptors) == 0:
        print("NO DESCRIBABLE CLUSTERS")
        return
    
    descriptors = np.array(descriptors)
    # print(descriptors.shape)
    # dif = scipy.spatial.distance_matrix(descriptors, descriptors)
    # maxdist = np.max(dif.flatten())
    
    submap.descriptor_positions = cluster_centroids
    submap.descriptors = descriptors# # #}

def compute_word_vector(submap, vocab_kdtree, vocab_centroids):
    word_idxs = vocab_tree.query(submap.descriptors)[1]
    present_words, counts = np.unique(word_idxs, return_counts=True)
    n_words = vocab_centroids.shape[0]

    res = np.zeros(n_words)
    for i in range(len(present_words)):
        res[present_words[i]] = counts[i]
    res = res / np.linalg.norm(res)
    submap.word_vector = res

    

num_points = 10000
# mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
#                        faces=[[0, 1, 2]])
# mesh = trimesh.load('Bunny.stl')
# mesh = trimesh.load('Normandy.stl')
# mesh = trimesh.load('House.stl')
# mesh = trimesh.load('Minecraft.stl')
# mesh = trimesh.load('model.sdf')
# points, normals = sample_mesh(mesh, num_points)

# n_horiz_bins = 20
# n_vert_bins = 6
n_horiz_bins = 6
n_vert_bins = 3
n_vocab_clusters = 6
desc_size = n_vert_bins * n_horiz_bins


# fpath = rospkg.RosPack().get_path('spatial_ai') + "/memories/almost_loop.pickle"
# fpath = rospkg.RosPack().get_path('spatial_ai') + "/memories/big_ep.pickle"
# fpath = rospkg.RosPack().get_path('spatial_ai') + "/memories/forest_uav.pickle"
fpath = rospkg.RosPack().get_path('spatial_ai') + "/memories/schools_loop1.pickle"
mchunk = CoherentSpatialMemoryChunk.load(fpath)

# MERGE SUBMAPS OF MCHUNK
merging_factor = 1
n_submaps_old = len(mchunk.submaps)
new_submaps = []
n_new_maps = int(n_submaps_old / merging_factor)

print("N OLD MAPS: " + str(n_submaps_old))
for i in range(n_new_maps):
    endindex = (i+1)*merging_factor
    if endindex > n_submaps_old:
        endindex = n_submaps_old
    print("indices:")
    if mchunk.submaps[i].points is None or mchunk.submaps[i].surfel_points is None:
        print("NONE PTS IN SUBMAP! SKIPPING!")
        continue
    indices = range(i*merging_factor, endindex)
    print(indices)
    new_submaps.append(mchunk.mergeSubmaps(indices))

mchunk.submaps = new_submaps
print("MERGING DONE")

for submap in mchunk.submaps:
    detect_and_describe_3d_key_areas(submap, n_horiz_bins, n_vert_bins)

total_descriptors = []

# total_descriptors = np.array(total_descriptors)
total_descriptors = None
total_descriptors_points = None
for m in range(len(mchunk.submaps)):
    tpoints = transformPoints(mchunk.submaps[m].descriptor_positions, mchunk.submaps[m].T_global_to_own_origin)
    # print(tpoints.shape)
    if total_descriptors_points   is None:
        total_descriptors_points  = tpoints
        total_descriptors = mchunk.submaps[m].descriptors
    else:
        total_descriptors_points = np.concatenate((total_descriptors_points, tpoints), axis=0)
        total_descriptors = np.concatenate((total_descriptors , mchunk.submaps[m].descriptors.reshape(mchunk.submaps[m].descriptors.shape[0], desc_size)), axis=0)

print("POINTS N DESCRIPTOR SHAPES:")
print(total_descriptors.shape)
print(total_descriptors_points.shape)

# VOCAB COMPUTATION
print("COMPUTING VOCAB")
vocab_centroids, distortion = scipy.cluster.vq.kmeans(total_descriptors, n_vocab_clusters, iter=200)
vocab_tree = KDTree(vocab_centroids)
print("DONE VOCAB")

# COMPUTE WORD VECTORS
wordvecs = []
for submap in mchunk.submaps:
    compute_word_vector(submap, vocab_tree, vocab_centroids)
    wordvecs.append(submap.word_vector)
wordvecs = np.array(wordvecs)
print("DONE WORDVECS")

# wordvecs_difs = scipy.spatial.distance_matrix(wordvecs, wordvecs)
# maxdist = np.max(dif.flatten())
# wordvecs_similarities = np.outer(wordvecs, wordvecs)
# wordvecs_db = np.outer(wordvecs, wordvecs)
# print(wordvecs_similarities)
# print("TESTSIM:")
# print(np.dot(wordvecs[0, :], wordvecs[1, :]))

print("DONE WORD SIMILARITIES OF SUBMAPS")


word_idxs = vocab_tree.query(total_descriptors)[1]
present_words = np.unique(word_idxs)


# VIS SIMILARITY TO ALL SUBMAPS
for i in range(len(mchunk.submaps)):
    query_scores = np.dot(wordvecs, wordvecs[i, :].T)
    minmax_scores = query_scores[np.where(np.arange(len(mchunk.submaps)) != i)]
    score_min = np.min(minmax_scores)
    score_max = np.max(minmax_scores)
    print(i)
    scores_recaled =  (query_scores - score_min) / (score_max - score_min)
    # scores_recaled =  query_scores
    print(query_scores)
    print(scores_recaled)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    pov_centroid = np.mean(transformPoints(mchunk.submaps[i].descriptor_positions, mchunk.submaps[i].T_global_to_own_origin), axis=0)

    lw_max = 3

    for j in range(len(mchunk.submaps)):
        tpoints = transformPoints(mchunk.submaps[j].descriptor_positions, mchunk.submaps[j].T_global_to_own_origin)
        # clr = 'r'
        # if j != i:
        #     clr = 'b'
        # ax.scatter(tpoints[:, 0], tpoints[:, 1], tpoints[:, 2],
        #            c=clr, marker='o', label='AAA')
        siz = 30
        if j != i:
            siz = 5
        ax.scatter(tpoints[:, 0], tpoints[:, 1], tpoints[:, 2],
                   marker='o',s=siz)
        point1 = pov_centroid
        point2 = np.mean(tpoints, axis=0)

        if j != i:
            score = scores_recaled[j]

            bar_size = 5
            point1 = point2 
            point2 = point1 + np.array([0, 0, bar_size * score])
            point4 = point1 + np.array([5, 0, 0])
            point3 = point4 + np.array([0, 0, bar_size])

            ax.plot([point4[0], point3[0]], [point4[1], point3[1]], [point4[2], point3[2]], c='k', linewidth = 5)
            ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], c='r', linewidth = 5)


            # ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], c='k', linewidth = lw_max * score)

    plt.show()

# VIS WORDS IN COLOR FROM WHOLE MAP
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# for word in present_words:

#     cluster_mask = word_idxs == word
#     not_mask = np.logical_not(cluster_mask )
#     ax.scatter(total_descriptors_points[cluster_mask, 0], total_descriptors_points[cluster_mask, 1], total_descriptors_points[cluster_mask, 2], marker='x', s=10)

# plt.show()



# VIS SIMILARITY
# for i in range(n_clusters):
#     print("CLUSTER " + str(i))
#     pts_mask = pts_cluster_idxs == i
#     if not np.any(pts_mask):
#         print("EMPTTY CLUSTER!")
#         continue
#     fig = plt.figure()
#     # ax = fig.add_subplot(111, projection='3d')
#     ax = plt.axes(projection='3d')

#     ax.scatter(points[pts_mask, 0], points[pts_mask, 1], points[pts_mask, 2], marker='x', c='b')
#     not_mask = np.logical_not(pts_mask)
#     ax.scatter(points[not_mask, 0], points[not_mask, 1], points[not_mask, 2], marker='x', c='c', alpha=0.2)

#     # ax.scatter(codebook[:, 0], codebook[:, 1], codebook[:,2], c='r')

#     print(np.sort(dif[i,:]))
#     for j in range(n_clusters):
#         if j == i:
#             continue

#         relative_maxdist = np.max(dif[i, :])
#         # lw = max_linewidth * (1 - (dif[i,j] / maxdist))
#         lw = max_linewidth * (1 - (dif[i,j] / relative_maxdist))
#         point1 = codebook[i, :]
#         point2 = codebook[j, :]
#         ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], c='r', linewidth = lw)
#         # ax.scatter(codebook[j, 0], codebook[j, 1], codebook[j,2], c='r', s=lw*3)

#     # ax.auto_scale_xyz()

#     plt.show()



# ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='x')
# ax.scatter(codebook[:, 0], codebook[:, 1], codebook[:,2], c='r')

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

