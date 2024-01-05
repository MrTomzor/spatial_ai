import copy
import rospy
from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import EmptyResponse as EmptySrvResponse
import threading

import heapq

import dbow
import rospkg

from spatial_ai.common_spatial import *

from sensor_msgs.msg import Image, CompressedImage, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import mrs_msgs.msg
import std_msgs.msg
from scipy.spatial.transform import Rotation
import scipy
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
import inspect
from shapely import geometry

import trimesh
import rtree

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from scipy.spatial import Delaunay, delaunay_plot_2d
import matplotlib.pyplot as plt
import io
import copy

from scipy.spatial.transform import Rotation
import tf
import tf2_ros
import tf2_geometry_msgs  # for transforming geometry_msgs
import tf.transformations as tfs
from geometry_msgs.msg import TransformStamped


import gtsam
from gtsam import DoglegOptimizer
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np
from gtsam.symbol_shorthand import L, X
from gtsam.examples import SFMdata
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=W0611

import sys
from termcolor import colored, cprint

class SubmapBuilderModule:
    def __init__(self, w, h, K, camera_frame_id, odom_orig_frame_id, fcu_frame, tf_listener, T_imu_to_cam, T_fcu_to_imu ):# # #{
        self.width = w
        self.height = h
        self.K = K
        self.camera_frame = camera_frame_id
        self.odom_frame = odom_orig_frame_id
        self.fcu_frame = fcu_frame
        self.tf_listener = tf_listener
        self.T_imu_to_cam = T_imu_to_cam
        self.T_fcu_to_imu = T_fcu_to_imu 

        self.do_frontiers = True

        self.bridge = CvBridge()
        self.prev_image = None
        self.prev_time = None
        self.proper_triang = False

        self.spheremap = None
        # self.mchunk.submaps = []
        self.mchunk = CoherentSpatialMemoryChunk()

        self.keyframes = []
        self.noprior_triangulation_points = None
        self.odomprior_triangulation_points = None
        self.spheremap_mutex = threading.Lock()
        self.predicted_traj_mutex = threading.Lock()

        # SRV
        # self.vocab_srv = rospy.Service("save_vocabulary", EmptySrv, self.saveCurrentVisualDatabaseToVocabFile)
        # self.save_episode_full = rospy.Service("save_episode_full", EmptySrv, self.saveEpisodeFull)
        # self.return_home_srv = rospy.Service("home", EmptySrv, self.return_home)

        # VIS PUB
        self.spheremap_outline_pub = rospy.Publisher('spheres', MarkerArray, queue_size=10)
        self.spheremap_freespace_pub = rospy.Publisher('spheremap_freespace', MarkerArray, queue_size=10)

        self.recent_submaps_vis_pub = rospy.Publisher('recent_submaps_vis', MarkerArray, queue_size=10)
        self.path_planning_vis_pub = rospy.Publisher('path_planning_vis', MarkerArray, queue_size=10)
        self.visual_similarity_vis_pub = rospy.Publisher('visual_similarity_vis', MarkerArray, queue_size=10)
        self.unsorted_vis_pub = rospy.Publisher('unsorted_markers', MarkerArray, queue_size=10)

        # self.kp_pub = rospy.Publisher('tracked_features_img', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('estim_depth_img', Image, queue_size=1)
        self.marker_pub = rospy.Publisher('/vo_odom', Marker, queue_size=10)


        print("T_imu(fcu)_to_cam")
        print(self.T_imu_to_cam)


        # LOAD PARAMS
        self.verbose_submap_construction = rospy.get_param("local_mapping/verbose_submap_construction")
        self.carryover_dist = rospy.get_param("local_mapping/carryover_dist")
        self.uav_radius = rospy.get_param("local_nav/uav_radius")
        self.min_planning_odist = rospy.get_param("local_nav/min_planning_odist")

        self.marker_scale = rospy.get_param("marker_scale")
        self.n_sphere_samples_per_update = rospy.get_param("local_mapping/n_sphere_samples_per_update")
        self.fragmenting_travel_dist = rospy.get_param("local_mapping/smap_fragmentation_dist")
        self.max_sphere_sampling_z = rospy.get_param("local_mapping/max_sphere_sampling_z")
        self.visual_kf_addition_heading = 3.14159 /2
        self.visual_kf_addition_dist = 2
        
        # # #}

    # CORE

    def camera_update_iter(self, msg, slam_ids = None):# # #{
        with ScopedLock(self.spheremap_mutex):
            if self.verbose_submap_construction:
                print("PCL MSG")

            if self.verbose_submap_construction:
                print("PCL MSG PROCESSING NOW")
            comp_start_time = time.time()

            # CHECK TRAVELED DIST
            if not self.spheremap is None:
                T_global_to_fcu = lookupTransformAsMatrix(self.odom_frame, self.fcu_frame, self.tf_listener)
                T_fcu_relative_to_smap_start  = np.linalg.inv(self.spheremap.T_global_to_own_origin) @ T_global_to_fcu

                new_kf = SubmapKeyframe(T_fcu_relative_to_smap_start)


                # CHECK IF NEW ENOUGH
                # TODO - check in near certainly connected submaps
                tooclose = False
                for kf in self.spheremap.visual_keyframes:
                    # TODO - scaling
                    if kf.euclid_dist(new_kf) < self.visual_kf_addition_dist and kf.heading_dif(new_kf) < self.visual_kf_addition_heading:
                        tooclose = True
                        break

                if len(self.spheremap.visual_keyframes) > 0:
                    dist_bonus = new_kf.euclid_dist(self.spheremap.visual_keyframes[-1])
                    heading_bonus = new_kf.heading_dif(self.spheremap.visual_keyframes[-1]) * 0.2

                    self.spheremap.traveled_context_distance += dist_bonus + heading_bonus

                # print("KFS: adding new visual keyframe!")
                self.spheremap.visual_keyframes.append(new_kf)

            # Read Points from PCL msg# #{
            pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            pc_list = []
            for point in pc_data:
                x, y, z = point
                pc_list.append([x, y, z])
            if len(pc_list) == 0:
                return
            point_cloud_array = np.array(pc_list, dtype=np.float32)

            # OpenVINS outputs always some point at origin, and it is not really useful for us, breaks hull shit
            #TODO - solve what to do with this in future
            nonzero_pts = np.array(np.linalg.norm(point_cloud_array, axis=1) > 0)
            if not np.any(nonzero_pts):
                if self.verbose_submap_construction:
                    print("NO POINTS?")
                return
            point_cloud_array = point_cloud_array[nonzero_pts, :]
            # print("PCL ARRAY SHAPE:")
            # print(point_cloud_array.shape)
# # #}

            # DECIDE WHETHER TO UPDATE SPHEREMAP OR INIT NEW ONE# #{
            T_global_to_fcu = lookupTransformAsMatrix(self.odom_frame, self.fcu_frame, self.tf_listener)

            self.submap_unpredictability_signal = 0
            init_new_spheremap = False
            memorized_transform_to_prev_map = None

            init_rad = self.min_planning_odist
            min_rad = init_rad

            if self.spheremap is None:
                init_new_spheremap = True
            else:
                if self.submap_unpredictability_signal > 1:
                    pass
                else:
                    max_split_score = 100
                    split_score = 0
                    secs_since_creation = (rospy.get_rostime() - self.spheremap.creation_time).to_sec()

                    # TIME BASED
                    # if secs_since_creation > 70:
                    #     split_score = max_split_score

                    # CONTEXT DIST BASED
                    if self.spheremap.traveled_context_distance > self.fragmenting_travel_dist:
                        split_score = max_split_score
                    else:
                        if self.verbose_submap_construction:
                            print("TRAVELED CONTEXT DISTS: " + str(self.spheremap.traveled_context_distance))

                    if split_score >= max_split_score:
                        # SAVE OLD SPHEREMAP
                        print("SPLITTING SUBMAP!")
                        # TODO give it the transform to the next ones origin from previous origin
                        memorized_transform_to_prev_map = np.linalg.inv(self.spheremap.T_global_to_own_origin) @ T_global_to_fcu

                        self.mchunk.submaps.append(self.spheremap)
                        init_new_spheremap = True
                        self.visualize_episode_submaps()
            if init_new_spheremap:
                # INIT NEW SPHEREMAP
                print("INFO: initing new spheremap")

                pts_to_transfer = None
                radii_to_transfer = None

                # CREATE CONNECTION FROM AND TO THE PREV SPHEREMAP IF PREV MAP EXISTS!
                connection_to_prev_map = None
                if (not self.spheremap is None) and (not memorized_transform_to_prev_map is None):
                    print("CREATING CONNECTION TO PREV MAP")
                    pos_in_old_frame = memorized_transform_to_prev_map[:3,3].reshape((1,3))
                    pos_in_new_frame = np.zeros((1,3))
                    odist = self.spheremap.getMaxDistToFreespaceEdge(pos_in_old_frame)
                    print("ODIST: " + str(odist))

                    if odist < self.uav_radius:
                        odist = self.uav_radius
                        print("WARN! ODIST SMALLER THAN UAV RADIUS!")

                    self.spheremap.map2map_conns.append(MapToMapConnection(pos_in_old_frame, len(self.mchunk.submaps), odist))
                    connection_to_prev_map = MapToMapConnection(pos_in_new_frame, len(self.mchunk.submaps)-1, odist)
                    # else:
                    #     # print("ODIST NOT SAFE!!! NOT ADDING CONNECTION!")
                    #     TODO - 
                    # old_map_connection_pt = 

                    # GET SOME CLOSE SPHERES
                    if not self.spheremap.spheres_kdtree is None:
                        qres = self.spheremap.spheres_kdtree.query(pos_in_old_frame, k=10, distance_upper_bound=self.carryover_dist)
                        print("CARRYOVER:")
                        idxs = qres[1][0]
                        existing_mask = idxs != self.spheremap.points.shape[0]
                        print(np.sum(existing_mask ))
                        existing_mask = np.where(existing_mask)[0]
                        if np.any(existing_mask):
                            pts_to_transfer = self.spheremap.points[existing_mask, :]
                            radii_to_transfer = self.spheremap.radii[existing_mask]
                            pts_to_transfer = transformPoints(pts_to_transfer, np.linalg.inv(memorized_transform_to_prev_map))


                self.spheremap = SphereMap(init_rad, min_rad)
                self.spheremap.surfels_filtering_radius = 0.2


                if not connection_to_prev_map is None:
                    self.spheremap.map2map_conns.append(connection_to_prev_map)

                self.spheremap.memorized_transform_to_prev_map = memorized_transform_to_prev_map 
                print("PESSS")
                self.spheremap.T_global_to_own_origin = T_global_to_fcu
                self.spheremap.creation_time = rospy.get_rostime()

                if not pts_to_transfer is None:
                    self.spheremap.points = copy.deepcopy(pts_to_transfer)
                    self.spheremap.radii = copy.deepcopy(radii_to_transfer)
                    self.spheremap.connections = np.array([None for c in range(pts_to_transfer.shape[0])], dtype=object)
                    self.spheremap.updateConnections(np.arange(0, pts_to_transfer.shape[0]))
                    self.spheremap.labelSpheresByConnectivity()



                if len(self.mchunk.submaps) > 2:
                    self.saveEpisodeFull(None)
            # # #}

            # TRANSFORM SLAM PTS TO -----CAMERA FRAME---- AND COMPUTE THEIR PIXPOSITIONS
            # T_global_to_cam = T_global_to_imu @ self.T_imu_to_cam  #TODO fix this
            T_global_to_cam = T_global_to_fcu @ self.T_fcu_to_imu @ self.T_imu_to_cam

            transformed = transformPoints(point_cloud_array, np.linalg.inv(T_global_to_cam)).T

            positive_z_mask = transformed[2, :] > 0
            positive_z_points = transformed[:, positive_z_mask]
            positive_z_slam_ids = None
            if not slam_ids is None:
                positive_z_slam_ids = slam_ids[positive_z_mask]

            pixpos = getPixelPositions(positive_z_points, self.K)

            # COMPUTE DELAUNAY TRIANG OF VISIBLE SLAM POINTS
            # print("POS Z SLAM PTS")
            # print(positive_z_points.shape)
            # print("PIXPOS SHAPE")
            # print(pixpos.shape)
            if positive_z_points.shape[1] < 4:
                if self.verbose_submap_construction:
                    print("NOT ENAUGH PTS FOR DELAUNAY!")
                return
            if self.verbose_submap_construction:
                print("HAVE DELAUNAY:")
            tri = Delaunay(pixpos)

            # print("INPUT: " )
            # print(point_cloud_array)
            # print("TRANSFORMED TO CAM FRAME: " )
            # print(positive_z_points.T)

            # vis = self.visualize_depth(pixpos, tri)
            # self.depth_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))

            # CONSTRUCT OBSTACLE MESH
            comp_mesh = time.time()
            obstacle_mesh = trimesh.Trimesh(vertices=positive_z_points.T, faces = tri.simplices)

            # CONSTRUCT POLYGON OF PIXPOSs OF VISIBLE SLAM PTS
            hull = ConvexHull(pixpos)

            if self.verbose_submap_construction:
                print("POLY")
            img_polygon = geometry.Polygon(hull.points)

            # CONSTRUCT HULL MESH FROM 3D POINTS OF CONVEX 2D HULL OF PROJECTED POINTS
            # hullmesh_pts = positive_z_points[:, np.unique(hull.simplices)]
            hullmesh_pts = positive_z_points
            orig_pts = np.zeros((3, 1))
            hullmesh_pts = np.concatenate((hullmesh_pts, orig_pts), axis=1)
            zero_pt_index = hullmesh_pts.shape[1] - 1

            # FUCK IT JUST DO CONV HULL OF THESE 3D PTS(just start and hull pts)
            fov_hull = ConvexHull(hullmesh_pts.T)
            # print("HULLMESH PTS:")
            # print(hullmesh_pts.T)
            # print("CONVHULL PTS:")
            # print(fov_hull.points)

            # uniq_idxs_in_hull = np.unique(fov_hull.simplices)
            # simplices_reindexing = {uniq_idxs_in_hull[i]:i for i in range(uniq_idxs_in_hull.size)}
            # simplices_reindexed = [[simplices_reindexing[orig[0]], simplices_reindexing[orig[1]], simplices_reindexing[orig[2]]] for orig in fov_hull.simplices]

            # CONSTRUCT FOV MESH AND QUERY
            # fov_mesh = trimesh.Trimesh(vertices=fov_hull.points, faces = simplices_reindexed)
            fov_mesh = trimesh.Trimesh(vertices=fov_hull.points, faces = fov_hull.simplices)
            fov_mesh_query = trimesh.proximity.ProximityQuery(fov_mesh)

            # CONSTRUCT OBSTACLE POINT MESH AND QUERY
            obstacle_mesh_query = trimesh.proximity.ProximityQuery(obstacle_mesh)

            comp_time = time.time() - comp_mesh

            fr_samples = None
            if self.do_frontiers:
                # print("FRONTIERS SAMPLING")
                fr_samples, fr_face_indices = trimesh.sample.sample_surface_even(fov_mesh, 200)
                fr_samples2, fr_face_indices2 = trimesh.sample.sample_surface_even(obstacle_mesh, 50)
                fr_samples = np.concatenate((fr_samples, fr_samples2))
                # print(fr_samples.shape)
                # fr_sample_odists = obstacle_mesh_query.signed_distance(fr_samples)
                # fr_samples = fr_samples[fr_sample_odists > -0.1, :]
                # print(fr_samples.shape)

            if self.verbose_submap_construction:
                print("MESHING time: " + str((comp_time) * 1000) +  " ms")


            # ---UPDATE AND PRUNING STEP
            T_orig_to_current_cam = np.eye(4)
            T_delta_odom  = np.eye(4)
            if not self.spheremap is None: # TODO remove
                # transform existing sphere points to current camera frame
                n_spheres_old = self.spheremap.points.shape[0]

                T_delta_odom = np.linalg.inv(self.spheremap.T_global_to_own_origin) @ T_global_to_fcu

                T_orig_to_current_cam = ( T_delta_odom @ self.T_fcu_to_imu @ self.T_imu_to_cam)

                # project sphere points to current camera frame
                transformed_old_points  = transformPoints(self.spheremap.points, np.linalg.inv(T_orig_to_current_cam))

                # Filter out spheres with z below zero or above the max z of obstacle points
                # TODO - use dist rather than z for checking
                max_vis_z = np.max(positive_z_points[2, :])
                z_ok_idxs = np.logical_and(transformed_old_points[:, 2] > 0, transformed_old_points[:, 2] <= max_vis_z)

                z_ok_points = transformed_old_points[z_ok_idxs , :] # remove spheres with negative z
                worked_sphere_idxs = np.arange(n_spheres_old)[z_ok_idxs ]
                # print("POSITIVE Z AND NOT BEHIND OBSTACLE MESH:" + str(np.sum(z_ok_idxs )) + "/" + str(n_spheres_old))

                # check the ones that are projected into the 2D hull
                old_pixpos = getPixelPositions(z_ok_points.T, self.K)
                inhull = np.array([img_polygon.contains(geometry.Point(old_pixpos[i, 0], old_pixpos[i, 1])) for i in range(old_pixpos.shape[0])])
                
                # print("OLD PTS IN HULL:" + str(np.sum(inhull)) + "/" + str(z_ok_points.shape[0]))

                if np.any(inhull):
                    # remove spheres not projecting to conv hull in 2D
                    visible_old_points = z_ok_points[inhull]
                    worked_sphere_idxs = worked_sphere_idxs[inhull]
                    # print("IN HULL: " + str(worked_sphere_idxs.size))

                    # get mesh distances for these updatable spheres
                    old_spheres_fov_dists = np.abs(fov_mesh_query.signed_distance(visible_old_points))
                    old_spheres_obs_dists = np.abs(obstacle_mesh_query.signed_distance(visible_old_points))
                    upperbound_combined = np.minimum(old_spheres_fov_dists, old_spheres_obs_dists)

                    should_decrease_radius = old_spheres_obs_dists < self.spheremap.radii[worked_sphere_idxs]
                    could_increase_radius = upperbound_combined > self.spheremap.radii[worked_sphere_idxs]

                    for i in range(worked_sphere_idxs.size):
                        if should_decrease_radius[i]:
                            self.spheremap.radii[worked_sphere_idxs[i]] = old_spheres_obs_dists[i]
                        elif could_increase_radius[i]:
                            self.spheremap.radii[worked_sphere_idxs[i]] = upperbound_combined[i]

                    # FIND WHICH SMALL SPHERES TO PRUNE AND STOP WORKING WITH THEM, BUT REMEMBER INDICES TO KILL THEM IN THE END
                    idx_picker = self.spheremap.radii[worked_sphere_idxs[i]] < self.spheremap.min_radius
                    toosmall_idxs = worked_sphere_idxs[idx_picker]
                    shouldkeep = np.full((n_spheres_old , 1), True)
                    shouldkeep[toosmall_idxs] = False
                    shouldkeep = shouldkeep .flatten()

                    # print("SHOULDKEEP:")
                    # print(shouldkeep)

                    worked_sphere_idxs = worked_sphere_idxs[np.logical_not(idx_picker)].flatten()

                    # self.spheremap.consistencyCheck()
                    # RE-CHECK CONNECTIONS
                    self.spheremap.updateConnections(worked_sphere_idxs)

                    # AT THE END, PRUNE THE EXISTING SPHERES THAT BECAME TOO SMALL (delete their pos, radius and conns)
                    # self.spheremap.consistencyCheck()
                    self.spheremap.removeSpheresIfRedundant(worked_sphere_idxs)

                    # self.spheremap.removeNodes(np.where(idx_picker)[0])

            # TODO fix - by raycasting!!!
            max_sphere_sampling_z = self.max_sphere_sampling_z

            n_sampled = self.n_sphere_samples_per_update
            sampling_pts = np.random.rand(n_sampled, 2)  # Random points in [0, 1] range for x and y
            sampling_pts = sampling_pts * [self.width, self.height]

            # CHECK THE SAMPLING DIRS ARE INSIDE THE 2D CONVEX HULL OF 3D POINTS
            inhull = np.array([img_polygon .contains(geometry.Point(p[0], p[1])) for p in sampling_pts])
            # print(inhull)
            if not np.any(inhull):
                if self.verbose_submap_construction:
                    print("NONE IN HULL")
                self.spheremap.labelSpheresByConnectivity()
                return
            sampling_pts = sampling_pts[inhull, :]

            n_sampled = sampling_pts.shape[0]

            # NOW PROJECT THEM TO 3D SPACE
            sampling_pts = np.concatenate((sampling_pts.T, np.full((1, n_sampled), 1)))

            invK = np.linalg.inv(self.K)
            sampling_pts = invK @ sampling_pts
            rand_z = np.random.rand(1, n_sampled) * max_sphere_sampling_z
            rand_z[rand_z > max_sphere_sampling_z] = max_sphere_sampling_z

            # FILTER PTS - CHECK THAT THE MAX DIST IS NOT BEHIND THE OBSTACLE MESH BY RAYCASTING
            ray_hit_pts, index_ray, index_tri = obstacle_mesh.ray.intersects_location(
            ray_origins=np.zeros(sampling_pts.shape).T, ray_directions=sampling_pts.T)

            # MAKE THEM BE IN RAND POSITION BETWEEN CAM AND MESH HIT POSITION
            sampling_pts =  (np.random.rand(n_sampled, 1) * ray_hit_pts).T
            # TODO - add max sampling dist


            orig_3dpt_indices_in_hull = np.unique(hull.simplices)

            # TRY ADDING NEW SPHERES AT SAMPLED POSITIONS
            new_spheres_fov_dists = np.abs(fov_mesh_query.signed_distance(sampling_pts.T))
            new_spheres_obs_dists = np.abs(obstacle_mesh_query.signed_distance(sampling_pts.T))

            mindists = np.minimum(new_spheres_obs_dists, new_spheres_fov_dists)
            new_sphere_idxs = mindists > min_rad

            n_spheres_before_adding = self.spheremap.points.shape[0]
            n_spheres_to_add = np.sum(new_sphere_idxs)
            if self.verbose_submap_construction:
                print("PUTATIVE SPHERES THAT PASSED FIRST RADIUS CHECKS: " + str(n_spheres_to_add))
            if n_spheres_to_add == 0:
                if self.verbose_submap_construction:
                    print("NO NEW SPHERES TO ADD")
                self.spheremap.labelSpheresByConnectivity()
                return

            # print(mindists.shape)
            # print(mindists[new_sphere_idxs].shape)

            # TRANSFORM POINTS FROM CAM ORIGIN TO SPHEREMAP ORIGIN! - DRAW OUT!
            # new_sphere_locations_in_spheremap_origin = transformPoints(sampling_pts[:, new_sphere_idxs].T, T_orig_to_current_cam)
            new_sphere_locations_in_spheremap_origin = transformPoints(sampling_pts[:, new_sphere_idxs].T, T_orig_to_current_cam)

            # print(new_sphere_locations_in_spheremap_origin.shape)
            # print(self.spheremap.points.shape)

            self.spheremap.points = np.concatenate((self.spheremap.points, new_sphere_locations_in_spheremap_origin))
            self.spheremap.radii = np.concatenate((self.spheremap.radii.flatten(), mindists[new_sphere_idxs].flatten()))
            self.spheremap.connections = np.concatenate((self.spheremap.connections.flatten(), np.array([None for i in range(n_spheres_to_add)], dtype=object).flatten()))

            # ADD SPHERE AT CURRENT POSITION!
            pos_in_smap_frame = T_delta_odom[:3, 3].reshape((1,3)) 
            self.spheremap.points = np.concatenate((self.spheremap.points, pos_in_smap_frame))
            self.spheremap.radii = np.concatenate((self.spheremap.radii.flatten(), np.array([self.uav_radius]) ))
            self.spheremap.connections = np.concatenate((self.spheremap.connections.flatten(), np.array([None], dtype=object).flatten()))
            n_spheres_to_add += 1

            self.spheremap.updateConnections(np.arange(n_spheres_before_adding, n_spheres_before_adding+n_spheres_to_add))
            new_idxs = np.arange(self.spheremap.radii.size)[self.spheremap.radii.size - n_spheres_to_add : self.spheremap.radii.size]
            self.spheremap.removeSpheresIfRedundant(new_idxs)

            self.spheremap.labelSpheresByConnectivity()

            comp_time = time.time() - comp_start_time
            if self.verbose_submap_construction:
                print("SPHEREMAP integration time: " + str((comp_time) * 1000) +  " ms")
                print("N SPHERES: " + str(self.spheremap.points.shape[0]))

            comp_start_time = time.time()
            self.spheremap.spheres_kdtree = KDTree(self.spheremap.points)
            self.spheremap.max_radius = np.max(self.spheremap.radii)
            comp_time = time.time() - comp_start_time
            if self.verbose_submap_construction:
                print("Sphere KDTree computation: " + str((comp_time) * 1000) +  " ms")

            # HANDLE ADDING/REMOVING VISIBLE 3D POINTS
            comp_start_time = time.time()

            visible_pts_in_spheremap_frame = transformPoints(positive_z_points.T, T_orig_to_current_cam)
            self.spheremap.updateSurfels(visible_pts_in_spheremap_frame , pixpos, tri.simplices, positive_z_slam_ids)

            if self.do_frontiers:
                self.spheremap.updateFrontiers(transformPoints(fr_samples, T_orig_to_current_cam))

            comp_time = time.time() - comp_start_time
            if self.verbose_submap_construction:
                print("SURFELS integration time: " + str((comp_time) * 1000) +  " ms")

            # VISUALIZE CURRENT SPHERES
            self.visualize_spheremap()
            comp_time = time.time() - comp_start_time
            if self.verbose_submap_construction:
                print("SPHEREMAP visualization time: " + str((comp_time) * 1000) +  " ms")
# # #}

    # UTILS

    def saveEpisodeFull(self, req):# # #{
        # with ScopedLock(self.spheremap_mutex):
        print("SAVING EPISODE MEMORY CHUNK")

        fpath = rospkg.RosPack().get_path('spatial_ai') + "/memories/last_episode.pickle"
        self.mchunk.submaps.append(self.spheremap)
        self.mchunk.save(fpath)
        self.mchunk.submaps.pop()

        print("EPISODE SAVED TO " + str(fpath))
        return EmptySrvResponse()# # #}

    def get_visited_viewpoints_global(self):# # #{
        res_pts = None
        res_headings = None
        for smap in [self.spheremap] + self.mchunk.submaps:
            if len(smap.visual_keyframes) > 0:
                pts_smap = np.array([kf.position for kf in smap.visual_keyframes])
                headings_smap = np.array([kf.heading for kf in smap.visual_keyframes])
                pts_global, headings_global = transformViewpoints(pts_smap, headings_smap, smap.T_global_to_own_origin)

                if res_pts is None:
                    res_pts = pts_global
                    res_headings = headings_global
                else:
                    res_pts = np.concatenate((res_pts, pts_global))
                    res_headings = np.concatenate((res_headings, headings_global))

        return res_pts, res_headings# # #}

    # VISUALIZE

    def visualize_episode_submaps(self):# # #{
        marker_array = MarkerArray()

        max_maps_to_vis = 20

        if max_maps_to_vis > len(self.mchunk.submaps):
            max_maps_to_vis = len(self.mchunk.submaps)
        if len(self.mchunk.submaps) == 0:
            return

        print("VISUALIZING MAPS:" )
        print(max_maps_to_vis)

        clr_index = 0
        max_clrs = 4
        for i in range(max_maps_to_vis):
            idx = len(self.mchunk.submaps)-(1+i)
            print(idx)
            clr_index = idx % max_clrs
            # if self.mchunk.submaps[idx].memorized_transform_to_prev_map is None:
            #     break
            print("T")
            print(self.mchunk.submaps[idx].T_global_to_own_origin)
            self.get_spheremap_marker_array(marker_array, self.mchunk.submaps[idx], self.mchunk.submaps[idx].T_global_to_own_origin, alternative_look = True, do_connections = False, do_surfels = True, do_spheres = False, do_map2map_conns=False, ms=self.marker_scale, clr_index = clr_index)

        self.recent_submaps_vis_pub.publish(marker_array)

        return
# # #}

    def visualize_spheremap(self):# # #{
        if self.spheremap is None:
            return

        marker_array = MarkerArray()
        self.get_spheremap_marker_array(marker_array, self.spheremap, self.spheremap.T_global_to_own_origin, ms=self.marker_scale, do_spheres=False, do_surfels=True, do_frontiers=True)
        self.spheremap_outline_pub.publish(marker_array)

        marker_array = MarkerArray()
        self.get_spheremap_marker_array(marker_array, self.spheremap, self.spheremap.T_global_to_own_origin, ms=self.marker_scale, do_spheres=True, do_surfels=False, do_keyframes=False)
        self.spheremap_freespace_pub.publish(marker_array)
# # #}
    
    def visualize_depth(self, pixpos, tri):# # #{
        rgb = np.repeat(copy.deepcopy(self.new_frame)[:, :, np.newaxis], 3, axis=2)

        for i in range(len(tri.simplices)):
            a = pixpos[tri.simplices[i][0], :].astype(int)
            b = pixpos[tri.simplices[i][1], :].astype(int)
            c = pixpos[tri.simplices[i][2], :].astype(int)

            rgb = cv2.line(rgb, (a[0], a[1]), (b[0], b[1]), (255, 0, 0), 3)
            rgb = cv2.line(rgb, (a[0], a[1]), (c[0], c[1]), (255, 0, 0), 3)
            rgb = cv2.line(rgb, (c[0], c[1]), (b[0], b[1]), (255, 0, 0), 3)

        res = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return res
# # #}

    def get_spheremap_marker_array(self, marker_array, smap, T_inv, alternative_look=False, do_connections=False,  do_surfels=True, do_spheres=True, do_keyframes=False, do_normals=False, do_map2map_conns=True, do_frontiers=False, ms=1, clr_index =0, alpha = 1):# # #{
        # T_vis = np.linalg.inv(T_inv)
        T_vis = T_inv
        pts = transformPoints(smap.points, T_vis)

        marker_id = 0
        if len(marker_array.markers) > 0:
            marker_id = marker_array.markers[-1].id + 1

        if do_map2map_conns and len(smap.map2map_conns) > 0:
            n_conns = len(smap.map2map_conns)

            # CONN PTS
            untr_pts = np.array([c.pt_in_first_map_frame for c in smap.map2map_conns])
            untr_pts = untr_pts.reshape((n_conns, 3))
            spts = transformPoints(untr_pts, T_vis)

            marker = Marker()
            # marker.header.frame_id = "global"  # Adjust the frame_id as needed
            marker.header.frame_id = self.odom_frame  # Adjust the frame_id as needed
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            # marker.scale.x = 1.2  # Adjust the size of the points
            # marker.scale.y = 1.2
            marker.scale.x = ms *2.2  # Adjust the size of the points
            marker.scale.y = ms *2.2
            marker.color.a = 1
            marker.color.r = 0.5
            marker.color.b = 1
            # if alternative_look:
            #     marker.color.r = 0.5
            #     marker.color.b = 1

            marker.id = marker_id
            marker_id += 1

            # Convert the 3D points to Point messages
            points_msg = [Point(x=spts[i, 0], y=spts[i, 1], z=spts[i, 2]) for i in range(n_conns)]
            marker.points = points_msg
            marker_array.markers.append(marker)

            # LINES BETWEEN CONN POINTS
            line_marker = Marker()
            line_marker.header.frame_id = self.odom_frame  # Set your desired frame_id
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD
            line_marker.scale.x =ms* 1  # Line width
            line_marker.color.a = 1  # Alpha
            line_marker.color.r = 0.5  # Alpha
            line_marker.color.b = 1  # Alpha

            # if alternative_look:
            #     line_marker.scale.x = 0.1

            line_marker.id = marker_id
            marker_id += 1

            for i in range(n_conns):
                for j in range(i+1, n_conns):
                    point1 = Point()
                    point2 = Point()

                    p1 = spts[i, :]
                    p2 = spts[j, :]
                    
                    point1.x = p1[0]
                    point1.y = p1[1]
                    point1.z = p1[2]
                    point2.x = p2[0]
                    point2.y = p2[1]
                    point2.z = p2[2]
                    line_marker.points.append(point1)
                    line_marker.points.append(point2)
            marker_array.markers.append(line_marker)

        if do_connections:
            line_marker = Marker()
            line_marker.header.frame_id = self.odom_frame  # Set your desired frame_id
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD
            line_marker.scale.x = ms* 0.2  # Line width
            line_marker.color.a = 1.0  # Alpha
            if alternative_look:
                line_marker.scale.x =ms * 0.1

            line_marker.id = marker_id
            marker_id += 1

            for i in range(smap.connections.shape[0]):
                if smap.connections[i] is None:
                    continue
                for j in range(len(smap.connections[i])):
                    point1 = Point()
                    point2 = Point()

                    p1 = pts[i, :]
                    p2 = pts[smap.connections[i].flatten()[j], :]
                    
                    point1.x = p1[0]
                    point1.y = p1[1]
                    point1.z = p1[2]
                    point2.x = p2[0]
                    point2.y = p2[1]
                    point2.z = p2[2]
                    line_marker.points.append(point1)
                    line_marker.points.append(point2)
            marker_array.markers.append(line_marker)

        if do_surfels:
            if not smap.surfel_points is None:
                spts = transformPoints(smap.surfel_points, T_vis)

                marker = Marker()
                marker.header.frame_id = self.odom_frame  # Adjust the frame_id as needed
                marker.type = Marker.POINTS
                marker.action = Marker.ADD
                marker.pose.orientation.w = 1.0
                # marker.scale.x = 1.2  # Adjust the size of the points
                # marker.scale.y = 1.2
                marker.scale.x = ms *0.9  # Adjust the size of the points
                marker.scale.y = ms *0.9
                marker.color.a = alpha
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                if alternative_look:
                    marker.color.a = 0.5
                    if clr_index == 0:
                        marker.color.r = 1.0
                        marker.color.g = 1.0
                        marker.color.b = 0.0
                    elif clr_index == 1:
                        marker.color.r = 0.0
                        marker.color.g = 1.0
                        marker.color.b = 1.0
                    elif clr_index == 2:
                        marker.color.r = 0.4
                        marker.color.g = 0.7
                        marker.color.b = 1.0
                    elif clr_index == 42:
                        marker.color.r = 0.0
                        marker.color.g = 0.0
                        marker.color.b = 0.4
                    else:
                        marker.color.r = 1.0
                        marker.color.g = 0.0
                        marker.color.b = 1.0

                marker.id = copy.deepcopy(marker_id)
                marker_id += 1

                # Convert the 3D points to Point messages
                n_surfels = spts.shape[0]
                points_msg = [Point(x=spts[i, 0], y=spts[i, 1], z=spts[i, 2]) for i in range(n_surfels)]
                marker.points = points_msg
                marker_array.markers.append(marker)
        if do_frontiers:
            if not smap.frontier_points is None:
                spts = transformPoints(smap.frontier_points, T_vis)

                marker = Marker()
                marker.header.frame_id = self.odom_frame  # Adjust the frame_id as needed
                marker.type = Marker.POINTS
                marker.action = Marker.ADD
                marker.pose.orientation.w = 1.0
                # marker.scale.x = 1.2  # Adjust the size of the points
                # marker.scale.y = 1.2
                marker.scale.x = ms *1.2  # Adjust the size of the points
                marker.scale.y = ms *1.2
                marker.color.a = 0.7
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 1.0

                marker.id = copy.deepcopy(marker_id)
                marker_id += 1

                # Convert the 3D points to Point messages
                n_frontiers = spts.shape[0]
                points_msg = [Point(x=spts[i, 0], y=spts[i, 1], z=spts[i, 2]) for i in range(n_frontiers)]
                marker.points = points_msg
                marker_array.markers.append(marker)

        if do_normals:
            arrowlen = 0.6 * ms
            pts1 = transformPoints(smap.surfel_points, T_vis)
            pts2 = transformPoints(smap.surfel_points + smap.surfel_normals * arrowlen, T_vis)
            if not smap.surfel_points is None:
                for i in range(smap.surfel_points.shape[0]):
                    marker = Marker()
                    marker.header.frame_id = self.odom_frame  # Change this frame_id if necessary
                    marker.header.stamp = rospy.Time.now()
                    marker.type = Marker.ARROW
                    marker.action = Marker.ADD
                    marker.id = marker_id
                    marker_id += 1

                    # Set the scale
                    marker.scale.x = ms *0.25
                    marker.scale.y = ms *0.5
                    marker.scale.z = ms *0.5

                    marker.color.a = 1
                    marker.color.r = 1.0
                    marker.color.g = 0.5
                    marker.color.b = 0.0
                    if alternative_look:
                        marker.color.r = 0.7
                        marker.color.g = 0.0
                        marker.color.b = 1.0

                    pt1 = pts1[i]
                    pt2 = pts2[i]
                    points_msg = [Point(x=pt1[0], y=pt1[1], z=pt1[2]), Point(x=pt2[0], y=pt2[1], z=pt2[2])]
                    marker.points = points_msg

                    # Add the marker to the MarkerArray
                    marker_array.markers.append(marker)

        if do_keyframes:
            n_kframes = len(smap.visual_keyframes)
            if n_kframes > 0:
                kframes_pts = np.array([kf.position for kf in smap.visual_keyframes])
                kframes_pts = transformPoints(kframes_pts, T_vis)
                for i in range(n_kframes):
                    marker = Marker()
                    marker.header.frame_id = self.odom_frame  # Change this frame_id if necessary
                    marker.header.stamp = rospy.Time.now()
                    marker.type = Marker.ARROW
                    marker.action = Marker.ADD
                    marker.id = marker_id
                    marker_id += 1

                    # # Set the position (sphere center)
                    # marker.pose.position.x = kframes_pts[i][0]
                    # marker.pose.position.y = kframes_pts[i][1]
                    # marker.pose.position.z = kframes_pts[i][2]

                    # Set the scale
                    marker.scale.x = ms *0.4
                    marker.scale.y = ms *2.0
                    marker.scale.z = ms *1.0

                    marker.color.a = 1
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    if alternative_look:
                        marker.color.r = 1.0
                        marker.color.g = 0.0
                        marker.color.b = 1.0

                    arrowlen = 2
                    map_heading = transformationMatrixToHeading(T_vis)
                    xbonus = arrowlen * np.cos(smap.visual_keyframes[i].heading + map_heading)
                    ybonus = arrowlen * np.sin(smap.visual_keyframes[i].heading + map_heading)
                    points_msg = [Point(x=kframes_pts[i][0], y=kframes_pts[i][1], z=kframes_pts[i][2]), Point(x=kframes_pts[i][0]+xbonus, y=kframes_pts[i][1]+ybonus, z=kframes_pts[i][2])]
                    marker.points = points_msg

                    # Add the marker to the MarkerArray
                    marker_array.markers.append(marker)

        if do_spheres:
            for i in range(smap.points.shape[0]):
                marker = Marker()
                marker.header.frame_id = self.odom_frame  # Change this frame_id if necessary
                marker.header.stamp = rospy.Time.now()
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.id = marker_id
                marker_id += 1

                # Set the position (sphere center)
                marker.pose.position.x = pts[i][0]
                marker.pose.position.y = pts[i][1]
                marker.pose.position.z = pts[i][2]

                # Set the scale (sphere radius)
                marker.scale.x = 2 * smap.radii[i]
                marker.scale.y = 2 * smap.radii[i]
                marker.scale.z = 2 * smap.radii[i]

                marker.color.a = 0.1
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                if alternative_look:
                    marker.color.a = 0.05
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0

                # Add the marker to the MarkerArray
                marker_array.markers.append(marker)

        return True
    # # #}

