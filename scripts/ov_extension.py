#!/usr/bin/env python

# #{ imports

import copy
import rospy
from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import EmptyResponse as EmptySrvResponse
import threading

import heapq

import dbow
import rospkg

# import common_spatial
from spatial_ai.common_spatial import *
# from scommon_spatial import *
# print(inspect.getmembers(common_spatial))
# from common_spatial import SphereMap

# from sensor_msgs.msg import Image
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import mrs_msgs.msg
import std_msgs.msg
from scipy.spatial.transform import Rotation
import scipy
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
# import pcl
import inspect
# import shapely
# from shapely.geometry import Point2D
# from shapely.geometry.polygon import Polygon
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

# #}
class ScopedLock:
    def __init__(self, mutex):
        # self.lock = threading.Lock()
        self.lock = mutex

    def __enter__(self):
        print("LOCKING MUTEX")
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("UNLOCKING MUTEX")
        self.lock.release()

# #{ global variables
STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 200
kMaxNumFeature = 2000
# #}

# #{ structs and util functions
def getPixelPositions(pts, K):
    # pts = 3D points u wish to project
    pixpos = K @ pts 
    pixpos = pixpos / pixpos[2, :]
    return pixpos[:2, :].T

# LKPARAMS
lk_params = dict(winSize  = (31, 31),
                 maxLevel = 3,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def featureTracking(image_ref, image_cur, px_ref, tracking_stats):
    '''
    Performs tracking and returns correspodning well tracked features (kp1 and kp2 have same size and correspond to each other)
    '''
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]

    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    return kp1, kp2, tracking_stats[st == 1]

class TrackingStat:
    def __init__(self):
        self.age = 0
        self.invdepth_measurements = 0
        self.invdepth_mean = 0
        self.invdepth_sigma2 = 1
        self.prev_points = []
        self.invdepth_buffer = []

class KeyFrame:
    def __init__(self, odom_msg, img_timestamp, T_odom):
        self.triangulated_points = []
        self.odom_msg = odom_msg
        self.img_timestamp = img_timestamp
        self.T_odom = T_odom

# #}

# #{ class NavNode:
class NavNode:
    def __init__(self):# # #{
        self.node_initialized = False
        self.node_offline = False
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

        # SRV
        self.vocab_srv = rospy.Service("save_vocabulary", EmptySrv, self.saveCurrentVisualDatabaseToVocabFile)
        self.save_episode_full = rospy.Service("save_episode_full", EmptySrv, self.saveEpisodeFull)

        # TIMERS
        self.planning_frequency = 0.2
        self.planning_timer = rospy.Timer(rospy.Duration(1.0 / self.planning_frequency), self.planning_loop_iter)

        # PLANNING PUB
        self.path_for_trajectory_generator_pub = rospy.Publisher('/uav1/trajectory_generation/path', mrs_msgs.msg.Path, queue_size=10)

        # VIS PUB
        self.slam_points = None
        self.slam_pcl_pub = rospy.Publisher('extended_slam_points', PointCloud, queue_size=10)

        self.spheremap_spheres_pub = rospy.Publisher('spheres', MarkerArray, queue_size=10)
        self.recent_submaps_vis_pub = rospy.Publisher('recent_submaps_vis', MarkerArray, queue_size=10)
        self.path_planning_vis_pub = rospy.Publisher('path_planning_vis', MarkerArray, queue_size=10)
        self.visual_similarity_vis_pub = rospy.Publisher('visual_similarity_vis', MarkerArray, queue_size=10)

        self.kp_pub = rospy.Publisher('tracked_features_img', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('estim_depth_img', Image, queue_size=1)
        self.marker_pub = rospy.Publisher('/vo_odom', Marker, queue_size=10)
        self.kp_pcl_pub = rospy.Publisher('tracked_features_space', PointCloud, queue_size=10)
        self.kp_pcl_pub_invdepth = rospy.Publisher('tracked_features_space_invdepth', PointCloud, queue_size=10)

        # --Load calib
        # UNITY
        self.K = np.array([642.8495341420769, 0, 400, 0, 644.5958939934509, 300, 0, 0, 1]).reshape((3,3))
        self.imu_to_cam_T = np.array( [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0.0, 0.0, 0.0, 1.0]])
        self.width = 800
        self.height = 600
        ov_slampoints_topic = '/ov_msckf/points_slam'
        img_topic = '/robot1/camera1/raw'
        # img_topic = '/robot1/camera1/image'
        odom_topic = '/ov_msckf/odomimu'
        self.marker_scale = 1

        # BLUEFOX UAV
        # self.K = np.array([227.4, 0, 376, 0, 227.4, 240, 0, 0, 1]).reshape((3,3))
        # self.imu_to_cam_T = np.eye(4)
        # self.width = 752
        # self.height = 480
        # ov_slampoints_topic = '/ov_msckf/points_slam'
        # img_topic = '/uav1/vio/camera/image_raw'
        # odom_topic = '/ov_msckf/odomimu'
        # self.marker_scale = 0.5

        print("IMUTOCAM", self.imu_to_cam_T)
        self.P = np.zeros((3,4))
        self.P[:3, :3] = self.K
        print(self.P)

        # --SUB
        self.sub_cam = rospy.Subscriber(img_topic, Image, self.image_callback, queue_size=10000)

        self.odom_buffer = []
        self.odom_buffer_maxlen = 1000
        self.sub_odom = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback, queue_size=10000)
        self.sub_slam_points = rospy.Subscriber(ov_slampoints_topic, PointCloud2, self.points_slam_callback, queue_size=3)
        # self.sub_odom = rospy.Subscriber('/ov_msckf/poseimu', PoseWithCovarianceStamped, self.odometry_callback, queue_size=10000)

        self.tf_broadcaster = tf.TransformBroadcaster()

        # self.orb = cv2.ORB_create(nfeatures=3000)
        self.orb = cv2.ORB_create(nfeatures=30)

        # LOAD VOCAB FOR DBOW
        # vocab_path = rospkg.RosPack().get_path('spatial_ai') + "/vision_data/vocabulary.pickle"
        # self.visual_vocab = dbow.Vocabulary.load(vocab_path)
        # self.test_db = dbow.Database(self.visual_vocab)
        # self.test_db_n_kframes = 0
        # self.test_db_indexing = {}



        # NEW
        self.new_frame = None
        self.last_frame = None
        self.last_img_stamp = None
        self.new_img_stamp = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        # self.focal = cam.fx
        # self.pp = (cam.cx, cam.cy)
        self.focal = 643.3520818301457
        self.pp = (400, 300)
        self.tracking_bin_width = 100
        self.min_features_per_bin = 1
        self.max_features_per_bin = 2
        self.tracking_history_len = 4
        self.last_tried_landmarks_pxs = None

        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.detector = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)

        self.tracking_colors = np.random.randint(0, 255, (100, 3)) 

        self.n_frames = 0
        self.tracked_features = []

        self.local_nav_goal = None
        self.local_nav_start_time = None
        self.local_reaching_dist = 3

        self.uav_radius = 0.5
        self.min_planning_odist = 0.8
        self.max_planning_odist = 2
        self.safety_weight = 0.5
        
        self.node_initialized = True
        # # #}

    def select_random_reachable_goal_viewpoint(self):# # #{
        print("SELECTING RANDOM REACHABLE GOAL VP")
        T_global_to_imu = self.odom_msg_to_transformation_matrix(self.odom_buffer[-1])
        current_pos_in_smap_frame = (np.linalg.inv(self.spheremap.T_global_to_imu_at_start) @ T_global_to_imu)[:3, 3]

        # GET LARGEST SEGMENT SEG_ID IN SOME RADIUS AORUND CURRENT POSITION, FIND GOAL IN THAT REACHABILITY SEGMENT
        if self.spheremap.spheres_kdtree is None:
            print("GOT NO SPHERES!!!")
            return
        query_res = self.spheremap.spheres_kdtree.query(current_pos_in_smap_frame, k = 5, distance_upper_bound = 3)[1]
        bad_mask = query_res == self.spheremap.points.shape[0]
        if np.all(bad_mask):
            print("WARN: NO NEARBY SPHERES TO FIND REACAHBLE VIEWPOINT")
            return None
        near_sphere_ids = query_res[np.logical_not(bad_mask)]
        seg_ids = self.spheremap.connectivity_labels[near_sphere_ids]

        # GET ALL SEG_IDS
        unique_segs = np.unique(seg_ids)
        counts = self.spheremap.connectivity_segments_counts[unique_segs]
        largest_seg_id = unique_segs[np.argmax(counts)]
        print("LARGEST SEG N SPHERES: " + str(self.spheremap.connectivity_segments_counts[largest_seg_id]))

        # REACHABLE SPHERES:
        reachable_centroids = self.spheremap.points[self.spheremap.connectivity_labels == largest_seg_id]
        deltavecs = reachable_centroids - current_pos_in_smap_frame
        dists = np.linalg.norm(deltavecs, axis = 1)
        
        # best_idx = np.argmax(dists)
        best_idx = np.argmax(deltavecs[:, 0]) # MOST INFRONT
        print("SELECTING GOAL METRES AWAY: " + str(dists[best_idx]))
        print(reachable_centroids[best_idx, :])
        print(reachable_centroids[best_idx, :].shape)

        res_point_in_odom_frame = transformPoints(reachable_centroids[best_idx, :].reshape((1,3)), self.spheremap.T_global_to_imu_at_start)
        print("GOAL IN ODOM_FRAME: ")
        print(res_point_in_odom_frame )
        return Viewpoint(res_point_in_odom_frame, None)

# # #}

    def find_path_rrt_multiple_submaps(self, start_vp, goal_vp, maps, sampling_dist=0.2):# # #{
        return None
# # #}

    def send_path_to_trajectory_generator(self, path):# # #{
        print("SENDING PATH TO TRAJ GENERATOR, LEN: " + str(len(path)))
        msg = mrs_msgs.msg.Path()
        msg.header = std_msgs.msg.Header()
        # msg.header.frame_id = 'global'
        msg.header.frame_id = 'uav1/vio_origin'
        # msg.header.frame_id = 'uav1/fcu'
        # msg.header.frame_id = 'uav1/local_origin'
        msg.header.stamp = rospy.Time.now()
        msg.use_heading = False
        msg.fly_now = True
        msg.stop_at_waypoints = False
        arr = []
        print(msg.points)

        for p in path:
            ref = mrs_msgs.msg.Reference()
            ref.position = Point(x=-p[0], y=-p[1], z=p[2])
            # arr.append(ref)
            msg.points.append(ref)
        # print("ARR:")
        # print(arr)
        # msg.points = arr

        self.path_for_trajectory_generator_pub.publish(msg)
        print("SENT PATH TO TRAJ GENERATOR")

        return None
# # #}

    def planning_loop_iter(self, event=None):# # #{
        print("pes")
        with ScopedLock(self.spheremap_mutex):
            print("pespes")
            if self.spheremap is None:
                return
            if not self.node_initialized:
                return
            print("kocka")

            T_global_to_imu = self.odom_msg_to_transformation_matrix(self.odom_buffer[-1])
            pos_odom_frame = T_global_to_imu[:3, 3]
            print("ODOM FRAME POS:")
            print(pos_odom_frame)
            heading_odom_frame = transformationMatrixToHeading(T_global_to_imu)
            current_vp = Viewpoint(pos_odom_frame, heading_odom_frame)

            # FIND GOAL IF NONE
            max_reaching_time = 5
            if self.local_nav_goal is None or (rospy.get_rostime() - self.local_nav_start_time).to_sec() > max_reaching_time:
                self.local_nav_goal = self.select_random_reachable_goal_viewpoint()
                self.local_nav_start_time = rospy.get_rostime()
                return

            # CHECK IF REACHED
            dist_to_goal = np.linalg.norm(self.local_nav_goal.position - pos_odom_frame)
            print("DIST TO GOAL: " + str(dist_to_goal))
            if dist_to_goal < self.local_reaching_dist:
                print("GOAL REACHED!!!")
                self.local_nav_goal = None
                return

            # VISUALIZE GOAL POSITION
            # TODO

            # FIND PATH TO GOAL AND SEND IT TO TRAJECTORY GENERATOR
            # TRY A* PATH PLANNING FOR NOW!
            T_global_to_smap_origin = np.linalg.inv(self.spheremap.T_global_to_imu_at_start)
            print("T:")
            print(T_global_to_smap_origin)
            current_pos_in_smap_frame = (T_global_to_smap_origin @ T_global_to_imu)[:3, 3]
            print("CURRENT POS IN SMAP FRAME:")
            print(current_pos_in_smap_frame)
            goal_pos_in_smap_frame = transformPoints(self.local_nav_goal.position, T_global_to_smap_origin)

            pathfindres = self.findPathAstarInSubmap(self.spheremap, current_pos_in_smap_frame, goal_pos_in_smap_frame)
            if not pathfindres is None:
                print("FOUND SOME PATH!")
                print(pathfindres)
                # pathfindres = np.concatenate(pathfindres, goal_pos_in_smap_frame.reshape((3,1)))

                print("TRANSFORMED PATH:")
                path_in_global_frame = transformPoints(pathfindres, self.spheremap.T_global_to_imu_at_start)
                print(path_in_global_frame )

                # IGNORE VPS TOO CLOSE TO UAV
                ignoring_dist = 1
                # TODO - use predicted trajectory to not stop like a retard
                first_to_not_ignore = path_in_global_frame.shape[0] - 1
                for i in range(path_in_global_frame.shape[0] - 1):
                    if np.linalg.norm(path_in_global_frame - current_pos_in_smap_frame) > ignoring_dist:
                        first_to_not_ignore = i
                        break
                print("IGNORING PTS: " + str(first_to_not_ignore))
                path_in_global_frame = path_in_global_frame[i:, :]

                print("GONNA SEND THIS PATH:")
                print(path_in_global_frame )


                self.send_path_to_trajectory_generator(path_in_global_frame)
            else:
                print("NO PATH FOUND! SELECTING NEW GOAL!")
                self.local_nav_goal = None
            # planning_smaps = [self.spheremap]
            # rrt_path = self.find_path_rrt_multiple_submaps(


# # #}

    def findGlobalPath(self, start_vp, goal_vp, goal_vp_map_index):
        # VIEWPOINTS SHOULD BE IN RELATIVE COORDS OF THE SUBMAPS!!!
        # TEST TOPOLOGICAL REACHABILITY FIRST!
        print("GLOBAL PATHFINDING TO POINT AND SUBMAP_ID:")
        print(goal_vp.position)
        print(goal_vp_map_index)
    
        n_old_submaps = len(self.mchunk.submaps)
        reached_flags = np.full((n_old_submaps), False)
        reaching_costs = np.zeros((n_old_submaps))

        # open_set = []
        # Priority queue for efficient retrieval of the node with the lowest total cost
        
        # while open_set:
        #     current_f, current_index, parent = heapq.heappop(open_set)
        #     smap = self.mchunk.submaps[current_index]
        #     reached_flags[current_index] = True

        #     if current_index == goal_vp_map_index:
        #         path = [current_index]
        #         while not parent is None:
        #             path.append(parent)
        #             current_index = parent
        #             parent = closed_set[current_index]
        #         break

        #     for conn in self.spheremap.map2map_conns:
        #         if not reached_flags[conn.second_map_id]:
        #             euclid_dist = np.linalg.norm(conn.pt_in_first_map_frame - start_vp.position)
        #             heapq.heappush(open_set, (euclid_dist, conn.second_map_id, None))


        open_set = []
        # closed_set = set()
        closed_set = {}

        # Priority queue for efficient retrieval of the node with the lowest total cost

        # heapq.heappush(open_set, (0, start_node_index, None))
        # g_score = {start_node_index: 0}
        g_score = {}

        for conn in self.spheremap.map2map_conns:
            euclid_dist = np.linalg.norm(conn.pt_in_first_map_frame - start_vp.position)
            heapq.heappush(open_set, (euclid_dist, conn.second_map_id, n_old_submaps))
            print("STARTING CONN TO: " + str(conn.second_map_id))
            closed_set[conn.second_map_id] = n_old_submaps
            g_score[conn.second_map_id] = euclid_dist

        # Dictionary to store the cost of reaching each node from the start

        # Dictionary to store the estimated total cost from start to goal passing through the node
        # startned_dist = np.linalg.norm(smap.points[start_node_index] - goal_node_pt)
        # print("PATHFIND: startnend dist:" + str(startned_dist))
        # f_score = {start_node_index: startned_dist }
        # print("TOPO PLANNING! N SUBMAPS IN HISTORY: " + str(n_old_submaps))

        path = []
        while open_set:
            current_f, current_index, parent = heapq.heappop(open_set)
            # print("POPPED: " + str(current_index) + " F: " + str(current_f))

            smap = self.mchunk.submaps[current_index]
            reached_flags[current_index] = True

            if current_index == goal_vp_map_index:
                path = [current_index]
                # while not parent is None:
                while parent != n_old_submaps:
                    path.append(parent)
                    current_index = parent
                    parent = closed_set[current_index]
                break

            # closed_set.add(current_index)

            conns = smap.map2map_conns
            if conns is None:
                continue

            for c in conns:
                neighbor_index = c.second_map_id
                # print("NEIGHBOR: " + str(neighbor_index))
                if neighbor_index == n_old_submaps or reached_flags[neighbor_index]:
                    continue
                pt_prev = smap.getPointOfConnectionToSubmap(parent)
                pt_next = c.pt_in_first_map_frame 

                # euclid_dist = np.linalg.norm(smap.points[current_index] - smap.points[neighbor_index])
                euclid_dist = np.linalg.norm(pt_prev - pt_next)
                tentative_g = g_score[current_index] + euclid_dist

                # if neighbor_index not in g_score or tentative_g < g_score[neighbor_index]:
                if neighbor_index not in g_score or tentative_g < g_score[neighbor_index]:
                    g_score[neighbor_index] = tentative_g
                    heapq.heappush(open_set, (g_score[neighbor_index], neighbor_index, current_index))
                    closed_set[neighbor_index] = current_index
        path.reverse()
        print("TOPO PLANNING FOUND PATH:")
        print(path)

        # RETURNING THIS TOPOLOGICAL PATH




    def points_slam_callback(self, msg):# # #{
        if self.node_offline:
            return
        print("PCL MSG")
        with ScopedLock(self.spheremap_mutex):
            print("PCL MSG PROCESSING NOW")
            comp_start_time = time.time()

            # Read Points from OpenVINS# #{
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
                print("NO POINTS?")
                return
            point_cloud_array = point_cloud_array[nonzero_pts, :]
# # #}

            # DECIDE WHETHER TO UPDATE SPHEREMAP OR INIT NEW ONE# #{
            T_global_to_imu = self.odom_msg_to_transformation_matrix(self.odom_buffer[-1])
            self.submap_unpredictability_signal = 0
            init_new_spheremap = False
            memorized_transform_to_prev_map = None
            init_rad = 0.5
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
                    if self.spheremap.traveled_context_distance > 7:
                        split_score = max_split_score
                    else:
                        print("TRAVELED CONTEXT DISTS: " + str(self.spheremap.traveled_context_distance))

                    if split_score >= max_split_score:
                        # SAVE OLD SPHEREMAP
                        print("SPLITTING!")
                        # TODO give it the transform to the next ones origin from previous origin
                        memorized_transform_to_prev_map = np.linalg.inv(self.spheremap.T_global_to_own_origin) @ T_global_to_imu
                        self.mchunk.submaps.append(self.spheremap)
                        init_new_spheremap = True
                        self.visualize_episode_submaps()
            if init_new_spheremap:
                # INIT NEW SPHEREMAP
                print("INFO: initing new spheremap")

                # CREATE CONNECTION FROM AND TO THE PREV SPHEREMAP IF PREV MAP EXISTS!
                connection_to_prev_map = None
                if (not self.spheremap is None) and (not memorized_transform_to_prev_map is None):
                    print("CREATING CONNECTION TO PREV MAP")
                    pos_in_old_frame = memorized_transform_to_prev_map[:3,3].reshape((1,3))
                    pos_in_new_frame = np.zeros((1,3))
                    odist = self.spheremap.getDistFromObstaclesAtPoint(pos_in_old_frame)
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

                self.spheremap = SphereMap(init_rad, min_rad)
                self.spheremap.surfels_filtering_radius = 0.2

                if not connection_to_prev_map is None:
                    self.spheremap.map2map_conns.append(connection_to_prev_map)

                self.spheremap.memorized_transform_to_prev_map = memorized_transform_to_prev_map 
                # self.spheremap.T_global_to_own_origin = T_global_to_imu @ self.imu_to_cam_T
                self.spheremap.T_global_to_own_origin = T_global_to_imu 
                self.spheremap.T_global_to_imu_at_start = T_global_to_imu
                self.spheremap.creation_time = rospy.get_rostime()

            if len(self.mchunk.submaps) > 2:
                self.saveEpisodeFull(None)
            # # #}

            # ---CURRENT SPHEREMAP UPDATE WITH CURRENTLY TRIANGULATED POINTS # #{

            # TRANSFORM SLAM PTS TO IMAGE AND COMPUTE THEIR PIXPOSITIONS
            if len(self.odom_buffer) == 0:
                print("WARN: ODOM BUFFER EMPY!")
                return
            transformed = transformPoints(point_cloud_array, (self.imu_to_cam_T @ np.linalg.inv(T_global_to_imu))).T
            positive_z_idxs = transformed[2, :] > 0
            final_points = transformed[:, positive_z_idxs]

            pixpos = self.getPixelPositions(final_points)

            # COMPUTE DELAUNAY TRIANG OF VISIBLE SLAM POINTS
            print("HAVE DELAUNAY:")
            tri = Delaunay(pixpos)
            # vis = self.visualize_depth(pixpos, tri)
            # self.depth_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))

            # CONSTRUCT OBSTACLE MESH
            comp_mesh = time.time()
            obstacle_mesh = trimesh.Trimesh(vertices=final_points.T, faces = tri.simplices)

            # CONSTRUCT POLYGON OF PIXPOSs OF VISIBLE SLAM PTS
            hull = ConvexHull(pixpos)
            print("POLY")
            img_polygon = geometry.Polygon(hull.points)

            # CONSTRUCT HULL MESH FROM 3D POINTS OF CONVEX 2D HULL OF PROJECTED POINTS
            hullmesh_pts = final_points[:, np.unique(hull.simplices)]
            orig_pts = np.zeros((3, 1))
            hullmesh_pts = np.concatenate((hullmesh_pts, orig_pts), axis=1)
            zero_pt_index = hullmesh_pts.shape[1] - 1

            # FUCK IT JUST DO CONV HULL OF THESE 3D PTS(just start and hull pts)
            fov_hull = ConvexHull(hullmesh_pts.T)

            uniq_idxs_in_hull = np.unique(fov_hull.simplices)
            simplices_reindexing = {uniq_idxs_in_hull[i]:i for i in range(uniq_idxs_in_hull.size)}
            simplices_reindexed = [[simplices_reindexing[orig[0]], simplices_reindexing[orig[1]], simplices_reindexing[orig[2]]] for orig in fov_hull.simplices]

            # CONSTRUCT FOV MESH AND QUERY
            fov_mesh = trimesh.Trimesh(vertices=fov_hull.points, faces = simplices_reindexed)
            fov_mesh_query = trimesh.proximity.ProximityQuery(fov_mesh)

            # CONSTRUCT OBSTACLE POINT MESH AND QUERY
            obstacle_mesh_query = trimesh.proximity.ProximityQuery(obstacle_mesh)

            comp_time = time.time() - comp_mesh
            print("MESHING time: " + str((comp_time) * 1000) +  " ms")


            # ---UPDATE AND PRUNING STEP
            T_current_cam_to_orig = np.eye(4)
            T_delta_odom  = np.eye(4)
            if not self.spheremap is None:
                # transform existing sphere points to current camera frame
                # T_spheremap_orig_to_current_cam = self.imu_to_cam_T @ T_global_to_imu @ np.linalg.inv(self.spheremap.T_global_to_own_origin) 
                # T_current_cam_to_orig = np.linalg.inv(T_spheremap_orig_to_current_cam)
                # T_spheremap_orig_to_current_cam = np.linalg.inv(self.imu_to_cam_T)
                # print("T DELTA ODOM")
                # print(T_delta_odom)
                # print("GLOBAL TO OWN ORIGIN")
                # print(self.spheremap.T_global_to_own_origin)
                # print("ORIG TO NOW")
                # print(T_spheremap_orig_to_current_cam )
                n_spheres_old = self.spheremap.points.shape[0]

                T_delta_odom = np.linalg.inv(self.spheremap.T_global_to_imu_at_start) @ T_global_to_imu
                # T_current_cam_to_orig = self.imu_to_cam_T @ T_delta_odom @ np.linalg.inv(self.imu_to_cam_T)

                T_current_cam_to_orig = T_delta_odom @ np.linalg.inv(self.imu_to_cam_T)

                print("FINAL MATRIX")
                print(T_current_cam_to_orig)

                # project sphere points to current camera frame
                transformed_old_points  = transformPoints(self.spheremap.points, np.linalg.inv(T_current_cam_to_orig))

                print("TRANSFORMED EXISTING SPHERE POINTS TO CURRENT CAMERA FRAME!")

                # Filter out spheres with z below zero or above the max z of obstacle points
                # TODO - use dist rather than z for checking
                max_vis_z = np.max(final_points[2, :])
                print(max_vis_z )
                z_ok_idxs = np.logical_and(transformed_old_points[:, 2] > 0, transformed_old_points[:, 2] <= max_vis_z)

                z_ok_points = transformed_old_points[z_ok_idxs , :] # remove spheres with negative z
                worked_sphere_idxs = np.arange(n_spheres_old)[z_ok_idxs ]
                print("POSITIVE Z AND NOT BEHIND OBSTACLE MESH:" + str(np.sum(z_ok_idxs )) + "/" + str(n_spheres_old))

                # check the ones that are projected into the 2D hull
                old_pixpos = self.getPixelPositions(z_ok_points.T)
                inhull = np.array([img_polygon.contains(geometry.Point(old_pixpos[i, 0], old_pixpos[i, 1])) for i in range(old_pixpos.shape[0])])
                
                print("OLD PTS IN HULL:" + str(np.sum(inhull)) + "/" + str(z_ok_points.shape[0]))

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


            # ---EXPANSION STEP #{
            max_sphere_sampling_z = 60

            n_sampled = 50
            sampling_pts = np.random.rand(n_sampled, 2)  # Random points in [0, 1] range for x and y
            sampling_pts = sampling_pts * [self.width, self.height]

            # CHECK THE SAMPLING DIRS ARE INSIDE THE 2D CONVEX HULL OF 3D POINTS
            inhull = np.array([img_polygon .contains(geometry.Point(p[0], p[1])) for p in sampling_pts])
            # print(inhull)
            if not np.any(inhull):
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
            print("PUTATIVE SPHERES THAT PASSED FIRST RADIUS CHECKS: " + str(n_spheres_to_add))
            if n_spheres_to_add == 0:
                self.spheremap.labelSpheresByConnectivity()
                return

            # print(mindists.shape)
            # print(mindists[new_sphere_idxs].shape)

            # TRANSFORM POINTS FROM CAM ORIGIN TO SPHEREMAP ORIGIN! - DRAW OUT!
            new_sphere_locations_in_spheremap_origin = transformPoints(sampling_pts[:, new_sphere_idxs].T, T_current_cam_to_orig)
            # print(new_sphere_locations_in_spheremap_origin.shape)
            # print(self.spheremap.points.shape)

            self.spheremap.points = np.concatenate((self.spheremap.points, new_sphere_locations_in_spheremap_origin))
            self.spheremap.radii = np.concatenate((self.spheremap.radii.flatten(), mindists[new_sphere_idxs].flatten()))
            self.spheremap.connections = np.concatenate((self.spheremap.connections.flatten(), np.array([None for i in range(n_spheres_to_add)], dtype=object).flatten()))

            self.spheremap.updateConnections(np.arange(n_spheres_before_adding, n_spheres_before_adding+n_spheres_to_add))

            new_idxs = np.arange(self.spheremap.radii.size)[self.spheremap.radii.size - n_spheres_to_add : self.spheremap.radii.size]
            self.spheremap.removeSpheresIfRedundant(new_idxs)

            self.spheremap.labelSpheresByConnectivity()
            # # #}

            comp_time = time.time() - comp_start_time
            print("SPHEREMAP integration time: " + str((comp_time) * 1000) +  " ms")

            comp_start_time = time.time()
            self.spheremap.spheres_kdtree = KDTree(self.spheremap.points)
            self.spheremap.max_radius = np.max(self.spheremap.radii)
            comp_time = time.time() - comp_start_time
            print("Sphere KDTree computation: " + str((comp_time) * 1000) +  " ms")

            # HANDLE ADDING/REMOVING VISIBLE 3D POINTS
            comp_start_time = time.time()

            visible_pts_in_spheremap_frame = transformPoints(final_points.T, T_current_cam_to_orig)
            self.spheremap.updateSurfels(visible_pts_in_spheremap_frame , pixpos, tri.simplices)

            comp_time = time.time() - comp_start_time
            print("SURFELS integration time: " + str((comp_time) * 1000) +  " ms")

            # VISUALIZE CURRENT SPHERES
            self.visualize_spheremap()
            comp_time = time.time() - comp_start_time
            print("SPHEREMAP visualization time: " + str((comp_time) * 1000) +  " ms")

            # TRY PATHFINDING TO START OF CURRENT SPHEREMAP
            # pathfindres = self.findPathAstarInSubmap(self.spheremap, T_delta_odom[:3, 3], np.array([0,0,0]))
            if len(self.mchunk.submaps) > 0:
                self.findGlobalPath(Viewpoint(T_current_cam_to_orig[:3,3].reshape((3,1))), Viewpoint(np.zeros((3,1))), 0)

            # self.node_offline = not self.spheremap.consistencyCheck()
# # #}

    def saveEpisodeFull(self, req):# # #{
        self.node_offline = True
        print("SAVING EPISODE MEMORY CHUNK")

        fpath = rospkg.RosPack().get_path('spatial_ai') + "/memories/last_episode.pickle"
        self.mchunk.submaps.append(self.spheremap)
        self.mchunk.save(fpath)
        self.mchunk.submaps.pop()

        self.node_offline = False
        print("EPISODE SAVED TO " + str(fpath))
        return EmptySrvResponse()# # #}

    def saveCurrentVisualDatabaseToVocabFile(self, req):# # #{
        self.node_offline = True
        n_words = 1000
        depth = 3

        # vocab = dbow.Vocabulary()
        vocab = EmptyClass()
        vocab.__class__ = dbow.Vocabulary
        
        descriptors = []
        n_keyframes = 0
        n_total_kfs = 0
        for smap in self.mchunk.submaps:
            for kf in smap.visual_keyframes:
                n_total_kfs += 1
        if not self.spheremap is None:
            for kf in self.spheremap.visual_keyframes:
                n_total_kfs += 1

        print("VOCAB N KFS:" + str(n_keyframes))
        for smap in self.mchunk.submaps:
            for kf in smap.visual_keyframes:
                # descriptors.append(kf.descs)
                descriptors = descriptors + kf.descs
                n_keyframes += 1
                print("VOCAB SAVING " + str(n_keyframes) + "/" + str(n_total_kfs))
        if not self.spheremap is None:
            for kf in self.spheremap.visual_keyframes:
                # descriptors.append(kf.descs)
                descriptors = descriptors + kf.descs
                n_keyframes += 1
                print("VOCAB SAVING " + str(n_keyframes) + "/" + str(n_total_kfs))

        descriptors = np.array(descriptors)
        vocab.root_node = dbow.Node(descriptors)
        words = dbow.initialize_tree(vocab.root_node, n_words, depth)
        vocab.words = [dbow.Word.from_node(node) for node in words]
        for word in vocab.words:
            word.update_weight(n_keyframes)

        vocab_path = rospkg.RosPack().get_path('spatial_ai') + "/vision_data/vocabulary_tmp.pickle"
        vocab.save(vocab_path)

        print("VOCAB SAVE SUCCESS")
        return EmptySrvResponse()# # #}

    def findPathAstarInSubmap(self, smap, startpoint, endpoint, maxdist_to_graph=10, min_safe_dist=0.5, max_safe_dist=3, safety_weight=1):# # #{
        print("PATHFIND: STARTING, FROM TO:")
        print(startpoint)
        print(endpoint)

        if smap.points is None: 
            print("PATHFIND: MAP IS EMPTY!!!")
            return None

        # FIND NEAREST NODES TO START AND END
        dist_from_startpoint = np.linalg.norm((smap.points - startpoint), axis=1) - smap.radii
        start_node_index = np.argmin(dist_from_startpoint)
        start_seg_id = smap.connectivity_labels[start_node_index]

        # ONLY CONSIDER GOAL POINTS IN THE SAME CONNECTIVITY REGION
        seg_mask = smap.connectivity_labels == start_seg_id
        mask_idxs = np.where(seg_mask)[0]
        # print(mask_idxs)
        # dist_from_endpoint = np.linalg.norm((smap.points - endpoint), axis=1) - smap.radii
        dist_from_endpoint = np.linalg.norm((smap.points[seg_mask, :] - endpoint), axis=1) - smap.radii[seg_mask]
        end_node_index_masked = np.argmin(dist_from_endpoint)
        # print(end_node_index_masked)
        end_node_index = mask_idxs[end_node_index_masked]
        # print(end_node_index)

        if(dist_from_startpoint[start_node_index ] > maxdist_to_graph or dist_from_endpoint[end_node_index_masked] > maxdist_to_graph):
            print("PATHFIND: start or end too far form graph!")
            print(dist_from_startpoint[start_node_index])
            print(dist_from_endpoint[end_node_index_masked])
            return None
        print("PATHFIND: node indices:")
        print(start_node_index)
        print(end_node_index)
        print("PATHFIND: node segments:")
        print(start_seg_id)
        print(smap.connectivity_labels[end_node_index])

        goal_node_pt = smap.points[end_node_index]
        open_set = []
        # closed_set = set()
        closed_set = {start_node_index: None}

        # Priority queue for efficient retrieval of the node with the lowest total cost
        heapq.heappush(open_set, (0, start_node_index, None))

        # Dictionary to store the cost of reaching each node from the start
        g_score = {start_node_index: 0}

        # Dictionary to store the estimated total cost from start to goal passing through the node
        startned_dist = np.linalg.norm(smap.points[start_node_index] - goal_node_pt)
        print("PATHFIND: startnend dist:" + str(startned_dist))
        f_score = {start_node_index: startned_dist }

        path = []
        while open_set:
            current_f, current_index, parent = heapq.heappop(open_set)
            # print("POPPED: " + str(current_index) + " F: " + str(current_f))

            if current_index == end_node_index:
                path = [current_index]
                while not parent is None:
                    path.append(parent)
                    current_index = parent
                    parent = closed_set[current_index]
                break

            # closed_set.add(current_index)

            conns = smap.connections[current_index]
            # print("CONNS:")
            # print(conns)
            if conns is None:
                continue

            for neighbor_index in conns:
                euclid_dist = np.linalg.norm(smap.points[current_index] - smap.points[neighbor_index])
                tentative_g = g_score[current_index] + euclid_dist

                if neighbor_index not in g_score or tentative_g < g_score[neighbor_index]:
                    g_score[neighbor_index] = tentative_g
                    f_score[neighbor_index] = tentative_g + np.linalg.norm(smap.points[neighbor_index] - goal_node_pt)
                    heapq.heappush(open_set, (f_score[neighbor_index], neighbor_index, current_index))
                    closed_set[neighbor_index] = current_index

        path.reverse()
        print("PATHFIND: RES PATH: ")
        print(path)
        print("N CLOSED NODES: " + str(len(closed_set.keys())))

        self.visualize_trajectory(smap.points[np.array(path), :], smap.T_global_to_own_origin, startpoint, endpoint)

        if len(path) == 0:
            return None
        return smap.points[np.array(path), :]
# # #}

    def getPixelPositions(self, pts):# # #{
        # pts = 3D points u wish to project
        return getPixelPositions(pts, self.K)# # #}

    def get_closest_time_odom_msg(self, stamp):# # #{
        bestmsg = None
        besttimedif = 0
        for msg in self.odom_buffer:
            # print(msg.header.stamp.to_sec())
            tdif2 = np.abs((msg.header.stamp - stamp).to_nsec())
            if bestmsg == None or tdif2 < besttimedif:
                bestmsg = msg
                besttimedif = tdif2
        final_tdif = (msg.header.stamp - stamp).to_sec()
        # print("TARGET")
        # print(stamp)
        # print("FOUND")
        # print(msg.header.stamp)
        # if not bestmsg is None:
            # print("found msg with time" + str(msg.header.stamp.to_sec()) + " for time " + str(stamp.to_sec()) +" tdif: " + str(final_tdif))
        return bestmsg# # #}

    def odometry_callback(self, msg):# # #{
        self.odom_buffer.append(msg)
        if len(self.odom_buffer) > self.odom_buffer_maxlen:
            self.odom_buffer.pop(0)# # #}
                    
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

    def control_features_population(self):# # #{
        wbins = self.width // self.tracking_bin_width
        hbins = self.height // self.tracking_bin_width
        found_total = 0

        if self.px_cur is None or len(self.px_cur) == 0:
            self.px_cur = self.detector.detect(self.new_frame)
            if self.px_cur is None or len(self.px_cur) == 0:
                return
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
            self.tracking_stats = np.array([TrackingStat() for x in self.px_cur], dtype=object)

        # print("STTAS:")
        # print(self.tracking_stats.shape)
        # print(self.px_cur.shape)

        self.px_cur_prev = copy.deepcopy(self.px_cur)
        self.px_cur = self.px_cur[:0, :]

        self.tracking_stats_prev = copy.deepcopy(self.tracking_stats)
        self.tracking_stats = self.tracking_stats[:0]

        n_culled = 0

        for xx in range(wbins):
            for yy in range(hbins):
                # count how many we have there and get the points in there:
                ul = np.array([xx * self.tracking_bin_width , yy * self.tracking_bin_width ])  
                lr = np.array([ul[0] + self.tracking_bin_width , ul[1] + self.tracking_bin_width]) 

                inidx = np.all(np.logical_and(ul <= self.px_cur_prev, self.px_cur_prev <= lr), axis=1)
                # print(inidx)
                inside_points = []
                inside_stats = np.array([], dtype=object)

                n_existing_in_bin = 0
                if np.any(inidx):
                    inside_points = self.px_cur_prev[inidx]
                    inside_stats = self.tracking_stats_prev[inidx]
                    n_existing_in_bin = inside_points.shape[0]

                # print(n_existing_in_bin )

                if n_existing_in_bin > self.max_features_per_bin:
                    # CUTOFF POINTS ABOVE MAXIMUM, SORTED BY AGE
                    ages = np.array([-pt.age for pt in inside_stats])
                    # idxs = np.arange(self.max_features_per_bin)
                    # np.random.shuffle(idxs)
                    idxs = np.argsort(ages)
                    surviving_idxs = idxs[:self.max_features_per_bin]
                    n_culled_this_bin = n_existing_in_bin - self.max_features_per_bin

                    self.px_cur = np.concatenate((self.px_cur, inside_points[surviving_idxs , :]))
                    self.tracking_stats = np.concatenate((self.tracking_stats, inside_stats[surviving_idxs ]))

                    # LET THE ONES WITH MANY DEPTH MEASUREMENTS LIVE
                    # print(n_culled_this_bin)
                    spared_inside_idxs = []
                    n_spared = 0
                    for i in range(n_culled_this_bin):
                        if inside_stats[idxs[self.max_features_per_bin + i]].invdepth_measurements > 1:
                            spared_inside_idxs.append(self.max_features_per_bin + i)
                            n_spared += 1
                    if n_spared > 0:
                        spared_inside_idxs = np.array(spared_inside_idxs)

                        self.px_cur = np.concatenate((self.px_cur, inside_points[idxs[spared_inside_idxs] , :]))
                        self.tracking_stats = np.concatenate((self.tracking_stats, inside_stats[idxs[spared_inside_idxs] ]))

                    n_culled += n_culled_this_bin - n_spared
                    # self.px_cur = np.concatenate((self.px_cur, inside_points[:self.max_features_per_bin, :]))

                elif n_existing_in_bin < self.min_features_per_bin:
                    # ADD THE EXISTING
                    if n_existing_in_bin > 0:
                        self.px_cur = np.concatenate((self.px_cur, inside_points))
                        self.tracking_stats = np.concatenate((self.tracking_stats, inside_stats))

                    # FIND NEW ONES
                    locally_found = self.detector.detect(self.new_frame[ul[1] : lr[1], ul[0] : lr[0]])
                    n_found_in_bin = len(locally_found)
                    if n_found_in_bin == 0:
                        continue
                    locally_found = np.array([x.pt for x in locally_found], dtype=np.float32)

                    # be sure to not add too many!
                    if n_existing_in_bin + n_found_in_bin > self.max_features_per_bin:
                        n_to_add = int(self.max_features_per_bin - n_existing_in_bin)

                        shuf = np.arange(n_to_add)
                        np.random.shuffle(shuf)

                        # locally_found = locally_found[:n_to_add]
                        locally_found = locally_found[shuf, :]

                    found_total += len(locally_found)

                    # ADD THE NEW ONES
                    locally_found[:, 0] += ul[0]
                    locally_found[:, 1] += ul[1]
                    self.px_cur = np.concatenate((self.px_cur, locally_found))
                    # self.tracking_stats = np.array([TrackingStat for x in locally_found], dtype=object)
                    self.tracking_stats = np.concatenate((self.tracking_stats, np.array([TrackingStat() for x in locally_found], dtype=object)))
                else:
                    # JUST COPY THEM
                    self.px_cur = np.concatenate((self.px_cur, inside_points))
                    # self.tracking_stats += inside_stats
                    self.tracking_stats = np.concatenate((self.tracking_stats, inside_stats))

        print("FOUND IN BINS: " + str(found_total) + " CULLED: " + str(n_culled))
        print("CURRENT FEATURES: " + str(self.px_cur.shape[0]))

        # FIND FEATS IF ZERO!
        if(self.px_cur.shape[0] == 0):
            print("ZERO FEATURES! FINDING FEATURES")
            self.px_cur = self.detector.detect(self.new_frame)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
# # #}
        
    def odom_msg_to_transformation_matrix(self, closest_time_odom_msg):# # #{
        odom_p = np.array([closest_time_odom_msg.pose.pose.position.x, closest_time_odom_msg.pose.pose.position.y, closest_time_odom_msg.pose.pose.position.z])
        odom_q = np.array([closest_time_odom_msg.pose.pose.orientation.x, closest_time_odom_msg.pose.pose.orientation.y,
            closest_time_odom_msg.pose.pose.orientation.z, closest_time_odom_msg.pose.pose.orientation.w])
        T_odom = np.eye(4)
        # print("R:")
        # print(Rotation.from_quat(odom_q).as_matrix())
        T_odom[:3,:3] = Rotation.from_quat(odom_q).as_matrix()
        T_odom[:3, 3] = odom_p
        return T_odom# # #}

    def image_callback(self, msg):# # #{
        if self.node_offline:
            return
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        comp_start_time = rospy.get_rostime()

        # SHIFT LAST AND NEW
        self.last_img_stamp = self.new_img_stamp 
        self.new_img_stamp  = msg.header.stamp

        self.last_frame = self.new_frame
        self.new_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.px_ref = self.px_cur

        print("FRAME " + str(self.n_frames))

        comp_start_time = time.time()

        # RETURN IF FIRST FRAME
        if self.n_frames == 0:
            # FIND FIRST FEATURES
            self.n_frames = 1
            # self.last_img_stamp = stamp
            return
        self.n_frames += 1


        # ADD NEW VISUAL KEYFRAME IF NEW ENOUGH
        if self.spheremap is None:
            return

        # GET ODOM MSG CLOSEST TO CURRENT IMG TIMESTAMP
        closest_time_odom_msg = self.get_closest_time_odom_msg(self.new_img_stamp)
        T_odom = self.odom_msg_to_transformation_matrix(closest_time_odom_msg)
        T_odom_relative_to_smap_start = np.linalg.inv(self.spheremap.T_global_to_own_origin) @ T_odom

        new_kf = SubmapKeyframe(T_odom_relative_to_smap_start)


        # CHECK IF NEW ENOUGH
        # TODO - check in near certainly connected submaps
        for kf in self.spheremap.visual_keyframes:
            # TODO - scaling
            if kf.euclid_dist(new_kf) < 5 and kf.heading_dif(new_kf) < 3.14159 /4:
                print("KFS: not novel enough, N keyframes: " + str(len(self.spheremap.visual_keyframes)))
                return

        if len(self.spheremap.visual_keyframes) > 0:
            dist_bonus = new_kf.euclid_dist(self.spheremap.visual_keyframes[-1])
            heading_bonus = new_kf.heading_dif(self.spheremap.visual_keyframes[-1]) * 0.2

            self.spheremap.traveled_context_distance += dist_bonus + heading_bonus

        print("KFS: adding new visual keyframe!")
        comp_start_time = time.time()
        # COMPUTE DESCRIPTION AND ADD THE KF
        kps, descs = self.orb.detectAndCompute(self.new_frame, None)
        print(len(kps))
        descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
        new_kf.descs = descs

        # self.test_db.add(descs)
        # self.test_db_n_kframes += 1
        self.spheremap.visual_keyframes.append(new_kf)



        # self.test_db_indexing[self.test_db_n_kframes - 1] = (len(self.mchunk.submaps), len(self.spheremap.visual_keyframes)-1) 

        # scores = self.test_db.query(descs)

        # self.visualize_keyframe_scores(scores, new_kf)
        # print("SCORES:")
        # print(scores)



        # TODO - get some vocab into here SOMEHOW
        # TODO - describe, check difference from other KFs

        comp_time = time.time() - comp_start_time
        print("KFS: kf addition time: " + str((comp_time) * 1000) +  " ms")# # #}

        # ---------------------------FUCK TRACKING FOR NOW
        return
        # IF YOU CAN - TRACK
        if not self.px_ref is None:
            print("BEFORE TRACKING: " + str(self.px_ref.shape[0]))
            self.px_ref, self.px_cur, self.tracking_stats = featureTracking(self.last_frame, self.new_frame, self.px_ref, self.tracking_stats)

            for i in range(self.px_ref.shape[0]):
                self.tracking_stats[i].age += 1
                self.tracking_stats[i].prev_points.append((self.px_ref[i,0], self.px_ref[i,1]))
                if len(self.tracking_stats[i].prev_points) > self.tracking_history_len:
                    self.tracking_stats[i].prev_points.pop(0)

            print("AFTER TRACKING: " + str(self.px_cur.shape[0]))

        keyframe_time_threshold = 0.15
        keyframe_distance_threshold = 2.5

        time_since_last_keyframe = None
        dist_since_last_keyframe = None

        # CHECK IF SHOULD ADD KEYFRAME FOR TRIANGULATION OF POINTS
        # TODO - and prev keyframe parallax condition
        if len(self.keyframes) > 0:
            time_since_last_keyframe = (self.new_img_stamp - self.keyframes[-1].img_timestamp).to_sec()
            p2 = closest_time_odom_msg.pose.pose.position
            p1 = self.keyframes[-1].odom_msg.pose.pose.position
            dist_since_last_keyframe = np.linalg.norm(np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z], dtype=np.float32))

        if self.px_ref is None or ( time_since_last_keyframe > keyframe_time_threshold and dist_since_last_keyframe > keyframe_distance_threshold):
            print("ATTEMPTING TO ADD NEW KEYFRAME! " + str(len(self.keyframes)) + ", dist: " + str(dist_since_last_keyframe) + ", time: " + str(time_since_last_keyframe))

            # NUMPIFY THE CURRENT ODOMETRY TRANSFORMATION MATRIX
            T_odom = self.odom_msg_to_transformation_matrix(closest_time_odom_msg)

            # IF YOU CAN - FIRST TRIANGULATE WITHOUT SCALE! - JUST TO DISCARD OUTLIERS
            can_triangulate = not self.px_cur is None
            if can_triangulate:
                print("TRIANGULATING WITH OWN ")

                # TRIANG BETWEEN PX_CUR AND THEIR PIXEL POSITIONS IN THE PREVIOUS KEYFRAME
                self.px_lkf = np.array([self.tracking_stats[i].prev_keyframe_pixel_pos for i in range(self.px_cur.shape[0])], dtype=np.float32)

                E, mask = cv2.findEssentialMat(self.px_cur, self.px_lkf, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                mask = (mask > 0.5).flatten()
                self.px_cur = self.px_cur[mask, :]
                self.px_lkf = self.px_lkf[mask, :]
                self.tracking_stats = self.tracking_stats[mask]
                print("AFTER OUTLIER REJECTION: " + str(self.px_cur.shape[0]))

                # FIND TRANSFORMATION WITHOUT ODOM PRIOR
                R, t = self.decomp_essential_mat(E, self.px_lkf, self.px_cur)
                self.noprior_triangulation_points = self.triangulated_points
                T_noprior = np.eye(4)
                T_noprior[:3, :3] = R
                T_noprior[:3, 3] = t


                # NOW USE THE TRANSF. MATRIX GIVEN BY OPENVINS ODOMETRY AND USE IT TO TRIANGULATE THE POINTS BETWEEN KEYFRAMES
                T_delta_odom = np.linalg.inv(self.keyframes[-1].T_odom) @ T_odom
                print("T_NOPRIOR:")
                print(T_noprior)
                print("T_delta_ODOM:")
                print(T_delta_odom)

                P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T_delta_odom)

                # Triangulate the 3D points
                hom_Q1 = cv2.triangulatePoints(self.P, P, self.px_lkf.T, self.px_cur.T)
                # Also seen from cam 2
                hom_Q2 = np.matmul(T_delta_odom, hom_Q1)

                # Un-homogenize
                uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
                uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

                # self.odomprior_triangulation_points = uhom_Q2
                self.odomprior_triangulation_points = uhom_Q2

                # USE NOPRIOR PTS AND JUST GET SCALE FROM THE ODOMETRY
                if(not np.any(np.isnan(np.linalg.norm(t)))):
                    scale_modifier = np.linalg.norm(T_delta_odom[:3,3]) / np.linalg.norm(t)
                    print("ORIG TRANSLATION SCALE: "  + str(np.linalg.norm(t)))
                    print("ODOM TRANSLATION SCALE: "  + str(np.linalg.norm(T_delta_odom[:3,3])))

                    print("SCALE MODIFIER: "  + str(scale_modifier))
                    self.noprior_triangulation_points = scale_modifier * self.noprior_triangulation_points 

                    # ADD DEPTH ESTIMATION TO THE POINT
                    for i in range(self.noprior_triangulation_points.shape[1]):
                        MINDEPTH = 0.01
                        depth_meas = np.linalg.norm(self.noprior_triangulation_points[:, i])
                        if depth_meas >= MINDEPTH:
                            invdepth_meas = 1.0 / depth_meas
                            self.tracking_stats[i].invdepth_buffer.append(invdepth_meas)
                            # self.invdepth_meas_sigma2 = 0.01
                            self.invdepth_meas_sigma2 = 0.001
                            self.invdepth_meas_sigma2_init = 0.01
                            if (self.tracking_stats[i].invdepth_measurements) == 0:
                                self.tracking_stats[i].invdepth_mean = invdepth_meas
                                self.tracking_stats[i].invdepth_sigma2 = self.invdepth_meas_sigma2_init
                            else:
                                self.tracking_stats[i].invdepth_mean = (self.tracking_stats[i].invdepth_mean * self.invdepth_meas_sigma2 + invdepth_meas * self.tracking_stats[i].invdepth_sigma2) / (self.invdepth_meas_sigma2 + self.tracking_stats[i].invdepth_sigma2) 
                                self.tracking_stats[i].invdepth_sigma2 = self.invdepth_meas_sigma2 * self.tracking_stats[i].invdepth_sigma2 / (self.invdepth_meas_sigma2 + self.tracking_stats[i].invdepth_sigma2)
                            # print("--MEAS INDEX: " + str(self.tracking_stats[i].invdepth_measurements))
                            # print("MEAS MEAS: " + str(invdepth_meas) )
                            # print("ESTIM MEAN: " + str(self.tracking_stats[i].invdepth_mean) )
                            # print("ESTIM COV: " + str(self.tracking_stats[i].invdepth_sigma2) )
                            avg = np.mean(np.array([x for x in self.tracking_stats[i].invdepth_buffer]))
                            # print("ESTIM AVG: " + str(avg) )
                            self.tracking_stats[i].invdepth_measurements += 1
                            self.proper_triang = True


            # VISUALIZE THE TRIANGULATED SHIT
            if self.proper_triang:
                self.proper_triang = False
                self.visualize_keypoints_in_space(True)
                self.visualize_keypoints_in_space(False)



            # CONTROL FEATURE POPULATION - ADDING AND PRUNING
            self.control_features_population()

            # RETURN IF STILL CANT FIND ANY, NOT ADDING KEYFRAME
            if(self.px_cur is None):
                print("--WARNING! NO FEATURES FOUND!")
                return

            # HAVE ENOUGH POINTS, ADD KEYFRAME
            print("ADDED NEW KEYFRAME! KF: " + str(len(self.keyframes)))
            new_kf = KeyFrame(closest_time_odom_msg, self.new_img_stamp, T_odom)

            # ADD SLAM POINTS INTO THIS KEYFRAME (submap)
            new_kf.slam_points = self.slam_points
            self.slam_points = None

            self.keyframes.append(new_kf)

            # STORE THE PIXELPOSITIONS OF ALL CURRENT POINTS FOR THIS GIVEN KEYFRAME 
            for i in range(self.px_cur.shape[0]):
                self.tracking_stats[i].prev_keyframe_pixel_pos = self.px_cur[i, :]




        # if self.tracking_stats.shape[0] != self.px_cur.shape[0]:
        #     print("FUCK!")

        # if (not self.noprior_triangulation_points is None) and self.tracking_stats.shape[0] != self.noprior_triangulation_points.shape[1]:
        #     print("FUCK2!") # THIS HAPPENS BECAUSE POPULATION CONTROL IS AFTER THE FUCKING THING!

        # VISUALIZE FEATURES
        vis = self.visualize_tracking()
        self.kp_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))
        self.visualize_slampoints_in_space()

        comp_time = time.time() - comp_start_time
        print("computation time: " + str((comp_time) * 1000) +  " ms")# # #}

    def visualize_tracking(self):# # #{
        # rgb = np.zeros((self.new_frame.shape[0], self.new_frame.shape[1], 3), dtype=np.uint8)
        # print(self.new_frame.shape)
        rgb = np.repeat(copy.deepcopy(self.new_frame)[:, :, np.newaxis], 3, axis=2)
        # rgb = np.repeat((self.new_frame)[:, :, np.newaxis], 3, axis=2)

        if not self.px_cur is None:

            ll = np.array([0, 0])  # lower-left
            ur = np.array([self.width, self.height])  # upper-right
            inidx = np.all(np.logical_and(ll <= self.px_cur, self.px_cur <= ur), axis=1)
            inside_pix_idxs = self.px_cur[inidx].astype(int)

            # rgb[inside_pix_idxs[:, 1],inside_pix_idxs[:, 0], 0] = 255
            growsize = 7
            minsize = 4
            # print("SHAPES")
            # print(self.triangulated_points.shape[1])
            # print(self.px_cur.shape[0])

            for i in range(inside_pix_idxs.shape[0]):
                size = self.tracking_stats[inidx][i].age / growsize
                if size > growsize:
                    size = growsize
                size += minsize
                # prevpt = self.tracking_stats[inidx][i].prev_points[-1]
                # rgb = cv2.circle(rgb, (int(prevpt[0]), int(prevpt[1])), int(size), 
                #                (0, 255, 0), 3) 
                rgb = cv2.circle(rgb, (inside_pix_idxs[i,0], inside_pix_idxs[i,1]), int(size), 
                               (255, 0, 255), -1) 
                # rgb = cv2.circle(rgb, (inside_pix_idxs[i,0], inside_pix_idxs[i,1]), 5, 
                #                (255, 0, 255), 2) 

                # triang_pix = self.K.dot(self.triangulated_points[:, inidx][:, i])
                # triang_pix = triang_pix  / triang_pix[2]
                # rgb = cv2.line(rgb, (int(triang_pix[0]), int(triang_pix[1])), (inside_pix_idxs[i,0], inside_pix_idxs[i,1]), (255, 0, 0), 3)
                # rgb = cv2.circle(rgb, (int(triang_pix[0]), int(triang_pix[1])), int(size), 
                #                (0, 0, 255), 3) 

            if not self.noprior_triangulation_points is None:
                for i in range(self.noprior_triangulation_points .shape[1]):
                    pixpos = self.K.dot(self.noprior_triangulation_points [:, i])
                    pixpos = pixpos / pixpos[2]
                    rgb = cv2.circle(rgb, (int(pixpos[0]), int(pixpos[1])), minsize+growsize+2, 
                                   (0, 0, 255), 2) 

            if not self.odomprior_triangulation_points  is None:
                for i in range(self.odomprior_triangulation_points  .shape[1]):
                    pixpos = self.K.dot(self.odomprior_triangulation_points [:, i])
                    pixpos = pixpos / pixpos[2]
                    rgb = cv2.circle(rgb, (int(pixpos[0]), int(pixpos[1])), minsize+growsize+5, 
                                   (0, 255, 255), 2) 

        res = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return res
    
    def publish_pose_msg(self):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.header.frame_id = "mission_origin"
        tf_msg.child_frame_id = "cam_odom"

        # Set the translation 
        tf_msg.transform.translation.x = self.cur_t[0]
        tf_msg.transform.translation.y = self.cur_t[1]
        tf_msg.transform.translation.z = self.cur_t[2]

        # Set the rotation 
        r = Rotation.from_matrix(self.cur_R)
        quat = r.as_quat()
        # quat = pose.rotation().toQuaternion()
        tf_msg.transform.rotation.x = quat[0]
        tf_msg.transform.rotation.y = quat[1]
        tf_msg.transform.rotation.z = quat[2]
        tf_msg.transform.rotation.w = quat[3]


        # Broadcast the TF transform
        self.tf_broadcaster.sendTransformMessage(tf_msg)
# # #}

    def visualize_keypoints(self, img, kp):# # #{
        rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for k in kp:
            rgb[int(k.pt[1]), int(k.pt[0]), 0] = 255
        flow_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return flow_vis
# # #}

    def visualize_slampoints_in_space(self):# # #{
        point_cloud = PointCloud()
        point_cloud.header.stamp = rospy.Time.now()
        point_cloud.header.frame_id = 'global'  # Set the frame ID according to your robot's configuration


        for k in range(len(self.keyframes)):
            kps = self.keyframes[k].slam_points
            if kps is None:
                return
            for i in range(kps.shape[0]):
                # if kps[2, i] > 0:
                point1 = Point32()
                point1.x = kps[i, 0] 
                point1.y = kps[i, 1] 
                point1.z = kps[i, 2] 
                point_cloud.points.append(point1)

        # for i in range(self.slam_points.shape[0]):
        #     # if kps[2, i] > 0:
        #     point1 = Point32()
        #     point1.x = kps[0, i] 
        #     point1.y = kps[1, i] 
        #     point1.z = kps[2, i] 
        #     point_cloud.points.append(point1)

        self.slam_pcl_pub.publish(point_cloud)
# # #}

    def visualize_keypoints_in_space(self, use_invdepth):# # #{
        point_cloud = PointCloud()
        point_cloud.header.stamp = rospy.Time.now()
        # point_cloud.header.frame_id = 'mission_origin'  # Set the frame ID according to your robot's configuration
        point_cloud.header.frame_id = 'cam0'  # Set the frame ID according to your robot's configuration

        # maxnorm = np.max(np.linalg.norm(self.triangulated_points), axis=0)
        # kps = 10 * self.triangulated_points / maxnorm
        # kps = self.odomprior_triangulation_points
        kps = self.noprior_triangulation_points
        if kps is None:
            return

        for i in range(self.tracking_stats.shape[0]):
            # if kps[2, i] > 0:
            scaling_factor = 1
            if use_invdepth and self.tracking_stats[i].invdepth_measurements > 0:
                invdepth = self.tracking_stats[i].invdepth_mean
                if invdepth > 0:
                    scaling_factor = (1/invdepth) / np.linalg.norm(kps[:, i])
            point1 = Point32()
            point1.x = kps[0, i] * scaling_factor
            point1.y = kps[1, i] * scaling_factor
            point1.z = kps[2, i] * scaling_factor
            point_cloud.points.append(point1)

        if use_invdepth:
            self.kp_pcl_pub_invdepth.publish(point_cloud)
        else:
            self.kp_pcl_pub.publish(point_cloud)
# # #}

    def visualize_trajectory(self,points_untransformed , T_vis, start=None, goal=None):# # #{
        marker_array = MarkerArray()
        print(points_untransformed.shape)
        pts = transformPoints(points_untransformed, T_vis)

        marker_id = 0
        if not start is None:
            start = transformPoints(start.reshape((1,3)), T_vis)
            goal = transformPoints(goal.reshape((1,3)), T_vis)

            marker = Marker()
            marker.header.frame_id = "global"  # Change this frame_id if necessary
            marker.header.stamp = rospy.Time.now()
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.id = marker_id
            marker_id += 1

            # Set the position (sphere center)
            marker.pose.position.x = start[0, 0]
            marker.pose.position.y = start[0, 1]
            marker.pose.position.z = start[0, 2]

            marker.scale.x = 2
            marker.scale.y = 2
            marker.scale.z = 2

            marker.color.a = 1
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)

            marker = copy.deepcopy(marker)
            marker.id = marker_id
            marker_id += 1
            marker.pose.position.x = goal[0, 0]
            marker.pose.position.y = goal[0, 1]
            marker.pose.position.z = goal[0, 2]
            marker.color.a = 1
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            # marker_array.markers.append(copy.deepcopy(marker))
            marker_array.markers.append(marker)


        # VISUALIZE START AND GOAL
        if not points_untransformed is None:
            line_marker = Marker()
            line_marker = Marker()
            line_marker.header.frame_id = "global"  # Set your desired frame_id
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.5  # Line width
            line_marker.color.a = 1.0  # Alpha
            line_marker.color.r = 1.0  
            line_marker.color.b = 1.0  

            line_marker.id = marker_id
            marker_id += 1

            for i in range(pts.shape[0]-1):
                point1 = Point()
                point2 = Point()

                p1 = pts[i, :]
                p2 = pts[i+1, :]
                
                point1.x = p1[0]
                point1.y = p1[1]
                point1.z = p1[2]
                point2.x = p2[0]
                point2.y = p2[1]
                point2.z = p2[2]
                line_marker.points.append(point1)
                line_marker.points.append(point2)
            marker_array.markers.append(line_marker)

        self.path_planning_vis_pub .publish(marker_array)
# # #}

    def visualize_keyframe_scores(self, scores, new_kf):# # #{
        marker_array = MarkerArray()

        new_kf_global_pos = transformPoints(new_kf.pos.reshape((1,3)), self.spheremap.T_global_to_own_origin)
        s_min = np.min(scores[:(len(scores)-1)])
        s_max = np.max(scores[:(len(scores)-1)])
        scores = (scores - s_min) / (s_max-s_min)

        marker_id = 0
        for i in range(self.test_db_n_kframes - 1):
            # assuming new_kf is the latest in the scores

            smap_idx, kf_idx = self.test_db_indexing[i]
            kf_pos_in_its_map = None
            T_vis = None
            if smap_idx >= len(self.mchunk.submaps):
                kf_pos_in_its_map = self.spheremap.visual_keyframes[kf_idx].pos
                T_vis = self.spheremap.T_global_to_own_origin
            else:
                # print("IDXS")
                # print(smap_idx)
                # print(kf_idx)
                kf_pos_in_its_map = self.mchunk.submaps[smap_idx].visual_keyframes[kf_idx].pos
                T_vis = self.mchunk.submaps[smap_idx].T_global_to_own_origin
            second_kf_global_pos = transformPoints(kf_pos_in_its_map.reshape((1,3)), T_vis)

            line_marker = Marker()
            line_marker.header.frame_id = "global"  # Set your desired frame_id
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD

            line_marker.scale.x = 1 * scores[i]
            line_marker.color.a = 1.0
            line_marker.color.r = 1.0

            line_marker.id = marker_id
            marker_id += 1

            point1 = Point() 
            point2 = Point()

            p1 = new_kf_global_pos
            p2 = second_kf_global_pos 
            
            point1.x = p1[0, 0]
            point1.y = p1[0, 1]
            point1.z = p1[0, 2]
            point2.x = p2[0, 0]
            point2.y = p2[0, 1]
            point2.z = p2[0, 2]

            line_marker.points.append(point1)
            line_marker.points.append(point2)
            marker_array.markers.append(line_marker)

        self.visual_similarity_vis_pub.publish(marker_array)
        return
    # # #}

    def visualize_episode_submaps(self):# # #{
        marker_array = MarkerArray()

        max_maps_to_vis = 20

        if max_maps_to_vis < len(self.mchunk.submaps):
            max_maps_to_vis = len(self.mchunk.submaps)

        clr_index = 0
        max_clrs = 4
        for i in range(max_maps_to_vis):
            idx = len(self.mchunk.submaps)-(1+i)
            clr_index = idx % max_clrs
            if self.mchunk.submaps[idx].memorized_transform_to_prev_map is None:
                break
            self.get_spheremap_marker_array(marker_array, self.mchunk.submaps[-(i+1)], self.mchunk.submaps[-(i+1)].T_global_to_own_origin, alternative_look = True, do_connections = False, do_surfels = True, do_spheres = False, ms=self.marker_scale, clr_index = clr_index)

        self.recent_submaps_vis_pub.publish(marker_array)

        return
# # #}

    def visualize_spheremap(self):# # #{
        if self.spheremap is None:
            return

        # FIRST CLEAR MARKERS
        # clear_msg = MarkerArray()
        # marker = Marker()
        # marker.id = 0
        # # marker.ns = self.marker_ns
        # marker.action = Marker.DELETEALL
        # clear_msg.markers.append(marker)
        # self.spheremap_spheres_pub.publish(clear_msg)

        marker_array = MarkerArray()

        self.get_spheremap_marker_array(marker_array, self.spheremap, self.spheremap.T_global_to_own_origin, ms=self.marker_scale)

        self.spheremap_spheres_pub.publish(marker_array)
# # #}
    
    def get_spheremap_marker_array(self, marker_array, smap, T_inv, alternative_look=False, do_connections=False,  do_surfels=True, do_spheres=True, do_keyframes=True, do_normals=False, do_map2map_conns=True, ms=1, clr_index =0):# # #{
        # T_vis = np.linalg.inv(T_inv)
        T_vis = T_inv
        pts = transformPoints(smap.points, T_vis)

        marker_id = 0
        if len(marker_array.markers) > 0:
            marker_id = marker_array.markers[-1].id

        if do_map2map_conns and len(smap.map2map_conns) > 0:
            n_conns = len(smap.map2map_conns)

            # CONN PTS
            untr_pts = np.array([c.pt_in_first_map_frame for c in smap.map2map_conns])
            untr_pts = untr_pts.reshape((n_conns, 3))
            spts = transformPoints(untr_pts, T_vis)

            marker = Marker()
            marker.header.frame_id = "global"  # Adjust the frame_id as needed
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
            line_marker.header.frame_id = "global"  # Set your desired frame_id
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
            line_marker.header.frame_id = "global"  # Set your desired frame_id
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
                    # trpt = transformPoints(np.array([[smap.points[i, :]], smap.points[smap.connections[i][j], :]]), T_vis)
                    # p1 = smap.points[i, :]
                    # p2 = smap.points[smap.connections[i][j], :][0]
                    # p1 = trpt[0, :]
                    # p2 = trpt[1, :]

                    p1 = pts[i, :]
                    p2 = pts[smap.connections[i].flatten()[j], :]
                    # print(p1)
                    # print(p2)
                    # print("KURVA")
                    # print(smap.connections[i].flatten())
                    # print(smap.connections[i][j])
                    
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
                marker.header.frame_id = "global"  # Adjust the frame_id as needed
                marker.type = Marker.POINTS
                marker.action = Marker.ADD
                marker.pose.orientation.w = 1.0
                # marker.scale.x = 1.2  # Adjust the size of the points
                # marker.scale.y = 1.2
                marker.scale.x = ms *0.9  # Adjust the size of the points
                marker.scale.y = ms *0.9
                marker.color.a = 1.0
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
                    else:
                        marker.color.r = 1.0
                        marker.color.g = 0.0
                        marker.color.b = 1.0

                marker.id = marker_id
                marker_id += 1


                # Convert the 3D points to Point messages
                n_surfels = spts.shape[0]
                points_msg = [Point(x=spts[i, 0], y=spts[i, 1], z=spts[i, 2]) for i in range(n_surfels)]
                marker.points = points_msg
                marker_array.markers.append(marker)
        if do_normals:
            arrowlen = 0.6 * ms
            pts1 = transformPoints(smap.surfel_points, T_vis)
            pts2 = transformPoints(smap.surfel_points + smap.surfel_normals * arrowlen, T_vis)
            if not smap.surfel_points is None:
                for i in range(smap.surfel_points.shape[0]):
                    marker = Marker()
                    marker.header.frame_id = "global"  # Change this frame_id if necessary
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
                kframes_pts = np.array([kf.pos for kf in smap.visual_keyframes])
                kframes_pts = transformPoints(kframes_pts, T_vis)
                for i in range(n_kframes):
                    marker = Marker()
                    marker.header.frame_id = "global"  # Change this frame_id if necessary
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
                    marker.scale.x = ms *0.8
                    marker.scale.y = ms *2.2
                    marker.scale.z = ms *1.5

                    marker.color.a = 1
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    if alternative_look:
                        marker.color.r = 1.0
                        marker.color.g = 0.0
                        marker.color.b = 1.0

                    arrowlen = 4
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
                marker.header.frame_id = "global"  # Change this frame_id if necessary
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

                marker.color.a = 0.15
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

    def decomp_essential_mat(self, E, q1, q2):# # #{
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
        def sum_z_cal_relative_scale(R, t, store=False):
            # Get the transformation matrix
            T = self._form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            if store:
                return uhom_Q2

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair

        # store the unhomogenized keypoints for the correct pair
        self.triangulated_points = sum_z_cal_relative_scale(R1, t, True)

        t = t * relative_scale

        return [R1, t]
# # #}

    @staticmethod# # #{
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T# # #}

# #}

if __name__ == '__main__':
    rospy.init_node('visual_odom_node')
    optical_flow_node = NavNode()
    rospy.spin()
    cv2.destroyAllWindows()
