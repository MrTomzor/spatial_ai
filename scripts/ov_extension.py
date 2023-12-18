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

# #}
class ScopedLock:
    def __init__(self, mutex):
        # self.lock = threading.Lock()
        self.lock = mutex

    def __enter__(self):
        # print("LOCKING MUTEX")
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # print("UNLOCKING MUTEX")
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
        self.predicted_traj_mutex = threading.Lock()

        # SRV
        self.vocab_srv = rospy.Service("save_vocabulary", EmptySrv, self.saveCurrentVisualDatabaseToVocabFile)
        self.save_episode_full = rospy.Service("save_episode_full", EmptySrv, self.saveEpisodeFull)
        self.return_home_srv = rospy.Service("home", EmptySrv, self.return_home)

        # TIMERS
        self.planning_frequency = 1
        self.planning_timer = rospy.Timer(rospy.Duration(1.0 / self.planning_frequency), self.planning_loop_iter)

        # PLANNING PUB
        self.path_for_trajectory_generator_pub = rospy.Publisher('/uav1/trajectory_generation/path', mrs_msgs.msg.Path, queue_size=10)

        # VIS PUB
        self.slam_points = None
        # self.slam_pcl_pub = rospy.Publisher('extended_slam_points', PointCloud, queue_size=10)

        self.spheremap_outline_pub = rospy.Publisher('spheres', MarkerArray, queue_size=10)
        self.spheremap_freespace_pub = rospy.Publisher('spheremap_freespace', MarkerArray, queue_size=10)

        self.recent_submaps_vis_pub = rospy.Publisher('recent_submaps_vis', MarkerArray, queue_size=10)
        self.path_planning_vis_pub = rospy.Publisher('path_planning_vis', MarkerArray, queue_size=10)
        self.visual_similarity_vis_pub = rospy.Publisher('visual_similarity_vis', MarkerArray, queue_size=10)
        self.unsorted_vis_pub = rospy.Publisher('unsorted_markers', MarkerArray, queue_size=10)

        # self.kp_pub = rospy.Publisher('tracked_features_img', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('estim_depth_img', Image, queue_size=1)
        self.marker_pub = rospy.Publisher('/vo_odom', Marker, queue_size=10)
        # self.kp_pcl_pub = rospy.Publisher('tracked_features_space', PointCloud, queue_size=10)
        # self.kp_pcl_pub_invdepth = rospy.Publisher('tracked_features_space_invdepth', PointCloud, queue_size=10)

        self.tf_listener = tf.TransformListener()

        # --Load calib
        # UNITY
        # self.K = np.array([642.8495341420769, 0, 400, 0, 644.5958939934509, 300, 0, 0, 1]).reshape((3,3))
        # self.imu_to_cam_T = np.array( [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0.0, 0.0, 0.0, 1.0]])
        # self.width = 800
        # self.height = 600
        # ov_slampoints_topic = '/ov_msckf/points_slam'
        # img_topic = '/robot1/camera1/raw'
        # # img_topic = '/robot1/camera1/image'
        # odom_topic = '/ov_msckf/odomimu'
        # self.marker_scale = 1

        # BLUEFOX UAV
        # self.K = np.array([227.4, 0, 376, 0, 227.4, 240, 0, 0, 1]).reshape((3,3))
        # self.P = np.zeros((3,4))
        # self.P[:3, :3] = self.K
        # print(self.P)

        # self.T_imu_to_cam = np.eye(4)
        # self.T_fcu_to_imu = np.eye(4)
        # self.width = 752
        # self.height = 480
        # ov_slampoints_topic = '/ov_msckf/points_slam'
        # img_topic = '/uav1/vio/camera/image_raw'
        # odom_topic = '/ov_msckf/odomimu'
        # self.imu_frame = 'imu'
        # self.fcu_frame = 'uav1/fcu'
        # self.camera_frame = 'cam0'
        # self.odom_frame = 'global'
        # self.marker_scale = 0.5

        # # Get the transform
        # self.tf_listener.waitForTransform(self.fcu_frame, self.imu_frame, rospy.Time(), rospy.Duration(4.0))
        # (trans, rotation) = self.tf_listener.lookupTransform(self.fcu_frame, self.imu_frame, rospy.Time(0))
        # rotation_matrix = tfs.quaternion_matrix(rotation)
        # print(rotation_matrix)
        # self.T_fcu_to_imu[:3, :3] = rotation_matrix[:3,:3]
        # self.T_fcu_to_imu[:3, 3] = trans
        # print("T_fcu_to_imu")
        # print(self.T_fcu_to_imu)

        # self.tf_listener.waitForTransform(self.imu_frame, self.camera_frame, rospy.Time(), rospy.Duration(4.0))
        # (trans, rotation) = self.tf_listener.lookupTransform(self.imu_frame, self.camera_frame, rospy.Time(0))
        # rotation_matrix = tfs.quaternion_matrix(rotation)
        # print(rotation_matrix)
        # self.T_imu_to_cam[:3, :3] = rotation_matrix[:3,:3]
        # self.T_imu_to_cam[:3, 3] = trans
        # print("T_imu_to_cam")
        # print(self.T_imu_to_cam)

        # self.carryover_dist = 8
        # self.uav_radius = 0.6
        # self.safety_replanning_trigger_odist = 0.6
        # self.min_planning_odist = 0.8
        # self.max_planning_odist = 2



        # TELLo (imu = fcu)
        self.K = np.array([933.5640667549508, 0.0, 500.5657553739987, 0.0, 931.5001605952165, 379.0130687255228, 0.0, 0.0, 1.0]).reshape((3,3))

        self.T_imu_to_cam = np.eye(4)
        self.T_fcu_to_imu = np.eye(4)
        self.width = 960
        self.height = 720
        ov_slampoints_topic = 'extended_slam_points'
        img_topic = '/uav1/tellopy_wrapper/rgb/image_raw'
        odom_topic = '/uav1/estimation_manager/odom_main'

        self.imu_frame = 'imu'
        self.fcu_frame = 'uav1/fcu'
        self.odom_frame = 'uav1/passthrough_origin'
        self.camera_frame = "uav1/rgb"

        self.marker_scale = 0.15

        self.tf_listener.waitForTransform(self.fcu_frame, self.camera_frame, rospy.Time(), rospy.Duration(4.0))
        (trans, rotation) = self.tf_listener.lookupTransform(self.fcu_frame, self.camera_frame, rospy.Time(0))
        rotation_matrix = tfs.quaternion_matrix(rotation)
        self.T_imu_to_cam[:3, :3] = rotation_matrix[:3,:3]
        self.T_imu_to_cam[:3, 3] = trans
        # self.T_imu_to_cam = np.linalg.inv(self.T_imu_to_cam)
        print("T_imu(fcu)_to_cam")
        print(self.T_imu_to_cam)

        self.carryover_dist = 4
        self.uav_radius = 0.2
        self.safety_replanning_trigger_odist = 0.2
        self.min_planning_odist = 0.2
        self.max_planning_odist = 2

        

        # --SUB
        self.sub_cam = rospy.Subscriber(img_topic, Image, self.image_callback, queue_size=10000)

        ptraj_topic = '/uav1/control_manager/mpc_tracker/prediction_full_state'
        self.sub_predicted_trajectory = rospy.Subscriber(ptraj_topic, mrs_msgs.msg.MpcPredictionFullState, self.predicted_trajectory_callback, queue_size=10000)

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

        self.global_roadmap = None
        self.global_roadmap_index = None
        self.global_roadmap_len = None
        self.roadmap_start_time = None

        self.local_nav_start_time = rospy.get_rostime()
        self.local_reaching_dist = 3
        self.last_traj_send_time =  rospy.get_rostime()
        self.traj_min_duration = 10

        self.currently_navigating_pts = None
        self.current_goal_vp_global = None
        self.reaching_dist = 0.5
        self.reaching_angle = np.pi/2

        self.max_goal_vp_pathfinding_times = 3
        self.current_goal_vp_pathfinding_times = 0

        # META PARAMS
        self.state = 'explore'
        self.n_sphere_samples_per_update = 100


        self.fspace_bonus_mod = 2
        self.safety_weight = 5
        self.fragmenting_travel_dist = 20
        self.visual_kf_addition_heading = 3.14159 /2
        self.visual_kf_addition_dist = 2

        # ROOMBA PARAMS
        self.roomba_progress_lasttime = None
        self.roomba_dirvec_global = None
        self.roomba_motion_start_global = None
        self.roomba_value_weight = 5
        self.roomba_progress_index = 0
        self.roomba_progress_step = 3
        self.roomba_progress_max_time_per_step = 50
        self.roomba_doubletime = False

        self.roomba_bounds_global = [-20, 20, -30, 30, -10, 20]

        self.verbose_submap_construction = True

        self.predicted_trajectory_pts_global = None

        
        self.node_initialized = True
        # # #}

    # --SERVICE CALLBACKS
    def return_home(self, req):# # #{
        print("RETURN HOME SRV")
        with ScopedLock(self.spheremap_mutex):
            self.state = 'home'
        return EmptySrvResponse()# # #}

    # -- MEMORY MANAGING
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

    # --SUBSCRIBE CALLBACKS

    def lookupTransformAsMatrix(self, frame1, frame2):# # #{
        (trans, rotation) = self.tf_listener.lookupTransform(frame1, frame2, rospy.Time(0)) #Time0 = latest

        rotation_matrix = tfs.quaternion_matrix(rotation)
        res = np.eye(4)
        res[:3, :3] = rotation_matrix[:3,:3]
        res[:3, 3] = trans
        return res# # #}

    def points_slam_callback(self, msg):# # #{
        if self.node_offline:
            return

        if self.verbose_submap_construction:
            print("PCL MSG")
        with ScopedLock(self.spheremap_mutex):
            if self.verbose_submap_construction:
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
                if self.verbose_submap_construction:
                    print("NO POINTS?")
                return
            point_cloud_array = point_cloud_array[nonzero_pts, :]
            print("PCL ARRAY SHAPE:")
            print(point_cloud_array.shape)
# # #}

            # DECIDE WHETHER TO UPDATE SPHEREMAP OR INIT NEW ONE# #{
            T_global_to_imu = self.odom_msg_to_transformation_matrix(self.odom_buffer[-1])
            T_global_to_fcu = T_global_to_imu @ np.linalg.inv(self.T_fcu_to_imu)

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
                        # memorized_transform_to_prev_map = np.linalg.inv(self.spheremap.T_global_to_own_origin) @ T_global_to_imu
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
                # self.spheremap.T_global_to_own_origin = T_global_to_imu @ self.imu_to_cam_T
                print("PESSS")
                # fcu_to_cam0 = self.T_imu_to_cam0 @ self.T_fcu_to_imu
                print(T_global_to_imu)
                # print(T_global_to_imu@ self.imu_to_cam_T)
                print(T_global_to_fcu )
                self.spheremap.T_global_to_own_origin = T_global_to_fcu
                # print(
                # self.spheremap.T_global_to_own_origin = T_global_to_imu 

                # self.spheremap.T_global_to_own_origin = T_global_to_imu
                self.spheremap.creation_time = rospy.get_rostime()

                if not pts_to_transfer is None:
                    self.spheremap.points = pts_to_transfer 
                    self.spheremap.radii = radii_to_transfer 
                    self.spheremap.connections = np.array([None for c in range(pts_to_transfer.shape[0])], dtype=object)
                    self.spheremap.updateConnections(np.arange(0, pts_to_transfer.shape[0]))
                    self.spheremap.labelSpheresByConnectivity()



                if len(self.mchunk.submaps) > 2:
                    self.saveEpisodeFull(None)
            # # #}

            if len(self.odom_buffer) == 0:
                if self.verbose_submap_construction:
                    print("WARN: ODOM BUFFER EMPY!")
                return

            # TRANSFORM SLAM PTS TO -----CAMERA FRAME---- AND COMPUTE THEIR PIXPOSITIONS
            T_global_to_cam = T_global_to_imu @ self.T_imu_to_cam 
            transformed = transformPoints(point_cloud_array, np.linalg.inv(T_global_to_cam)).T
            # transformed = transformPoints(point_cloud_array, (self.imu_to_cam_T @ np.linalg.inv(T_global_to_imu))).T

            positive_z_idxs = transformed[2, :] > 0
            final_points = transformed[:, positive_z_idxs]

            pixpos = self.getPixelPositions(final_points)

            # COMPUTE DELAUNAY TRIANG OF VISIBLE SLAM POINTS
            if final_points.shape[0] < 3:
                if self.verbose_submap_construction:
                    print("NOT ENAUGH PTS FOR DELAUNAY!")
                return
            if self.verbose_submap_construction:
                print("HAVE DELAUNAY:")
            tri = Delaunay(pixpos)

            # print("INPUT: " )
            # print(point_cloud_array)
            # print("TRANSFORMED TO CAM FRAME: " )
            # print(final_points.T)

            vis = self.visualize_depth(pixpos, tri)
            self.depth_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))

            # CONSTRUCT OBSTACLE MESH
            comp_mesh = time.time()
            obstacle_mesh = trimesh.Trimesh(vertices=final_points.T, faces = tri.simplices)

            # CONSTRUCT POLYGON OF PIXPOSs OF VISIBLE SLAM PTS
            hull = ConvexHull(pixpos)

            if self.verbose_submap_construction:
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

            if self.verbose_submap_construction:
                print("MESHING time: " + str((comp_time) * 1000) +  " ms")


            # ---UPDATE AND PRUNING STEP
            T_orig_to_current_cam = np.eye(4)
            T_delta_odom  = np.eye(4)
            if not self.spheremap is None: # TODO remove
                # transform existing sphere points to current camera frame
                # T_spheremap_orig_to_current_cam = self.imu_to_cam_T @ T_global_to_imu @ np.linalg.inv(self.spheremap.T_global_to_own_origin) 
                # T_orig_to_current_cam = np.linalg.inv(T_spheremap_orig_to_current_cam)
                # T_spheremap_orig_to_current_cam = np.linalg.inv(self.imu_to_cam_T)
                # print("T DELTA ODOM")
                # print(T_delta_odom)
                # print("GLOBAL TO OWN ORIGIN")
                # print(self.spheremap.T_global_to_own_origin)
                # print("ORIG TO NOW")
                # print(T_spheremap_orig_to_current_cam )
                n_spheres_old = self.spheremap.points.shape[0]

                # T_delta_odom = np.linalg.inv(self.spheremap.T_global_to_own_origin) @ T_global_to_imu
                T_delta_odom = np.linalg.inv(self.spheremap.T_global_to_own_origin) @ T_global_to_fcu

                # T_orig_to_current_cam = T_delta_odom @ np.linalg.inv(self.imu_to_cam_T)
                T_orig_to_current_cam = ( T_delta_odom @ self.T_fcu_to_imu @ self.T_imu_to_cam)

                # project sphere points to current camera frame
                transformed_old_points  = transformPoints(self.spheremap.points, np.linalg.inv(T_orig_to_current_cam))

                # Filter out spheres with z below zero or above the max z of obstacle points
                # TODO - use dist rather than z for checking
                max_vis_z = np.max(final_points[2, :])
                z_ok_idxs = np.logical_and(transformed_old_points[:, 2] > 0, transformed_old_points[:, 2] <= max_vis_z)

                z_ok_points = transformed_old_points[z_ok_idxs , :] # remove spheres with negative z
                worked_sphere_idxs = np.arange(n_spheres_old)[z_ok_idxs ]
                # print("POSITIVE Z AND NOT BEHIND OBSTACLE MESH:" + str(np.sum(z_ok_idxs )) + "/" + str(n_spheres_old))

                # check the ones that are projected into the 2D hull
                old_pixpos = self.getPixelPositions(z_ok_points.T)
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
            max_sphere_sampling_z = 10

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

            visible_pts_in_spheremap_frame = transformPoints(final_points.T, T_orig_to_current_cam)
            self.spheremap.updateSurfels(visible_pts_in_spheremap_frame , pixpos, tri.simplices)

            comp_time = time.time() - comp_start_time
            if self.verbose_submap_construction:
                print("SURFELS integration time: " + str((comp_time) * 1000) +  " ms")

            # VISUALIZE CURRENT SPHERES
            self.visualize_spheremap()
            comp_time = time.time() - comp_start_time
            if self.verbose_submap_construction:
                print("SPHEREMAP visualization time: " + str((comp_time) * 1000) +  " ms")

            # TRY PATHFINDING TO START OF CURRENT SPHEREMAP
            # pathfindres = self.findPathAstarInSubmap(self.spheremap, T_delta_odom[:3, 3], np.array([0,0,0]))

            # VISUALIZE PATH HOME
            # if len(self.mchunk.submaps) > 0:
            #     startpos = T_current_cam_to_orig[:3,3].reshape((1,3))
            #     endpos = np.zeros((1,3))
            #     goal_map_id = 0
            #     plan_res = self.findGlobalPath(Viewpoint(startpos), Viewpoint(endpos), goal_map_id)
            #     self.visualize_roadmap(plan_res, transformPoints(startpos,self.spheremap.T_global_to_own_origin), transformPoints(endpos, self.mchunk.submaps[goal_map_id].T_global_to_own_origin))

            # self.node_offline = not self.spheremap.consistencyCheck()
# # #}

    def odometry_callback(self, msg):# # #{
        self.odom_buffer.append(msg)
        if len(self.odom_buffer) > self.odom_buffer_maxlen:
            self.odom_buffer.pop(0)# # #}

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

        # print("FRAME " + str(self.n_frames))

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

        # TODO - FIX TRANSFORMS

        # GET ODOM MSG CLOSEST TO CURRENT IMG TIMESTAMP
        # closest_time_odom_msg = self.get_closest_time_odom_msg(self.new_img_stamp)
        # T_odom = self.odom_msg_to_transformation_matrix(closest_time_odom_msg)
        # T_odom_relative_to_smap_start = np.linalg.inv(self.spheremap.T_global_to_own_origin) @ T_odom

        latest_odom_msg  = self.get_closest_time_odom_msg(self.new_img_stamp)
        T_global_to_imu = self.odom_msg_to_transformation_matrix(latest_odom_msg)
        T_global_to_fcu = T_global_to_imu @ np.linalg.inv(self.T_fcu_to_imu)
        T_fcu_relative_to_smap_start  = np.linalg.inv(self.spheremap.T_global_to_own_origin) @ T_global_to_fcu
#         T_fcu_relative_to_smap_start = np.linalg.inv(self.spheremap.T_global_to_own_origin) @ T_odom

        new_kf = SubmapKeyframe(T_fcu_relative_to_smap_start)


        # CHECK IF NEW ENOUGH
        # TODO - check in near certainly connected submaps
        for kf in self.spheremap.visual_keyframes:
            # TODO - scaling
            if kf.euclid_dist(new_kf) < self.visual_kf_addition_dist and kf.heading_dif(new_kf) < self.visual_kf_addition_heading:
                # print("KFS: not novel enough, N keyframes: " + str(len(self.spheremap.visual_keyframes)))
                return

        if len(self.spheremap.visual_keyframes) > 0:
            dist_bonus = new_kf.euclid_dist(self.spheremap.visual_keyframes[-1])
            heading_bonus = new_kf.heading_dif(self.spheremap.visual_keyframes[-1]) * 0.2

            self.spheremap.traveled_context_distance += dist_bonus + heading_bonus

        print("KFS: adding new visual keyframe!")
        self.spheremap.visual_keyframes.append(new_kf)
        return


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
        print("KFS: kf addition time: " + str((comp_time) * 1000) +  " ms")

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

    def predicted_trajectory_callback(self, msg):# # #{
        with ScopedLock(self.predicted_traj_mutex):
            # PARSE THE MSG
            pts = np.array([[pt.x, pt.y, pt.z] for pt in msg.position])
            headings = np.array([h for h in msg.heading])
            msg_frame = msg.header.frame_id

            # GET CURRENT ODOM MSG
            # latest_odom_msg = self.odom_buffer[-1]
            # T_global_to_imu = self.odom_msg_to_transformation_matrix(latest_odom_msg)
            # T_global_to_fcu = T_global_to_imu @ np.linalg.inv(self.T_fcu_to_imu)


            T_msg_to_fcu = self.lookupTransformAsMatrix(msg_frame, self.fcu_frame)
            T_fcu_to_global = self.lookupTransformAsMatrix(self.fcu_frame, 'global')

            T_msg_to_global = T_msg_to_fcu @ T_fcu_to_global

            pts_global, headings_global = transformViewpoints(pts, headings, np.linalg.inv(T_msg_to_global))
            # print("PTS GLOBAL: ")
            # print(pts_global)

            # print("PTS FCU: ")
            pts_fcu, headings_fcu = transformViewpoints(pts_global, headings_global, T_fcu_to_global)
            # print(pts_fcu)
            # print(headings_fcu)

            self.predicted_trajectory_stamps = msg.stamps
            self.predicted_trajectory_pts_global = pts_global
            self.predicted_trajectory_headings_global = headings_global

    # # #}

    # --PLANNING PIPELINE

    def get_future_predicted_trajectory_global_frame(self):# # #{
        now = rospy.get_rostime()

        stamps = None
        pts = None
        headings = None

        if self.predicted_trajectory_pts_global is None:
            return None, None, None

        with ScopedLock(self.predicted_traj_mutex):
            stamps = self.predicted_trajectory_stamps
            pts = self.predicted_trajectory_pts_global
            headings = self.predicted_trajectory_headings_global

        n_pred_pts = pts.shape[0]

        first_future_index = -1
        for i in range(n_pred_pts):
            if (stamps[i] - now).to_sec() > 0:
                first_future_index = i
                break
        if first_future_index == -1:
            print("WARN! - predicted trajectory is pretty old!")
            return None, None, None

        pts_global = pts[first_future_index:, :]
        headings_global = headings[first_future_index:]
        relative_times = np.array([(stamps[i] - now).to_sec() for i in range(first_future_index, n_pred_pts)])

        return pts_global, headings_global, relative_times

    # # #}

    def find_unsafe_pt_idxs_on_global_path(self, pts_global, headings_global, min_odist, clearing_dist = -1, other_submap_idxs = None):# # #{
        n_pts = pts_global.shape[0]
        res = []

        pts_in_smap_frame = transformPoints(pts_global, np.linalg.inv(self.spheremap.T_global_to_own_origin))
        dists_from_start = np.linalg.norm(pts_global - pts_global[0, :], axis=1)

        odists = []
        for i in range(n_pts):
            odist_fs = self.spheremap.getMaxDistToFreespaceEdge(pts_in_smap_frame[i, :]) * self.fspace_bonus_mod
            odist_surf = self.spheremap.getMinDistToSurfaces(pts_in_smap_frame[i, :])
            odist = min(odist_fs, odist_surf)

            odists.append(odist)
            if odist < min_odist and dists_from_start[i] > clearing_dist:
                res.append(i)
        odists = np.array(odists)
        # print("ODISTS RANKED:")
        # print("ODISTS:")
        # # odists = np.sort(odists)
        # print(odists)
        return np.array(res), odists# # #}
        
    def planning_loop_iter(self, event=None):# # #{
        # print("pes")
        with ScopedLock(self.spheremap_mutex):
            # self.dump_forward_flight_astar_iter()
            self.dumb_forward_flight_rrt_iter()
    # # #}

    def postproc_path(self, pos_in_smap_frame, headings_in_smap_frame):# # #{
        # MAKE IT SO THAT NOT TOO MANY PTS CLOSE TOGETHER!!! unless close to obstacles!
        # TODO ALLOW NO HEADING CHANGE IF STILL MOVING IN FOV!
        keep_idxs = []
        n_pts = pos_in_smap_frame.shape[0]

        # prev_pos = start_vp.positon
        # prev_heading = start_vp.heading

        prev_pos = pos_in_smap_frame[0, :]

        min_travers_dist = 0.7

        for i in range(n_pts):
            dist = np.linalg.norm(prev_pos - pos_in_smap_frame[i,:])
            if dist > min_travers_dist:
                keep_idxs.append(i)
                prev_pos = pos_in_smap_frame[i,:]

        print("postprocessed path to " + str(len(keep_idxs)) + " / " + str(n_pts))
        if len(keep_idxs) == 0:
            return None, None

        keep_idxs = np.array(keep_idxs)
        return pos_in_smap_frame[keep_idxs, :], headings_in_smap_frame[keep_idxs]


    # # #}

    def dumb_forward_flight_rrt_iter(self):# # #{
        # print("--DUMB FORWARD FLIGHT RRT ITER")
        return
        if self.spheremap is None:
            return
        if not self.node_initialized:
            return

        # GET CURRENT POS IN ODOM FRAME
        latest_odom_msg = self.odom_buffer[-1]
        T_global_to_imu = self.odom_msg_to_transformation_matrix(latest_odom_msg)
        T_global_to_fcu = T_global_to_imu @ np.linalg.inv(self.T_fcu_to_imu)
        T_smap_origin_to_fcu = np.linalg.inv(self.spheremap.T_global_to_own_origin) @ T_global_to_fcu

        pos_fcu_in_global_frame = T_global_to_fcu[:3, 3]
        heading_odom_frame = transformationMatrixToHeading(T_global_to_imu)

        # GET START VP IN SMAP FRAME
        T_smap_frame_to_fcu = np.linalg.inv(self.spheremap.T_global_to_own_origin) @ T_global_to_imu @ np.linalg.inv(self.T_fcu_to_imu)
        heading_in_smap_frame = transformationMatrixToHeading(T_smap_frame_to_fcu)
        planning_start_vp = Viewpoint(T_smap_frame_to_fcu[:3, 3], heading_in_smap_frame)

        planning_time = 0.5
        current_maxodist = self.max_planning_odist
        current_minodist = self.min_planning_odist

        # CHECK ROOMBA DIR
        roombatime =  self.roomba_progress_max_time_per_step*2 if self.roomba_doubletime else self.roomba_progress_max_time_per_step
        if self.roomba_progress_lasttime is None or (rospy.get_rostime() - self.roomba_progress_lasttime).to_sec() > roombatime:
            print("--RRR------CHANGING ROOMBA DIR!")
            theta = np.random.rand(1) * 2 * np.pi
            dirvec2 = np.array([np.cos(theta), np.sin(theta)]).reshape((1,2))

            if not self.roomba_dirvec_global is None:
                # FIND VERY DIFFERENT DIR!
                while dirvec2.dot(self.roomba_dirvec_global[0,:2]) > 0.1:
                    print("RAND")
                    theta = np.random.rand(1) * 2 * np.pi
                    dirvec2 = np.array([np.cos(theta), np.sin(theta)]).reshape((1,2))

            self.roomba_dirvec_global = np.zeros((1,3))
            self.roomba_dirvec_global[0, :2] = dirvec2 
            print(self.roomba_dirvec_global)

            self.roomba_progress_lasttime = rospy.get_rostime()
            self.roomba_motion_start_global = pos_fcu_in_global_frame

            # if self.roomba_progress_index == 0:
            #     print("didnt reach far, giving this dir 2x the time!")
            #     self.roomba_doubletime = True
            # else:
            #     self.roomba_doubletime = False
            self.roomba_progress_index = 0
        else:
            # Update roomba progress
            current_progress = np.linalg.norm(pos_fcu_in_global_frame - self.roomba_motion_start_global)
            current_index = current_progress // self.roomba_progress_step
            if current_index > self.roomba_progress_index:
                print("--RRR------ROOMBA PROGRESS TO INDEX " + str(current_index))
                self.roomba_progress_index = current_index 
                self.roomba_progress_lasttime = rospy.get_rostime()
            # TODO - replanning totally trigger

        arrow_pos = copy.deepcopy(pos_fcu_in_global_frame)
        arrow_pos[2] += 10
        self.visualize_arrow(arrow_pos.flatten(), (arrow_pos + self.roomba_dirvec_global*5).flatten(), r=1, g=0.5, marker_idx=0)

        if not self.current_goal_vp_global is None:
            # VISUALZIE GOAL
            arrow_pos = self.current_goal_vp_global.position.flatten()
            ghd = self.current_goal_vp_global.heading
            arrow_pos2 = arrow_pos + np.array([np.cos(ghd), np.sin(ghd), 0]) * 2
            self.visualize_arrow(arrow_pos, arrow_pos2, r=0.7, g=0.7, marker_idx=2)

            # CHECK IF REACHED
            dist_to_goal = np.linalg.norm(pos_fcu_in_global_frame - self.current_goal_vp_global.position)
            # print("DIST FROM GOAL:")
            # print(dist_to_goal)
            if dist_to_goal < self.reaching_dist:
                print("-------- GOAL REACHED! ----------------")
                self.current_goal_vp_global = None


        evading_obstacle = False

        # IF NAVIGATING ALONG PATH, AND ALL IS OK, DO NOT REPLAN. IF NOT PROGRESSING OR REACHED END - CONTINUE FURTHER IN PLANNIGN NEW PATH!
        if not self.currently_navigating_pts is None:
            path_reaching_dist = self.reaching_dist
            dists_from_path_pts = np.linalg.norm(self.currently_navigating_pts - pos_fcu_in_global_frame, axis = 1)
            # print("DISTS FROM PATH")
            # print(dists_from_path_pts)
            closest_path_pt_idx = np.argmin(dists_from_path_pts)
            if closest_path_pt_idx > self.currently_navigating_reached_node_idx:
                print("MOVED BY PTS: " + str(closest_path_pt_idx - self.currently_navigating_reached_node_idx))
                self.currently_navigating_reached_node_idx = closest_path_pt_idx 
                self.trajectory_following_moved_time = rospy.get_rostime()
            if closest_path_pt_idx == self.currently_navigating_pts.shape[0] - 1:
                print("REACHED GOAL OF CURRENTLY SENT PATH!")
                self.currently_navigating_pts = None
            else:
                if (rospy.get_rostime() - self.trajectory_following_moved_time).to_sec() > 8:
                    print("NOT PROGRESSING ON PATH FOR LONG TIME! THROWING AWAY PATH!")
                    self.currently_navigating_pts = None
                    return
                else:
                    print("PROGRESSING ALONG PATH - PROGRESS: "+ str(self.currently_navigating_reached_node_idx) + "/" + str(self.currently_navigating_pts.shape[0]) )

                    # CHECK SAFETY OF TRAJECTORY!
                    # GET RELEVANT PTS IN PREDICTED TRAJ!
                    future_pts, future_headings, relative_times = self.get_future_predicted_trajectory_global_frame()
                    if not future_pts is None:
                        future_pts_dists = np.linalg.norm(future_pts - pos_fcu_in_global_frame, axis=1)
                        unsafe_idxs, odists = self.find_unsafe_pt_idxs_on_global_path(future_pts, future_headings, self.safety_replanning_trigger_odist, 0)
                        if unsafe_idxs.size > 0:

                            time_to_unsafety = relative_times[unsafe_idxs[0]]
                            triggering_time = 2

                            if time_to_unsafety < triggering_time:
                                print("-----TRIGGERING UNSAFETY REPLANNING!!!")

                                print("FOUND UNSAFE PT IN PREDICTED TRAJECTORY AT INDEX:" + str(unsafe_idxs[0]) + " DIST: " + str(future_pts_dists[unsafe_idxs[0]]))
                                print("ODISTS UP TO PT:")
                                print(odists[:unsafe_idxs[0]+1])
                                print("DISTS UP TO PT:")
                                print(future_pts_dists[:unsafe_idxs[0]+1])
                                print("FUTURE PTS:")
                                print(future_pts)
                                print("FCU POS:")
                                print(pos_fcu_in_global_frame)
                                print("TIME TO UNSAFETY: " + str(time_to_unsafety) +" / " + str(triggering_time))

                                planning_time = time_to_unsafety * 0.5
                                print("PLANNING TIME: " +str(planning_time))

                                # VISUALIZE OFFENDING PT
                                arrow_pos2 = future_pts[unsafe_idxs[0], :]
                                arrow_pos = copy.deepcopy(arrow_pos2)
                                arrow_pos[2] -= 10
                                self.visualize_arrow(arrow_pos.flatten(), arrow_pos2.flatten(), r=0.5, scale=0.5, marker_idx=1)

                                # if time_to_unsafety < planning_time

                                # print("SOME PT IS UNSAFE! DISCARDING PLAN AND PLANNING FROM CURRENT VP!")
                                # self.currently_navigating_pts = None

                                after_planning_time = relative_times > planning_time + 0.2
                                if not np.any(after_planning_time):
                                    print("PLANNING TIME IS LONGER THAN PREDICTED TRAJ! SHORTENING IT AND PLANNING FROM CURRENT POS IN SMAP FRAME!")
                                    planning_time *= 0.5
                                else:
                                    start_idx = np.where(after_planning_time)[0][0]
                                    print("PLANNING FROM VP OF INDEX " + str(start_idx) + " AT DIST " + str(future_pts_dists[start_idx]))
                                    startpos_smap_frame, startheading_smap_frame = transformViewpoints(future_pts[start_idx, :].reshape((1,3)), np.array([future_headings[start_idx]]), np.linalg.inv(self.spheremap.T_global_to_own_origin))
                                    planning_start_vp = Viewpoint(startpos_smap_frame, startheading_smap_frame[0])
                                    self.currently_navigating_pts = None
                                evading_obstacle = True
                            # TODO - replaning to goal trigger. If failed -> replanning to any other goal. If failed -> evasive maneuvers n wait!
                            else:
                                return
                        else:
                            return
                    else:
                        return


        # FIND PATHS WITH RRT
        best_path_pts = None
        best_path_headings = None
        save_last_pos_as_goal = False

        if not self.current_goal_vp_global is None:

            # FIND PATH TO GOAL
            goal_vp_smap_pos, goal_vp_smap_heading = transformViewpoints(self.current_goal_vp_global.position.reshape((1,3)), np.array([self.current_goal_vp_global.heading]), np.linalg.inv(self.spheremap.T_global_to_own_origin))
            best_path_pts, best_path_headings = self.find_paths_rrt(planning_start_vp , max_comp_time = planning_time, min_odist = current_minodist, max_odist = current_maxodist, mode = 'to_goal', goal_vp_smap = Viewpoint(goal_vp_smap_pos, goal_vp_smap_heading))

            if best_path_pts is None:
                print("NO PATH FOUND TO GOAL VP! TRY: " + str(self.current_goal_vp_pathfinding_times) + "/" + str(self.max_goal_vp_pathfinding_times))
                self.current_goal_vp_pathfinding_times += 1
                if self.current_goal_vp_pathfinding_times > self.max_goal_vp_pathfinding_times:
                    print("GOAL VP UNREACHABLE TOO MANY TIMES, DISCARDING IT!")
                    self.current_goal_vp_global = None
                return

        # elif not evading_obstacle:
        else:
            best_path_pts, best_path_headings = self.find_paths_rrt(planning_start_vp , max_comp_time = planning_time, min_odist = current_minodist, max_odist = current_maxodist)

            if best_path_pts is None:
                print("NO OK VIABLE REACHABLE GOALS FOUND BY RRT!")
                return

            print("FOUND SOME REACHABLE GOAL, SETTING IT AS GOAL AND SENDING PATH TO IT!")
            save_last_pos_as_goal = True

        min_path_len = 1
        if np.linalg.norm(best_path_pts[-1, :] - planning_start_vp.position) < min_path_len:
            print("FOUND BEST PATH TOO SHORT!")
            return

        # GET FCU TRANSFORM NOWW! AFTER 0.5s OF PLANNING
        latest_odom_msg = self.odom_buffer[-1]
        T_global_to_imu = self.odom_msg_to_transformation_matrix(latest_odom_msg)
        T_global_to_fcu = T_global_to_imu @ np.linalg.inv(self.T_fcu_to_imu)
        T_smap_origin_to_fcu = np.linalg.inv(self.spheremap.T_global_to_own_origin) @ T_global_to_fcu

        pts_fcu, headings_fcu = transformViewpoints(best_path_pts, best_path_headings, np.linalg.inv(T_smap_origin_to_fcu))
        pts_global, headings_global = transformViewpoints(best_path_pts, best_path_headings, self.spheremap.T_global_to_own_origin)

        if save_last_pos_as_goal:
            self.current_goal_vp_global = Viewpoint(pts_global[-1, :], headings_global[-1])
            self.current_goal_vp_pathfinding_times = 0

        print("PTS IN FCU FRAME (SHOULD START NEAR ZERO!):")
        print(pts_fcu)
        print("HEADINGS IN GLOBAL FRAME (SHOULD START NEAR ZERO!):") # GLOBAL IS ROTATED BY 180 DEG FROM VIO ORIGIN!
        print(headings_fcu)

        # SEND IT AND SET FLAGS
        # if headings_fcu.size > 1:
        #     self.send_path_to_trajectory_generator(pts_fcu[1:, :], headings_fcu[1:])
        # else:
        #     self.send_path_to_trajectory_generator(pts_fcu, headings_fcu)
        self.send_path_to_trajectory_generator(pts_fcu, headings_fcu)

        self.currently_navigating_pts = pts_global
        self.currently_navigating_headings = headings_global
        self.trajectory_following_moved_time = rospy.get_rostime()
        self.currently_navigating_reached_node_idx = -1

        self.visualize_trajectory(pts_global, np.eye(4), headings_global, do_line = False, frame_id = "global")

        return
    # # #}

    def find_paths_rrt(self, start_vp, visualize = True, max_comp_time=0.5, max_step_size = 0.5, max_spread_dist=15, min_odist = 0.1, max_odist = 0.5, max_iter = 700006969420, goal_vp_smap=None, p_greedy = 0.3, mode='find_goals'):# # #{
        # print("RRT: STARTING, FROM TO:")
        # print(start_vp)

        bounds = np.ones((1,3)) * max_spread_dist
        # bounds += 10

        epicenter = start_vp.position 

        max_conn_size = max_step_size * 1.05

        comp_start_time = rospy.get_rostime()
        n_iters = 0
        n_unsafe = 0
        n_rewirings = 0

        # INIT TREE
        n_nodes = 1
        tree_pos = start_vp.position.reshape((1, 3))
        tree_headings = np.array([start_vp.heading])
        odists = np.array([self.spheremap.getMaxDistToFreespaceEdge(start_vp.position)])
        parent_indices = np.array([-1])
        # child_indices = np.array([-1])
        child_indices = [[]]
        total_costs = np.array([0])

        # so always can move!
        if odists < min_odist:
            odist = min_odist

        while True:
            time_left = max_comp_time - (rospy.get_rostime() - comp_start_time).to_sec()
            if time_left <= 0 or n_iters >= max_iter:
                break
            n_iters += 1

            # SAMPLE PT IN SPACE
            sampling_goal_pt = (np.random.rand(1, 3)*2 - np.ones((1,3))) * bounds * 0.6  + epicenter
            if not (goal_vp_smap is None) and np.random.rand() < p_greedy:
                sampling_goal_pt = goal_vp_smap.position

            # FIND NEAREST POINT TO THE SAMPLING DIRPOINT
            dists = np.array(np.linalg.norm(sampling_goal_pt - tree_pos, axis=1)).flatten()
            nearest_index = np.argmin(dists)

            new_node_pos = None
            if dists[nearest_index] < max_step_size:
                new_node_pos = sampling_goal_pt 
            else:
                new_node_pos = tree_pos[nearest_index, :] + (sampling_goal_pt - tree_pos[nearest_index, :]) * (max_step_size / dists[nearest_index])
                # print("DIST FROM NEAREST TREE NODE:" + str(np.linalg.norm(new_node_pos - tree_pos[nearest_index, :])))
                dists = np.linalg.norm(new_node_pos - tree_pos, axis=1).flatten()
                nearest_index = np.argmin(dists)

            # CHECK IF POINT IS SAFE
            new_node_pos = new_node_pos.reshape((1,3))

            # IF IN SPHERE
            new_node_fspace_dist = self.spheremap.getMaxDistToFreespaceEdge(new_node_pos) * self.fspace_bonus_mod 
            if new_node_fspace_dist < min_odist:
                n_unsafe += 1
                continue

            # new_node_odist = self.spheremap.getMaxDistToFreespaceEdge(new_node_pos)
            new_node_odist = self.spheremap.getMinDistToSurfaces(new_node_pos)
            if new_node_odist < min_odist:
                n_unsafe += 1
                continue
            new_node_odist = min(new_node_odist, new_node_fspace_dist)

            # FIND ALL CONNECTABLE PTS
            connectable_mask = dists < max_conn_size
            n_connectable = np.sum(connectable_mask)
            if n_connectable == 0:
                print("WARN!!! no connectable nodes!!")
                print("MIN DIST: " + str(np.min(dists)))
            connectable_indices = np.where(connectable_mask)[0]

            # COMPUTE COST TO MOVE FROM NEAREST NODE TO THIS
            # travelcosts = dists[connectable_mask]
            dirvecs = (new_node_pos - tree_pos[connectable_mask, :])
            potential_headings = np.arctan2(dirvecs[:, 1], dirvecs[:,0])
            # potential_headings = np.arctan2(-dirvecs[:, 0], dirvecs[:,2])

            # for i in range(dirvecs.shape[0]):
            #     print("dirvec:")
            #     print(dirvecs[i,:])
            #     print("HEADING: " + str(potential_headings[i]))

            heading_difs = np.abs(np.unwrap(potential_headings - tree_headings[connectable_mask]))
            # safety_costs = np.array([self.compute_safety_cost(odists, dists[idx]) for idx in connectable_indices]).flatten() * self.safety_weight
            safety_costs = self.compute_safety_cost(new_node_odist, min_odist, max_odist) * dists[connectable_mask] * self.safety_weight
            travelcosts = dists[connectable_mask] + heading_difs * 0.0 + safety_costs

            potential_new_total_costs = total_costs[connectable_mask] + travelcosts
            new_node_parent_idx2 = np.argmin(potential_new_total_costs)

            newnode_total_cost = potential_new_total_costs[new_node_parent_idx2]
            newnode_heading = potential_headings[new_node_parent_idx2]
            new_node_parent_idx = connectable_indices[new_node_parent_idx2]

            # ADD NODE TO TREE
            tree_pos = np.concatenate((tree_pos, new_node_pos))
            tree_headings = np.concatenate((tree_headings, np.array([newnode_heading]) ))
            odists = np.concatenate((odists, np.array([new_node_odist]) ))
            total_costs = np.concatenate((total_costs, np.array([newnode_total_cost]) ))
            parent_indices = np.concatenate((parent_indices, np.array([new_node_parent_idx]) ))
            # child_indices[new_node_parent_idx] = n_nodes
            child_indices[new_node_parent_idx].append(n_nodes)
            # child_indices = np.concatenate((child_indices, np.array([-1]) ))
            child_indices.append([])
            n_nodes += 1

            # CHECK IF CAN REWIRE
            do_rewiring = False
            if do_rewiring and n_connectable > 1:
                # connectable_mask[new_node_parent_idx] = False
                # connectable_indices = np.where(connectable_mask)
                # print("N CONNECT: " + str(n_connectable))
                # print(travelcosts.shape)

                new_total_costs = newnode_total_cost + travelcosts
                difs_from_old_cost = new_total_costs - (total_costs[:(n_nodes-1)])[connectable_mask] 

                rewiring_mask = difs_from_old_cost < 0 
                rewiring_mask[new_node_parent_idx2] = False

                n_to_rewire = np.sum(rewiring_mask)
                if n_to_rewire > 0:
                    n_rewirings += 1
                    rewiring_idxs = connectable_indices[rewiring_mask]

                    for i in range(n_to_rewire):
                        node_to_rewire = rewiring_idxs[i]
                        prev_parent = parent_indices[rewiring_idxs[i]]

                        print("REWIRING!")
                        print(node_to_rewire)
                        print(child_indices[prev_parent])
                        # CHANGE PARENT OF OLD NODE TO NEWLY ADDED
                        child_indices[prev_parent].remove(node_to_rewire)
                        parent_indices[node_to_rewire] = n_nodes - 1
                        child_indices[n_nodes-1].append(node_to_rewire)

                        # SPREAD CHANGE IN COST TO ITS CHILDREN
                        openset = [rewiring_idxs[i]]
                        discount = -difs_from_old_cost[i]
                        while len(openset) > 0:
                            print("OPENSET:")
                            print(node_to_rewire)
                            print(openset)
                            expnd = openset.pop()
                            total_costs[expnd] -= discount
                            openset = openset + child_indices[expnd]



        print("SPREAD FINISHED, HAVE " + str(tree_pos.shape[0]) + " NODES. ITERS: " +str(n_iters) + " UNSAFE: " + str(n_unsafe) + " REWIRINGS: " + str(n_rewirings))
        if visualize:
            self.visualize_rrt_tree(self.spheremap.T_global_to_own_origin, tree_pos, None, odists, parent_indices)

        # FIND ALL HEADS, EVALUATE THEM
        comp_start_time = rospy.get_rostime()
        heads_indices = np.array([i for i in range(tree_pos.shape[0]) if len(child_indices[i]) == 0])
        print("N HEADS: " + str(heads_indices.size))

        # GET GLOBAL AND FCU FRAME POINTS
        heads_global = transformPoints(tree_pos[heads_indices, :], self.spheremap.T_global_to_own_origin)

        latest_odom_msg = self.odom_buffer[-1]
        T_global_to_imu = self.odom_msg_to_transformation_matrix(latest_odom_msg)
        T_global_to_fcu = T_global_to_imu @ np.linalg.inv(self.T_fcu_to_imu)

        heads_values = None
        acceptance_thresh = -1000
        heads_scores = np.full((1, heads_indices.size), acceptance_thresh-1).flatten()
        heads_values = np.full((1, heads_indices.size), acceptance_thresh-1).flatten()

        if mode == 'find_goals':
            acceptance_thresh = 0
            toofar = np.logical_not(check_points_in_box(heads_global, self.roomba_bounds_global))
            if np.all(toofar):
                print("ALL PTS OUTSIDE OF ROOMBA BBX! SETTING NOVELTY BASED ON DISTANCE IN DIRECTION TO CENTER")
                current_fcu_in_global = T_global_to_fcu[:3, 3].reshape((1,3))
                dirvecs = heads_global - current_fcu_in_global
                dir_to_center = - current_fcu_in_global / np.linalg.norm(current_fcu_in_global)
                dists_towards_center = (dirvecs @ dir_to_center.T).flatten()

                heads_values = self.roomba_value_weight * dists_towards_center
            else:
                current_fcu_in_global = T_global_to_fcu[:3, 3].reshape((1,3))
                dirvecs = heads_global - current_fcu_in_global
                roombadir = self.roomba_dirvec_global 
                dists_in_dir = (dirvecs @ roombadir.T).flatten()

                heads_values = self.roomba_value_weight * dists_in_dir
                heads_values[toofar] = acceptance_thresh-1

            heads_scores = heads_values - total_costs[heads_indices] 
        elif mode == 'to_goal':
            dists_from_goal = np.linalg.norm(tree_pos[heads_indices, :] - goal_vp_smap.position, axis = 1)
            # heading_difs_from_goal = tree_headings[heads_indices] - goal_vp.heading
            reaching_goal = dists_from_goal <= self.reaching_dist
            print("N REACHING GOAL: " + str(np.sum(reaching_goal)))
            print(heads_scores.shape)
            print(reaching_goal.shape)
            print(heads_indices.shape)

            heads_values[reaching_goal] = acceptance_thresh + 1
            heads_scores[reaching_goal] = -total_costs[heads_indices[reaching_goal]]

        best_head = np.argmax(heads_scores)
        print("BEST HEAD: " + str(best_head))
        best_node_index = heads_indices[best_head]
        print("-----BEST NODE VALUE (mode:"+mode+ ") : " +str(heads_values[best_head]))
        print("-----BEST NODE COST: " +str(total_costs[best_node_index]))

        if heads_values[best_head] <= acceptance_thresh:
            print("-----BEST HEAD IS BELOW ACCEPTANCE THRESHOLD, RETURNING NO VIABLE PATHS!!!")
            return None, None


        # RECONSTRUCT PATH FROM THERE
        path_idxs = [best_node_index]
        parent = parent_indices[best_node_index]

        dvec = tree_pos[best_node_index, :] - tree_pos[0, :]
        tree_headings[best_node_index] = start_vp.heading

        while parent >= 0:
            path_idxs.append(parent)
            parent = parent_indices[parent]

        print("RETURNING OF LEN: " + str(len(path_idxs)))
        path_idxs.reverse()
        path_idxs = np.array(path_idxs)

        # return tree_pos[path_idxs, :], tree_headings[path_idxs]
        pp_pts, pp_headings = self.postproc_path(tree_pos[path_idxs, :], tree_headings[path_idxs]) #WILL ALWAYS DISCARD ORIGIN!
        return pp_pts, pp_headings

    # # #}

    def set_global_roadmap(rm):# # #{
        self.global_roadmap = rm
        self.global_roadmap_index = 0
        self.global_roadmap_len = rm.shape[0]
        self.roadmap_start_time = rospy.Time.now()
    # # #}

    def send_path_to_trajectory_generator(self, path, headings=None):# # #{
        print("SENDING PATH TO TRAJ GENERATOR, LEN: " + str(len(path)))
        msg = mrs_msgs.msg.Path()
        msg.header = std_msgs.msg.Header()
        # msg.header.frame_id = 'global'
        # msg.header.frame_id = 'uav1/vio_origin' #FCU!!!!
        msg.header.frame_id = 'uav1/fcu'
        # msg.header.frame_id = 'uav1/local_origin'
        msg.header.stamp = rospy.Time.now()
        msg.use_heading = not headings is None
        # msg.use_heading = False
        msg.fly_now = True
        msg.stop_at_waypoints = False
        arr = []

        n_pts = path.shape[0]
        for i in range(n_pts):
        # for p in path:
            p = path[i, :]
            ref = mrs_msgs.msg.Reference()
            xx = p[0]
            yy = p[1]
            zz = p[2]
            ref.position = Point(x=xx, y=yy, z=zz)
            if not headings is None:
                ref.heading = headings[i]
                # ref.heading = 0
                # ref.heading = np.arctan2(yy, xx)
                # ref.heading = ((headings[i]+ np.pi) + np.pi) % (2 * np.pi) - np.pi
            msg.points.append(ref)
        # print("ARR:")
        # print(arr)
        # msg.points = arr

        self.path_for_trajectory_generator_pub.publish(msg)
        print("SENT PATH TO TRAJ GENERATOR, N PTS: " + str(n_pts))

        return None
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

    def compute_safety_cost(self, node2_radius, dist):# # #{
        if node2_radius < self.min_planning_odist:
            node2_radius = self.min_planning_odist
        if node2_radius > self.max_planning_odist:
            return 0
        saf = (node2_radius - self.min_planning_odist) / (self.max_planning_odist  - self.min_planning_odist)
        return saf * saf * dist
    # # #}

    def compute_safety_cost(self, node2_radius, minodist, maxodist):# # #{
        if node2_radius < minodist:
            node2_radius = minodist
        if node2_radius > maxodist:
            return 0
        saf = (node2_radius - minodist) / (maxodist  - minodist)
        return saf * saf
    # # #}


    # --VISUALIZATIONS
    def visualize_arrow(self, pos, endpos, frame_id='global', r=1,g=0,b=0, scale=1,marker_idx=0):
        marker_array = MarkerArray()

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.id = marker_idx

        # Set the scale
        marker.scale.x = scale *0.4
        marker.scale.y = scale *2.0
        marker.scale.z = scale *1.0

        marker.color.a = 1
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b

        points_msg = [Point(x=pos[0], y=pos[1], z=pos[2]), Point(x=endpos[0], y=endpos[1], z=endpos[2])]
        marker.points = points_msg

        # Add the marker to the MarkerArray
        marker_array.markers.append(marker)
        self.unsorted_vis_pub.publish(marker_array)

    def visualize_rrt_tree(self, T_vis, tree_pos, tree_headings, odists, parent_indices):# # #{
        marker_array = MarkerArray()
        pts = transformPoints(tree_pos, T_vis)

        marker_id = 0
        if not tree_pos is None:
            line_marker = Marker()
            line_marker.header.frame_id = "global"  # Set your desired frame_id
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.04  # Line width
            line_marker.color.a = 1.0  # Alpha
            line_marker.color.r = 0.5  
            line_marker.color.b = 1.0  

            line_marker.id = marker_id
            marker_id += 1
            line_marker.ns = "rrt"

            for i in range(1, pts.shape[0]):
                point1 = Point()
                point2 = Point()

                p1 = pts[i, :]
                # p2 = pts[i+1, :]
                p2 = pts[parent_indices[i], :]
                
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

    def visualize_roadmap(self,pts, start=None, goal=None, reached_idx = None):# # #{
        marker_array = MarkerArray()
        # pts = transformPoints(points_untransformed, T_vis)

        marker_id = 0
        if not start is None:
            # start = transformPoints(start.reshape((1,3)), T_vis)
            # goal = transformPoints(goal.reshape((1,3)), T_vis)

            marker = Marker()
            marker.header.frame_id = "global"  # Change this frame_id if necessary
            marker.ns = "roadmap"
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
            marker.scale.z = 4

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
        if not pts is None:
            line_marker = Marker()
            
            line_marker.header.frame_id = "global"  # Set your desired frame_id
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.5  # Line width
            line_marker.color.a = 1.0  # Alpha
            line_marker.color.r = 1.0  
            line_marker.color.b = 1.0  

            line_marker.id = marker_id
            line_marker.ns = "roadmap"
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

    def visualize_trajectory(self,points_untransformed, T_vis, headings=None,do_line=True, start=None, goal=None, frame_id="global"):# # #{
        marker_array = MarkerArray()
        print(points_untransformed.shape)
        pts = transformPoints(points_untransformed, T_vis)

        marker_id = 0
        if not start is None:
            start = transformPoints(start.reshape((1,3)), T_vis)
            goal = transformPoints(goal.reshape((1,3)), T_vis)

            marker = Marker()
            marker.header.frame_id = frame_id  # Change this frame_id if necessary
            marker.header.stamp = rospy.Time.now()
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.id = marker_id
            marker.ns = "path"
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
        if do_line and not points_untransformed is None:
            line_marker = Marker()
            line_marker = Marker()
            line_marker.header.frame_id = frame_id    # Set your desired frame_id
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.1  # Line width
            line_marker.color.a = 1.0  # Alpha
            line_marker.color.r = 0.0  
            line_marker.color.b = 0.0  

            line_marker.id = marker_id
            marker_id += 1
            line_marker.ns = "path"

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

        if not headings is None:
           for i in range(pts.shape[0]):
                marker = Marker()
                marker.header.frame_id = frame_id   # Change this frame_id if necessary
                marker.header.stamp = rospy.Time.now()
                marker.type = Marker.ARROW
                marker.action = Marker.ADD
                marker.id = marker_id
                marker.ns = "path"
                marker_id += 1

                # Set the scale
                marker.scale.x = self.marker_scale *0.5
                marker.scale.y = self.marker_scale *1.2
                marker.scale.z = self.marker_scale *0.8

                marker.color.a = 1
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 0.0

                arrowlen = 1
                map_heading = transformationMatrixToHeading(T_vis)
                xbonus = arrowlen * np.cos(headings[i] + map_heading)
                ybonus = arrowlen * np.sin(headings[i] + map_heading)
                points_msg = [Point(x=pts[i][0], y=pts[i][1], z=pts[i][2]), Point(x=pts[i][0]+xbonus, y=pts[i][1]+ybonus, z=pts[i][2])]
                marker.points = points_msg
                marker_array.markers.append(marker)

        self.path_planning_vis_pub .publish(marker_array)
# # #}

    def visualize_keyframe_scores(self, scores, new_kf):# # #{
        marker_array = MarkerArray()

        new_kf_global_pos = transformPoints(new_kf.position.reshape((1,3)), self.spheremap.T_global_to_own_origin)
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
            # if self.mchunk.submaps[idx].memorized_transform_to_prev_map is None:
            #     break
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

        marker_array = MarkerArray()
        self.get_spheremap_marker_array(marker_array, self.spheremap, self.spheremap.T_global_to_own_origin, ms=self.marker_scale, do_spheres=False, do_surfels=True)
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

    def get_spheremap_marker_array(self, marker_array, smap, T_inv, alternative_look=False, do_connections=False,  do_surfels=True, do_spheres=True, do_keyframes=False, do_normals=False, do_map2map_conns=True, ms=1, clr_index =0):# # #{
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

    # --UTILS
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
        

# #}

if __name__ == '__main__':
    rospy.init_node('spheremap_mapper_node')
    optical_flow_node = NavNode()
    rospy.spin()
    cv2.destroyAllWindows()
