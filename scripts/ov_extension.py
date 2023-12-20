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

from spatial_ai.common_spatial import *
from spatial_ai.fire_slam_module import *
from spatial_ai.submap_builder_module import *
from spatial_ai.local_navigator_module import *

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
        self.save_episode_full = rospy.Service("save_episode_full", EmptySrv, self.saveEpisodeFull)
        self.return_home_srv = rospy.Service("home", EmptySrv, self.return_home)

        # TIMERS
        self.planning_frequency = 1
        self.planning_timer = rospy.Timer(rospy.Duration(1.0 / self.planning_frequency), self.planning_loop_iter)

        # VIS PUB
        self.slam_points = None

        self.path_planning_vis_pub = rospy.Publisher('path_planning_vis', MarkerArray, queue_size=10)
        self.visual_similarity_vis_pub = rospy.Publisher('visual_similarity_vis', MarkerArray, queue_size=10)
        self.unsorted_vis_pub = rospy.Publisher('unsorted_markers', MarkerArray, queue_size=10)

        self.marker_pub = rospy.Publisher('/vo_odom', Marker, queue_size=10)

        self.tf_listener = tf.TransformListener()

        # --Load calib
        # UNITY
        self.K = np.array([642.8495341420769, 0, 400, 0, 644.5958939934509, 300, 0, 0, 1]).reshape((3,3))
        self.T_imu_to_cam = np.eye(4)
        self.T_fcu_to_imu = np.eye(4)
        self.width = 800
        self.height = 600
        # ov_slampoints_topic = '/ov_msckf/points_slam'
        ov_slampoints_topic = 'extended_slam_points'
        img_topic = '/robot1/camera1/raw'
        # img_topic = '/robot1/camera1/image'
        odom_topic = '/ov_msckf/odomimu'
        self.imu_frame = 'imu'
        # self.fcu_frame = 'uav1/fcu'
        self.fcu_frame = 'imu'
        self.camera_frame = 'cam0'
        self.odom_frame = 'global'
        self.marker_scale = 1
        self.slam_kf_dist_thr = 4
        self.smap_fragmentation_dist = 3000

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
        # self.K = np.array([933.5640667549508, 0.0, 500.5657553739987, 0.0, 931.5001605952165, 379.0130687255228, 0.0, 0.0, 1.0]).reshape((3,3))

        # self.T_imu_to_cam = np.eye(4)
        # self.T_fcu_to_imu = np.eye(4)
        # self.width = 960
        # self.height = 720
        # ov_slampoints_topic = 'extended_slam_points'
        # img_topic = '/uav1/tellopy_wrapper/rgb/image_raw'
        # odom_topic = '/uav1/estimation_manager/odom_main'

        # self.imu_frame = 'imu'
        # self.fcu_frame = 'uav1/fcu'
        # self.odom_frame = 'uav1/passthrough_origin'
        # self.camera_frame = "uav1/rgb"

        # self.marker_scale = 0.15
        # self.slam_kf_dist_thr = 0.5
        # self.smap_fragmentation_dist = 10


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

        # --INITIALIZE MODULES

        # FIRESLAM
        self.fire_slam_module = FireSLAMModule(self.width, self.height, self.K, self.camera_frame, self.odom_frame, self.tf_listener)
        self.fire_slam_module.kf_dist_thr = self.slam_kf_dist_thr
        self.fire_slam_module.marker_scale = self.marker_scale

        # SUBMAP BUILDER
        self.submap_builder_input_mutex = threading.Lock()
        self.submap_builder_input_pcl = None
        self.submap_builder_input_point_ids = None

        self.submap_builder_module = SubmapBuilderModule(self.width, self.height, self.K, self.camera_frame, self.odom_frame,self.fcu_frame, self.tf_listener, self.T_imu_to_cam, self.T_fcu_to_imu)
        self.submap_builder_module.marker_scale = self.marker_scale
        self.submap_builder_module.fragmenting_travel_dist = self.smap_fragmentation_dist

        self.submap_builder_rate = 10
        self.submap_builder_timer = rospy.Timer(rospy.Duration(1.0 / self.submap_builder_rate), self.submap_builder_update_iter)

        # LOCAL NAVIGATOR
        ptraj_topic = '/uav1/control_manager/mpc_tracker/prediction_full_state'
        output_path_topic = '/uav1/trajectory_generation/path'
        self.local_navigator_module = LocalNavigatorModule(self.submap_builder_module, ptraj_topic, output_path_topic)

        # --SUB
        self.sub_cam = rospy.Subscriber(img_topic, Image, self.image_callback, queue_size=10000)

        self.odom_buffer = []
        self.odom_buffer_maxlen = 1000
        self.sub_odom = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback, queue_size=10000)

        self.tf_broadcaster = tf.TransformBroadcaster()

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
        with ScopedLock(self.spheremap_mutex):
            self.submap_builder_module.saveEpisodeFull(None)
        return EmptySrvResponse()# # #}

    # --SUBSCRIBE CALLBACKS

    def planning_loop_iter(self, event):# # #{
        print("PLANNING ITER")
        if not self.node_initialized:
            return
        with ScopedLock(self.spheremap_mutex):
            self.local_navigator_module.planning_loop_iter()
    # # #}

    def lookupTransformAsMatrix(self, frame1, frame2):# # #{
        return lookupTransformAsMatrix(frame1, frame2, self.tf_listener)
    # # #}

    def odometry_callback(self, msg):# # #{
        self.odom_buffer.append(msg)
        if len(self.odom_buffer) > self.odom_buffer_maxlen:
            self.odom_buffer.pop(0)# # #}

    def image_callback(self, msg):# # #{
        if self.node_offline:
            return

        # UPDATE VISUAL SLAM MODULE, PASS INPUT TO SUBMAP BUILDER IF NEW INPUT
        self.fire_slam_module.image_callback(msg)

        if self.fire_slam_module.has_new_pcl_data:
            pcl_msg, point_ids = self.fire_slam_module.get_visible_pointcloud_metric_estimate(visualize=True)
            # print("PCL MSG TYPE:")
            # print(pcl_msg)
            with ScopedLock(self.submap_builder_input_mutex):
                self.submap_builder_input_pcl = pcl_msg
                self.submap_builder_input_point_ids = point_ids


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



        # # #}

    def submap_builder_update_iter(self, event=None):# # #{

        # copy deepcopy the input data, its not that big! then can leave mutex!! (so rest of img callback is not held up)
        pcl_msg = None
        points_info = None

        if self.submap_builder_input_pcl is None:
            print("NO PCL FOR UPDATE YET!")
            return

        with ScopedLock(self.submap_builder_input_mutex):
            pcl_msg = copy.deepcopy(self.submap_builder_input_pcl)
            points_info = copy.deepcopy(self.submap_builder_input_point_ids)

        with ScopedLock(self.spheremap_mutex):
            self.submap_builder_module.camera_update_iter(pcl_msg, points_info) 
    # # #}

    # --VISUALIZATIONS
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

    # --UTILS

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

# #}

if __name__ == '__main__':
    rospy.init_node('ultra_navigation_node')
    optical_flow_node = NavNode()
    rospy.spin()
    cv2.destroyAllWindows()
