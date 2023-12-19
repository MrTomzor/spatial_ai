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

class LocalNavigatorModule:
    def __init__(self):# # #{

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

        # INITIALIZE MODULES

        self.fire_slam_module = FireSLAMModule(self.width, self.height, self.K, self.camera_frame, self.odom_frame, self.tf_listener)

        self.submap_builder_input_mutex = threading.Lock()
        self.submap_builder_input_pcl = None
        self.submap_builder_input_point_info = None

        # self.submap_builder_module = None #TODO
        # self.submap_builder_rate = 10
        # self.submap_builder_timer = rospy.Timer(rospy.Duration(1.0 / self.submap_builder_rate), self.submap_builder_update_callback)

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

