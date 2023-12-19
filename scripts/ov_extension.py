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

        # PLANNING PUB
        self.path_for_trajectory_generator_pub = rospy.Publisher('/uav1/trajectory_generation/path', mrs_msgs.msg.Path, queue_size=10)

        # VIS PUB
        self.slam_points = None

        self.path_planning_vis_pub = rospy.Publisher('path_planning_vis', MarkerArray, queue_size=10)
        self.visual_similarity_vis_pub = rospy.Publisher('visual_similarity_vis', MarkerArray, queue_size=10)
        self.unsorted_vis_pub = rospy.Publisher('unsorted_markers', MarkerArray, queue_size=10)

        self.marker_pub = rospy.Publisher('/vo_odom', Marker, queue_size=10)

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
        self.submap_builder_input_point_ids = None

        self.submap_builder_module = SubmapBuilderModule(self.width, self.height, self.K, self.camera_frame, self.odom_frame,self.fcu_frame, self.tf_listener, self.T_imu_to_cam, self.T_fcu_to_imu)
        self.submap_builder_rate = 10
        self.submap_builder_timer = rospy.Timer(rospy.Duration(1.0 / self.submap_builder_rate), self.submap_builder_update_callback)

        # --SUB
        self.sub_cam = rospy.Subscriber(img_topic, Image, self.image_callback, queue_size=10000)

        ptraj_topic = '/uav1/control_manager/mpc_tracker/prediction_full_state'
        self.sub_predicted_trajectory = rospy.Subscriber(ptraj_topic, mrs_msgs.msg.MpcPredictionFullState, self.predicted_trajectory_callback, queue_size=10000)

        self.odom_buffer = []
        self.odom_buffer_maxlen = 1000
        self.sub_odom = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback, queue_size=10000)

        # self.sub_slam_points = rospy.Subscriber(ov_slampoints_topic, PointCloud2, self.points_slam_callback, queue_size=3)

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


        # ADD NEW VISUAL KEYFRAME IF NEW ENOUGH
        if self.spheremap is None:
            return

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


        comp_time = time.time() - comp_start_time
        print("KFS: kf addition time: " + str((comp_time) * 1000) +  " ms")


        # # #}

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

    def submap_builder_update_callback(self, event=None):# # #{

        # copy deepcopy the input data, its not that big! then can leave mutex!! (so rest of img callback is not held up)
        pcl_msg = None
        points_info = None

        if self.submap_builder_input_pcl is None:
            print("NO PCL FOR UPDATE YET!")
            return

        with ScopedLock(self.submap_builder_input_mutex):
            pcl_msg = copy.deepcopy(self.submap_builder_input_pcl)
            points_info = copy.deepcopy(self.submap_builder_input_point_ids)

        self.submap_builder_module.camera_update_iter(pcl_msg, points_info) 
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

        self.visualize_trajectory(pts_global, np.eye(4), headings_global, do_line = False, frame_id = self.odom_frame)

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
    def visualize_arrow(self, pos, endpos, frame_id=None, r=1,g=0,b=0, scale=1,marker_idx=0):# # #{
        marker_array = MarkerArray()
        if frame_id is None:
            frame_id = self.odom_frame

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
        self.unsorted_vis_pub.publish(marker_array)# # #}

    def visualize_rrt_tree(self, T_vis, tree_pos, tree_headings, odists, parent_indices):# # #{
        marker_array = MarkerArray()
        pts = transformPoints(tree_pos, T_vis)

        marker_id = 0
        if not tree_pos is None:
            line_marker = Marker()
            line_marker.header.frame_id = self.odom_frame  # Set your desired frame_id
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

    def visualize_roadmap(self,pts, start=None, goal=None, reached_idx = None):# # #{
        marker_array = MarkerArray()
        # pts = transformPoints(points_untransformed, T_vis)

        marker_id = 0
        if not start is None:
            # start = transformPoints(start.reshape((1,3)), T_vis)
            # goal = transformPoints(goal.reshape((1,3)), T_vis)

            marker = Marker()
            marker.header.frame_id = self.odom_frame  # Change this frame_id if necessary
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
            
            line_marker.header.frame_id = self.odom_frame  # Set your desired frame_id
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

    def visualize_trajectory(self,points_untransformed, T_vis, headings=None,do_line=True, start=None, goal=None, frame_id=None):# # #{
        if frame_id is None:
            frame_id = self.odom_frame
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
    rospy.init_node('spheremap_mapper_node')
    optical_flow_node = NavNode()
    rospy.spin()
    cv2.destroyAllWindows()
