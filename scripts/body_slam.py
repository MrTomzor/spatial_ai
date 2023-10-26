#!/usr/bin/env python

import rospy
# from sensor_msgs.msg import Image
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
import sensor_msgs.point_cloud2 as pc2
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


STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 200
kMaxNumFeature = 2000

def transformPoints(pts, T):
    # pts = Nx3 matrix, T = transformation matrix to apply
    res = np.concatenate((pts.T, np.full((1, pts.shape[0]), 1)))
    res = T @ res 
    res = res / res[3, :] # unhomogenize
    return res[:3, :].T

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

class Tracked2DPoint:
    def __init__(self, pos, keyframe_id):
        self.keyframe_observations = {keyframe_id: pos}
        self.current_pos = pos
        self.age = 1

        self.last_observed_keyframe_id = keyframe_id
        self.body_index = None
        self.active = True

    def addObservation(self, pt, keyframe_id):
        # self.last_observed_keyframe_id = np.max(keyframe_id, self.last_observed_keyframe_id)
        # self.last_observed_keyframe_id = np.max(keyframe_id, self.last_observed_keyframe_id)
        if self.last_observed_keyframe_id < keyframe_id:
            self.last_observed_keyframe_id = keyframe_id
        # self.keyframe_observations[keyframe_id] = [u,v]
        self.keyframe_observations[keyframe_id] = pt

    def getAge(self):
        return len(self.keyframe_observations)

class KeyFrame:
    def __init__(self, img_timestamp):
        self.triangulated_points = []
        self.img_timestamp = img_timestamp

class OdomNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.prev_image = None
        self.prev_time = None
        self.proper_triang = False

        self.spheremap = None
        self.keyframes = []
        self.keyframe_idx = 0

        self.tracked_2d_points = {}
        self.next_2d_point_id = 0


        self.noprior_triangulation_points = None
        self.odomprior_triangulation_points = None

        self.active_2d_points_ids = []

        self.slam_points = None
        self.slam_pcl_pub = rospy.Publisher('extended_slam_points', PointCloud, queue_size=10)

        self.kp_pub = rospy.Publisher('tracked_features_img', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('estim_depth_img', Image, queue_size=1)
        self.marker_pub = rospy.Publisher('/vo_odom', Marker, queue_size=10)
        self.kp_pcl_pub = rospy.Publisher('tracked_features_space', PointCloud, queue_size=10)
        self.kp_pcl_pub_invdepth = rospy.Publisher('tracked_features_space_invdepth', PointCloud, queue_size=10)

        self.sub_cam = rospy.Subscriber('/robot1/camera1/raw', Image, self.image_callback, queue_size=10000)

        self.odom_buffer = []
        self.odom_buffer_maxlen = 1000
        self.sub_odom = rospy.Subscriber('/ov_msckf/odomimu', Odometry, self.odometry_callback, queue_size=10000)
        # self.sub_slam_points = rospy.Subscriber('/ov_msckf/points_slam', PointCloud2, self.points_slam_callback, queue_size=3)
        # self.sub_odom = rospy.Subscriber('/ov_msckf/poseimu', PoseWithCovarianceStamped, self.odometry_callback, queue_size=10000)

        self.tf_broadcaster = tf.TransformBroadcaster()

        self.orb = cv2.ORB_create(nfeatures=3000)

        # Load calib
        self.K = np.array([642.8495341420769, 0, 400, 0, 644.5958939934509, 300, 0, 0, 1]).reshape((3,3))
        # self.K = np.array([, 644.5958939934509, 400.0503960299562, 300.5824096896595]).reshape((3,3))
        self.imu_to_cam_T = np.array( [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0.0, 0.0, 0.0, 1.0]])
        print("IMUTOCAM", self.imu_to_cam_T)

        # self.K = np.array([642.8495341420769, 644.5958939934509, 400.0503960299562, 300.5824096896595]).reshape((3,3))
        self.P = np.zeros((3,4))
        self.P[:3, :3] = self.K
        print(self.P)

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
        self.width = 800
        self.height = 600
        self.tracking_bin_width = 100
        self.min_features_per_bin = 1
        self.max_features_per_bin = 2
        self.tracking_history_len = 4
        self.node_offline = False
        self.last_tried_landmarks_pxs = None

        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.detector = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)

        self.tracking_colors = np.random.randint(0, 255, (100, 3)) 

        self.n_frames = 0

    def getPixelPositions(self, pts):
        # pts = 3D points u wish to project
        pixpos = self.K @ pts 
        pixpos = pixpos / pixpos[2, :]
        return pixpos[:2, :].T

    def odometry_callback(self, msg):
        self.odom_buffer.append(msg)
        if len(self.odom_buffer) > self.odom_buffer_maxlen:
            self.odom_buffer.pop(0)

    def visualize_depth(self, pixpos, tri):
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

    def insertNew2DPoint(self, pixpos):
        return

    def control_features_population(self):
        wbins = self.width // self.tracking_bin_width
        hbins = self.height // self.tracking_bin_width
        found_total = 0

        # FIND PTS IN PT DICT THAT WERE SEEN IN PREVIOUS FRAME (NOT LOST SINCE PREV KEYFRAME)
        active_ids  = [pt_id for pt_id, pt in self.tracked_2d_points.items() if pt.active]

        active_pix = None

        if len(active_ids ) == 0:
            new_px = self.detector.detect(self.new_frame)
            if new_px is None or len(new_px) == 0:
                return
            n_new_px = len(new_px)
            print("N FOUND IN BEGINNING: " + str(n_new_px))

            active_pix = np.array([x.pt for x in new_px], dtype=np.float32)

            # ADD NEWLY DETECTED POINTS TO THE DICT FOR TRACKING
            new_ids = range(self.next_2d_point_id, self.next_2d_point_id + n_new_px)
            self.next_2d_point_id += n_new_px
            active_ids = new_ids
            for i in range(n_new_px):
                pt_object = Tracked2DPoint(new_px[i].pt, self.keyframe_idx)
                # self.tracked_2d_points.insert(new_ids[i], pt_object) 
                self.tracked_2d_points[new_ids[i]] = pt_object
        else:
            active_pix  = np.array([self.tracked_2d_points[pt_id].current_pos for pt_id in active_ids], dtype=np.float32)

        # if self.px_cur is None or len(self.px_cur) == 0:
        #     self.px_cur = self.detector.detect(self.new_frame)
        #     if self.px_cur is None or len(self.px_cur) == 0:
        #         return
        #     self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
        #     self.tracking_stats = np.array([TrackingStat() for x in self.px_cur], dtype=object)

        active_ids = np.array(active_ids)

        new_ids = []
        new_px = []
        deletion_ids = []

        n_culled = 0

        # FIND THE IDXS OF THE ACTIV IDS WHICH TO DELETE, AND ACCUMULATE NEW POINT POSITIONS TO INIT
        for xx in range(wbins):
            for yy in range(hbins):
                # count how many we have there and get the points in there:
                ul = np.array([xx * self.tracking_bin_width , yy * self.tracking_bin_width])  
                lr = np.array([ul[0] + self.tracking_bin_width , ul[1] + self.tracking_bin_width]) 

                inidx = np.all(np.logical_and(ul <= active_pix, active_pix <= lr), axis=1)
                # print(inidx)
                inside_points = []
                inside_ids = []

                n_existing_in_bin = 0
                if np.any(inidx):
                    inside_points = active_pix[inidx]
                    inside_ids = active_ids[inidx]
                    n_existing_in_bin = inside_points.shape[0]

                if n_existing_in_bin > self.max_features_per_bin:
                    # CUTOFF POINTS ABOVE MAXIMUM, SORTED BY AGE
                    ages = np.array([-self.tracked_2d_points[pt_id].age for pt_id in inside_ids])

                    idxs = np.argsort(ages)
                    surviving_idxs = idxs[:self.max_features_per_bin]
                    n_culled_this_bin = n_existing_in_bin - self.max_features_per_bin

                    # self.px_cur = np.concatenate((self.px_cur, inside_points[surviving_idxs , :]))
                    # self.tracking_stats = np.concatenate((self.tracking_stats, inside_stats[surviving_idxs ]))

                    deletion_ids = deletion_ids + [inside_ids[i] for i in idxs[(self.max_features_per_bin-1):]]

                    # TODO - LET THE ONES WITH MANY DEPTH MEASUREMENTS LIVE

                elif n_existing_in_bin < self.min_features_per_bin:
                    # ADD THE EXISTING
                    # if n_existing_in_bin > 0:
                    #     self.px_cur = np.concatenate((self.px_cur, inside_points))
                    #     self.tracking_stats = np.concatenate((self.tracking_stats, inside_stats))

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
                    # locally_found[:, 0] += ul[0]
                    # locally_found[:, 1] += ul[1]
                    # self.px_cur = np.concatenate((self.px_cur, locally_found))
                    new_px = new_px + [locally_found]
                else:
                    # JUST COPY THEM
                    # self.px_cur = np.concatenate((self.px_cur, inside_points))
                    # self.tracking_stats = np.concatenate((self.tracking_stats, inside_stats))
                    continue

        # DELETION = DEACTIVATION OF POINT IN DICT
        # nondel = np.full( (1, len(active_ids)), True)
        # nondel[deletion_ids] =  False
        # nondel_i

        # NONDEACTIVATED POINTS ... ARENT CHANGED AT ALL! OH JUST INCREASED AGE (NUMBER OF IMGS LIVED)
        for px_id in active_ids:
            self.tracked_2d_points[px_id].age += 1
        for del_id in deletion_ids:
            # self.tracked_2d_points[active_ids[del_idx]].age -= 1
            # self.tracked_2d_points[active_ids[del_idx]].active = False
            self.tracked_2d_points[del_id].age -= 1
            self.tracked_2d_points[del_id].active = False


        print("DEACTIVATED: " + str(len(deletion_ids)))

        n_added = 0
        for batch in new_px:
            n_pts = batch.shape[0]
            n_added += n_pts
            # print(batch)

            new_ids = range(self.next_2d_point_id, self.next_2d_point_id + n_pts)
            self.next_2d_point_id += n_pts
            for i in range(n_pts):
                pt_object = Tracked2DPoint(batch[i, :], self.keyframe_idx)
                self.tracked_2d_points[new_ids[i]] = pt_object

        print("ADDED: " + str(n_added))

        # TODO - add the found points to dict!!!

        # print("CURRENT FEATURES: " + str(self.px_cur.shape[0]))

        # # FIND FEATS IF ZERO!
        # if(self.px_cur.shape[0] == 0):
        #     print("ZERO FEATURES! FINDING FEATURES")
        #     self.px_cur = self.detector.detect(self.new_frame)
        #     self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)

        
    def odom_msg_to_transformation_matrix(self, odom_msg):
        odom_p = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z])
        odom_q = np.array([odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w])
        T_odom = np.eye(4)
        # print("R:")
        # print(Rotation.from_quat(odom_q).as_matrix())
        T_odom[:3,:3] = Rotation.from_quat(odom_q).as_matrix()
        T_odom[:3, 3] = odom_p
        return T_odom

    def visualBatchOptimization(self, kf_idxs, point_idxs):
        print("OPTIMIZING!")

    def image_callback(self, msg):
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


        # IF YOU CAN - TRACK
        # if not self.px_ref is None:

        # TODO update this iteratively so its faster, this doesnt scale!
        self.active_2d_points_ids = [pt_id for pt_id, pt in self.tracked_2d_points.items() if pt.active]

        print("N_ACTIV 2D POINTS: " + str(len(self.active_2d_points_ids)))
        if len(self.active_2d_points_ids) > 0:
            # print("BEFORE TRACKING: " + str(self.px_ref.shape[0]))
            # self.px_ref, self.px_cur, self.tracking_stats = featureTracking(self.last_frame, self.new_frame, self.px_ref, self.tracking_stats)

            # DO TRACKING
            px_ref = np.array([self.tracked_2d_points[p].current_pos for p in self.active_2d_points_ids], dtype=np.float32)
            print("PX REF SHAPE")
            print(px_ref.shape)
            # print(px_ref)
            kp2, st, err = cv2.calcOpticalFlowPyrLK(self.last_frame, self.new_frame, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]

            st = st.reshape(st.shape[0])
            # kp1 = px_ref[st == 1]
            # kp2 = kp2[st == 1]

            # ADD OBSERVATION POSITIONS AND THIS_FRAME INDEX FOR ALL 2D POINTS IN 2D POINT DICT
            for i in range(px_ref.shape[0]):
                pt_id = self.active_2d_points_ids[i]
                if st[i] == 1:
                    self.tracked_2d_points[pt_id].addObservation(kp2[i], self.keyframe_idx)
                    self.tracked_2d_points[pt_id].current_pos = kp2[i]
                else:
                    self.tracked_2d_points[pt_id].active = False


            # print("AFTER TRACKING: " + str(self.px_cur.shape[0]))

        # TODO - REMOVE 2D POINTS THAT HAVE NOT BEEN OBSERVED FOR N FRAMES

        keyframe_time_threshold = 0.1
        keyframe_distance_threshold = 2.5

        time_since_last_keyframe = None
        dist_since_last_keyframe = None

        # CHECK IF SHOULD ADD KEYFRAME FOR TRIANGULATION OF POINTS
        # TODO - and prev keyframe parallax condition
        if len(self.keyframes) > 0:
            time_since_last_keyframe = (self.new_img_stamp - self.keyframes[-1].img_timestamp).to_sec()

        # if self.px_ref is None or ( time_since_last_keyframe > keyframe_time_threshold and dist_since_last_keyframe > keyframe_distance_threshold):
        if time_since_last_keyframe is None or time_since_last_keyframe > keyframe_time_threshold:
            # print("ATTEMPTING TO ADD NEW KEYFRAME! " + str(len(self.keyframes)) + ", dist: " + str(dist_since_last_keyframe) + ", time: " + str(time_since_last_keyframe))
            print("ATTEMPTING TO ADD NEW KEYFRAME! " + str(len(self.keyframes)))

            # IF YOU CAN - FIRST TRIANGULATE WITHOUT SCALE! - JUST TO DISCARD OUTLIERS
            # can_triangulate = not self.px_cur is None TODO
            # can_triangulate = True
            n_optim_keyfames = 4
            can_init = self.keyframe_idx >= n_optim_keyfames 
            if can_init:
                print("INITIALIIZNG")
                optim_kf_idxs = range(self.keyframe_idx - n_optim_keyfames - 1, self.keyframe_idx)
                optim_pt_ids = []

                # FIND WHICH PTS ARE OBSERVED FROM AT LEAST 2 OF THE OPTIMIZED KFS
                # TODO - and are not part of some body
                for pt_id, pt in self.tracked_2d_points.items():
                    point_kfs_in_optim_kfs = []
                    for k in optim_kf_idxs:
                        if k in pt.keyframe_observations.keys():
                            point_kfs_in_optim_kfs.append(k)
                    if len(point_kfs_in_optim_kfs) > 1:
                        optim_pt_ids.append(pt_id)

                print("OPTIM KF IDXS: ")
                print(optim_kf_idxs)
                print("OPTIM POINT IDXS: ")
                print(optim_pt_ids)
                self.visualBatchOptimization(optim_kf_idxs, optim_pt_ids)

            # CONTROL FEATURE POPULATION - ADDING AND PRUNING
            self.control_features_population()

            # RETURN IF STILL CANT FIND ANY, NOT ADDING KEYFRAME
            # if(self.px_cur is None):
            if len(self.active_2d_points_ids) < 4:
                print("--WARNING! NOT ENOUGH FEATURES FOUND EVEN AFTER POPULATION CONTROL! NOT ADDING KEYFRAME! NUM TRACKED PTS: " + str(len(self.active_2d_points_ids)))
                return

            # HAVE ENOUGH POINTS, ADD KEYFRAME
            print("ADDED NEW KEYFRAME! KF: " + str(len(self.keyframes)))
            new_kf = KeyFrame(self.new_img_stamp)

            # ADD SLAM POINTS INTO THIS KEYFRAME (submap)
            new_kf.slam_points = self.slam_points
            self.slam_points = None

            self.keyframes.append(new_kf)
            self.keyframe_idx += 1

            # STORE THE PIXELPOSITIONS OF ALL CURRENT POINTS FOR THIS GIVEN KEYFRAME 
            # for i in range(self.px_cur.shape[0]):
            #     self.tracking_stats[i].prev_keyframe_pixel_pos = self.px_cur[i, :]

        # VISUALIZE FEATURES
        vis = self.visualize_tracking()
        self.kp_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))
        # self.visualize_slampoints_in_space()

        comp_time = time.time() - comp_start_time
        print("computation time: " + str((comp_time) * 1000) +  " ms")


    def visualize_tracking(self):
        # rgb = np.zeros((self.new_frame.shape[0], self.new_frame.shape[1], 3), dtype=np.uint8)
        # print(self.new_frame.shape)
        rgb = np.repeat(copy.deepcopy(self.new_frame)[:, :, np.newaxis], 3, axis=2)
        # rgb = np.repeat((self.new_frame)[:, :, np.newaxis], 3, axis=2)

        px_cur = np.array([self.tracked_2d_points[p].current_pos for p in self.active_2d_points_ids])
        print("PX CUR SHAPE:")
        print(px_cur.shape)
        print(px_cur)

        if not px_cur is None and px_cur.size > 0:

            ll = np.array([0, 0])  # lower-left
            ur = np.array([self.width, self.height])  # upper-right
            inidx = np.all(np.logical_and(ll <= px_cur, px_cur <= ur), axis=1)
            inside_pix_idxs = px_cur[inidx].astype(int)

            growsize = 7
            minsize = 4

            for i in range(inside_pix_idxs.shape[0]):
                # size = self.tracking_stats[inidx][i].age / growsize
                # if size > growsize:
                #     size = growsize
                # size += minsize
                size = minsize

                rgb = cv2.circle(rgb, (inside_pix_idxs[i,0], inside_pix_idxs[i,1]), int(size), 
                               (255, 0, 255), -1) 

                # rgb = cv2.circle(rgb, (inside_pix_idxs[i,0], inside_pix_idxs[i,1]), 5, 
                #                (255, 0, 255), 2) 

                # triang_pix = self.K.dot(self.triangulated_points[:, inidx][:, i])
                # triang_pix = triang_pix  / triang_pix[2]
                # rgb = cv2.line(rgb, (int(triang_pix[0]), int(triang_pix[1])), (inside_pix_idxs[i,0], inside_pix_idxs[i,1]), (255, 0, 0), 3)
                # rgb = cv2.circle(rgb, (int(triang_pix[0]), int(triang_pix[1])), int(size), 
                #                (0, 0, 255), 3) 

            # if not self.noprior_triangulation_points is None:
            #     for i in range(self.noprior_triangulation_points .shape[1]):
            #         pixpos = self.K.dot(self.noprior_triangulation_points [:, i])
            #         pixpos = pixpos / pixpos[2]
            #         rgb = cv2.circle(rgb, (int(pixpos[0]), int(pixpos[1])), minsize+growsize+2, 
            #                        (0, 0, 255), 2) 

            # if not self.odomprior_triangulation_points  is None:
            #     for i in range(self.odomprior_triangulation_points  .shape[1]):
            #         pixpos = self.K.dot(self.odomprior_triangulation_points [:, i])
            #         pixpos = pixpos / pixpos[2]
            #         rgb = cv2.circle(rgb, (int(pixpos[0]), int(pixpos[1])), minsize+growsize+5, 
            #                        (0, 255, 255), 2) 

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

    def visualize_keypoints(self, img, kp):
        rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for k in kp:
            rgb[int(k.pt[1]), int(k.pt[0]), 0] = 255
        flow_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return flow_vis

    def visualize_slampoints_in_space(self):
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

    def visualize_keypoints_in_space(self, use_invdepth):
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

    def visualize_spheremap(self, T_current_cam_to_orig):
        if self.spheremap is None:
            return

        marker_array = MarkerArray()

        # point_cloud = PointCloud()
        # point_cloud.header.stamp = rospy.Time.now()
        # point_cloud.header.frame_id = 'global'  # Set the frame ID according to your robot's configuration
        T_vis = np.linalg.inv(T_current_cam_to_orig)
        pts = transformPoints(self.spheremap.points, T_vis)

        for i in range(self.spheremap.points.shape[0]):
            marker = Marker()
            marker.header.frame_id = "cam0"  # Change this frame_id if necessary
            marker.header.stamp = rospy.Time.now()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = i

            # Set the position (sphere center)
            marker.pose.position.x = pts[i][0]
            marker.pose.position.y = pts[i][1]
            marker.pose.position.z = pts[i][2]

            # Set the scale (sphere radius)
            marker.scale.x = 2 * self.spheremap.radii[i]
            marker.scale.y = 2 * self.spheremap.radii[i]
            marker.scale.z = 2 * self.spheremap.radii[i]

            marker.color.a = 0.1
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            # Add the marker to the MarkerArray
            marker_array.markers.append(marker)

        line_marker = Marker()
        line_marker.header.frame_id = "cam0"  # Set your desired frame_id
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.2  # Line width
        line_marker.color.a = 1.0  # Alpha

        for i in range(self.spheremap.connections.shape[0]):
            if self.spheremap.connections[i] is None:
                continue
            for j in range(len(self.spheremap.connections[i])):
                point1 = Point()
                point2 = Point()
                # trpt = transformPoints(np.array([[self.spheremap.points[i, :]], self.spheremap.points[self.spheremap.connections[i][j], :]]), T_vis)
                # p1 = self.spheremap.points[i, :]
                # p2 = self.spheremap.points[self.spheremap.connections[i][j], :][0]
                # p1 = trpt[0, :]
                # p2 = trpt[1, :]

                p1 = pts[i, :]
                p2 = pts[self.spheremap.connections[i].flatten()[j], :]
                # print(p1)
                # print(p2)
                # print("KURVA")
                # print(self.spheremap.connections[i].flatten())
                # print(self.spheremap.connections[i][j])
                
                point1.x = p1[0]
                point1.y = p1[1]
                point1.z = p1[2]
                point2.x = p2[0]
                point2.y = p2[1]
                point2.z = p2[2]
                line_marker.points.append(point1)
                line_marker.points.append(point2)
        marker_array.markers.append(line_marker)

        self.spheremap_spheres_pub.publish(marker_array)

    def decomp_essential_mat(self, E, q1, q2):
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

    @staticmethod
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
        return T


if __name__ == '__main__':
    rospy.init_node('visual_odom_node')
    optical_flow_node = OdomNode()
    rospy.spin()
    cv2.destroyAllWindows()