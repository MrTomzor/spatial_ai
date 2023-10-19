#!/usr/bin/env python

import rospy
# from sensor_msgs.msg import Image
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from visualization_msgs.msg import Marker
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
        self.prev_points = []

class OdomNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.prev_image = None
        self.prev_time = None

        self.kp_pub = rospy.Publisher('tracked_features_img', Image, queue_size=1)
        self.marker_pub = rospy.Publisher('/vo_odom', Marker, queue_size=10)
        self.kp_pcl_pub = rospy.Publisher('tracked_features_space', PointCloud, queue_size=10)

        self.sub_cam = rospy.Subscriber('/robot1/camera1/raw', Image, self.image_callback, queue_size=10000)

        self.odom_buffer = []
        self.odom_buffer_maxlen = 1000
        self.sub_odom = rospy.Subscriber('/ov_msckf/odomimu', Odometry, self.odometry_callback, queue_size=10000)

        self.tf_broadcaster = tf.TransformBroadcaster()

        self.orb = cv2.ORB_create(nfeatures=3000)

        # Load calib
        self.K = np.array([642.8495341420769, 0, 400, 0, 644.5958939934509, 300, 0, 0, 1]).reshape((3,3))
        # self.K = np.array([, 644.5958939934509, 400.0503960299562, 300.5824096896595]).reshape((3,3))

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
        self.max_features_per_bin = 4
        self.tracking_history_len = 4
        self.node_offline = False
        self.last_tried_landmarks_pxs = None

        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.detector = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)

        self.tracking_colors = np.random.randint(0, 255, (100, 3)) 

        self.n_frames = 0

        self.tracked_features = []

    def get_closest_time_odom_msg(self, stamp):
        bestmsg = None
        besttimedif = 0
        for msg in self.odom_buffer:
            tdif2 = np.power(msg.header.stamp.to_sec() - stamp.to_sec(), 2)
            if bestmsg == None or tdif2 < besttimedif:
                bestmsg = msg
                besttimedif = tdif2
        return bestmsg

    def odometry_callback(self, msg):
        self.odom_buffer.append(msg)
        if len(self.odom_buffer) > self.odom_buffer_maxlen:
            self.odom_buffer.pop(0)

    def control_features_population(self):
        wbins = self.width // self.tracking_bin_width
        hbins = self.height // self.tracking_bin_width
        found_total = 0

        if self.px_ref is None or len(self.px_ref) == 0:
            self.px_ref = self.detector.detect(self.new_frame)
            if self.px_ref is None or len(self.px_ref) == 0:
                return
            self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
            self.tracking_stats = np.array([TrackingStat() for x in self.px_ref], dtype=object)

        print("STTAS:")
        print(self.tracking_stats.shape)
        print(self.px_ref.shape)

        self.px_ref_prev = copy.deepcopy(self.px_ref)
        self.px_ref = self.px_ref[:0, :]

        self.tracking_stats_prev = copy.deepcopy(self.tracking_stats)
        self.tracking_stats = self.tracking_stats[:0]

        n_culled = 0

        for xx in range(wbins):
            for yy in range(hbins):
                # count how many we have there and get the points in there:
                ul = np.array([xx * self.tracking_bin_width , yy * self.tracking_bin_width ])  
                lr = np.array([ul[0] + self.tracking_bin_width , ul[1] + self.tracking_bin_width]) 

                inidx = np.all(np.logical_and(ul <= self.px_ref_prev, self.px_ref_prev <= lr), axis=1)
                # print(inidx)
                inside_points = []
                inside_stats = np.array([], dtype=object)

                n_existing_in_bin = 0
                if np.any(inidx):
                    inside_points = self.px_ref_prev[inidx]
                    inside_stats = self.tracking_stats_prev[inidx]
                    n_existing_in_bin = inside_points.shape[0]

                # print(n_existing_in_bin )

                if n_existing_in_bin > self.max_features_per_bin:
                    # CUTOFF POINTS ABOVE MAXIMUM, SORTED BY AGE
                    ages = np.array([-pt.age for pt in inside_stats])
                    # idxs = np.arange(self.max_features_per_bin)
                    # np.random.shuffle(idxs)
                    idxs = np.argsort(ages)
                    idxs = idxs[:self.max_features_per_bin]
                    n_culled += n_existing_in_bin - self.max_features_per_bin

                    self.px_ref = np.concatenate((self.px_ref, inside_points[idxs, :]))
                    self.tracking_stats = np.concatenate((self.tracking_stats, inside_stats[idxs]))
                    # self.px_ref = np.concatenate((self.px_ref, inside_points[:self.max_features_per_bin, :]))

                elif n_existing_in_bin < self.min_features_per_bin:
                    # ADD THE EXISTING
                    if n_existing_in_bin > 0:
                        self.px_ref = np.concatenate((self.px_ref, inside_points))
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
                    self.px_ref = np.concatenate((self.px_ref, locally_found))
                    # self.tracking_stats = np.array([TrackingStat for x in locally_found], dtype=object)
                    self.tracking_stats = np.concatenate((self.tracking_stats, np.array([TrackingStat() for x in locally_found], dtype=object)))
                else:
                    # JUST COPY THEM
                    self.px_ref = np.concatenate((self.px_ref, inside_points))
                    # self.tracking_stats += inside_stats
                    self.tracking_stats = np.concatenate((self.tracking_stats, inside_stats))

        print("FOUND IN BINS: " + str(found_total) + " CULLED: " + str(n_culled))
        print("CURRENT FEATURES: " + str(self.px_ref.shape[0]))

        # FIND FEATS IF ZERO!
        if(self.px_ref.shape[0] == 0):
            print("ZERO FEATURES! FINDING FEATURES")
            self.px_ref = self.detector.detect(self.new_frame)
            self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        

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

        if self.n_frames == 0:
            # FIND FIRST FEATURES
            self.n_frames = 1
            # self.last_img_stamp = stamp
            return
        self.n_frames += 1

        # CONTROL FEATURE POPULATION
        self.control_features_population()

        # RETURN IF STILL CANT FIND ANY
        if(len(self.px_ref) == 0):
            print("--WARNING! NO FEATURES FOUND!")
            return

        # TRACK
        print("BEFORE TRACKING: " + str(self.px_ref.shape[0]))
        self.px_ref, self.px_cur, self.tracking_stats = featureTracking(self.last_frame, self.new_frame, self.px_ref, self.tracking_stats)

        for i in range(self.px_ref.shape[0]):
            self.tracking_stats[i].age += 1
            self.tracking_stats[i].prev_points.append((self.px_ref[i,0], self.px_ref[i,1]))
            if len(self.tracking_stats[i].prev_points) > self.tracking_history_len:
                self.tracking_stats[i].prev_points.pop(0)

        print("AFTER TRACKING: " + str(self.px_cur.shape[0]))

        # TRY TO ESTIMATE FEATURES POINTS AND SELF MOTION (WITHOUT ODOM FOR NOW)

        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        mask = (mask > 0.5).flatten()
        # print("MASK: ")
        # print(mask.shape)
        self.px_cur = self.px_cur[mask, :]
        self.px_ref = self.px_ref[mask, :]
        self.tracking_stats = self.tracking_stats[mask]
        print("AFTER OUTLIER REJECTION: " + str(self.px_cur.shape[0]))

        R, t = self.decomp_essential_mat(E, self.px_ref, self.px_cur)
        self.last_triangulation_R = R
        self.last_triangulation_t = t


        self.gtsam_triang_points = None
        # self.pose_estim_gtsam()
        # try:
        #     self.pose_estim_gtsam()
        # except Exception as error:
        #     cprint("GTSAM ERROR:", 'red')
        #     cprint(error, 'red')
        #     # self.node_offline = True

        # closest_time_odom_msg = self.get_closest_time_odom_msg(self.new_img_stamp)
        # closest_time_prev_odom_msg = self.get_closest_time_odom_msg(self.last_img_stamp)

        # VISUALIZE FEATURES
        vis = self.visualize_tracking()
        self.kp_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))

        self.visualize_keypoints_in_space()

        comp_time = time.time() - comp_start_time
        print("computation time: " + str((comp_time) * 1000) +  " ms")

    def pose_estim_gtsam(self):
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(
            2, 1.0)  # two pixel in u and v

        # K = gtsam.Cal3_S2(50.0, 50.0, 0.0, 50.0, 50.0)
        # K = gtsam.Cal3_S2(self.K[0,0], self.K[0,0], 0.0, self.width, self.height)
        K = gtsam.Cal3_S2(self.K[0,0], self.K[0,0], 0.0, self.width/2, self.height/2)
        # self.K = np.array([642.8495341420769, 0, 400, 0, 644.5958939934509, 300, 0, 0, 1]).reshape((3,3))

        # Create an iSAM2 object. Unlike iSAM1, which performs periodic batch steps
        # to maintain proper linearization and efficient variable ordering, iSAM2
        # performs partial relinearization/reordering at each step. A parameter
        # structure is available that allows the user to set various properties, such
        # as the relinearization threshold and type of linear solver. For this
        # example, we we set the relinearization threshold small so the iSAM2 result
        # will approach the batch result.
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.01)
        parameters.relinearizeSkip = 1
        isam = gtsam.ISAM2(parameters)

        # Create a Factor Graph and Values to hold the new data
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()

        n_current_features = self.px_cur.shape[0]
        optim_steps = self.tracking_history_len + 1
        print("OPTIM STEPS: " + str(optim_steps))

        landmark_index = -1

        first_landmark_first_frame_pix = None
        min_parallax_pixels = 10
        landmark_idxs_to_feature_idxs = []
        # Add factors for each landmark observation

        # shuffle landmark idxs to not take just the first ones in top left corner
        landmark_selection_idxs = np.arange(n_current_features)
        np.random.shuffle(landmark_selection_idxs)

        self.last_tried_landmarks_pxs = None

        for k in range(n_current_features):
            i = landmark_selection_idxs[k]

            # CHECK IF OLD ENOUGH HISTORY
            if len(self.tracking_stats[i].prev_points) < self.tracking_history_len :
                continue
            # CHECK IF ENOUGH PARALLAX
            parallax_pix = np.linalg.norm(self.px_cur[i, :] - self.tracking_stats[i].prev_points[-1])
            if parallax_pix < min_parallax_pixels:
                continue
            # # MAX N LANDMARKS
            # if landmark_index > 5:
            #     break

            landmark_index += 1
            landmark_idxs_to_feature_idxs.append(i)

            # Add current observation
            current_pt = self.px_cur[i, :]
            graph.push_back(gtsam.GenericProjectionFactorCal3_S2(
                current_pt, measurement_noise, X(optim_steps-1), L(landmark_index), K))

            print("PREV PTS LEN: " + str(len(self.tracking_stats[i].prev_points)))
            # Add previous observations
            for j in range(len(self.tracking_stats[i].prev_points)):
               current_pt = self.tracking_stats[i].prev_points[-j-1]
               current_pt = np.array([current_pt[0], current_pt[1]], dtype=np.float64)

               graph.push_back(gtsam.GenericProjectionFactorCal3_S2(
                   current_pt, measurement_noise, X(optim_steps-2-j), L(landmark_index), K))
            if landmark_index == 0:
               current_pt = self.tracking_stats[i].prev_points[-1]
               first_landmark_first_frame_pix = np.array([current_pt[0], current_pt[1]], dtype=np.float64)


        print("NUM OPTIMIZED LANDMARKS:" + str(landmark_index+1))
        if landmark_index < 3:
            print("NOT ENOUGH USABLE LANDMARKS, RETURNING")
            return 1

        self.last_tried_landmarks_pxs = self.px_cur[landmark_idxs_to_feature_idxs]

        # Add a prior on pose x0
        # pose_noise = gtsam.noiseModel.Diagonal.Sigmas(0.1 * np.array(
        #     # [0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))  # 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
        #     [0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))  # 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
        # prior_first_pose = gtsam.Pose3()
        # print("PRIOR FIRST POSE:")
        # print(prior_first_pose)
        # graph.push_back(gtsam.PriorFactorPose3(X(0), prior_first_pose, pose_noise))

        # Add a prior final pose - set it as 0
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(
            # [0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))  # 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
            [0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))  # 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
        prior_first_pose = gtsam.Pose3()
        print("POSE PRIOR:")
        print(prior_first_pose)
        graph.push_back(gtsam.PriorFactorPose3(X(optim_steps-1), prior_first_pose, pose_noise))


        # Add a prior on landmark l0
        # TODO try projecting keypoint from first frame with camera matrix! scale will be off but whatever

        point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

        # ADD PRIOR ON FIRST LANDMARK FROM ITS TRIANG POSITION
        l0_triang_point = self.triangulated_points[:, landmark_selection_idxs[0]]
        graph.push_back(gtsam.PriorFactorPoint3(
            L(0), gtsam.Point3(l0_triang_point[0], l0_triang_point[1], l0_triang_point[2]), point_noise))  # add directly to graph

        # Add initial guesses to all observed landmarks
        for i in range(landmark_index+1):
            triang_point = self.triangulated_points[:, landmark_selection_idxs[i]]
            initial_estimate.insert(L(i), gtsam.Point3(triang_point[0], triang_point[1], triang_point[2]))
            #AND PRIOR
            # graph.push_back(gtsam.PriorFactorPoint3(
            #     L(i), gtsam.Point3(triang_point[0], triang_point[1], triang_point[2]), point_noise))  # add directly to graph

        # Add initial guess for previous pose (from triangulation) as 0
        initial_estimate.insert(X(optim_steps-1), prior_first_pose)

        # Add initial guess for previous pose (from triangulation) as 0
        second_to_last_pose = gtsam.Pose3(gtsam.Rot3(self.last_triangulation_R.T), gtsam.Point3(-self.last_triangulation_t))
        initial_estimate.insert(X(optim_steps-2), second_to_last_pose)

        # for all other x, set the pose estimate as the second-to-last as well
        for i in range(optim_steps-2):
            initial_estimate.insert(X(optim_steps-3-i), second_to_last_pose)
            # initial_estimate.insert(X(i), prior_first_pose.compose(gtsam.Pose3(
            #     gtsam.Rot3.Rodrigues(0, 0, 0), gtsam.Point3(0, 0, 0.1 * i))))
            # initial_estimate.insert(X(i), gtsam.Pose3().compose(gtsam.Pose3(
            #     gtsam.Rot3.Rodrigues(-0.1, 0.2, 0.25), gtsam.Point3(0.05, -0.10, 0.20))))

        # print("INITIAL ESTIM:")
        # print(initial_estimate)
        # print("GRAPH:")
        # print(graph)

        # Update iSAM with the new factors
        # isam.update(graph, initial_estimate)
        # isam.update()
        # isam.update()
        # isam.update()
        # isam.update()
        # isam.update()
        # isam.update()
        # isam.update()
        # result = isam.calculateEstimate()

        params = gtsam.DoglegParams()
        # params.setVerbosity('VALUES')
        params.setVerbosity('TERMINATION')
        optimizer = DoglegOptimizer(graph, initial_estimate, params)
        print('Optimizing:')
        result = optimizer.optimize()

        # Each call to iSAM2 update(*) performs one iteration of the iterative nonlinear solver.
        # If accuracy is desired at the expense of time, update(*) can be called additional
        # times to perform multiple optimizer iterations every step.
        # isam.update()
        print("yeehaw")
        # current_estimate = isam.calculateEstimate()
        # print("****************************************************")
        # print("Frame", i, ":")
        # for j in range(i + 1):
        #     print(X(j), ":", current_estimate.atPose3(X(j)))

        print("RESULTING LAST TWO POSES:")
        print( result.atPose3(X(optim_steps-1)))
        print( result.atPose3(X(optim_steps-2)))
        print("INITIAL LAST TWO POSES:")
        print(prior_first_pose)
        print(second_to_last_pose)

        self.gtsam_triang_points = []
        for j in range(landmark_index+1):
            # print(L(j), ":", current_estimate.atPoint3(L(j)))
            pt = result.atPoint3(L(j))
            self.gtsam_triang_points.append(pt)

            print("L" + str(j) + " - INITIALE ESTIMATE POINT:")
            triang_point = self.triangulated_points[:, landmark_selection_idxs[j]]
            print(triang_point)

            print("RESULTING POINT:")
            print(pt)

        # # visual_ISAM2_plot(current_estimate)

        # # Clear the factor graph and values for the next iteration
        graph.resize(0)
        initial_estimate.clear()


    def visualize_tracking(self):
        # rgb = np.zeros((self.new_frame.shape[0], self.new_frame.shape[1], 3), dtype=np.uint8)
        # print(self.new_frame.shape)
        rgb = np.repeat(copy.deepcopy(self.new_frame)[:, :, np.newaxis], 3, axis=2)
        # rgb = np.repeat((self.new_frame)[:, :, np.newaxis], 3, axis=2)

        ll = np.array([0, 0])  # lower-left
        ur = np.array([self.width, self.height])  # upper-right
        inidx = np.all(np.logical_and(ll <= self.px_cur, self.px_cur <= ur), axis=1)
        inside_pix_idxs = self.px_cur[inidx].astype(int)

        rgb[inside_pix_idxs[:, 1],inside_pix_idxs[:, 0], 0] = 255
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
            prevpt = self.tracking_stats[inidx][i].prev_points[-1]
            # rgb = cv2.circle(rgb, (int(prevpt[0]), int(prevpt[1])), int(size), 
            #                (0, 255, 0), 3) 
            # rgb = cv2.circle(rgb, (inside_pix_idxs[i,0], inside_pix_idxs[i,1]), int(size), 
            #                (255, 0, 255), -1) 
            # rgb = cv2.circle(rgb, (inside_pix_idxs[i,0], inside_pix_idxs[i,1]), 5, 
            #                (255, 0, 255), 2) 

            triang_pix = self.K.dot(self.triangulated_points[:, inidx][:, i])
            triang_pix = triang_pix  / triang_pix[2]
            rgb = cv2.line(rgb, (int(triang_pix[0]), int(triang_pix[1])), (inside_pix_idxs[i,0], inside_pix_idxs[i,1]), (255, 0, 0), 3)
            rgb = cv2.circle(rgb, (int(triang_pix[0]), int(triang_pix[1])), int(size), 
                           (0, 0, 255), 3) 

        # for i in range(self.triangulated_points.shape[1]):
        #     pixpos = self.K.dot(self.triangulated_points[:, i])
        #     pixpos = pixpos / pixpos[2]
        #     rgb = cv2.circle(rgb, (int(pixpos[0]), int(pixpos[1])), minsize+growsize+2, 
        #                    (0, 0, 255), 2) 

        if self.last_tried_landmarks_pxs is not None:
            for i in range(self.last_tried_landmarks_pxs.shape[0]):
                pixpos = self.last_tried_landmarks_pxs[i, :]
                rgb = cv2.circle(rgb, (int(pixpos[0]), int(pixpos[1])), minsize+growsize+5, 
                               (255, 150, 0), 6) 

        if not self.gtsam_triang_points is None:
            for i in range(len(self.gtsam_triang_points)):
                pixpos = self.K.dot(self.gtsam_triang_points[i])
                pixpos = pixpos / pixpos[2]
                rgb = cv2.circle(rgb, (int(pixpos[0]), int(pixpos[1])), minsize+growsize+8, 
                               (255, 0, 0), 6) 



        flow_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return flow_vis
    
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

    def visualize_keypoints_in_space(self):
        point_cloud = PointCloud()
        point_cloud.header.stamp = rospy.Time.now()
        point_cloud.header.frame_id = 'mission_origin'  # Set the frame ID according to your robot's configuration

        # Sample points
        # for i in range(kp.shape[1]):
        #     if kp[2, i] > 0:
        #         point1 = Point32()
        #         point1.x = kp[0, i]
        #         point1.y = kp[1, i]
        #         point1.z = kp[2, i]
        #         point_cloud.points.append(point1)
        maxnorm = np.max(np.linalg.norm(self.triangulated_points), axis=0)
        kps = 10 * self.triangulated_points / maxnorm

        for i in range(self.triangulated_points.shape[1]):
            if kps[2, i] > 0:
                point1 = Point32()
                point1.x = kps[0, i]
                point1.y = kps[1, i]
                point1.z = kps[2, i]
                point_cloud.points.append(point1)

        self.kp_pcl_pub.publish(point_cloud)

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
