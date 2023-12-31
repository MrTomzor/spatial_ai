
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

from scipy.spatial.transform import Rotation as R

# #}

class LocalNavigatorModule:
    def __init__(self, mapper, ptraj_topic, output_path_topic):# # #{

        self.mapper = mapper
        self.odom_frame = mapper.odom_frame
        self.fcu_frame = mapper.fcu_frame
        self.tf_listener = mapper.tf_listener

        self.predicted_traj_mutex = threading.Lock()

        # OUTPUT PATH PUB
        self.path_for_trajectory_generator_pub = rospy.Publisher(output_path_topic, mrs_msgs.msg.Path, queue_size=10)

        # VIS PUB
        self.path_planning_vis_pub = rospy.Publisher('path_planning_vis', MarkerArray, queue_size=10)
        self.unsorted_vis_pub = rospy.Publisher('unsorted_markers', MarkerArray, queue_size=10)

        # PARAMS
        self.planning_enabled = rospy.get_param("local_nav/enabled")
        self.marker_scale = rospy.get_param("marker_scale") * 1.2
        self.path_step_size = rospy.get_param("local_nav/max_rrt_step_size")
        self.max_heading_change_per_m = rospy.get_param("local_nav/max_heading_change_per_m")

        self.safety_replanning_trigger_odist = rospy.get_param("local_nav/safety_replanning_trigger_odist")
        self.min_planning_odist = rospy.get_param("local_nav/min_planning_odist")
        self.max_planning_odist = rospy.get_param("local_nav/max_planning_odist")
        self.path_abandon_time = rospy.get_param("local_nav/path_abandon_time")

        self.planning_clearing_dist = rospy.get_param("local_nav/clearing_dist")
        self.output_path_resolution = rospy.get_param("local_nav/out_path_resolution")


        # PREDICTED TRAJ
        self.sub_predicted_trajectory = rospy.Subscriber(ptraj_topic, mrs_msgs.msg.MpcPredictionFullState, self.predicted_trajectory_callback, queue_size=10000)
        self.predicted_trajectory_pts_global = None

        # PLANNING PARAMS
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
        self.reaching_dist = rospy.get_param("local_nav/reaching_dist")
        self.reaching_angle = np.pi/2

        self.max_goal_vp_pathfinding_times = 3
        self.current_goal_vp_pathfinding_times = 0

        self.fspace_bonus_mod = 2
        self.safety_weight = 5

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
        
        # # #}

    def predicted_trajectory_callback(self, msg):# # #{
        with ScopedLock(self.predicted_traj_mutex):
            # PARSE THE MSG
            pts = np.array([[pt.x, pt.y, pt.z] for pt in msg.position])
            headings = np.array([h for h in msg.heading])
            msg_frame = msg.header.frame_id

            # GET CURRENT ODOM MSG
            T_msg_to_fcu = lookupTransformAsMatrix(msg_frame, self.fcu_frame, self.tf_listener)
            # T_fcu_to_global = lookupTransformAsMatrix(self.fcu_frame, 'global', self.tf_listener)
            T_fcu_to_global = lookupTransformAsMatrix(self.fcu_frame, self.odom_frame, self.tf_listener)

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

        pts_in_smap_frame = transformPoints(pts_global, np.linalg.inv(self.mapper.spheremap.T_global_to_own_origin))
        dists_from_start = np.linalg.norm(pts_global - pts_global[0, :], axis=1)

        odists = []
        for i in range(n_pts):
            odist_fs = self.mapper.spheremap.getMaxDistToFreespaceEdge(pts_in_smap_frame[i, :]) * self.fspace_bonus_mod
            odist_surf = self.mapper.spheremap.getMinDistToSurfaces(pts_in_smap_frame[i, :])
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
        
    def planning_loop_iter(self):# # #{
        if not self.planning_enabled:
            return
        with ScopedLock(self.mapper.spheremap_mutex):
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

        min_travers_dist = self.output_path_resolution

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
        if self.mapper.spheremap is None:
            return

        # GET CURRENT POS IN ODOM FRAME
        # latest_odom_msg = self.odom_buffer[-1]
        # T_global_to_imu = self.odom_msg_to_transformation_matrix(latest_odom_msg)
        # T_global_to_fcu = T_global_to_imu @ np.linalg.inv(self.T_fcu_to_imu)
        T_global_to_fcu = lookupTransformAsMatrix(self.odom_frame, self.fcu_frame, self.tf_listener)
        T_smap_origin_to_fcu = np.linalg.inv(self.mapper.spheremap.T_global_to_own_origin) @ T_global_to_fcu

        pos_fcu_in_global_frame = T_global_to_fcu[:3, 3]

        # GET START VP IN SMAP FRAME
        # T_smap_frame_to_fcu = np.linalg.inv(self.mapper.spheremap.T_global_to_own_origin) @ T_global_to_imu @ np.linalg.inv(self.T_fcu_to_imu)
        T_smap_frame_to_fcu = np.linalg.inv(self.mapper.spheremap.T_global_to_own_origin) @ T_global_to_fcu
        heading_in_smap_frame = transformationMatrixToHeading(T_smap_frame_to_fcu)
        planning_start_vp = Viewpoint(T_smap_frame_to_fcu[:3, 3], heading_in_smap_frame)

        planning_time = 1.0
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
                if (rospy.get_rostime() - self.trajectory_following_moved_time).to_sec() > self.path_abandon_time:
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
                                    startpos_smap_frame, startheading_smap_frame = transformViewpoints(future_pts[start_idx, :].reshape((1,3)), np.array([future_headings[start_idx]]), np.linalg.inv(self.mapper.spheremap.T_global_to_own_origin))
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
            goal_vp_smap_pos, goal_vp_smap_heading = transformViewpoints(self.current_goal_vp_global.position.reshape((1,3)), np.array([self.current_goal_vp_global.heading]), np.linalg.inv(self.mapper.spheremap.T_global_to_own_origin))
            best_path_pts, best_path_headings = self.find_paths_rrt(planning_start_vp , max_comp_time = planning_time, min_odist = current_minodist, max_odist = current_maxodist, mode = 'to_goal', goal_vp_smap = Viewpoint(goal_vp_smap_pos, goal_vp_smap_heading), max_step_size = self.path_step_size)

            if best_path_pts is None:
                print("NO PATH FOUND TO GOAL VP! TRY: " + str(self.current_goal_vp_pathfinding_times) + "/" + str(self.max_goal_vp_pathfinding_times))
                self.current_goal_vp_pathfinding_times += 1
                if self.current_goal_vp_pathfinding_times > self.max_goal_vp_pathfinding_times:
                    print("GOAL VP UNREACHABLE TOO MANY TIMES, DISCARDING IT!")
                    self.current_goal_vp_global = None
                return

        # elif not evading_obstacle:
        else:
            best_path_pts, best_path_headings = self.find_paths_rrt(planning_start_vp , max_comp_time = planning_time, min_odist = current_minodist, max_odist = current_maxodist, max_step_size = self.path_step_size)

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
        # latest_odom_msg = self.odom_buffer[-1]
        # T_global_to_imu = self.odom_msg_to_transformation_matrix(latest_odom_msg)
        # T_global_to_fcu = T_global_to_imu @ np.linalg.inv(self.T_fcu_to_imu)
        T_global_to_fcu = lookupTransformAsMatrix(self.odom_frame, self.fcu_frame, self.tf_listener)

        T_smap_origin_to_fcu = np.linalg.inv(self.mapper.spheremap.T_global_to_own_origin) @ T_global_to_fcu

        pts_fcu, headings_fcu = transformViewpoints(best_path_pts, best_path_headings, np.linalg.inv(T_smap_origin_to_fcu))
        pts_global, headings_global = transformViewpoints(best_path_pts, best_path_headings, self.mapper.spheremap.T_global_to_own_origin)

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

    def find_paths_rrt(self, start_vp, visualize = True, max_comp_time=0.5, max_step_size = 0.5, max_spread_dist=10, min_odist = 0.1, max_odist = 0.5, max_iter = 700006969420, goal_vp_smap=None, p_greedy = 0.3, mode='find_goals'):# # #{
        # print("RRT: STARTING, FROM TO:")
        # print(start_vp)

        bounds = np.ones((1,3)) * max_spread_dist
        # bounds += 10

        epicenter = start_vp.position 

        max_conn_size = max_step_size * 1.05

        comp_start_time = rospy.get_rostime()
        n_iters = 0
        n_unsafe_fspace = 0
        n_unsafe_obs = 0
        n_odom_unsafe  = 0
        n_rewirings = 0

        # INIT TREE
        n_nodes = 1
        tree_pos = start_vp.position.reshape((1, 3))
        tree_headings = np.array([start_vp.heading])
        odists = np.array([self.mapper.spheremap.getMaxDistToFreespaceEdge(start_vp.position)])
        parent_indices = np.array([-1])
        # child_indices = np.array([-1])
        child_indices = [[]]
        total_costs = np.array([0])
        T_fcu_to_cam = lookupTransformAsMatrix(self.fcu_frame, self.mapper.camera_frame, self.tf_listener)
        w = self.mapper.width
        h = self.mapper.height
        K = self.mapper.K

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
            new_node_fspace_dist = self.mapper.spheremap.getMaxDistToFreespaceEdge(new_node_pos) * self.fspace_bonus_mod 
            if new_node_fspace_dist < min_odist:
                # IF AT LEAST IN SPHERE AND WITHIN CLEARING DIST -> IS OK
                if new_node_fspace_dist < 0 or np.linalg.norm(new_node_pos - start_vp.position) > self.planning_clearing_dist:
                    n_unsafe_fspace += 1
                    continue

            # new_node_odist = self.mapper.spheremap.getMaxDistToFreespaceEdge(new_node_pos)
            new_node_odist = self.mapper.spheremap.getMinDistToSurfaces(new_node_pos)
            if new_node_odist < min_odist:
                n_unsafe_obs += 1
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

            heading_difs = (np.unwrap(potential_headings - tree_headings[connectable_mask]))

            # CLAMP HEADING DIFS!
            heading_overshoot_ratios = (np.abs(heading_difs) / dists[connectable_mask]) / self.max_heading_change_per_m 
            overshoot_mask = heading_overshoot_ratios > 1.0
            # print("OVERSHOOT MASK: " + str(np.sum(overshoot_mask)) + "/" + str(np.size(overshoot_mask)))
            heading_difs[overshoot_mask] = heading_difs[overshoot_mask] / heading_overshoot_ratios[overshoot_mask]
            potential_headings = np.unwrap(tree_headings[connectable_mask] + heading_difs) 

            # safety_costs = np.array([self.compute_safety_cost(odists, dists[idx]) for idx in connectable_indices]).flatten() * self.safety_weight
            safety_costs = self.compute_safety_cost(new_node_odist, min_odist, max_odist) * dists[connectable_mask] * self.safety_weight
            travelcosts = dists[connectable_mask] + safety_costs

            potential_new_total_costs = total_costs[connectable_mask] + travelcosts
            new_node_parent_idx2 = np.argmin(potential_new_total_costs)

            newnode_total_cost = potential_new_total_costs[new_node_parent_idx2]
            newnode_heading = potential_headings[new_node_parent_idx2]
            new_node_parent_idx = connectable_indices[new_node_parent_idx2]

            # CHECK IF ODOM PTS VISIBLE
            T_smap_orig_to_head_fcu = posAndHeadingToMatrix(new_node_pos, newnode_heading)
            T_smap_orig_to_head_cam = T_smap_orig_to_head_fcu @ T_fcu_to_cam 
            T_head_cam_to_smap_orig = np.linalg.inv(T_smap_orig_to_head_cam)
            

            # IF NO SURFELS IN MAP, FUCK IT ALL!
            if self.mapper.spheremap.surfel_points is None:
                continue

            surfel_points_in_camframe = transformPoints(self.mapper.spheremap.surfel_points, T_head_cam_to_smap_orig)
            only_rotmatrix = np.eye(4)
            only_rotmatrix[:3,:3] = T_head_cam_to_smap_orig[:3,:3]

            surfel_normals_in_camframe = transformPoints(self.mapper.spheremap.surfel_normals, only_rotmatrix)

            # visible_pts_mask = getVisiblePoints(surfel_points_in_camframe, surfel_normals_in_camframe, np.pi/3, 100, w, h, K, check_normals=False)
            # n_visible = np.sum(visible_pts_mask)
            # print("VISIBLE:" +str(n_visible))
            # if n_visible < 3:
            #     n_odom_unsafe += 1
            #     continue


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



        print("SPREAD FINISHED, HAVE " + str(tree_pos.shape[0]) + " NODES. ITERS: " +str(n_iters) + " F-UNSAFE: " + str(n_unsafe_fspace) + " O-UNSAFE: " + str(n_unsafe_obs) + " ODOM_UNSAFE: " + str(n_odom_unsafe) + " REWIRINGS: " + str(n_rewirings))
        if visualize:
            self.visualize_rrt_tree(self.mapper.spheremap.T_global_to_own_origin, tree_pos, None, odists, parent_indices)

        # FIND ALL HEADS, EVALUATE THEM
        comp_start_time = rospy.get_rostime()
        heads_indices = np.array([i for i in range(tree_pos.shape[0]) if len(child_indices[i]) == 0])
        print("N HEADS: " + str(heads_indices.size))

        # GET GLOBAL AND FCU FRAME POINTS
        # heads_global, heads_headings_global = transformViewpoints(tree_pos[heads_indices, :], tree_headings[heads_indices], self.mapper.spheremap.T_global_to_own_origin)
        heads_global = transformPoints(tree_pos[heads_indices, :], self.mapper.spheremap.T_global_to_own_origin)

        # latest_odom_msg = self.odom_buffer[-1] #TODO fix
        # T_global_to_imu = self.odom_msg_to_transformation_matrix(latest_odom_msg)
        # T_global_to_fcu = T_global_to_imu @ np.linalg.inv(self.T_fcu_to_imu)
        T_global_to_fcu = lookupTransformAsMatrix(self.odom_frame, self.fcu_frame, self.tf_listener)

        heads_values = None
        acceptance_thresh = -1000
        heads_scores = np.full((1, heads_indices.size), acceptance_thresh-1).flatten()
        heads_values = np.full((1, heads_indices.size), acceptance_thresh-1).flatten()

        if mode == 'find_goals':
            acceptance_thresh = 0
            mode2 = 'frontiers'
            if mode2 == 'frontiers':
                infovals = np.full((1, heads_indices.size), 0).flatten()
                # T_smap_orig_to_fcu = np.eye(4)
                # T_smap_orig_to_fcu[:3, :3] = headingToTransformationMatrix(start_vp.heading)
                # T_smap_orig_to_fcu[:3, 3] = start_vp.position

                # T_smap_orig_to_fcu = np.linalg.inv(self.mapper.spheremap.T_global_to_own_origin) @ T_global_to_fcu
                # T_smap_orig_to_cam = T_smap_orig_to_fcu @ lookupTransformAsMatrix(self.fcu_frame, self.mapper.camera_frame, self.tf_listener)
                # frontier_points_in_camframe = transformPoints(self.mapper.spheremap.frontier_points, np.linalg.inv(T_smap_orig_to_cam))
                # frontier_normals_in_camframe = transformPoints(self.mapper.spheremap.frontier_normals, np.linalg.inv(T_smap_orig_to_cam))

                if not self.mapper.spheremap.frontier_points is None:
                    # print("FCU TO CAM COMP")
                    # print(T_fcu_to_cam)

                    for i in range(heads_indices.size):
                        idx = heads_indices[i]
                        print("HEAD INDEX: " + str(idx))
                        startdist = np.linalg.norm(tree_pos[idx] - start_vp.position)
                        if startdist < self.reaching_dist:
                            print("TOO CLOSE")
                            continue
                        T_smap_orig_to_head_fcu = posAndHeadingToMatrix(tree_pos[idx, :], tree_headings[idx])
                        T_smap_orig_to_head_cam = T_smap_orig_to_head_fcu @ T_fcu_to_cam 
                        T_head_cam_to_smap_orig = np.linalg.inv(T_smap_orig_to_head_cam)

                        frontier_points_in_camframe = transformPoints(self.mapper.spheremap.frontier_points, T_head_cam_to_smap_orig)
                        only_rotmatrix = np.eye(4)
                        only_rotmatrix[:3,:3] = T_head_cam_to_smap_orig[:3,:3]
                        frontier_normals_in_camframe = transformPoints(self.mapper.spheremap.frontier_normals, only_rotmatrix)

                        # TODO - visitedness of frontiers - check which are seen by cam!!!
                        # normals_not_too_down_mask = frontier_points_in_camframe[:, 0] > -0.2

                        # print(frontier_points_in_camframe)

                        visible_pts_mask = getVisiblePoints(frontier_points_in_camframe, frontier_normals_in_camframe, np.pi/2, 15, w, h, K, verbose=True)
                        n_visible = np.sum(visible_pts_mask)
                        print("N VISIBLE FRONTIERS: " + str(n_visible))
                        if n_visible > 0:
                            # heads_values[i] = n_visible * 100
                            heads_values[i] = startdist * 100
            else:
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
        print("BEST HEAD IDX: " + str(best_node_index))
        print("-----BEST NODE VALUE (mode:"+mode+ ") : " +str(heads_values[best_head]))
        print("-----BEST NODE COST: " +str(total_costs[best_node_index]))

        if heads_values[best_head] <= acceptance_thresh:
            print("-----BEST HEAD IS BELOW ACCEPTANCE THRESHOLD, RETURNING NO VIABLE PATHS!!!")
            return None, None


        # RECONSTRUCT PATH FROM THERE
        path_idxs = [best_node_index]
        parent = parent_indices[best_node_index]

        dvec = tree_pos[best_node_index, :] - tree_pos[0, :]
        # tree_headings[best_node_index] = start_vp.heading

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
        msg.header.frame_id = self.fcu_frame
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
            # line_marker.scale.x = 0.04 * self.marker_scale  # Line width
            line_marker.scale.x = 0.4 * self.marker_scale  # Line width
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


