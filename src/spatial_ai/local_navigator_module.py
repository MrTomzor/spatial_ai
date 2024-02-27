
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


class NavRoadmap:# # #{
    def __init__(self, points, headings = None):
        self.points = points
        self.headings = headings
        if headings is None:
            self.headings = np.full((points.shape[0]), None).flatten()

    # def __init__(self, points):
    #     self.points = points
    #     self.headings = np.full((points.shape[0]), None).flatten()
# # #}

class ExplorationGoal:
    def __init__(self, vp):
        self.viewpoint = vp
        self.num_observations = 0


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
        self.global_path_planning_vis_pub = rospy.Publisher('path_planning_vis_global', MarkerArray, queue_size=10)
        self.unsorted_vis_pub = rospy.Publisher('unsorted_markers', MarkerArray, queue_size=10)
        self.exploration_goals_vis_pub = rospy.Publisher('exploration_goals', MarkerArray, queue_size=10)

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
        self.roadmap = None
        self.roadmap_index = None
        self.roadmap_start_time = None
        self.roadmap_failing_time = 10
        self.roadmap_navigation_failed  = False
        self.roadmap_navigation_success = False

        self.local_nav_start_time = rospy.get_rostime()
        self.local_reaching_dist = 3
        self.traj_min_duration = 10

        self.last_path_send_time = rospy.Time.now()
        self.current_goal_vp_global = None
        self.reaching_dist = rospy.get_param("local_nav/reaching_dist")
        self.reaching_angle = np.pi/2

        self.max_goal_vp_pathfinding_times = 3
        self.current_goal_vp_pathfinding_times = 0

        self.fspace_bonus_mod = 2
        self.safety_weight = rospy.get_param("local_nav/safety_weight")

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

        # EXPLO PARAMS
        self.exploration_state = "local"
        self.exploration_goals = None
        self.goal_blocking_dist = 10
        self.goal_blocking_angle = np.pi/2
        self.goal_max_obsvs = 1
        
        # # #}

    # --PLANNING CORE

    def find_headingpath_astar_smap_frame(self, planning_start_vp, max_comp_time, min_odist, max_odist, goal_vp_smap, constant_heading_states_end = 0):# # #{
        # FIND PATH WITHOUT HEADINGS USING ASTAR ON SMAP
        # TODO
        raw_pts, path_cost = self.findPathAstarInSubmap(self.mapper.spheremap, planning_start_vp.position, goal_vp_smap.position, maxdist_to_graph=10, min_safe_dist=self.min_planning_odist, max_safe_dist=self.max_planning_odist, safety_weight = self.safety_weight)
        if raw_pts is None:
            return None, None, None
        n_pts = raw_pts.shape[0]

        # FIND HEADINGS FOR THE PATH THAT WOULD SUFFICE ALL CONSTRAINTS

        # SIMPLEST APPROACH - COMPUTE FRONT-FLYING HEADINGS, AND THEN GO FROM THE END AND ITERATIVELY TRY SHORTER TIMES WHEN TO START TURNING
        # COMPUTE FRONT FLYING HEADINGS
        front_flight_headings = np.full((n_pts), planning_start_vp.heading).flatten()
        prev_pt = planning_start_vp.position
        prev_heading = planning_start_vp.heading

        for i in range(n_pts):
            dirvec = (raw_pts[i] - prev_pt).flatten()
            dist = np.linalg.norm(dirvec)

            # CLAMP
            front_heading = np.arctan2(dirvec[1], dirvec[0])
            heading_dif = np.unwrap(np.array(front_heading).flatten() - np.array(prev_heading).flatten()) 

            max_allowed_heading_change = self.max_heading_change_per_m * dist
            if np.abs(heading_dif) > max_allowed_heading_change:
                front_flight_headings[i] = np.unwrap(prev_heading + max_allowed_heading_change * np.sign(heading_dif))


            prev_pt = raw_pts[i]
            prev_heading = front_flight_headings[i]

        # IF NO ALIGNING AT END NEEDED, JUST FLY
        if not goal_vp_smap.use_heading:
            final_pts = np.concatenate((raw_pts, goal_vp_smap.position), axis = 0)
            final_headings = np.concatenate((front_flight_headings.flatten(), np.array([front_flight_headings[-1]]).flatten()))
            return final_pts, final_headings, path_cost

        # ITERATIVELY CHECK HOW CLOSE TO THE GOAL YOU SHOULD ROTATE TO GOAL HEADING
        # TODO - is goal vp in path? NO!! ONLY THRUS SPHERES!!! n_pts-1 is idx of last sphere before goal!!!
        turning_start_idx = None
        goal_heading = goal_vp_smap.heading
        prev_pt = goal_vp_smap.position
        summed_dist = 0

        for i in range(n_pts):
            start_idx = n_pts - 1 - i
            dist = np.linalg.norm(raw_pts[start_idx] - prev_pt)
            summed_dist += dist

            heading_dif_to_goal = np.unwrap(goal_heading - front_flight_headings[start_idx])
            if np.abs(heading_dif_to_goal) < summed_dist * self.max_heading_change_per_m:
                turning_start_idx = start_idx
                break

        if turning_start_idx is None:
            print("ASTAR SENDABLE PATH PLANNING -- cant find good index to start turning!!!")
            return None, None, None

        print("ASTAR SENDABLE PATH PLANNING -- TURNING START IDX: " + str(turning_start_idx) + "/" + str(n_pts+1))

        # NOW MODIFY THE HEADING CHANGES FROM TURNING START (if too soon, just stay at end heading!)
        prev_pt = raw_pts[turning_start_idx]
        prev_heading = front_flight_headings[turning_start_idx]
        aligning_headings = front_flight_headings

        for i in range(turning_start_idx+1, n_pts):
            heading_dif_to_goal = np.unwrap(goal_heading - prev_heading)
            dist = np.linalg.norm(raw_pts[i] - prev_pt)

            # CLOSE ENOUGH
            max_allowed_heading_change = self.max_heading_change_per_m * dist
            if np.abs(heading_dif_to_goal) < max_allowed_heading_change:
                aligning_headings[i:] = goal_heading
                break

            # ELSE, MOVE AS MUCH AS POSSIBLE TO THE GOAL HEADING
            new_heading = np.unwrap(prev_heading + max_allowed_heading_change * np.sign(heading_dif_to_goal))
            aligning_headings[i] = new_heading

            prev_pt = raw_pts[i]
            prev_heading = aligning_headings[i]

        # ATTACH END VP
        # TODO
        final_pts = np.concatenate((raw_pts, goal_vp_smap.position), axis = 0)
        final_headings = np.concatenate((aligning_headings, goal_vp_smap.heading), axis = 0)

        return final_pts, final_headings, path_cost
    # # #}

    def find_paths_rrt_smap_frame(self, start_vp, visualize = True, max_comp_time=0.5, max_step_size = 0.5, max_spread_dist=15, min_odist = 0.1, max_odist = 0.5, max_iter = 700006969420, goal_vp_smap=None, p_greedy = 0.3, mode='find_goals', other_submap_idxs = None):# # #{
        print("RRT START:")
        print(start_vp.position)
        print(start_vp.heading)
        print(start_vp.position.shape)
        print(start_vp.heading.shape)
        # start_vp.heading = start_vp.heading[0]
        # print("RRT: STARTING, FROM TO:")
        # print(start_vp)
        
        n_prev_smaps = 0
        prev_smaps = []
        prev_smaps_T = []
        if not other_submap_idxs is None:
            n_prev_smaps = len(other_submap_idxs)
            prev_smaps = [self.mapper.mchunk.submaps[idx] for idx in other_submap_idxs]
            prev_smaps_T = [np.linalg.inv(self.mapper.mchunk.submaps[idx].T_global_to_own_origin) @ self.mapper.spheremap.T_global_to_own_origin for idx in other_submap_idxs]

            

        bounds = np.ones((1,3)) * max_spread_dist * 2
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
        tree_headings = np.array([start_vp.heading]).flatten()
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


        # MAIN SPREAD LOOP# #{
        while True:
            time_left = max_comp_time - (rospy.get_rostime() - comp_start_time).to_sec()
            if time_left <= 0 or n_iters >= max_iter:
                break
            n_iters += 1

            # SAMPLE PT IN SPACE
            sampling_goal_pt = (np.random.rand(1, 3)*2 - np.ones((1,3))) * bounds * 0.5  + epicenter
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
                    # CHECK WHETHER NOT IN FSPACE OF OTHER MAPS
                    is_unsafe_in_other_maps = True
                    for omi in range(n_prev_smaps):
                        pt_in_prev_smap = transformPoints(new_node_pos, prev_smaps_T[omi])
                        new_node_fspace_dist = prev_smaps[omi].getMaxDistToFreespaceEdge(pt_in_prev_smap) * self.fspace_bonus_mod 
                        if new_node_fspace_dist >= min_odist:
                            is_unsafe_in_other_maps = False
                            break
                    if is_unsafe_in_other_maps :
                        n_unsafe_fspace += 1
                        continue

            # new_node_odist = self.mapper.spheremap.getMaxDistToFreespaceEdge(new_node_pos)

            # CHECK IF NOT TOO CLOSE TO OBSTACLES
            new_node_odist = self.mapper.spheremap.getMinDistToSurfaces(new_node_pos)
            if new_node_odist < min_odist:
                n_unsafe_obs += 1
                continue

            # CHECK ALSO OTHER SUBMAPS IF ANY SURFACES TOO CLOSE IN THEM
            is_unsafe_in_other_maps = False
            for omi in range(n_prev_smaps):
                pt_in_prev_smap = transformPoints(new_node_pos, prev_smaps_T[omi])
                new_node_fspace_dist = prev_smaps[omi].getMaxDistToFreespaceEdge(pt_in_prev_smap) * self.fspace_bonus_mod 
                if new_node_fspace_dist >= min_odist:
                    is_unsafe_in_other_maps = False
                    break

            if is_unsafe_in_other_maps:
                n_unsafe_obs +=1
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
            dirvecs = (new_node_pos - tree_pos[connectable_mask, :])
            potential_headings = np.arctan2(dirvecs[:, 1], dirvecs[:,0])

            heading_difs = (np.unwrap(potential_headings.flatten() - tree_headings[connectable_mask].flatten()))

            # CLAMP HEADING DIFS!
            heading_overshoot_ratios = (np.abs(heading_difs) / dists[connectable_mask]) / self.max_heading_change_per_m 
            overshoot_mask = heading_overshoot_ratios > 1.0
            # print("OVERSHOOT MASK: " + str(np.sum(overshoot_mask)) + "/" + str(np.size(overshoot_mask)))
            heading_difs[overshoot_mask] = heading_difs[overshoot_mask] / heading_overshoot_ratios[overshoot_mask]
            potential_headings = np.unwrap(tree_headings[connectable_mask].flatten() + heading_difs) 

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

        # # #}

        print("SPREAD FINISHED, HAVE " + str(tree_pos.shape[0]) + " NODES. ITERS: " +str(n_iters) + " F-UNSAFE: " + str(n_unsafe_fspace) + " O-UNSAFE: " + str(n_unsafe_obs) + " ODOM_UNSAFE: " + str(n_odom_unsafe) + " REWIRINGS: " + str(n_rewirings))
        if visualize:
            self.visualize_rrt_tree(self.mapper.spheremap.T_global_to_own_origin, tree_pos, None, odists, parent_indices)

        # FIND ALL LEAFS, EVALUATE THEM# #{
        comp_start_time = rospy.get_rostime()
        heads_indices = np.array([i for i in range(tree_pos.shape[0]) if len(child_indices[i]) == 0])
        print("N HEADS: " + str(heads_indices.size))

        # GET GLOBAL AND FCU FRAME POINTS
        heads_global, heads_headings_global = transformViewpoints(tree_pos[heads_indices, :], tree_headings[heads_indices], self.mapper.spheremap.T_global_to_own_origin)
        # heads_global = transformPoints(tree_pos[heads_indices, :], self.mapper.spheremap.T_global_to_own_origin)
        T_global_to_fcu = lookupTransformAsMatrix(self.odom_frame, self.fcu_frame, self.tf_listener)

        print("CUR FCU IN GLOBAL:")
        print(T_global_to_fcu[:3,3])

        heads_values = None
        acceptance_thresh = -1000
        heads_scores = np.full((1, heads_indices.size), acceptance_thresh-1).flatten()
        heads_values = np.full((1, heads_indices.size), acceptance_thresh-1).flatten()

        if mode == 'find_goals':
            acceptance_thresh = 0
            mode2 = 'frontiers'
            mode3 = 'forward'
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

                        visible_pts_mask = getVisiblePoints(frontier_points_in_camframe, frontier_normals_in_camframe, np.pi/2, 15, w, h, K, verbose=False)
                        n_visible = np.sum(visible_pts_mask)
                        print("N VISIBLE FRONTIERS: " + str(n_visible))
                        if n_visible > 0:
                            # heads_values[i] = n_visible * 100 # VALUE OF FRONTIERS
                            # heads_values[i] = startdist * 100 # VALUE OF DIST FROM CUR POS
                            current_fcu_in_global = T_global_to_fcu[:3, 3].reshape((1,3))
                            global_pos = heads_global[i]

                            # heads_values[i] = (global_pos - current_fcu_in_global).flatten()[0] * 5 # VALUE OF DIST IN FORWARD DIR!
                            # heads_values[i] = np.linalg.norm((global_pos - current_fcu_in_global).flatten()) * 5 # VALUE OF DIST IN ANY DIR!
                            heads_values[i] = np.random.rand() * 50 # VALUE OF DIST IN ANY DIR!

                            if heads_values[i] <= acceptance_thresh:
                                heads_values[i] = acceptance_thresh + 1
                        else:
                            heads_values[i] = acceptance_thresh - 1
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
            print("CLOSEST NODE:")
            print(np.min(dists_from_goal))

            heads_values[reaching_goal] = acceptance_thresh + 1
            heads_scores[reaching_goal] = -total_costs[heads_indices[reaching_goal]]
# # #}

        # IF MODE IS TO FIND GOALS, RETURN ALL END VPS THAT ARE ACCEPTABLE, AND NOT BLOCKED
        if mode == 'find_goals':
            acceptable_goals = []
            for i in range(heads_values.size):
                if heads_values[i] < acceptance_thresh:
                    continue
                # CHECK IF NOT BLOCKED BY ANY
                potential_vp = Viewpoint(heads_global[i, :], heads_headings_global[i])

                # ADD TO RES
                acceptable_goals.append(potential_vp)
            print("RRT FOUND " + str(len(acceptable_goals)) + " ACCEPTABLE NEW GOALS")
            return acceptable_goals

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

    def findPathAstarInSubmap(self, smap, startpoint, endpoint, maxdist_to_graph=10, min_safe_dist=0.5, max_safe_dist=3, safety_weight=1):# # #{
        print("PATHFIND: STARTING, FROM TO:")
        print(startpoint)
        print(endpoint)

        if smap.points is None: 
            print("PATHFIND: MAP IS EMPTY!!!")
            return None, None

        # FIND NEAREST NODES TO START AND END
        dist_from_startpoint = np.linalg.norm((smap.points - startpoint), axis=1) - smap.radii
        start_node_index = np.argmin(dist_from_startpoint)

        # ONLY CONSIDER GOAL POINTS IN THE SAME CONNECTIVITY REGION
        dist_from_endpoint = np.linalg.norm((smap.points - endpoint), axis=1) - smap.radii
        end_node_index = np.argmin(dist_from_endpoint)

        if(dist_from_startpoint[start_node_index] > maxdist_to_graph or dist_from_endpoint[end_node_index] > maxdist_to_graph):
            print("PATHFIND: start or end too far form graph!")
            print(dist_from_startpoint[start_node_index])
            print(dist_from_endpoint[end_node_index])
            print("N SPHERES:")
            print(smap.points.shape[0])
            return None, None
        print("PATHFIND: node indices:")
        print(start_node_index)
        print(end_node_index)

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
                safety_cost = self.compute_safety_cost(smap.radii[neighbor_index], min_safe_dist, max_safe_dist) * euclid_dist * self.safety_weight
                tentative_g = g_score[current_index] + euclid_dist + safety_cost

                if neighbor_index not in g_score or tentative_g < g_score[neighbor_index]:
                    g_score[neighbor_index] = tentative_g
                    f_score[neighbor_index] = tentative_g + np.linalg.norm(smap.points[neighbor_index] - goal_node_pt)
                    heapq.heappush(open_set, (f_score[neighbor_index], neighbor_index, current_index))
                    closed_set[neighbor_index] = current_index

        path.reverse()
        print("PATHFIND: RES PATH: ")
        print(path)
        print("N CLOSED NODES: " + str(len(closed_set.keys())))

        if len(path) == 0:
            return None, None

        self.visualize_trajectory(smap.points[np.array(path), :], smap.T_global_to_own_origin, headings=None, do_line=True, start=startpoint, goal=endpoint, pub=self.global_path_planning_vis_pub)

        return smap.points[np.array(path), :], g_score[end_node_index]
# # #}

    def planning_loop_iter(self):# # #{
        if not self.planning_enabled:
            return
        with ScopedLock(self.mapper.spheremap_mutex):
            # self.dumb_forward_flight_rrt_iter()
            T_global_to_fcu = lookupTransformAsMatrix(self.odom_frame, self.fcu_frame, self.tf_listener)
            T_smap_origin_to_fcu = np.linalg.inv(self.mapper.spheremap.T_global_to_own_origin) @ T_global_to_fcu
            current_vp_smap_frame = Viewpoint(T=T_smap_origin_to_fcu) 

            self.tryAddingGoalsNearby(current_vp_smap_frame)
            self.visualize_exploration_goals()
            self.exploration_logic()

            # self.visualize_exploration_goals()

            # if self.roadmap is None or self.roadmap_navigation_failed:
            #     print("TMP - pathfinding home")
            #     T_global_to_fcu = lookupTransformAsMatrix(self.odom_frame, self.fcu_frame, self.tf_listener)
            #     T_smap_origin_to_fcu = np.linalg.inv(self.mapper.spheremap.T_global_to_own_origin) @ T_global_to_fcu
            #     pos_fcu_in_global_frame = T_global_to_fcu[:3, 3]

            #     path_home_pts, path_cost = self.findPathAstarInSubmap(self.mapper.spheremap, T_smap_origin_to_fcu[:3, 3].T, np.zeros((3,1)).T, maxdist_to_graph=10, min_safe_dist=self.min_planning_odist, max_safe_dist=self.max_planning_odist, safety_weight = self.safety_weight)  
            #     path_home_pts = transformPoints(path_home_pts, self.mapper.spheremap.T_global_to_own_origin)

            #     rm = NavRoadmap(path_home_pts)
            #     self.set_roadmap(rm)

            self.roadmap_following_iter()

    # # #}
    # def homing_logic(self):

    # --MRS SYSTEM ATTACHMENTS

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

    def get_future_predicted_trajectory_global_frame(self):# # #{
        now = rospy.get_rostime()

        stamps = None
        pts = None
        headings = None

        if self.predicted_trajectory_pts_global is None:
            # return None, None, None

            T_global_to_fcu = lookupTransformAsMatrix(self.odom_frame, self.fcu_frame, self.tf_listener)
            pos = T_global_to_fcu[:3, 3].reshape((1,3))
            # pos[0] += 0.001
            heading = transformationMatrixToHeading(T_global_to_fcu)

            return pos, heading.reshape((1,1)), np.array([0])

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
            print("CHECKING ODIST: " + str(odist))

            odists.append(odist)
            if odist < min_odist and dists_from_start[i] > clearing_dist:
                res.append(i)
        odists = np.array(odists)
        # print("ODISTS RANKED:")
        # print("ODISTS:")
        # # odists = np.sort(odists)
        # print(odists)
        return np.array(res), odists# # #}

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
        self.last_path_send_time = rospy.Time.now()

        return None
# # #}


    # --ROADMAP STUFF

    def roadmap_following_iter(self, mode="astar"):# # #{
        if self.mapper.spheremap is None:
            return
        if self.roadmap is None:
            return
        if self.roadmap_navigation_failed:
            return

        # GET CURRENT POS IN ODOM FRAME, GET PLANNING TIME# #{
        T_global_to_fcu = lookupTransformAsMatrix(self.odom_frame, self.fcu_frame, self.tf_listener)
        T_smap_origin_to_fcu = np.linalg.inv(self.mapper.spheremap.T_global_to_own_origin) @ T_global_to_fcu

        current_pos = T_global_to_fcu[:3, 3].T
        current_heading = transformationMatrixToHeading(T_global_to_fcu)

        T_smap_frame_to_fcu = np.linalg.inv(self.mapper.spheremap.T_global_to_own_origin) @ T_global_to_fcu
        heading_in_smap_frame = transformationMatrixToHeading(T_smap_frame_to_fcu)
        planning_start_vp = Viewpoint(T_smap_frame_to_fcu[:3, 3], heading_in_smap_frame)

        planning_time = 1.0
        current_maxodist = self.max_planning_odist
        current_minodist = self.min_planning_odist
        # # #}

        # UPDATE CARROT GOAL# #{
        carrot_dist = 15

        roadmap_len = self.roadmap.points.shape[0]
        i = self.roadmap_index + 1
        while i < roadmap_len:
            pos = self.roadmap.points[i, :]
            dist = np.linalg.norm(current_pos - pos)
            if dist < carrot_dist:
                self.roadmap_index = i
                print("ROADMAP - increased carrot index to " + str(i))
                # TODO - if intermittent goals have some vp req, you must pass it too
                self.last_carrot_update = rospy.Time.now()
            i += 1

        arrow_pos = self.roadmap.points[self.roadmap_index, :].flatten() - np.array([0,0,10]).flatten()
        arrow_pos2 = self.roadmap.points[self.roadmap_index, :].flatten()
        self.visualize_arrow(arrow_pos, arrow_pos2, r=0.5,g=0.5, scale=0.5, marker_idx=42)
        # # #}

        # CHECK IF END REACHED AND CAN RAISE SUCCESS FLAG# #{
        goal_pos = self.roadmap.points[self.roadmap_index, :].reshape((1,3))
        goal_heading = self.roadmap.headings[self.roadmap_index]

        # TODO - fix better, just wanna align at end!
        # if self.roadmap_index != roadmap_len-1:
        #     goal_heading = None
        goal_heading = None

        goal_dist = np.linalg.norm(current_pos - goal_pos)
        heading_reached = True

        if not goal_heading is None:
            heading_dif = np.abs(np.unwrap(goal_heading - current_heading))
            if heading_dif > self.reaching_angle:
                heading_reached = False
        else:
            goal_heading = np.array([current_heading])

        if goal_dist < self.reaching_dist and heading_reached:
            print("-----------------------ROADMAP NAVIGATION SUCCESS----------------------")
            self.roadmap_navigation_success = True
        # # #}

        # CHECK IF NOT PROGRESSING FOR A LONG TIME# #{
        if (rospy.Time.now() - self.last_carrot_update).to_sec() > self.roadmap_failing_time and not self.roadmap_navigation_success:
            print("ROADMAP - not progressing for a long time, setting as failed!")
            self.roadmap_navigation_failed = True
            return
        # # #}

        # CHECK SAFETY OF TRAJECTORY! AND IF UNSAFE, SELECT NEW PLANNING TIME# #{
        # GET RELEVANT PTS IN PREDICTED TRAJ!
        planning_time = 1
        planning_start_vp = Viewpoint(current_pos, current_heading)
        future_pts, future_headings, relative_times = self.get_future_predicted_trajectory_global_frame()
        enhancing_path = False

        if not future_pts is None:
            future_pts_dists = np.linalg.norm(future_pts - current_pos, axis=1)
            unsafe_idxs, odists = self.find_unsafe_pt_idxs_on_global_path(future_pts, future_headings, self.safety_replanning_trigger_odist, -1)
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
                    print(current_pos)
                    print("TIME TO UNSAFETY: " + str(time_to_unsafety) +" / " + str(triggering_time))

                    planning_time = time_to_unsafety * 0.5
                    if planning_time <= 0.2:
                        planning_time = 0.2
                    print("PLANNING TIME: " +str(planning_time))

                    # VISUALIZE OFFENDING PT
                    arrow_pos2 = future_pts[unsafe_idxs[0], :]
                    arrow_pos = copy.deepcopy(arrow_pos2)
                    arrow_pos[2] -= 10
                    self.visualize_arrow(arrow_pos.flatten(), arrow_pos2.flatten(), r=0.5, scale=0.5, marker_idx=1)

                    after_planning_time = relative_times > planning_time + 0.2
                    if not np.any(after_planning_time):
                        print("PLANNING TIME IS LONGER THAN PREDICTED TRAJ! SHORTENING IT AND PLANNING FROM CURRENT POS IN SMAP FRAME!")
                        planning_time *= 0.5
                    # else:
                    #     start_idx = np.where(after_planning_time)[0][0]
                    #     print("PLANNING FROM VP OF INDEX " + str(start_idx) + " AT DIST " + str(future_pts_dists[start_idx]))
                    #     startpos_smap_frame, startheading_smap_frame = transformViewpoints(future_pts[start_idx, :].reshape((1,3)), np.array([future_headings[start_idx]]), np.linalg.inv(self.mapper.spheremap.T_global_to_own_origin))
                    #     planning_start_vp = Viewpoint(startpos_smap_frame, startheading_smap_frame[0])
                    evading_obstacle = True
                # TODO - replaning to goal trigger. If failed -> replanning to any other goal. If failed -> evasive maneuvers n wait!
                else:
                    # return
                    # print("replanning for enhancing path")
                    enhancing_path = True
            else:
                # return
                # print("replanning for enhancing path")
                enhancing_path = True
        else:
            # return
            # print("replanning for enhancing path")
            enhancing_path = True
        # # #}

        # IF TIME TO PERIODICALLY REPLAN / PREDICTED TRAJ UNSAFE / CARROT GOAL MOVED A LOT, COMPUTE NEW PATH TO CURRENT CARROT AND SEND IT
        if enhancing_path and (rospy.Time.now() - self.last_path_send_time).to_sec() < 4:
        # if enhancing_path:
            return

        # GET THE START POSE ALONG PREDICTED TRAJECTORY UP TO planning_time IN THE FUTURE
        if not future_pts is None:
            after_planning_time = relative_times > planning_time + 0.2
            start_idx = None
            if not np.any(after_planning_time):
                start_idx = relative_times.size - 1
                pass
            else:
                start_idx = np.where(after_planning_time)[0][0]
            print("ROADMAP - HAVE FUTURE PTS, PLANNING FROM VP OF INDEX " + str(start_idx) + " AT DIST " + str(future_pts_dists[start_idx]))
            startpos_smap_frame, startheading_smap_frame = transformViewpoints(future_pts[start_idx, :].reshape((1,3)), np.array([future_headings[start_idx]]), np.linalg.inv(self.mapper.spheremap.T_global_to_own_origin))
            planning_start_vp = Viewpoint(startpos_smap_frame, startheading_smap_frame[0])

        # FIND PATH TO GOAL
        best_path_pts = None
        best_path_headings = None
        save_last_pos_as_goal = False
        trusted_submap_idxs = None
        if len(self.mapper.mchunk.submaps) > 0:
            trusted_submap_idxs = [len(self.mapper.mchunk.submaps) - 1]

        # TRANSFORM GOAL VP TO SPHEREMAP COORDS
        print("SHAPEZ")
        print(goal_pos.shape)
        goal_vp_smap_pos, goal_vp_smap_heading = transformViewpoints(goal_pos, goal_heading, np.linalg.inv(self.mapper.spheremap.T_global_to_own_origin))
        goal_vp_smap_heading = None # TODO - remove
        print("GOAL IN SMAP FRAME:")
        print(goal_vp_smap_pos)
        print("GOAL IN ODOM FRAME:")
        print(goal_pos)

        # COMPUTE THE PATH
        best_path_pts = None 
        best_path_headings = None

        if mode == "rrt":
            max_spread_dist = np.linalg.norm(planning_start_vp.position - goal_vp_smap_pos) * 1.2
            best_path_pts, best_path_headings = self.find_paths_rrt_smap_frame(planning_start_vp , max_comp_time = planning_time, min_odist = current_minodist, max_odist = current_maxodist, mode = 'to_goal', goal_vp_smap = Viewpoint(goal_vp_smap_pos, goal_vp_smap_heading), max_step_size = self.path_step_size, other_submap_idxs = trusted_submap_idxs, max_spread_dist=max_spread_dist)
        elif mode == "astar":
            best_path_pts, best_path_headings, path_cost = self.find_headingpath_astar_smap_frame(planning_start_vp , max_comp_time = planning_time, min_odist = current_minodist, max_odist = current_maxodist, goal_vp_smap = Viewpoint(goal_vp_smap_pos, goal_vp_smap_heading))

    
        if best_path_pts is None:
            print("NO PATH FOUND TO GOAL VP! TRY: " + str(self.current_goal_vp_pathfinding_times) + "/" + str(self.max_goal_vp_pathfinding_times))
            self.current_goal_vp_pathfinding_times += 1
            if self.current_goal_vp_pathfinding_times > self.max_goal_vp_pathfinding_times:
                print("GOAL VP UNREACHABLE TOO MANY TIMES, DISCARDING IT!")
                # self.current_goal_vp_global = None
                self.roadmap_navigation_failed = True
            return


        min_path_len = 1
        if np.linalg.norm(best_path_pts[-1, :] - planning_start_vp.position) < min_path_len:
            print("FOUND BEST PATH TOO SHORT!")
            return

        # GET FCU TRANSFORM NOWW! AFTER planning_time OF PLANNING
        T_global_to_fcu = lookupTransformAsMatrix(self.odom_frame, self.fcu_frame, self.tf_listener)

        T_smap_origin_to_fcu = np.linalg.inv(self.mapper.spheremap.T_global_to_own_origin) @ T_global_to_fcu

        pts_fcu, headings_fcu = transformViewpoints(best_path_pts, best_path_headings, np.linalg.inv(T_smap_origin_to_fcu))
        pts_global, headings_global = transformViewpoints(best_path_pts, best_path_headings, self.mapper.spheremap.T_global_to_own_origin)

        if save_last_pos_as_goal:
            self.current_goal_vp_global = Viewpoint(pts_global[-1, :], headings_global[-1])
            self.current_goal_vp_pathfinding_times = 0

        # print("PTS IN FCU FRAME (SHOULD START NEAR ZERO!):")
        # print(pts_fcu)
        # print("HEADINGS IN GLOBAL FRAME (SHOULD START NEAR ZERO!):") # GLOBAL IS ROTATED BY 180 DEG FROM VIO ORIGIN!
        # print(headings_fcu)

        # SEND IT
        self.send_path_to_trajectory_generator(pts_fcu, headings_fcu)
        self.trajectory_following_moved_time = rospy.get_rostime()

        self.visualize_trajectory(pts_global, np.eye(4), headings_global, do_line = False, frame_id = self.odom_frame, pub = self.path_planning_vis_pub)

    # # #}

    def set_roadmap(self, roadmap):# # #{
        print("NAV: SETTING ROADMAP OF LEN " + str(roadmap.points.shape[0]))
        self.roadmap_navigation_failed  = False
        self.roadmap_navigation_success = False
        self.roadmap_index = 0
        self.roadmap = roadmap
        self.last_carrot_update = rospy.Time.now()
    # # #}

    def stop_following_roadmap_and_stop(self):# # #{
        # RESET ROADMAP STUFF
        self.roadmap = None
        self.roadmap_index = 0

        # SEND CURRENT POSE AS REFERENCE
        T_global_to_fcu = lookupTransformAsMatrix(self.odom_frame, self.fcu_frame, self.tf_listener)
        # self.send_path_to_trajectory_generator(T_global_to_fcu[:3,3].T.reshape((1,3)), [transformationMatrixToHeading(T_global_to_fcu)])
        print(" KINDA STOPPING ")
    # # #}


    # --EXPLORATION STUFF

    def getBestGoalRoadmap(self):# # #{
        T_global_to_fcu = lookupTransformAsMatrix(self.odom_frame, self.fcu_frame, self.tf_listener)
        T_smap_origin_to_fcu = np.linalg.inv(self.mapper.spheremap.T_global_to_own_origin) @ T_global_to_fcu
        current_vp_smap_frame = Viewpoint(T=T_smap_origin_to_fcu) 

        if self.exploration_goals is None:
            return None

        n_tries = self.exploration_goals.size
        paths_to_goals = []
        paths_costs = []

        for i in range(n_tries):
            goal = self.exploration_goals[i]
            if goal.num_observations >= self.goal_max_obsvs:
                continue

            # FIND PATH TO GOAL, SAVE IT AND COST, IF FOUND
            # TRANSFORM GOAL TO SMAP FRAME
            goal_pos_smap, goal_heading_smap = transformViewpoints(goal.viewpoint.position.reshape((1,3)), np.array([goal.viewpoint.heading]), np.linalg.inv(self.mapper.spheremap.T_global_to_own_origin))
            goal_vp_smap = Viewpoint(goal_pos_smap, goal_heading_smap)

            # FIND HEADING PATH IF POSSIBLE
            path_pts, path_headings, path_cost = self.find_headingpath_astar_smap_frame(current_vp_smap_frame , max_comp_time = 1, min_odist = self.min_planning_odist, max_odist = self.max_planning_odist, goal_vp_smap = goal_vp_smap)
            if not path_pts is None:
                paths_to_goals.append([path_pts, path_headings])
                paths_costs.append(path_cost)

        # IF NO VPS REACHABLE, RETURN 
        if len(paths_to_goals) == 0:
            return None

        # RANK THE PATHS AND VPS
        scores = np.array(paths_costs)
        best_goal_idx = np.argmin(scores)
        best_path_pts = paths_to_goals[best_goal_idx][0]
        best_path_headings = paths_to_goals[best_goal_idx][1]

        # TRANSFORMED BEST PATH TO GLOBAL FRAME
        global_pts, global_headings = transformViewpoints(best_path_pts, best_path_headings, self.mapper.spheremap.T_global_to_own_origin)

        # RETURN ROADMAP IN GLOBAL FRAME
        return NavRoadmap(global_pts, global_headings)
    # # #}

    def isViewpointBlockedByExplorationGoals(self, vp):# # #{
        if self.exploration_goals is None:
            return False, None

        explo_points = np.array([goal.viewpoint.position for goal in self.exploration_goals])
        explo_headings = np.array([goal.viewpoint.heading for goal in self.exploration_goals]).flatten()

        dists = np.linalg.norm(vp.position - explo_points, axis = 1).flatten()

        # heading_difs = np.unwrap(explo_headings - vp.heading)
        heading_difs = hdif(explo_headings - vp.heading)

        gp = heading_difs > np.pi
        lp = heading_difs < -np.pi
        heading_difs[gp] = np.pi * 2 - heading_difs[gp] 
        heading_difs[lp] = np.pi * 2 + heading_difs[lp] 

        distfailed = dists < self.goal_blocking_dist
        headingfailed = np.abs(heading_difs) < self.goal_blocking_angle

        blocking_mask = np.logical_and(distfailed, headingfailed)

        if not np.any(blocking_mask):
            return False, None

        blocking_goals = self.exploration_goals[blocking_mask]
        return True, blocking_goals

    # # #}

    def tryAddingGoalsNearby(self, planning_start_vp, search_dist=30, planning_time=0.5):# # #{
        frontier_vps = self.find_paths_rrt_smap_frame(planning_start_vp, max_comp_time = planning_time, min_odist = self.min_planning_odist, max_odist = self.max_planning_odist, max_step_size = self.path_step_size, max_spread_dist = search_dist, mode = 'find_goals')
        n_added = 0



        if len(frontier_vps ) > 0:
            acceptable_goals = []
            n_added = 0
            for i in range(len(frontier_vps)):
                is_blocked, by_who = self.isViewpointBlockedByExplorationGoals(frontier_vps[i])
                if is_blocked:
                    continue
                new_goal = ExplorationGoal(frontier_vps[i])
                if self.exploration_goals is None:
                    self.exploration_goals = np.array([new_goal], dtype=object)
                else:
                    self.exploration_goals = np.concatenate((self.exploration_goals, np.array([new_goal], dtype=object)))
                n_added += 1
            if n_added == 0:
                return 0
            print("EXPLORATION - now have num of goals: " + str(self.exploration_goals.size))
        return n_added
    # # #}

    def exploration_logic(self):# # #{
        T_global_to_fcu = lookupTransformAsMatrix(self.odom_frame, self.fcu_frame, self.tf_listener)
        T_smap_origin_to_fcu = np.linalg.inv(self.mapper.spheremap.T_global_to_own_origin) @ T_global_to_fcu
        current_vp_smap_frame = Viewpoint(T=T_smap_origin_to_fcu) 

        if self.roadmap_navigation_failed:
            print("ROADMAP NAV TO EXPLORATION GLOBAL GOAL FAILED, SWITCHING TO LOCAL EXPLO")
            self.stop_following_roadmap_and_stop()
            self.exploration_state == 'local'
            return

        elif self.roadmap_navigation_success:
            # TODO - MARK NEARBY VPS AS VISITED
            pass

        if self.roadmap is None:
            # FIND A* PATHS TO ALL KNOWN GOALS
            global_goal_roadmap = self.getBestGoalRoadmap()

            # IF NONE FOUND, SWITCH TO LOCAL SEARCH AND TRY SEARCHING HERE (todo -increase counter)
            if global_goal_roadmap is None:
                self.tryAddingGoalsNearby()
                self.visualize_exploration_goals()
                # TODO - search for nearby goals, N times
                # TODO - counter++
                return
            print("EXPLORATION - SETTING NEW ROADMAP TO GOAL!!!")

            # IF SOME IS GOOD, SET IT AS NAVGOAL AND FUCKING GO TO IT
            self.set_roadmap(global_goal_roadmap)

    # # #}

    # --UTILS

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
            line_marker.scale.x = 0.2 * self.marker_scale  # Line width
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

        self.path_planning_vis_pub.publish(marker_array)
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

    def visualize_trajectory(self,points_untransformed, T_vis, headings=None,do_line=True, start=None, goal=None, frame_id=None, pub=None):# # #{
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

        pub.publish(marker_array)
# # #}

    def visualize_exploration_goals(self, scale = 1):# # #{
        marker_array = MarkerArray()
        if self.exploration_goals is None:
            return
        frame_id = self.odom_frame

        for i in range(self.exploration_goals.size):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time.now()
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.id = i

            # Set the scale
            marker.scale.x = scale *0.4
            marker.scale.y = scale *2.0
            marker.scale.z = scale *1.0

            marker.color.a = 1
            marker.color.r = 1
            marker.color.g = 1
            marker.color.b = 0

            map_heading = self.exploration_goals[i].viewpoint.heading
            pt = self.exploration_goals[i].viewpoint.position

            arrowlen = 3
            xbonus = arrowlen * np.cos(map_heading)
            ybonus = arrowlen * np.sin(map_heading)
            points_msg = [Point(x=pt[0], y=pt[1], z=pt[2]), Point(x=pt[0]+xbonus, y=pt[1]+ybonus, z=pt[2])]
            marker.points = points_msg

            # Add the marker to the MarkerArray
            marker_array.markers.append(marker)

        self.exploration_goals_vis_pub.publish(marker_array)# # #}


