
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

def map_match_score(n_inliers, rmse):
    return n_inliers / rmse

class MultiMapMatch:
    def __init__(self, submap_idxs1, submap_idxs2, mchunk1, mchunk2):
        self.idxs1 = submap_idxs1
        self.idxs2 = submap_idxs2
        self.mchunk1 = mchunk1
        self.mchunk2 = mchunk2
        self.n_tries = 0

    def add_measurement(self, trans, n_inliers, rmse):
        if self.n_tries == 0 or map_match_score(n_inliers, rmse) > map_match_score(self.n_inliers, self.rmse):
            print("MEASUREMENT IS BETTER THAN PREV, MEAS INDEX: " + str(self.n_tries))
            self.n_inliers = n_inliers
            self.rmse = rmse
            self.trans = trans
            self.n_tries += 1
            # TODO - n offending pts in freespace

    def same_submaps(self, match2):
        if sorted(self.idxs1) == sorted(match2.idxs1) and sorted(self.idxs2) == sorted(match2.idxs2):
            return True
        return False

class GlobalNavigatorModule:
    def __init__(self, mapper, local_navigator):# # #{
        self.multimap_matches = []

        self.mapper = mapper
        self.local_navigator = local_navigator
        self.odom_frame = mapper.odom_frame
        self.fcu_frame = mapper.fcu_frame
        self.tf_listener = mapper.tf_listener

        # VIS PUB
        # self.path_planning_vis_pub = rospy.Publisher('path_planning_vis', MarkerArray, queue_size=10)
        # self.unsorted_vis_pub = rospy.Publisher('unsorted_markers', MarkerArray, queue_size=10)
        self.matching_result_vis = rospy.Publisher('map_matching_result_vis', MarkerArray, queue_size=10)
        self.multimap_matches_vis = rospy.Publisher('multimap_matches', MarkerArray, queue_size=10)


        self.marker_scale = 0.15
        self.path_step_size = 0.5
        # self.max_heading_change_per_m = np.pi / 10
        self.max_heading_change_per_m = np.pi / 6

        self.safety_replanning_trigger_odist = 0.2
        self.min_planning_odist = 0.2
        self.max_planning_odist = 2

        # # PREDICTED TRAJ
        # self.sub_predicted_trajectory = rospy.Subscriber(ptraj_topic, mrs_msgs.msg.MpcPredictionFullState, self.predicted_trajectory_callback, queue_size=10000)
        # self.predicted_trajectory_pts_global = None


        # LOAD OTHER MAP MCHUNK
        self.planning_enabled = rospy.get_param("global_nav/enabled")
        self.testing_mchunk_filename = rospy.get_param("global_nav/testing_mchunk_filename")
        mchunk_filepath = rospkg.RosPack().get_path('spatial_ai') + "/memories/" + self.testing_mchunk_filename
        self.test_mchunk = CoherentSpatialMemoryChunk.load(mchunk_filepath)

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
        self.reaching_dist = 1
        self.reaching_angle = np.pi/2

        self.max_goal_vp_pathfinding_times = 3
        self.current_goal_vp_pathfinding_times = 0

        self.fspace_bonus_mod = 2
        self.safety_weight = 5

        self.last_matching_time = None
        self.matching_min_interval = 1

        self.best_current_match = None
        self.best_current_match_score = None
        
        # # #}

    def test_matching(self):# # #{
        print("N SUBMAPS IN OLD MAP:")
        print(len(self.test_mchunk.submaps))

        mchunk1 = self.mapper.mchunk
        mchunk2 = self.test_mchunk
        
        start1 = len(mchunk1.submaps) - 1
        if start1 < 0:
            print("NOT ENOUGH SUBMAPS IN CURRENT MAP")
            return
        
        start2 = np.random.randint(0, len(mchunk2.submaps))
        # start2 = 0
        print("START2: " + str(start2))
        if start2 < 0:
            print("NOT ENOUGH SUBMAPS IN OLD MAP")
            return

        max_submaps = 20
        # TODO - check by SIZE (of radii of traveled dists!) rather than max submaps!!!

        idxs1, transforms1 = getConnectedSubmapsWithTransforms(mchunk1, start1, max_submaps)
        idxs2, transforms2 = getConnectedSubmapsWithTransforms(mchunk2, start2, max_submaps)

        print("N MAPS FOR MATCHING IN CHUNK1: " + str(len(idxs1)))
        print("N MAPS FOR MATCHING IN CHUNK2: " + str(len(idxs2)))
        if len(idxs1) == 0 or len(idxs2) == 0:
            print("NOT ENOUGH MAPS FOR MATCHING")
            return

        # SCROUNGE ALL MAP MATCHING DATA
        matching_data1 = getMapMatchingDataSimple(mchunk1, idxs1, transforms1)
        matching_data2 = getMapMatchingDataSimple(mchunk2, idxs2, transforms2)

        matching_data1 = copy.deepcopy(matching_data1)
        matching_data2 = copy.deepcopy(matching_data2)

        # PERFORM MATCHING!
        T_res, score_res = matchMapGeomSimple(matching_data1, matching_data2)
        if T_res is None:
            print("MATCHING FAILED!")
            return

        # VISUALIZE MATCH OVERLAP!!
        print("MATCHING DONE!!!")
        T_odom_chunk1 = mchunk1.submaps[start1].T_global_to_own_origin
        # T_vis_chunk1 = [T_odom_chunk1 @ tr for tr in transforms1]

        # T_odom_chunk2 = mchunk2.submaps[start2].T_global_to_own_origin
        T_vis_chunk2 = [T_odom_chunk1 @ T_res @ tr for tr in transforms2]
        # print(T_odom_chunk1)
        print("T_VIS:")

        print("MATCHING DATA INLIER RATIOS :")
        # print(matching_data1.submap_overlap_ratios)
        # print(matching_data2.submap_overlap_ratios)

        # VISUALIZE OVERLAYED MATCHING SUBMAPS
        marker_array = MarkerArray()
        for i in range(len(idxs2)):
            cmap = plt.get_cmap('viridis')  # You can use other colormaps as well
            rgba_color = cmap(matching_data2.submap_overlap_ratios[i])
            # rgba_color = cmap(1)
            rgb = rgba_color[:3]

            self.mapper.get_spheremap_marker_array(marker_array, mchunk2.submaps[idxs2[i]], T_vis_chunk2[i], alternative_look = True, do_connections = False, do_surfels = True, do_spheres = False, do_map2map_conns=False, ms=self.mapper.marker_scale, clr_index = 42, alpha = 1, rgb = rgb)
            # print("INLIER RATIO: " + str(matching_data2.submap_overlap_ratios[i]))
            # TODO - vis only the maps that were put into data!!!

        self.matching_result_vis.publish(marker_array)
# # #}

    def main_iter(self):# # #{
        print("N SUBMAPS IN OLD MAP:")
        print(len(self.test_mchunk.submaps))

        mchunk1 = self.mapper.mchunk
        mchunk2 = self.test_mchunk
        
        # start1 = len(mchunk1.submaps) - 1
        if len(mchunk1.submaps) == 0:
            print("NO SUBMAPS IN MCHUNK1")
            return
        start1 = np.random.randint(0, len(mchunk1.submaps))
        print("START1: " + str(start1))
        
        if len(mchunk2.submaps) == 0:
            print("NO SUBMAPS IN MCHUNK1")
            return
        start2 = np.random.randint(0, len(mchunk2.submaps))
        print("START2: " + str(start2))

        max_submaps = 3
        # TODO - check by SIZE (of radii of traveled dists!) rather than max submaps!!!

        idxs1, transforms1 = getConnectedSubmapsWithTransforms(mchunk1, start1, max_submaps)
        idxs2, transforms2 = getConnectedSubmapsWithTransforms(mchunk2, start2, max_submaps)


        print("N MAPS FOR MATCHING IN CHUNK1: " + str(len(idxs1)))
        print("N MAPS FOR MATCHING IN CHUNK2: " + str(len(idxs2)))
        if len(idxs1) == 0 or len(idxs2) == 0:
            print("NOT ENOUGH MAPS FOR MATCHING")
            return

        # SCROUNGE ALL MAP MATCHING DATA
        matching_data1 = getMapMatchingDataSimple(mchunk1, idxs1, transforms1)
        matching_data2 = getMapMatchingDataSimple(mchunk2, idxs2, transforms2)

        matching_data1 = copy.deepcopy(matching_data1)
        matching_data2 = copy.deepcopy(matching_data2)

        # PERFORM MATCHING!
        T_res, n_inliers, rmse = matchMapGeomSimple(matching_data1, matching_data2)
        if T_res is None:
            print("MATCHING FAILED!")
            return

        # COMPUTE PERCENTAGE OF INLIERS wrt THE SMALLER MAP
        n_inliers = n_inliers / np.min(np.array([matching_data1.surfel_pts.shape[0], matching_data2.surfel_pts.shape[0]]))

        # VISUALIZE MATCH OVERLAP!!
        print("MATCHING DONE!!!")
        T_odom_chunk1 = mchunk1.submaps[start1].T_global_to_own_origin
        T_vis_chunk2 = [T_odom_chunk1 @ T_res @ tr for tr in transforms2]

        print("MATCHING DATA INLIER RATIOS :")
        # print(matching_data1.submap_overlap_ratios)
        # print(matching_data2.submap_overlap_ratios)

        # VISUALIZE OVERLAYED MATCHING SUBMAPS
        marker_array = MarkerArray()
        for i in range(len(idxs2)):
            cmap = plt.get_cmap('viridis')  # You can use other colormaps as well
            rgba_color = cmap(matching_data2.submap_overlap_ratios[i])
            # rgba_color = cmap(1)
            rgb = rgba_color[:3]

            self.mapper.get_spheremap_marker_array(marker_array, mchunk2.submaps[idxs2[i]], T_vis_chunk2[i], alternative_look = True, do_connections = False, do_surfels = True, do_spheres = False, do_map2map_conns=False, ms=self.mapper.marker_scale, clr_index = 42, alpha = 1, rgb = rgb)
            # print("INLIER RATIO: " + str(matching_data2.submap_overlap_ratios[i]))
            # TODO - vis only the maps that were put into data!!!
        self.matching_result_vis.publish(marker_array)

        # STORE MATCH RESULT!
        # new_match = MultiMapMatch(idxs1, idxs2, mchunk1, mchunk2)
        print("RMSE:" + str(rmse))
        print("N_INLIERS:" + str(n_inliers))
        print("-> SCORE: " + str(map_match_score(n_inliers, rmse)))
        new_match = MultiMapMatch([start1], [start2], mchunk1, mchunk2)
        similar_match = None
        for match in self.multimap_matches:
            if new_match.same_submaps(match):
                similar_match = match
                break
        if similar_match is None:
            print("INITING NEW MATCH!")
            new_match.add_measurement(T_res, n_inliers, rmse)
            self.multimap_matches.append(new_match)
        else:
            print("SIMILAR MATCH FOUND!")
            similar_match.add_measurement(T_res, n_inliers, rmse)

        # VISUALIZE MATCHES (OR PARTICLE FILTER TIMEr!)
        self.visualize_matches()

    # # #}

    def visualize_matches(self):
        # TODO - draw big map in the sky of other mchunk
        mchunk1 = self.mapper.mchunk
        mchunk2 = self.test_mchunk

        mchunk_centroid_odom = mchunk2.compute_centroid_odom()
        trans_vis = np.array([0, 0, 100])
        T_common = np.eye(4)
        T_common[:3, 3] = trans_vis - mchunk_centroid_odom

        marker_array = MarkerArray()
        for smap in mchunk2.submaps:
            T_vis = T_common @ smap.T_global_to_own_origin 
            # self.mapper.get_spheremap_marker_array(marker_array, smap, T_vis, alternative_look = True, do_connections = False, do_surfels = False, do_spheres = False, do_map2map_conns=True, do_centroids = True, ms=self.mapper.marker_scale)
            self.mapper.get_spheremap_marker_array(marker_array, smap, T_vis, alternative_look = True, do_connections = False, do_surfels = True, do_spheres = False, do_map2map_conns=False, do_centroids = False, ms=self.mapper.marker_scale, alpha=0.5)
            # self.mapper.get_spheremap_marker_array(marker_array, smap, T_vis, alternative_look = True, do_connections = False, do_surfels = False, do_spheres = False, do_map2map_conns=True, do_centroids = True, ms=self.mapper.marker_scale, alpha=1)


        # TODO - compute best score out of all matches, normalize scores OR just abs scores

        # COMPUTE RANKED MATCHES FOR EACH SUBMAP IN CHUNK1
        print("SORTING MATCH RANKINGS")
        match_rankings = []
        for match_data in self.multimap_matches:
            found_ranking_idx = False
            idx1 = match_data.idxs1[0]
            idx2 = match_data.idxs2[0]
            score = map_match_score(match_data.n_inliers, match_data.rmse)

            for i in range(len(match_rankings)):
                if match_rankings[i][0] == idx1:
                    found_ranking_idx = True
                    match_rankings[i][1].append(score)
                    match_rankings[i][2].append(idx2)
                    break
            if not found_ranking_idx:
                match_rankings.append([idx1, [score], [idx2]])
        # NOW SORT FOR EACH IDX1
        for i in range(len(match_rankings)):
            match_rankings[i][1] = np.array(match_rankings[i][1])
            match_rankings[i][2] = np.array(match_rankings[i][2])

            argsorted = np.argsort(-match_rankings[i][1])
            match_rankings[i][1] = match_rankings[i][1][argsorted]
            match_rankings[i][2] = match_rankings[i][2][argsorted]

        max_vis_matches_per_idx1 = 3

        marker_id = marker_array.markers[-1].id+1
        # for match in self.multimap_matches:
        for ranking in match_rankings:
            # find transform of other smap in current odom frame (in the big map above)
            # smap1 = mchunk1.submaps[match.idxs1[0]]
            # smap2 = mchunk2.submaps[match.idxs2[0]]
            # score = map_match_score(match.n_inliers, match.rmse)
            smap1 = mchunk1.submaps[ranking[0]]
            # n_vis = ranking[1].size if ranking[1].size < max_vis_matches_per_idx1 else max_vis_matches_per_idx1
            n_vis = ranking[1].size
            relativize = True

            for i in range(n_vis):
                smap2 = mchunk2.submaps[ranking[2][i]]
                score = ranking[1][i]

                T_vis1 = smap1.T_global_to_own_origin 
                centroid_trans1 = np.eye(4)
                centroid_trans1[:3, 3] = smap1.centroid
                T_vis1 = T_vis1 @ centroid_trans1

                T_vis2 = T_common @ smap2.T_global_to_own_origin 
                centroid_trans2 = np.eye(4)
                centroid_trans2[:3, 3] = smap2.centroid
                T_vis2 = T_vis2 @ centroid_trans2

                rgb = [1,0,0, 1]
                if i > max_vis_matches_per_idx1:
                    rgb = [0.2,0.2,0.8, 0.8]
                # draw line between them! (getlinemarker) thiccness related to score!
                # print("LINE MARKER:")
                pos1 = T_vis1[:3,3].flatten()
                pos2 = T_vis2[:3,3].flatten()
                if score < 0.01:
                    score = 0.01
                if np.any(np.isnan(score)):
                    print("NAN SCORE!!!!")
                    score = 0.01
                    # TODO - relativize

                # print("SCORE: " + str(score))
                # print(pos1)
                # print(pos2)
                line_marker = getLineMarker(pos1, pos2, score, rgb, self.mapper.odom_frame, marker_id)
                # line_marker.ns = 'lines'
                marker_id += 1
                marker_array.markers.append(line_marker)

        self.multimap_matches_vis.publish(marker_array)

        return


