
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
    if n_inliers == 0:
        return 0
    # return n_inliers / rmse
    # return n_inliers 
    return 1.0 / rmse 
    # return n_inliers / rmse 

class MultiMapMatch:# # #{
    def __init__(self, submap_idxs1, submap_idxs2, mchunk1, mchunk2):
        self.idxs1 = submap_idxs1
        self.idxs2 = submap_idxs2
        self.mchunk1 = mchunk1
        self.mchunk2 = mchunk2
        self.n_tries = 0
        self.score = 0

    def add_measurement(self, trans, n_inliers, rmse):
        if self.n_tries == 0 or map_match_score(n_inliers, rmse) > map_match_score(self.n_inliers, self.rmse):
            print("MEASUREMENT IS BETTER THAN PREV, MEAS INDEX: " + str(self.n_tries))
            self.n_inliers = n_inliers
            self.rmse = rmse
            self.trans = trans
            self.n_tries += 1
            self.score = map_match_score(n_inliers, rmse)
            # TODO - n offending pts in freespace

    def same_submaps(self, match2):
        if sorted(self.idxs1) == sorted(match2.idxs1) and sorted(self.idxs2) == sorted(match2.idxs2):
            return True
        return False
# # #}

class LocalizationParticle:# # #{
    def __init__(self, poses1, idxs1, poses2, idxs2, sum_odom_error):
        self.poses1 = poses1
        self.idxs1 = idxs1
        self.poses2 = poses2
        self.idxs2 = idxs2
        self.sum_odom_error = sum_odom_error
        self.n_assocs = np.sum(np.array([not a is None for a in idxs2]))
# # #}

class GlobalNavigatorModule:
    def __init__(self, mapper, local_navigator):# # #{
        self.multimap_matches = []
        self.localization_particles = None
        self.matches_mutex = threading.Lock()

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
        self.particle_vis = rospy.Publisher('particle_vis', MarkerArray, queue_size=10)


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

    def update_matches(self):# # #{
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
# # #}

    def main_iter(self):# # #{
        # UPDATE AND VISUALIZE MATCHES
        self.update_matches()
        self.visualize_matches()

        # SAMPLE NEW PARTICLES
        self.localization_particles = None
        self.sample_and_propagate_particle(max_submaps = 30, add=True)

        if not self.localization_particles is None:
            print("PUBLISHING PARTICLE MARKERS")
            marker_array = MarkerArray()
            self.get_particle_markers_detailed(marker_array, self.localization_particles[0])
            self.particle_vis.publish(marker_array)


        # n_particle_samples = 10
        # for i in range(n_particle_samples):
        #     self.sample_and_propagate_particle(max_submaps = 30, add=True)

        # if not self.localization_particles is None:
        #     n_kept_particles = 100
        #     if n_kept_particles > self.localization_particles:
        #         n_kept_particles = self.localization_particles


        #     # RANK PARTICLES
        #     particle_scores = np.array([len(p.poses1) for p in self.localization_particles])
        #     ranked_particles = np.argsort(-particle_scores)
        #     kept_particle_idxs = ranked_particles[:n_kept_particles]
        #     self.localization_particles = self.localization_particles[kept_particle_idxs] #SORTED

        #     # VIS PARTICLES
        #     cmap = plt.get_cmap('viridis')  # You can use other colormaps as well
        #     colors = [cmap(1.0 - (1.0 *i) / n_kept_particles)[:3] for i in range(n_kept_particles)]

        #     marker_array = MarkerArray()
        #     for i in range(n_kept_particles):
        #         self.get_particle_markers(marker_array, self.localization_particles[i], colors[i])
        #     self.particle_vis.publish(marker_array)


    # # #}

    def get_particle_markers_detailed(self, marker_array, particle):# # #{
        # VIS ASSOC MAPS IN THE SKY
        mchunk1 = self.mapper.mchunk
        mchunk2 = self.test_mchunk
        n_poses = len(particle.poses2)

        mchunk_centroid_odom = mchunk2.compute_centroid_odom()
        trans_vis = np.array([0, 0, 100])
        T_common = np.eye(4)
        T_common[:3, 3] = trans_vis - mchunk_centroid_odom

        # VIS ACTIVE MAPS
        for idx in range(len(mchunk2.submaps)):
            smap = mchunk2.submaps[idx]
            T_vis = T_common @ smap.T_global_to_own_origin 

            rgb = [0.3, 0.3, 0.3]
            alpha = 1
            # if idx in particle.idxs2:
            if idx == particle.idxs2[0]:
                # print("RED!")
                rgb = [1, 0, 0]
                alpha = 1
            elif idx in particle.idxs2:
                rgb = [1, 0.7, 0.2]
                alpha = 1
                # print("T_vis:")
                # print(T_vis)

            self.mapper.get_spheremap_marker_array(marker_array, smap, T_vis, alternative_look = True, do_connections = False, do_surfels = True, do_spheres = False, do_map2map_conns=False, do_centroids = False, ms=self.mapper.marker_scale, rgb = rgb, alpha = alpha)

        # VIS PATH OF PARTICLE in MAP2
        print("PATHS!")
        pts1 = []
        pts2 = []
        pts3 = []
        for i in range(n_poses):
            # T_odom1_map2 = particle.poses1[i] @ np.linalg.inv(particle.freshest_odom12_T) #TODO - STRETCHED
            # T_odom1_map2 = np.linalg.inv(particle.freshest_odom12_T) @ particle.poses1[i]
            T_odom1_map2 = particle.freshest_odom12_T @ particle.poses1[i] # TODO - SHAPE OK BUT WEIRD TRANSLATION (LATERAL)
            # T_odom1_map2 = particle.poses1[i] @ particle.freshest_odom12_T
            T_odom2_map2 = particle.poses2[i]

            vis1 = T_common @ T_odom1_map2
            vis2 = T_common @ T_odom2_map2
            vis3 = particle.poses1[i]

            # pts1.append(T_odom1_map2[:3, 3].flatten())
            # pts2.append(T_odom2_map2[:3, 3].flatten())
            pts1.append(vis1[:3, 3].flatten())
            pts2.append(vis2[:3, 3].flatten())
            pts3.append(vis3[:3, 3].flatten())

        pts1 = np.array(pts1)
        # print(pts1.shape)
        pts2 = np.array(pts2)
        pts3 = np.array(pts3)
        # print(pts1)
        # print(pts2)

        marker_id = marker_array.markers[-1].id+1

        line1 = getLineMarkerPts(pts1, 2, [0, 0, 1, 1], self.mapper.odom_frame, 0, ns='lines')
        line2 = getLineMarkerPts(pts2, 1.5, [1, 1, 0, 1], self.mapper.odom_frame, 1, ns='lines')
        line3 = getLineMarkerPts(pts3, 2, [0, 0, 1, 1], self.mapper.odom_frame, 2, ns='lines')
        marker_array.markers.append(line1)
        marker_array.markers.append(line2)
        marker_array.markers.append(line3)
        

        # print("N detailed markers:")
        # print(len(marker_array.markers))

        return
# # #}

    def get_particle_markers(self, marker_array, particles):# # #{
        return
# # #}

    def visualize_matches(self):# # #{
        # TODO - draw big map in the sky of other mchunk
        mchunk1 = self.mapper.mchunk
        mchunk2 = self.test_mchunk

        mchunk_centroid_odom = mchunk2.compute_centroid_odom()
        trans_vis = np.array([0, 0, -100])
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
        for match_index in range(len(self.multimap_matches)):
        # for match_data in self.multimap_matches:
            match_data = self.multimap_matches[match_index]
            found_ranking_idx = False
            idx1 = match_data.idxs1[0]
            idx2 = match_data.idxs2[0]
            score = map_match_score(match_data.n_inliers, match_data.rmse)
            # match_index = 

            for i in range(len(match_rankings)):
                if match_rankings[i][0] == idx1:
                    found_ranking_idx = True
                    match_rankings[i][1].append(score)
                    match_rankings[i][2].append(idx2)
                    match_rankings[i][3].append(match_index)
                    break
            if not found_ranking_idx:
                match_rankings.append([idx1, [score], [idx2], [match_index]])
        # NOW SORT FOR EACH IDX1
        for i in range(len(match_rankings)):
            match_rankings[i][1] = np.array(match_rankings[i][1])
            match_rankings[i][2] = np.array(match_rankings[i][2])
            match_rankings[i][3] = np.array(match_rankings[i][3])

            argsorted = np.argsort(-match_rankings[i][1])
            match_rankings[i][1] = match_rankings[i][1][argsorted]
            match_rankings[i][2] = match_rankings[i][2][argsorted]
            match_rankings[i][3] = match_rankings[i][3][argsorted]
        self.matches_ranked = match_rankings

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
            scores = ranking[1]
            # print(scores)
            maxscore = np.max(scores)
            minscore = np.min(scores)

            for i in range(n_vis):
                if i > max_vis_matches_per_idx1:
                    continue
                smap2 = mchunk2.submaps[ranking[2][i]]
                score = ranking[1][i]
                match_index = ranking[3][i]
                # T_res = np.linalg.inv(self.multimap_matches[match_index].trans)
                T_res = self.multimap_matches[match_index].trans

                T_vis1 = smap1.T_global_to_own_origin 
                centroid_trans1 = np.eye(4)
                centroid_trans1[:3, 3] = smap1.centroid
                T_vis1 = T_vis1 @ centroid_trans1

                T_vis2 = T_common @ smap2.T_global_to_own_origin 
                centroid_trans2 = np.eye(4)
                centroid_trans2[:3, 3] = smap2.centroid
                # T_vis2 = T_vis2 @ centroid_trans2
                # T_vis2 = T_vis2 @ centroid_trans2 @ T_res
                # T_vis2 = np.linalg.inv(T_res) @ (T_vis2 @ centroid_trans2)
                # T_vis2 =  T_vis2 @ np.linalg.inv(T_res)
                T_vis2 =  T_vis2 @ T_res

                rgb = [1,0,0, 1]
                if i > 0:
                    rgb = [0.2,0.2,0.8, 0.8]
                # draw line between them! (getlinemarker) thiccness related to score!
                # print("LINE MARKER:")
                pos1 = T_vis1[:3,3].flatten()
                pos2 = T_vis2[:3,3].flatten()
                if np.any(np.isnan(score)):
                    print("NAN SCORE!!!!")
                    score = 0.01
                    # TODO - relativize
                if relativize:
                    max_thicc = 2
                    if scores.size > 1 and (maxscore - minscore) > 0.0001:
                        score = max_thicc * (score - minscore) / (maxscore - minscore)
                    # else:
                    #     score = max_thicc
                if score < 0.01:
                    score = 0.01


                # print("SCORE: " + str(score))
                # print(pos1)
                # print(pos2)
                line_marker = getLineMarker(pos1, pos2, score, rgb, self.mapper.odom_frame, marker_id)
                # line_marker.ns = 'lines'
                marker_id += 1
                marker_array.markers.append(line_marker)

        self.multimap_matches_vis.publish(marker_array)

        return
# # #}

    def get_potential_matches(self, map1_idx):# # #{
        idxs = []
        scores = []
        trans = []

        for match in self.multimap_matches:
            if match.idxs1[0] == map1_idx:
                idxs.append(match.idxs2[0])
                scores.append(match.score)
                trans.append(match.trans)
        return idxs, scores, trans

        # matches_data = None
        # for mr in self.matches_ranked:
        #     if mr[0] == map1_idx:
        #         matches_data = mr
        #         break
        # if match_data is None:
        #     print("NO MATCHES FOR THIS MAP IDX YET!")
        #     return None, None, None
        # self.matches_ranked = match_rankings
# # #}

    def sample_and_propagate_particle(self, max_submaps, add=True):# # #{
        # TODO - lock matches mutex, matches cant change!!!

        mchunk1 = self.mapper.mchunk
        mchunk2 = self.test_mchunk

        n_maps_in_chunk1 = len(mchunk1.submaps)
        if n_maps_in_chunk1 < 2:
            print("NOT ENOUGH MAPS FOR PARTICLE PROPAGATION")
            return
        n_propagated_maps = n_maps_in_chunk1
        if n_propagated_maps > max_submaps:
            n_propagated_maps = max_submaps

        # ownmap_idxs = mchunk1.submaps[(n_maps_in_chunk1-n_propagated_maps-1):(n_maps_in_chunk1)]
        ownmap_idxs = np.arange(n_maps_in_chunk1-n_propagated_maps, n_maps_in_chunk1)
        print("N PROPAGATED MAPS:" + str(n_propagated_maps))
        print(ownmap_idxs)
        # ownmap_idxs.reverse()
        ownmap_idxs = np.flip(ownmap_idxs)

        print("PROPAGATING")
        localized = False

        own_poses = []
        assoc_poses = []
        submap_idx_associations = []
        odom2odom_transforms = []
        n_assoc_maps = 0
        last_odom2odom_T = None
        fresest_odom12_T = None

        for i in range(n_propagated_maps):
            map1_idx = ownmap_idxs[i]
            pose1_odom = mchunk1.submaps[map1_idx].T_global_to_own_origin

            # RETRIEVE potential matches with their absolute similarity scores
            potential_idxs, potential_scores, potential_transforms = self.get_potential_matches(map1_idx)
            n_potential_matches = len(potential_idxs)

            # IF ASSOCIATED BEFORE -- filter out matches taht would be too far from integrated pose in map2
            # TODO

            # CHOOSE some association or no association
            assoc_idx = None
            assoc_icp_trans = None
            if n_potential_matches > 0:
                prob_no_assoc = -1 # TODO - if bad similarity (all similarities same, no "firing", also allow low assoc. WHen disctinct -> likely assoc)

                # COMPUTE WEIGHTS FOR SELECTING MATCHES
                potential_scores = np.array(potential_scores)
                minscore = np.min(potential_scores)
                maxscore = np.max(potential_scores)
                score_span = (maxscore - minscore)
                if score_span > 0.0001 and np.random.rand() > prob_no_assoc:
                    norm_scores = (potential_scores - minscore) / score_span
                    norm_scores += 0.1
                    prob_distrib = norm_scores / np.sum(norm_scores)
                    print("PROB DISTRIB:")
                    print(prob_distrib)

                    # CHOOSE
                    choice_match_idx = np.random.choice(n_potential_matches, 1, p=prob_distrib)[0]
                    print("CHOSEN INDEX:")
                    print(choice_match_idx)
                    assoc_idx = potential_idxs[choice_match_idx]
                    assoc_icp_trans = potential_transforms[choice_match_idx]

            # IF ASSOCIATED NOW -- set pose in map2 to the associated pose
            T_cur_odom2odom = None
            pose2_odom = None # save none if not yet assoc, so that the arrays have same len!
            if not assoc_idx is None:
                pose2_odom = mchunk2.submaps[assoc_idx].T_global_to_own_origin
                pose2_odom = pose2_odom @ assoc_icp_trans 
                # pose2_odom_plus_icp = pose2_odom @ assoc_icp_trans 

                # pose2_odom_plus_icp = assoc_icp_trans @ pose2_odom  
                # T_cur_odom2odom = TODO
                if fresest_odom12_T is None:
                    # fresest_odom12_T = np.linalg.inv(pose1_odom) @ pose2_odom_plus_icp
                    # fresest_odom12_T = np.linalg.inv(pose1_odom) @ pose2_odom

                    # fresest_odom12_T =  pose2_odom @ np.linalg.inv(pose1_odom) #TODO - WORKS!!! kinda

                    fresest_odom12_T =  (pose2_odom) @ np.linalg.inv(pose1_odom)

                    # fresest_odom12_T = np.linalg.inv(pose2_odom_plus_icp) @ pose1_odom

                n_assoc_maps += 1
            else:
                print("NO ASSOC, BREAKING FOR NOW!")
                break
                # IF NOT ASSOCIATED NOW BUT BEFORE -- PROPAGATE the associated position in second map by own odom
                if n_assoc_maps > 0 and len(own_poses) > 1:
                    own_delta_odom = np.linalg.inv(own_poses[-2]) @ own_poses[-1]

                    transformed_delta_odom = np.linalg.inv(last_odom2odom_T @ own_poses[-2]) @ (last_odom2odom_T @own_poses[-1])

                    # pose2_odom = last_odom2odom_T @ own_delta_odom @ assoc_poses[-1]
                    pose2_odom = transformed_delta_odom @ assoc_poses[-1]

            # IF ASSOCIATED and ASSOCIATED BEFORE -- compute translation and rotation error! and accumulate it!
            own_poses.append(pose1_odom)
            assoc_poses.append(pose2_odom)
            submap_idx_associations.append(assoc_idx)
            odom2odom_transforms.append(T_cur_odom2odom)

        if n_assoc_maps == 0:
            print("ZERO ASSOCIATIONS! EXITING")
            return

        if add:
            p = LocalizationParticle(own_poses, ownmap_idxs, assoc_poses, submap_idx_associations, 0)
            p.freshest_odom12_T = fresest_odom12_T 

            if self.localization_particles is None:
                self.localization_particles = np.array([p], dtype=object)
            else:
                self.localization_particles = np.concatenate((self.localization_particles, p)) 
            print("PARTICLE ADDED!")

        # THIS IS HARD, LETS JUST ALWAYS ASSOCIATE!!

        # GO BACK THRU THE MAP FROM FIRST ASSOC AND PROPAGATE WHERE WE COULD BE NOW!! -- ALL THE POSES!! (i guess)
        # latest_assoc_path_index = None
        # for i in range(len(own_poses)):
        #     if not submap_idx_associations[i] is None:
        #         latest_assoc_path_index = i
        # print("LATEST ASSOC INDEX (going from now-submap) IS " + str(latest_assoc_path_index))
        # extrapolating_transform_map1 = np.linalg.inv(own_poses[latest_assoc_path_index]) @ own_poses[0]
        # extrapolating_transform_map1[:3,:3] = np.eye(3) # JUST TRANSLATION FOR EXTRAPOLATION

        # T_first_odom2odom = odom2odom_transforms[latest_assoc_path_index]
        # rotmatrix = np.eye(4)
        # rotmatrix[:3, :3] = T_first_odom2odom[:3,:3] #rot from odom1 to odm2
        # T_delta_odom_map2 = np.linalg.inv(rotmatrix) @ extrapolating_transform_map1

        # extrapolated_current_pose_map2 = T_delta_odom_map2 @ mchunk2.submaps[submap_idx_associations[latest_assoc_path_index]].T_global_to_own_origin 

# # #}
                    


