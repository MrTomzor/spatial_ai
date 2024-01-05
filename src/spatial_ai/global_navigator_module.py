
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

class GlobalNavigatorModule:
    def __init__(self, mapper, local_navigator):# # #{

        self.mapper = mapper
        self.local_navigator = local_navigator
        self.odom_frame = mapper.odom_frame
        self.fcu_frame = mapper.fcu_frame
        self.tf_listener = mapper.tf_listener

        # VIS PUB
        # self.path_planning_vis_pub = rospy.Publisher('path_planning_vis', MarkerArray, queue_size=10)
        # self.unsorted_vis_pub = rospy.Publisher('unsorted_markers', MarkerArray, queue_size=10)
        self.matching_result_vis = rospy.Publisher('map_matching_result_vis', MarkerArray, queue_size=10)


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

    def main_iter(self):# # #{
        print("N SUBMAPS IN OLD MAP:")
        print(len(self.test_mchunk.submaps))

        mchunk1 = self.mapper.mchunk
        mchunk2 = self.test_mchunk
        
        start1 = len(mchunk1.submaps) - 1
        if star1 < 0:
            print("NOT ENOUGH SUBMAPS IN CURRENT MAP")
            return
        
        start2 = len(mchunk2.submaps) - 1
        if start2 < 0:
            print("NOT ENOUGH SUBMAPS IN OLD MAP")
            return

        max_submaps = 3
        idxs1, transforms1 = getConnectedSubmapsWithTransforms(mchunk1, start1, max_submaps)
        idxs2, transforms2 = getConnectedSubmapsWithTransforms(mchunk2, start2, max_submaps)

        print("N MAPS FOR MATCHING IN CHUNK1: " + str(len(idxs1)))
        print("N MAPS FOR MATCHING IN CHUNK2: " + str(len(idxs2)))

        # SCROUNGE ALL MAP MATCHING DATA
        matching_data1 = getMapMatchingDataSimple(mchunk1, idxs1, transforms1)
        matching_data2 = getMapMatchingDataSimple(mchunk2, idxs2, transforms2)

        # PERFORM MATCHING!
        T_res, score_res = matchMapGeomSimple(matching_data1, matching_data2)

        # VISUALIZE MATCH OVERLAP!!
        print("MATCHING DONE!!!")
        # T_odom_chunk1 = mchunk1.sumaps[start1].T_global_to_own_origin
        # T_vis_chunk1 = [T_odom_chunk1 @ tr for tr in transforms1]

        T_odom_chunk2 = mchunk2.sumaps[start2].T_global_to_own_origin
        T_vis_chunk2 = [T_odom_chunk1 @ T_res @ tr for tr in transforms2]

        marker_array = MarkerArray()
        for i in range(len(idxs2)):
            mapper.get_spheremap_marker_array(marker_array, mchunk2.submaps[idxs2[i]], T_vis_chunk2[i], alternative_look = True, do_connections = False, do_surfels = True, do_spheres = False, do_map2map_conns=False, ms=self.mapper.marker_scale, clr_index = 0, alpha = 0.5)

        self.matching_result_vis.publish(marker_array)

    # # #}


