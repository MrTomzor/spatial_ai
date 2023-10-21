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
    return res.T

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

class SphereMap:
    def __init__(self, init_radius, min_radius):
        self.points = np.array([0,0,0]).reshape((1,3))
        # self.radii = np.array([init_radius])
        self.radii = np.array([init_radius]).reshape((1,1))
        self.connections = np.array([[]])

        self.min_radius = min_radius
        self.max_radius = init_radius
    
    def update_in_fov(self, cam_T, visible_obstacle_pts, cam_K):
        # visible_obstacle_pts = visible obstacle points, position is relative to current pose of camera
        # cam_T = transformation from origin of spheremap to current camera pose
        # cam_K = projection matrix of cam
        print("UPDATINGI IN FOV")

class OdomNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.prev_image = None
        self.prev_time = None
        self.proper_triang = False

        self.spheremap = None
        self.keyframes = []
        self.noprior_triangulation_points = None
        self.odomprior_triangulation_points = None

        self.slam_points = None
        self.slam_pcl_pub = rospy.Publisher('extended_slam_points', PointCloud, queue_size=10)

        self.spheremap_spheres_pub = rospy.Publisher('spheres', MarkerArray, queue_size=10)

        self.kp_pub = rospy.Publisher('tracked_features_img', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('estim_depth_img', Image, queue_size=1)
        self.marker_pub = rospy.Publisher('/vo_odom', Marker, queue_size=10)
        self.kp_pcl_pub = rospy.Publisher('tracked_features_space', PointCloud, queue_size=10)
        self.kp_pcl_pub_invdepth = rospy.Publisher('tracked_features_space_invdepth', PointCloud, queue_size=10)

        self.sub_cam = rospy.Subscriber('/robot1/camera1/raw', Image, self.image_callback, queue_size=10000)

        self.odom_buffer = []
        self.odom_buffer_maxlen = 1000
        self.sub_odom = rospy.Subscriber('/ov_msckf/odomimu', Odometry, self.odometry_callback, queue_size=10000)
        self.sub_slam_points = rospy.Subscriber('/ov_msckf/points_slam', PointCloud2, self.points_slam_callback, queue_size=10000)
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

        self.tracked_features = []

    def get_closest_time_odom_msg(self, stamp):
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
        return bestmsg

    def odometry_callback(self, msg):
        self.odom_buffer.append(msg)
        if len(self.odom_buffer) > self.odom_buffer_maxlen:
            self.odom_buffer.pop(0)

    def integrate_slam_points_to_keyframe(self, point_cloud_array ):
        comp_start_time = time.time()

        n_new_points = point_cloud_array.size

        mindist = 1
        mindist2 = mindist * mindist

        if self.slam_points is None and n_new_points > 0:
            self.slam_points = point_cloud_array
        elif not self.slam_points is None:

            print("ADDED NEW")
            self.slam_points = np.concatenate((self.slam_points, point_cloud_array))
            distances = scipy.spatial.distance_matrix(self.slam_points, self.slam_points)
            print("HAVE DISTMATRIX")
            far_enough = distances > mindist2
            keep = np.array([False for i in range(self.slam_points.shape[0])])
            for i in range(self.slam_points.shape[0]):
                far_enough[i,i] = True
                keep[i] = np.all(far_enough[:, i])
            # idxs = np.any(distances > mindist2)
            print("HAVE KEEP")
            self.slam_points = self.slam_points[keep, :]
            print("HAVE NEW PTS")

        print("PCL MSG DONE")
        comp_time = time.time() - comp_start_time
        print("PCL integration time: " + str((comp_time) * 1000) +  " ms")
                    
    def get_distances_from_fov_borders(self, points):
        # vec1 = 
        invK = self.K
        ul = self.K

    def points_slam_callback(self, msg):
        print("PCL MSG")

        pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pc_list = []

        for point in pc_data:
            x, y, z = point
            pc_list.append([x, y, z])

        point_cloud_array = np.array(pc_list, dtype=np.float32)

        # INTEGRATE
        self.integrate_slam_points_to_keyframe(point_cloud_array)

        # ---SPHEREMAP STUFF
        if len(pc_list) == 0:
            return

        comp_start_time = time.time()
        # TRANSFORM SLAM PTS TO IMAGE AND COMPUTE THEIR PIXPOSITIONS
        if len(self.odom_buffer) == 0:
            return
        T_global_to_imu = self.odom_msg_to_transformation_matrix(self.odom_buffer[-1])
        transformed = transformPoints(point_cloud_array, (self.imu_to_cam_T @ np.linalg.inv(T_global_to_imu))).T

        # hom = np.concatenate((point_cloud_array.T, np.full((1, point_cloud_array.shape[0]), 1)))
        # transformed = (self.imu_to_cam_T @ np.linalg.inv(T_global_to_imu)) @ hom

        transformed = transformed / transformed [3, :] # unhomogenize

        positive_z_idxs = transformed[2, :] > 0
        final_points = transformed[:3, positive_z_idxs]

        pixpos = self.K @ final_points 
        pixpos = pixpos / pixpos[2, :]
        pixpos = pixpos[:2, :].T

        # COMPUTE DELAUNAY TRIANG OF VISIBLE SLAM POINTS
        print("HAVE DELAUNAY:")
        tri = Delaunay(pixpos)
        print(tri.simplices)

        vis = self.visualize_depth(pixpos, tri)
        self.depth_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))

        # CONSTRUCT OBSTACLE MESH
        obstacle_mesh = trimesh.Trimesh(vertices=final_points.T, faces = tri.simplices)

        # ---UPDATE AND PRUNING STEP
        T_current_cam_to_orig = np.eye(4)
        if not self.spheremap is None:
            # transform existing sphere points to current camera frame
            # T_spheremap_orig_to_current_cam = self.imu_to_cam_T @ T_global_to_imu @ np.linalg.inv(self.spheremap.T_global_to_own_origin) 
            # T_current_cam_to_orig = np.linalg.inv(T_spheremap_orig_to_current_cam)
            # T_spheremap_orig_to_current_cam = np.linalg.inv(self.imu_to_cam_T)
            # print("T DELTA ODOM")
            # print(T_delta_odom)
            # print("GLOBAL TO OWN ORIGIN")
            # print(self.spheremap.T_global_to_own_origin)
            # print("ORIG TO NOW")
            # print(T_spheremap_orig_to_current_cam )
            n_spheres_old = self.spheremap.points.shape[0]

            T_delta_odom = np.linalg.inv(self.spheremap.T_global_to_imu_at_start) @ T_global_to_imu
            T_current_cam_to_orig = self.imu_to_cam_T @ T_delta_odom @ np.linalg.inv(self.imu_to_cam_T)

            print("FINAL MATRIX")
            print(T_current_cam_to_orig)

            # project sphere points
            transformed_old_points  = transformPoints(self.spheremap.points, T_current_cam_to_orig)
            # print(" OLD IN SPHEREMAP ORIGIN")
            # print(self.spheremap.points)
            # print(" OLD IN CURRENT FRAME")
            # print(transformed_old_points.T)

            print("TRANSFORMED EXISTING SPHERE POINTS TO CURRENT CAMERA FRAME!")
            positive_z_idxs_old = transformed_old_points[:, 2] > 0

            print("POSITIVE Z:" + str(np.sum(positive_z_idxs_old)) + "/" + str(n_spheres_old))

        # ---EXPANSION STEP
        max_sphere_sampling_z = 60

        n_sampled = 20
        sampling_pts = np.random.rand(n_sampled, 2)  # Random points in [0, 1] range for x and y
        sampling_pts = sampling_pts   * [self.width, self.height]

        # CHECK THE SAMPLING DIRS ARE INSIDE THE 2D CONVEX HULL OF 3D POINTS
        hull = ConvexHull(pixpos)

        print("POLY")
        polygon = geometry.Polygon(hull.points)

        inhull = np.array([polygon.contains(geometry.Point(p[0], p[1])) for p in sampling_pts])
        print(inhull)
        if not np.any(inhull):
            print("NONE IN HULL")
            return
        sampling_pts = sampling_pts[inhull, :]

        n_sampled = sampling_pts.shape[0]

        # NOW PROJECT THEM TO 3D SPACE
        sampling_pts = np.concatenate((sampling_pts.T, np.full((1, n_sampled), 1)))

        invK = np.linalg.inv(self.K)
        sampling_pts = invK @ sampling_pts
        rand_z = np.random.rand(1, n_sampled) * max_sphere_sampling_z
        print("SPHERE SAMPLING PTS:" )
        print(sampling_pts)

        # FILTER PTS - CHECK THAT THE MAX DIST IS NOT BEHIND THE OBSTACLE MESH BY RAYCASTING
        ray_hit_pts, index_ray, index_tri = obstacle_mesh.ray.intersects_location(
        ray_origins=np.zeros(sampling_pts.shape).T, ray_directions=sampling_pts.T)
        print("RAYS")
        print(index_ray)

        # MAKE THEM BE IN RAND POSITION BETWEEN CAM AND MESH HIT POSITION
        sampling_pts =  (np.random.rand(n_sampled, 1) * ray_hit_pts).T
        # TODO - add max sampling dist


        orig_3dpt_indices_in_hull = np.unique(hull.simplices)
        # CONSTRUCT HULL MESH FROM 3D POINTS OF CONVEX 2D HULL OF PROJECTED POINTS
        hullmesh_pts = final_points[:, np.unique(hull.simplices)]
        orig_pts = np.zeros((3, 1))
        # print(hullmesh_pts.shape)
        # print(orig_pts.shape)
        hullmesh_pts = np.concatenate((hullmesh_pts, orig_pts), axis=1)
        zero_pt_index = hullmesh_pts.shape[1] - 1

        # FUCK IT JUST DO CONV HULL OF THESE 3D PTS(just start and hull pts)
        fov_hull = ConvexHull(hullmesh_pts.T)

        uniq_idxs_in_hull = np.unique(fov_hull.simplices)
        simplices_reindexing = {uniq_idxs_in_hull[i]:i for i in range(uniq_idxs_in_hull.size)}
        simplices_reindexed = [[simplices_reindexing[orig[0]], simplices_reindexing[orig[1]], simplices_reindexing[orig[2]]] for orig in fov_hull.simplices]

        # CONSTRUCT FOV MESH AND QUERY
        fov_mesh = trimesh.Trimesh(vertices=fov_hull.points, faces = simplices_reindexed)
        fov_mesh_query = trimesh.proximity.ProximityQuery(fov_mesh)

        # CONSTRUCT OBSTACLE POINT MESH AND QUERY
        obstacle_mesh_query = trimesh.proximity.ProximityQuery(obstacle_mesh)

        # TRY ADDING NEW SPHERES AT SAMPLED POSITIONS
        new_spheres_fov_dists = np.abs(fov_mesh_query.signed_distance(sampling_pts.T))
        new_spheres_obs_dists = np.abs(obstacle_mesh_query.signed_distance(sampling_pts.T))
        
        init_rad = 0.5
        min_rad = init_rad
        if self.spheremap is None:
            self.spheremap = SphereMap(init_rad, min_rad)
            self.spheremap.T_global_to_own_origin = T_global_to_imu @ self.imu_to_cam_T
            self.spheremap.T_global_to_imu_at_start = T_global_to_imu

        mindists = np.minimum(new_spheres_obs_dists, new_spheres_fov_dists)
        new_sphere_idxs = mindists > min_rad

        n_spheres_to_add = np.sum(new_sphere_idxs)
        print("PUTATIVE SPHERES THAT PASSED FIRST RADIUS CHECKS: " + str(n_spheres_to_add))
        if n_spheres_to_add == 0:
            return

        # print(mindists.shape)
        # print(mindists[new_sphere_idxs].shape)

        # TRANSFORM POINTS FROM CAM ORIGIN TO SPHEREMAP ORIGIN! - DRAW OUT!
        self.spheremap.points = np.concatenate((self.spheremap.points, sampling_pts[:, new_sphere_idxs].T))
        self.spheremap.radii = np.concatenate((self.spheremap.radii.flatten(), mindists[new_sphere_idxs].flatten()))

        # mindists = np.min(new_spheres_fov_dists , new_spheres_obs_dists )
        # print("MINDISTS:")
        # print(new_spheres_fov_dists )
        # print(new_spheres_obs_dists )

        # points = np.array([[0,0,5], [0, 0, 10], [0, 0, 15], [0, 0, 20], [0, 0, 25]])
        
        # print("DISTS:")
        # print(obstacle_mesh_query.signed_distance(points))

        # # CHECK DIST AGAINST FOV MESH (is mesh formed by camera pos and the obstacles (NARROWER THAN FOV)
        # distances_from_fov = self.get_distances_from_fov_borders(points)


        # CHECK RADII OF EXISTING SPHERES AGAINST THE FOV MESH (IF PROJECTED TO CAM)

        # pixpos = self.K @

        comp_time = time.time() - comp_start_time
        print("SPHEREMAP integration time: " + str((comp_time) * 1000) +  " ms")

        # VISUALIZE CURRENT SPHERES
        self.visualize_spheremap()


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


    def control_features_population(self):
        wbins = self.width // self.tracking_bin_width
        hbins = self.height // self.tracking_bin_width
        found_total = 0

        if self.px_cur is None or len(self.px_cur) == 0:
            self.px_cur = self.detector.detect(self.new_frame)
            if self.px_cur is None or len(self.px_cur) == 0:
                return
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
            self.tracking_stats = np.array([TrackingStat() for x in self.px_cur], dtype=object)

        print("STTAS:")
        print(self.tracking_stats.shape)
        print(self.px_cur.shape)

        self.px_cur_prev = copy.deepcopy(self.px_cur)
        self.px_cur = self.px_cur[:0, :]

        self.tracking_stats_prev = copy.deepcopy(self.tracking_stats)
        self.tracking_stats = self.tracking_stats[:0]

        n_culled = 0

        for xx in range(wbins):
            for yy in range(hbins):
                # count how many we have there and get the points in there:
                ul = np.array([xx * self.tracking_bin_width , yy * self.tracking_bin_width ])  
                lr = np.array([ul[0] + self.tracking_bin_width , ul[1] + self.tracking_bin_width]) 

                inidx = np.all(np.logical_and(ul <= self.px_cur_prev, self.px_cur_prev <= lr), axis=1)
                # print(inidx)
                inside_points = []
                inside_stats = np.array([], dtype=object)

                n_existing_in_bin = 0
                if np.any(inidx):
                    inside_points = self.px_cur_prev[inidx]
                    inside_stats = self.tracking_stats_prev[inidx]
                    n_existing_in_bin = inside_points.shape[0]

                # print(n_existing_in_bin )

                if n_existing_in_bin > self.max_features_per_bin:
                    # CUTOFF POINTS ABOVE MAXIMUM, SORTED BY AGE
                    ages = np.array([-pt.age for pt in inside_stats])
                    # idxs = np.arange(self.max_features_per_bin)
                    # np.random.shuffle(idxs)
                    idxs = np.argsort(ages)
                    surviving_idxs = idxs[:self.max_features_per_bin]
                    n_culled_this_bin = n_existing_in_bin - self.max_features_per_bin

                    self.px_cur = np.concatenate((self.px_cur, inside_points[surviving_idxs , :]))
                    self.tracking_stats = np.concatenate((self.tracking_stats, inside_stats[surviving_idxs ]))

                    # LET THE ONES WITH MANY DEPTH MEASUREMENTS LIVE
                    # print(n_culled_this_bin)
                    spared_inside_idxs = []
                    n_spared = 0
                    for i in range(n_culled_this_bin):
                        if inside_stats[idxs[self.max_features_per_bin + i]].invdepth_measurements > 1:
                            spared_inside_idxs.append(self.max_features_per_bin + i)
                            n_spared += 1
                    if n_spared > 0:
                        spared_inside_idxs = np.array(spared_inside_idxs)

                        self.px_cur = np.concatenate((self.px_cur, inside_points[idxs[spared_inside_idxs] , :]))
                        self.tracking_stats = np.concatenate((self.tracking_stats, inside_stats[idxs[spared_inside_idxs] ]))

                    n_culled += n_culled_this_bin - n_spared
                    # self.px_cur = np.concatenate((self.px_cur, inside_points[:self.max_features_per_bin, :]))

                elif n_existing_in_bin < self.min_features_per_bin:
                    # ADD THE EXISTING
                    if n_existing_in_bin > 0:
                        self.px_cur = np.concatenate((self.px_cur, inside_points))
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
                    self.px_cur = np.concatenate((self.px_cur, locally_found))
                    # self.tracking_stats = np.array([TrackingStat for x in locally_found], dtype=object)
                    self.tracking_stats = np.concatenate((self.tracking_stats, np.array([TrackingStat() for x in locally_found], dtype=object)))
                else:
                    # JUST COPY THEM
                    self.px_cur = np.concatenate((self.px_cur, inside_points))
                    # self.tracking_stats += inside_stats
                    self.tracking_stats = np.concatenate((self.tracking_stats, inside_stats))

        print("FOUND IN BINS: " + str(found_total) + " CULLED: " + str(n_culled))
        print("CURRENT FEATURES: " + str(self.px_cur.shape[0]))

        # FIND FEATS IF ZERO!
        if(self.px_cur.shape[0] == 0):
            print("ZERO FEATURES! FINDING FEATURES")
            self.px_cur = self.detector.detect(self.new_frame)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)

        
    def odom_msg_to_transformation_matrix(self, closest_time_odom_msg):
        odom_p = np.array([closest_time_odom_msg.pose.pose.position.x, closest_time_odom_msg.pose.pose.position.y, closest_time_odom_msg.pose.pose.position.z])
        odom_q = np.array([closest_time_odom_msg.pose.pose.orientation.x, closest_time_odom_msg.pose.pose.orientation.y,
            closest_time_odom_msg.pose.pose.orientation.z, closest_time_odom_msg.pose.pose.orientation.w])
        T_odom = np.eye(4)
        # print("R:")
        # print(Rotation.from_quat(odom_q).as_matrix())
        T_odom[:3,:3] = Rotation.from_quat(odom_q).as_matrix()
        T_odom[:3, 3] = odom_p
        return T_odom

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


        # GET ODOM MSG CLOSEST TO CURRENT IMG TIMESTAMP
        closest_time_odom_msg = self.get_closest_time_odom_msg(self.new_img_stamp)

        # IF YOU CAN - TRACK
        if not self.px_ref is None:
            print("BEFORE TRACKING: " + str(self.px_ref.shape[0]))
            self.px_ref, self.px_cur, self.tracking_stats = featureTracking(self.last_frame, self.new_frame, self.px_ref, self.tracking_stats)

            for i in range(self.px_ref.shape[0]):
                self.tracking_stats[i].age += 1
                self.tracking_stats[i].prev_points.append((self.px_ref[i,0], self.px_ref[i,1]))
                if len(self.tracking_stats[i].prev_points) > self.tracking_history_len:
                    self.tracking_stats[i].prev_points.pop(0)

            print("AFTER TRACKING: " + str(self.px_cur.shape[0]))

        keyframe_time_threshold = 0.15
        keyframe_distance_threshold = 2.5

        time_since_last_keyframe = None
        dist_since_last_keyframe = None

        # CHECK IF SHOULD ADD KEYFRAME FOR TRIANGULATION OF POINTS
        # TODO - and prev keyframe parallax condition
        if len(self.keyframes) > 0:
            time_since_last_keyframe = (self.new_img_stamp - self.keyframes[-1].img_timestamp).to_sec()
            p2 = closest_time_odom_msg.pose.pose.position
            p1 = self.keyframes[-1].odom_msg.pose.pose.position
            dist_since_last_keyframe = np.linalg.norm(np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z], dtype=np.float32))

        if self.px_ref is None or ( time_since_last_keyframe > keyframe_time_threshold and dist_since_last_keyframe > keyframe_distance_threshold):
            print("ATTEMPTING TO ADD NEW KEYFRAME! " + str(len(self.keyframes)) + ", dist: " + str(dist_since_last_keyframe) + ", time: " + str(time_since_last_keyframe))

            # NUMPIFY THE CURRENT ODOMETRY TRANSFORMATION MATRIX
            T_odom = self.odom_msg_to_transformation_matrix(closest_time_odom_msg)

            # IF YOU CAN - FIRST TRIANGULATE WITHOUT SCALE! - JUST TO DISCARD OUTLIERS
            can_triangulate = not self.px_cur is None
            if can_triangulate:
                print("TRIANGULATING WITH OWN ")

                # TRIANG BETWEEN PX_CUR AND THEIR PIXEL POSITIONS IN THE PREVIOUS KEYFRAME
                self.px_lkf = np.array([self.tracking_stats[i].prev_keyframe_pixel_pos for i in range(self.px_cur.shape[0])], dtype=np.float32)

                E, mask = cv2.findEssentialMat(self.px_cur, self.px_lkf, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                mask = (mask > 0.5).flatten()
                self.px_cur = self.px_cur[mask, :]
                self.px_lkf = self.px_lkf[mask, :]
                self.tracking_stats = self.tracking_stats[mask]
                print("AFTER OUTLIER REJECTION: " + str(self.px_cur.shape[0]))

                # FIND TRANSFORMATION WITHOUT ODOM PRIOR
                R, t = self.decomp_essential_mat(E, self.px_lkf, self.px_cur)
                self.noprior_triangulation_points = self.triangulated_points
                T_noprior = np.eye(4)
                T_noprior[:3, :3] = R
                T_noprior[:3, 3] = t


                # NOW USE THE TRANSF. MATRIX GIVEN BY OPENVINS ODOMETRY AND USE IT TO TRIANGULATE THE POINTS BETWEEN KEYFRAMES
                T_delta_odom = np.linalg.inv(self.keyframes[-1].T_odom) @ T_odom
                print("T_NOPRIOR:")
                print(T_noprior)
                print("T_delta_ODOM:")
                print(T_delta_odom)

                P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T_delta_odom)

                # Triangulate the 3D points
                hom_Q1 = cv2.triangulatePoints(self.P, P, self.px_lkf.T, self.px_cur.T)
                # Also seen from cam 2
                hom_Q2 = np.matmul(T_delta_odom, hom_Q1)

                # Un-homogenize
                uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
                uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

                # self.odomprior_triangulation_points = uhom_Q2
                self.odomprior_triangulation_points = uhom_Q2

                # USE NOPRIOR PTS AND JUST GET SCALE FROM THE ODOMETRY
                if(not np.any(np.isnan(np.linalg.norm(t)))):
                    scale_modifier = np.linalg.norm(T_delta_odom[:3,3]) / np.linalg.norm(t)
                    print("ORIG TRANSLATION SCALE: "  + str(np.linalg.norm(t)))
                    print("ODOM TRANSLATION SCALE: "  + str(np.linalg.norm(T_delta_odom[:3,3])))

                    print("SCALE MODIFIER: "  + str(scale_modifier))
                    self.noprior_triangulation_points = scale_modifier * self.noprior_triangulation_points 

                    # ADD DEPTH ESTIMATION TO THE POINT
                    for i in range(self.noprior_triangulation_points.shape[1]):
                        MINDEPTH = 0.01
                        depth_meas = np.linalg.norm(self.noprior_triangulation_points[:, i])
                        if depth_meas >= MINDEPTH:
                            invdepth_meas = 1.0 / depth_meas
                            self.tracking_stats[i].invdepth_buffer.append(invdepth_meas)
                            # self.invdepth_meas_sigma2 = 0.01
                            self.invdepth_meas_sigma2 = 0.001
                            self.invdepth_meas_sigma2_init = 0.01
                            if (self.tracking_stats[i].invdepth_measurements) == 0:
                                self.tracking_stats[i].invdepth_mean = invdepth_meas
                                self.tracking_stats[i].invdepth_sigma2 = self.invdepth_meas_sigma2_init
                            else:
                                self.tracking_stats[i].invdepth_mean = (self.tracking_stats[i].invdepth_mean * self.invdepth_meas_sigma2 + invdepth_meas * self.tracking_stats[i].invdepth_sigma2) / (self.invdepth_meas_sigma2 + self.tracking_stats[i].invdepth_sigma2) 
                                self.tracking_stats[i].invdepth_sigma2 = self.invdepth_meas_sigma2 * self.tracking_stats[i].invdepth_sigma2 / (self.invdepth_meas_sigma2 + self.tracking_stats[i].invdepth_sigma2)
                            print("--MEAS INDEX: " + str(self.tracking_stats[i].invdepth_measurements))
                            print("MEAS MEAS: " + str(invdepth_meas) )
                            print("ESTIM MEAN: " + str(self.tracking_stats[i].invdepth_mean) )
                            print("ESTIM COV: " + str(self.tracking_stats[i].invdepth_sigma2) )
                            avg = np.mean(np.array([x for x in self.tracking_stats[i].invdepth_buffer]))
                            print("ESTIM AVG: " + str(avg) )
                            self.tracking_stats[i].invdepth_measurements += 1
                            self.proper_triang = True


            # VISUALIZE THE TRIANGULATED SHIT
            if self.proper_triang:
                self.proper_triang = False
                self.visualize_keypoints_in_space(True)
                self.visualize_keypoints_in_space(False)



            # CONTROL FEATURE POPULATION - ADDING AND PRUNING
            self.control_features_population()

            # RETURN IF STILL CANT FIND ANY, NOT ADDING KEYFRAME
            if(self.px_cur is None):
                print("--WARNING! NO FEATURES FOUND!")
                return

            # HAVE ENOUGH POINTS, ADD KEYFRAME
            print("ADDED NEW KEYFRAME! KF: " + str(len(self.keyframes)))
            new_kf = KeyFrame(closest_time_odom_msg, self.new_img_stamp, T_odom)

            # ADD SLAM POINTS INTO THIS KEYFRAME (submap)
            new_kf.slam_points = self.slam_points
            self.slam_points = None

            self.keyframes.append(new_kf)

            # STORE THE PIXELPOSITIONS OF ALL CURRENT POINTS FOR THIS GIVEN KEYFRAME 
            for i in range(self.px_cur.shape[0]):
                self.tracking_stats[i].prev_keyframe_pixel_pos = self.px_cur[i, :]




        # if self.tracking_stats.shape[0] != self.px_cur.shape[0]:
        #     print("FUCK!")

        # if (not self.noprior_triangulation_points is None) and self.tracking_stats.shape[0] != self.noprior_triangulation_points.shape[1]:
        #     print("FUCK2!") # THIS HAPPENS BECAUSE POPULATION CONTROL IS AFTER THE FUCKING THING!

        # VISUALIZE FEATURES
        vis = self.visualize_tracking()
        self.kp_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))
        self.visualize_slampoints_in_space()

        comp_time = time.time() - comp_start_time
        print("computation time: " + str((comp_time) * 1000) +  " ms")


    def visualize_tracking(self):
        # rgb = np.zeros((self.new_frame.shape[0], self.new_frame.shape[1], 3), dtype=np.uint8)
        # print(self.new_frame.shape)
        rgb = np.repeat(copy.deepcopy(self.new_frame)[:, :, np.newaxis], 3, axis=2)
        # rgb = np.repeat((self.new_frame)[:, :, np.newaxis], 3, axis=2)

        if not self.px_cur is None:

            ll = np.array([0, 0])  # lower-left
            ur = np.array([self.width, self.height])  # upper-right
            inidx = np.all(np.logical_and(ll <= self.px_cur, self.px_cur <= ur), axis=1)
            inside_pix_idxs = self.px_cur[inidx].astype(int)

            # rgb[inside_pix_idxs[:, 1],inside_pix_idxs[:, 0], 0] = 255
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
                # prevpt = self.tracking_stats[inidx][i].prev_points[-1]
                # rgb = cv2.circle(rgb, (int(prevpt[0]), int(prevpt[1])), int(size), 
                #                (0, 255, 0), 3) 
                rgb = cv2.circle(rgb, (inside_pix_idxs[i,0], inside_pix_idxs[i,1]), int(size), 
                               (255, 0, 255), -1) 
                # rgb = cv2.circle(rgb, (inside_pix_idxs[i,0], inside_pix_idxs[i,1]), 5, 
                #                (255, 0, 255), 2) 

                # triang_pix = self.K.dot(self.triangulated_points[:, inidx][:, i])
                # triang_pix = triang_pix  / triang_pix[2]
                # rgb = cv2.line(rgb, (int(triang_pix[0]), int(triang_pix[1])), (inside_pix_idxs[i,0], inside_pix_idxs[i,1]), (255, 0, 0), 3)
                # rgb = cv2.circle(rgb, (int(triang_pix[0]), int(triang_pix[1])), int(size), 
                #                (0, 0, 255), 3) 

            if not self.noprior_triangulation_points is None:
                for i in range(self.noprior_triangulation_points .shape[1]):
                    pixpos = self.K.dot(self.noprior_triangulation_points [:, i])
                    pixpos = pixpos / pixpos[2]
                    rgb = cv2.circle(rgb, (int(pixpos[0]), int(pixpos[1])), minsize+growsize+2, 
                                   (0, 0, 255), 2) 

            if not self.odomprior_triangulation_points  is None:
                for i in range(self.odomprior_triangulation_points  .shape[1]):
                    pixpos = self.K.dot(self.odomprior_triangulation_points [:, i])
                    pixpos = pixpos / pixpos[2]
                    rgb = cv2.circle(rgb, (int(pixpos[0]), int(pixpos[1])), minsize+growsize+5, 
                                   (0, 255, 255), 2) 

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

    def visualize_spheremap(self):
        if self.spheremap is None:
            return

        marker_array = MarkerArray()

        # point_cloud = PointCloud()
        # point_cloud.header.stamp = rospy.Time.now()
        # point_cloud.header.frame_id = 'global'  # Set the frame ID according to your robot's configuration


        for i in range(self.spheremap.points.shape[0]):
            marker = Marker()
            marker.header.frame_id = "cam0"  # Change this frame_id if necessary
            marker.header.stamp = rospy.Time.now()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = i

            # Set the position (sphere center)
            marker.pose.position.x = self.spheremap.points[i][0]
            marker.pose.position.y = self.spheremap.points[i][1]
            marker.pose.position.z = self.spheremap.points[i][2]

            # Set the scale (sphere radius)
            marker.scale.x = 2 * self.spheremap.radii[i]
            marker.scale.y = 2 * self.spheremap.radii[i]
            marker.scale.z = 2 * self.spheremap.radii[i]

            marker.color.a = 0.6
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            # Add the marker to the MarkerArray
            marker_array.markers.append(marker)

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
