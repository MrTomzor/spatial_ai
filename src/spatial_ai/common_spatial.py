#!/usr/bin/env python

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
import scipy
import pickle
import rospy
import tf.transformations as tfs
from scipy.spatial.transform import Rotation as R


# common utils# #{


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


def getPixelPositions(pts, K):
    # pts = 3D points u wish to project
    pixpos = K @ pts 
    pixpos = pixpos / pixpos[2, :]
    return pixpos[:2, :].T

def getVisiblePoints(pts, normals, max_angle, max_dist, w, h, K, verbose=False, check_normals=True):
    n_pts = pts.shape[0]
    pixpos = getPixelPositions(pts.T, K)
    inside_fov = np.logical_and(np.logical_and(pixpos[:, 0] > 0, pixpos[:, 1] > 0), np.logical_and(pixpos[:, 0] < w, pixpos[:, 1] < h))
    inside_fov = np.logical_and(pts[:, 2] > 0, inside_fov)
    
    if verbose:
        print("PTS: " + str(n_pts))
        print("INSIDE_FOV: " + str(np.sum(inside_fov)))

    dists = np.linalg.norm(pts, axis=1)
    pt_dirs = pts / dists.reshape((dists.size, 1))

    ok_angles = np.full(inside_fov.shape, True)
    if check_normals:
        arccos = np.sum(np.multiply(pt_dirs, -normals), axis = 1) / 2
        min_arccos = np.cos(max_angle)
        ok_angles = arccos > min_arccos

        if verbose:
            print("MIN ARCCOS:")
            print(min_arccos)
            print("VECS:")
            print(pt_dirs)
            print(normals)
            print(np.multiply(pt_dirs, normals))

            print("DOTPRODS:")
            print(arccos)
            print("OK ANGLES: " + str(np.sum(ok_angles)))

    ok_dists = dists < max_dist
    if verbose:
        print("OK DISTS: " + str(np.sum(ok_dists)))

    visible_pts_mask = np.logical_and(inside_fov, np.logical_and(ok_dists, ok_angles))
    return visible_pts_mask 



def lookupTransformAsMatrix(frame1, frame2, tf_listener):# # #{
    (trans, rotation) = tf_listener.lookupTransform(frame1, frame2, rospy.Time(0)) #Time0 = latest

    rotation_matrix = tfs.quaternion_matrix(rotation)
    res = np.eye(4)
    res[:3, :3] = rotation_matrix[:3,:3]
    res[:3, 3] = trans
    return res# # #}

class EmptyClass(object):
    pass

def transformationMatrixToHeading(T):
    return np.arctan2(T[1,0], T[0, 0])

def headingToTransformationMatrix(h):
    return R.from_euler('z', h, degrees=False).as_matrix()

def posAndHeadingToMatrix(pos, h):
    res = np.eye(4)
    res[:3, 3] = pos
    res[:3, :3] = headingToTransformationMatrix(h)
    return res

def transformPoints(pts, T):
    # pts = NxM matrix, T = transformation matrix to apply
    M = pts.shape[1]
    res = np.concatenate((pts.T, np.full((1, pts.shape[0]), 1)))
    res = T @ res 
    res = res / res[M, :] # unhomogenize
    return res[:M, :].T

def transformViewpoints(pts, headings, T):
    # pts = Nx3 matrix, T = transformation matrix to apply
    res = np.concatenate((pts.T, np.full((1, pts.shape[0]), 1)))
    res = T @ res 
    res = res / res[3, :] # unhomogenize

    hdif = transformationMatrixToHeading(T)
    res_heads = np.unwrap(headings + hdif)

    return res[:3, :].T, res_heads

def check_points_in_box(points, bounds):
    """
    Check if all 3D points fall within the specified box.

    Parameters:
    - points: NumPy array of shape (N, 3) representing 3D points.
    - x1, x2: Range of x values for the box.
    - y1, y2: Range of y values for the box.
    - z1, z2: Range of z values for the box.

    Returns:
    - A boolean array indicating whether each point is inside the box.
    """
    x1 = bounds[0]
    x2 = bounds[1]
    y1 = bounds[2]
    y2 = bounds[3]
    z1 = bounds[4]
    z2 = bounds[5]
    within_x = np.logical_and(points[:, 0] >= x1, points[:, 0] <= x2)
    within_y = np.logical_and(points[:, 1] >= y1, points[:, 1] <= y2)
    within_z = np.logical_and(points[:, 2] >= z1, points[:, 2] <= z2)

    # Combine the conditions for all three dimensions
    points_within_box = np.logical_and(np.logical_and(within_x, within_y), within_z)

    return points_within_box

class Viewpoint(object):
    def __init__(self, position, heading=None):
        self.position = position
        self.heading = heading
        self.use_heading = not (heading is None)
# # #}

class SubmapKeyframe:# # #{
    def __init__(self, T):
        self.position = T[:3,3]
        self.heading = transformationMatrixToHeading(T)
        # print("HEADING")
        # print(T)
        # print(self.heading)

    def euclid_dist(self, other_kf): 
        return np.linalg.norm(self.position - other_kf.position)

    def heading_dif(self, other_kf): 
        dif = np.abs(other_kf.heading - self.heading)
        if dif > 3.14159/2:
            dif = 3.14159 - dif
        return dif
# # #}

class MapToMapConnection:
    def __init__(self, pt_in_first_map_frame, second_map_id, rad):
        self.pt_in_first_map_frame = pt_in_first_map_frame
        self.second_map_id = second_map_id
        self.radius_at_creation = rad

# #{ class SphereMap
class SphereMap:
    def __init__(self, init_radius, min_radius):# # #{
        self.spheres_kdtree = None
        self.points = np.array([0,0,0]).reshape((1,3))
        self.radii = np.array([init_radius]).reshape((1,1))
        self.connections = np.array([None], dtype=object)
        self.visual_keyframes = []
        self.traveled_context_distance = 0
        self.surfels_filtering_radius = 1
        self.map2map_conns = []

        self.surfels_kdtree = None
        self.surfel_points = None
        self.surfel_slam_ids = None
        self.surfel_radii = None
        self.surfel_normals = None
        self.connectivity_labels = None

        self.frontier_points = None
        self.frontier_normals = None

        self.min_radius = min_radius
        self.max_radius = init_radius# # #}

    def getPointOfConnectionToSubmap(self, idx):
        for conn in self.map2map_conns:
            print(conn.second_map_id)
            if conn.second_map_id == idx:
                return conn.pt_in_first_map_frame
        return None

    def updateFrontiers(self, fr_samples):# # #{
        # SIMPLEST: 
        n_test_new_pts = fr_samples.shape[0]

        filtering_radius = 2

        # FILTER OUT THE NEW ONES TOO CLOSE TO PREVIOUS ONES
        pts_survived_first_mask = np.full((n_test_new_pts), True)
        if not self.frontier_points is None:
            n_existign_points = self.frontier_points.shape[0]
            existing_points_distmatrix = scipy.spatial.distance_matrix(fr_samples, self.frontier_points)
            pts_survived_first_mask = np.sum(existing_points_distmatrix > filtering_radius, axis=1) == n_existign_points

        pts_survived_filter_with_mappoints = fr_samples[pts_survived_first_mask, :]
        n_survived_first_filter = pts_survived_filter_with_mappoints.shape[0]

        # pts_added_mask = np.full((n_survived_first_filter), False)
        pts_added_mask = np.full((n_survived_first_filter), True)
        n_added = np.sum(pts_added_mask)
        print("N FRONTIERS SAMPLED: " +str(n_test_new_pts))
        print("N FRONTIERS FAR ENOUGH FROM OLD: " +str(n_survived_first_filter))
        self_distmatrix = scipy.spatial.distance_matrix(pts_survived_filter_with_mappoints, pts_survived_filter_with_mappoints)

        # ADD NEW PTS IF NOT BAD AGAINST OTHERS OR AGAINST DISTMATRIX

        # n_added = 0
        # for i in range(n_survived_first_filter):
        #     if n_added == 0 or np.all(np.sum(self_distmatrix[pts_added_mask, :] > filtering_radius, axis=1) == n_survived_first_filter - 1):
        #         n_added += 1
        #         pts_added_mask[i] = True

        if n_added > 0:
            if self.frontier_points is None:
                self.frontier_points = pts_survived_filter_with_mappoints[pts_added_mask]
            else:
                self.frontier_points = np.concatenate((self.frontier_points, pts_survived_filter_with_mappoints[pts_added_mask]))
        print("N ADDED: " +str(n_added))

        # NOW CHECK ALL FRONTIERS IF DEEP ENOUGH IN FREESPACE -> DELETE THEM
        deleting_inside_dist = 0
        toofar_from_spheres_dist = 2
        affection_distance = self.max_radius + toofar_from_spheres_dist 

        max_n_affecting_spheres = 5

        skdtree_query = self.spheres_kdtree.query(self.frontier_points, k=max_n_affecting_spheres, distance_upper_bound=affection_distance)
        n_spheres = self.points.shape[0]

        keep_frontiers_mask = np.full((self.frontier_points.shape[0]), True)
        self.frontier_normals = np.zeros((self.frontier_points.shape[0], 3))

        # CHECK IF NOT DELETED BY SPHERE AND COMPUTE NORMAL IF NOT
        for i in range(self.frontier_points.shape[0]):
            # found_neighbors = np.array([x for x in skdtree_query[1][i] if x != n_spheres])

            found_neighbors = skdtree_query[1][i]
            existing_mask = found_neighbors != n_spheres
            # print("FOUND NEIGHBORS: " + str(found_neighbors.shape[0]))
            # print(skdtree_query[1][i])
            # print(skdtree_query[0][i])

            n_considered = np.sum(existing_mask)
            if n_considered == 0:
                keep_frontiers_mask[i] = False
                continue

            found_dists = skdtree_query[0][i][existing_mask]
            found_sphere_indicies = skdtree_query[1][i][existing_mask]
            found_radii = self.radii[found_sphere_indicies]

            # CHECK IF DELETED BY THE SPHERES
            dists_from_spheres_edge = found_dists - found_radii
            if np.any(np.logical_or(dists_from_spheres_edge < deleting_inside_dist, dists_from_spheres_edge > toofar_from_spheres_dist)):
            # if np.any(dists_from_spheres_edge < deleting_inside_dist):
                keep_frontiers_mask[i] = False
                continue

            # CHECK IF NOT TOO CLOSE TO SURFELS
            # surf_min_dist = filtering_radius * 2
            surf_dist = np.min(np.linalg.norm(self.surfel_points - self.frontier_points[i,:], axis=1))
            if surf_dist < filtering_radius:
                keep_frontiers_mask[i] = False
                continue

            # CHECK IF CLEAR NORMAL (disallow small pts in nothingness
            normals = (self.points[found_sphere_indicies, :] - self.frontier_points[i].reshape(1,3)) / found_dists.reshape(n_considered, 1)
            fr_normal = (np.sum(normals, axis = 0) / n_considered).reshape((3,1))
            fr_normal_mag = np.linalg.norm(fr_normal)
            fr_normal = fr_normal / fr_normal_mag

            if np.any((normals).dot(fr_normal)) <= 0:
                keep_frontiers_mask[i] = False
                continue
            if fr_normal[2, 0] < -0.2:
                keep_frontiers_mask[i] = False
                continue

            self.frontier_normals[i, :] = fr_normal.flatten()


        # SAVE NEW PTS AND THEIR NORMALS
        n_keep = np.sum(keep_frontiers_mask)
        self.frontier_points = self.frontier_points[keep_frontiers_mask, :]
        # self.frontier_normals = self.frontier_normals[keep_frontiers_mask, :] / np.linalg.norm(self.frontier_normals[keep_frontiers_mask, :], axis=1).reshape(n_keep, 1)
        self.frontier_normals = self.frontier_normals[keep_frontiers_mask, :] 
        print("N KEEP: " +str(n_keep))
        print("N CUR FRONTIERS: " + str(self.frontier_points.shape[0]))


    # # #}

    def updateSurfels(self, visible_points, pixpos, simplices, slam_ids=None):# # #{
        # Compute normals measurements for the visible points, all should be pointing towards the camera
        # What frame are the points in?
        # print("UPDATING SURFELS")

        n_test_new_pts = visible_points.shape[0]

        # TODO do based on map scale!!!
        filtering_radius = self.surfels_filtering_radius


        # FIRST CHECK WHICH INPUT PTS ALREADY EXIST IN THE MAP AND JUST UPDATE THEIR POSITIONS
        if not self.surfel_slam_ids is None:
            are_new_pts_mask = np.full(n_test_new_pts, True)
            for i in range(n_test_new_pts):
                input_slam_id = slam_ids[i]
                findres = np.where(self.surfel_slam_ids == input_slam_id)[0]
                if findres.size == 0:
                    continue
                elif findres.size > 1:
                    print("ERROR! MORE SURFELS WITH SLAM ID " + str(input_slam_id))
                pt_idx_in_map_surfels = findres[0]

                # MOVE THE POINT
                self.surfel_points[pt_idx_in_map_surfels, :] = visible_points[i, :]
                are_new_pts_mask[i] = False
            n_are_new = np.sum(are_new_pts_mask)
            # print("MOVED PTS: " + str(n_test_new_pts - n_are_new))

            # REMOVE PTS FROM INPUT THAT WERE SEEN ALREADY
            visible_points = visible_points[are_new_pts_mask, :]
            if not slam_ids is None:
                slam_ids = slam_ids[are_new_pts_mask]
            n_test_new_pts = n_are_new


        # FILTER OUT THE NEW ONES TOO CLOSE TO PREVIOUS ONES
        pts_survived_first_mask = np.full((n_test_new_pts), True)
        if not self.surfel_points is None:
            n_existign_points = self.surfel_points.shape[0]
            # print("N EXISTING SURFELS: " + str(n_existign_points))
            existing_points_distmatrix = scipy.spatial.distance_matrix(visible_points, self.surfel_points)
            # pts_survived_first_mask = np.all(existing_points_distmatrix > filtering_radius, axis = 1)
            pts_survived_first_mask = np.sum(existing_points_distmatrix > filtering_radius, axis=1) == n_existign_points
            # print(np.sum(existing_points_distmatrix > filtering_radius, axis=1))

            # print("N SURVIVED FIRST MASK:" + str(np.sum(pts_survived_first_mask)))

        # pts_survived_filter_with_mappoints = visible_points[pts_survived_first_mask]
        pts_survived_filter_with_mappoints = visible_points[pts_survived_first_mask, :]
        if not slam_ids is None:
            slam_ids = slam_ids[pts_survived_first_mask]

        n_survived_first_filter = pts_survived_filter_with_mappoints.shape[0]
        # print("N SURVIVED FILTERING WITH OLD POINTS: " + str(n_survived_first_filter))

        pts_added_mask = np.full((n_survived_first_filter), False)
        self_distmatrix = scipy.spatial.distance_matrix(pts_survived_filter_with_mappoints, pts_survived_filter_with_mappoints)

        # ADD NEW PTS IF NOT BAD AGAINST OTHERS OR AGAINST DISTMATRIX
        n_added = 0
        for i in range(n_survived_first_filter):
            if n_added == 0 or np.all(np.sum(self_distmatrix[pts_added_mask, :] > filtering_radius, axis=1) == n_survived_first_filter - 1):
                n_added += 1
                pts_added_mask[i] = True

        if n_added > 0:
            if self.surfel_points is None:
                self.surfel_points = pts_survived_filter_with_mappoints[pts_added_mask]
                if not slam_ids is None:
                    self.surfel_slam_ids = slam_ids[pts_added_mask]
            else:
                self.surfel_points = np.concatenate((self.surfel_points, pts_survived_filter_with_mappoints[pts_added_mask]))
                if not slam_ids is None:
                    self.surfel_slam_ids = np.concatenate((self.surfel_slam_ids, slam_ids[pts_added_mask]))

        # COMPUTE POINT NORMALS AND DELETE THEM IF OCCUPIED BY SPHERES
        affection_distance = self.max_radius * 1.2
        deleting_inside_dist = -0.1

        max_n_affecting_spheres = 5
        skdtree_query = self.spheres_kdtree.query(self.surfel_points, k=max_n_affecting_spheres, distance_upper_bound=affection_distance)
        n_spheres = self.points.shape[0]
        # print("N SPHERES: " + str(n_spheres))

        self.surfels_that_have_normals = np.full((self.surfel_points.shape[0]), False)
        keep_surfels_mask = np.full((self.surfel_points.shape[0]), True)
        self.surfel_normals = np.zeros((self.surfel_points.shape[0], 3))

        # CHECK IF NOT DELETED BY SPHERE AND COMPUTE NORMAL IF NOT
        for i in range(self.surfel_points.shape[0]):
            # found_neighbors = np.array([x for x in skdtree_query[1][i] if x != n_spheres])

            found_neighbors = skdtree_query[1][i]
            existing_mask = found_neighbors != n_spheres
            # print("FOUND NEIGHBORS: " + str(found_neighbors.shape[0]))
            # print(skdtree_query[1][i])
            # print(skdtree_query[0][i])

            n_considered = np.sum(existing_mask)
            if n_considered == 0:
                continue

            found_dists = skdtree_query[0][i][existing_mask]
            found_sphere_indicies = skdtree_query[1][i][existing_mask]
            found_radii = self.radii[found_sphere_indicies]
            # found_neighbors = np.array(found_neighbors)

            # CHECK IF DELETED BY THE SPHERES
            dists_from_spheres_edge = found_dists - found_radii
            # print(dists_from_spheres_edge )

            # TODO figure better way to trigger this
            if slam_ids is None:
                if np.any(dists_from_spheres_edge < deleting_inside_dist):
                    keep_surfels_mask[i] = False
                    continue

            # COMPUTE NORMALS FROM NEAR SPHERES
            normals = (self.points[found_sphere_indicies, :] - self.surfel_points[i].reshape(1,3)) / found_dists.reshape(n_considered, 1)
            self.surfel_normals[i, :] = np.sum(normals, axis = 0) / n_considered

        # SAVE NEW PTS AND THEIR NORMALS
        n_keep = np.sum(keep_surfels_mask)
        self.surfel_points = self.surfel_points[keep_surfels_mask, :]
        # self.surfel_normals = self.surfel_normals[keep_surfels_mask, :] / n_considered
        self.surfel_normals = self.surfel_normals[keep_surfels_mask, :] / np.linalg.norm(self.surfel_normals[keep_surfels_mask, :], axis=1).reshape(n_keep, 1)

        # self.surfel_kdtree = 

        return True
    # # #}

    def consistencyCheck(self):# # #{
        print("CONSISTENCY CHECK")
        n_nodes = self.points.shape[0]
        for i in range(n_nodes):
            conns = self.connections[i]
            if conns is None:
                continue
            for c in conns:
                otherconns = self.connections[c]
                if otherconns is None or i not in otherconns:
                    print("ERROR! CONNECTION INCONSISTENCY AT NODE INDEX " + str(i) + " AND " + str(c))
                    print(conns)
                    print(otherconns)
                    return False
        return True# # #}

    def getMinDistToSurfaces(self, pt):# # #{
        if self.surfel_points is None:
            return np.inf
        dists = np.linalg.norm(self.surfel_points - pt, axis=1)

        # inside_mask = dists < 0
        # if not np.any(inside_mask):
        #     return -1
        # largest_inside_dist = np.max(-dists[inside_mask])
        return np.min(dists)# # #}

    def getMaxDistToFreespaceEdge(self, pt):# # #{
        # query_res = self.spheres_kdtree.query(pt.reshape((1,3), k=10, distance_upper_bound=affection_distance)
        dists = np.linalg.norm(self.points - pt, axis=1)
        dists = dists - self.radii

        inside_mask = dists < 0
        if not np.any(inside_mask):
            return -1
        largest_inside_dist = np.max(-dists[inside_mask])
        return largest_inside_dist# # #}

    def labelSpheresByConnectivity(self):
        n_nodes = self.points.shape[0]
        self.connectivity_labels = np.full((n_nodes), -1)
        self.connectivity_segments_counts = []

        seg_id = 0
        for i in range(n_nodes):
            if self.connectivity_labels[i] < 0:
                # FLOODFILL INDEX
                openset = [i]
                self.connectivity_labels[i] = seg_id
                
                n_labeled = 1
                while len(openset) > 0:
                    expansion_node = openset.pop()
                    conns = self.connections[expansion_node]
                    if conns is None:
                        continue
                    for conn_id in conns:
                        if self.connectivity_labels[conn_id] < 0:
                            self.connectivity_labels[conn_id] = seg_id
                            openset.append(conn_id)
                            n_labeled += 1
                self.connectivity_segments_counts.append(n_labeled)
                # print("SEG SIZE: "  +str(n_labeled))
                seg_id += 1
        self.connectivity_segments_counts = np.array(self.connectivity_segments_counts)
        self.max_radius = np.max(self.radii)

        # print("DISCONNECTED REGIONS: " + str(seg_id))

    def wouldPruningNodeMakeConnectedNodesNotFullGraph(self, idx):# # #{
        conns = self.connections[idx]
        if conns is None:
            return False
        if conns.size == 1:
            return False

        # print("WP FOR NODE " + str(idx) + " WITH CONNS " + str(len(conns)))

        frontier = [conns[0]]
        visited = [conns[0], idx]
        while len(frontier) > 0:
            # print("FRONTIER: ")
            # print(frontier)
            popped = frontier.pop()
            # visited.append(popped)
            popped_conns = self.connections[popped]
            for c in popped_conns:
                if (not c in visited) and c in conns:
                    frontier.append(c)
                    visited.append(c)
        # print("VISITED AT END: ")
        # print(visited)
        if len(visited) == conns.size + 1:
            return False
        return True
    # # #}

    def removeNodes(self, toosmall_idxs):# # #{
        # FIRST DESTROY CONNECTIONS TO THE PRUNED SPHERES
        for i in range(toosmall_idxs.size):
            idx = toosmall_idxs[i]
            if (not self.connections[idx] is None) and (not len(self.connections[idx]) == 0):
                for j in range(len(self.connections[idx])):
                    other_idx = self.connections[idx][j]
                    otherconn = self.connections[other_idx]
                    if otherconn.size == 1: #ASSUMING the other sphere has at least 1 connection, which should be to this node
                        self.connections[other_idx] = None
                    else:
                        # print("KOKOT")
                        # print(self.connections[idx][j])
                        # print(self.connections[idx][j].shape)
                        self.connections[other_idx] = np.array([x for x in self.connections[other_idx] if x != idx]).flatten()

        shouldkeep = np.full((self.points.shape[0] , 1), True)
        shouldkeep[toosmall_idxs] = False
        shouldkeep = shouldkeep.flatten()

        index_remapping = np.full((self.points.shape[0] , 1), -1)
        incr = 0
        for i in range(self.points.shape[0]):
            if shouldkeep[i]:
                index_remapping[i] = incr
                incr += 1


        # THEN KILL THE SPHERES
        self.points = self.points[shouldkeep, :]
        self.radii = self.radii[shouldkeep]
        self.connections = self.connections[shouldkeep]

        # GO THRU ALL SURVIVING NODES AND REMAP THEIR CONNECTION INDICES
        for i in range(self.radii.size):
            if not self.connections[i] is None and self.connections[i].size > 0 and not self.connections[i][0] is None:
                self.connections[i] = index_remapping[self.connections[i]].flatten()# # #}

    # #{ def removeSpheresIfRedundant(self, worked_sphere_idxs):
    def removeSpheresIfRedundant(self, worked_sphere_idxs):
        # CHECK IF ADJACENT NODES ARE ALL CONNECTED TOGETHER - SO GRAPH IS NEVER TORN
        shouldkeep = np.full((self.points.shape[0] , 1), True)
        # shouldkeep[toosmall_idxs] = False
        # shouldkeep = shouldkeep.flatten()
        n_remove = 0

        # SORT FROM SMALLEST TO BIGGEST
        worked_sphere_idxs = worked_sphere_idxs[np.argsort(self.radii[worked_sphere_idxs])]

        for idx in worked_sphere_idxs:
            # fg = connectedNodesFormFullGraph()
            conns = self.connections[idx]
            if self.radii[idx] < self.min_radius:
                shouldkeep[idx] = False
                n_remove+=1
                continue

            if conns is None or conns.size == 0:
                continue

            # dont take into account the ones decided for deletion
            conns = ([c for c in conns if shouldkeep[c]])
            if len(conns) == 0 or conns[0] is None:
                continue
            conns = np.array(conns, dtype=int)

            pos = self.points[idx]
            radius = self.radii[idx] 

            distvectors = self.points[conns, :] - pos
            norms = np.linalg.norm(distvectors, axis=1)
            others_radii = self.radii[conns]
            # print("RAD:")
            peeking_dists = (norms - others_radii) + radius

            peek_thresh = 2
            ratio_thresh = 1
            # pdt = peeking_dists < peek_thresh
            magic = norms < ((others_radii + radius) / 2) * 0.5
            # trsh = others_radii > ratio_thresh * radius
            # wp = self.wouldPruningNodeMakeConnectedNodesNotFullGraph(idx)
            # smallerenough = 

            # if (not wp) and np.any(magic):
            if np.any(magic):
                shouldkeep[idx] = False
                n_remove += 1
        # print("REMOVING REDUNDANT: " + str(n_remove))
        if n_remove > 0:
            remove_idxs = np.where(np.logical_not(shouldkeep))[0]
            # print(remove_idxs)
            # print(remove_idxs.shape)
            self.removeNodes(remove_idxs)
    # #}
    
    # #{ def updateConnections(self, worked_sphere_idxs):
    def updateConnections(self, worked_sphere_idxs):
        # print("UPDATING CONNECTIONS FOR " + str(worked_sphere_idxs.size) + " SPHERES")
        for idx in worked_sphere_idxs:
            prev_connections = self.connections[idx]
            intersecting = self.getIntersectingSpheres(self.points[idx, :], self.radii[idx])
            intersecting[idx] = False
            if not np.any(intersecting):
                self.connections[idx] = None
            else:
                newconn = np.where(intersecting)[0]
                self.connections[idx] = newconn.flatten()

            # FOR X2 THAT USED TO BE CONNECTED TO X1 AND ARE NOT ANYMORE, REMOVE X1 FROM X2s CONNECTIONS
            if not prev_connections is None:
                detached_sphere_idxs = [x for x in prev_connections.flatten()] #was in old
                if not self.connections[idx] is None:
                    for remain_conn in self.connections[idx]: #is in new
                        if remain_conn in detached_sphere_idxs: #was in old
                            # print("REM CON")
                            # print(remain_conn)
                            detached_sphere_idxs.remove(remain_conn)

                if len(detached_sphere_idxs) > 0:
                    for det_idx in detached_sphere_idxs:
                        if not det_idx is None:
                            # WARNING! THIS FUCKER RETURNS [None]
                            dif = np.setdiff1d(self.connections[det_idx], np.array([idx], dtype=int))
                            # print("DIFF")
                            # print(dif)
                            if len(dif) == 1 and dif[0] is None:
                                self.connections[det_idx] = None
                            else:
                                self.connections[det_idx] = dif.flatten()
                        else:
                            print("WARN! DET IDX IS NONE")

            # FOR X2 THAT WERE NOT CONNECTED TO X1 AND NOW ARE, ADD X1 TO THEIR CONNS
            # WARNING! THIS FUCKER RETURNS [None]
            not_in_old_yes_in_new = np.setdiff1d(self.connections[idx], prev_connections)
            if not_in_old_yes_in_new.size > 0 and not not_in_old_yes_in_new[0] is None:
                for j in not_in_old_yes_in_new:
                    if self.connections[j] is None:
                        self.connections[j] = np.array([idx])
                    else:
                        # print(self.connections[j].shape)
                        self.connections[j] = np.concatenate((self.connections[j], np.array([idx])))


            # print("NEWCONNS:")
            # print(self.connections[idx])

        return
    # #}

    # #{ def getIntersectingSpheres(self, position, radius):
    def getIntersectingSpheres(self, position, radius):
        distvectors = self.points - position

        # rad2 = radius * radius
        # norms2 = np.sum(np.multiply(distvectors,distvectors), 1)
        # intersecting_idxs = norms2

        norms = np.linalg.norm(distvectors, axis=1)

        intersecting = norms < self.radii + radius
        # print("FOUND INTERSECTIONS: " + str(np.sum(intersecting)))
        return intersecting 
    # #}
# #}

class CoherentSpatialMemoryChunk:# # #{
    def __init__(self):
        self.submaps = []
        self.total_traveled_context_distance = 0

    def mergeSubmaps(self, indices):
        '''merges submaps into the first one and returns it'''
        map1 = self.submaps[indices[0]]
        for i in indices:
            if i == indices[0]:
                continue
            map2 = self.submaps[i]

            relative_T = np.linalg.inv(map1.T_global_to_own_origin) @ map2.T_global_to_own_origin
            relative_rot_T = np.eye(4)
            relative_rot_T[:3, :3] = relative_T[:3, :3]

            sphere_centers_t = transformPoints(map2.points, relative_T)
            surfel_centers_t = transformPoints(map2.surfel_points, relative_T)
            normals_t = transformPoints(map2.surfel_normals, relative_rot_T)

            map1.points = np.concatenate((map1.points, sphere_centers_t))
            map1.surfel_points = np.concatenate((map1.surfel_points, surfel_centers_t))
            map1.surfel_normals = np.concatenate((map1.surfel_normals, normals_t))

            # TODO - handle connections of spheres

        return map1


    def addSubmap(self, smap):
        self.submaps.append(smap)

    def save(self, path):
        print("MCHUNK SAVING " + str(len(self.submaps)) + " SUBMAPS")
        with open(path, "wb") as output:
            pickle.dump(self.__dict__, output, -1)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as mfile:
            cls_dict = pickle.load(mfile)
        mem = cls.__new__(cls)
        mem.__dict__.update(cls_dict)
        return mem
# # #}
