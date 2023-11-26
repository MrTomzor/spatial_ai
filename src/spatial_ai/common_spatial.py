#!/usr/bin/env python

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
import scipy
import pickle


# common utils# #{
class EmptyClass(object):
    pass

def transformationMatrixToHeading(T):
    return np.arctan2(T[1,0], T[0, 0])

def transformPoints(pts, T):
    # pts = Nx3 matrix, T = transformation matrix to apply
    res = np.concatenate((pts.T, np.full((1, pts.shape[0]), 1)))
    res = T @ res 
    res = res / res[3, :] # unhomogenize
    return res[:3, :].T
# # #}

class SubmapKeyframe:# # #{
    def __init__(self, T):
        self.pos = T[:3,3]
        self.heading = transformationMatrixToHeading(T)
        print("HEADING")
        print(T)
        print(self.heading)

    def euclid_dist(self, other_kf): 
        return np.linalg.norm(self.pos - other_kf.pos)

    def heading_dif(self, other_kf): 
        dif = np.abs(other_kf.heading - self.heading)
        if dif > 3.14159/2:
            dif = 3.14159 - dif
        return dif
# # #}

# #{ class SphereMap
class SphereMap:
    def __init__(self, init_radius, min_radius):# # #{
        self.points = np.array([0,0,0]).reshape((1,3))
        self.radii = np.array([init_radius]).reshape((1,1))
        self.connections = np.array([None], dtype=object)
        self.visual_keyframes = []
        self.traveled_context_distance = 0

        self.surfel_points = None
        self.surfel_radii = None
        self.surfel_normals = None

        self.min_radius = min_radius
        self.max_radius = init_radius# # #}

    def updateSurfels(self, visible_points, pixpos, simplices):# # #{
        # Compute normals measurements for the visible points, all should be pointing towards the camera
        # What frame are the points in?
        # print("UPDATING SURFELS")
        n_test_new_pts = visible_points.shape[0]

        # TODO do based on map scale!!!
        filtering_radius = 1

        pts_survived_first_mask = np.full((n_test_new_pts), True)
        if not self.surfel_points is None:
            n_existign_points = self.surfel_points.shape[0]
            # print("N EXISTING SURFELS: " + str(n_existign_points))
            existing_points_distmatrix = scipy.spatial.distance_matrix(visible_points, self.surfel_points)
            # pts_survived_first_mask = np.all(existing_points_distmatrix > filtering_radius, axis = 1)
            pts_survived_first_mask = np.sum(existing_points_distmatrix > filtering_radius, axis=1) == n_existign_points
            # print(np.sum(existing_points_distmatrix > filtering_radius, axis=1))

            # print("N SURVIVED FIRST MASK:" + str(np.sum(pts_survived_first_mask)))

        pts_survived_filter_with_mappoints = visible_points[pts_survived_first_mask]

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
            else:
                self.surfel_points = np.concatenate((self.surfel_points, pts_survived_filter_with_mappoints[pts_added_mask]))

        # COMPUTE POINT NORMALS AND KILL THEM IF OCCUPIED BY SPHERES
        affection_distance = self.max_radius * 1.2
        killing_inside_dist = -0.1

        max_n_affecting_spheres = 5
        skdtree_query = self.spheres_kdtree.query(self.surfel_points, k=max_n_affecting_spheres, distance_upper_bound=affection_distance)
        n_spheres = self.points.shape[0]
        print("N SPHERES: " + str(n_spheres))

        self.surfels_that_have_normals = np.full((self.surfel_points.shape[0]), False)
        keep_surfels_mask = np.full((self.surfel_points.shape[0]), True)
        self.surfel_normals = np.zeros((self.surfel_points.shape[0], 3))

        # CHECK IF NOT KILLED BY SPHERE AND COMPUTE NORMAL IF NOT
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

            # CHECK IF KILLED BY THE SPHERES
            dists_from_spheres_edge = found_dists - found_radii
            # print(dists_from_spheres_edge )

            if np.any(dists_from_spheres_edge < killing_inside_dist):
                keep_surfels_mask[i] = False
                # print("KILLING")
                continue

            # COMPUTE NORMALS FROM NEAR SPHERES
            normals = (self.points[found_sphere_indicies, :] - self.surfel_points[i].reshape(1,3)) / found_dists.reshape(n_considered, 1)
            self.surfel_normals[i, :] = np.sum(normals, axis = 0) / n_considered

        # SAVE NEW PTS AND THEIR NORMALS
        n_keep = np.sum(keep_surfels_mask)
        self.surfel_points = self.surfel_points[keep_surfels_mask, :]
        # self.surfel_normals = self.surfel_normals[keep_surfels_mask, :] / n_considered
        self.surfel_normals = self.surfel_normals[keep_surfels_mask, :] / np.linalg.norm(self.surfel_normals[keep_surfels_mask, :], axis=1).reshape(n_keep, 1)


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

    def labelSpheresByConnectivity(self):
        n_nodes = self.points.shape[0]
        self.connectivity_labels = np.full((n_nodes), -1)

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
                # print("SEG SIZE: "  +str(n_labeled))
                seg_id += 1

        print("DISCONNECTED REGIONS: " + str(seg_id))

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
            if conns is None or conns.size == 0:
                continue

            # dont take into account the ones decided for deletion
            conns = ([c for c in conns if shouldkeep[c]])
            if len(conns)== 0:
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
            magic = norms < ((others_radii + radius) / 2) * 0.9
            trsh = others_radii > ratio_thresh * radius
            # print("P")
            # print(norms )
            # print(peeking_dists )
            # print(radius)

            # print("MAGIC")
            # print(magic)
            # print(trsh)
            wp = self.wouldPruningNodeMakeConnectedNodesNotFullGraph(idx)
            # print(wp)

            if (not wp) and np.any(magic):
            # if np.any(np.logical_and(pdt, trsh)):
                shouldkeep[idx] = False
                n_remove += 1
        print("REMOVING REDUNDANT: " + str(n_remove))
        if n_remove > 0:
            remove_idxs = np.where(np.logical_not(shouldkeep))[0]
            print(remove_idxs)
            print(remove_idxs.shape)
            self.removeNodes(remove_idxs)
    # #}
    
    # #{ def updateConnections(self, worked_sphere_idxs):
    def updateConnections(self, worked_sphere_idxs):
        print("UPDATING CONNECTIONS FOR " + str(worked_sphere_idxs.size) + " SPHERES")
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

class CoherentSpatialMemoryChunk:
    def __init__(self):# # #{
        self.submaps = []
        self.total_traveled_context_distance = 0

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

