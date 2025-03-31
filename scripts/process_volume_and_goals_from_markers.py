#!/usr/bin/env python

import rospy
import rosbag
import sys
import matplotlib.pyplot as plt
from visualization_msgs.msg import MarkerArray
import csv
import numpy as np

OCTOMAP_FREESPACE_TOPICNAME = "/uav1/octomap_local_vis/free_cells_vis_array_throttled"
SPHERES_FREESPACE_TOPICNAME = "/spheremap_freespace"
GOALS_TOPICNAME = "/exploration_goals"
SPHERES_CELLSIZE = 1
OCTOMAP_CELLSIZE = 0.5

def world_to_grid(coord, cellsize):
    """Convert a world coordinate to a discrete grid index."""
    return int(round(coord / cellsize))

def mark_sphere_explored(grid, center, radius, cellsize):
    """Mark cells in a hash map that intersect a given sphere."""
    x_c, y_c, z_c = center
    r_cells = int(radius / cellsize)  # Radius in grid cells

    # Iterate over grid cells inside the bounding box
    for x in range(world_to_grid(x_c - radius, cellsize), world_to_grid(x_c + radius, cellsize) + 1):
        for y in range(world_to_grid(y_c - radius, cellsize), world_to_grid(y_c + radius, cellsize) + 1):
            for z in range(world_to_grid(z_c - radius, cellsize), world_to_grid(z_c + radius, cellsize) + 1):
                # Convert grid indices back to world coordinates
                x_w = x * cellsize
                y_w = y * cellsize
                z_w = z * cellsize
                
                # Check if the grid cell center is inside the sphere
                if (x_w - x_c) ** 2 + (y_w - y_c) ** 2 + (z_w - z_c) ** 2 <= radius ** 2:
                    grid[(x, y, z)] = True  # Store in hash map (dictionary)


# def compute_free_space(markerarray, cell_size):
    # Get bounds

def process_explored_volume_octomap(bag, cellsize, skip_factor):
    timestamps = []
    explored_space = []
    start_time = None
    
    process_every_nth = int(1 / skip_factor)
    message_index = 0
    for topic, msg, t in bag.read_messages(topics=[OCTOMAP_FREESPACE_TOPICNAME ]):

        if message_index % process_every_nth == 0:
            # print("PROC, index: " + str(message_index))
            if start_time is None:
                start_time = t.to_sec()
            
            # print("MSG, n markers: " + str(len(msg.markers)))

            explored_volume = 0
            marker_sizes = []
            # EACH MARKER - CUBE ARRAY
            index = 0
            for marker in msg.markers:
                # radius = marker.scale.x / 2.0  # Assuming scale.x is diameter
                sidelen_mod = (len(msg.markers) - index)
                cube_volume = np.power(OCTOMAP_CELLSIZE * sidelen_mod, 3) 
                n_cubes = len(marker.points)
                # print(marker.type)
                # print("sidelen_mod:" + str(sidelen_mod))
                # print("vol:" + str(cube_volume))
                # print("cubes:" + str(n_cubes))

                explored_volume += n_cubes * cube_volume
                index += 1

            # marker_sizes = np.array(marker_sizes)
            # marker_sizes_unique = np.unique(marker_sizes)
            # print("UNIQUE SIZES:") 
            # print(marker_sizes_unique)


            # Compute explored volume
            print("Explored volume: " + str(explored_volume))

            # add data
            timestamps.append(t.to_sec() - start_time)
            explored_space.append(explored_volume)

        message_index += 1

    return timestamps, explored_space

def process_explored_volume_spheres(bag, cellsize, skip_factor):
    timestamps = []
    explored_space = []
    start_time = None
    grid = {}
    
    process_every_nth = int(1 / skip_factor)
    message_index = 0
    for topic, msg, t in bag.read_messages(topics=[SPHERES_FREESPACE_TOPICNAME ]):

        if message_index % process_every_nth == 0:
            # print("PROC, index: " + str(message_index))
            if start_time is None:
                start_time = t.to_sec()
            
            # print("MSG, n markers: " + str(len(msg.markers)))
            # total_radius = sum(marker.scale.x / 2.0 for marker in msg.markers)  # Assuming scale.x is diameter

            # Mark cells in grid as explored
            for marker in msg.markers:
                radius = marker.scale.x / 2.0  # Assuming scale.x is diameter
                center = (marker.pose.position.x, marker.pose.position.y, marker.pose.position.z)
                mark_sphere_explored(grid, center, radius, cellsize)


            # Compute explored volume
            explored_cells = len(grid)  # Unique occupied cells
            explored_volume = explored_cells * (cellsize ** 3)
            print("Explored volume: " + str(explored_volume))

            # add data
            timestamps.append(t.to_sec() - start_time)
            explored_space.append(explored_volume)

        message_index += 1

    return timestamps, explored_space

def process_visited_goals(bag, skip_factor):
    timestamps = []
    visited_vps = []
    nonvisited_vps = []
    start_time = None
    
    process_every_nth = int(1 / skip_factor)
    message_index = 0
    for topic, msg, t in bag.read_messages(topics=[GOALS_TOPICNAME]):

        if message_index % process_every_nth == 0:
            print("PROC, index: " + str(message_index))
            if start_time is None:
                start_time = t.to_sec()
            
            print("MSG, n markers: " + str(len(msg.markers)))

            # Mark cells in grid as explored
            n_explored = 0
            n_unexplored = 0
            thresh = 0.8
            for marker in msg.markers:
                if marker.color.r > thresh:
                    n_unexplored += 1
                else:
                    n_explored += 1
                # radius = marker.scale.x / 2.0  # Assuming scale.x is diameter
                # center = (marker.pose.position.x, marker.pose.position.y, marker.pose.position.z)
                # mark_sphere_explored(grid, center, radius, cellsize)
            print("Explored vs Unexplored: " + str(n_explored) + "/" + str(n_unexplored))

            # add data
            timestamps.append(t.to_sec() - start_time)
            visited_vps.append(n_explored)
            nonvisited_vps.append(n_unexplored)

        message_index += 1

    return timestamps, visited_vps, nonvisited_vps

def plot_explored_space(timestamps, explored_space):
    plt.figure()
    plt.plot(timestamps, explored_space, marker='o', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Total Explored Free Space (sum of sphere radii)')
    plt.title('Explored Free Space Over Time')
    plt.grid()
    plt.show()

# def save_data_to_csv(filename, timestamps, explored_space, visited_vps, nonvisited_vps):
#     with open(filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Time (s)", "Explored Volume", "Visited VPs", "Unvisited VPs"])  # Column headers
#         for t, v, visited, unvisited in zip(timestamps, explored_space, visited_vps, nonvisited_vps):
#             writer.writerow([t, v, visited, unvisited])
#     print(f"Data saved to {filename}")
def save_data_to_csv(filename, timestamps, explored_space):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (s)", "Explored Volume"])  # Column headers
        for t, v in zip(timestamps, explored_space):
            writer.writerow([t, v])
    print(f"Data saved to {filename}")

def main():
    if len(sys.argv) < 2:
        print("Usage: rosrun your_package script.py <bagfile_path> <>")
        sys.exit(1)
    
    bagfile_path = sys.argv[1]
    output_data_path_spheres = bagfile_path + "_processed_spheres.txt"
    output_data_path_octomap = bagfile_path + "_processed_octomap.txt"
    skip_factor = float(sys.argv[2])

    try:
        bag = rosbag.Bag(bagfile_path, 'r')
    except Exception as e:
        print(f"Error opening bag file: {e}")
        sys.exit(1)

    endvals = {}

    # VOLUME PROC - get both octomap and sphere data
    # print("processing spheremaps volume")
    # timestamps_spheres, explored_space_spheres = process_explored_volume_spheres(bag, SPHERES_CELLSIZE, skip_factor)
    # if len(explored_space_spheres) > 0:
    #     endvals['Total Explored Volume Spheres'] = explored_space_spheres[-1]
    
    print("processing octomap volume")
    timestamps_octomap, explored_space_octomap = process_explored_volume_octomap(bag, OCTOMAP_CELLSIZE, skip_factor)
    endvals['Total Explored Volume Octomap'] = explored_space_octomap[-1]
    # plot_explored_space_spheres(timestamps, explored_space_spheres)


    # GOALS PROC
    # timestamps_vps, visited_vps, nonvisited_vps = process_visited_goals(bag, 1)
    # # plot_explored_space_spheres(timestamps_vps, visited_vps)
    # endvals['Total visited VPs'] = visited_vps[-1]
    # endvals['Final unvisited VPs'] = nonvisited_vps[-1]

    # PRINT FINAL VALS
    print("ENDVALS:")
    for key in endvals.keys():
        print(key)
        print(endvals[key])


    # SAVE DATA
    # save_data_to_csv(output_data_path_spheres, timestamps_spheres, explored_space_spheres)
    save_data_to_csv(output_data_path_octomap, timestamps_octomap, explored_space_octomap)

    bag.close()

if __name__ == "__main__":
    main()
