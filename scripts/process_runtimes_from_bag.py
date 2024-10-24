import rosbag
import matplotlib.pyplot as plt
from mrs_msgs.msg import Float64ArrayStamped
import numpy as np
import sys

# Load the ROS bag and the relevant topic
def process_rosbag(bag_path):
    # Initialize lists to store data for plotting
    timestamps = []
    runtime_values = []

    # Open the ROS bag
    with rosbag.Bag(bag_path, 'r') as bag:
        # Iterate over all messages on the /spheremap_runtimes topic
        for topic, msg, t in bag.read_messages(topics=['/spheremap_runtimes']):
            if topic == '/spheremap_runtimes':
                # Extract the timestamp (could be msg.header.stamp if you want exact ROS time)
                timestamps.append(msg.header.stamp.to_sec())

                # Extract the array of computation times
                runtime_values.append(msg.values)

    return timestamps, runtime_values

def plot_runtimes(timestamps, runtime_values):
    # Convert timestamps and runtime values into numpy arrays for easier processing
    timestamps = np.array(timestamps)
    timestamps = timestamps  - np.min(timestamps  )
    runtime_values = np.array(runtime_values)

    # Plot the runtimes of individual parts over time
    plt.figure(figsize=(10, 6))

    # Transpose runtime_values so that we can plot each part separately
    labels = ["Polyhedron construction", "Obstacle+Frontier points", "Old Spheres", "New Spheres", "Connectivity+Sparsification"]
    num_parts = runtime_values.shape[1]
    for part_idx in range(num_parts):
        # plt.plot(timestamps, runtime_values[:, part_idx], label=f'Part {part_idx + 1}')
        plt.plot(timestamps, runtime_values[:, part_idx], label=labels[part_idx])
    # plot also sum
    plt.plot(timestamps, np.sum(runtime_values, axis = 1), label=f'Total Runtime')

    plt.xlabel('Time (s)')
    plt.ylabel('Runtime (s)')
    plt.title('Runtimes of Individual Parts Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Path to the ROS bag file
    bag_path = sys.argv[1]

    
    # Process the rosbag
    timestamps, runtime_values = process_rosbag(bag_path)
    
    # Plot the runtimes
    plot_runtimes(timestamps, runtime_values)
