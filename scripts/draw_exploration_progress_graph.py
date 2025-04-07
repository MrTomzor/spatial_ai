import csv
import sys
import matplotlib.pyplot as plt
import os
from collections import defaultdict

def read_data_from_txt(filename):
    # timestamps, explored_space, visited_vps, nonvisited_vps = [], [], [], []
    timestamps, explored_space = [], [] 

    with open(filename, mode='r') as file:
        next(file)  # skip header
        for line in file:
            # print(line)
            # t, v, visited, unvisited = line.strip().split(",")
            vals = line.strip().split(",")
            t = vals[0]
            v = vals[1]
            timestamps.append(float(t))
            explored_space.append(float(v))
            # visited_vps.append(int(visited))
            # nonvisited_vps.append(int(unvisited))

    # return timestamps, explored_space, visited_vps, nonvisited_vps
    return timestamps, explored_space

def plot_exploration_progress_simple(datas):
    # plt.figure()
    for experiment in datas:
        plt.plot(experiment['timestamps'], experiment['explored_volumes'], marker='o', linestyle='-', label = experiment['name'])
    plt.xlabel('time (s)')
    plt.ylabel('total explored free space (sum of sphere radii)')
    plt.title('explored free space over time')
    plt.legend()
    plt.grid()
    plt.show()

def plot_exploration_progress_clustered(datas):
    colors = {}
    colors['openspace'] = 'green'
    colors['basemono'] = 'red'
    colors['octomap'] = 'blue'

    for experiment in datas:
        clr = 'black'
        if experiment['method'] in colors.keys():
            clr = colors[experiment['method']]
        plt.plot(experiment['timestamps'], experiment['explored_volumes'], linestyle='-', color = clr)
    plt.xlabel('time (s)')
    plt.ylabel('total explored free space (sum of sphere radii)')
    plt.title('explored free space over time')
    # plt.legend()
    plt.grid()
    plt.show()

def extract_table_data(datas):
    # todo - print these values for each method: number of experiments, average volume explored, max volume explored, number of crashes across all experiments
    method_stats = defaultdict(lambda: {
        'num_experiments': 0,
        'total_volume': 0.0,
        'max_volume': 0.0,
        'num_crashes': 0
    })

    for experiment in datas:
        method = experiment['method']
        final_volume = experiment['explored_volumes'][-1] if experiment['explored_volumes'] else 0.0
        
        method_stats[method]['num_experiments'] += 1
        method_stats[method]['total_volume'] += final_volume
        method_stats[method]['max_volume'] = max(method_stats[method]['max_volume'], final_volume)
        if experiment['crash']:
            method_stats[method]['num_crashes'] += 1

    print("\n=== Table Summary ===")
    for method, stats in method_stats.items():
        avg_volume = stats['total_volume'] / stats['num_experiments'] if stats['num_experiments'] > 0 else 0.0
        print(f"Method: {method}")
        print(f"  Number of Experiments: {stats['num_experiments']}")
        print(f"  Average Volume Explored: {avg_volume:.2f}")
        print(f"  Max Volume Explored: {stats['max_volume']:.2f}")
        print(f"  Number of Crashes: {stats['num_crashes']}")
        print("---------------------------")

def extract_datas(files):
    datas = []
    for i in range(len(files)):
        filename = files[i]
        print("reading experiment " + str(i) + "from file: " + filename)
        timestamps, explored_volumes = read_data_from_txt(filename)
        experiment = {}
        experiment['timestamps'] = timestamps
        experiment['explored_volumes'] = explored_volumes 

        experiment['method'] = 'unknown'
        if "nofake" in filename:
            experiment['method'] = 'basemono'
        elif "fake" in filename:
            experiment['method'] = 'openspace'
        elif "astar" in filename:
            experiment['method'] = 'octomap'

        experiment['crash'] = "crash" in filename

        experiment['name'] = filename

        datas.append(experiment)
    return datas

def plot_progress_from_all_files(rootpath):

    # get all data files
    data_files = []
    for dirpath, dirnames, filenames in os.walk(rootpath):
        for file in filenames:
            if file.endswith(".txt"):
                full_path = os.path.join(dirpath, file)
                data_files.append(full_path)
    nfiles = len(data_files)
    print("n found files: " + str(nfiles))

    # go thru all and extract data from file
    datas = []
    datas = extract_datas(data_files)
    # index = 0
    # for data_file in data_files:
    #     index += 1
    #     print(f"\nprocessing file {index}/{nfiles}: {data_file}")

    #     do_octomap = false
    #     # todo - draw all into graph

    extract_table_data(datas)

    plot_exploration_progress_clustered(datas)

    

    return true

def main():
    if len(sys.argv) > 1:
        files = []
        print("processing given bagfiles, simple")
        for i in range(len(sys.argv) - 1):
            files += sys.argv[i+1]
        datas = extract_datas(files)

        print(f"read {len(datas)} experiments")
        plot_exploration_progress_simple(datas)
    else:
        rootpath = os.getcwd()
        print("processing all data in this folder: " + rootpath)
        plot_progress_from_all_files(rootpath)

if __name__ == "__main__":
    main()
