import csv
import sys
import matplotlib.pyplot as plt
import os

def read_data_from_txt(filename):
    # timestamps, explored_space, visited_vps, nonvisited_vps = [], [], [], []
    timestamps, explored_space = [], [] 

    with open(filename, mode='r') as file:
        next(file)  # Skip header
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
    plt.xlabel('Time (s)')
    plt.ylabel('Total Explored Free Space (sum of sphere radii)')
    plt.title('Explored Free Space Over Time')
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
    plt.xlabel('Time (s)')
    plt.ylabel('Total Explored Free Space (sum of sphere radii)')
    plt.title('Explored Free Space Over Time')
    # plt.legend()
    plt.grid()
    plt.show()

def extract_datas(files):
    datas = []
    for i in range(len(files)):
        filename = files[i]
        print("reading experiment " + str(i) + "from file: " + filename)
        timestamps, explored_volumes = read_data_from_txt(filename)
        experiment = {}
        experiment['timestamps'] = timestamps
        experiment['explored_volumes'] = explored_volumes 
        # experiment['name'] = "Experiment " + str(i)
        experiment['method'] = 'unknown'
        if "nofake" in filename:
            experiment['method'] = 'basemono'
        elif "fake" in filename:
            experiment['method'] = 'openspace'
        elif "astar" in filename:
            experiment['method'] = 'octomap'
        experiment['name'] = filename
        datas.append(experiment)
    return datas

def plot_progress_from_all_files(rootpath):

    # GET ALL DATA FILES
    data_files = []
    for dirpath, dirnames, filenames in os.walk(rootpath):
        for file in filenames:
            if file.endswith(".txt"):
                full_path = os.path.join(dirpath, file)
                data_files.append(full_path)
    nfiles = len(data_files)
    print("N found files: " + str(nfiles))

    # GO THRU ALL AND EXTRACT DATA FROM FILE
    datas = []
    datas = extract_datas(data_files)
    # index = 0
    # for data_file in data_files:
    #     index += 1
    #     print(f"\nProcessing file {index}/{nfiles}: {data_file}")

    #     do_octomap = False
    #     # TODO - draw all into graph
    plot_exploration_progress_clustered(datas)

    

    return True

def main():
    if len(sys.argv) > 1:
        files = []
        print("Processing given bagfiles, simple")
        for i in range(len(sys.argv) - 1):
            files += sys.argv[i+1]
        datas = extract_datas(files)

        print(f"read {len(datas)} experiments")
        plot_exploration_progress_simple(datas)
    else:
        rootpath = os.getcwd()
        print("Processing all data in this folder: " + rootpath)
        plot_progress_from_all_files(rootpath)

if __name__ == "__main__":
    main()
