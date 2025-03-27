import csv
import sys
import matplotlib.pyplot as plt

def read_data_from_txt(filename):
    # timestamps, explored_space, visited_vps, nonvisited_vps = [], [], [], []
    timestamps, explored_space = [], [] 

    with open(filename, mode='r') as file:
        next(file)  # Skip header
        for line in file:
            # print(line)
            t, v, visited, unvisited = line.strip().split(",")
            timestamps.append(float(t))
            explored_space.append(float(v))
            # visited_vps.append(int(visited))
            # nonvisited_vps.append(int(unvisited))

    # return timestamps, explored_space, visited_vps, nonvisited_vps
    return timestamps, explored_space

def plot_exploration_progress(datas):
    # plt.figure()
    for experiment in datas:
        plt.plot(experiment['timestamps'], experiment['explored_volumes'], marker='o', linestyle='-', label = experiment['name'])
    plt.xlabel('Time (s)')
    plt.ylabel('Total Explored Free Space (sum of sphere radii)')
    plt.title('Explored Free Space Over Time')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    datas = []
    for i in range(len(sys.argv) - 1):
        filename = sys.argv[i + 1]
        print("reading experiment " + str(i) + "from file: " + filename)
        timestamps, explored_volumes = read_data_from_txt(filename)
        experiment = {}
        experiment['timestamps'] = timestamps
        experiment['explored_volumes'] = explored_volumes 
        experiment['name'] = "Experiment " + str(i)
        datas.append(experiment)
    print(f"read {len(datas)} experiments")
    plot_exploration_progress(datas)


if __name__ == "__main__":
    main()
