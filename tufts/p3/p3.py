"""
    p3.py
    Programming Project 3 for Machine Learning
    By: Morgan Ciliv
"""

import math
import copy
#import bool

import numpy as np

# Parse the arff files
# Put arff parser here

# Removes endline characters
def read_file(file_name):
    with open(file_name) as file:
        file_lines = [line[:-1] for line in file]
    file_lines = [line.strip() for line in file_lines]
    return file_lines

# Writes training file back out with new name
def write_file(num_train_examples, new_data_for_file):
    filename = "train" + str(num_train_examples) + ".arff"
    with open(filename, 'w+') as file:
        for line in new_data_for_file:
            line = line + "\n"
            file.write(line)
    return filename

def calc_cluster_scatter(means, file_data):
    cluster_scatter = 0.0
    for i, mean in enumerate(means):
        for j, point in enumerate(file_data["points"]):
            if file_data["clusters"][j] == i:
                cluster_scatter += get_euclidean(point, mean) ** 2
    return cluster_scatter

def counts(val, list):
    counts = 0
    for i in list:
        if i == val:
            counts += 1
    return counts

# num of occurances / total
def prob(val, list):
    return float(counts(val, list)) / np.size(list)

# H(X)
def entropy(X):
    uniques = np.unique(X)
    entropy = 0.0
    for val in uniques:
        p = prob(val, X)
        entropy += -1.0 * p * np.log(p)
    return entropy


def means(X, points):
    labels = np.unique(X)
    point_dim = size(points[0])
    sums = np.zeros((np.size(labels), points_dim))
    counts = np.zeros(np.size(labels))
    for i, point in enumerate(points):
        sums[X[i]-1] = sums[X[i]-1] + point
        counts[X[i]-1] += 1
    np.expand_dims(counts)
    print(shape(counts))
    return sums / counts

# returns the corresponding labels of cluster y for X
def labels_from_cluster(X, x_points, Y, y_points):
    y_labels = np.zeros(x_points.size())
    for i, x in enumerate(X):
        closest = 0
        min_dist = math.inf
        for j, y_mean in enumerate(means(Y, y_points)):
            if get_euclidean(x_points[i], y_mean) < min_dist:
                min_dist = get_euclidean(x_points[i], y_mean)
                y_labels[i] = j+1
    return y_labels

# H(X|Y)
def conditional_entropy(X, x_points, Y, y_points):
    # Constrain list to just where Y
    entropy = 0
    y_labels = labels_from_cluster(X, x_points, Y, y_points)
    y_labels_unique = np.unique(y_labels)
    for unique in y_labels_unique:
        xs_in_y = []
        for i, y in enumerate(y_labels):
            if unique == y:
                xs_in_y.append(X[i])
            xs_in_y_arr = np.array(xs_in_y)
            for j, x in enumerate(X):
                p = prob(x, xs_in_y_arr)
                entropy += -1.0 * p * math.log(p)

# I(X,Y) = H(X) - H(X|Y)
def mutual_information(X, Y):
    # Should call conditional_entropy with more params, but ran out of time
    return entropy(X) - conditional_entropy(X, Y)

# NMIsum = 2I(U,V) / (H(U) + H(V))
def calc_NMI(data_file):
    numerator = 2 * mutual_information(data_file["clusters"],
        data_file["labels"])
    denominator = entropy(data_file["clusters"]) + entropy(data_file["labels"])
    return float(numerator) / denominator

data = {"artdata0.5.arff": {}, "artdata1.arff": {}, "artdata2.arff": {},
    "artdata3.arff": {}, "artdata4.arff": {}, "soybean-processed.arff": {},
    "ionosphere.arff": {}, "iris.arff": {}}

def get_euclidean(a, b):
    return np.linalg.norm(a - b)

file = "artdata0.5.arff"
data_lines = read_file(file)
for i, line in enumerate(data_lines):
    if line == "@DATA":
        data_line = i + 1
        break
split = [line.split(',') for line in data_lines[data_line:]]
points = np.array([[float(elem) for elem in line[:-1]] for line in split])
labels = np.array([int(line[-1]) for line in split])
data[file]["points"] = points
data[file]["labels"] = labels
num_points = len(points)
data[file]["clusters"] = np.full(num_points, 0)
data[file]["dists"] = np.full(num_points, math.inf)

# k-means iteratively updates
# - the means
# - the cluster assignments
# --> cluster can become empty during the update
# ---> In this case, the mean of any empty cluster is initialized with a
#      random example from the data set.
# Note: Do not use class label for the distance calculation
k = 3
num_features = 5

# Initialize to random points of the data set such that none of the means are
# the same
means = np.zeros((3,5))
for i in range(k):
        means[i] = data[file]["points"][np.random.randint(num_points)]
        j = i-1
        while j >= 0:
            if np.array_equal(means[i], means[j]):
                means[i] = data[file]["points"][np.random.randint(num_points)]
            else:
                j -= 1

while True:
    for cluster in range(k):
        for i, point in enumerate(data[file]["points"]):
            dist = get_euclidean(means[cluster], point)
            if dist < data[file]["dists"][i]:
                data[file]["clusters"][i] = cluster
                data[file]["dists"][i] = dist

    occurances = np.zeros((3,1))
    for cluster in data[file]["clusters"]:
        occurances[cluster] += 1
    for cluster, occurance in enumerate(occurances):
        if occurance == 0:
            data[file]["clusters"][np.random.randint(num_points)] = cluster

    means_prev = copy.deepcopy(means)
    means.fill(0)
    occurances.fill(0)

    for i, cluster in enumerate(data[file]["clusters"]):
        means[cluster] += data[file]["points"][i]
        occurances[cluster] += 1

    # Assures
    for cluster, occurance in enumerate(occurances):
        if occurance == 0:
            means[cluster] = data[file]["points"][np.random.randint(num_points)]
            duplicates = False
            clust = 0
            while i < k or duplicates:
                duplicates = False
                if cluster != clust and means[cluster] == means[clust]:
                    rand_idx = np.random.randint(num_points)
                    means[cluster] = data[file]["points"][rand_idx]
                    duplicates = True
                    i = 0
    means = np.divide(means, occurances)
    print(means)
    if np.array_equal(means, means_prev):
        break
    else:
        means_prev = copy.deepcopy(means)

print(calc_cluster_scatter(means, data[file]))
#print(calc_NMI(data[file]))
