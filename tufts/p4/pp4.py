"""
    pp4.py: Main file for programming project 4
    About: Implements a feedforward neural network from ARFF files
    To use: Call learn(width, depth, train_file, test_file)
            "width": width of each of the hidden layers
            "depth": number of hidden layers
            "train_file": ".arff" file with the training data
            "test_file": ".arff" file with the testing data
            
    Course: Machine Learning at Tufts
    Professor: Roni Khardon
    By: Morgan Ciliv
    Date: 1 December 2017
"""

import numpy as np
import ANN
import ANN_math

FILE_838 = "838.arff"
FILE_OPTDIGITS_TRAIN = "optdigits_train.arff"
FILE_OPTDIGITS_TEST = "optdigits_test.arff"
FILE_BASIC = "basic.arff"

def file_rows(file):
    with open(file) as file_text:
        rows = [row[:-1] for row in file_text]
        return rows

def file_markers(rows):
    markers = {"relation": 0, "attribute": [], "data": 0}
    for row_num, row in enumerate(rows):
        if len(row) > 0 and row[0] == "@":
            marker = row.split(" ", 1)[0][1:]
            if type(markers[marker]) is list:
                markers[marker].append(row_num)
            else:
                markers[marker] = row_num
    return markers

def features_and_labels(data_list):
    features = [example[:-1] for example in data_list]
    features = np.array(features, dtype=int)
    labels = [example[-1] for example in data_list]
    labels = np.array(labels, dtype=int)[:, np.newaxis]
    return {"features": features, "labels": labels}

def output_classes(rows, markers):
    for attribute_row in markers["attribute"]:
        letter_sequences = rows[attribute_row].split(" ")
        if letter_sequences[1].lower() == "class":
            classes = letter_sequences[-1][1:-1].split(",")
            classes = np.array(classes, dtype=int)[:, np.newaxis]
            return classes

def file_data(file):
    rows = file_rows(file)
    markers = file_markers(rows)
    data_start = markers["data"] + 1
    data_list = [row.split(",") for row in rows[data_start:]]
    data = features_and_labels(data_list)
    data["classes"] = output_classes(rows, markers)
    return data

def learn(width, depth, train_file, test_file, iters=5000):
    train_data, test_data = file_data(train_file), file_data(test_file)
    ann = ANN.ArtificialNeuralNetwork(train_data, test_data, width, depth,
        iters)
    print("Correct/Incorrect for Training Set: ", ann.correct_div_incorrect_examples("train"))
    print("Correct/Incorrect for Test Set: ",
            ann.correct_div_incorrect_examples("test"))
#        ann.plot_error_v_iter(description, "test")
    return ann

def test_file_markers_838():
    rows = file_rows(FILE_838)
    markers = file_markers(rows)
    assert(markers["relation"] == 0)
    for i, row_num in enumerate(markers["attribute"]):
        assert(markers["attribute"][i] == markers["relation"] + 2 + i)
    assert(markers["data"] == 12)

def print_data():
    print(file_data(FILE_838))

def test_description(width, depth):
    return ("ANN with Width of " + str(width) + " and Depth of " +
            str(depth) + " on Optdigits Dataset")

def train_and_testANN(width, depth, train_file, test_file, iters=5000):
        description = test_description(width, depth)
        print("\nTest ", description)
        print("_______________________________________________________________")
        ann = learn(width, depth, FILE_OPTDIGITS_TRAIN, FILE_OPTDIGITS_TEST,
            iters)
        print("Correct/Incorrect for Training Set: ", ann.correct_div_incorrect_examples("train"))
        print("Correct/Incorrect for Test Set: ",
            ann.correct_div_incorrect_examples("test"))
        ann.plot_error_v_iter(description, "test")

def test_ANN():
    description = "838 Dataset"
    print("\nTest ", description, ":")
    print("___________________________________________________________________")
    width = 3
    depth = 1
    ann = learn(width, depth, FILE_838, FILE_838)
    print("Correct/Incorrect for Training Set: ",
        ann.correct_div_incorrect_examples("train"))
    ann.plot_error_v_iter(description, "train")
    print("Correct/Incorrect for Test Set: ",
        ann.correct_div_incorrect_examples("test"))
    ann.plot_error_v_iter(description, "test")
    ann.layer2_activations()

    depth = 3
    for width in range(5, 41, 5):
        test_ANN_with_test_set(width, depth, FILE_OPTDIGITS_TRAIN,
        FILE_OPTDIGITS_TEST, iters=200)

    width = 10
    for depth in range(0, 6):
        test_ANN_with_test_set(width, depth, FILE_OPTDIGITS_TRAIN,
        FILE_OPTDIGITS_TEST, iters=200)

def test_ANN_basic():
    print("Testing basic\n")
    width = 2
    depth = 2
    train_file = FILE_BASIC
    test_file = FILE_BASIC
    learn(width, depth, train_file, test_file)

def test_ANN_math():
    assert(nn_math.sigmoid(0) == 1/2)
    assert(nn_math.sigmoid(-1, threshold=0) == 1/2)
    assert(nn_math.sigmoid(-1) > 0.268 and nn_math.sigmoid(-1) < 0.269)
    assert(nn_math.sigmoid_gradient(0) == 1/4)

def test_data(file):
    print(file_data(file))

def test():
    test_data(FILE_OPTDIGITS_TRAIN)
    test_file_markers_838()
    test_ANN_math()
    test_ANN()

def main():
    test_ANN()

if __name__ == "__main__":
    main()
