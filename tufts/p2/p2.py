# Morgan Ciliv
# Programming Project 2: Machine Learning

import subprocess
import re
import matplotlib.pyplot as plt
import random
import shutil
import numpy
import math

FILES = ["amazon_cells_labelled.txt", "imdb_labelled.txt", "yelp_labelled.txt"]
COMPANIES = ["amazon", "imdb", "yelp"]
def read_label_sent(file_name):
    with open(file_name) as file:
        file_lines = [line.strip("\n") for line in file]
        file_lines = [re.split(r'\t+', line) for line in file_lines]
        data = [[line[0].split(' '), int(line[1])] for line in file_lines]
    return data

# Split into test and training sets
def split_data(data):
    split_mark = int(len(data) * .8)
    train = data[:split_mark]
    test = data[split_mark:]
    return {"train": train, "test": test}

data_sets = {}
for file in FILES:
    data_sets[file.split("_")[0]] = split_data(read_label_sent(file))
print(data_sets)

# Fill data structures with corresponding data
num_positives = {"amazon": 0, "imdb": 0, "yelp": 0}
num_negatives = {"amazon": 0, "imdb": 0, "yelp": 0}
pos_word_counts = {"amazon": {}, "imdb": {}, "yelp": {}}
neg_word_counts = {"amazon": {}, "imdb": {}, "yelp": {}}
for company in COMPANIES:
    for example in data_sets[company]["train"]:
        if example[1]:
            num_positives[company] += 1
            for word in example[0]:
                if word not in pos_word_counts[company].keys():
                    pos_word_counts[company][word] = 1
                    neg_word_counts[company][word] = 0
                else:
                    pos_word_counts[company][word] += 1
        else:
            num_negatives[company] += 1
            for word in example[0]:
                if word not in neg_word_counts[company].keys():
                    neg_word_counts[company][word] = 1
                    pos_word_counts[company][word] = 0
                else:
                    neg_word_counts[company][word] += 1

# Log probabilities
accuracies = {"amazon": 0, "imdb": 0, "yelp": 0}
for company in COMPANIES:
    accuracy = 0
    pos_class = num_positives[company]
    neg_class = num_negatives[company]
    num_words = float(len(pos_word_counts[company].keys()))
    for example in data_sets[company]["test"]:
        for word in example[0]:
            if word in pos_word_counts[company].keys():
                if pos_word_counts[company][word] > 0:
                    pos_class += math.log(pos_word_counts[company][word] /
                                          num_words)
                if neg_word_counts[company][word] > 0:
                    neg_class += math.log(neg_word_counts[company][word] /
                                      num_words)
        if pos_class >= neg_class:
            pred_class = 1
        else:
            pred_class = 0
        if pred_class == example[1]:
            accuracy += 1.0
    accuracy /= num_words
    accuracies[company] = accuracy
print(accuracies)

# Smoothing
