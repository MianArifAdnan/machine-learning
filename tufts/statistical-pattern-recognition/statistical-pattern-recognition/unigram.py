################################################################################
# unigram.py
# By: Morgan Ciliv
#
# Statistical Pattern Recognition - Programming Project 1
################################################################################

import math
import numpy as np
from scipy import misc as spm
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
#                                   Part 1
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#                                   Setup data
# ------------------------------------------------------------------------------

# Functions to setup data
def file_to_list(file_name):
    with open(file_name, 'r') as f:
        list = [word for line in f for word in line.split()]
        f.close()
    return list

def list_to_file(list, file_name):
    with open(file_name, 'w+') as f:
        for elem in list:
            f.write("%s " % elem)
    f.close()

def file_to_dict(file_name):
    with open(file_name, 'r') as f:
        dict = eval(f.read())
    return dict

def dict_to_file(dict, file_name):
    with open(file_name, 'w+') as f:
        f.write(str(dict))
    f.close()

# Creates dictionary of words with key as the word and value as the count
def get_counts_dict(list):
    counts_dict = {}
    for elem in list:
        if elem in counts_dict.keys():
            counts_dict[elem] += 1
        else:
            counts_dict[elem] = 1
    return counts_dict

# Data constants
TRAIN_WORDS = file_to_list("training_data.txt")
TEST_WORDS = file_to_list("test_data.txt")
N = len(TRAIN_WORDS) # Same length as test_data
TRAIN_SET_SIZES = [N/128, N/64, N/16, N/4, N]
TRAIN_SETS = [TRAIN_WORDS[:size] for size in TRAIN_SET_SIZES]
ALL_WORDS = TRAIN_WORDS + TEST_WORDS
try:
    DICT = file_to_dict("dictionary.txt")
except:
    DICT = get_counts_dict(ALL_WORDS)
    dict_to_file(DICT, "dictionary.txt")
K = len(DICT)

# ------------------------------------------------------------------------------
# Train a unigram model with a dirichlet prior distribution using 3 different
# methods of training.
#
# 3 methods for calculating P(next word = k-th word of vocabulary):
#
# 1. MLE: Maximum Likelihood Estimation
# 2. MAP: Maximum a Posteriori Estimation
# 3. Predictive Distribution (PD)
# ------------------------------------------------------------------------------

# Calculates perplexity of a numpy array
def perplexity(prob_list):
    return (np.exp((-1.0 / N) * np.sum(np.log(prob_list)))).item()

# Dirichlet Prior Distriubtion with parameter 'alpha'
ONES = np.ones(K)
ALPHA_PRIME = 2.0
ALPHA = ALPHA_PRIME * ONES
ALPHA_0 = ALPHA_PRIME * K # More general, but more expensive: np.sum(ALPHA)
M = np.array(DICT.values(), dtype=float)

# 1. MLE: Maximum Likelihood Estimate
probs_MLE = [M / N for N in TRAIN_SET_SIZES]
perplexities_MLE = [perplexity(p) for p in probs_MLE]
print(type(perplexities_MLE[0]))
print(perplexities_MLE)

# 2. MAP: Maximum a Posteriori Estimate
probs_MAP = [(M + ALPHA - 1) / (N + ALPHA_0 - K) for N in TRAIN_SET_SIZES]
perplexities_MAP = [perplexity(p) for p in probs_MAP]
print(perplexities_MAP)

# 3. Predictive Distribution
probs_PD = [(M + ALPHA) / (N + ALPHA_0) for N in TRAIN_SET_SIZES]
perplexities_PD = [perplexity(p) for p in probs_PD]
print(perplexities_PD)

# Plot of perplexities versus the train set size
plt.figure(1)
a, = plt.plot(TRAIN_SET_SIZES, perplexities_MLE, '.-', label="MLE")
b, = plt.plot(TRAIN_SET_SIZES, perplexities_MAP, '.-', label="MAP")
c, = plt.plot(TRAIN_SET_SIZES, perplexities_PD, '.-', label="Pred Dist")
plt.title("Perplexities vs. Size of Training Set")
plt.xlabel("Size of Training Set")
plt.ylabel("Perplexity")
plt.legend([a, b, c], ["MLE", "MAP", "Pred Dist"])
plt.show()

# ------------------------------------------------------------------------------
#                               Part 2
# ------------------------------------------------------------------------------

alpha_primes = np.arange(1,11, dtype=float)
print alpha_primes

alphas = np.array([alpha_prime * ONES for alpha_prime in alpha_primes])
alpha_0s = np.array([sum(alpha) for alpha in alphas])
print alphas
print alpha_0s

def sum_logs(x):
    x = int(x)
    sumlogs = 0
    for i in range(1,x):
        sumlogs += math.log(i)
    return sumlogs

## LogEvidence function: P(Data|ALPHA)
p_evidences = np.array([], dtype=float)
for i, alpha in enumerate(alphas):
    p_evid_num = np.sum(np.array([sum_logs(x) for x in (alphas[i] + M)]))
    p_evid_den_calc1 = sum_logs(N / 128) # Cancel the factorials
    p_evid_den_calc2 = np.sum(np.array([sum_logs(x) for x in alphas[i]]))
    p_evid_den = p_evid_den_calc1 * p_evid_den_calc2
    np.append(p_evidences, p_evid_num / p_evid_den)
print p_evidences
