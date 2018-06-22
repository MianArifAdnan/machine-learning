# README.md for PP4
# Course: ML-Tufts
# By: Morgan Ciliv
# Date: 2 December 2017

## Goal:
The goal of this programming project is to allow a user to...

## Files:
pp4.py: main
ANN.py: implements an artificial neural network class
ANN_math.py: implements the activation functions and their gradients for the ANN

## Files to import:
pp4.py: Contains functions for preparation of data for the ANN

In python:
import pp4

## Functions to call:
learn(width, depth, train_file, test_file, iters=5000)
        "width": width of each of the hidden layers
        "depth": number of hidden layers
        "trainfile": ".arff" train file with data
        "testfile": ".arff" test file with data
        "iters": number of iterations to run on the training set
            
