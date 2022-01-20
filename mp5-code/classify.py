# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    W = np.zeros(len(train_set[0]) + 1)
    b = 0
    for t in range(max_iter):
        for i in range(len(train_set)):
            y = train_labels[i]
            x = np.append(train_set[i], 1)
            if y == 0:
                y = -1
            y_star = np.sign(np.dot(W, x))
            if y_star != y:
                W = np.add(W, learning_rate * y * x)
    b = W[-1]
    W = np.delete(W, -1)
    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    W, b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    predicts = []
    for i in range(len(dev_set)):
        img = dev_set[i]
        f = np.dot(W, img) + b
        if f < 0:
            predicts.append(0)
        else:
            predicts.append(1)
    return predicts

def classifyKNN(train_set, train_labels, dev_set, k):
    # TODO: Write your code here
    predicts = []
    for test_img in dev_set:
        nearest_n = []
        n = []
        for i in range(len(train_set)):
            train_img = train_set[i]
            dist = np.sum(np.square(np.subtract(train_img, test_img)))
            n.append((dist, train_labels[i]))
        n.sort(key=lambda x: x[0])
        for i in range(k):
            nearest_n.append(n[i][1])
        animal_count = 0
        not_animal_count = 0
        for neighbor in nearest_n:
            if neighbor == 1:
                animal_count = animal_count + 1
            else:
                not_animal_count = not_animal_count + 1
        if animal_count > not_animal_count:
            predicts.append(1)
        else:
            predicts.append(0)
    return predicts
