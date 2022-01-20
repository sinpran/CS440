# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for MP3. You should only modify code
within this file and the last two arguments of line 34 in mp3.py
and if you want-- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as numpy
import math
from collections import Counter


def naiveBayes(train_set, train_labels, dev_set, smooth_p=1.0, pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smooth_p - The smoothing parameter --laplace (1.0 by default)
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set

    positive_c = Counter()
    negative_c = Counter()
    positive_N = 0
    negative_N = 0
    for i in range(0, len(train_set)):
        doc = train_set[i]
        if train_labels[i] == 1:
            for word in doc:
                positive_c.update({word:1})
                positive_N = positive_N+1
        if train_labels[i] == 0:
            for word in doc:
                negative_c.update({word:1})
                negative_N = negative_N+1

    predicted_labels = []
    for i in range(0, len(dev_set)):
        doc = dev_set[i]
        positive_posterior = math.log(pos_prior)
        negative_posterior = math.log(1-pos_prior)
        for word in doc:
            likelihood = 0
            unlikelihood = 0
            if word in positive_c:
                likelihood = (positive_c[word] + smooth_p)/(positive_N + (smooth_p*len(positive_c)))
            else:
                likelihood = smooth_p/(positive_N + (smooth_p*len(positive_c)))
            if word in negative_c:
                unlikelihood = (negative_c[word] + smooth_p)/(negative_N + (smooth_p*len(negative_c)))
            else:
                unlikelihood = smooth_p/(negative_N + (smooth_p*len(negative_c)))
            positive_posterior = positive_posterior + math.log(likelihood)
            negative_posterior = negative_posterior + math.log(unlikelihood)
        if positive_posterior > negative_posterior:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)
    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)
    print(predicted_labels)
    return predicted_labels

def bigramBayes(train_set, train_labels, dev_set, unigram_smooth_p=1.0, bigram_smooth_p=1.0, bigram_lambda=0.5,pos_prior=0.8):

    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    unigram_smooth_p - The smoothing parameter for unigram model (same as above) --laplace (1.0 by default)
    bigram_smooth_p - The smoothing parameter for bigram model (1.0 by default)
    bigram_lambda - Determines what fraction of your prediction is from the bigram model and what fraction is from the unigram model. Default is 0.5
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set using a bigram model
    positive_c_uni = Counter()
    negative_c_uni = Counter()
    positive_c_bi = Counter()
    negative_c_bi = Counter()
    positive_N_uni, negative_N_uni, positive_N_bi, negative_N_bi = 0
    for i in range(0, len(train_set)):
        doc = train_set[i]
        if train_labels[i] == 1:
            for j in range(0, len(doc)):
                word1 = doc[j]
                positive_c_uni.update({word1 : 1})
                positive_N_uni = positive_N_uni + 1
                if j+1 < len(doc):
                    word2 = doc[j+1]
                    positive_c_bi.update({word1 + " " + word2 : 1})
                    positive_N_bi = positive_N_bi + 1
        if train_labels[i] == 0:
            for j in range(0, len(doc)):
                word1 = doc[j]
                negative_c_uni.update({word1 : 1})
                negative_N_uni = negative_N_uni + 1
                if j+1 < len(doc):
                    word2 = doc[j+1]
                    negative_c_bi.update({word1 + " " + word2 : 1})
                    negative_N_bi = negative_N_bi + 1


    predicted_labels = []
    for i in range(0, len(dev_set)):
        doc = dev_set[i]
        positive_posterior1 = math.log(pos_prior)
        negative_posterior1 = math.log(1-pos_prior)
        positive_posterior2 = math.log(pos_prior)
        negative_posterior2 = math.log(1-pos_prior)
        for j in range(0, len(doc)):
            likelihood1 = 0
            unlikelihood1 = 0
            word1 = doc[j]
            if word1 in positive_c_uni:
                likelihood1 = (positive_c_uni[word1] + unigram_smooth_p)/(positive_N_uni + (unigram_smooth_p*len(positive_c_uni)))
            else:
                likelihood1 = unigram_smooth_p/(positive_N_uni + (unigram_smooth_p*len(positive_c_uni)))
            if word1 in negative_c_uni:
                unlikelihood1 = (negative_c_uni[word1] + unigram_smooth_p)/(negative_N_uni + (unigram_smooth_p*len(negative_c_uni)))
            else:
                unlikelihood1 = unigram_smooth_p/(negative_N_uni + (unigram_smooth_p*len(negative_c_uni)))
            if j+1 < len(doc):
                likelihood2 = 0
                unlikelihood2 = 0
                word2 = doc[j+1]
                bigram = word1 + " " + word2
                if bigram in positive_c_bi:
                    likelihood2 = (positive_c_bi[bigram] + bigram_smooth_p)/(positive_N_bi + (bigram_smooth_p*len(positive_c_bi)))
                else:
                    likelihood2 = bigram_smooth_p/(positive_N_bi + (bigram_smooth_p*len(positive_c_bi)))
                if bigram in negative_c_bi:
                    unlikelihood2 = (negative_c_bi[bigram] + bigram_smooth_p)/(negative_N_bi + (bigram_smooth_p*len(negative_c_bi)))
                else:
                    unlikelihood2 = bigram_smooth_p/(negative_N_bi + (bigram_smooth_p*len(negative_c_bi)))

            positive_posterior1 = positive_posterior1 + math.log(likelihood1)
            negative_posterior1 = negative_posterior1 + math.log(unlikelihood1)
            positive_posterior2 = positive_posterior2 + math.log(likelihood2)
            negative_posterior2 = negative_posterior2 + math.log(unlikelihood2)

        positive_prob_bigram = ((1-bigram_lambda)*positive_posterior1)+(bigram_lambda*positive_posterior2)
        negative_prob_bigram = ((1-bigram_lambda)*negative_posterior1)+(bigram_lambda*negative_posterior2)
        if positive_prob_bigram > negative_prob_bigram:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)
    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)
    print(predicted_labels)
    return predicted_labels
