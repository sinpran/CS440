"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

import math
from collections import Counter
import numpy as np

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    word_dict = {}
    tag_counter = Counter()
    for sentence in train:
        for item in sentence:
            word = item[0]
            tag = item[1]
            tag_counter.update({tag:1})
            if word not in word_dict:
                word_dict[word] = {tag:1}
            elif tag not in word_dict[word]:
                word_dict[word][tag] = 1
            else:
                word_dict[word][tag] = word_dict[word][tag] + 1
    predicts = []
    for sentence in test:
        sentence_list = []
        for word in sentence:
            if word in word_dict:
                tag = max(word_dict[word], key=word_dict[word].get)
            else:
                tag = max(tag_counter, key=tag_counter.get)
            sentence_list.append((word,tag))
        predicts.append(sentence_list)
    return predicts
