"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import math
from collections import Counter
import numpy as np


def viterbi_1(train, test):
	'''
	TODO: implement the simple Viterbi algorithm. This function has time out limitation for 3 mins.
	input:  training data (list of sentences, with tags on the words)
			E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
			test data (list of sentences, no tags on the words)
			E.g [[word1,word2...]]
	output: list of sentences with tags on the words
			E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
	'''

	tag_counter = Counter()
	for sentence in train:
		for pair in sentence:
			tag = pair[1]
			tag_counter.update({tag : 1})
	transition = {}
	transition["s"] = Counter()
	emission = {}
	vocab = Counter()
	for tag in tag_counter:
		transition[tag] = Counter()
		for tag2 in tag_counter:
			transition[tag].update({tag2 : 0})
		transition["s"].update({tag : 0})
		emission[tag] = Counter()
	for sentence in train:
		for j in range(len(sentence)):
			word1 = sentence[j][0]
			tag1 = sentence[j][1]
			if j == 0:
				transition["s"].update({tag1 : 1})
			if j+1 < len(sentence):
				tag2 = sentence[j+1][1]
				transition[tag1].update({tag2 : 1})
			emission[tag1].update({word1 : 1})
			vocab.update({word1 : 1})

	k = 0.00001
	unseen = {}
	for tag1 in transition:
		if tag1 == "s":
			N = len(train)
		else:
			N = tag_counter[tag1]
		X = len(transition[tag1])
		for tag2 in transition[tag1]:
			count = transition[tag1][tag2]
			transition[tag1][tag2] = math.log((count + k)/(N + k*X))
		if tag1 != "s":
			N = tag_counter[tag1]
			X = len(emission[tag1])
			for word in emission[tag1]:
				count = emission[tag1][word]
				emission[tag1][word] = math.log((count + k)/(N + k*(X+1)))
			unseen[tag1] = math.log(k/(N + k*(X+1)))

	predicts = []
	count = 0
	for sentence in test:
		viterbi = []
		backpointer = []
		predicted_sent = []
		viterbi.append({})
		backpointer.append({})
		for tag in emission:
			if sentence[0] in emission[tag]:
				viterbi[0][tag] = transition["s"][tag] + emission[tag][sentence[0]]
			else:
				viterbi[0][tag] = transition["s"][tag] + unseen[tag]
			backpointer[0][tag] = None
		for t in range(1, len(sentence)):
			word = sentence[t]
			viterbi.append({})
			backpointer.append({})
			for tag in emission:
				if word in emission[tag]:
					b = emission[tag][word]
				else:
					b = unseen[tag]
				maximum = viterbi_max(viterbi[t-1], tag, transition, b)
				viterbi[t][tag] = maximum[1]
				backpointer[t][tag] = maximum[0]
				count = maximum[2] + count

		best_path = []
		best_prob = 0
		for end in backpointer[-1]:
			temp_path = []
			temp_path.append(end)
			temp_prob = viterbi[-1][end]
			prev = backpointer[-1][end]
			for t in reversed(range(0, len(backpointer)-1)):
				temp_path.append(prev)
				temp_prob = temp_prob + viterbi[t][prev]
				prev = backpointer[t][prev]
			if len(best_path) == 0 or temp_prob > best_prob:
				best_path = temp_path
				best_prob = temp_prob
		for i in range(len(sentence)):
			predicted_sent.append((sentence[i], best_path[len(best_path)-i-1]))
		predicts.append(predicted_sent)
	print(count)
	return predicts

def viterbi_max(viterbi, cur_tag, a, b):
	max_tag = cur_tag
	max_val = viterbi[cur_tag]+a[cur_tag][cur_tag]+b
	count = 0
	for prev_tag in viterbi:
		if viterbi[prev_tag]+a[prev_tag][cur_tag]+b > max_val:
			max_val = viterbi[prev_tag]+a[prev_tag][cur_tag]+b
			max_tag = prev_tag
		count = count + 1
	return (max_tag, max_val, count)