"""

Ze Xuan Ong
1/21/19
Minor modifications to make it more Pythonic and to streamline piping

David Bamman
2/14/14

Python port of train_hmm.pl:

Noah A. Smith
2/21/08
Code for maximum likelihood estimation of a bigram HMM from
column-formatted training data.

Usage:  train_hmm.py tags-file text-file hmm-file

The training data should consist of one line per sequence, with
states or symbols separated by whitespace and no trailing whitespace.
The initial and final states should not be mentioned; they are
implied.
The output format is the HMM file format as described in viterbi.pl.

"""

# Modified version for trigram model
# Add linear interpolation

import sys
import re
import numpy as np

from collections import defaultdict

# Files
TAG_FILE = sys.argv[1]
TOKEN_FILE = sys.argv[2]
OUTPUT_FILE = sys.argv[3]

# Vocabulary
vocab = {}
OOV_WORD = "OOV"
INIT_STATE = "init"
FINAL_STATE = "final"

# Set & list
tag_set = set()
token_set = set()
tag_list = []
token_list = []

# Transition and emission probabilities
emissions = {}
emissions_total = defaultdict(lambda: 0)

transitions = {}
transitions_total = defaultdict(lambda: 0)

tri_transitions = {}
tri_transitions_total = defaultdict(lambda: 0)

with open(TAG_FILE) as tag_file, open(TOKEN_FILE) as token_file:
    for tag_string, token_string in zip(tag_file, token_file):
        tags = re.split("\s+", tag_string.rstrip())
        tokens = re.split("\s+", token_string.rstrip())
        pairs = zip(tags, tokens)

        prevtag = INIT_STATE
        prev_prevtag = INIT_STATE

        for (tag, token) in pairs:

            # this block is a little trick to help with out-of-vocabulary (OOV)
            # words.  the first time we see *any* word token, we pretend it
            # is an OOV.  this lets our model decide the rate at which new
            # words of each POS-type should be expected (e.g., high for nouns,
            # low for determiners).

            tag_set.add(tag)
            token_set.add(token)
            tag_list.append(tag)
            token_list.append(token)

            if token not in vocab:
                vocab[token] = 1
                token = OOV_WORD

            if tag not in emissions:
                emissions[tag] = defaultdict(lambda: 0)
            if prevtag not in transitions:
                transitions[prevtag] = defaultdict(lambda: 0)
            if (prev_prevtag, prevtag) not in tri_transitions:
                tri_transitions[(prev_prevtag, prevtag)] = defaultdict(lambda: 0)

            # increment the emission/transition observation
            emissions[tag][token] += 1
            emissions_total[tag] += 1
            
            transitions[prevtag][tag] += 1
            transitions_total[prevtag] += 1

            tri_transitions[(prev_prevtag, prevtag)][tag] += 1
            tri_transitions_total[(prev_prevtag, prevtag)] += 1

            prev_prevtag = prevtag
            prevtag = tag

        # don't forget the stop probability for each sentence
        if prevtag not in transitions:
            transitions[prevtag] = defaultdict(lambda: 0)

        transitions[prevtag][FINAL_STATE] += 1
        transitions_total[prevtag] += 1

        if (prev_prevtag, prevtag) not in tri_transitions:
            tri_transitions[(prev_prevtag, prevtag)] = defaultdict(lambda: 0)

        tri_transitions[(prev_prevtag, prevtag)][FINAL_STATE] += 1
        tri_transitions_total[(prev_prevtag, prevtag)] += 1

# Define probability
uni_gram_prob = {}
bi_gram_prob = {}
tri_gram_prob = {}

bi_gram_count = {}
tri_gram_count = {}

for tag in tag_set:
    uni_gram_prob[tag] = tag_list.count(tag) / len(tag_list)

for prevtag in transitions:
    for tag in transitions[prevtag]:
        bi_gram_count[(prevtag, tag)] = transitions[prevtag][tag]
        bi_gram_prob[(prevtag, tag)] = float(transitions[prevtag][tag] / transitions_total[prevtag])

for (prev_prevtag, prevtag) in tri_transitions:
    for tag in tri_transitions[(prev_prevtag, prevtag)]:
        tri_gram_count[(prev_prevtag, prevtag, tag)] = tri_transitions[(prev_prevtag, prevtag)][tag]
        tri_gram_prob[(prev_prevtag, prevtag, tag)] = float(tri_transitions[(prev_prevtag, prevtag)][tag] / tri_transitions_total[(prev_prevtag, prevtag)])

# Define probability with imputation
tag_tot_set = tag_set
tag_tot_set.add(INIT_STATE)
tag_tot_set.add(FINAL_STATE)

uni_gram_prob = {}
bi_gram_prob = {}
tri_gram_prob = {}

bi_gram_count = {}
tri_gram_count = {}

for tag in tag_tot_set:
    try:
        uni_gram_prob[tag] = tag_list.count(tag) / len(tag_list)
    except KeyError as e:
        uni_gram_prob[tag] = float(1e-10)

for prevtag in tag_tot_set:
    for tag in tag_tot_set:
        try:
            bi_gram_count[(prevtag, tag)] = transitions[prevtag][tag]
            bi_gram_prob[(prevtag, tag)] = float(transitions[prevtag][tag] / transitions_total[prevtag])
        except KeyError as e:
            bi_gram_prob[(prevtag, tag)] = float(1e-10)

for prev_prevtag in tag_tot_set:
    for prevtag in tag_tot_set:
        for tag in tag_tot_set:
            try:
                tri_gram_count[(prev_prevtag, prevtag, tag)] = tri_transitions[(prev_prevtag, prevtag)][tag]
                tri_gram_prob[(prev_prevtag, prevtag, tag)] = float(
                    tri_transitions[(prev_prevtag, prevtag)][tag] / tri_transitions_total[(prev_prevtag, prevtag)])
            except KeyError as e:
                tri_gram_prob[(prev_prevtag, prevtag, tag)] = float(1e-10)

# Find linear interpolation parameters
N_token = len(token_list)

lambda_1, lambda_2, lambda_3 = 0, 0, 0

for prev_prevtag in tag_tot_set:
    for prevtag in tag_tot_set:
        for tag in tag_tot_set:
            f_123 = tri_gram_prob[(prev_prevtag, prevtag, tag)]
            f_12 = bi_gram_prob[(prev_prevtag, prevtag)]
            f_23 = bi_gram_prob[(prevtag, tag)]
            f_2 = uni_gram_prob[prevtag]
            f_3 = uni_gram_prob[tag]

            cases ={}
            cases['case_3'] = (f_123 - 1) / (f_12 - 1)
            cases['case_2'] = (f_23 - 1) / (f_2 - 1)
            cases['case_1'] = (f_3 - 1) / (N_token - 1)

            max_case = max(cases, key=cases.get)
            if max_case == 'case_3':
                lambda_3 += tri_gram_prob[(prev_prevtag, prevtag, tag)]
            elif max_case == 'case_2':
                lambda_2 += tri_gram_prob[(prev_prevtag, prevtag, tag)]
            elif max_case == 'case_1':
                lambda_1 += tri_gram_prob[(prev_prevtag, prevtag, tag)]

lambdas = np.array([lambda_1, lambda_2, lambda_3]) / (lambda_1 + lambda_2 + lambda_3)

# Assign probability
uni_gram_prob_int = {}
bi_gram_prob_int = {}
tri_gram_prob_int = {}

for tag in tag_tot_set:
    try:
        uni_gram_prob_int[tag] = uni_gram_prob[tag]
        if uni_gram_prob_int[tag] == 0:
            uni_gram_prob_int[tag] = float(1e-10)
    except KeyError as e:
        uni_gram_prob_int[tag] = float(1e-10)

uni_gram_prob = uni_gram_prob_int

for prevtag in tag_tot_set:
    for tag in tag_tot_set:
        try:
            bi_gram_prob_int[(prevtag, tag)] = bi_gram_prob[(prevtag, tag)]
            if bi_gram_prob_int[(prevtag, tag)] == 0:
                bi_gram_prob_int[(prevtag, tag)] = float(1e-10)
        except KeyError as e:
            bi_gram_prob_int[(prevtag, tag)] = float(1e-10)

bi_gram_prob = bi_gram_prob_int

for prev_prevtag in tag_tot_set:
    for prevtag in tag_tot_set:
        for tag in tag_tot_set:
            try:
                # tri_gram_prob[(prev_prevtag, prevtag, tag)] = tri_gram_prob[(prev_prevtag, prevtag, tag)]
                p_1 = uni_gram_prob[tag]
                p_2 = bi_gram_prob[(prevtag, tag)]
                p_3 = tri_gram_prob[(prev_prevtag, prevtag, tag)]
                tri_gram_prob_int[(prev_prevtag, prevtag, tag)] = lambdas[0] * p_1 + lambdas[1] * p_2 + lambdas[2] * p_3
                if tri_gram_prob_int[(prev_prevtag, prevtag, tag)] == 0:
                    tri_gram_prob_int[(prev_prevtag, prevtag, tag)] = float(1e-10)
            except KeyError as e:
                tri_gram_prob_int[(prev_prevtag, prevtag, tag)] = float(1e-10)

tri_gram_prob = tri_gram_prob_int


# Write output to output_file
with open(OUTPUT_FILE, "w") as f:
    for prevtag in tag_tot_set:
        for tag in tag_tot_set:
            f.write("trans {} {} {}\n"
                    .format(prevtag, tag, bi_gram_prob[(prevtag, tag)]))

    for prev_prevtag in tag_tot_set:
        for prevtag in tag_tot_set:
            for tag in tag_tot_set:
                f.write("tri_trans {} {} {} {}\n"
                        .format(prev_prevtag, prevtag, tag, tri_gram_prob[(prev_prevtag, prevtag, tag)]))

    for tag in emissions:
        for token in emissions[tag]:
            f.write("emit {} {} {}\n"
                    .format(tag, token, emissions[tag][token] / emissions_total[tag]))

