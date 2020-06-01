"""
Ze Xuan Ong
14 Jan 2019

Parallel Viteribi in Python, adapted from Jocelyn Huang's version of
Viterbi implemented for NLP Assignment 6, which was adapted from
Noah A. Smith's viterbi.pl (yes that is Perl not Python)

Largest modification is with regards to the usage of multiple
processes to handle process the output text in parallel.

Usage: python viterbi.py <HMM_FILE> <TEXT_FILE> <OUTPUT_FILE>

Apart from writing the output to a file, the program also prints
the number of text lines read and processed, and the time taken
for the entire program to run in seconds. This may be useful to
let you know how much time you have to get a coffee in subsequent
iterations.

Provides about 10x speed-up based on bigram HMM. This is largely
NOT due to parallelization, but due to removal of redundant
loops. Perhaps only 5% of the speed-up can be attributed to running
in parallel, largely because of the sizes of input provided for the
assignment.

If you consider modifying this to work for a trigram model, the
speed-up may be more significant and obvious. For reference when I took
this course and modified the original Viterbi to run trigrams,
it took about 20 minutes on my machine (2.4 GHz Intel Core i5). I
would think anything less than that is great for productivity

"""

import math
import sys
import time
import multiprocessing as mp

from collections import defaultdict

# Magic strings and numbers
NUM_PROCESSES = mp.cpu_count()     # Uses all cores available
PROCESSOR_RATIO = 2
HMM_FILE = sys.argv[1]
TEXT_FILE = sys.argv[2]
OUTPUT_FILE = sys.argv[3]
TRANSITION_TAG = "trans"
EMISSION_TAG = "emit"
OOV_WORD = "OOV"         # check that the HMM file uses this same string
INIT_STATE = "init"      # check that the HMM file uses this same string
FINAL_STATE = "final"    # check that the HMM file uses this same string

# Transition and emission probabilities
# Structured as a nested defaultdict in defaultdict, with inner defaultdict
#   returning 0.0 as a default value, since dirty KeyErrors are equivalent to
#   zero probabilities
#
# The advantage of this is that one can add redundant transition probabilities
transition = defaultdict(lambda: defaultdict(lambda: 1.0))
emission = defaultdict(lambda: defaultdict(lambda: 1.0))

# Store states to iterate over for HMM
# Store vocab to check for OOV words
states = set()
vocab = set()


# Actual Viterbi function that takes a list of lines of text as input
# The original version (and most versions) take in a single line of text.
# This is to reduce process creation/tear-down overhead by allowing us
#   to chunk up the input and divide it amongst the processes without
#   resource sharing
# NOTE: the state and vocab sets are still shared but it does not seem
#       to impact performance by much
def viterbi(lines):

    ret = [""] * len(lines)
    for (index, line) in enumerate(lines):

        w = line.split()

        # Setup Viterbi for this sentence:
        # V[x][y] where x is the index of the word in the line
        #   and y is a state in the set of states
        V = defaultdict(lambda: {})

        # Initialize backtrace so we get recover the path:
        # back[x][y] where x is the index of the word in the line
        #   and y is the previous state with the highest probability
        back = defaultdict(lambda: defaultdict(lambda: ""))

        # Initialize V with init state
        # REDUNDANT
        V[-1][INIT_STATE] = 0.0

        # Iterate over each word in the line
        for i in range(len(w)):

            # If word not in vocab, replace with OOV word
            if w[i] not in vocab:
                w[i] = OOV_WORD

            # Iterate over all possible current states:
            for q in states:

                # If this emission is impossible for this state, move on
                emission_prob = emission[q][w[i]]
                if emission_prob > 0.0 or emission_prob == 1.0:
                    continue

                # Iterate over all possible previous states:
                # Specifically, qq -> q
                for qq in states:

                    # If this transition is impossible for this state, move on
                    transition_prob = transition[qq][q]
                    if transition_prob > 0.0 or transition_prob == 1.0:
                        continue

                    # Calculate the (log) probability of:
                    # 1. The highest probability of the previous state being qq
                    # 2. The probability of transition qq -> q
                    # 3. The probability of emission q -> w[i]
                    try:
                        v = V[i - 1][qq] + transition_prob + emission_prob
                    except KeyError as e:
                        continue

                    # Replace if probability is higher
                    current = V[i].get(q, None)
                    if not current or v > V[i][q]:
                        V[i][q] = v
                        back[i][q] = qq

        # Handle final state    
        best_final_state = None
        for q in states:
            transition_prob = transition[q][FINAL_STATE]
            if transition_prob >= 0:
                continue

            try:
                v = V[len(w) - 1][q] + transition_prob
            except KeyError as e:
                continue

            # Replace if probability is higher
            current = V[len(w)].get(FINAL_STATE, None)
            if not current or v > V[len(w)][FINAL_STATE]:
                V[len(w)][FINAL_STATE] = v
                best_final_state = q

        # Backtrace from the best_final_state
        if best_final_state:

            output = []
            # Step from len(w) to 0
            for i in range(len(w) - 1, -1, -1):
                output.append(best_final_state)
                best_final_state = back[i][best_final_state]

            # Reverse the output and join as string
            ret[index] = " ".join(output[::-1])

        # If no best_final_state e.g. could not find transition to terminate
        # then return empty string
        else:
            ret[index] = ""

    # Return a list of processed lines
    return ret


# Chunk a list l into n sublists
def chunks(l, n):
    chunk_size = len(l) // n
    return [[l[i:i + chunk_size]] for i in range(0, len(l), chunk_size)]


# Main method
def main():

    # Mark start time
    t0 = time.time()

    # Read HMM transition and emission probabilities
    with open(HMM_FILE, "r") as f:
        for line in f:
            line = line.split()

            # Read transition
            # Example line: trans NN NNPS 9.026968067100463e-05
            # Read in states as qq -> q
            if line[0] == TRANSITION_TAG:
                (qq, q, trans_prob) = line[1:4]
                transition[qq][q] = math.log(float(trans_prob))
                states.add(qq)
                states.add(q)

            # Read in states as q -> w
            elif line[0] == EMISSION_TAG:
                (q, w, emit_prob) = line[1:4]
                emission[q][w] = math.log(float(emit_prob))
                states.add(q)
                vocab.add(w)

            # Ignores all other lines

    # Read lines from text file and then split by number of processes
    text_file_lines = []
    with open(TEXT_FILE, "r") as f:
        text_file_lines = f.readlines()

    # If too few lines of text, run on single process
    results = []
    if len(text_file_lines) <= NUM_PROCESSES * PROCESSOR_RATIO:
        results = viterbi(text_file_lines, states, vocab)

    # Otherwise divide workload amongst process threads
    else:
        slices = chunks(text_file_lines, NUM_PROCESSES)
        pool = mp.Pool(processes=NUM_PROCESSES)        
        
        # starmap 
        results = pool.starmap(viterbi, slices)

    # Print output to file
    with open(OUTPUT_FILE, "w") as f:
        for lines in results:
            for line in lines:
                f.write(line)
                f.write("\n")

    # Mark end time
    t1 = time.time()

    # Print info to stdout
    print("Processed {} lines".format(len(text_file_lines)))
    print("Time taken to run: {}".format(t1 - t0))

if __name__ == "__main__":    
    main()
