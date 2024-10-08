# implementing viterbi algorithm
import math
import os
import numpy as np
from data_loader import load_data_2_dicts
from Levenshtein import distance
import time

vocab, _, bigram, _ = load_data_2_dicts('data')

# for viterbi algorithm we also need to find the predecessors for each state
bigram_path = os.path.join('data', 'bigram_counts.txt')
reversed_bigram = {}
with open(bigram_path, 'r') as f:
    for line in f:
        past, present, prob = line.split(" ")
        past, present, prob = int(past), int(present), float(prob)
        if present not in reversed_bigram.keys():
            reversed_bigram[present] = []
        reversed_bigram[present].append(past)
# replace states with no connection with empty lists
for i in vocab.keys():
    if i not in reversed_bigram.keys():
        reversed_bigram[i] =[]

def emission_prob(u, v):
    """caculates the emission probabilities

    Args:
        u (str): emission
        v (str): expected word
    """
    k = distance(u, v)
    lamda = 0.01
    return k * np.log10(lamda) - np.log10(math.factorial(k)) 

def viterbi(obsv_seq):

    obsv_seq = obsv_seq.split(' ')
    
    # viterbi functions
    num_states = list(vocab.keys())[-1]
    num_obsv = len(obsv_seq)
    viterbi_function = np.full((num_states, num_obsv), fill_value=-np.inf)
    viterbi_max_states = np.full((num_states, num_obsv), fill_value=0)
    # print(viterbi_max_states)

    # initialization for t = 1
    t = 1
    initial_state = '<s>'
    initial_state_index = 153
    for state in bigram[initial_state_index].keys():
        viterbi_function[state - 1, t-1] = bigram[initial_state_index][state] + emission_prob(u = obsv_seq[t-1], v = vocab[state])
    
    # iteration
    for t in range(2, num_obsv + 1):
        
        # travers through each state
        for state in vocab.keys():
            max_prob = float('-inf')
            max_state = 0

            # for all the states that links to this state
            for previous_state in reversed_bigram[state]:
                prob = viterbi_function[previous_state - 1, t-2] + bigram[previous_state][state]
                if prob > max_prob:
                    max_prob = prob
                    max_state = previous_state -1
            # add the emission probability
            max_prob += emission_prob(u = obsv_seq[t-1], v = vocab[state])
            
            # add the viterbi table
            viterbi_function[state - 1, t-1] = max_prob
            viterbi_max_states[state - 1, t-1] = max_state
    # print(viterbi_function[50:55,:])

    # termination
    best_last_prob = np.max(viterbi_function[:, -1])
    best_last_state = np.argmax(viterbi_function[:, -1]) + 1

    # back tracking
    best_path = np.zeros(num_obsv)
    best_path[-1] = best_last_state
    for t in range(num_obsv-2, -1, -1):
        best_path[t] = viterbi_max_states[int(best_path[t + 1]) - 1, t + 1] + 1
    
    # convert best path into words
    corrected_sentences = str()
    for i in best_path:
        corrected_sentences = corrected_sentences + vocab[i] + " "

    return corrected_sentences, best_path, best_last_prob



if __name__ == "__main__":
    # print(emission_prob("hello", "haalo"))   
    sentences = ["I think hat twelve thousand pounds",
                 "she haf heard them", 
                 "She was ulreedy quit live",
                 "John Knightly wasn’t hard at work",
                 "he said nit word by" ]

    count = 1
    for sentence in sentences:
        start = time.time()
        corrected_sentence, _, max_prob = viterbi(sentence)
        end = time.time()
        print("Sentence \# ", count)
        print("_______________")
        print("Incorrect Sentence       : ", sentence)
        print("Corrected Sentence       : ", corrected_sentence)
        print("Max Prob                 : ", max_prob)
        print("Time taken to processe   : ", end - start)
        print("\n\n")
        count += 1