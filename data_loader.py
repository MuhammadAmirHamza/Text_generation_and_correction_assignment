import os

def load_data_2_dicts(path = None):
    """this function takes the path of the data containing the data files for the project of text generation:
    the folder must consist of these four files:
    1. vocab.txt
    2. unigram.txt
    3. bigram.txt
    4. trigram.txt

    Args:
        path (_type_, optional): the path of folder where files are located. Defaults to None if the data files are present in the same folder.

    Returns:
        dict: 
    """
    if path is not None:
        vocab_path = os.path.join(path, 'vocab.txt')
        unigram_path = os.path.join(path, 'unigram_counts.txt')
        bigram_path = os.path.join(path, 'bigram_counts.txt')
        trigram_path = os.path.join(path, 'trigram_counts.txt')
    else:
        vocab_path = 'vocab.txt'
        unigram_path = 'unigram_counts.txt'
        bigram_path = 'bigram_counts.txt'
        trigram_path = 'trigram_counts.txt'
        
    # loading the data
    words_mapping = {}
    with open(vocab_path, 'r') as file:
        for x in file:
            index, value = x.split(" ")
            index = int(index)
            value = value[: -1]
            if index not in words_mapping.keys():
                words_mapping[index] = value
    #print(words_mapping[152])

    unigram_data = {}
    with open(unigram_path, 'r') as file:
        for x in file:
            index, prob = x.split(" ")
            index = int(index)
            prob = float(prob)
            if index not in unigram_data.keys():
                unigram_data[index] = prob
    # print(unigram_data)


    bigram_data = {}
    with open(bigram_path, 'r') as file:
        for x in file:
            past, present, prob = x.split(" ")
            past = int(past)
            present = int(present)
            prob = float(prob)
            if past not in bigram_data.keys():
                bigram_data[past] = {}
            bigram_data[past][present] = prob
    # print(bigram_data[2])

    trigram_data = {}
    with open(trigram_path, 'r') as file:
        for x in file:
            past_2, past_1, present, prob = x.split(" ")
            past_2, past_1, present, prob = int(past_2), int(past_1), int(present), float(prob)
            if past_2 not in trigram_data.keys():
                trigram_data[past_2] = {}
            if past_1 not in trigram_data[past_2].keys():
                trigram_data[past_2][past_1] = {}
            trigram_data[past_2][past_1][present] = prob
    #print(trigram_data[28][738])

    return words_mapping, unigram_data, bigram_data, trigram_data

if __name__ == "__main__":
    a, b, c, d = load_data_2_dicts('data')
    print(a)
