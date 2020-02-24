# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""

import numpy as np
import string
import pickle
import re

TRAIN_FN       = "data/train_self_original_no_cands.txt"
TEST_FN        = "data/test_self_original_no_cands.txt"
TRAIN_PKL_FN   = "data/train_set.pkl"
TEST_PKL_FN    = "data/test_set.pkl"
MODEL_IMAGE_FN = "models/model.png"
MODEL_FN       = "models/model.h5"

'''
Data is in the following format
1 persona 
2 persona 
3 persona 
4 persona
5 persona (sometimes)
5 human \t bot
6 human \t bot
7 human \t bot
8 human \t bot
9 human \t bot
1 persona
2 persona
...

human to bot utterances can range in the number of replies per persona.
This is the no_cands version, which means candidate replies which are not
the ground truth are included within possible replies of the bot within the dataset. 
This is useful when training for multi-task problem to produce utterance sequence 
but also classify which reply from the candidates is the ground truth.
https://github.com/huggingface/transfer-learning-conv-ai/blob/master/example_entry.py

'''

def remove_contractions(sentence):
    sentence = re.sub(r"won\'t", "will not", sentence)
    sentence = re.sub(r"can\'t", "can not", sentence)

    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    return sentence

''' Returns the index of a single persona within a list of already stored personas 
    ValueError exception if the single_pers cannot be found in personas '''
def get_persona_index(personas, single_pers):
    # each persona has 5 sentences however in the dataset most times the fifth 
    # sentence is excluded for whatever reason. Also the same personas
    # have the same sentences but in different order and sometimes with different
    # punctuation and spelling
    
    #if single_pers == a:
        #breakpoint()
    
    # first number character for comparison
    single_pers_re = [sentence[1:] for sentence in single_pers]
    
    shared_sentences = len(single_pers) - 1
    
    for p_index, persona in enumerate(personas):
        persona_re = [sentence[1:] for sentence in persona]
        # different personas don't share the same 4 sentences
        count = 0
        for sentence in single_pers_re:
            if sentence in persona_re:
                count += 1
                if count >= shared_sentences:
                    # these are the same personas, save one with 5 sentences and return index
                    if(len(single_pers) == 5 and len(persona) != 5):
                        personas[p_index] = single_pers
                    return p_index
    raise ValueError("Could not find persona {}".format(single_pers))
                    
''' Returns 
    persona array [["persona_line_1, ... persona_line_4"], ...]
    array of message, reply, persona index triplets e.g. [["hi how are you", "hi good", 5], ...]
'''
def load_dataset(filename):
    personas = [] 
    conversations = []
    
    with open(filename, 'rt') as lines:
        single_pers = []
        # read text file line by line
        for line in lines:
            line = line.strip()
            if "your persona:" in line and line[0] in ['1', '2', '3', '4', '5']:
                single_pers.append(line)
            else:
                # add persona to list of personas
                if len(single_pers) != 0:
                    personas.append(single_pers)
                    p_index = len(personas) - 1
                    single_pers = []
                # line is a message and reply seperated by tab
                # which is ascociated with the last read persona
                pair = line.split('\t')
                conversations.append([pair[0], pair[1], p_index])
    return personas, conversations
    
''' Takes [["message text", "reply text", pindex], ...] 
    and returns the cleaned version in numpy array '''
def clean_triples(msg_reply):
    cleaned = []
    
    for triple in msg_reply:
        clean_msg_reply = []
        
        # triple[0] is message starting with a number
        # triple[1] is the reply to learn 
        # triple[2] the persona index
        triple[0] = clean_line(remove_first_num(triple[0]))
        clean_msg_reply.append(triple[0])
        triple[1] = clean_line(triple[1])
        clean_msg_reply.append(triple[1])
        clean_msg_reply.append(triple[2])
        
        cleaned.append(clean_msg_reply)
        
    return np.array(cleaned)
        

''' Clean a single line of text by
    removing non-printable characters
    make lower case
    removing punctuation apart from full stop and comma 
    removes contractions e.g. i've -> i have'''
def clean_line(line):
    # only include printable characters and remove punctuation
    # apart from full stop and comma characters
    punc = string.punctuation
    punc = punc.replace('.', "")
    punc = punc.replace(',', "")

    re_punc = re.compile('[%s]' % re.escape(punc))
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    
    line = remove_contractions(line)
    line = line.split()
    # make lower case
    line = [word.lower() for word in line]
    # remove punctuation
    line = [re_punc.sub('', w) for w in line]
    # remove non-printable chars
    line = [re_print.sub('', w) for w in line]
    
    return ' '.join(line)
    
''' Remove the first number from a string '''
def remove_first_num(strr):
    for i in range(len(strr)):
        if strr[i] in string.digits:
            # found the first number
            count = 1
            while True:
                if i + count >= len(strr) or strr[i + count] not in string.digits:
                    indicies = [x for x in range(i, i + count)]
                    cpy = [strr[j] for j in range(len(strr)) if j not in indicies]
                    return ''.join(cpy)
                count += 1
    # string does not contain any numbers
    return strr

''' Save an object using pickle as the filename '''   
def save_object(obj, filename, verbose=1):
    pickle.dump(obj, open(filename, 'wb'))
    if verbose == 1:
        print("Saved: %s" % filename)

''' Load a pickle object from file '''
def load_object(filename):
    return pickle.load(open(filename, 'rb'))

''' Given a list of cleaned lines ["sentence 1", "sentence 2", ...] 
    returns sentence with max num of words '''
def max_seq_length(lines):
    return max([len(line.split()) for line in lines])

''' Given a list of cleaned lines ["sentence 1", "sentence 2", ...] 
    returns the third quartile length of the sentences '''
def third_quartile_seq_length(lines):
    lengths = sorted([len(line.split()) for line in lines])
    return lengths[3 * (len(lengths) // 4)]

''' Given a list of cleaned lines ["sentence 1", "sentence 2", ...] 
    returns the mean length of the sentences '''
def mean_seq_length(lines):
    lengths = [len(line.split()) for line in lines]
    return sum(lengths) // len(lengths)

''' Saves the full dataset, train and test set 
    as numpy array of ([["message", "reply", p_index], ...], personas) in pickle files'''
def save_dataset(load_fn, save_fn):
    personas, msg_reply = load_dataset(load_fn)
    
    # clean the personas and message, reply pairs
    triples = clean_triples(msg_reply)
    for i in range(len(personas)):
        for j in range(len(personas[i])):
            personas[i][j] = clean_line(remove_first_num(personas[i][j]))
        personas[i] = ' '.join(personas[i])
    personas = np.array(personas)
    
    save_object((triples, personas), save_fn)
    
    # check that cleaned text is as intended
    for i in range(100):
        print('[%s]\n[%s] => [%s]' % (personas[int(triples[i, 2])], triples[i, 0], triples[i, 1]))
        print("\n")

if __name__ == '__main__':
    save_dataset(TRAIN_FN, TRAIN_PKL_FN)