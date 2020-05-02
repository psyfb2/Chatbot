# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""
import numpy as np
import tensorflow as tf
import textpreprocessing as pre
from math import log
from copy import copy
from functools import reduce

def beam_search(persona, msg, process_inputs, pred_function, max_reply_length, beam_length = 3):
    '''
    Beam Searh Framework

    Parameters
    ----------
    persona : str
        input persona
    msg : str
        input message
    process_inputs : function
        process_inputs(persona, msg) -> tokenized inputs which will be passed
        to pred_function
    pred_function : function
        pred_function(inputs, state, timestep) -> out_layer, new_state
        out_layer must be a 1D logits tensor with vocab size elements
        state will be None for the first calls to pred_function
        timestep is which output word is currently being predicted
    max_reply_length : int
        maximum number of tokens in the reply's
    beam_length : int, optional
        beam width to use The default is 3.

    Returns
    -------
    replys : list of str
        reply's from most likely to least likely

    '''
    
    '''
    No built in implementation of beam search for keras models so build our own, works by
    1. find beam length most likely next words for each of previous beam length
       network fragments
       
    2. find most likely beam length words from (beam length * vocab_size) possibilities
       using summed log likelihood probability
    
    3. save hidden state, ascociated output tokens and current probability
       for each most likely beam length token
    
    4. if the output token is EOS or out_seq_length reached make this beam a dead end
    
    5. repeat until all beams are dead ends
    
    6. pick most likely beam lenght sequences according to length normalized
       log likelihood objective function
    '''
    inputs = process_inputs(persona, msg)
    
    # beams will store [ [probability, state, word1, word2, ...], ... ]
    state = None
    beams = [ [0.0, state, pre.START_SEQ_TOKEN] for i in range(beam_length)]
    
    # store beam length ^ 2 most likely words [ [probability, word_index, beam_index], ... ]
    most_likely_words_all = [[0.0, 0, 0] for i in range(beam_length * beam_length)]
    
    timestep = 0
    
    beam_finished = lambda b : b[-1] == pre.END_SEQ_TOKEN or len(b) - 2 >= max_reply_length
    while not reduce(lambda a, b : a and b , map(beam_finished, beams)):
    
        # find beam length most likely words out of all beams 
        # (vocab size * beam length) possibilities
        for b_index in range(len(beams)):
            b = beams[b_index]
            prev_word = b[-1]
            
            if prev_word == pre.END_SEQ_TOKEN:
                # dead end beam so don't generate a new token, update states 
                # and leave most_likely_words for this beam constant
                continue
            
            logits, b[1] = pred_function(inputs)
            decoder_input = tf.expand_dims([tokenizer.word_index[prev_word]], 0)
            out_softmax_layer, _, b[context_vec], *b[initial_state] = decoder_model([decoder_input, encoder_out, b[context_vec], b[initial_state]])
            
            # store beam length most likely words and there probabilities for this beam
            out_softmax_layer = tf.nn.softmax(out_softmax_layer[0]).numpy()
            most_likely_indicies = out_softmax_layer.argsort()[-beam_length:][::-1]
            
            i_ = 0
            for i in range(beam_length * b_index, beam_length * (b_index + 1) ):
                # summed log likelihood probability
                most_likely_words_all[i][0] = b[prob] + log(
                    out_softmax_layer[most_likely_indicies[i_]]) 
                
                # word_index in tokenizer
                most_likely_words_all[i][1] = most_likely_indicies[i_]
                
                # beam index
                most_likely_words_all[i][2] = b_index
                i_ += 1
            
        if prev_word == pre.START_SEQ_TOKEN:
            # on first run of beam search choose beam length most likely unique words
            # as this will prevent simply running greedy search beam length times
            most_likely_words = most_likely_words_all[:beam_length]
        else:
            # chose beam length most likely words out of beam length ^ 2 possible words
            # by their length normalized log likelihood probabilities descending
            # using a beam penalty of 1.0
            most_likely_words = sorted(
                most_likely_words_all, key = lambda l:l[0] / (len(beams[l[2]]) - context_vec - 1), reverse=True)[:beam_length]
        
            # most_likely_words must remain constant for dead end beams
            # so make sure index in most_likely_words == b_index for dead end beams
            for i in range(len(most_likely_words)):
                if beams[most_likely_words[i][2]][-1] == pre.END_SEQ_TOKEN and i != most_likely_words[i][2]:
                    # swap index of dead end in beams with i in most_likely_words
                    temp = most_likely_words[i]
                    most_likely_words[i] = most_likely_words[most_likely_words[i][2]]
                    most_likely_words[most_likely_words[i][2]] = temp
                    
            
        # save network fragments for the most likely words
        temp_beams = [[] for i in range(beam_length)]
        for i in range(len(most_likely_words)):
            old_beam = copy(beams[most_likely_words[i][2]])
            old_beam[prob] = most_likely_words[i][0]
            
            # prevent EOS token being continually appended for dead end beams
            if old_beam[-1] != pre.END_SEQ_TOKEN:
                old_beam.append(pre.index_to_word(most_likely_words[i][1], tokenizer))
            
            # beam state was already saved in the first step
            
            temp_beams[i] = old_beam
            
        beams = temp_beams
        
        timestep += 1
    
    # sort beams by length normalized log likelihood descending and only keep text
    replys = [" ".join(b[context_vec + 2:-1]) 
              for b in sorted(beams, key = lambda b:b[0] / (len(b) - context_vec - 1), reverse=True)]
    
    # return list of beam length reply's from most likely to least likely
    return replys

