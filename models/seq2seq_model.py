# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""
import numpy as np
import text_preprocessing as pre
import tensorflow as tf
from math import log
from copy import copy
from functools import reduce
from attention import Attention
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import LSTM, Dense, Embedding, TimeDistributed, Input, Bidirectional, Concatenate, Dropout
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

def pre_train_seq2seq_movie(LSTM_DIM, EPOCHS, BATCH_SIZE, CLIP_NORM, DROPOUT, train_by_batch):
    vocab, persona_length, msg_length, reply_length = pre.get_vocab()
    tokenizer = pre.fit_tokenizer(vocab)
    vocab_size = len(tokenizer.word_index) + 1
    vocab = None
    
    in_seq_length = persona_length + msg_length
    out_seq_length = reply_length
    
    print('Vocab size: %d' % vocab_size)
    print('Input sequence length: %d' % in_seq_length)
    print('Output sequence length: %d' % out_seq_length)
    
    movie_conversations = pre.load_movie_dataset(pre.MOVIE_FN)
    movie_conversations = movie_conversations[:5] ######################### 
    
    encoder_input  = movie_conversations[:, 0]
    decoder_input  = np.array([pre.START_SEQ_TOKEN + ' ' + row[1] for row in movie_conversations])
    decoder_target = np.array([row[1] + ' ' + pre.END_SEQ_TOKEN for row in movie_conversations])
    
    raw = encoder_input[:20]
    
    # integer encode training data
    encoder_input  = pre.encode_sequences(tokenizer, in_seq_length, encoder_input)
    decoder_input  = pre.encode_sequences(tokenizer, out_seq_length, decoder_input)
    decoder_target = pre.encode_sequences(tokenizer, out_seq_length, decoder_target)
    
    # load GloVe embeddings
    embedding_matrix = pre.load_glove_embedding(tokenizer, pre.GLOVE_FN)
    
    # ------ seq2seq model definition and training ------ #
    n_units = LSTM_DIM
    dropout = DROPOUT
    src_timesteps = in_seq_length
    tar_timesteps = out_seq_length
    
    e_dim = embedding_matrix.shape[1]
    embedding = Embedding(vocab_size, e_dim, weights=[embedding_matrix], input_length=None , trainable=True, mask_zero=True)
    
    # LSTM 4-layer Bi-directional encoder
    input_utterence = Input(shape=(src_timesteps,))
    encoder_inputs = embedding(input_utterence)
    
    encoder1 = Bidirectional(LSTM(n_units, return_sequences=True, return_state=True))
    encoder_output, h1_fwd, c1_fwd, _, _ = encoder1(encoder_inputs)
    
    encoder2 = Bidirectional(LSTM(n_units, return_sequences=True, return_state=True))
    encoder_output, h2_fwd, c2_fwd, _, _ = encoder2(encoder_output)
    
    encoder3 = Bidirectional(LSTM(n_units, return_sequences=True, return_state=True))
    encoder_output, h3_fwd, c3_fwd, _, _ = encoder3(encoder_output)
    
    encoder4 = Bidirectional(LSTM(n_units, return_sequences=True, return_state=True))
    encoder_output, h4_fwd, c4_fwd, _, _ = encoder4(encoder_output)
    
    # hidden layer between encoder final state and decoder initial state has shown to be better
    encoder_dense_h1 = Dense(n_units, activation="tanh")
    initial_state_h1 = encoder_dense_h1(h1_fwd)
    
    encoder_dense_c1 = Dense(n_units, activation="tanh")
    initial_state_c1 = encoder_dense_c1(c1_fwd)
    
    encoder_dense_h2 = Dense(n_units, activation="tanh")
    initial_state_h2 = encoder_dense_h2(h2_fwd)
    
    encoder_dense_c2 = Dense(n_units, activation="tanh")
    initial_state_c2 = encoder_dense_c2(c2_fwd)
    
    encoder_dense_h3 = Dense(n_units, activation="tanh")
    initial_state_h3 = encoder_dense_h3(h3_fwd)
    
    encoder_dense_c3 = Dense(n_units, activation="tanh")
    initial_state_c3 = encoder_dense_c3(c3_fwd)
    
    encoder_dense_h4 = Dense(n_units, activation="tanh")
    initial_state_h4 = encoder_dense_h4(h4_fwd)
    
    encoder_dense_c4 = Dense(n_units, activation="tanh")
    initial_state_c4 = encoder_dense_c4(c4_fwd)
    
    
    # LSTM 4-layer decoder using teacher forcing
    target_utterence = Input(shape=(tar_timesteps,))
    decoder_inputs = embedding(target_utterence)
    
    decoder1 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder1(decoder_inputs, initial_state=[initial_state_h1, initial_state_c1])
    
    decoder2 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder2(decoder_outputs, initial_state=[initial_state_h2, initial_state_c2])
    
    decoder3 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder3(decoder_outputs, initial_state=[initial_state_h3, initial_state_c3])
    
    decoder4 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder4(decoder_outputs, initial_state=[initial_state_h4, initial_state_c4])
    
    # Bahdanau attention
    attn_layer = Attention()
    attn_out, attn_states = attn_layer([encoder_output, decoder_outputs])

    # Concat context vector from attention and decoder outputs for prediction
    decoder_concat_outputs = Concatenate(axis=-1)([decoder_outputs, attn_out])
    
    # apply softmax over the whole vocab for every decoder output hidden state
    dense1 = Dense(n_units * 3, activation="relu")
    outputs = dense1(decoder_concat_outputs)
    outputs = Dropout(dropout)(outputs)
    dense2 = Dense(vocab_size, activation="softmax")
    outputs = dense2(outputs)
    
    # treat as a multi-class classification problem where need to predict
    # P(yi | x1, x2, ..., xn, y1, y2, ..., yi-1)
    model = Model([input_utterence, target_utterence], outputs)
    model.compile(optimizer=Adam(clipnorm=CLIP_NORM), loss='categorical_crossentropy')
    model.summary()
    plot_model(model, to_file=pre.SEQ2SEQ_MODEL_IMAGE_FN, show_shapes=True)
    
    if train_by_batch:
        train_on_batches(model, encoder_input, decoder_input, decoder_target, vocab_size, BATCH_SIZE, EPOCHS)
    else:
        decoder_target = pre.encode_output(decoder_target, vocab_size)
        model.fit([encoder_input, decoder_input], decoder_target,
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=pre.VERBOSE)  
    

    # cannot use trained model directly for inference because of teacher forcing
    # don't have the ground truth for the target sequence during inference
    # so instead:
    # encoder model will encode input sequence and return hidden anc cell states
    # decoder model receives start of sequence token as first input to generate one word
    # make this generated word the input for the next time step
    # repeat until end of sequence token predicted or max seq length reached
    encoder_model = Model(inputs=input_utterence, 
                          outputs=[encoder_output,
                                   initial_state_h1, initial_state_c1,
                                   initial_state_h2, initial_state_c2,
                                   initial_state_h3, initial_state_c3,
                                   initial_state_h4, initial_state_c4])
    
    
    inf_decoder_utterence = Input(shape=(1,))
    encoder_input_states  = Input(shape=(src_timesteps, n_units * 2))
    
    inf_decoder_input_h1  = Input(shape=(n_units,))
    inf_decoder_input_c1  = Input(shape=(n_units,))
    
    inf_decoder_input_h2  = Input(shape=(n_units,))
    inf_decoder_input_c2  = Input(shape=(n_units,))
    
    inf_decoder_input_h3  = Input(shape=(n_units,))
    inf_decoder_input_c3  = Input(shape=(n_units,))
    
    inf_decoder_input_h4  = Input(shape=(n_units,))
    inf_decoder_input_c4  = Input(shape=(n_units,))

    inf_decoder_input     = embedding(inf_decoder_utterence)
    
    inf_decoder_out, inf_decoder_h1, inf_decoder_c1 = decoder1(
        inf_decoder_input, initial_state=[inf_decoder_input_h1, inf_decoder_input_c1])
    
    inf_decoder_out, inf_decoder_h2, inf_decoder_c2 = decoder2(
        inf_decoder_out, initial_state=[inf_decoder_input_h2, inf_decoder_input_c2])
    
    inf_decoder_out, inf_decoder_h3, inf_decoder_c3 = decoder3(
        inf_decoder_out, initial_state=[inf_decoder_input_h3, inf_decoder_input_c3])
    
    inf_decoder_out, inf_decoder_h4, inf_decoder_c4 = decoder4(
        inf_decoder_out, initial_state=[inf_decoder_input_h4, inf_decoder_input_c4])
    
    inf_attn_out, inf_attn_states = attn_layer([encoder_input_states, inf_decoder_out])
    inf_decoder_concat = Concatenate(axis=-1)([inf_decoder_out, inf_attn_out])
    
    inf_output = dense1(inf_decoder_concat)
    inf_output = dense2(inf_output)
    
    decoder_model = Model(
        inputs=[inf_decoder_utterence, encoder_input_states,
                inf_decoder_input_h1, inf_decoder_input_c1, 
                inf_decoder_input_h2, inf_decoder_input_c2, 
                inf_decoder_input_h3, inf_decoder_input_c3, 
                inf_decoder_input_h4, inf_decoder_input_c4],
        outputs=[inf_output, inf_attn_states, 
                 inf_decoder_h1, inf_decoder_c1, 
                 inf_decoder_h2, inf_decoder_c2, 
                 inf_decoder_h3, inf_decoder_c3, 
                 inf_decoder_h4, inf_decoder_c4])
    # ------ ------ #
    
    print("Finished Pre-training on movie conversations for %d epochs" % EPOCHS)
    
    # save the models
    model.save(pre.SEQ2SEQ_MODEL_FN)
    encoder_model.save(pre.SEQ2SEQ_ENCODER_MODEL_FN)
    decoder_model.save(pre.SEQ2SEQ_DECODER_MODEL_FN) 
    
    # do some dummy text generation
    for i in range(len(raw)):
        reply, _ = generate_reply_seq2seq(encoder_model, decoder_model, tokenizer, raw[i], in_seq_length, out_seq_length)
        print("Message:", raw[i])
        print("Reply:", reply + "\n")
    
    return model, encoder_model, decoder_model, tokenizer, in_seq_length, out_seq_length
    
def pre_train_seq2seq_dailydialogue(model, encoder_model, decoder_model, tokenizer, in_seq_length, out_seq_length, train_by_batch, BATCH_SIZE, EPOCHS):
    conversations = pre.load_dailydialogue_dataset()
    conversations = conversations[:5] ############################################
    vocab_size = len(tokenizer.word_index) + 1
    
    encoder_input  = conversations[:, 0]
    decoder_input  = np.array([pre.START_SEQ_TOKEN + ' ' + row[1] for row in conversations])
    decoder_target = np.array([row[1] + ' ' + pre.END_SEQ_TOKEN for row in conversations])
    
    raw = encoder_input[:20]
    
    # integer encode training data
    encoder_input  = pre.encode_sequences(tokenizer, in_seq_length, encoder_input)
    decoder_input  = pre.encode_sequences(tokenizer, out_seq_length, decoder_input)
    decoder_target = pre.encode_sequences(tokenizer, out_seq_length, decoder_target)
    
    if train_by_batch:
        train_on_batches(model, encoder_input, decoder_input, decoder_target, vocab_size, BATCH_SIZE, EPOCHS)
    else:
        decoder_target = pre.encode_output(decoder_target, vocab_size)
        model.fit([encoder_input, decoder_input], decoder_target,
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=pre.VERBOSE)  
        
    model.save(pre.SEQ2SEQ_MODEL_FN)
    encoder_model.save(pre.SEQ2SEQ_ENCODER_MODEL_FN)
    decoder_model.save(pre.SEQ2SEQ_DECODER_MODEL_FN) 
    
    print("Finished Pre-training on daily dialogue for %d epochs" % EPOCHS)
    
    # do some dummy text generation
    for i in range(len(raw)):
        reply, _ = generate_reply_seq2seq(encoder_model, decoder_model, tokenizer, raw[i], in_seq_length, out_seq_length)
        print("Message:", raw[i])
        print("Reply:", reply + "\n")
    
    return model, encoder_model, decoder_model
    
    

def train_seq2seq(LSTM_DIM=512, EPOCHS=100, BATCH_SIZE=128, CLIP_NORM=5, DROPOUT=0.2, train_by_batch=True):
    ''' Train sequence to sequence model with attention ans save the model to file '''
    
    # pretrain the model on movie conversations to get a sense of the english language
    model, encoder_model, decoder_model, tokenizer, in_seq_length, out_seq_length = pre_train_seq2seq_movie(
        LSTM_DIM, 1, BATCH_SIZE, CLIP_NORM, DROPOUT, train_by_batch)
    vocab_size = len(tokenizer.word_index) + 1
    
    # pretrain the model on daily dialogue to get a sense for natural conversations
    model, encoder_model, decoder_model = pre_train_seq2seq_dailydialogue(model, encoder_model, decoder_model, tokenizer, in_seq_length, out_seq_length, train_by_batch, BATCH_SIZE, 0)
    
    train_personas, train = pre.load_dataset(pre.TRAIN_FN)
    train = train[:1]
    
    # train is a numpy array containing triples [message, reply, persona_index]
    # personas is an numpy array of strings for the personas
    
    # need 3 arrays to fit the seq2seq model as will be using teacher forcing
    # encoder_input which is persona prepended to message, integer encoded and padded
    # for all messages so will have shape (utterances, in_seq_length)
    
    # decoder_input which is the reponse to the input, integer_encoded and padded
    # for all messages so will have shape (utterances, out_seq_length)
    
    # decoder_target which is the response to the input with an end of sequence token
    # appended, one_hot encoded and padded. shape (utterances, out_seq_length, vocab_size)
    encoder_input  = np.array([train_personas[int(row[2])] + ' ' + pre.SEP_SEQ_TOKEN + ' ' + row[0] for row in train])
    decoder_input  = np.array([pre.START_SEQ_TOKEN + ' ' + row[1] for row in train])
    decoder_target = np.array([row[1] + ' ' + pre.END_SEQ_TOKEN for row in train])
    
    raw = encoder_input[:20]
    
    # integer encode training data
    encoder_input  = pre.encode_sequences(tokenizer, in_seq_length, encoder_input)
    decoder_input  = pre.encode_sequences(tokenizer, out_seq_length, decoder_input)
    decoder_target = pre.encode_sequences(tokenizer, out_seq_length, decoder_target)
    
    
    if train_by_batch:
        train_on_batches(model, encoder_input, decoder_input, decoder_target, vocab_size, BATCH_SIZE, EPOCHS)
    else:
        decoder_target = pre.encode_output(decoder_target, vocab_size)
        model.fit([encoder_input, decoder_input], decoder_target,
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=pre.VERBOSE)     
    
    # save the models
    model.save(pre.SEQ2SEQ_MODEL_FN)
    encoder_model.save(pre.SEQ2SEQ_ENCODER_MODEL_FN)
    decoder_model.save(pre.SEQ2SEQ_DECODER_MODEL_FN) 
    
    print("Finished training on PERSONA-CHAT for %d epochs" % EPOCHS)
    
    # do some dummy text generation
    for i in range(len(raw)):
        reply, _ = generate_reply_seq2seq(encoder_model, decoder_model, tokenizer, raw[i], in_seq_length, out_seq_length)
        print("Message:", raw[i])
        print("Reply:", reply + "\n")

def generate_reply_seq2seq(encoder_model, decoder_model, tokenizer, input_msg, in_seq_length, out_seq_length):
    ''' Generates a reply for a trained sequence to sequence model using greedy search '''
    
    input_seq = pre.encode_sequences(tokenizer, in_seq_length, [input_msg])
    
    # get the hidden and cell state from the encoder
    encoder_outputs, h1, c1, h2, c2, h3, c3, h4, c4 = encoder_model.predict(input_seq)
    
    reply = []
    attn_weights = []
    prev_word = pre.START_SEQ_TOKEN
    while True:
        out_softmax_layer, attn, h1, c1, h2, c2, h3, c3, h4, c4 = decoder_model.predict(
            [pre.encode_sequences(tokenizer, 1, prev_word)[0], encoder_outputs,
             h1, c1, h2, c2, h3, c3, h4, c4])
        
        # get predicted word by looking at highest node in output softmax layer
        word_index = np.argmax(out_softmax_layer[0, -1, :])
        attn_weights.append((word_index, attn))
        prev_word = pre.index_to_word(word_index, tokenizer)
        
        if prev_word == pre.END_SEQ_TOKEN or len(reply) >= out_seq_length or prev_word == None:
            break
        
        reply.append(prev_word)
    
    return " ".join(reply), attn_weights

def beam_search_seq2seq(encoder_model, decoder_model, tokenizer, input_msg, in_seq_length, out_seq_length, beam_length = 3):
    ''' Generates a reply for a trained sequence to sequence model using beam search '''
    
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
    
    input_seq = pre.encode_sequences(tokenizer, in_seq_length, [input_msg])
    encoder_outputs, h1, c1, h2, c2, h3, c3, h4, c4 = encoder_model.predict(input_seq)
    
    # beams will store [ [probability, states, word1, word2, ...], ... ]
    beams = [ [0.0, h1, c1, h2, c2, h3, c3, h4, c4, pre.START_SEQ_TOKEN] for i in range(beam_length)]
    
    # store beam length ^ 2 most likely words [ [probability, word_index, beam_index], ... ]
    most_likely_words = [[0.0, 0, 0] for i in range(beam_length * beam_length)]
    
    prob = 0
    h1 = 1
    c1 = 2
    h2 = 3
    c2 = 4
    h3 = 5
    c3 = 6
    h4 = 7
    c4 = 8
    
    beam_finished = lambda b : b[-1] == pre.END_SEQ_TOKEN or len(b) - c4 - 1 >= out_seq_length
    while not reduce(lambda a, b : a and b , map(beam_finished, beams)):
    
        # find beam length most likely words out of all beams (vocab_size * beam length possibilities)
        for b_index in range(len(beams)):
            b = beams[b_index]
            prev_word = b[-1]
            
            if prev_word == pre.END_SEQ_TOKEN:
                # dead end beam so don't generate a new token, update states 
                # and leave most_likely_words for this beam constant
                continue
                
            out_softmax_layer, _, b[h1], b[c1], b[h2], b[c2], b[h3], b[c3], b[h4], b[c4] = decoder_model.predict(
                    [pre.encode_sequences(tokenizer, 1, prev_word)[0], encoder_outputs,
                     b[h1], b[c1], b[h2], b[c2], b[h3], b[c3], b[h4], b[c4]])
            
            # store beam length most likely words and there probabilities for this beam
            out_softmax_layer = out_softmax_layer[0, -1, :]
            most_likely_indicies = out_softmax_layer.argsort()[-beam_length:][::-1]
     
            i_ = 0
            for i in range(beam_length * b_index, beam_length * (b_index + 1) ):
                # summed log likelihood probability
                most_likely_words[i][0] = beams[b_index][prob] + log(
                    out_softmax_layer[most_likely_indicies[i_]]) 
                
                # word_index in tokenizer
                most_likely_words[i][1] = most_likely_indicies[i_]
                
                # beam index
                most_likely_words[i][2] = b_index
                i_ += 1
            
        # chose beam length most likely words out of beam length ^ 2 possible words
        # by their length normalized log likelihood probabilities descending
        # using a beam penalty of 1.0
        most_likely_words = sorted(
            most_likely_words, key = lambda l:l[0] / (len(beams[l[2]]) - c4 - 1), reverse=True)[:beam_length]
        
        # save network fragments for the most likely words
        temp_beams = [[] for i in range(beam_length)]
        for i in range(most_likely_words):
            old_beam = copy(beams[most_likely_words[i][2]])
            old_beam[prob] = most_likely_words[i][0]
            
            # prevent EOS token being continually appended for dead end beams
            if old_beam[-1] != pre.END_SEQ_TOKEN:
                old_beam.append(pre.index_to_word(most_likely_words[i][1], tokenizer))
            
            # beam state was already saved in the first step
            
            temp_beams[i] = old_beam
            
        beams = temp_beams
    
    # sort beams by length normalized log likelihood descending and only keep text
    replys = [" ".join(b[c4 + 2:-1]) 
              for b in sorted(beams, key = lambda b:b[0] / (len(b) - c4 - 1), reverse=True)]
    
    # return list of beam length reply's from most likely to least likely
    return replys


def train_on_batches(model, encoder_input, decoder_input, decoder_target, vocab_size, BATCH_SIZE, EPOCHS):
    ''' Trains the model batch by batch for the purposes of reducing memory usage 
        give integer encoded decoder target, will one hot encode this per-batch '''
        
    for epoch in range(EPOCHS):
        l = 0.0
        
        for i in range(0, encoder_input.shape[0] - BATCH_SIZE + 1, BATCH_SIZE):
            batch_encoder_input = encoder_input[i:i+BATCH_SIZE]
            batch_decoder_input = decoder_input[i:i+BATCH_SIZE]
            batch_decoder_target = pre.encode_output(
                decoder_target[i:i+BATCH_SIZE], vocab_size)
            
            model.train_on_batch(
                [batch_encoder_input, batch_decoder_input], batch_decoder_target)
            
            l = model.evaluate(
                [batch_encoder_input, batch_decoder_input], batch_decoder_target, verbose=pre.VERBOSE)
            
            if pre.VERBOSE != 0:
                print("BATCH %d / %d - loss: %f" % (i + BATCH_SIZE, encoder_input.shape[0], l))
        
        print("EPOCH %d loss: %f" % (epoch + 1, l))
