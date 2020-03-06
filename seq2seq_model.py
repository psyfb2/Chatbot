# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""
import numpy as np
import text_preprocessing as pre
import tensorflow as tf
from attention import AttentionLayer
from keras.utils.vis_utils import plot_model
from tensorflow.keras.layers import LSTM, Dense, Embedding, TimeDistributed, Input, Bidirectional, Concatenate
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

def train_seq2seq(LSTM_DIM=512, EPOCHS=10, BATCH_SIZE=64, CLIP_NORM=5):
    train, train_personas = pre.load_object(pre.TRAIN_PKL_FN)
    # fit tokenizer over training data and start, stop tokens
    tokenizer = pre.fit_tokenizer( np.concatenate([train_personas, train[:, 0], train[:, 1], np.array([pre.START_SEQ_TOKEN, pre.END_SEQ_TOKEN])]) )
    vocab_size = len(tokenizer.word_index) + 1
    
    # train is a numpy array containing triples [message, reply, persona_index]
    # personas is an numpy array of strings for the personas
    
    # need 3 arrays to fit the seq2seq model as will be using teacher forcing
    # encoder_input which is persona prepended to message, integer encoded and padded
    # for all messages so will have shape (utterances, in_seq_length)
    
    # decoder_input which is the reponse to the input, integer_encoded and padded
    # for all messages so will have shape (utterances, out_seq_length)
    
    # decoder_target which is the response to the input with an end of sequence token
    # appended, one_hot encoded and padded. shape (utterances, out_seq_length, vocab_size)
    encoder_input  = np.array([train_personas[int(row[2])] + ' ' + row[0] for row in train])
    decoder_input  = np.array([pre.START_SEQ_TOKEN + ' ' + row[1] for row in train])
    decoder_target = np.array([row[1] + ' ' + pre.END_SEQ_TOKEN for row in train])
    
    raw = encoder_input
    
    in_seq_length = pre.max_seq_length(encoder_input)
    out_seq_length = pre.max_seq_length(decoder_input)
    
    print('Vocab size: %d' % vocab_size)
    print('Input sequence length: %d' % in_seq_length)
    print('Output sequence length: %d' % out_seq_length)

    # prepare training data
    encoder_input  = pre.encode_sequences(tokenizer, in_seq_length, encoder_input)
    decoder_input  = pre.encode_sequences(tokenizer, out_seq_length, decoder_input)
    decoder_target = pre.encode_sequences(tokenizer, out_seq_length, decoder_target)
    decoder_target = pre.encode_output(decoder_target, vocab_size)
    

    # load GloVe embeddings
    embedding_matrix = pre.load_glove_embedding(tokenizer)
    
    
    # ------ Model Definition ------ #
    src_timesteps = in_seq_length
    tar_timesteps = out_seq_length
    n_units = LSTM_DIM
    
    # encoder and decoder share the same embedding
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
    attn_layer = AttentionLayer()
    attn_out, attn_states = attn_layer([encoder_output, decoder_outputs])

    # Concat context vector from attention and decoder outputs for prediction
    decoder_concat_outputs = Concatenate(axis=-1)([decoder_outputs, attn_out])
    
    # apply softmax over the whole vocab for every decoder output hidden state
    dense1 = Dense(n_units * 3, activation="relu")
    outputs = dense1(decoder_concat_outputs)
    dense2 = Dense(vocab_size, activation="softmax")
    outputs = dense2(outputs)
    
    # treat as a multi-class classification problem where need to predict
    # P(yi | x1, x2, ..., xn, y1, y2, ..., yi-1)
    model = Model([input_utterence, target_utterence], outputs)
    model.compile(optimizer=Adam(clipnorm=CLIP_NORM), loss='categorical_crossentropy')
    model.summary()
    model.fit([encoder_input, decoder_input], decoder_target,
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)     
    plot_model(model, to_file=pre.SEQ2SEQ_MODEL_IMAGE_FN, show_shapes=True)

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
        outputs=[inf_output, inf_attn_out, 
                 inf_decoder_h1, inf_decoder_c1, 
                 inf_decoder_h2, inf_decoder_c2, 
                 inf_decoder_h3, inf_decoder_c3, 
                 inf_decoder_h4, inf_decoder_c4])
    # ------ ------ #
    
    # save the tokenizer and model so it can be used for prediction
    pre.save_object(tokenizer, pre.SEQ2SEQ_TOKENIZER_PKL_FN)
    pre.save_object(out_seq_length, pre.SEQ2SEQ_MAX_OUT_LEN_PKL_FN)
    pre.save_object(in_seq_length, pre.SEQ2SEQ_MAX_IN_LEN_PKL_FN)
    model.save(pre.SEQ2SEQ_MODEL_FN)
    encoder_model.save(pre.SEQ2SEQ_ENCODER_MODEL_FN)
    decoder_model.save(pre.SEQ2SEQ_DECODER_MODEL_FN) 
    
    # do some dummy text generation
    for i in range(20):
        input_seq = encoder_input[i:i+1]
        reply, _ = generate_reply_seq2seq(input_seq, out_seq_length, tokenizer, encoder_model, decoder_model)
        print("Message:", raw[i])
        print("Reply:", reply + "\n")
        
def generate_reply_seq2seq(input_seq, max_out_seq_length, tokenizer, encoder_model, decoder_model):
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
        attn_weights.append((word_index, attn_weights))
        prev_word = pre.index_to_word(word_index, tokenizer)
        
        if prev_word == pre.END_SEQ_TOKEN or len(reply) >= max_out_seq_length:
            break
        
        reply.append(prev_word)
    
    return " ".join(reply), attn_weights

''' Trains the model batch by batch for the purposes of reducing memory usage 
    give non one-hot encoded decoder target, will one hot encode this per-batch '''
def train_on_batch(model, encoder_input, decoder_input, decoder_target, vocab_size, BATCH_SIZE, EPOCHS, verbose=1):
    for epoch in range(EPOCHS):
        losses = []
        for i in range(0, encoder_input.shape[0] - BATCH_SIZE, BATCH_SIZE):
            batch_encoder_input = encoder_input[i:i+BATCH_SIZE]
            batch_decoder_input = decoder_input[i:i+BATCH_SIZE]
            batch_decoder_target = pre.encode_output(
                decoder_target[i:i+BATCH_SIZE], vocab_size)
            
            model.train_on_batch(
                [batch_encoder_input, batch_decoder_input], batch_decoder_target)
            
            l = model.evaluate(
                [batch_encoder_input, batch_decoder_input], batch_decoder_target)
            
            losses.append(l)
            
            if verbose == 1:
                print("BATCH %d / %d - loss: %f" % (i, int((encoder_input.shape[0] - BATCH_SIZE) / BATCH_SIZE), l))
            
        print("Mean loss in epoch %d : %f : " % (epoch + 1, np.mean(losses)))
            