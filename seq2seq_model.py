# -*- coding: utf-8 -*-
"""
@author: Fady
"""
import numpy as np
import text_preprocessing as pre
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import TimeDistributed
from keras.layers import Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

LSTM_DIM = 512
EPOCHS = 10
BATCH_SIZE = 64
CLIP_NORM = 5
DROPOUT = 0.2

def generate_reply_seq2seq(input_seq, max_out_seq_length, tokenizer, encoder_model, decoder_model):
    # get the hidden and cell state from the encoder
    hidden_state, cell_state = encoder_model.predict(input_seq)
    
    reply = []
    prev_word = pre.START_SEQ_TOKEN
    while True:
        out_softmax_layer, hidden_state, cell_state = decoder_model.predict(
            [pre.encode_sequences(tokenizer, 1, prev_word)[0], hidden_state, cell_state])
        
        # get predicted word by looking at highest node in output softmax layer
        word_index = np.argmax(out_softmax_layer[0, -1, :])
        prev_word = pre.index_to_word(word_index, tokenizer)
        
        if prev_word == pre.END_SEQ_TOKEN or len(reply) >= max_out_seq_length:
            break
        
        reply.append(prev_word)
    
    return " ".join(reply)


def train_seq2seq():
    # load training and validation set
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
    
    # encoder and decoder share the same embedding
    e_dim = embedding_matrix.shape[1]
    embedding = Embedding(vocab_size, e_dim, weights=[embedding_matrix], input_length=None , trainable=True, mask_zero=True)
    
    
    # ------ Model Definition ------ #
    src_timesteps = in_seq_length
    tar_timesteps = out_seq_length
    n_units = LSTM_DIM
    
    # LSTM 4-layer encoder
    input_utterence = Input(shape=(src_timesteps,))
    encoder_inputs = embedding(input_utterence)
    encoder = LSTM(n_units, return_state=True, dropout=DROPOUT)
    # get back the last hidden state and last cell state from the encoder LSTM
    encoder_output, hidden_state, cell_state = encoder(encoder_inputs)
    
    
    # LSTM 4-layer decoder using teacher forcing
    target_utterence = Input(shape=(tar_timesteps,))
    decoder_inputs = embedding(target_utterence)
    
    # decoder will output hidden state of all it's timesteps along with
    # last hidden state and last cell state which is used for inference model
    decoder = LSTM(n_units, return_sequences=True, return_state=True, dropout=DROPOUT)
    decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=[hidden_state, cell_state])
    # apply softmax over the whole vocab for every decoder output hidden state
    dense = Dense(vocab_size, activation="softmax")
    outputs = dense(decoder_outputs)
    
    # treat as a multi-class classification problem where need to predict
    # P(yi | x1, x2, ..., xn, y1, y2, ..., yi-1)
    model = Model([input_utterence, target_utterence], outputs)
    model.compile(optimizer=Adam(clipnorm=CLIP_NORM), loss='categorical_crossentropy')
    model.summary()
    model.fit([encoder_input, decoder_input], decoder_target,
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)     
    plot_model(model, to_file=pre.MODEL_IMAGE_FN, show_shapes=True)

    # cannot use trained model directly for inference because of teacher forcing
    # don't have the ground truth for the target sequence during inference
    # so instead:
    # encoder model will encode input sequence and return hidden anc cell states
    # decoder model receives start of sequence token as first input to generate one word
    # make this generated word the input for the next time step
    # repeat until end of sequence token predicted or max seq length reached
    
    batch_size = 1
    inf_encoder_utterence = Input(batch_shape=(batch_size, src_timesteps))
    inf_encoder_input = embedding(inf_encoder_utterence)
    inf_encoder_out, inf_encoder_hidden_state, inf_encoder_cell_state = encoder(inf_encoder_input)
    encoder_model = Model(inputs=inf_encoder_utterence, 
                          outputs=[inf_encoder_hidden_state, inf_encoder_cell_state])
    
    
    inf_decoder_utterence = Input(batch_shape=(batch_size, 1))
    inf_decoder_input_h   = Input(batch_shape=(batch_size, n_units))
    inf_decoder_input_c   = Input(batch_shape=(batch_size, n_units))
    inf_decoder_input     = embedding(inf_decoder_utterence)
    
    inf_decoder_out, inf_decoder_state_h, inf_decoder_state_c = decoder(
        inf_decoder_input, initial_state=[inf_decoder_input_h, inf_decoder_input_c])
    inf_output = dense(inf_decoder_out)
    decoder_model = Model(
        inputs=[inf_decoder_utterence, inf_decoder_input_h, inf_decoder_input_c],
        outputs=[inf_output, inf_decoder_state_h, inf_decoder_state_c])    
    # ------ ------ #
    
    # save the tokenizer and model so it can be used for prediction
    pre.save_object(tokenizer, pre.TOKENIZER_PKL_FN)
    pre.save_object(out_seq_length, pre.MAX_OUT_LEN_PKL_FN)
    pre.save_object(in_seq_length, pre.MAX_IN_LEN_PKL_FN)
    model.save(pre.MODEL_FN)
    encoder_model.save(pre.ENCODER_MODEL_FN)
    decoder_model.save(pre.DECODER_MODEL_FN) 
    
    # do some dummy text generation
    for i in range(20):
        input_seq = encoder_input[i:i+1]
        reply = generate_reply_seq2seq(input_seq, out_seq_length, tokenizer, encoder_model, decoder_model)
        print("Message:", raw[i])
        print("Reply:", reply + "\n")
