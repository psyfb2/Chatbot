# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""
import numpy as np
import text_preprocessing as pre
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RepeatVector, TimeDistributed, Dense, Embedding, LSTM
from tensorflow.keras.utils import plot_model

def autoencoder_model(vocab_size, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=src_timesteps, trainable=True, mask_zero=True))
    
    # encoder
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    
    # decoder
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.summary()
    plot_model(model, to_file=pre.AUTOENC_MODEL_IMAGE_FN, show_shapes=True)
    return model

def train_autoencoder(LSTM_DIMS=512, EPOCHS=10, BATCH_SIZE=64, CLIP_NORM=5, train_by_batch=True):
    encoder_input = np.array(['hello my friend'])
    decoder_target = np.array(['hello my friend'])
    
    tokenizer = pre.fit_tokenizer(["hello my friend"], oov_token=False)
    
    encoder_input  = pre.encode_sequences(tokenizer, 3, encoder_input)
    decoder_target = pre.encode_sequences(tokenizer, 3, decoder_target)
    
    model = autoencoder_model(4, 3, 3, LSTM_DIMS)
    
    decoder_target = pre.encode_output(decoder_target, 4)
    model.fit(encoder_input, decoder_target,
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1) 
    
    
     
