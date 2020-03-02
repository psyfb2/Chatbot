# -*- coding: utf-8 -*-
"""
@author: Fady
"""
import numpy as np
import text_preprocessing as pre
from keras.models import Sequential
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.utils.vis_utils import plot_model

def autoencoder_model(vocab_size, src_timesteps, tar_timesteps, n_units, embedding_matrix):
    model = Sequential()
    e_dim = embedding_matrix.shape[1]
    model.add(Embedding(vocab_size, e_dim, weights=[embedding_matrix], input_length=src_timesteps, trainable=True, mask_zero=True))
    
    # encoder
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    
    # decoder
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.summary()
    plot_model(model, to_file=pre.MODEL_IMAGE_FN, show_shapes=True)
    return model


    
''' Given source sentence, generate the model inference as a target sentence '''
def generate_reply_autoencoder(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)
    prediction = prediction[0]
    integers = [np.argmax(vec) for vec in prediction]
    
    target = []
    for i in integers:
        word = pre.index_to_word(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)