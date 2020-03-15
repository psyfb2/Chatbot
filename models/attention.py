# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""
import tensorflow as tf
import os
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, LSTM

class AttentionLayer(Layer):
    '''
    Implements additive Bahdanau style attention
    https://arxiv.org/pdf/1508.04025.pdf
    '''
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        # at(s) = softmax(V dotproduct W1.ht + W2.hs + b) 
        # where ht is the current decoder hidden state
        # and hs is encoder hidden state at timestep s
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V  = Dense(1)
    
    def call(self, decoder_state, encoder_states):
        '''
            decoder_state shape is (batch_size, lstm_units)
            encoder_state shape is (batch_size, src_seq_length, lstm_units * 2)
            * 2 only if bidirectional encoder
            
            change decoder_state shape to (batch_sie, 1, lstm_units)
            to add this state after dense layer to all encoder states 
            after their dense layer
        '''
        decoder_states = tf.expand_dims(decoder_state, 1)
        
        score = self.V(tf.nn.tanh( self.W1(decoder_states) + self.W2(encoder_states) ))
        attn_weights = tf.nn.softmax(score, axis=1)
        
        # attn_weights(s) indicates how important encoder state s is to
        # decoder output t. Find weighted sum of encoder states
        c = attn_weights * encoder_states
        c = tf.reduce_sum(c, axis=1)
        
        return c, attn_weights
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config
    