# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""
import tensorflow as tf
import os
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, LSTM


class AttentionLayer(Layer):
    def __init__(self, units):
        # at(s) = softmax(V dotproduct W1.ht + W2.hs + b) 
        # where ht is the current decoder hidden state
        # and hs is encoder hidden state at timestep s
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V  = Dense(1)
    
    def call(self, inputs):
        '''
            inputs = [all encoder outputs, all decoder outputs]
        '''
        enc_outputs, dec_outputs = inputs
        
        def attn_weights(dec_state, fake_state):
            '''
                calculate scores for all encoder states given a single decoder state
                decoder_state shape => (batch_size, tar_timesteps)
                return shape        => (batch_size, src_timesteps)
            '''
            # convert to shape (batch_size, 1, tar_timesteps) for addition
            dec_state_time_axis = tf.expand_dims(dec_state, 1)
            
            # scores has shape (batch_size, src_timesteps)
            scores = self.V(tf.nn.tanh( self.W1(dec_state_time_axis) + self.W2(enc_outputs) ))
            
            attn_weights = tf.nn.softmax(scores, axis=1)
            
            return attn_weights, [attn_weights]
        
        def context_vector(attn_weights, fake_state):
            pass
            #context_vec = attn_weights * 
            
            
            
        


"""
class Attention(Layer):
    '''
    Implements additive Bahdanau style attention
    https://arxiv.org/pdf/1508.04025.pdf
    '''
    def __init__(self, units):
        super(Attention, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        enc_shape = input_shape[0]
        dec_shape = input_shape[1]
        
        # at(s) = softmax(V dotproduct W1.ht + W2.hs + b) 
        # where ht is the current decoder hidden state
        # and hs is encoder hidden state at timestep s
        
        self.W1 = self.add_weight(name="W1", 
                                  shape=tf.TensorShape(enc_shape[2], self.units),
                                  initializer="uniform",
                                  trainable=True)
        
        self.W2 = self.add_weight(name="W2",
                                  shape=tf.TensorShape(dec_shape[2], self.units),
                                  initializer='uniform',
                                  trainable=True)
        
        self.b = self.add_weight(name="b",
                                 shape=tf.TensorShape(self.units),
                                 initializer='uniform',
                                 trainable=True)
        
        self.V = self.add_weight(name="V",
                                 shape=tf.TensorShape(self.units, 1),
                                 initializer='uniform',
                                 trainable=True)
        
        super(Attention, self).build(input_shape)
        
    def call(self, inputs):
        '''
            inputs = [all encoder outputs, all decoder outputs]
        '''
        encoder_outputs, decoder_outputs = inputs
        
        def attn_weights(ht, ignore_state):
            '''
                calculate attention weights with shape (batch_size, src_timesteps)
                from a single decoder hidden state ht
            '''
            dec_hidden_units = ht.shape[-1]
            src_timesteps, enc_hidden_units = encoder_outputs.shape[1], encoder_outputs.shape[-1]
            
            # calculate W1.hs for all hs in encoder_outputs
            # go from (batch_size, src_timesteps, units) 
            #      -> (batch_size * src_timesteps, units)
            encoder_outputs_reshaped = tf.reshape(encoder_outputs, [-1, enc_hidden_units])
            w1_hs = 
            
"""
            
            
    