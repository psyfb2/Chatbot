# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import rnn


class Attention(Layer):
    def __init__(self, **kwargs):
        ''' 
        Implements additive bahdanau style attention
        https://arxiv.org/pdf/1508.04025.pdf
        '''
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        enc_shape = input_shape[0]
        dec_shape = input_shape[1]

        self.W1 = self.add_weight(name='W1',
                                   shape=tf.TensorShape((enc_shape[2], enc_shape[2])),
                                   initializer='uniform',
                                   trainable=True)
        
        self.W2 = self.add_weight(name='W2',
                                   shape=tf.TensorShape((dec_shape[2], enc_shape[2])),
                                   initializer='uniform',
                                   trainable=True)
        
        self.b = self.add_weight(name='b',
                                 shape=tf.TensorShape((1, 1, enc_shape[2])),
                                 initializer='uniform',
                                 trainable=True)
        
        self.V = self.add_weight(name='V',
                                   shape=tf.TensorShape((enc_shape[2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(Attention, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=0):
        '''
            inputs: [all_enc_outputs, all_dec_outputs]
        '''
        
        all_enc_outputs, all_dec_outputs = inputs
        
        if verbose != 0:
            print('encoder states =>', all_enc_outputs.shape)
            print('decoder states =>', all_dec_outputs.shape)

        def calc_attn_weights(dec_state, fake_state):
            src_timesteps, e_units = all_enc_outputs.shape[1], all_enc_outputs.shape[2]

            ''' Find W1.hs for all encoder states '''
            # <= (batch_size * src_timesteps, e_units)
            all_hs = tf.reshape(all_enc_outputs, (-1, e_units))
            
            # <= (batch_size, src_timesteps, e_units)
            W1_hs = tf.reshape(tf.matmul(all_hs, self.W1), (-1, src_timesteps, e_units))
            
            if verbose != 0:
                print('W1.hs:', W1_hs.shape)

            ''' Find W2.ht for a single decoder state '''
            # <= (batch_size, 1, e_units)
            W2_ht = tf.expand_dims(tf.matmul(dec_state, self.W2), 1)  
            
            if verbose != 0:
                print('W2.ht:', W2_ht.shape)

            ''' Find tanh(W1.hs + W2.ht + b) '''
            # <= (batch_size * src_timesteps, e_units)
            attn_weights = tf.nn.tanh(tf.reshape(W1_hs + W2_ht + self.b, (-1, e_units)))
            
            if verbose:
                print('W1.hs + W2.ht:', attn_weights.shape)

            ''' 
                Compress self.units down to one score 
                and find softmax over all encoder timestep scores 
            '''
            # <= (batch_size, src_timesteps)
            attn_weights = tf.reshape(tf.matmul(attn_weights, self.V), (-1, src_timesteps))
            # <= (batch_size, src_timesteps)
            attn_weights = tf.nn.softmax(attn_weights)

            if verbose != 0:
                print('attention weights: ', attn_weights.shape)

            return attn_weights, [attn_weights]

        def calc_context_vector(attn_weights, states):
            ''' Find context vector given attn_weights for a single decoder state '''
            # <= (batch_size, e_units)
            context_vector = tf.reduce_sum(all_enc_outputs * tf.expand_dims(attn_weights, -1), axis=1)
            
            if verbose != 0:
                print('context vector:', context_vector.shape)
                
            return context_vector, [context_vector]

        def init_fake_state(inputs, hidden_size):
            # <= (batch_size, src_timesteps, e_units)
            fake_state = tf.zeros_like(inputs)  
            
            # <= (batch_size)
            fake_state = tf.reduce_sum(fake_state, axis=[1, 2])  
            
            # <= (batch_size, 1)
            fake_state = tf.expand_dims(fake_state, axis=1)  
            
            # <= (batch_size, e_units)
            fake_state = tf.tile(fake_state, [1, hidden_size])  
            
            return fake_state

        c = init_fake_state(all_enc_outputs, all_enc_outputs.shape[-1])
        e = init_fake_state(all_enc_outputs, all_enc_outputs.shape[1])

        # attn_weights => (batch_size, tar_timesteps, src_timesteps)
        out, attn_weights, _ = rnn(
            calc_attn_weights, all_dec_outputs, [e],
        )

        # context_vectors => (batch_size, tar_timesteps, e_units)
        out, context_vectors, _ = rnn(
            calc_context_vector, attn_weights, [c],
        )

        return context_vectors, attn_weights

    def compute_output_shape(self, input_shape):
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]
    