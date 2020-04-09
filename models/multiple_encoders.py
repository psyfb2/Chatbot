# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""
import numpy as np
import text_preprocessing as pre
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from time import time
from math import log
from copy import copy
from functools import reduce
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

LSTM_DIM = 512
CLIP_NORM = 5.0
DROPOUT = 0.2

class MultipleEncoder(tf.keras.Model):
    ''' 
        1 Layer Bidirectional LSTM for each of persona and message encoder
        there hidden states are added together and can be passed to the decoder
    '''
    def __init__(self, vocab_size, embedding, n_units, batch_size):
        super(MultipleEncoder, self).__init__()
        self.n_units = n_units
        self.batch_size = batch_size
        
        self.embedding = embedding
        
        self.persona_lstm1 = Bidirectional(
            LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_persona_lstm1"), name="enc_persona_lstm1_bi")
        
        self.msg_lstm1 = Bidirectional(
            LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_msg_lstm1"), name="enc_msg_lstm1_bi")
    @tf.function
    def call(self, inputs):
        '''
        inputs => [persona, msg]
        returns: encoder_persona_states, encoder_msg_states, h1, c1
        '''
        persona, msg = inputs[0], inputs[1]
        persona_embed = self.embedding(persona)
        msg_embed = self.embedding(msg)
        
        encoder_persona_states, persona_h1, persona_c1, _, _ = self.persona_lstm1(persona_embed)
        encoder_msg_states, msg_h1, msg_c1, _, _ = self.msg_lstm1(msg_embed)
        
        # add the hidden states of the persona and message encoder
        h1 = tf.math.add(persona_h1, msg_h1)
        c1 = tf.math.add(persona_c1, msg_c1)
        
        return encoder_persona_states, encoder_msg_states, h1, c1


class MultipleDecoder(tf.keras.Model):
    ''' 1 layer attentive LSTM which performs attention on two seperate encoders '''
    def __init__(self, vocab_size, embedding, n_units, batch_size):
        super(MultipleDecoder, self).__init__()
        self.batch_size = batch_size
        self.n_units = n_units
        
        self.embedding = embedding
        
        self.lstm1 = LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="dec_lstm1")
        
        # attention
        # Ct(s) = V tanh(W1 hs + W2 ht)
        # where hs is encoder state at timestep s and ht is the current
        # decoder timestep (which is at timestep t)
        self.persona_W1 = Dense(n_units)
        self.persona_W2 = Dense(n_units)
        self.persona_V  = Dense(1)
        
        self.msg_W1 = Dense(n_units)
        self.msg_W2 = Dense(n_units)
        self.msg_V = Dense(1)
        
        # from_logits=True in loss function, it will apply the softmax there for us
        self.out_dense1 = Dense(vocab_size)
    
    @tf.function
    def call(self, inputs):
        '''
        inputs => [input_word, encoder_persona_outputs, encoder_msg_outputs, context_vec_concat, [h1, c1]]
        returns => decoder_output, persona_attn_weights, msg_attn_weights, context_vec_concat, h1, c1
        '''
        input_word, encoder_persona_outputs, encoder_msg_outputs, context_vec_concat, hidden = inputs
        h1, c1 = hidden
        
        input_embed = self.embedding(input_word)
        
        # feed previous context vector as input into LSTM at current timestep
        input_embed = tf.concat([tf.expand_dims(context_vec_concat, 1), input_embed], axis=-1)
        
        decoder_output, h1, c1 = self.lstm1(input_embed, initial_state=[h1, c1])
        
        # => (batch_size, 1, n_units)
        decoder_state = tf.expand_dims(h1, 1)
        
        # ------ Attention on Persona Encoder  ------ #
        # score shape => (batch_size, src_timesteps, 1)
        score = self.persona_V(
            tf.nn.tanh(self.persona_W1(encoder_persona_outputs) + self.persona_W2(decoder_state)) )
        
        persona_attn_weights = tf.nn.softmax(score, axis=1)
        
        # context vector is a weighted sum of attention weights with encoder outputs
        persona_context_vec = persona_attn_weights * encoder_persona_outputs
        # => (batch_size, n_units * 2)
        persona_context_vec = tf.reduce_sum(persona_context_vec, axis=1)
        # ------ ------ #
        
        # ------ Attention on Message Encoder  ------ #
        # score shape => (batch_size, src_timesteps, 1)
        score = self.msg_V(
            tf.nn.tanh(self.msg_W1(encoder_msg_outputs) + self.msg_W2(decoder_state)) )
        
        msg_attn_weights = tf.nn.softmax(score, axis=1)
        
        # context vector is a weighted sum of attention weights with encoder outputs
        msg_context_vec = msg_attn_weights * encoder_msg_outputs
        # => (batch_size, n_units * 2)
        msg_context_vec = tf.reduce_sum(msg_context_vec, axis=1)
        # ------ ------ #
        
        # => (batch_size, n_units * 4)
        context_vec_concat = tf.concat([persona_context_vec, msg_context_vec], axis=-1)
        
        # (batch_size, 1, n_units) => (batch_size, n_units)
        decoder_output = tf.reshape(decoder_output, (-1, decoder_output.shape[2]))
        
        # => (batch_size, n_units * 5)
        decoder_output = tf.concat([decoder_output, context_vec_concat], axis=-1)
        
        decoder_output = Dropout(DROPOUT)(decoder_output )
        decoder_output = self.out_dense1(decoder_output)
        
        return decoder_output, persona_attn_weights, msg_attn_weights, context_vec_concat, h1, c1
        
class DeepMultipleEncoder(tf.keras.Model):
    ''' 
        4 Layer Bidirectional LSTM for each of persona and message encoder
        there hidden states are added together and can be passed to the decoder
    '''
    def __init__(self, vocab_size, embedding, n_units, batch_size):
        super(DeepMultipleEncoder, self).__init__()
        self.n_units = n_units
        self.batch_size = batch_size
        
        self.embedding = embedding
        
        self.persona_lstm1 = Bidirectional(
            LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_persona_lstm1"), name="enc_persona_lstm1_bi")
        self.persona_lstm2 = Bidirectional(
            LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_persona_lstm2"), name="enc_persona_lstm2_bi")
        self.persona_lstm3 = Bidirectional(
            LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_persona_lstm3"), name="enc_persona_lstm3_bi")
        self.persona_lstm4 = Bidirectional(
            LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_persona_lstm4"), name="enc_persona_lstm4_bi")
        
        self.msg_lstm1 = Bidirectional(
            LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_msg_lstm1"), name="enc_msg_lstm1_bi")
        self.msg_lstm2 = Bidirectional(
            LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_msg_lstm2"), name="enc_msg_lstm2_bi")
        self.msg_lstm3 = Bidirectional(
            LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_msg_lstm3"), name="enc_msg_lstm3_bi")
        self.msg_lstm4 = Bidirectional(
            LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_msg_lstm4"), name="enc_msg_lstm4_bi")
        
    @tf.function
    def call(self, inputs):
        '''
        inputs => [persona, msg]
        returns: encoder_persona_states, encoder_msg_states, h1, c1, h2, c2, h3, c3, h4, c4
        '''
        persona, msg = inputs[0], inputs[1]
        persona_embed = self.embedding(persona)
        msg_embed = self.embedding(msg)
        
        encoder_persona_states, persona_h1, persona_c1, _, _ = self.persona_lstm1(persona_embed)
        encoder_persona_states, persona_h2, persona_c2, _, _ = self.persona_lstm2(encoder_persona_states)
        encoder_persona_states, persona_h3, persona_c3, _, _ = self.persona_lstm3(encoder_persona_states)
        encoder_persona_states, persona_h4, persona_c4, _, _ = self.persona_lstm4(encoder_persona_states)
        
        encoder_msg_states, msg_h1, msg_c1, _, _ = self.msg_lstm1(msg_embed)
        encoder_msg_states, msg_h2, msg_c2, _, _ = self.msg_lstm2(encoder_msg_states)
        encoder_msg_states, msg_h3, msg_c3, _, _ = self.msg_lstm3(encoder_msg_states)
        encoder_msg_states, msg_h4, msg_c4, _, _ = self.msg_lstm4(encoder_msg_states)
        
        # add the hidden states of the persona and message encoder
        h1 = tf.math.add(persona_h1, msg_h1)
        c1 = tf.math.add(persona_c1, msg_c1)
        
        h2 = tf.math.add(persona_h2, msg_h2)
        c2 = tf.math.add(persona_c2, msg_c2)
        
        h3 = tf.math.add(persona_h3, msg_h3)
        c3 = tf.math.add(persona_c3, msg_c3)
        
        h4 = tf.math.add(persona_h4, msg_h4)
        c4 = tf.math.add(persona_c4, msg_c4)
        
        return encoder_persona_states, encoder_msg_states, h1, c1, h2, c2, h3, c3, h4, c4


class DeepMultipleDecoder(tf.keras.Model):
    ''' 4 layer attentive LSTM which performs attention on two seperate encoders '''
    def __init__(self, vocab_size, embedding, n_units, batch_size):
        super(DeepMultipleDecoder, self).__init__()
        self.batch_size = batch_size
        self.n_units = n_units
        
        self.embedding = embedding
        
        self.lstm1 = LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="dec_lstm1")
        self.lstm2 = LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="dec_lstm2")
        self.lstm3 = LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="dec_lstm3")
        self.lstm4 = LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="dec_lstm4")
        
        # attention
        # Ct(s) = V tanh(W1 hs + W2 ht)
        # where hs is encoder state at timestep s and ht is the current
        # decoder timestep (which is at timestep t)
        self.persona_W1 = Dense(n_units)
        self.persona_W2 = Dense(n_units)
        self.persona_V  = Dense(1)
        
        self.msg_W1 = Dense(n_units)
        self.msg_W2 = Dense(n_units)
        self.msg_V = Dense(1)
        
        # from_logits=True in loss function, it will apply the softmax there for us
        self.out_dense1 = Dense(vocab_size)
    
    @tf.function
    def call(self, inputs):
        '''
        inputs => [input_word, encoder_persona_outputs, encoder_msg_outputs, context_vec_concat, [h1, c1, h2, c2, h3, c3, h4, c4]]
        returns => decoder_output, persona_attn_weights, msg_attn_weights, context_vec_concat, h1, c1, h2, c2, h3, c3, h4, c4
        '''
        input_word, encoder_persona_outputs, encoder_msg_outputs, context_vec_concat, hidden = inputs
        h1, c1, h2, c2, h3, c3, h4, c4 = hidden
        
        input_embed = self.embedding(input_word)
        
        # feed previous context vector as input into LSTM at current timestep
        input_embed = tf.concat([tf.expand_dims(context_vec_concat, 1), input_embed], axis=-1)
        
        decoder_output, h1, c1 = self.lstm1(input_embed, initial_state=[h1, c1])
        decoder_output, h2, c2 = self.lstm2(decoder_output, initial_state=[h2, c2])
        decoder_output, h3, c3 = self.lstm3(decoder_output, initial_state=[h3, c3])
        decoder_output, h4, c4 = self.lstm4(decoder_output, initial_state=[h4, c4])
        
        # => (batch_size, 1, n_units)
        decoder_state = tf.expand_dims(h4, 1)
        
        # ------ Attention on Persona Encoder  ------ #
        # score shape => (batch_size, src_timesteps, 1)
        score = self.persona_V(
            tf.nn.tanh(self.persona_W1(encoder_persona_outputs) + self.persona_W2(decoder_state)) )
        
        persona_attn_weights = tf.nn.softmax(score, axis=1)
        
        # context vector is a weighted sum of attention weights with encoder outputs
        persona_context_vec = persona_attn_weights * encoder_persona_outputs
        # => (batch_size, n_units * 2)
        persona_context_vec = tf.reduce_sum(persona_context_vec, axis=1)
        # ------ ------ #
        
        # ------ Attention on Message Encoder  ------ #
        # score shape => (batch_size, src_timesteps, 1)
        score = self.msg_V(
            tf.nn.tanh(self.msg_W1(encoder_msg_outputs) + self.msg_W2(decoder_state)) )
        
        msg_attn_weights = tf.nn.softmax(score, axis=1)
        
        # context vector is a weighted sum of attention weights with encoder outputs
        msg_context_vec = msg_attn_weights * encoder_msg_outputs
        # => (batch_size, n_units * 2)
        msg_context_vec = tf.reduce_sum(msg_context_vec, axis=1)
        # ------ ------ #
        
        # => (batch_size, n_units * 4)
        context_vec_concat = tf.concat([persona_context_vec, msg_context_vec], axis=-1)
        
        # (batch_size, 1, n_units) => (batch_size, n_units)
        decoder_output = tf.reshape(decoder_output, (-1, decoder_output.shape[2]))
        
        # => (batch_size, n_units * 5)
        decoder_output = tf.concat([decoder_output, context_vec_concat], axis=-1)
        
        decoder_output = Dropout(DROPOUT)(decoder_output )
        decoder_output = self.out_dense1(decoder_output)
        
        return decoder_output, persona_attn_weights, msg_attn_weights, context_vec_concat, h1, c1, h2, c2, h3, c3, h4, c4
    
def loss_function(label, pred, loss_object):
    '''
    Calculate loss for a single prediction
    '''
    mask = tf.math.logical_not(tf.math.equal(label, 0))
    loss_ = loss_object(label, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)            

def calc_val_loss(batches_per_epoch, encoder, decoder, tokenizer, val_dataset, loss_object):
    total_loss = 0
    
    for (batch, (persona, msg, decoder_target)) in enumerate(val_dataset.take(batches_per_epoch)):
        loss = 0
        
        encoder_persona_states, encoder_msg_states, *initial_state = encoder([persona, msg])
        
        decoder_input = tf.expand_dims([tokenizer.word_index[pre.START_SEQ_TOKEN]] * encoder_persona_states.shape[0], 1)
        context_vec_concat = tf.zeros(
                (encoder_persona_states.shape[0], encoder_persona_states.shape[-1] + encoder_msg_states[-1]))
    
        for t in range(1, decoder_target.shape[1]):
            predictions, _, _, context_vec_concat, *initial_state = decoder([decoder_input, encoder_persona_states, encoder_msg_states, context_vec_concat, initial_state])
            
            loss += loss_function(decoder_target[:, t], predictions, loss_object)
            
            decoder_input = tf.expand_dims(decoder_target[:, t], 1)
        
        batch_loss = (loss / int(decoder_target.shape[1]))
        total_loss += batch_loss
    
    return total_loss
        

def train(encoder_persona_input, encoder_msg_input, decoder_target, encoder, decoder, tokenizer, loss_object, optimizer, save_best_model, deep_lstm, BATCH_SIZE, EPOCHS, PATIENCE):
    ''' Train seq2seq model, creates a validation set and uses early stopping '''
    
    @tf.function
    def train_step(persona, msg, decoder_target):
        '''
        Perform training on a single batch
        persona  => (batch_size, persona_length)
        msg => (batch_size, msg_length)
        decoder_target shape => (batch_size, out_seq_length)
        '''
        loss = 0
        
        with tf.GradientTape() as tape:
            encoder_persona_states, encoder_msg_states, *initial_state = encoder([persona, msg])
            
            decoder_input = tf.expand_dims([tokenizer.word_index[pre.START_SEQ_TOKEN]] * BATCH_SIZE, 1)
            context_vec_concat = tf.zeros(
                (encoder_persona_states.shape[0], encoder_persona_states.shape[-1] + encoder_msg_states[-1]))
            
            # Teacher forcing, ground truth for previous word input to the decoder at current timestep
            for t in range(1, decoder_target.shape[1]):
                predictions, _, _, context_vec_concat, *initial_state = decoder([decoder_input, encoder_persona_states, encoder_msg_states, context_vec_concat, initial_state])
                
                loss += loss_function(decoder_target[:, t], predictions, loss_object)
                
                decoder_input = tf.expand_dims(decoder_target[:, t], 1)
            
        # backpropegate loss
        batch_loss = (loss / int(decoder_target.shape[1]))
            
        variables = encoder.trainable_variables + decoder.trainable_variables
    
        gradients = tape.gradient(loss, variables)
            
        optimizer.apply_gradients(zip(gradients, variables))
            
        return batch_loss
    
    breakpoint() ##########
    
    encoder_persona_input, encoder_persona_input_val, encoder_msg_input, encoder_msg_input_val, decoder_target, decoder_target_val = train_test_split(
        encoder_persona_input, encoder_msg_input, decoder_target, test_size=0.05)
    
    dataset = tf.data.Dataset.from_tensor_slices((encoder_persona_input, encoder_msg_input,  decoder_target))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((encoder_persona_input_val, encoder_msg_input_val,  decoder_target_val))
    val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    batches_per_epoch = len(encoder_persona_input) // BATCH_SIZE
    batches_per_epoch_val = len(encoder_persona_input_val) // BATCH_SIZE
        
    min_val_loss = float("inf")
    no_improvement_counter = 0
    
    for epoch in range(EPOCHS):
        start = time()
        
        total_loss = 0
        
        for (batch, (persona, msg, decoder_target)) in enumerate(dataset.take(batches_per_epoch)):
            batch_loss = train_step(persona, msg, decoder_target)
            total_loss += batch_loss
            
            if pre.VERBOSE == 1:
                print("Epoch %d: Batch %d: Loss %f" % (epoch + 1, batch + 1, batch_loss.numpy()))
        
        val_loss = calc_val_loss(batches_per_epoch_val, encoder, decoder, tokenizer, val_dataset, loss_object)
        
        if val_loss < min_val_loss:
            if save_best_model:
                print("Saving model as best val loss decreased from %f to %f" % (min_val_loss, val_loss))
                save_seq2seq(encoder, decoder, deep_lstm)
            no_improvement_counter = 0
            min_val_loss = val_loss
        else:
            no_improvement_counter += 1
        
        print("Epoch %d --- %d sec: Loss %f, val_loss: %f" % (epoch + 1, time() - start, total_loss / batches_per_epoch, val_loss / batches_per_epoch_val))
        
        if no_improvement_counter >= PATIENCE:
            print("Early stopping, no improvement over minimum in %d epochs" % PATIENCE)
            return epoch + 1
    
    save_seq2seq(encoder, decoder, deep_lstm)
    return EPOCHS

def seed_weights(seq2seq_encoder, seq2seq_decoder, multple_encoder, multiple_decoder, deep_lstm):
    ''' Seed weights of multiple encoder model with the weights of a previously trained sequence to sequence model '''
    pass
            

def train_multiple_encoders(EPOCHS, BATCH_SIZE, PATIENCE, deep_lstm=False):
    vocab, persona_length, msg_length, reply_length = pre.get_vocab()
    tokenizer = pre.fit_tokenizer(vocab)
    vocab_size = len(tokenizer.word_index) + 1
    vocab = None

    out_seq_length = reply_length
    
    print('Vocab size: %d' % vocab_size)
    print('Persona sequence length: %d' % persona_length)
    print('Message sequence length: %d' % msg_length)
    print('Output sequence length: %d' % out_seq_length)
    
    embedding_matrix = pre.load_glove_embedding(tokenizer, pre.GLOVE_FN)
    embedding_matrix = Embedding(vocab_size, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=True, mask_zero=True, name="tied_embedding")
    
    if deep_lstm:
        seq2seq_encoder = tf.saved_model.load(pre.SEQ2SEQ_ENCODER_DEEP_MODEL_FN)
        seq2seq_decoder = tf.saved_model.load(pre.SEQ2SEQ_DECODER_DEEP_MODEL_FN)
        
        encoder = DeepMultipleEncoder(vocab_size, embedding_matrix, LSTM_DIM, BATCH_SIZE)
        decoder = DeepMultipleDecoder(vocab_size, embedding_matrix, LSTM_DIM, BATCH_SIZE)
    else:
        seq2seq_encoder = tf.saved_model.load(pre.SEQ2SEQ_ENCODER_MODEL_FN)
        seq2seq_decoder = tf.saved_model.load(pre.SEQ2SEQ_DECODER_MODEL_FN)
        
        encoder = MultipleEncoder(vocab_size, embedding_matrix, LSTM_DIM, BATCH_SIZE)
        decoder = MultipleDecoder(vocab_size, embedding_matrix, LSTM_DIM, BATCH_SIZE)
    
    # seed the weights of this model with previously trained seq2seq model
    seed_weights(seq2seq_encoder, seq2seq_decoder, encoder, decoder, deep_lstm)
    
    # ------ Train on PERSONA-CHAT ------ #
    train_personas, train_data = pre.load_dataset(pre.TRAIN_FN)
    
    # train is a numpy array containing triples [message, reply, persona_index]
    # personas is an numpy array of strings for the personas
    
    encoder_persona_input  = np.array([train_personas[int(row[2])] for row in train_data])
    encoder_msg_input = train_data[:, 0]
    decoder_target = np.array([pre.START_SEQ_TOKEN + ' ' + row[1] + ' ' + pre.END_SEQ_TOKEN for row in train_data])
    
    train_data, train_personas = None, None
    
    persona_raw = encoder_persona_input[:20]
    msg_raw = encoder_msg_input[:20]
    
    # integer encode training data
    encoder_persona_input  = pre.encode_sequences(tokenizer, persona_length, encoder_persona_input)
    encoder_msg_input = pre.encode_sequences(tokenizer, msg_length, encoder_msg_input)
    decoder_target = pre.encode_sequences(tokenizer, out_seq_length, decoder_target)
    
    optimizer = Adam(clipnorm=CLIP_NORM)
    loss_func = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    epochs = train(encoder_persona_input, encoder_msg_input, decoder_target, encoder, decoder, tokenizer, loss_func, optimizer, True, deep_lstm, BATCH_SIZE, EPOCHS, PATIENCE)
    
    print("Finished Training on PERSONA-CHAT for %d epochs" % epochs)
    
    # do some dummy text generation
    for i in range(len(persona_raw)):
        reply, persona_attn_weights, msg_attn_weights = generate_reply_multiple_encoder(encoder, decoder, tokenizer, persona_raw[i], msg_raw[i], persona_length, msg_length, out_seq_length)
        print("Persona:", persona_raw[i])
        print("Message:", msg_raw[i])
        print("Reply:", reply + "\n")
    # ------ ------ #


def save_seq2seq(encoder, decoder, deep_lstm):
    ''' Save the encoder and decoder as tensorflow models to file '''
    if deep_lstm:
        encoder_fn = pre.SEQ2SEQ_ENCODER_DEEP_MODEL_FN
        decoder_fn = pre.SEQ2SEQ_DECODER_DEEP_MODEL_FN
        decoder_states_spec = []
        for i in range(1, 5):
            decoder_states_spec.append(tf.TensorSpec(shape=[None, LSTM_DIM], dtype=tf.float32, name='h%d' % i))
            decoder_states_spec.append(tf.TensorSpec(shape=[None, LSTM_DIM], dtype=tf.float32, name='c%d' % i))
    else:
        encoder_fn = pre.SEQ2SEQ_ENCODER_MODEL_FN
        decoder_fn = pre.SEQ2SEQ_DECODER_MODEL_FN
        decoder_states_spec = [
            tf.TensorSpec(shape=[None, LSTM_DIM], dtype=tf.float32, name='h1'), 
            tf.TensorSpec(shape=[None, LSTM_DIM], dtype=tf.float32, name='c1')]
        
    tf.saved_model.save(encoder, encoder_fn , signatures=encoder.call.get_concrete_function(
        [
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='persona'),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='message')
        ]))
    
    tf.saved_model.save(decoder, decoder_fn, signatures=decoder.call.get_concrete_function(
        [
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='input_word'), 
            tf.TensorSpec(shape=[None, None, LSTM_DIM * 2], dtype=tf.float32, name="encoder_persona_states"),
            tf.TensorSpec(shape=[None, None, LSTM_DIM * 2], dtype=tf.float32, name="encoder_msg_states"),
            tf.TensorSpec(shape=[None, LSTM_DIM * 4], dtype=tf.float32, name="context_vec_concat"),
            decoder_states_spec
        ]))
    
def plot_attention(attn_weights, message, reply):
    ''' Visualize attention weights '''
    fig = plt.figure(figsize=(15, 15))
    axis = fig.add_subplot(1, 1, 1)
    
    axis.matshow(attn_weights, cmap='viridis')
    
    font_size = {'fontsize' : 12}
    
    axis.set_xticklabels([''] + message.split(' '), fontdict=font_size, rotation=90)
    axis.set_yticklabels([''] + reply.split(' '), fontdict=font_size)
    
    axis.xaxis.set_major_locator(ticker.MultipleLocator(1))
    axis.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def generate_reply_multiple_encoder(encoder, decoder, tokenizer, persona, msg, persona_length, msg_length, out_seq_length):
    '''
    Generates a reply for a trained multiple encoder sequence to sequence model using greedy search 
    '''
    persona = pre.encode_sequences(tokenizer, persona_length, [persona])
    persona = tf.convert_to_tensor(persona)
    
    msg = pre.encode_sequences(tokenizer, persona_length, [msg])
    msg = tf.convert_to_tensor(msg)
    
    persona_attn_weights = np.zeros((out_seq_length, persona_length))
    msg_attn_weights = np.zeros((out_seq_length, msg_length))
    
    encoder_persona_states, encoder_msg_states, *initial_state = encoder([persona, msg])
    
    decoder_input = tf.expand_dims([tokenizer.word_index[pre.START_SEQ_TOKEN]], 0)
    context_vec_concat = tf.zeros(
                (encoder_persona_states.shape[0], encoder_persona_states.shape[-1] + encoder_msg_states[-1]))
    
    reply = []
    for t in range(out_seq_length):
        softmax_layer, persona_attn_score, msg_attn_score, context_vec_concat, *initial_state = decoder([decoder_input, encoder_persona_states, encoder_msg_states, context_vec_concat, initial_state])
        
        persona_attn_score = tf.reshape(persona_attn_score, (-1,))
        persona_attn_weights[t] = persona_attn_score.numpy()
        
        msg_attn_score = tf.reshape(msg_attn_score, (-1,))
        msg_attn_weights[t] = msg_attn_score.numpy()
        
        # get predicted word by looking at highest node in output softmax layer
        word_index = tf.argmax(softmax_layer[0]).numpy()
        word = pre.index_to_word(word_index, tokenizer)
    
        if word == pre.END_SEQ_TOKEN:
            break
        
        reply.append(word)
        
        decoder_input = tf.expand_dims([word_index], 0)
    
    return " ".join(reply), persona_attn_weights, msg_attn_weights