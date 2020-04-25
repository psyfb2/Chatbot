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

class Encoder(tf.keras.Model):
    ''' 1 Layer Bidirectional LSTM '''
    def __init__(self, vocab_size, embedding, n_units, batch_size, use_segment_embedding, segment_embedding_dim):
        super(Encoder, self).__init__()
        self.n_units = n_units
        self.batch_size = batch_size
        
        self.embedding = embedding
        
        # segment embedding are used so that this model can better distinguish between persona and message segments
        # pad segment vectors with 0's exactly like word vectors
        if use_segment_embedding:
            # segment_embedding_dim must be the same as output_dim of word embedding
            self.segment_embedding = Embedding(3, segment_embedding_dim, trainable=True, mask_zero=True, name="segment_embedding")
        else:
            # use a zero segment embedding which will have no effect on the model
            self.segment_embedding = Embedding(3, segment_embedding_dim, weights=[np.zeros((3, segment_embedding_dim))], trainable=False, mask_zero=True, name="segment_embedding")
        
        self.lstm1 = Bidirectional(
            LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_lstm1"), name="enc_lstm1_bi")
        
    @tf.function
    def call(self, inputs):
        '''
        inputs => [input_utterence, segment_tokens]
        returns => encoder_states, h1, c1
        '''
        input_utterence, segment_tokens = inputs
        input_embed = self.embedding(input_utterence)
        segment_embed = self.segment_embedding(segment_tokens)
        
        combined_embed = tf.add(input_embed, segment_embed)
        
        encoder_states, h1, c1, _, _ = self.lstm1(combined_embed)
        
        return encoder_states, h1, c1


class Decoder(tf.keras.Model):
    ''' 1 layer attentive LSTM '''
    def __init__(self, vocab_size, embedding, n_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.n_units = n_units
        
        self.embedding = embedding
        
        self.lstm1 = LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="dec_lstm1")
        
        # attention
        # Ct(s) = V tanh(W1 hs + W2 ht)
        # where hs is encoder state at timestep s and ht is the current
        # decoder timestep (which is at timestep t)
        self.W1 = Dense(n_units)
        self.W2 = Dense(n_units)
        self.V  = Dense(1)
        
        # from_logits=True in loss function, it will apply the softmax there for us
        self.out_dense1 = Dense(vocab_size)
    
    @tf.function
    def call(self, inputs):
        '''
        inputs => [input_word, encoder_outputs, context_vec, [h1, c1]
        returns => decoder_output, attn_weights, context_vec, h1, c1
        '''
        input_word, encoder_outputs, context_vec, hidden = inputs
        h1, c1 = hidden
        
        input_embed = self.embedding(input_word)
        
        # feed previous context vector as input into LSTM at current timestep
        input_embed = tf.concat([tf.expand_dims(context_vec, 1), input_embed], axis=-1)
        
        decoder_output, h1, c1 = self.lstm1(input_embed, initial_state=[h1, c1])
        
        # ------ Attention ------ #
        # => (batch_size, 1, n_units)
        decoder_state = tf.expand_dims(h1, 1)
        
        # score shape => (batch_size, src_timesteps, 1)
        score = self.V(
            tf.nn.tanh(self.W1(encoder_outputs) + self.W2(decoder_state)) )
        
        attn_weights = tf.nn.softmax(score, axis=1)
        
        # context vector is a weighted sum of attention weights with encoder outputs
        context_vec = attn_weights * encoder_outputs
        # => (batch_size, n_units * 2)
        context_vec = tf.reduce_sum(context_vec, axis=1)
        # ------ ------ #
        
        # (batch_size, 1, n_units) => (batch_size, n_units)
        decoder_output = tf.reshape(decoder_output, (-1, decoder_output.shape[2]))
        
        # => (batch_size, n_units * 3)
        decoder_output = tf.concat([decoder_output, context_vec], axis=-1)
        
        decoder_output = Dropout(DROPOUT)(decoder_output )
        decoder_output = self.out_dense1(decoder_output)
        
        return decoder_output, attn_weights, context_vec, h1, c1


class DeepEncoder(tf.keras.Model):
    ''' 4 Layer Bidirectional LSTM '''
    def __init__(self, vocab_size, embedding, n_units, batch_size, use_segment_embedding, segment_embedding_dim):
        super(DeepEncoder, self).__init__()
        self.n_units = n_units
        self.batch_size = batch_size
        
        self.embedding = embedding
        
        if use_segment_embedding:
            self.segment_embedding = Embedding(3, segment_embedding_dim, trainable=True, mask_zero=True, name="segment_embedding")
        else:
            self.segment_embedding = Embedding(3, segment_embedding_dim, weights=[np.zeros((3, segment_embedding_dim))], trainable=False, mask_zero=True, name="segment_embedding")
        
        self.lstm1 = Bidirectional(
            LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_lstm1"), name="enc_lstm1_bi")
        
        self.lstm2 = Bidirectional(
            LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_lstm2"), name="enc_lstm2_bi")
        
        self.lstm3 = Bidirectional(
            LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_lstm3"), name="enc_lstm3_bi")
        
        self.lstm4 = Bidirectional(
            LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_lstm4"), name="enc_lstm4_bi")
        
    @tf.function
    def call(self, inputs):
        '''
        inputs => [input_utterence, segment_tokens]
        returns: encoder_states, h1, c1, h2, c2, h3, c3, h4, c4
        '''
        input_utterence, segment_tokens = inputs
        
        input_embed = self.embedding(input_utterence)
        segment_embed = self.segment_embedding(segment_tokens)
        
        combined_embed = tf.add(input_embed, segment_embed)
        
        encoder_states, h1, c1, _, _ = self.lstm1(combined_embed)
        encoder_states, h2, c2, _, _ = self.lstm2(encoder_states)
        encoder_states, h3, c3, _, _ = self.lstm3(encoder_states)
        encoder_states, h4, c4, _, _ = self.lstm4(encoder_states)
        
        return encoder_states, h1, c1, h2, c2, h3, c3, h4, c4


class DeepDecoder(tf.keras.Model):
    ''' 4 layer attentive LSTM '''
    def __init__(self, vocab_size, embedding, n_units, batch_size):
        super(DeepDecoder, self).__init__()
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
        self.W1 = Dense(n_units)
        self.W2 = Dense(n_units)
        self.V  = Dense(1)
        
        # from_logits=True in loss function, it will apply the softmax there for us
        self.out_dense1 = Dense(vocab_size)
    
    @tf.function
    def call(self, inputs):
        '''
        inputs => [input_word, encoder_outputs, context_vec, [h1, c1, h2, c2, h3, c3, h4, c4]]
        returns => decoder_output, attn_weights, context_vec, h1, c1, h2, c2, h3, c3, h4, c4
        '''
        input_word, encoder_outputs, context_vec, hidden = inputs
        h1, c1, h2, c2, h3, c3, h4, c4 = hidden
        
        input_embed = self.embedding(input_word)
        input_embed = tf.concat([tf.expand_dims(context_vec, 1), input_embed], axis=-1)
        
        decoder_output, h1, c1 = self.lstm1(input_embed, initial_state=[h1, c1])
        decoder_output, h2, c2 = self.lstm2(decoder_output, initial_state=[h2, c2])
        decoder_output, h3, c3 = self.lstm3(decoder_output, initial_state=[h3, c3])
        decoder_output, h4, c4 = self.lstm4(decoder_output, initial_state=[h4, c4])
        
        # ------ Attention ------ #
        # => (batch_size, 1, n_units)
        decoder_state = tf.expand_dims(h4, 1)
        
        # score shape => (batch_size, src_timesteps, 1)
        score = self.V(
            tf.nn.tanh(self.W1(encoder_outputs) + self.W2(decoder_state)) )
        
        attn_weights = tf.nn.softmax(score, axis=1)
        
        # context vector is a weighted sum of attention weights with encoder outputs
        context_vec = attn_weights * encoder_outputs
        # => (batch_size, n_units * 2)
        context_vec = tf.reduce_sum(context_vec, axis=1)
        # ------ ------ #
        
        # (batch_size, 1, n_units) => (batch_size, n_units)
        decoder_output = tf.reshape(decoder_output, (-1, decoder_output.shape[2]))
        
        # => (batch_size, n_units * 3)
        decoder_output = tf.concat([decoder_output, context_vec], axis=-1)
        
        decoder_output = Dropout(DROPOUT)(decoder_output )
        decoder_output = self.out_dense1(decoder_output)
        
        return decoder_output, attn_weights, context_vec, h1, c1, h2, c2, h3, c3, h4, c4
    
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
    
    for (batch, (encoder_input, segment_input, decoder_target)) in enumerate(val_dataset.take(batches_per_epoch)):
        loss = 0
        
        encoder_outputs, *initial_state = encoder([encoder_input, segment_input])
        
        decoder_input = tf.expand_dims([tokenizer.word_index[pre.START_SEQ_TOKEN]] * encoder_input.shape[0], 1)
        context_vec = tf.zeros((encoder_outputs.shape[0], encoder_outputs.shape[-1]))
    
        for t in range(1, decoder_target.shape[1]):
            predictions, _, context_vec, *initial_state = decoder([decoder_input, encoder_outputs, context_vec, initial_state])
            
            loss += loss_function(decoder_target[:, t], predictions, loss_object)
            
            decoder_input = tf.expand_dims(decoder_target[:, t], 1)
        
        batch_loss = (loss / int(decoder_target.shape[1]))
        total_loss += batch_loss
    
    return total_loss

@tf.function
def train_step(encoder_input, segment_input, decoder_target, encoder, decoder, loss_object, tokenizer, optimizer, BATCH_SIZE):
    '''
    Perform training on a single batch
    encoder_input shape  => (batch_size, in_seq_length)
    segment_input shape  => (batch_size, in_seq_length)
    decoder_target shape => (batch_size, out_seq_length)
    '''
    loss = 0
    
    with tf.GradientTape() as tape:
        encoder_outputs, *initial_state = encoder([encoder_input, segment_input])
        
        decoder_input = tf.expand_dims([tokenizer.word_index[pre.START_SEQ_TOKEN]] * BATCH_SIZE, 1)
        context_vec = tf.zeros((encoder_outputs.shape[0], encoder_outputs.shape[-1]))
        
        # Teacher forcing, ground truth for previous word input to the decoder at current timestep
        for t in range(1, decoder_target.shape[1]):
            predictions, _, context_vec, *initial_state = decoder([decoder_input, encoder_outputs, context_vec, initial_state])
            
            loss += loss_function(decoder_target[:, t], predictions, loss_object)
            
            decoder_input = tf.expand_dims(decoder_target[:, t], 1)
        
    # backpropegate loss
    batch_loss = (loss / int(decoder_target.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)
    
    optimizer.apply_gradients(zip(gradients, variables))
    
    return batch_loss
        

def train(dataset, val_dataset, batches_per_epoch, batches_per_epoch_val, encoder, decoder, tokenizer, loss_object, optimizer, save_best_model, deep_lstm, BATCH_SIZE, EPOCHS, MIN_EPOCHS, PATIENCE):
    ''' Train seq2seq model, uses early stopping on the validation set '''
    min_val_loss = float("inf")
    no_improvement_counter = 0

    for epoch in range(EPOCHS):
        start = time()
        
        total_loss = 0
        
        for (batch, (encoder_input, segment_input, decoder_target)) in enumerate(dataset.take(batches_per_epoch)):
            
            batch_loss = train_step(encoder_input, segment_input, decoder_target, encoder, decoder, loss_object, tokenizer, optimizer, BATCH_SIZE)
            total_loss += batch_loss
            
            if pre.VERBOSE == 1:
                print("Epoch %d: Batch %d / %d: Loss %f" % (epoch + 1, batch + 1, batches_per_epoch, batch_loss.numpy()))
        
        
        if val_dataset != None:
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
        else:
            print("Epoch %d --- %d sec: Loss %f" % (epoch + 1, time() - start, total_loss / batches_per_epoch))
            
        if epoch + 1 == MIN_EPOCHS:
            print("Saving model as min epochs %d reached", MIN_EPOCHS)
            save_seq2seq(encoder, decoder, deep_lstm)
        
        if no_improvement_counter >= PATIENCE and epoch > MIN_EPOCHS:
            print("Early stopping, no improvement over minimum in %d epochs" % PATIENCE)
            return epoch + 1
        
    print("Saving model as all %d epochs have been completed" % EPOCHS)
    save_seq2seq(encoder, decoder, deep_lstm)
    return EPOCHS   
            

def train_seq2seq(EPOCHS, BATCH_SIZE, PATIENCE, MIN_EPOCHS, deep_lstm=False, use_segment_embedding=True):
    vocab, persona_length, msg_length, reply_length = pre.get_vocab()
    tokenizer = pre.fit_tokenizer(vocab)
    vocab_size = len(tokenizer.word_index) + 1
    vocab = None
    
    print('Vocab size: %d' % vocab_size)
    print('Input sequence length: %d' % (persona_length + msg_length))
    print('Output sequence length: %d' % reply_length)
    
    BUFFER_SIZE = 15000

    # ------ Pretrain on Movie dataset ------ #
    movie_epochs = 15
    movie_conversations = pre.load_movie_dataset(pre.MOVIE_FN)
    
    encoder_input  = movie_conversations[:, 0]
    decoder_target = np.array([pre.START_SEQ_TOKEN + ' ' + row[1] + ' ' + pre.END_SEQ_TOKEN for row in movie_conversations])
    
    movie_conversations = None
    
    raw = encoder_input[:20]

    # integer encode training data
    segment_input  = np.array([pre.generate_segment_array(msg,  persona_length + msg_length, no_persona=True) for msg in encoder_input])
    encoder_input  = pre.encode_sequences(tokenizer, persona_length + msg_length, encoder_input)
    decoder_target = pre.encode_sequences(tokenizer, reply_length, decoder_target)
    
    dataset = tf.data.Dataset.from_tensor_slices((encoder_input, segment_input,  decoder_target)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    batches_per_epoch = len(encoder_input) // BATCH_SIZE
    
    encoder_input, segment_input, decoder_target = None, None, None

    # load GloVe embeddings, make the embeddings for encoder and decoder tied https://www.aclweb.org/anthology/E17-2025.pdf
    embedding_matrix = pre.load_glove_embedding(tokenizer, pre.GLOVE_FN)
    e_dim = embedding_matrix.shape[1]
    embedding_matrix = Embedding(vocab_size, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=True, mask_zero=True, name="tied_embedding")

    if deep_lstm:
        encoder = DeepEncoder(vocab_size, embedding_matrix, LSTM_DIM, BATCH_SIZE, use_segment_embedding, e_dim)
        decoder = DeepDecoder(vocab_size, embedding_matrix, LSTM_DIM, BATCH_SIZE)
    else:
        encoder = Encoder(vocab_size, embedding_matrix, LSTM_DIM, BATCH_SIZE, use_segment_embedding, e_dim)
        decoder = Decoder(vocab_size, embedding_matrix, LSTM_DIM, BATCH_SIZE)
    
    optimizer = Adam(clipnorm=CLIP_NORM)
    # will give labels as integers instead of one-hot so use sparse CCE
    loss_func = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    movie_epochs = train(dataset, None, batches_per_epoch, None, encoder, decoder, tokenizer, loss_func, optimizer, False, deep_lstm, BATCH_SIZE, EPOCHS, 0, PATIENCE)
    
    print("Finished Pre-training on Cornell Movie Dataset for %d epochs" % movie_epochs)
    
    # do some dummy text generation
    for i in range(len(raw)):
        reply, attn_weights = generate_reply_seq2seq(encoder, decoder, tokenizer, raw[i], persona_length + msg_length, reply_length)
        print("Message:", raw[i])
        print("Reply:", reply + "\n")
    # ------ ------ #

    
    # ------ Pretrain on Daily Dialogue ------ #
    daily_epochs = 15
    conversations = pre.load_dailydialogue_dataset()
    
    encoder_input  = conversations[:, 0]
    decoder_target = np.array([pre.START_SEQ_TOKEN + ' ' + row[1] + ' ' + pre.END_SEQ_TOKEN for row in conversations])
    
    conversations = None
    
    raw = encoder_input[:20]
    
    # integer encode training data
    segment_input  = np.array([pre.generate_segment_array(msg, persona_length + msg_length, no_persona=True) for msg in encoder_input])
    encoder_input  = pre.encode_sequences(tokenizer, persona_length + msg_length, encoder_input)
    decoder_target = pre.encode_sequences(tokenizer, reply_length, decoder_target)
    
    dataset = tf.data.Dataset.from_tensor_slices((encoder_input, segment_input,  decoder_target)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    batches_per_epoch = len(encoder_input) // BATCH_SIZE
    
    encoder_input, segment_input, decoder_target = None, None, None
    
    daily_epochs = train(dataset, None, batches_per_epoch, None, encoder, decoder, tokenizer, loss_func, optimizer, False, deep_lstm, BATCH_SIZE, EPOCHS, 0, PATIENCE)
    
    print("Finished Pre-training on Daily Dialogue for %d epochs" % daily_epochs)
    
    # do some dummy text generation
    for i in range(len(raw)):
        reply, attn_weights = generate_reply_seq2seq(encoder, decoder, tokenizer, raw[i], persona_length + msg_length, reply_length)
        print("Message:", raw[i])
        print("Reply:", reply + "\n")
    # ------ ------ #
    

    # ------ Train on PERSONA-CHAT ------ #
    train_personas, train_data = pre.load_dataset(pre.TRAIN_FN)
    
    # train is a numpy array containing triples [message, reply, persona_index]
    # personas is an numpy array of strings for the personas

    encoder_input  = np.array(
        [pre.truncate(train_personas[int(row[2])], persona_length) + ' ' + pre.SEP_SEQ_TOKEN + ' ' + row[0] for row in train_data])
    decoder_target = np.array(
        [pre.START_SEQ_TOKEN + ' ' + row[1] + ' ' + pre.END_SEQ_TOKEN for row in train_data])
    
    train_data, train_personas = None, None
    
    raw = encoder_input[:20]
    
    # integer encode training data
    segment_input  = np.array([pre.generate_segment_array(msg, persona_length + msg_length) for msg in encoder_input])
    encoder_input  = pre.encode_sequences(tokenizer, persona_length + msg_length, encoder_input)
    decoder_target = pre.encode_sequences(tokenizer, reply_length, decoder_target)
    
    dataset = tf.data.Dataset.from_tensor_slices((encoder_input, segment_input,  decoder_target)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    batches_per_epoch = len(encoder_input) // BATCH_SIZE
    
    encoder_input, segment_input, decoder_target = None, None, None
    
    # load validation set
    val_personas, val_data = pre.load_dataset(pre.VALID_FN)
    
    encoder_input  = np.array(
        [pre.truncate(val_personas[int(row[2])], persona_length) + ' ' + pre.SEP_SEQ_TOKEN + ' ' + row[0] for row in val_data])
    decoder_target = np.array(
        [pre.START_SEQ_TOKEN + ' ' + row[1] + ' ' + pre.END_SEQ_TOKEN for row in val_data])
    
    val_personas, val_data = None, None
    
    val_raw = encoder_input[:20]
    
    segment_input  = np.array([pre.generate_segment_array(msg, persona_length + msg_length) for msg in encoder_input])
    encoder_input  = pre.encode_sequences(tokenizer, persona_length + msg_length, encoder_input)
    decoder_target = pre.encode_sequences(tokenizer, reply_length, decoder_target)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((encoder_input, segment_input,  decoder_target)).shuffle(BUFFER_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
    val_batches_per_epoch = len(encoder_input) // BATCH_SIZE
    
    epochs = train(dataset, val_dataset, batches_per_epoch, val_batches_per_epoch, encoder, decoder, tokenizer, loss_func, optimizer, True, deep_lstm, BATCH_SIZE, EPOCHS, MIN_EPOCHS, PATIENCE)
    
    print("Finished Training on PERSONA-CHAT for %d epochs" % epochs)
    
    # do some dummy text generation
    for i in range(len(raw)):
        reply, attn_weights = generate_reply_seq2seq(encoder, decoder, tokenizer, raw[i], persona_length + msg_length, reply_length)
        print("Message:", raw[i])
        print("Reply:", reply + "\n")
        
    for i in range(len(val_raw)):
        reply, attn_weights = generate_reply_seq2seq(encoder, decoder, tokenizer, val_raw[i], persona_length + msg_length, reply_length)
        print("Message:", val_raw[i])
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
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='input_utterence'),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='segment_tokens')
        ]
        ))
    
    tf.saved_model.save(decoder, decoder_fn, signatures=decoder.call.get_concrete_function(
        [
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='input_word'), 
            tf.TensorSpec(shape=[None, None, LSTM_DIM * 2], dtype=tf.float32, name="encoder_output"),
            tf.TensorSpec(shape=[None, LSTM_DIM * 2], dtype=tf.float32, name="context_vec"),
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

def generate_reply_seq2seq(encoder, decoder, tokenizer, input_msg, in_seq_length, out_seq_length):
    '''
    Generates a reply for a trained sequence to sequence model using greedy search 
    '''
    input_seq = pre.encode_sequences(tokenizer, in_seq_length, [input_msg])
    input_seq = tf.convert_to_tensor(input_seq)
    
    # generate the segment for the input_msg by using seperator token 
    segment_input  = np.array([pre.generate_segment_array(input_msg, in_seq_length)])
    segment_input  = tf.convert_to_tensor(segment_input)
    
    attn_weights = np.zeros((out_seq_length, in_seq_length))
    
    encoder_out, *initial_state = encoder([input_seq, segment_input])
    
    decoder_input = tf.expand_dims([tokenizer.word_index[pre.START_SEQ_TOKEN]], 0)
    context_vec = tf.zeros((encoder_out.shape[0], encoder_out.shape[-1]))
    
    reply = []
    for t in range(out_seq_length):
        softmax_layer, attn_score, context_vec, *initial_state = decoder([decoder_input, encoder_out, context_vec, initial_state])
        
        attn_score = tf.reshape(attn_score, (-1,))
        attn_weights[t] = attn_score.numpy()
        
        # get predicted word by looking at highest node in output softmax layer
        word_index = tf.argmax(softmax_layer[0]).numpy()
        word = pre.index_to_word(word_index, tokenizer)
    
        if word == pre.END_SEQ_TOKEN:
            break
        
        reply.append(word)
        
        decoder_input = tf.expand_dims([word_index], 0)
    
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
    input_seq = tf.convert_to_tensor(input_seq)
    
    segment_input  = np.array([pre.generate_segment_array(input_msg, in_seq_length)])
    segment_input  = tf.convert_to_tensor(segment_input)
    
    encoder_out, *initial_state = encoder_model([input_seq, segment_input])
    
    context_vec = tf.zeros((encoder_out.shape[0], encoder_out.shape[-1]))
    
    # beams will store [ [probability, states, context_vec, word1, word2, ...], ... ]
    beams = [ [0.0, initial_state, context_vec, pre.START_SEQ_TOKEN] for i in range(beam_length)]
    
    prob = 0
    initial_state = 1
    context_vec = 2
    
    # store beam length ^ 2 most likely words [ [probability, word_index, beam_index], ... ]
    most_likely_words_all = [[0.0, 0, 0] for i in range(beam_length * beam_length)]
    
    beam_finished = lambda b : b[-1] == pre.END_SEQ_TOKEN or len(b) - context_vec - 1 >= out_seq_length
    while not reduce(lambda a, b : a and b , map(beam_finished, beams)):
    
        # find beam length most likely words out of all beams (vocab_size * beam length possibilities)
        for b_index in range(len(beams)):
            b = beams[b_index]
            prev_word = b[-1]
            
            if prev_word == pre.END_SEQ_TOKEN:
                # dead end beam so don't generate a new token, update states 
                # and leave most_likely_words for this beam constant
                continue
            
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
    
    # sort beams by length normalized log likelihood descending and only keep text
    replys = [" ".join(b[context_vec + 2:-1]) 
              for b in sorted(beams, key = lambda b:b[0] / (len(b) - context_vec - 1), reverse=True)]
    
    # return list of beam length reply's from most likely to least likely
    return replys
    