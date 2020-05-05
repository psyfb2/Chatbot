# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""
import evaluate
import numpy as np
import text_preprocessing as pre
import tensorflow as tf
from time import time
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import GRU, Dense, Embedding, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from beamsearch import beam_search

# hyperparameters
LSTM_DIM = 512
DROPOUT = 0.2

# global variables
# global variables
loss_object = SparseCategoricalCrossentropy(from_logits=True,
                                            reduction='none')
optimizer = Adam()

encoder = None
decoder = None
tokenizer = None
checkpoint = None
checkpoint_manager = None
batches_per_epoch = None
batches_per_epoch_val = None

class MultipleEncoder(tf.keras.Model):
    ''' 
        1 Layer Bidirectional GRU for each of persona and message encoder
        their hidden states are combined together and can be passed to the decoder
    '''
    def __init__(self, vocab_size, embedding, n_units, batch_size):
        super(MultipleEncoder, self).__init__()
        self.n_units = n_units
        self.batch_size = batch_size
        
        self.embedding = embedding
        
        self.persona_gru1 = Bidirectional(
            GRU(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_persona_gru1"))
        
        self.msg_gru1 = Bidirectional(
            GRU(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_msg_lstm1"))
        
        self.h1_dense = Dense(n_units, activation="tanh")
    
    
    @tf.function
    def call(self, inputs):
        '''
        inputs => [persona, msg, initial_state]
        returns: encoder_persona_states, encoder_msg_states, h1
        '''
        persona, msg, initial_state = inputs
        
        persona_embed = self.embedding(persona)
        msg_embed = self.embedding(msg)
        
        encoder_persona_states, persona_h1, _ = self.persona_gru1(persona_embed, initial_state=[initial_state, initial_state])
        encoder_msg_states, msg_h1, _ = self.msg_gru1(msg_embed)
        
        # add the hidden states of the persona and message encoder
        h1 = tf.concat([persona_h1, msg_h1], axis=-1)
        h1 = self.h1_dense(h1)
        
        return encoder_persona_states, encoder_msg_states, h1
    
    def create_initial_state(self):
        return tf.zeros((self.batch_size, self.n_units))


class MultipleDecoder(tf.keras.Model):
    ''' 1 layer attentive LSTM which performs attention on two seperate encoders '''
    def __init__(self, vocab_size, embedding, n_units, batch_size):
        super(MultipleDecoder, self).__init__()
        self.batch_size = batch_size
        self.n_units = n_units
        
        self.embedding = embedding
        
        self.gru1 = GRU(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="dec_gru1")
        
        # attention
        self.persona_W1 = Dense(n_units)
        self.persona_W2 = Dense(n_units)
        self.persona_V  = Dense(1)
        
        self.msg_W1 = Dense(n_units)
        self.msg_W2 = Dense(n_units)
        self.msg_V = Dense(1)
        
        self.dropout = Dropout(DROPOUT)
        
        # from_logits=True in loss function, it will apply the softmax there for us
        self.out_dense1 = Dense(vocab_size)
    
    @tf.function
    def call(self, inputs):
        '''
        inputs => [input_word, encoder_persona_outputs, encoder_msg_outputs, is_training, [h1]]
        returns => decoder_output, persona_attn_weights, msg_attn_weights, h1
        '''
        input_word, encoder_persona_outputs, encoder_msg_outputs, is_training, hidden = inputs
        h1 = hidden[0]
        
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
        
        input_embed = self.embedding(input_word)
        
        # feed context vector as input into GRU at current timestep
        input_embed = tf.concat([tf.expand_dims(context_vec_concat, 1), input_embed], axis=-1)
        
        decoder_output, h1 = self.gru1(input_embed, initial_state=h1)
        
        # (batch_size, 1, n_units) => (batch_size, n_units)
        decoder_output = tf.reshape(decoder_output, (-1, decoder_output.shape[2]))
                
        decoder_output = self.dropout(decoder_output, training=is_training)
        decoder_output = self.out_dense1(decoder_output)
        
        return decoder_output, persona_attn_weights, msg_attn_weights, h1
        
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
        
        self.persona_gru1 = Bidirectional(
            GRU(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_persona_gru1"))
        self.persona_gru2 = Bidirectional(
            GRU(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_persona_gru2"))
        self.persona_gru3 = Bidirectional(
            GRU(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_persona_gru3"))
        self.persona_gru4 = Bidirectional(
            GRU(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_persona_gru4"))
        
        self.msg_gru1 = Bidirectional(
            GRU(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_msg_gru1"))
        self.msg_gru2 = Bidirectional(
            GRU(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_msg_gru2"))
        self.msg_gru3 = Bidirectional(
            GRU(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_msg_gru3"))
        self.msg_gru4 = Bidirectional(
            GRU(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_msg_gru4"))
        
        self.h1_dense = Dense(n_units, activation="tanh")
        self.h2_dense = Dense(n_units, activation="tanh")
        self.h3_dense = Dense(n_units, activation="tanh")
        self.h4_dense = Dense(n_units, activation="tanh")
        
    @tf.function
    def call(self, inputs):
        '''
        inputs => [persona, msg, initial_state]
        returns: encoder_persona_states, encoder_msg_states, h1, h2, h3, h4
        '''
        persona, msg, initial_state = inputs
        persona_embed = self.embedding(persona)
        msg_embed = self.embedding(msg)
        
        encoder_persona_states, persona_h1, _ = self.persona_gru1(persona_embed, initial_state=[initial_state, initial_state])
        encoder_persona_states, persona_h2, _ = self.persona_gru2(encoder_persona_states, initial_state=[initial_state, initial_state])
        encoder_persona_states, persona_h3, _ = self.persona_gru3(encoder_persona_states, initial_state=[initial_state, initial_state])
        encoder_persona_states, persona_h4, _ = self.persona_gru4(encoder_persona_states, initial_state=[initial_state, initial_state])
        
        encoder_msg_states, msg_h1, _ = self.msg_gru1(msg_embed, initial_state=[initial_state, initial_state])
        encoder_msg_states, msg_h2, _ = self.msg_gru2(encoder_msg_states, initial_state=[initial_state, initial_state])
        encoder_msg_states, msg_h3, _ = self.msg_gru3(encoder_msg_states, initial_state=[initial_state, initial_state])
        encoder_msg_states, msg_h4, _ = self.msg_gru4(encoder_msg_states, initial_state=[initial_state, initial_state])
        
        # concat the hidden states of the persona and message encoder
        h1 = tf.concat([persona_h1, msg_h1], axis=-1)
        h2 = tf.concat([persona_h2, msg_h2], axis=-1)
        h3 = tf.concat([persona_h3, msg_h3], axis=-1)
        h4 = tf.concat([persona_h4, msg_h4], axis=-1)
        
        h1 = self.h1_dense(h1)
        h2 = self.h2_dense(h2)
        h3 = self.h3_dense(h3)
        h4 = self.h4_dense(h4)
        
        return encoder_persona_states, encoder_msg_states, h1, h2, h3, h4
    
    def create_initial_state(self):
        return tf.zeros((self.batch_size, self.n_units))


class DeepMultipleDecoder(tf.keras.Model):
    ''' 4 layer attentive LSTM which performs attention on two seperate encoders '''
    def __init__(self, vocab_size, embedding, n_units, batch_size):
        super(DeepMultipleDecoder, self).__init__()
        self.batch_size = batch_size
        self.n_units = n_units
        
        self.embedding = embedding
        
        self.gru1 = GRU(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="dec_gru1")
        self.gru2 = GRU(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="dec_gru2")
        self.gru3 = GRU(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="dec_gru3")
        self.gru4 = GRU(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="dec_gru4")
        
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
        
        self.dropout = Dropout(DROPOUT)
        
        # from_logits=True in loss function, it will apply the softmax there for us
        self.out_dense1 = Dense(vocab_size)
    
    @tf.function
    def call(self, inputs):
        '''
        inputs => [input_word, encoder_persona_outputs, encoder_msg_outputs, is_training, [h1, h2, h3, h4]]
        returns => decoder_output, persona_attn_weights, msg_attn_weights, h1, h2, h3, h4
        '''
        input_word, encoder_persona_outputs, encoder_msg_outputs, is_training, hidden = inputs
        h1, h2, h3, h4 = hidden
        
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
        
        input_embed = self.embedding(input_word)
        
        # feed context vector as input into GRU at current timestep
        input_embed = tf.concat([tf.expand_dims(context_vec_concat, 1), input_embed], axis=-1)
        
        decoder_output, h1 = self.gru1(input_embed, initial_state=h1)
        decoder_output, h2 = self.gru2(decoder_output, initial_state=h2)
        decoder_output, h3 = self.gru3(decoder_output, initial_state=h3)
        decoder_output, h4 = self.gru4(decoder_output, initial_state=h4)
        
        # (batch_size, 1, n_units) => (batch_size, n_units)
        decoder_output = tf.reshape(decoder_output, (-1, decoder_output.shape[2]))
                
        decoder_output = self.dropout(decoder_output, training=is_training)
        decoder_output = self.out_dense1(decoder_output)
        
        return decoder_output, persona_attn_weights, msg_attn_weights, h1, h2, h3, h4
    
def loss_function(label, pred, loss_object):
    '''
    Calculate mean batch loss for a single timestep
    '''
    # do not calculate loss for padding values
    mask = tf.math.logical_not(tf.math.equal(label, 0))
    loss_ = loss_object(label, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)            

def calc_val_loss(batches_per_epoch, val_dataset):
    total_loss = 0
    
    enc_state = encoder.create_initial_state()
    
    for (batch, (persona, msg, decoder_target)) in enumerate(val_dataset.take(batches_per_epoch)):
        loss = 0

        encoder_persona_states, encoder_msg_states, *initial_state = encoder(
            [persona, msg, enc_state])
        
        decoder_input = tf.expand_dims([tokenizer.word_index[pre.START_SEQ_TOKEN]] * encoder_persona_states.shape[0], 1)
    
        for t in range(1, decoder_target.shape[1]):
            predictions, _, _, *initial_state = decoder([decoder_input, encoder_persona_states, encoder_msg_states, False, initial_state])
            
            loss += loss_function(decoder_target[:, t], predictions, loss_object)
            
            decoder_input = tf.expand_dims(decoder_target[:, t], 1)
        
        batch_loss = (loss / int(decoder_target.shape[1]))
        total_loss += batch_loss
    
    return total_loss

@tf.function
def train_step(persona, msg, decoder_target, BATCH_SIZE):
    '''
    Perform training on a single batch
    persona  => (batch_size, persona_length)
    msg => (batch_size, msg_length)
    decoder_target shape => (batch_size, out_seq_length)
    '''
    loss = 0

    enc_state = encoder.create_initial_state()
    
    with tf.GradientTape() as tape:
        encoder_persona_states, encoder_msg_states, *initial_state = encoder(
            [persona, msg, enc_state])
        
        decoder_input = tf.expand_dims([tokenizer.word_index[pre.START_SEQ_TOKEN]] * BATCH_SIZE, 1)
        
        # Teacher forcing, ground truth for previous word input to the decoder at current timestep
        for t in range(1, decoder_target.shape[1]):
            predictions, _, _, *initial_state = decoder([decoder_input, encoder_persona_states, encoder_msg_states, True, initial_state])
            
            loss += loss_function(decoder_target[:, t], predictions, loss_object)
            
            decoder_input = tf.expand_dims(decoder_target[:, t], 1)
        
    # backpropegate loss
    batch_loss = (loss / int(decoder_target.shape[1]))
        
    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)
        
    optimizer.apply_gradients(zip(gradients, variables))
        
    return batch_loss

def train(dataset, val_dataset, BATCH_SIZE, EPOCHS, MIN_EPOCHS, PATIENCE):
    ''' Train seq2seq model, with the use of early stopping '''        
    min_val_loss = float("inf")
    no_improvement_counter = 0
    
    for epoch in range(EPOCHS):
        start = time()
        
        total_loss = 0
        
        for (batch, (persona, msg, decoder_target)) in enumerate(dataset.take(batches_per_epoch)):
            batch_loss = train_step(persona, msg, decoder_target, BATCH_SIZE)
            total_loss += batch_loss
            
            if pre.VERBOSE == 1:
                print("Epoch %d: Batch %d / %d: Loss %f" % (epoch + 1, batch + 1, batches_per_epoch, batch_loss.numpy()))
        
        if val_dataset != None:
            val_loss = calc_val_loss(batches_per_epoch_val, val_dataset)
        
            if val_loss < min_val_loss:
                save_path = checkpoint_manager.save()
                print("Saving model as total best val loss decreased from %f to %f" % (min_val_loss, val_loss))
                print("seq2seq checkpoint saved: %s" % save_path)
                no_improvement_counter = 0
                min_val_loss = val_loss
            else:
                no_improvement_counter += 1
     
            print("Epoch %d --- %d sec: Loss %f, val_loss: %f" % (epoch + 1, time() - start, total_loss / batches_per_epoch, val_loss / batches_per_epoch_val))
        else:
            print("Epoch %d --- %d sec: Loss %f" % (epoch + 1, time() - start, total_loss / batches_per_epoch))
            
        if epoch + 1 == MIN_EPOCHS:
            save_path = checkpoint_manager.save()
            print("Saving model as min epochs %d reached" % MIN_EPOCHS)
            print("seq2seq checkpoint saved: %s" % save_path)
        
        if no_improvement_counter >= PATIENCE and epoch > MIN_EPOCHS:
            print("Early stopping, no improvement over minimum in %d epochs" % PATIENCE)
            return epoch + 1
    
    return EPOCHS            

def train_multiple_encoders(EPOCHS, BATCH_SIZE, PATIENCE, MIN_EPOCHS, deep_lstm=False, pre_train=True):
    global checkpoint
    global checkpoint_manager
    global encoder
    global decoder
    global tokenizer
    global batches_per_epoch
    global batches_per_epoch_val
    
    vocab, persona_length, msg_length, reply_length = pre.get_vocab()
    tokenizer = pre.fit_tokenizer(vocab)
    vocab_size = len(tokenizer.word_index) + 1
    vocab = None

    out_seq_length = reply_length
    
    print('Vocab size: %d' % vocab_size)
    print('Persona sequence length: %d' % persona_length)
    print('Message sequence length: %d' % msg_length)
    print('Output sequence length: %d' % out_seq_length)
    
    BUFFER_SIZE = 15000
    
    embedding_matrix = pre.load_glove_embedding(tokenizer, pre.GLOVE_FN)
    embedding_matrix = Embedding(vocab_size, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=True, mask_zero=True, name="tied_embedding")
    
    if deep_lstm:        
        encoder = DeepMultipleEncoder(vocab_size, embedding_matrix, LSTM_DIM, BATCH_SIZE)
        decoder = DeepMultipleDecoder(vocab_size, embedding_matrix, LSTM_DIM, BATCH_SIZE)
        fn = pre.MULTIENC_DEEP_CHECKPOINT_FN
    else:        
        encoder = MultipleEncoder(vocab_size, embedding_matrix, LSTM_DIM, BATCH_SIZE)
        decoder = MultipleDecoder(vocab_size, embedding_matrix, LSTM_DIM, BATCH_SIZE)
        fn = pre.MULTIENC_CHECKPOINT_FN
    
    checkpoint = tf.train.Checkpoint(encoder=encoder,
                                     decoder=decoder,
                                     optimizer=optimizer)
    
    checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                    fn,
                                                    max_to_keep=2)
    
    if pre_train:
        # ------ Pretrain on Movie dataset ------ #
        movie_epochs = 3
        movie_conversations = pre.load_movie_dataset(pre.MOVIE_FN)
    
        encoder_msg_input  = movie_conversations[:, 0]
        encoder_persona_input = np.array(["" for _ in encoder_msg_input])
        decoder_target = np.array([pre.START_SEQ_TOKEN + ' ' + row[1] + ' ' + pre.END_SEQ_TOKEN for row in movie_conversations])
    
        movie_conversations = None
        
        msg_raw = encoder_msg_input[:20]
        persona_raw = encoder_persona_input[:20]
        
        # integer encode training data
        encoder_persona_input = pre.encode_sequences(tokenizer, persona_length, encoder_persona_input)
        encoder_msg_input  = pre.encode_sequences(tokenizer, msg_length, encoder_msg_input)
        decoder_target = pre.encode_sequences(tokenizer, out_seq_length, decoder_target)
        
        dataset = tf.data.Dataset.from_tensor_slices((encoder_persona_input, encoder_msg_input,  decoder_target)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        batches_per_epoch = len(encoder_persona_input) // BATCH_SIZE
        
        encoder_persona_input, encoder_msg_input, decoder_target = None, None, None
        
        epochs = train(dataset, None, BATCH_SIZE, movie_epochs, 0, movie_epochs)
        
        print("Finished Pre-training on Movie dataset for %d epochs" % epochs)
        
        # do some dummy text generation
        for i in range(len(msg_raw)):
            reply, persona_attn_weights, msg_attn_weights = generate_reply_multiple_encoder(encoder, decoder, tokenizer, persona_raw[i], msg_raw[i], persona_length, msg_length, out_seq_length)
            print("Message:", msg_raw[i])
            print("Reply:", reply + "\n")
        # ------ ------ #
        
        # ------ Pretrain on Daily Dialogue ------ #
        daily_epochs = 3
        conversations = pre.load_dailydialogue_dataset()
        
        encoder_msg_input  = conversations[:, 0]
        encoder_persona_input = np.array(["" for _ in encoder_msg_input])
        decoder_target = np.array([pre.START_SEQ_TOKEN + ' ' + row[1] + ' ' + pre.END_SEQ_TOKEN for row in conversations])
        
        conversations = None
        
        msg_raw = encoder_msg_input[:20]
        persona_raw = encoder_persona_input[:20]
        
        # integer encode training data
        encoder_persona_input = pre.encode_sequences(tokenizer, persona_length, encoder_persona_input)
        encoder_msg_input  = pre.encode_sequences(tokenizer, msg_length, encoder_msg_input)
        decoder_target = pre.encode_sequences(tokenizer, out_seq_length, decoder_target)
        
        dataset = tf.data.Dataset.from_tensor_slices((encoder_persona_input, encoder_msg_input,  decoder_target)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        batches_per_epoch = len(encoder_persona_input) // BATCH_SIZE
        
        encoder_persona_input, encoder_msg_input, decoder_target = None, None, None
        
        epochs = train(dataset, None, BATCH_SIZE, daily_epochs, 0, daily_epochs)
        
        print("Finished Pre-training on Daily Dialogue for %d epochs" % daily_epochs)
        
        # do some dummy text generation
        for i in range(len(msg_raw)):
            reply, persona_attn_weights, msg_attn_weights = generate_reply_multiple_encoder(encoder, decoder, tokenizer, persona_raw[i], msg_raw[i], persona_length, msg_length, out_seq_length)
            print("Message:", msg_raw[i])
            print("Reply:", reply + "\n")
        # ------ ------ #

    # ------ Train on PERSONA-CHAT ------ #
    dataset, num_examples, persona_raw, msg_raw = data_pipeline(
        pre.TRAIN_FN, tokenizer, persona_length, msg_length, reply_length, BATCH_SIZE)
    
    val_dataset, num_examples_val, persona_raw_val, msg_raw_val = data_pipeline(
        pre.VALID_FN, tokenizer, persona_length, msg_length, reply_length, BATCH_SIZE)
    
    batches_per_epoch = num_examples // BATCH_SIZE
    batches_per_epoch_val = num_examples_val // BATCH_SIZE

    epochs = train(dataset, val_dataset, BATCH_SIZE, EPOCHS, MIN_EPOCHS, PATIENCE)
    
    print("Finished Training on PERSONA-CHAT for %d epochs" % epochs)
    
    # do some dummy text generation
    print("Responses from the training set:\n")
    for i in range(len(persona_raw)):
        reply, persona_attn_weights, msg_attn_weights = generate_reply_multiple_encoder(encoder, decoder, tokenizer, persona_raw[i], msg_raw[i], persona_length, msg_length, out_seq_length)
        print("Persona:", persona_raw[i])
        print("Message:", msg_raw[i])
        print("Reply:", reply + "\n")
    
    print("\nResponses from the validation set:\n")
    for i in range(len(persona_raw)):
        reply, persona_attn_weights, msg_attn_weights = generate_reply_multiple_encoder(encoder, decoder, tokenizer, persona_raw_val[i], msg_raw_val[i], persona_length, msg_length, out_seq_length)
        print("Persona:", persona_raw_val[i])
        print("Message:", msg_raw_val[i])
        print("Reply:", reply + "\n")
    # ------ ------ #
        
def data_pipeline(filename, tokenizer, persona_length, msg_length, out_seq_length, BATCH_SIZE, drop_remainder=True):
    ''' Load and integer encode persona chat dataset '''
    personas, data = pre.load_dataset(filename)
    
    encoder_persona_input  = np.array(
        [pre.START_SEQ_TOKEN + ' ' + personas[int(row[2])] + ' ' 
         + pre.END_SEQ_TOKEN for row in data])
    
    encoder_msg_input = np.array(
        [pre.START_SEQ_TOKEN + ' ' + row[0] + ' ' + pre.END_SEQ_TOKEN for row in data])
    decoder_target = np.array(
        [pre.START_SEQ_TOKEN + ' ' + row[1] + ' ' + pre.END_SEQ_TOKEN for row in data])
    
    persona_raw = encoder_persona_input[:20]
    msg_raw = encoder_msg_input[:20]
    
    encoder_persona_input  = pre.encode_sequences(tokenizer, persona_length, encoder_persona_input)
    encoder_msg_input = pre.encode_sequences(tokenizer, msg_length, encoder_msg_input)
    decoder_target = pre.encode_sequences(tokenizer, out_seq_length, decoder_target)
    
    dataset = tf.data.Dataset.from_tensor_slices((encoder_persona_input, encoder_msg_input,  decoder_target))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=drop_remainder)
    
    num_examples = len(encoder_persona_input)
    
    return dataset, num_examples, persona_raw, msg_raw


def generate_reply_multiple_encoder(encoder, decoder, tokenizer, persona, msg, persona_length, msg_length, out_seq_length):
    '''
    Generates a reply for a trained multiple encoder sequence to sequence model using greedy search 
    persona and msg should be wrapped around SOS and EOS tokens
    '''   
    persona = pre.encode_sequences(tokenizer, persona_length, [persona])
    persona = tf.convert_to_tensor(persona)
    
    msg = pre.encode_sequences(tokenizer, msg_length, [msg])
    msg = tf.convert_to_tensor(msg)
    
    persona_attn_weights = np.zeros((out_seq_length, persona_length))
    msg_attn_weights = np.zeros((out_seq_length, msg_length))
    
    enc_state = tf.zeros((1, LSTM_DIM))
    encoder_persona_states, encoder_msg_states, *initial_state = encoder(
        [persona, msg, enc_state])
    
    decoder_input = tf.expand_dims([tokenizer.word_index[pre.START_SEQ_TOKEN]], 0)
    
    reply = []
    for t in range(out_seq_length):
        softmax_layer, persona_attn_score, msg_attn_score, *initial_state = decoder(
            [decoder_input, encoder_persona_states, encoder_msg_states, False, initial_state])
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


class ChatBot(evaluate.BaseBot):
    def __init__(self, deep_model):
        '''
        Use this class from the outside for interaction and evaluation.

        Parameters
        ----------
        deep_model : boolean
            Load shallow or stacked trained model

        Returns
        -------
        None

        '''
        vocab, self.persona_length, self.msg_length, self.reply_length = pre.get_vocab()
        self.tokenizer = pre.fit_tokenizer(vocab)
        
        vocab_size = len(self.tokenizer.word_index) + 1
        
        # no need to load 300d glove embedding
        embedding_matrix = Embedding(vocab_size, 300, trainable=True, 
                                     mask_zero=True, name="tied_embedding")
        
        if deep_model:
            self.encoder = DeepMultipleEncoder(vocab_size, embedding_matrix, LSTM_DIM, 64)
            self.decoder = DeepMultipleDecoder(vocab_size, embedding_matrix, LSTM_DIM, 64)
            fn = pre.MULTIENC_DEEP_CHECKPOINT_FN
        else:
            self.encoder = MultipleEncoder(vocab_size, embedding_matrix, LSTM_DIM, 64)
            self.decoder = MultipleDecoder(vocab_size, embedding_matrix, LSTM_DIM, 64)
            fn = pre.MULTIENC_CHECKPOINT_FN
        
        ckpt = tf.train.Checkpoint(encoder=self.encoder, decoder=self.decoder)
        manager = tf.train.CheckpointManager(ckpt, fn,
                                             max_to_keep=2)
        
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            pass
        else:
            raise FileNotFoundError("Failed load seq2seq checkpoint")
        
        
        # for plotting attention
        self.persona_attn_weights = None
        self.msg_attn_weights = None
        self.last_persona = None
        self.last_msg = None
        self.last_reply = None
        
    
    def reply(self, persona, message):
        '''
        Generate a reply using the saved model

        Parameters
        ----------
        persona : str
            persona description
        message : str
            message

        Returns
        -------
        str
            reply

        '''
        self.last_persona = pre.START_SEQ_TOKEN + ' ' + persona + ' ' + pre.END_SEQ_TOKEN
        self.last_msg = pre.START_SEQ_TOKEN + ' ' + message + ' ' + pre.END_SEQ_TOKEN
        
        reply, self.persona_attn_weights, self.msg_attn_weights = (
             generate_reply_multiple_encoder(self.encoder, self.decoder, 
                                             self.tokenizer, persona, message, 
                                             self.persona_length, 
                                             self.msg_length, self.reply_length))
        
        self.last_reply = reply
        return self.last_reply
    
    def plot_attn(self):
        '''
        Plot attention for the last generated reply

        Returns
        -------
        None.

        '''
        if self.persona_attn_weights == None:
            return
        pre.plot_attention(self.persona_attn_weights, self.last_persona, self.last_reply)
        pre.plot_attention(self.msg_attn_weights, self.last_msg, self.last_reply)
        
        
    def beam_search_reply(self, persona, message, beam_width):
        '''
        Beam Search

        Parameters
        ----------
        persona : str
            persona description
        message : str
            message
        beam_width : int
            beam_width to use 

        Returns
        -------
        list of str
            beam_width replies from most likely to least likely

        '''
        def process_inputs(persona, msg):
            persona = pre.START_SEQ_TOKEN + ' ' + persona + ' ' + pre.END_SEQ_TOKEN
            msg = pre.START_SEQ_TOKEN + ' ' + msg + ' ' + pre.END_SEQ_TOKEN
            
            persona = pre.encode_sequences(self.tokenizer, self.persona_length, [persona])
            persona = tf.convert_to_tensor(persona)
            
            msg = pre.encode_sequences(self.tokenizer, self.msg_length, [msg])
            msg = tf.convert_to_tensor(msg)
            
            enc_state = tf.zeros((1, LSTM_DIM))
            encoder_persona_states, encoder_msg_states, *initial_state = self.encoder(
                [persona, msg, enc_state])
            
            return [encoder_persona_states, encoder_msg_states, initial_state]
        
        def pred_function(inputs, state, last_word):
            # decoder step
            decoder_input = tf.expand_dims([last_word], 0)
            
            if state is None:
                # first call to pred function
                encoder_persona_states, encoder_msg_states, initial_state = inputs
            else:
                encoder_persona_states, encoder_msg_states, initial_state = state
                
            logits, _, _, *initial_state = self.decoder(
            [decoder_input, encoder_persona_states, encoder_msg_states, False, initial_state])
            
            # return output and new state 
            return logits[0], [encoder_persona_states, encoder_msg_states, initial_state]
        
        sos = self.tokenizer.word_index[pre.START_SEQ_TOKEN]
        eos = self.tokenizer.word_index[pre.END_SEQ_TOKEN]
        
        replys = beam_search(persona, message, process_inputs, pred_function, 
                             self.reply_length, sos, eos, beam_width)
        
        replys_str = []
        for reply in replys:
            single_reply_str = []
            for i in reply:
                word = pre.index_to_word(i, self.tokenizer)
                single_reply_str.append(word)
            replys_str.append(" ".join(single_reply_str))
        
        return replys_str
    
    def eval_f1(self):
        '''
        Get test set F1 score: 2 . (precision * recall) / (precision + recall)
        where an F1 score is calculated for each reply
        Note: this can take some time

        Returns
        -------
        float
            mean F1 score

        '''
        get_reply = (lambda persona, msg : 
                     generate_reply_multiple_encoder(self.encoder, self.decoder,
                                                     self.tokenizer, 
                                                     pre.START_SEQ_TOKEN + ' ' + persona + ' '
                                                     + pre.END_SEQ_TOKEN,
                                                     pre.START_SEQ_TOKEN + ' ' + msg
                                                     + pre.END_SEQ_TOKEN, self.persona_length,
                                                     self.msg_length, self.reply_length)[0])
        return evaluate.f1(get_reply)
            
    
    def eval_ppl(self):
        '''
        Get test set perplexity
        Note: this can take some time
        
        Returns
        -------
           float
               perplexity meassure
        '''
        # perplexity is equivalant to cross entropy loss over all timesteps
        # over all training examples / number of training examples
        # not raising to the power of two in this meassure of perplexity
        # to prevent return value from being huge
        # but stil has the exactly the same scope for comparison
        # without exponentiation
        batch_size = 256
        dataset, num_examples, _, _ = data_pipeline(
        pre.TEST_FN, self.tokenizer, self.persona_length, 
        self.msg_length, self.reply_length, batch_size, False)

        ppl = 0
        
        enc_state = tf.zeros((batch_size, LSTM_DIM))
        
        for (persona, msg, decoder_target) in dataset:
            encoder_persona_states, encoder_msg_states, *initial_state = self.encoder(
                [persona, msg, enc_state])
            
            decoder_input = tf.expand_dims(
                [self.tokenizer.word_index[pre.START_SEQ_TOKEN]] * encoder_persona_states.shape[0], 1)
        
            for t in range(1, decoder_target.shape[1]):
                predictions, _, _, *initial_state = self.decoder(
                    [decoder_input, encoder_persona_states, encoder_msg_states, 
                     False, initial_state])
                
                ppl += evaluate.CCE_loss(decoder_target[:, t], predictions)
                
                decoder_input = tf.expand_dims(decoder_target[:, t], 1)
    
        ppl = ppl / num_examples
        return ppl.numpy()
    