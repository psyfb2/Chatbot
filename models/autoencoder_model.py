# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""
import numpy as np
import text_preprocessing as pre
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RepeatVector, TimeDistributed, Dense, Embedding, LSTM
from tensorflow.keras.utils import plot_model

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
    #plot_model(model, to_file=pre.AUTOENC_MODEL_IMAGE_FN, show_shapes=True)
    return model

def train_autoencoder(LSTM_DIMS=512, EPOCHS=10, BATCH_SIZE=64, CLIP_NORM=5, PATIENCE=5):
    vocab, persona_length, msg_length, reply_length = pre.get_vocab()
    tokenizer = pre.fit_tokenizer(vocab)
    # + 1 for padding 0 value not in word_index dictionary
    vocab_size = len(tokenizer.word_index) + 1
    vocab = None
    
    # feed the model data pairs of (persona + message, reply)
    _, train = pre.load_dataset(pre.TRAIN_FN)
    
    # train is a numpy array containing triples [message, reply, persona_index]

    # train this auto_encoder without using persona
    encoder_input  = train[:, 0]
    decoder_target = train[:, 1]
    
    raw = encoder_input[:20]
    
    in_seq_length = msg_length
    out_seq_length = reply_length
    
    print('Vocab size: %d' % vocab_size)
    print('Input sequence length: %d' % in_seq_length)
    print('Output sequence length: %d' % out_seq_length)

    # prepare training data
    encoder_input  = pre.encode_sequences(tokenizer, in_seq_length, encoder_input)
    decoder_target = pre.encode_sequences(tokenizer, out_seq_length, decoder_target)
    
    # load GloVe embeddings
    embedding_matrix = pre.load_glove_embedding(tokenizer, pre.GLOVE_FN)
    
    model = autoencoder_model(vocab_size, in_seq_length, out_seq_length, LSTM_DIMS, embedding_matrix)
    
    epochs = train_on_batches(model, encoder_input, decoder_target, vocab_size, BATCH_SIZE, EPOCHS, PATIENCE)
    
    print("Trained autoencoder model for %d epochs" % epochs)
    
    # do some dummy text generation
    for i in range(len(raw)):
        reply = generate_reply_autoencoder(model, tokenizer, raw[i], in_seq_length)
        print("Message:", raw[i])
        print("Reply:", reply + "\n")


''' Given source sentence, generate the model inference as a target sentence '''
def generate_reply_autoencoder(model, tokenizer, input_msg, in_seq_length):
    input_seq = pre.encode_sequences(tokenizer, in_seq_length, [input_msg])
    
    prediction = model.predict(input_seq, verbose=0)
    prediction = prediction[0]
    integers = [np.argmax(vec) for vec in prediction]
    
    target = []
    for i in integers:
        word = pre.index_to_word(i, tokenizer)
        if word is None:
            # using pad character as EOS token, None means EOS token reached
            break
        target.append(word)
    return ' '.join(target)

def train_on_batches(model, encoder_input, decoder_target, vocab_size, BATCH_SIZE, EPOCHS, PATIENCE):
    ''' Trains the model batch by batch for the purposes of reducing memory usage 
    give integer encoded decoder target, will one hot encode this per-batch 
    also will create a validation set from 5% of the train set and do early stopping'''
    encoder_input, encoder_input_val, decoder_target, decoder_target_val = train_test_split(encoder_input, decoder_target, shuffle=False, test_size=0.05)
    
    min_val_loss = float("inf")
    no_improvement_counter = 0
    
    for epoch in range(EPOCHS):
        loss = train_step(model, encoder_input, decoder_target, vocab_size, BATCH_SIZE, False)
        val_loss = train_step(model, encoder_input_val, decoder_target_val, vocab_size, BATCH_SIZE, True)
        
        if val_loss < min_val_loss:
            model.save(pre.AUTOENC_MODEL_FN)
            print("Saving model as best val loss decreased from %f to %f" % (min_val_loss, val_loss))
            no_improvement_counter = 0
            min_val_loss = val_loss
        else:
            no_improvement_counter += 1
            
        print("EPOCH %d loss: %f, val loss: %f" % (epoch + 1, loss, val_loss))
        
        if no_improvement_counter >= PATIENCE:
            print("Early stopping, no improvement over minimum in %d epochs" % PATIENCE)
            return epoch + 1
    
    model.save(pre.AUTOENC_MODEL_FN)
    
    return EPOCHS

def train_step(model, encoder_input, decoder_target, vocab_size, BATCH_SIZE, only_get_loss=False):
    loss = 0
    for i in range(0, encoder_input.shape[0] - BATCH_SIZE + 1, BATCH_SIZE):
        batch_encoder_input = encoder_input[i:i+BATCH_SIZE]
        batch_decoder_target = pre.encode_output(
            decoder_target[i:i+BATCH_SIZE], vocab_size)
        
        if only_get_loss:
            l = model.evaluate(
            batch_encoder_input, batch_decoder_target, verbose=0)
        else:
            model.train_on_batch(
                batch_encoder_input, batch_decoder_target)
            l = model.evaluate(
            batch_encoder_input, batch_decoder_target, verbose=0)
            
        loss += l
        if pre.VERBOSE != 0:
            print("BATCH %d / %d - loss: %f" % (i + BATCH_SIZE, encoder_input.shape[0], l))
    
    loss = loss / (encoder_input.shape[0] // BATCH_SIZE)
    return loss
        
        
        