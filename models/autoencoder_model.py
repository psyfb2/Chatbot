# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""
import numpy as np
import text_preprocessing as pre
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
    plot_model(model, to_file=pre.AUTOENC_MODEL_IMAGE_FN, show_shapes=True)
    return model

def train_autoencoder(LSTM_DIMS=512, EPOCHS=10, BATCH_SIZE=64, CLIP_NORM=5, train_by_batch=True):
    vocab, persona_length, msg_length, reply_length = pre.get_vocab()
    tokenizer = pre.fit_tokenizer(vocab)
    # + 1 for padding 0 value not in word_index dictionary
    vocab_size = len(tokenizer.word_index) + 1
    vocab = None
    
    # feed the model data pairs of (persona + message, reply)
    train_personas, train = pre.load_dataset(pre.TRAIN_FN)
    
    # train is a numpy array containing triples [message, reply, persona_index]
    # personas is an numpy array of strings for the personas

    encoder_input  = np.array([train_personas[int(row[2])] + ' ' + pre.SEP_SEQ_TOKEN + ' ' + row[0] for row in train])
    decoder_target = train[:, 1]
    
    raw = encoder_input[:20]
    
    in_seq_length = persona_length + msg_length
    out_seq_length = reply_length
    
    print('Vocab size: %d' % vocab_size)
    print('Input sequence length: %d' % in_seq_length)
    print('Output sequence length: %d' % out_seq_length)

    # prepare training data
    encoder_input  = pre.encode_sequences(tokenizer, in_seq_length, encoder_input)
    decoder_target = pre.encode_sequences(tokenizer, out_seq_length, decoder_target)
    
    # load GloVe embeddings
    embedding_matrix = pre.load_glove_embedding(tokenizer)
    
    model = autoencoder_model(vocab_size, in_seq_length, out_seq_length, LSTM_DIMS, embedding_matrix)
    
    if train_by_batch:
        train_on_batches(model, encoder_input, decoder_target, vocab_size, BATCH_SIZE, EPOCHS)
    else:
        decoder_target = pre.encode_output(decoder_target, vocab_size)
        model.fit(encoder_input, decoder_target,
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)     
    plot_model(model, to_file=pre.AUTOENC_MODEL_IMAGE_FN, show_shapes=True)
    
    # save the model so it can be used for prediction
    model.save(pre.AUTOENC_MODEL_FN)
    
    print("Trained autoencoder model for %d epochs" % EPOCHS)
    
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

''' Trains the model batch by batch for the purposes of reducing memory usage 
    give integer encoded decoder target, will one hot encode this per-batch '''
def train_on_batches(model, encoder_input, decoder_target, vocab_size, BATCH_SIZE, EPOCHS, verbose=1):
    for epoch in range(EPOCHS):
        losses = []
        for i in range(0, encoder_input.shape[0] - BATCH_SIZE + 1, BATCH_SIZE):
            batch_encoder_input = encoder_input[i:i+BATCH_SIZE]
            batch_decoder_target = pre.encode_output(
                decoder_target[i:i+BATCH_SIZE], vocab_size)
            
            model.train_on_batch(
                batch_encoder_input, batch_decoder_target)
            
            l = model.evaluate(
                batch_encoder_input, batch_decoder_target)
            
            losses.append(l)
            
            if verbose == 1:
                print("BATCH %d / %d - loss: %f" % (i + BATCH_SIZE, encoder_input.shape[0], l))
            
        print("Mean loss in epoch %d : %f : " % (epoch + 1, np.mean(losses)))