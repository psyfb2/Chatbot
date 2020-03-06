# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""
import numpy as np
import text_preprocessing as pre
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RepeatVector, TimeDistributed, Dense, Embedding, LSTM
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam

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

def train_autoencoder(LSTM_DIMS=512, EPOCHS=10, BATCH_SIZE=64, CLIP_NORM=5):
    # feed the model data pairs of (persona + message, reply)
    train, train_personas = pre.load_object(pre.TRAIN_PKL_FN)
    
    # train is a numpy array containing triples [message, reply, persona_index]
    # personas is an numpy array of strings for the personas
    
    tokenizer = pre.fit_tokenizer( np.concat(train[:, 0], train[:, 1]) )
    vocab_size = len(tokenizer.word_index) + 1

    encoder_input  = np.array([train_personas[int(row[2])] + ' ' + row[0] for row in train])
    decoder_target = train[:, 1]
    raw = encoder_input
    
    in_seq_length = pre.max_seq_length(encoder_input)
    out_seq_length = pre.max_seq_length(decoder_target)
    
    print('Vocab size: %d' % vocab_size)
    print('Input sequence length: %d' % in_seq_length)
    print('Output sequence length: %d' % out_seq_length)

    # prepare training data
    encoder_input  = pre.encode_sequences(tokenizer, in_seq_length, encoder_input)
    decoder_target = pre.encode_sequences(tokenizer, out_seq_length, decoder_target)
    decoder_target = pre.encode_output(decoder_target, vocab_size)

    # load GloVe embeddings
    embedding_matrix = pre.load_glove_embedding(tokenizer)
    
    model = autoencoder_model(vocab_size, in_seq_length, out_seq_length, LSTM_DIMS, embedding_matrix)
    model.compile(optimizer=Adam(clipnorm=CLIP_NORM), loss='categorical_crossentropy')
    model.summary()
    model.fit(encoder_input, decoder_target,
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)     
    plot_model(model, to_file=pre.AUTOENC_MODEL_IMAGE_FN, show_shapes=True)
    
    # save the tokenizer and model so it can be used for prediction
    pre.save_object(tokenizer, pre.AUTOENC_TOKENIZER_PKL_FN)
    pre.save_object(out_seq_length, pre.AUTOENC_MAX_OUT_LEN_PKL_FN)
    pre.save_object(in_seq_length, pre.AUTOENC_MAX_IN_LEN_PKL_FN)
    model.save(pre.AUTOENC_MODEL_FN)
    
    # do some dummy text generation
    for i in range(20):
        input_seq = encoder_input[i:i+1]
        reply, _ = generate_reply_autoencoder(model, tokenizer, input_seq)
        print("Message:", raw[i])
        print("Reply:", reply + "\n")


    
''' Given source sentence, generate the model inference as a target sentence '''
def generate_reply_autoencoder(model, tokenizer, input_seq):
    prediction = model.predict(input_seq, verbose=0)
    prediction = prediction[0]
    integers = [np.argmax(vec) for vec in prediction]
    
    target = []
    for i in integers:
        word = pre.index_to_word(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)