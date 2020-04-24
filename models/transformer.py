# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import text_preprocessing as pre
import tensorflow_datasets as tfds

SOS = 0
SEP = 1
EOS = 4


def create_subword_tokenizer():
    ''' Create subword tokenizer from PERSONA-CHAT dataset '''
    train_personas, train_data = pre.load_dataset(pre.TRAIN_FN)
    
    all_text = [p for p in train_personas] + [triplet[0] for triplet in train_data] + [triplet[1] for triplet in train_data]
    
    return tfds.features.text.SubwordTextEncoder.build_from_corpus(
        all_text, target_vocab_size=8192)

def encode_example(sentence, tokenizer):
    ''' Convert raw text sentence into subword tokenized form '''
    sentence = [SOS] + tokenizer.encode(sentence.numpy()) + [EOS]
    return sentence


def train_transformer(BATCH_SIZE):
    tokenizer = create_subword_tokenizer()
    sample_str = "optimus prime goodness"
    assert tokenizer.decode(tokenizer.encode(sample_str)) == sample_str
    
    BUFFER_SIZE = 20000
    
    

train_transformer()

'''
    # ------ Pretrain on Movie dataset ------ #
    movie_epochs = 15
    movie_conversations = pre.load_movie_dataset(pre.MOVIE_FN)
    
    encoder_input  = movie_conversations[:, 0]
    decoder_target = np.array([pre.START_SEQ_TOKEN + ' ' + row[1] + ' ' + pre.END_SEQ_TOKEN for row in movie_conversations])
    movie_conversations = None
    
    
    
    raw = encoder_input[:20]

    # integer encode training data
    segment_input  = np.array([generate_segment_array(msg, in_seq_length, no_persona=True) for msg in encoder_input])
    encoder_input  = pre.encode_sequences(tokenizer, in_seq_length, encoder_input)
    decoder_target = pre.encode_sequences(tokenizer, out_seq_length, decoder_target)
    
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
    # will give labels as integers instead of one hot so use sparse CCE
    loss_func = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    movie_epochs = train(encoder_input, segment_input, decoder_target, encoder, decoder, tokenizer, loss_func, optimizer, False, deep_lstm, BATCH_SIZE, movie_epochs, PATIENCE)
    
    print("Finished Pre-training on Cornell Movie Dataset for %d epochs" % movie_epochs)
    
    # do some dummy text generation
    for i in range(len(raw)):
        reply, attn_weights = generate_reply_seq2seq(encoder, decoder, tokenizer, raw[i], in_seq_length, out_seq_length)
        print("Message:", raw[i])
        print("Reply:", reply + "\n")
    # ------ ------ #

    
    # ------ Pretrain on Daily Dialogue ------ #
    daily_epochs = 50
    conversations = pre.load_dailydialogue_dataset()
    
    encoder_input  = conversations[:, 0]
    decoder_target = np.array([pre.START_SEQ_TOKEN + ' ' + row[1] + ' ' + pre.END_SEQ_TOKEN for row in conversations])
    
    conversations = None
    
    raw = encoder_input[:20]
    
    # integer encode training data
    segment_input  = np.array([generate_segment_array(msg, in_seq_length, no_persona=True) for msg in encoder_input])
    encoder_input  = pre.encode_sequences(tokenizer, in_seq_length, encoder_input)
    decoder_target = pre.encode_sequences(tokenizer, out_seq_length, decoder_target)
    
    daily_epochs = train(encoder_input, segment_input, decoder_target, encoder, decoder, tokenizer, loss_func, optimizer, False, deep_lstm, BATCH_SIZE, daily_epochs, PATIENCE)
    
    print("Finished Pre-training on Daily Dialogue for %d epochs" % daily_epochs)
    
    # do some dummy text generation
    for i in range(len(raw)):
        reply, attn_weights = generate_reply_seq2seq(encoder, decoder, tokenizer, raw[i], in_seq_length, out_seq_length)
        print("Message:", raw[i])
        print("Reply:", reply + "\n")
    # ------ ------ #
    
    
    # ------ Train on PERSONA-CHAT ------ #
    train_personas, train_data = pre.load_dataset(pre.TRAIN_FN)
    
    # train is a numpy array containing triples [message, reply, persona_index]
    # personas is an numpy array of strings for the personas

    encoder_input  = np.array([train_personas[int(row[2])] + ' ' + pre.SEP_SEQ_TOKEN + ' ' + row[0] for row in train_data])
    decoder_target = np.array([pre.START_SEQ_TOKEN + ' ' + row[1] + ' ' + pre.END_SEQ_TOKEN for row in train_data])
    
    train_data, train_personas = None, None
    
    raw = encoder_input[:20]
    
    # integer encode training data
    segment_input  = np.array([generate_segment_array(msg, in_seq_length) for msg in encoder_input])
    encoder_input  = pre.encode_sequences(tokenizer, in_seq_length, encoder_input)
    decoder_target = pre.encode_sequences(tokenizer, out_seq_length, decoder_target)
    
    epochs = train(encoder_input, segment_input, decoder_target, encoder, decoder, tokenizer, loss_func, optimizer, True, deep_lstm, BATCH_SIZE, EPOCHS, PATIENCE)
    
    print("Finished Training on PERSONA-CHAT for %d epochs" % epochs)
    
    # do some dummy text generation
    for i in range(len(raw)):
        reply, attn_weights = generate_reply_seq2seq(encoder, decoder, tokenizer, raw[i], in_seq_length, out_seq_length)
        print("Message:", raw[i])
        print("Reply:", reply + "\n")
    # ------ ------ #
'''