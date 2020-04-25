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
EOS = 2

PSN = 1
MSG = 2

def percentile_length(sentences, tokenizer, percentile):
    ''' [["hello how are you ?"], ...] => percentile length of subword encoded strings '''
    encoded_sentences = [tokenizer.encode(s) for s in sentences]
    lengths = sorted([len(e) for e in encoded_sentences])
    return lengths[int(((len(lengths)) * percentile) // 100)]

def create_subword_tokenizer():
    ''' Create subword tokenizer from PERSONA-CHAT dataset and 97.5th percentile of lengths '''
    train_personas, train_data = pre.load_dataset(pre.TRAIN_FN)
    val_personas, val_data = pre.load_dataset(pre.VALID_FN)
    
    all_text = [p for p in train_personas + val_personas] + [msg for msg in train_data[:, 0] + val_data[:, 0]] + [rply for rply in train_data[:, 1] + val_data[:, 1]]
    
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        all_text, target_vocab_size=8192)
    
    # get the percentile length of subwords of encoded examples 
    personas = np.concatenate([train_personas, val_personas])
    persona_length = percentile_length(personas, tokenizer, 99.5)
    
    messages = np.concatenate([train_data[:, 0], val_data[:, 0]])
    message_length = percentile_length(messages, tokenizer, 97.5)
    
    replys = np.concatenate([train_data[:, 1], val_data[:, 1]])
    reply_length = percentile_length(replys)
    
    # training examples with input sequence > persona_length + message_length
    # or reply > reply_length, should be dropped
    
    return tokenizer, persona_length, message_length, reply_length
    
    
    
    

def encode_sentence(persona, message, tokenizer, persona_length, msg_length):
    ''' Tokenize a persona + message sentence for the encoder '''
    
    if persona == "":
        encoded = tokenizer.encode(message)
        # truncate the message
        del encoded[:msg_length - 2]
        
        encoded = [tokenizer.vocab_size + SOS] + encoded + [tokenizer.vocab_size + EOS]
    else:
        persona = tokenizer.encode(persona)
        del persona[:persona_length - 1]
        
        message = tokenizer.encode(message)
        del message[:msg_length - 1]
        
        encoded = [tokenizer.vocab_size + SOS] +  persona + [tokenizer.vocab_size + SEP] + message + [tokenizer.vocab_size + EOS]
    
    # pad by persona_length + msg_length
    encoded = encoded + [0 for i in range(persona_length + msg_length - len(encoded))]
    return encoded


def get_angles(pos, i, d_model):
    angle_freq = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_freq

def positional_encoding(pos, d_model):
    ''' Get positional embeddings for words at pos 1,...,pos ''' 
    angle_radians = get_angles(np.arange(pos)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    
    # Sine at even indices in the array, 2i
    angle_radians[:, 0::2] = np.sin(angle_radians[:, 0::2])
    
    # Cosine at odd indices in the array, 2i+1
    angle_radians[:, 1::2] = np.cos(angle_radians[:, 1::2])
      
    pos_encoding = angle_radians[np.newaxis, ...]
      
    return tf.cast(pos_encoding, dtype=tf.float32)

def create_look_ahead_mask(size):
    ''' Ensure the decoder can only se'''
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    # => (reply_length, reply_length)
    return mask  

def create_padding_mask(seq):
    ''' Return tensor of same shape, with 1 where zero padding value is present, else 0 '''
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    # add extra dimensions to add the padding to the attention logits
    # => (batch_size, 1, 1, seq_len)
    return seq[:, tf.newaxis, tf.newaxis, :]  

def train_transformer(BATCH_SIZE):
    tokenizer = create_subword_tokenizer()
    sample_str = "optimus prime goodness"
    assert tokenizer.decode(tokenizer.encode(sample_str)) == sample_str
    
    BUFFER_SIZE = 20000
    
     # ------ Pretrain on Movie dataset ------ #
    movie_epochs = 15
    movie_conversations = pre.load_movie_dataset(pre.MOVIE_FN)
    
    encoder_input  = movie_conversations[:, 0]
    decoder_target = movie_conversations[:, 1]
    movie_conversations = None
    
    raw = encoder_input[:20]
    
    # integer encode sequences
    x = [encode_sentence("", msg, tokenizer, PERSONA_LENGTH, MSG_LENGTH) for msg in encoder_input]
    y = [encode_sentence("", reply, tokenizer, 0, REPLY_LENGTH) for reply in decoder_target]

    segment_input  = np.array([pre.generate_segment_array(msg, PERSONA_LENGTH + MSG_LENGTH, True) for msg in encoder_input])
    encoder_input  = np.array(x)
    decoder_target = np.array(y)
    
    
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (encoder_input, segment_input, decoder_target)).cache().shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    
    
    
    
    

train_transformer(64)

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