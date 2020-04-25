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
    
    personas = np.concatenate([train_personas, val_personas])
    messages = np.concatenate([train_data[:, 0], val_data[:, 0]])
    replys = np.concatenate([train_data[:, 1], val_data[:, 1]])

    all_text = [p for p in personas] + [msg for msg in messages] + [rply for rply in replys]
    
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        all_text, target_vocab_size=8192)
    
    # get the percentile length of subwords of encoded examples 
    in_seq_length = percentile_length(
        [train_personas[int(row[2])] + ' ' + row[0] for row in train_data], 
        tokenizer, 99.75)
    out_seq_length = percentile_length(train_data[:, 1], tokenizer, 99.5)
    
    # training examples with input sequence > in_seq_length
    # or reply > out_seq_length, should be dropped, +3 for SOS, EOS, SOS tokens
    return tokenizer, in_seq_length + 3, out_seq_length + 2

def encode_sentence(persona, message, tokenizer, max_length, drop_example=False):
    ''' Tokenize a sentences and pad/truncate according to max_length '''
    
    if persona == "":
        # encoding a reply
        encoded = [tokenizer.vocab_size + SOS] + tokenizer.encode(message) + [tokenizer.vocab_size + EOS]
    else:        
        encoded = [tokenizer.vocab_size + SOS] +  tokenizer.encode(persona) + [tokenizer.vocab_size + SEP] + tokenizer.encode(message) + [tokenizer.vocab_size + EOS]
    
    # pad, truncate or drop the example
    if len(encoded) > max_length:
        if drop_example:
            return None
        else:
            # truncate the message so it fits in the max length specified
            del encoded[:max_length]
    else:
        encoded = encoded + [0 for i in range(max_length - len(encoded))]
    return encoded

def encode_training_examples(personas, msg_reply_pairs, tokenizer, in_seq_length, out_seq_length):
    ''' Encode all training examples '''
    encoder_input = []
    decoder_target = []
    
    for i in range(len(msg_reply_pairs)):
        if personas == None:
            # no persona data
            msg = encode_sentence("", msg_reply_pairs[i, 0], tokenizer, in_seq_length, True)
            reply = encode_sentence("", msg_reply_pairs[i, 1], tokenizer, out_seq_length, True)
        else:
            # persona data included
            msg = encode_sentence(personas[int(msg_reply_pairs[i, 2])], 
                                  msg_reply_pairs[i, 0], tokenizer, in_seq_length, True)
            reply = encode_sentence("", msg_reply_pairs[i, 1], tokenizer, out_seq_length, True)
            
        # drop this training example if it exceeds maximum length
        if msg != None and reply != None:
            encoder_input.append(msg)
            decoder_target.append(reply)
            
    
    return np.array(encoder_input), np.array(decoder_target)

def generate_segment_list(encoded, pad_length, sep_index, no_persona=False):
    ''' Generates a list of segment indicies based on the seperator index '''
    sep_seq_found = no_persona
    segment = []
    c = 0
    
    for subword_index in encoded:
        if c >= pad_length or subword_index == 0:
            break
        
        if sep_seq_found:
            # message segment
            segment.append(MSG)
        else:
            # persona segment
            segment.append(PSN)
        if subword_index == sep_index:
            sep_seq_found = True
        c += 1
        
    # pad the segment array
    for i in range(pad_length - len(segment)):
        segment.append(0)
            
    return segment


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
    ''' Ensure the decoder can only see words it's generated '''
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    # => (reply_length, reply_length)
    return mask  

def create_padding_mask(seq):
    ''' Return tensor of same shape, with 1 where zero padding value is present, else 0 '''
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    # add extra dimensions to add the padding to the attention logits
    # => (batch_size, 1, 1, seq_len)
    return seq[:, tf.newaxis, tf.newaxis, :]  

def dot_product_attention(q, k, v, mask):
    ''' Calculate dot-product attention for Transformers
        q, k, v should have the same batch size and words per batch
        mask can be look-ahead or padding which effects its dimesnions
        
        Args:
          q: query shape => (..., m, d_k)
          k: key shape => (..., m, d_k)
          v: value shape => (..., m, d_v)
          mask: float tensor with shape broadcastable 
                to (..., m, m), defaults to None
          
        Returns:
          output, attention_weights
    '''
    # => (batch_size, m, m)
    q_kt = tf.matmul(q, k, tranpose_b=True)
    
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_q_kt = q_kt / tf.math.sqrt(dk)
    
    # mask this tensor by multiplying by pad positions by -infinite
    # so they will be approximately 0 in the softmax
    if mask is not None:
        scaled_q_kt += (mask * -1e9)
    
    attn_weights = tf.nn.softmax(scaled_q_kt, axis=-1)
    
    #> (batch_size, m, d_v)
    q_kt_v = tf.matmul(scaled_q_kt, v)
    
    return q_kt_v, attn_weights
    

def train_transformer(BATCH_SIZE):
    tokenizer, in_seq_length, out_seq_length = create_subword_tokenizer()
    sample_str = "optimus prime goodness"
    assert tokenizer.decode(tokenizer.encode(sample_str)) == sample_str
    
    print("Input Length: %d " % in_seq_length)
    print("Output Length: %d" % out_seq_length)
    
    BUFFER_SIZE = 15000
    
    # ------ Pretrain on Movie dataset ------ #
    movie_epochs = 15
    movie_conversations = pre.load_movie_dataset(pre.MOVIE_FN)
    
    raw = movie_conversations[:20, 0]
    #movie_conversations = None
    
    # integer encode sequences
    encoder_input, decoder_target = encode_training_examples(None, movie_conversations, tokenizer, in_seq_length, out_seq_length)
    segment_input  = np.array([generate_segment_list(encoded_msg, in_seq_length, True) for encoded_msg in encoder_input])
    
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