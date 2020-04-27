# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import text_preprocessing as pre
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Embedding
from time import time

# tokens
SOS = 0
SEP = 1
EOS = 2

PSN = 1
MSG = 2

# hyperparameters
D_MODEL = 1024
NUM_LAYERS = 6 
NUM_HEADS = 16
D_FF = 2048
DROPOUT = 0.1
USE_SEG_EMBEDDING = True

# globals
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
transformer = None
checkpoint = None
checkpoint_manage = None
optimizer = None


class MultiHeadAttention(tf.keras.layers.Layer):
    ''' Mult-Head Self Attention https://arxiv.org/pdf/1706.03762.pdf '''
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        if d_model % num_heads != 0:
            raise ValueError(
                "d_model {} must be divisible by num_heads {}".format(d_model, num_heads))
        
        # d_k, d_q == d_v, for a vectorised implementation
        self.d_k = d_model // num_heads
        
        # weights to tranform input into query, key and value matricies
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        
        # transforms Concat(h1,...,h_num_heads) into seq_length X d_model dimensional matrix
        self.linear = Dense(d_model)
        
    def split_heads(self, inputs):
        '''
        Split matrix multiple with input from call into inputs for each head
        inputs => [x, batch_size]
        converts x shape from (batch_size, seq_length, d_model)
                           => (batch_size, num_heads, seq_length, d_k)
        '''
        x, batch_size = inputs
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        ''' 
        Vectorised implementation of Multi Head Self Attention 
        inputs  => [Query, Key, Value, mask]
        outputs => [Concat(h1,...., h_n)W_o, attention_weights]
        '''
        q, k, v, mask = inputs
        batch_size = tf.shape(q)[0]
        
        # => (batch_size, seq_length, d_model)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        # => (batch_size, num_heads, seq_length, d_k)
        q = self.split_heads([q, batch_size])
        k = self.split_heads([k, batch_size])
        v = self.split_heads([v, batch_size])
        
        # attn => (batch_size, num_heads, seq_length, d_k)
        # attn_weights => (batch_size, num_heads, seq_length, seq_length)
        attn, attn_weights = dot_product_attention(q, k, v, mask)
        
        # => (batch_size, seq_length, num_heads, d_k)
        attn = tf.transpose(attn, perm=[0, 2, 1, 3])
        
        # => (batch_size, seq_length, d_model)
        concat_heads = tf.reshape(attn, (batch_size, -1, self.d_model))
        
        output = self.linear(concat_heads)
        
        return output, attn_weights
    
    
class EncoderLayer(tf.keras.layers.Layer):
    ''' Single Transformer Encoder Layer '''
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.attn_layer = MultiHeadAttention(d_model, num_heads)
        
        self.first_layernorm = LayerNormalization(epsilon=1e-6)
        self.first_dropout   = Dropout(dropout)
        
        self.mlp = MLP(d_model, d_ff)
        
        self.second_layernorm = LayerNormalization(epsilon=1e-6)
        self.second_dropout   = Dropout(dropout)
    
    def call(self, inputs):
        ''' 
        Multi Head Self Attention => layernorm => MLP => layernorm 
        inputs => [x, is_training, mask]
        output => layer_output
        mask is for padded values in x
        '''
        x, is_training, mask = inputs
        
        # => (batch_size, seq_length, d_model)
        attn_out, _ = self.attn_layer([x, x, x, mask])
        attn_out    = self.first_dropout(attn_out, training=is_training)
        attn_out    = self.first_layernorm(x + attn_out) # residual connection
        
        mlp_out = self.mlp(attn_out)
        mlp_out = self.second_dropout(mlp_out, training=is_training)
        layer_output  = self.second_layernorm(attn_out + mlp_out) # residual connection
        
        return layer_output


class DecoderLayer(tf.keras.layers.Layer):
    ''' Single Transformer Decoder Layer '''
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn            = MultiHeadAttention(d_model, num_heads)
        self.encoder_decoder_attn = MultiHeadAttention(d_model, num_heads)
        
        self.first_layernorm = LayerNormalization(epsilon=1e-6)
        self.first_dropout   = Dropout(dropout)
        
        self.second_layernorm = LayerNormalization(epsilon=1e-6)
        self.second_dropout   = Dropout(dropout)

        self.third_layernorm  = LayerNormalization(epsilon=1e-6)
        self.third_dropout    = Dropout(dropout)
                
        self.mlp = MLP(d_model, d_ff)
    
    def call(self, inputs):
        '''
        Masked Multi Head Self Attention => layernorm 
        => Multi Head Encoder-Decoder Attention => layernorm 
        => MLP => layernorm
        
        inputs  => [x, encoder_output, is_training, look_ahead_mask, padding_mask]
        outputs => layer_output, self_attn_weights, ed_attn_weights
        '''
        x, encoder_output, is_training, look_ahead_mask, padding_mask = inputs
        
        # encoder_output shape => (batch_size, in_seq_length, d_model)
        
        # => (batch_size, out_seq_length, d_model)
        self_attn_out, self_attn_weights = self.self_attn([x, x, x, look_ahead_mask])
        self_attn_out = self.first_dropout(self_attn_out, training=is_training)
        out1 = self.first_layernorm(self_attn_out + x)
        
        # => (batch_size, out_seq_length, d_model)
        ed_attn_out, ed_attn_weights = self.encoder_decoder_attn([out1,
                                                                  encoder_output, 
                                                                  encoder_output,
                                                                 padding_mask])
        ed_attn_out = self.second_dropout(ed_attn_out, training=is_training)
        out2 = self.second_layernorm(ed_attn_out + out1)
        
        mlp_out = self.mlp(out2)
        mlp_out = self.third_dropout(mlp_out, training=is_training)
        
        # => (batch_size, out_seq_length, d_model)
        layer_output = self.third_layernorm(mlp_out + out2)
        
        return layer_output, self_attn_weights, ed_attn_weights


class Encoder(tf.keras.layers.Layer):
    ''' Stacks N Transformer Encoder Layers '''
    def __init__(self, num_layers, d_model, num_heads, d_ff, embedding,
                 max_pos, use_segment_embedding, dropout=0.1):
        super(Encoder, self).__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        
        
        self.embedding = embedding
        self.pos_encoding = positional_encoding(max_pos, d_model)
        # segment embedding are used so that this model can
        # better distinguish between persona and message segments
        # segment embeddings should be padded
        if use_segment_embedding:
            # segment_embedding_dim must be the same as output_dim of word embedding
            self.segment_embedding = Embedding(3, d_model, trainable=True, name="segment_embedding")
        else:
            # use a zero segment embedding which will have no effect on the model
            self.segment_embedding = Embedding(3, d_model, 
                                               weights=[np.zeros((3, d_model))], 
                                               trainable=False,
                                               name="segment_embedding")
        
        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout)
                               for _ in range(num_layers)]
        
        self.dropout = Dropout(dropout)
    
    def call(self, inputs):
        '''
        Performs N layers of Multi-Head Self Attention
        inputs => [x, segment, is_training, mask]
        output => output
        x shape       => (batch_size, in_seq_length)
        segment shape => (batch_size, in_seq_length)
        output shape  => (batch_size, in_seq_length, d_model)
        '''
        x, segment, is_training, mask = inputs
        in_seq_length = tf.shape(x)[1]
        
        # word embedding + positional embedding + segment embedding
        x = self.embedding(x)
        # multiply embedding by sqrt of dimension https://arxiv.org/pdf/1608.05859.pdf
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = tf.add(x, self.pos_encoding[:, in_seq_length, :])
        x = tf.add(x, self.segment_embedding(segment))
        
        x = self.dropout(x, training=is_training)
        
        for i in range(self.num_layers):
            x = self.encoder_layers[i]([x, is_training, mask])
        
        return x


class Decoder(tf.keras.layers.Layer):
    ''' Stacks N Transformer Decoder Layers '''
    def __init__(self, num_layers, d_model, num_heads, d_ff,
                 embedding, max_pos, dropout=0.1):
        super(Decoder, self).__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        
        # no segment embedding needed for Decoder
        self.embedding = embedding
        self.pos_encoding = positional_encoding(max_pos, d_model)
        
        self.decoder_layers = [DecoderLayer(d_model, num_heads, d_ff, dropout)
                               for _ in range(num_layers)]
        
        self.dropout = Dropout(dropout)

    def call(self, inputs):
        ''' 
        inputs  => [x, encoder_out, is_training, look_ahead_mask, padding_mask]
        outputs => output, attn_weights
        x shape      => (batch_size, out_seq_length)
        output shape => (batch_size, out_seq_length, d_model)
        
        attn_weights is dictionary 
        {'self_attn0' : self attention weights, 
         'ed_attn0' : encoder decoder attention weights, ... layer N}
        '''
        x, encoder_out, is_training, look_ahead_mask, padding_mask = inputs
        
        out_seq_length = tf.shape(x)[1]
        attn_weights = {}
        
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = tf.add(x, self.pos_encoding[:, :out_seq_length, :])
        
        x = self.dropout(x, training=is_training)
        
        for i in range(self.num_layers):
            x, self_attn, ed_attn = self.decoder_layers[i]([x, encoder_out,
                                                            is_training, look_ahead_mask,
                                                            padding_mask])
            attn_weights['self_attn{}'.format(i)] = self_attn
            attn_weights['ed_attn{}'.format(i)]   = ed_attn
        
        return x, attn_weights
    
 
class Transformer(tf.keras.Model):
    ''' Q&A Transformer https://arxiv.org/pdf/1706.03762.pdf '''
    def __init__(self, d_model, num_layers, num_heads, d_ff, 
                 input_max_pos, output_max_pos, vocab_size,
                 use_segment_embedding, dropout=0.1):
        super(Transformer, self).__init__()
        
        tied_embedding = Embedding(vocab_size, d_model)
        
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, 
                               tied_embedding, input_max_pos, use_segment_embedding,
                               dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, 
                               tied_embedding, output_max_pos, dropout)
        
        self.output_layer = Dense(vocab_size)
    
    @tf.function
    def call(self, inputs):
        ''' 
        One Transformer Forward Pass
        inputs  => [msg, segment, reply, is_training, enc_mask, look_ahead_mask, dec_mask]
        outputs => output, attn_dict
        
        reply is sentence generated so far
        msg shape     => (batch_size, in_seq_length)
        segment shape => (batch_size, in_seq_length)
        reply shape   => (batch_size, out_seq_length)
        output shape  => (batch_size, out_seq_length, vocab_size)
        '''
        msg, segment, reply, is_training, enc_mask, look_ahead_mask, dec_mask = inputs
        
        encoder_output = self.encoder([msg, segment, is_training, enc_mask])
        
        decoder_output, attn_dict = self.decoder([reply, encoder_output, 
                                                  is_training, look_ahead_mask, 
                                                  dec_mask])
        output = self.output_layer(decoder_output)
        
        return output, attn_dict

class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    ''' Learning rate scheduler for Adam https://arxiv.org/pdf/1706.03762.pdf '''
    def __init__(self, d_model, warmup_steps=4000):
        super(TransformerSchedule, self).__init__()
        self.d_model = tf.cast(d_model, dtype=tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        return (tf.math.rsqrt(self.d_model) * tf.math.minimum(
            tf.math.rsqrt(step), step * (self.warmup_steps ** -1.5)))

def loss_function(label, pred):
    mask = tf.math.logical_not(tf.math.equal(label, 0))
    loss = loss_object(label, pred)
    
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


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
    q_kt = tf.matmul(q, k, transpose_b=True)
    
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

def MLP(d_model, hidden_units):
    ''' Feed Forward Transformer Layer '''
    return tf.keras.Sequential([Dense(hidden_units),
                                Dense(d_model)])

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

def create_masks(msg_seq, reply_seq):
    # mask for encoder self attention so it
    # doesn't consider pad values
    encoder_padding = create_padding_mask(msg_seq)
    
    # mask for encoder output, used in encoder_decoder attention
    decoder_padding = create_padding_mask(msg_seq)
    
    # mask for decoder self attention so it
    # doesn't consider pad values for future predictions
    look_ahead_mask = create_look_ahead_mask(tf.shape(reply_seq)[1])
    output_mask     = create_padding_mask(reply_seq)
    look_ahead_mask = tf.maximum(output_mask, look_ahead_mask)
    
    return encoder_padding, look_ahead_mask, decoder_padding

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

@tf.function(input_signature=[
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
def train_step(msg_batch, seg_batch, reply_batch):
    # use teacher forcing by passing Transformer the ground truth
    # as as the reply input
    reply_input_batch  = reply_batch[:, :-1]
    reply_target_batch = reply_batch[:, 1:]
    
    encoder_mask, look_ahead_mask, decoder_mask = create_masks(msg_batch, 
                                                               reply_input_batch)
    
    with tf.GradientTape() as tape:
        generated_sentences, _ = transformer([msg_batch, seg_batch, 
                                             reply_input_batch, True,
                                             encoder_mask, look_ahead_mask,
                                             decoder_mask])
        loss = loss_function(reply_target_batch, generated_sentences)
        
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    
    train_loss(loss)
    train_accuracy(loss)


def train(dataset, val_dataset, EPOCHS, MIN_EPOCHS, PATIENCE):
    for epoch in range(EPOCHS):
        elapsed = time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        
        for(batch, (msg, seg, reply)) in enumerate(dataset):
            train_step(msg, seg, reply)
            
            if pre.VERBOSE == 1:
                print("Epoch %d: Batch %d: Loss %f, Accuracy %f" %
                      (epoch + 1, batch, train_loss.result(), train_accuracy.result()))
            
        if (epoch + 1) % 10 == 0:
            # checkpoint model every 10 epochs
            save_path = checkpoint_manager.save()
            print("Epoch %d: Checkpoint saved at %s" % (epoch + 1, save_path))
        
        print("Epoch %d --- %d sec: Loss %f, Accuracy %f" % (epoch + 1, 
                                                             time() - elapsed,
                                                             train_loss.result(),
                                                             train_accuracy.result()))
def train_transformer(BATCH_SIZE):
    global transformer
    global checkpoint
    global checkpoint_manager
    global optimizer
    
    tokenizer, in_seq_length, out_seq_length = create_subword_tokenizer()
    # +3 for SOS, SEP, EOS tokens
    vocab_size = tokenizer.vocab_size + 3
    sample_str = "optimus prime goodness"
    assert tokenizer.decode(tokenizer.encode(sample_str)) == sample_str
    
    print("Input Length: %d " % in_seq_length)
    print("Output Length: %d" % out_seq_length)
    print("Vocab Size: %d" % vocab_size)
    
    BUFFER_SIZE = 15000
    
    # ------ Pretrain on Movie dataset ------ #
    movie_epochs = 15
    movie_conversations = pre.load_movie_dataset(pre.MOVIE_FN)
    movie_conversations = movie_conversations[:4, :]######
    pre.VERBOSE = 1
    
    raw = movie_conversations[:20, 0]
    
    # integer encode sequences
    encoder_input, decoder_target = encode_training_examples(None, movie_conversations, tokenizer, in_seq_length, out_seq_length)
    segment_input  = np.array([generate_segment_list(encoded_msg, in_seq_length, True) for encoded_msg in encoder_input])
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (encoder_input, segment_input, decoder_target)).cache().shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    
    movie_conversations = None
    encoder_input, decoder_target, segment_input = None, None, None
    
    transformer = Transformer(D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF,
                              vocab_size, vocab_size, vocab_size, 
                              USE_SEG_EMBEDDING, DROPOUT)
    
    optimizer = tf.keras.optimizers.Adam(TransformerSchedule(D_MODEL), 
                                         beta_1=0.9, beta_2=0.98, 
                                         epsilon=1e-9)
    
    # use checkpoints to retrain model from the last checkpoint
    # in-case more training is needed
    checkpoint = tf.train.Checkpoint(transformer=transformer,
                                     optimizer=optimizer)
    
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, 
                                                    pre.TRANSFORMER_CHECKPOINT_PATH,
                                                    max_to_keep=3)
    
    train(dataset, None, 10, 10, 10)
    
    
    
    

train_transformer(2)

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

if __name__ == "__main__":
    # do some tests to ensure Encoder and Decoder have correct dimensionality
    
    # ------ Encoder Layer ------ #
    encoder_layer = EncoderLayer(1024, 16, 2048)
    encoder_in_shape = (128, 105, 1024)

    encoder_out = encoder_layer([
        tf.random.uniform(encoder_in_shape), False, None])
    
    assert encoder_out.shape == encoder_in_shape
    # ------ ------ #
        
    # ------ Decoder Layer ------ #
    decoder_layer = DecoderLayer(1024, 16, 2048)
    decoder_in_shape = (128, 21, 1024)
    decoder_out, self_attn_weights, de_attn_weights = decoder_layer([
        tf.random.uniform(decoder_in_shape), encoder_out, 
        False, None, None])
    
    assert decoder_out.shape == decoder_in_shape
    
    assert self_attn_weights.shape == (decoder_in_shape[0], 
                                 16,
                                 decoder_in_shape[1], 
                                 decoder_in_shape[1])
    
    assert de_attn_weights.shape == (decoder_in_shape[0],
                               16,
                               decoder_in_shape[1], 
                               encoder_in_shape[1])
    # ------ ------ #
    
    # ------ Encoder ------ #
    encoder_embedding = Embedding(8136, 512)
    encoder = Encoder(6, 512, 8, 2048, encoder_embedding, 10000, True)
    encoder_in_shape = (32, 102)
    
    encoder_input = tf.random.uniform(encoder_in_shape, dtype=tf.int64, minval=0, maxval=8000)
    segment_input = tf.random.uniform(encoder_in_shape, dtype=tf.int64, minval=0, maxval=2)
    encoder_out = encoder([encoder_input, segment_input, False, None])
    
    assert encoder_out.shape == (encoder_in_shape[0], encoder_in_shape[1], 512)
    # ------ ------ #
    
    # ------ Decoder ------ #
    decoder = Decoder(6, 512, 8, 2048, encoder_embedding, 5000)
    decoder_in_shape = (32, 25)
    decoder_input = tf.random.uniform(decoder_in_shape, dtype=tf.int64, minval=0, maxval=8000)

    output, attn_dict = decoder([decoder_input, encoder_out, False, None, None])
    
    assert output.shape == (decoder_in_shape[0], decoder_in_shape[1], 512)
    assert attn_dict['ed_attn5'].shape == (decoder_in_shape[0], 8, decoder_in_shape[1], 
                                           encoder_in_shape[1])
    # ------ ------ #
    
    # ------ Transformer ------ #
    tied_embedding = Embedding(8136, 1024)
    optimus_prime = Transformer(1024, 6, 16, 2048, 10000, 5000, 
                                     8136, True)

    msg   = tf.random.uniform((128, 92), dtype=tf.int64, minval=0, maxval=8135)
    seg   = tf.random.uniform((128, 92), dtype=tf.int64, minval=0, maxval=2)
    reply = tf.random.uniform((128, 25), dtype=tf.int64, minval=0, maxval=8135)
    
    out, _ = optimus_prime([msg, seg, reply, False, None, None, None])
    
    assert out.shape == (128, 25, 8136)
    # ------ ------ #
    