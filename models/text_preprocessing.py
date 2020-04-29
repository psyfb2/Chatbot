# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""
import numpy as np
import string
import pickle
import re
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir))

TRAIN_FN            = os.path.join(PROJ_PATH, 'data', 'train_self_original_no_cands.txt')
VALID_FN            = os.path.join(PROJ_PATH, 'data', 'valid_self_original_no_cands.txt')
MOVIE_FN            = os.path.join(PROJ_PATH, 'data', 'movie_lines.txt')
DAILYDIALOGUE_FN    = os.path.join(PROJ_PATH, 'data', 'dialogues_text.txt')
VOCAB_FN            = os.path.join(PROJ_PATH, 'data', 'vocab.txt')
GLOVE_FN            = os.path.join(PROJ_PATH, 'data', 'glove.6B.300d.txt')

MULTIENC_ENCODER_MODEL_FN    = os.path.join(PROJ_PATH, 'saved_models', 'multiple_encoders_model')
MULTIENC_DECODER_MODEL_FN    = os.path.join(PROJ_PATH, 'saved_models', 'multiple_decoder_model')

MULTIENC_ENCODER_DEEP_MODEL_FN    = os.path.join(PROJ_PATH, 'saved_models', 'multiple_encoders_deep_model')
MULTIENC_DECODER_DEP_MODEL_FN    = os.path.join(PROJ_PATH, 'saved_models', 'multiple_decoder_deep_model')

SEQ2SEQ_ENCODER_MODEL_FN    = os.path.join(PROJ_PATH, 'saved_models', 'seq2seq_encoder_model')
SEQ2SEQ_DECODER_MODEL_FN    = os.path.join(PROJ_PATH, 'saved_models', 'seq2seq_decoder_model')

SEQ2SEQ_ENCODER_DEEP_MODEL_FN    = os.path.join(PROJ_PATH, 'saved_models', 'seq2seq_encoder_deep_model')
SEQ2SEQ_DECODER_DEEP_MODEL_FN    = os.path.join(PROJ_PATH, 'saved_models', 'seq2seq_decoder_deep_model')

TRANSFORMER_CHECKPOINT_PATH = os.path.join(PROJ_PATH, 'saved_models', 'transformer_checkpoints')
TRANSFORMER_MODEL_FN = os.path.join(PROJ_PATH, 'saved_models', 'transformer')

AUTOENC_MODEL_IMAGE_FN      = os.path.join(PROJ_PATH, 'saved_models', 'autoenc.png')
AUTOENC_MODEL_FN            = os.path.join(PROJ_PATH, 'saved_models', 'autoenc.h5')

VERBOSE = 0


# punctuation which will not be removed from training/test data
ALLOWED_CHARS = ['.', ',', '_', '?']
START_SEQ_TOKEN = "startseqq"
END_SEQ_TOKEN   = "stopseqq"
SEP_SEQ_TOKEN   = "sepseqq"
SEGMENT_PERSONA_INDEX   = 1
SEGMENT_MESSAGE_INDEX   = 2

def remove_allowed_chars(punc_str):
    for char in ALLOWED_CHARS:
        punc_str = punc_str.replace(char, "")
    return punc_str

'''
Data is in the following format
1 persona 
2 persona 
3 persona 
4 persona
5 persona (sometimes)
5 human \t bot
6 human \t bot
7 human \t bot
8 human \t bot
9 human \t bot
1 persona
2 persona
...

human to bot utterances can range in the number of replies per persona.
This is the no_cands version, which means candidate replies which are not
the ground truth are included within possible replies of the bot within the dataset. 
This is useful when training for multi-task problem to produce utterance sequence 
but also classify which reply from the candidates is the ground truth.

'''

''' Load GloVe embedding containing only those words within the tokenizer vocab '''
def load_glove_embedding(tokenizer, glove_filename):
    # load glove as dictionary {word : embedding, ...} for only the words in
    # the tokenizer vocabulary
    glove_file = open(glove_filename, mode="rt", encoding="utf-8")
    word_dict = dict()
    for line in glove_file:
        values = line.split()
        word = values[0]
        word_dict[word] = np.asarray(values[1:], dtype="float32")
    glove_file.close()
    
    # create an embedding matrix which is indexed by words in training docs
    vocab_size= len(tokenizer.word_index) + 1
    dimensions = 50
    if "100" in glove_filename:
        dimensions = 100
    elif "200" in glove_filename:
        dimensions = 200
    else:
        dimensions = 300
    
    embedding_matrix = np.zeros((vocab_size, dimensions))
    for word, unique_index in tokenizer.word_index.items():
        # get the embedding vector from the dictionary created above
        vec = word_dict.get(word)
        if vec is not None:
            embedding_matrix[unique_index] = vec
    
    return embedding_matrix

''' Returns a keras tokenizer fitted on the given text '''
def fit_tokenizer(lines, oov_token=True):
    if not oov_token:
         tokenizer = Tokenizer(filters=remove_allowed_chars('!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'))
    else:
        tokenizer = Tokenizer(filters=remove_allowed_chars('!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'), oov_token="UNK")
    tokenizer.fit_on_texts(lines)
    return tokenizer

''' Given a tokenizer, pad length, numpy array of cleaned lines
    returns numpy array of padded and integer encoded sequences
    e.g. ["a cleaned sentence", ...] => [[5, 20, 30], ...] '''
def encode_sequences(tokenizer, length, lines):
    X = tokenizer.texts_to_sequences(lines)
    X = pad_sequences(X, maxlen=length, padding='post')
    return X

''' Each sequeunce in sequences will produce a matrix of one hot encoded words.
    Use this to perform catagorical crossentropy on the target sequence '''
def encode_output(sequences, vocab_size):
    ylist = []
    for sequence in sequences:
        # each target sequence produces a matrix where each row is a word one hot encoded
        # and there are vocab_size columns
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y

''' Reverse mapping of tokenizer to get a word from a unique index '''
def index_to_word(integer, tokenizer):
    for w, i in tokenizer.word_index.items():
        if i == integer:
            return w
    return None

def truncate(s, length):
    split = s.split(' ')
    if len(split) <= length:
        return s
    else:
        return ' '.join(split[:length])

def generate_segment_array(sentence, pad_length, no_persona=False):
    ''' Generates a list of segment indicies based on the SEP_SEQ_TOKEN found in sentence '''
    sep_seq_found = no_persona
    segment = []
    c = 0
    
    for word in sentence.split(' '):
        if c >= pad_length:
            break
        
        if sep_seq_found:
            # message segment
            segment.append(SEGMENT_MESSAGE_INDEX)
        else:
            # persona segment
            segment.append(SEGMENT_PERSONA_INDEX)
        if word == SEP_SEQ_TOKEN:
            sep_seq_found = True
        c += 1
        
    # pad the segment array
    for i in range(pad_length - len(segment)):
        segment.append(0)
            
    return segment

def remove_contractions(sentence):
    sentence = re.sub(r"won\'t", "will not", sentence)
    sentence = re.sub(r"can\'t", "can not", sentence)

    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    return sentence

''' Returns the index of a single persona within a list of already stored personas 
    ValueError exception if the single_pers cannot be found in personas '''
def get_persona_index(personas, single_pers):
    # each persona has 5 sentences however in the dataset most times the fifth 
    # sentence is excluded for whatever reason. Also the same personas
    # have the same sentences but in different order and sometimes with different
    # punctuation and spelling
    
    #if single_pers == a:
        #breakpoint()
    
    # first number character for comparison
    single_pers_re = [sentence[1:] for sentence in single_pers]
    
    shared_sentences = len(single_pers) - 1
    
    for p_index, persona in enumerate(personas):
        persona_re = [sentence[1:] for sentence in persona]
        # different personas don't share the same 4 sentences
        count = 0
        for sentence in single_pers_re:
            if sentence in persona_re:
                count += 1
                if count >= shared_sentences:
                    # these are the same personas, save one with 5 sentences and return index
                    if(len(single_pers) == 5 and len(persona) != 5):
                        personas[p_index] = single_pers
                    return p_index
    raise ValueError("Could not find persona {}".format(single_pers))
                    
''' Returns cleaned
    persona array [["persona_line_1, ... persona_line_4"], ...]
    array of message, reply, persona index triplets e.g. [["hi how are you", "hi good", 5], ...]
'''
def load_dataset(filename, verbose=0):
    personas = [] 
    conversations = []
    
    with open(filename, 'rt') as lines:
        single_pers = []
        # read text file line by line
        for line in lines:
            line = line.strip()
            if "your persona:" in line and line[0] in ['1', '2', '3', '4', '5']:
                single_pers.append(line)
            else:
                # add persona to list of personas
                if len(single_pers) != 0:
                    personas.append(single_pers)
                    p_index = len(personas) - 1
                    single_pers = []
                # line is a message and reply seperated by tab
                # which is ascociated with the last read persona
                pair = line.split('\t')
                conversations.append([pair[0], pair[1], p_index])
                
    # clean the conversation and personas
    triples = clean_triples(conversations)
    for i in range(len(personas)):
        for j in range(len(personas[i])):
            personas[i][j] = clean_line(remove_first_num(personas[i][j]).replace("your persona", "").strip())
        personas[i] = ' '.join(personas[i])
    personas = np.array(personas)
    
    # check that cleaned text is as intended
    if verbose != 0:
        for i in range(100):
            assert len(triples[i]) == 3
            print('[%s]\n[%s] => [%s]' % (personas[int(triples[i, 2])], triples[i, 0], triples[i, 1]))
            print("\n")
    
    return personas, triples

''' Returns
    array of message, reply pairs e.g. [["give me the gun", "no i can't do that"], ...]
'''
def load_movie_dataset(filename=MOVIE_FN, verbose=0):
    conversations = []
    msg_reply = []
    
    with open(filename, 'r', encoding='latin-1') as lines:
        for line in lines:
            line = line.strip()
            line = line.split(' +++$+++ ')[-1]         
            
            # remove some specific tags in the dataset
            tags = ['<u>', '</u>', '<i>', '</i>', '<b>', '</b>']
            line = " ".join([w for w in line.split(" ") if w.lower() not in tags])

            line = clean_line(line)
            
            if len(msg_reply) >= 2:
                conversations.append(msg_reply)
                msg_reply = [msg_reply[1]]
            
            msg_reply.append(line)
    
    if verbose != 0:
        for i in range(100):
            assert len(conversations[i]) == 2
            print('[%s] => [%s]' % (conversations[i][0], conversations[i][1]))
            print("\n")
            
    return np.array(conversations)   

''' Returns
    array of message, reply pairs e.g. [['Can you take out the bins ?', 'Yes I can .'], ...]
'''
def load_dailydialogue_dataset(filename=DAILYDIALOGUE_FN, verbose=0):
    conversations = []
    
    with open(filename, 'r', encoding='utf-8') as lines:
        for line in lines:
            # each line is a conversation split with __eou__ for each utterence
            
            # remove and fix spacing around ’
            line = line.replace(" ’ ", "'")
            
            line = clean_line(line)
            
            utterences = line.split("__eou__")[:-1]
            for i in range(len(utterences) - 1):
                conversations.append([utterences[i].strip(), utterences[i + 1].strip()])
    
    if verbose != 0:
        for i in range(100):
            assert len(conversations[i]) == 2
            print('[%s] => [%s]' % (conversations[i][0], conversations[i][1]))
            print("\n")
            
    return np.array(conversations)
            

''' Build a vocab file from the PERSONA-CHAT dataset for tokenizer '''
def build_vocab_file(verbose=0):
    personas, triples = load_dataset(TRAIN_FN)
    personas2, triples2 = load_dataset(VALID_FN)
    
    # texts from PERSONA CHAT
    texts =  np.concatenate([personas, personas2, 
                        triples[:, 0], triples[:, 1], triples2[:, 0], 
                        triples2[:, 1]])
    
    
    tokenizer = fit_tokenizer(texts, oov_token=False)
    
    # remove infrequent words
    '''
    low_count_words = [w for w,c in tokenizer.word_counts.items() if c < 2]
    for w in low_count_words:
        del tokenizer.word_index[w]
        del tokenizer.word_docs[w]
        del tokenizer.word_counts[w]
    '''
    
    max_persona_len = percentile_length(np.concatenate([personas, personas2]), 99.8) 
    max_msg_len     = percentile_length(np.concatenate([triples[:, 0], triples2[:, 0]]), 99.5) + 3
    max_reply_len   = percentile_length(np.concatenate([triples[:, 1], triples2[:, 1]]), 99.5) + 3
    
    # write the vocab and max lengths to a file
    with open(VOCAB_FN, 'wt') as f:
        for w in tokenizer.word_index.keys():
            f.write(w + "\n")
        f.write(START_SEQ_TOKEN + "\n")
        f.write(SEP_SEQ_TOKEN + "\n")
        f.write(END_SEQ_TOKEN + "\n")
        f.write("max persona length: %d\n" % max_persona_len)
        f.write("max message length: %d\n" % max_msg_len)
        f.write("max reply length: %d" % max_reply_len)
    
    if verbose != 0:
        print("Max persona length: %d" % max_persona_len)
        print("Max message length: %d" % max_msg_len)
        print("Max reply length: %d" % max_reply_len)
        print("Vocab Lenght: %d" % len(tokenizer.word_index))

''' Returns numpy array of words in vocab,
    persona_length, max message length 
    and max reply length excluding any stop/start tokens
    if the vocab hasn't been built yet, will automatically build it
'''
def get_vocab(rebuild_vocab=False, verbose=0):
    vocab = []
    persona_length = 0
    message_length = 0
    reply_length = 0
    
    # if the vocab file already doesnt exists then build the file
    if not os.path.exists(VOCAB_FN) or rebuild_vocab:
        build_vocab_file(verbose)
    
    with open(VOCAB_FN, 'rt') as lines:
        for line in lines:
            line = line.strip()
            
            if "persona length:" in line:
                persona_length = int(line.split(" ")[-1])
                
            elif "message length:" in line:
                message_length = int(line.split(" ")[-1])
                
            elif "reply length:" in line:
                reply_length = int(line.split(" ")[-1])
                
            else:
                vocab.append(line)
     
    if verbose != 0:
        print("Max persona length: %d" % persona_length)
        print("Max message length: %d" % message_length)
        print("Max reply length: %d" % reply_length)
        print(vocab[:50])
            
    return np.array(vocab), persona_length, message_length, reply_length
            
    
''' Takes [["message text", "reply text", pindex], ...] 
    and returns the cleaned version in numpy array '''
def clean_triples(msg_reply):
    cleaned = []
    
    for triple in msg_reply:
        clean_msg_reply = []
        
        # triple[0] is message starting with a number
        # triple[1] is the reply to learn 
        # triple[2] the persona index
        triple[0] = clean_line(remove_first_num(triple[0]))
        clean_msg_reply.append(triple[0])
        triple[1] = clean_line(triple[1])
        clean_msg_reply.append(triple[1])
        clean_msg_reply.append(triple[2])
        
        cleaned.append(clean_msg_reply)
        
    return np.array(cleaned)
        

''' Clean a single line of text by
    removing non-printable characters
    make lower case
    removing punctuation apart from full stop and comma 
    removes contractions e.g. i've -> i have'''
def clean_line(line):
    # only include printable characters and remove punctuation
    # apart from full stop and comma characters
    punc = remove_allowed_chars(string.punctuation)

    re_punc = re.compile('[%s]' % re.escape(punc))
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    
    line = remove_contractions(line)
    line = line.split()
    # make lower case and remove any start or stop tokens (this should never be the case anyway)
    line = [word.lower() for word in line 
            if word not in [START_SEQ_TOKEN, SEP_SEQ_TOKEN, END_SEQ_TOKEN]]
    # remove punctuation
    line = [re_punc.sub('', w) for w in line]
    # remove non-printable chars
    line = [re_print.sub('', w) for w in line]
    
    line = ' '.join(line)
    
    # allowed punctuation needs to be spaced, e.g. hi, how are you. => hi , how are you .
    # dont apply this to '_' because of __silence__ token
    chars_to_space = ALLOWED_CHARS.copy()
    chars_to_space.remove('_')
    line = re.sub('([%s])' % "".join(chars_to_space), r' \1 ', line)
    # remove multiple consecutive  spaces
    line = re.sub('\s{2,}', ' ', line)
    
    return line.strip()
    
''' Remove the first number from a string '''
def remove_first_num(strr):
    for i in range(len(strr)):
        if strr[i] in string.digits:
            # found the first number
            count = 1
            while True:
                if i + count >= len(strr) or strr[i + count] not in string.digits:
                    indicies = [x for x in range(i, i + count)]
                    cpy = [strr[j] for j in range(len(strr)) if j not in indicies]
                    return ''.join(cpy)
                count += 1
    # string does not contain any numbers
    return strr

''' Save an object using pickle as the filename '''   
def save_object(obj, filename, verbose=1):
    pickle.dump(obj, open(filename, 'wb'))
    if verbose == 1:
        print("Saved: %s" % filename)

''' Load a pickle object from file '''
def load_object(filename):
    return pickle.load(open(filename, 'rb'))

''' Given a list of cleaned lines ["sentence 1", "sentence 2", ...] 
    returns sentence with max num of words '''
def max_seq_length(lines):
    return max([len(line.split(' ')) for line in lines])        

''' Given a list of cleaned lines ["sentence 1", "sentence 2", ...] 
    returns sentence length at the given percentile between 0 and 99 '''
def percentile_length(lines, percentile):
    lengths = sorted([len(line.split(' ')) for line in lines])
    return lengths[int(((len(lengths)) * percentile) // 100)]

if __name__ == '__main__':
    # test dataset loading works correctly
    #personas, triples = load_dataset(TRAIN_FN, verbose=1)
    #conversations = load_movie_dataset(MOVIE_FN, verbose=1)
    #dialogue_conversations = load_dailydialogue_dataset(DAILYDIALOGUE_FN, verbose=1)
    vocab, persona_length, message_length, reply_length = get_vocab(True, verbose=1)
    
    