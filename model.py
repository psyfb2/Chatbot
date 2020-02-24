# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""

import numpy as np
from nltk.translate.bleu_score import corpus_bleu
import text_preprocessing as pre
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

''' Returns a keras tokenizer fitted on the given text '''
def fit_tokenizer(lines):
    tokenizer = Tokenizer()
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

def seq2seq_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    
    # encoder
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    
    
    # decoder
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    
    # treat as a multi-class classification problem where need to predict
    # P(yi | x1, x2, ..., xn, y1, y2, ..., yi-1)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.summary()
    plot_model(model, to_file=pre.MODEL_IMAGE_FN, show_shapes=True)
    return model

''' Reverse mapping of tokenizer to get a word from a unique index '''
def word_to_index(integer, tokenizer):
    for w, i in tokenizer.word_index.items():
        if i == integer:
            return w
    return None
    
''' Given source sentence, generate the model inference as a target sentence '''
def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)
    prediction = prediction[0]
    integers = [np.argmax(vec) for vec in prediction]
    
    target = []
    for i in integers:
        word = word_to_index(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)

''' Given a model, matrix of the integer encoded sources, raw dataset, tokenizer
    print bleu scores for all examples in sources '''
def evaluate_by_auto_metrics(model, sources, dataset_not_encoded, tokenizer, verbose=1):
    target_sentences = []
    predicted_sentences = []
    
    for i, source in enumerate(sources):
        source = source.reshape((1, source.shape[0]))
        # source is a row vector of encoded integers, predict it using the model
        pred = predict_sequence(model, tokenizer, source)
        
        # get the real plain text for the source sentence and target sentence
        target, src = dataset_not_encoded[i]
        
        if verbose == 1 and i < 15:
            print('english=[%s], german=[%s], predicted=[%s]' % (src, target, pred))
        target_sentences.append(target)
        predicted_sentences.append(pred)
    
    # calculated the bleu scores
    print("BLEU-1: %f" % corpus_bleu(target_sentences, predicted_sentences, weights=(1.0, 0, 0, 0)))
    print("BLEU-2: %f" % corpus_bleu(target_sentences, predicted_sentences, weights=(0.5, 0.5, 0, 0)))
    print("BLEU-3: %f" % corpus_bleu(target_sentences, predicted_sentences, weights=(0.3, 0.3, 0.3, 0)))
    print("BLEU-4: %f" % corpus_bleu(target_sentences, predicted_sentences, weights=(0.25, 0.25, 0.25, 0.25)))
        

def train():
    # load training and validation set
    train, train_personas = pre.load_object(pre.TRAIN_PKL_FN)
    val, val_personas = pre.load_object(pre.VAL_PKL_FN)
    
    # train is a numpy array containing triples [message, reply, persona_index]
    # personas is an numpy array of strings for the personas
    
    # fit tokenizer over training data
    tokenizer = fit_tokenizer( np.concatenate([train_personas, train[:, 0], train[:, 1]]) )
    vocab_size = len(tokenizer.word_index) + 1
    
    # sequences are persona + msg (prepending the persona to message)
    in_sequences = np.array([train_personas[int(row[2])] + ' ' + row[0] for row in train])
    out_sequences = train[:, 1]
    in_seq_length = pre.seq_length(in_sequences)
    out_seq_length = pre.seq_length(out_sequences)
    
    val_in_sequences = np.array([val_personas[int(row[2])] + ' ' + row[0] for row in val])
    val_out_sequences = val[:, 1]
    
    print('Vocab size: %d' % vocab_size)
    print('Max input sequence length: %d' % in_seq_length)
    print('Max output sequence length: %d' % out_seq_length)

    # prepare training data
    trainX = encode_sequences(tokenizer, in_seq_length, in_sequences)
    trainY = encode_sequences(tokenizer, out_seq_length, out_sequences)
    trainY = encode_output(trainY, vocab_size)
    
    # validation data
    valX = encode_sequences(tokenizer, in_seq_length, val_in_sequences)
    valY = encode_sequences(tokenizer, out_seq_length, val_out_sequences)
    valY = encode_output(valY, vocab_size)
    
    model = seq2seq_model(vocab_size, vocab_size, in_seq_length, out_seq_length, 256)
    checkpoint = ModelCheckpoint(pre.MODEL_FN, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')
    model.fit(trainX, trainY, epochs=1, batch_size=64, validation_data=(valX, valY), 
              callbacks=[checkpoint], verbose=1)       
    
    # save the tokenizer so it can be used for prediction
    pre.save_object(tokenizer, pre.TOKENIZER_PKL_FN)
    
def evaluate():
    # load dataset
    dataset = pre.load_object(pre.CLEANED_PAIRS_PKL_FN)
    train = pre.load_object(pre.TRAIN_SET_FN)
    test = pre.load_object(pre.TEST_SET_FN)
    
    # english tokenizer
    eng_tokenizer = fit_tokenizer(dataset[:, 0])
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_length = pre.max_seq_length(dataset[:, 0])
    
    # german tokenizer
    ger_tokenizer = fit_tokenizer(dataset[:, 1])
    ger_vocab_size = len(ger_tokenizer.word_index) + 1
    ger_length = pre.max_seq_length(dataset[:, 1])
    
    trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
    testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
    model = load_model(pre.MODEL_FN)
    
    evaluate_by_auto_metrics(model, trainX, dataset, eng_tokenizer)
    evaluate_by_auto_metrics(model, testX, dataset, eng_tokenizer)
    
    
if __name__ == '__main__':
    train()