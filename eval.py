# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""

import numpy as np
import text_preprocessing as pre
import seq2seq_model as seq2seq
from nltk.translate.bleu_score import corpus_bleu
from keras.models import load_model



''' Given a model, matrix of the integer encoded sources, raw dataset, tokenizer
    print bleu scores for all examples in sources '''
def evaluate_by_auto_metrics(model, sources, dataset_not_encoded, tokenizer, verbose=1):
    target_sentences = []
    predicted_sentences = []
    
    for i, source in enumerate(sources):
        source = source.reshape((1, source.shape[0]))
        # source is a row vector of encoded integers, predict it using the model
        pred = pre.predict_sequence(model, tokenizer, source)
        
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
        

def evaluate():
    # load dataset
    dataset = pre.load_object(pre.CLEANED_PAIRS_PKL_FN)
    train = pre.load_object(pre.TRAIN_SET_FN)
    test = pre.load_object(pre.TEST_SET_FN)
    
    # english tokenizer
    eng_tokenizer = pre.fit_tokenizer(dataset[:, 0])
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_length = pre.max_seq_length(dataset[:, 0])
    
    # german tokenizer
    ger_tokenizer = pre.fit_tokenizer(dataset[:, 1])
    ger_vocab_size = len(ger_tokenizer.word_index) + 1
    ger_length = pre.max_seq_length(dataset[:, 1])
    
    trainX = pre.encode_sequences(ger_tokenizer, ger_length, train[:, 1])
    testX = pre.encode_sequences(ger_tokenizer, ger_length, test[:, 1])
    model = load_model(pre.MODEL_FN)
    
    evaluate_by_auto_metrics(model, trainX, dataset, eng_tokenizer)
    evaluate_by_auto_metrics(model, testX, dataset, eng_tokenizer)
    
    
if __name__ == '__main__':
    seq2seq.train_seq2seq()