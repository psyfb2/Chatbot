# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""

import numpy as np
import text_preprocessing as pre
import seq2seq_model as seq2seq
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.models import load_model



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
    pass
    
    
if __name__ == '__main__':
    seq2seq.train_seq2seq()