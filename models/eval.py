# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""

import argparse
import text_preprocessing as pre
import seq2seq_model as seq2seq
import autoencoder_model as autoenc
from tensorflow.keras.models import load_model


''' Evaluate a model by calculating Perplexity and F1 score '''
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
            print('msg=[%s], reply=[%s], predicted=[%s]' % (src, target, pred))
        target_sentences.append(target)
        predicted_sentences.append(pred)

''' Generate a reply given an input_msg (not included prepended persona), can calculate which model is specified from inputs '''
def reply(tokenizer, persona, input_msg, max_persona_length, max_message_length, max_reply_length, model=None, encoder_model=None, decoder_model=None):
    if model != None:
        # use auto encoder model, prepend persona to input message
        input_msg = persona + ' ' + pre.SEP_SEQ_TOKEN + ' ' + input_msg
        return autoenc.generate_reply_autoencoder(model, tokenizer, input_msg, max_persona_length + max_message_length)
    
    elif encoder_model != None and decoder_model != None:
        # use seq2seq model, prepend persona to input message
        input_msg = persona + ' ' + pre.SEP_SEQ_TOKEN + ' ' + input_msg
        reply, attn_weights = seq2seq.generate_reply_seq2seq(encoder_model, decoder_model, tokenizer, input_msg, max_persona_length + max_message_length, max_reply_length) 
        return reply
    
def str2bool(s):
    if isinstance(s, bool):
       return s
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    model_choices = ['autoenc', 'seq2seq', 'multiple_encoders', 'merge']
    parser = argparse.ArgumentParser(description="Train and Evaluate Different Chatbot Models")
    
    parser.add_argument("--train", default=None, type=str, choices=model_choices,
                        help="Name of the model to train")
    
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size for training, default = 128")
    
    parser.add_argument("--epochs", default=100, type=int,
                        help="epochs for training on PERSONA-CHAT dataset, default = 100")
    
    parser.add_argument("--batch_generator", default=True, type=str2bool,
                        help="Whether to generate sequence data batch by batch, when True will save considerable memory however training will be slower, default = True")
    
    
    parser.add_argument("--eval", default=None, type=str, choices=model_choices,
                        help="Name of the model to eval, must have already been trained.")
    
    parser.add_argument("--talk", default=None, type=str, choices=model_choices,
                        help="Load the chosen model which has been trained and talk through the command line")
    
    args = parser.parse_args()
    
    # 0 autoenc, 1 seq2seq, 3 multiple encoders, 4 merge
    
    if args.train != None:
        if args.train == model_choices[0]:
            autoenc.train_autoencoder(
                BATCH_SIZE=args.batch_size, EPOCHS=args.epochs, train_by_batch=args.batch_generator)
            
        elif args.train == model_choices[1]:
            seq2seq.train_seq2seq(
                BATCH_SIZE=args.batch_size, EPOCHS=args.epochs, train_by_batch=args.batch_generator)
            
        elif args.train == model_choices[2]:
            pass
        elif args.train == model_choices[3]:
            pass
    
    if args.eval != None:
        if args.eval == model_choices[0]:
            pass
        elif args.eval == model_choices[1]:
            pass
        elif args.eval == model_choices[2]:
            pass
        elif args.eval == model_choices[3]:
            pass

    if args.talk != None:
        model = None
        encoder_model = None
        decoder_model = None
        vocab, persona_length, msg_length, reply_length = pre.get_vocab()
        tokenizer = pre.fit_tokenizer(vocab)
        
        if args.talk == model_choices[0]:
            model = load_model(pre.AUTOENC_MODEL_FN)
            
        elif args.talk == model_choices[1]:
            encoder_model = load_model(pre.SEQ2SEQ_ENCODER_MODEL_FN)
            decoder_model = load_model(pre.SEQ2SEQ_DECODER_MODEL_FN)
            
        elif args.talk == model_choices[2]:
            pass
        elif args.talk == model_choices[3]:
            pass
        
        persona, _ = pre.load_dataset(pre.TRAIN_FN)
        persona = persona[0]
        print("Talking with %s model, enter <exit> to close this program\n" % args.talk)
        print("Persona: %s" % persona)#
        
        while True:
            msg = input("Enter your message: ")
            
            if msg == "<exit>":
                break
            
            msg = pre.clean_line(msg)
            response = reply(tokenizer, persona, msg, persona_length, msg_length, reply_length, model, encoder_model, decoder_model)
            
            print("Reply: %s\n" % response)
        