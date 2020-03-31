# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""

import argparse
import os
import text_preprocessing as pre
import seq2seq_model as seq2seq
import autoencoder_model as autoenc
import multiple_encoders as multenc
import tensorflow as tf
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
def reply(tokenizer, persona, input_msg, max_persona_length, max_message_length, max_reply_length, model=None, encoder_model=None, decoder_model=None, beam_width=3, show_beams=False):
    input_msg = pre.clean_line(input_msg)
    
    if model != None:
        # use auto encoder model, prepend persona to input message
        input_msg = persona + ' ' + pre.SEP_SEQ_TOKEN + ' ' + input_msg
        return autoenc.generate_reply_autoencoder(model, tokenizer, input_msg, max_persona_length + max_message_length)
    
    elif encoder_model != None and decoder_model != None:
        # use seq2seq model, prepend persona to input message
        #input_msg = persona + ' ' + pre.SEP_SEQ_TOKEN + ' ' + input_msg
        
        # beam search
        replys = seq2seq.beam_search_seq2seq(encoder_model, decoder_model, tokenizer, input_msg, max_persona_length + max_message_length, max_reply_length, beam_width)
        
        if show_beams:
            # display greedy search result as well all the beams
            # greedy search
            reply, _ = seq2seq.generate_reply_seq2seq(encoder_model, decoder_model, tokenizer, input_msg, max_persona_length + max_message_length, max_reply_length)
            print("Greedy Search:", reply, "\n")
            for r in replys:
                print("Beam Search:", r)
            print("")
        
        return replys[0]
    
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
    
    # ----------- Train Arguments ----------- #
    parser.add_argument("--train", default=None, type=str, choices=model_choices,
                        help="Name of the model to train")
    
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size for training, default = 64")
    
    parser.add_argument("--epochs", default=100, type=int,
                        help="epochs for training on PERSONA-CHAT dataset, default = 100")
    
    parser.add_argument("--batch_generator", default=True, type=str2bool,
                        help="Whether to generate sequence data batch by batch, when True will save considerable memory however training will be slower, default = True")
    
    parser.add_argument("--verbose", default=1, type=int,
                        help="Display progress bar for each batch during training, default = 1")
    
    parser.add_argument("--glove_filename", default="glove.6B.300d.txt", type=str,
                        help="The GLoVe filename to use e.g. glove.840B.300d.txt. Will automatically prepend data path")
    # ----------- ----------- #
    
    # ----------- Evaluation Arguments ----------- #
    parser.add_argument("--eval", default=None, type=str, choices=model_choices,
                        help="Name of the model to eval, must have already been trained.")
    # ----------- ----------- #
    
    # ----------- Talk Arguments ----------- #
    parser.add_argument("--talk", default=None, type=str, choices=model_choices,
                        help="Load the chosen model which has been trained and talk through the command line")
    
    parser.add_argument("--beam_width", default=3, type=int,
                        help="Beam width to use during beam search, default = 3")
    
    parser.add_argument("--show_beams", default=False, type=str2bool,
                        help="Generating responses uses beam search, chose whether to show not only the most likely sentence but beam width sentences")
    # ----------- ----------- #
    
    args = parser.parse_args()
    
    if tf.test.gpu_device_name():
        print("Using GPU Device: {}\n".format(tf.test.gpu_device_name()))
        
        # consume GPU memory dynamically instead allocating all the memory
        #gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        #tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    else:
        print("Using CPU\n")
    
    pre.GLOVE_FN = os.path.join(pre.PROJ_PATH, 'data', args.glove_filename)
    pre.VERBOSE  = args.verbose
    
    # 0 autoenc, 1 seq2seq, 3 multiple encoders, 4 merge
    
    if args.train != None:
        if args.train == model_choices[0]:
            autoenc.train_autoencoder(
                BATCH_SIZE=args.batch_size, EPOCHS=args.epochs, train_by_batch=args.batch_generator)
            
        elif args.train == model_choices[1]:
            seq2seq.train_seq2seq(
                args.epochs, args.batch_size)
            
        elif args.train == model_choices[2]:
            multenc.train_autoencoder() # just for testing, remove me
            
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
            pass
            
        elif args.talk == model_choices[2]:
            pass
        elif args.talk == model_choices[3]:
            pass
        
        persona, _ = pre.load_dataset(pre.TRAIN_FN)
        persona = persona[0]
        print("Talking with %s model, enter <exit> to close this program\n" % args.talk)
        print("Persona: %s" % persona)
        
        while True:
            msg = input("Enter your message: ")
            
            if msg == "<exit>":
                break
            
            response = reply(tokenizer, persona, msg, persona_length, msg_length, reply_length, model, encoder_model, decoder_model, args.beam_width, args.show_beams)
            
            print("Reply: %s\n" % response)
        