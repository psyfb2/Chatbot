#!/usr/bin/env python3
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
import transformer
import tensorflow as tf
from multiple_encoders import ChatBot as me_chatbot
from sys import exit
    
def input_number(prompt, limit):
    '''
    Get a number from the user between 0 - limit

    Parameters
    ----------
    limit : int
        max number allowed.
    prompt: str
        text to show the user.

    Returns
    -------
    int
        number entered by the user.

    '''
    while True:
        try:
            i = int(input(prompt))
        except Exception:
            print("Please only provide integer input")
        
        if i < 0 or i > limit:
            print("Please enter an integer between 1 and %d" % limit)
            continue
        
        return i
    
def get_chatbot(arg, model_choices):
    if arg == model_choices[0]:
        # autoencoder model wasn't good at all so leave it out for now
        return
    elif arg == model_choices[1]:
        return seq2seq.ChatBot(deep_model=False)
        
    elif arg == model_choices[2]:
        return seq2seq.ChatBot(deep_model=True)

    elif arg == model_choices[3]:
        return me_chatbot(deep_model=False)
    
    elif arg == model_choices[4]:
        return me_chatbot(deep_model=True)
    
    elif arg == model_choices[5]:
        return transformer.ChatBot()
    
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
    model_choices = ['autoenc', 'seq2seq', 'deep_seq2seq', 'multiple_encoders', 'deep_multiple_encoders', 'transformer']
    parser = argparse.ArgumentParser(description="Train and Evaluate Different Chatbot Models")
    
    # ----------- Train Arguments ----------- #
    parser.add_argument("--train", default=None, type=str, choices=model_choices,
                        help="Name of the model to train")
    
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size for training. It is reccomended to use highest possible batch size for transformer (e.g. 256) .")
    
    parser.add_argument("--epochs", default=100, type=int,
                        help="epochs for training on PERSONA-CHAT dataset")
    
    parser.add_argument("--early_stopping_patience", default=7, type=int,
                        help="number of consecutive epochs without improvement over validation loss minimum until training is stopped early")
    
    parser.add_argument("--segment_embedding", default=True, type=str2bool,
                        help="Whether or not to add an additional segment embedding to seq2seq models so that it can better distiguish persona from message.")
    
    parser.add_argument("--perform_pretraining", default=False, type=str2bool,
                        help="Perform pretraining on movie, dialy dialog datasets? default = False")
    
    parser.add_argument("--verbose", default=0, type=int,
                        help="Display loss for each batch during training")
    
    parser.add_argument("--min_epochs", default=15, type=int,
                        help="a minimum number of epochs which the model must be trained for on the PERSONA-CHAT dataset regardless of early stopping")
    
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
    
    parser.add_argument("--beam_search", default=False, type=str2bool,
                        help="Use beam search")
    
    parser.add_argument("--plot_attention", default=False, type=str2bool,
                        help="Plot attention weights for greedy search")
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
    
    # 0 autoenc, 1 seq2seq, 2 deep_seq2seq, 3 multiple_encoder, 4 deep_multiple_encoders, 5 transformer
    
    if args.train != None:
        if args.train == model_choices[0]:
            autoenc.train_autoencoder(
                BATCH_SIZE=args.batch_size, EPOCHS=args.epochs, PATIENCE=args.early_stopping_patience)
            
        elif args.train == model_choices[1]:
            seq2seq.train_seq2seq(
                args.epochs, args.batch_size, args.early_stopping_patience, 
                args.min_epochs, deep_lstm=False, use_segment_embedding=args.segment_embedding,
                pre_train=args.perform_pretraining)
            
        elif args.train == model_choices[2]:
            seq2seq.train_seq2seq(
                args.epochs, args.batch_size, args.early_stopping_patience, 
                args.min_epochs, deep_lstm=True, use_segment_embedding=args.segment_embedding,
                pre_train=args.perform_pretraining)
            
        elif args.train == model_choices[3]:
            multenc.train_multiple_encoders(args.epochs, args.batch_size, 
                                            args.early_stopping_patience, args.min_epochs,
                                            deep_lstm=False, pre_train=args.perform_pretraining)
        
        elif args.train == model_choices[4]:
            multenc.train_multiple_encoders(args.epochs, args.batch_size, 
                                            args.early_stopping_patience, args.min_epochs, 
                                            deep_lstm=True, pre_train=args.perform_pretraining)
        
        elif args.train == model_choices[5]:
            transformer.train_transformer(args.epochs, args.batch_size, 
                                          args.early_stopping_patience, args.min_epochs,
                                          args.segment_embedding)
    if args.eval != None:
        print("Calculating Perplexity, this can take some time")
        chatbot = get_chatbot(args.eval, model_choices)
        if chatbot == None:
            print("Uknown model", args.eval)
            exit()
        
        print("Perplexity:", chatbot.eval_ppl())
        
        print("Calculating F1, this can take some time")
        print("F1:", chatbot.eval_f1())

    if args.talk != None:
        print("Loading model")
        chatbot = get_chatbot(args.talk, model_choices)
        if chatbot == None:
            print("Uknown model", args.talk)
            exit()
            
        persona = pre.get_only_personas()
        persona_num = input_number("Please choose a persona number between 0 - %d: " % (len(persona) - 1),
                                   len(persona) - 1)
        persona = persona[persona_num]
        print("Persona: %s\n" % persona)
        
        print("Talking with %s model, enter <exit> to close this program\n" % args.talk)
        
        while True:
            msg = input("Message: ")
            
            if msg == "<exit>":
                break
            
            msg = pre.clean_line(msg)
            
            if not args.beam_search:
                reply = chatbot.reply(persona, msg)
            else:
                reply = chatbot.reply(persona, msg)
                #reply = chatbot.beam_search_reply(persona, msg, args.beam_width)
                #reply = reply[0]
            
            print(reply, "\n")
            
            if args.plot_attention and not args.beam_search:
                # keeping track of attention weights for beams is costly
                # so only do this for greedy search
                chatbot.plot_attn()
