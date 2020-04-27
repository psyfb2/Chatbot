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
import tensorflow as tf

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


def reply(tokenizer, persona, input_msg, max_persona_length, max_message_length, max_reply_length, model=None, encoder_model=None, decoder_model=None, include_greedy_search=False, plot_attn=False, beam_width=3):
    ''' 
    Generate a list of reply's from most likely to least
    given an input_msg (not included prepended persona)
    '''
    input_msg = pre.clean_line(input_msg)
    
    if model != None:
        # use auto encoder model, does not take into account persona
        return [autoenc.generate_reply_autoencoder(model, tokenizer, input_msg, max_message_length)]
    
    elif encoder_model != None and decoder_model != None:
        # use seq2seq model, prepend persona to input message
        input_msg = persona + ' ' + pre.SEP_SEQ_TOKEN + ' ' + input_msg
        
        # beam search
        replys = seq2seq.beam_search_seq2seq(encoder_model, decoder_model, tokenizer, input_msg, max_persona_length + max_message_length, max_reply_length, beam_width)
        
        # greedy search
        if plot_attn or include_greedy_search:
            reply, attn_weights = seq2seq.generate_reply_seq2seq(encoder_model, decoder_model, tokenizer, input_msg, max_persona_length + max_message_length, max_reply_length)
            # return beam search reply's with greeding search reply appended at the end
            replys.append(reply)
            
            if plot_attn:
                seq2seq.plot_attention(attn_weights[:len(reply.split(' ')), :len(input_msg.split(' '))], input_msg, reply)
        
        return replys
    
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
                        help="Batch size for training, default = 64")
    
    parser.add_argument("--epochs", default=100, type=int,
                        help="epochs for training on PERSONA-CHAT dataset, default = 100")
    
    parser.add_argument("--early_stopping_patience", default=15, type=int,
                        help="number of consecutive epochs without improvement over validation loss minimum until training is stopped early, default = 15")
    
    parser.add_argument("--segment_embedding", default=True, type=str2bool,
                        help="Whether or not to add an additional segment embedding to seq2seq models so that it can better distiguish persona from message. Only applies to seq2seq and deep_seq2seq, default = True")
    
    parser.add_argument("--verbose", default=1, type=int,
                        help="Display loss for each batch during training, default = 0")
    
    parser.add_argument("--min_epochs", default=0, type=int,
                        help="a minimum number of epochs which the model must be trained for on the PERSONA-CHAT dataset regardless of early stopping, default = 0")
    
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
                        help="Generating responses uses beam search, chose whether to show not only the most likely sentence but beam width sentences, default = False")
    
    parser.add_argument("--include_greedy_search", default=False, type=str2bool,
                        help="On top of beam width responses also show the reply of greedy search, default = False")
    
    parser.add_argument("--plot_attention", default=False, type=str2bool,
                        help="Plot the attention weights for a response generated by greedy search, default = False")
    # ----------- ----------- #
    
    args = parser.parse_args()
    
    if tf.test.gpu_device_name():
        print("Using GPU Device: {}\n".format(tf.test.gpu_device_name()))
        
        # consume GPU memory dynamically instead allocating all the memory
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
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
                args.min_epochs, deep_lstm=False, use_segment_embedding=args.use_segment_embedding)
            
        elif args.train == model_choices[2]:
            seq2seq.train_seq2seq(
                args.epochs, args.batch_size, args.early_stopping_patience, 
                args.min_epochs, deep_lstm=True, use_segment_embedding=args.use_segment_embedding)
            
        elif args.train == model_choices[3]:
            multenc.train_multiple_encoders(args.epochs, args.batch_size, args.early_stopping_patience, args.min_epochs ,deep_lstm=False)
        
        elif args.train == model_choices[4]:
            multenc.train_multiple_encoders(args.epochs, args.batch_size, args.early_stopping_patience, args.min_epochs, deep_lstm=True)
        
        elif args.train == model_choices[5]:
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
        elif args.eval == model_choices[4]:
            pass
        elif args.eval == model_choices[5]:
            pass

    if args.talk != None:
        model = None
        encoder_model = None
        decoder_model = None
        vocab, persona_length, msg_length, reply_length = pre.get_vocab()
        tokenizer = pre.fit_tokenizer(vocab)
        
        if args.talk == model_choices[0]:
            model = tf.keras.models.load_model(pre.AUTOENC_MODEL_FN)
        
        elif args.talk == model_choices[1]:
            encoder_model = tf.saved_model.load(pre.SEQ2SEQ_ENCODER_MODEL_FN)
            decoder_model = tf.saved_model.load(pre.SEQ2SEQ_DECODER_MODEL_FN)
            
        elif args.talk == model_choices[2]:
            encoder_model = tf.saved_model.load(pre.SEQ2SEQ_ENCODER_DEEP_MODEL_FN)
            decoder_model = tf.saved_model.load(pre.SEQ2SEQ_DECODER_DEEP_MODEL_FN)
        
        elif args.talk == model_choices[3]:
            pass
        
        elif args.eval == model_choices[4]:
            pass
        
        elif args.eval == model_choices[5]:
            pass
        
        persona, _ = pre.load_dataset(pre.TRAIN_FN)
        persona = persona[0]
        print("Talking with %s model, enter <exit> to close this program\n" % args.talk)
        print("Persona: %s" % persona)
        
        while True:
            msg = input("Enter your message: ")
            
            if msg == "<exit>":
                break
            
            responses = reply(tokenizer, persona, msg, persona_length, msg_length, reply_length, model, encoder_model, decoder_model, args.include_greedy_search, args.plot_attention, args.beam_width)
            
            print("Reply: %s" % (responses[0]))
            
            if args.show_beams or args.include_greedy_search:
                for i in range(1, len(responses)):
                    print("Reply %d: %s" % (i + 1, responses[i]))
                
            print("\n")
        