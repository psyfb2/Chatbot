# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""
import unittest
import numpy as np
from text_preprocessing import remove_first_num
from text_preprocessing import remove_contractions
from text_preprocessing import clean_line
from text_preprocessing import load_dataset, TRAIN_FN, VALID_FN
from transformer import DecoderLayer
from transformer import Transformer
from transformer import create_subword_tokenizer
from transformer import encode_training_examples
from transformer import generate_segment_list
from transformer import SOS, EOS, SEP, PSN, MSG

class UnitTests(unittest.TestCase):
    ''' Test preprocessing function, data pipeline and transformer dimensions '''
    def test_remove_first_number(self):
        self.assertEqual(remove_first_num("1 your persona: I am 15"), " your persona: I am 15")
        self.assertEqual(remove_first_num("6 just finished a 5 mile climb	wow exciting ! where did you go ?"), " just finished a 5 mile climb	wow exciting ! where did you go ?")
        self.assertEqual(remove_first_num(" no numbers to remove here"), " no numbers to remove here")
        self.assertEqual(remove_first_num(""), "")
        self.assertEqual(remove_first_num("143"), "")
        self.assertEqual(remove_first_num("your persona: I am 15"), "your persona: I am ")
        self.assertEqual(remove_first_num("5 i love to work out , what about you ?	that's nice i love my car"), " i love to work out , what about you ?	that's nice i love my car")
        self.assertEqual(remove_first_num("jefone16293nfen"), "jefonenfen")
        self.assertEqual(remove_first_num("1 12345 1 1 1 hello"), " 12345 1 1 1 hello")
        
    def test_remove_contractions(self):
        self.assertEqual(remove_contractions("1 your persona: i've a big spider"), "1 your persona: i have a big spider")
        self.assertEqual(remove_contractions("1 your persona: i'm 15 years old"), "1 your persona: i am 15 years old")
        self.assertEqual(remove_contractions("1 your persona: i can't sing"), "1 your persona: i can not sing")
        self.assertEqual(remove_contractions("1 your persona: i haven't cried in 5 years"), "1 your persona: i have not cried in 5 years")
        self.assertEqual(remove_contractions("you've"), "you have")

    def test_clean_line(self):
       self.assertEqual(clean_line("blah. blah? blah, blah . blah..."), "blah . blah ? blah , blah . blah . . .")
       self.assertEqual(clean_line("__silence__ yes, it's . &    £AND  maybe."), "__silence__ yes , it is . and maybe .")
       
    def test_data_pipeline(self):
        # integration test
        tokenizer, in_seq_length, out_seq_length = create_subword_tokenizer()
        train_personas, train_data = load_dataset(TRAIN_FN)
    
        encoder_input, decoder_target = encode_training_examples(train_personas, 
                                                                 train_data, tokenizer, 
                                                                 in_seq_length, out_seq_length)
        segment_input  = np.array([generate_segment_list(encoded_msg, 
                                   in_seq_length, tokenizer.vocab_size + SEP) for encoded_msg in encoder_input])
        def rem(arr, vocab_size):
            return [i for i in arr if i != 0 and i < vocab_size]
        
        def check_seg(arr, segment_arr, vocab_size):
            self.assertEqual(len(arr), len(segment_arr))
            self.assertEqual(arr[0], vocab_size + SOS)
            self.assertEqual(vocab_size + SEP in arr, True)
            
            
            for i in range(len(arr)):
                if arr[i] == vocab_size + SEP:
                    # check segment array matches with the input
                    self.assertEqual(segment_arr[i],  PSN)
                    self.assertEqual(len(set(segment_arr[:i+1])), 1)
                    
                    self.assertEqual(segment_arr[i + 1], MSG)
                    if segment_arr[-1] == 0:
                        # account for padding value
                        self.assertEqual(len(set(segment_arr[i+1:])), 2)
                    else:
                        self.assertEqual(len(set(segment_arr[i+1:])), 1)
                    break
        
        orig = (
            clean_line("i like to remodel homes. i like to go hunting. i like to shoot a bow. my favorite holiday is halloween.") +
            clean_line("hi , how are you doing ? i'm getting ready to do some cheetah chasing to stay in shape ."))
        self.assertEqual(tokenizer.decode(rem(encoder_input[0, :], tokenizer.vocab_size)), orig)
        
        orig = (
            clean_line("my mother lives with me. i like to plant flowers in my gardens. i like to take my dog for long walks. i have a lizard named nagini. i enjoy cooking for people.") +
            clean_line("i had a pair of glasses with green frames when i was younger"))
        self.assertEqual(tokenizer.decode(rem(encoder_input[-1, :], tokenizer.vocab_size)), orig)
        
        check_seg(encoder_input[0, :], segment_input[0, :], tokenizer.vocab_size)
        check_seg(encoder_input[-1, :], segment_input[-1, :], tokenizer.vocab_size)
        
        orig = clean_line("you must be very fast . hunting is one of my favorite hobbies .")
        self.assertEqual(tokenizer.decode(rem(decoder_target[0, :], tokenizer.vocab_size)), orig)
        
        orig = clean_line("let me guess , green is your favorite color too ?")
        self.assertEqual(tokenizer.decode(rem(decoder_target[-1, :], tokenizer.vocab_size)), orig)
        
        # validation data
        train_personas, train_data = load_dataset(VALID_FN)
        encoder_input, decoder_target = encode_training_examples(train_personas, 
                                                                 train_data, tokenizer, 
                                                                 in_seq_length, out_seq_length)
        segment_input  = np.array([generate_segment_list(encoded_msg, 
                                   in_seq_length, tokenizer.vocab_size + SEP) for encoded_msg in encoder_input])
        
        orig = (
            clean_line("i read twenty books a year. i'm a stunt double as my second job. i only eat kosher. i was raised in a single parent household.") +
            clean_line("hello what are doing today ?"))
        self.assertEqual(tokenizer.decode(rem(encoder_input[0, :], tokenizer.vocab_size)), orig)
        
        orig = (
            clean_line("i am a vegan and i love hummus. i love rollercoasters and sky diving. i do like watching cooking shows. i am not a good swimmer at all.") +
            clean_line("three and one on the way . the dog is like a child too ."))
        self.assertEqual(tokenizer.decode(rem(encoder_input[-1, :], tokenizer.vocab_size)), orig)
        
        check_seg(encoder_input[0, :], segment_input[0, :], tokenizer.vocab_size)
        check_seg(encoder_input[-1, :], segment_input[-1, :], tokenizer.vocab_size)
        
        orig = clean_line("i am good , i just got off work and tired , i have two jobs .")
        self.assertEqual(tokenizer.decode(rem(decoder_target[0, :], tokenizer.vocab_size)), orig)
        
        orig = clean_line("i love dogs want a husky but cant have one yet")
        self.assertEqual(tokenizer.decode(rem(decoder_target[-1, :], tokenizer.vocab_size)), orig)

if __name__ == "__main__":
    unittest.main()