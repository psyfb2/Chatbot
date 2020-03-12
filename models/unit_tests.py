# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""
import unittest
from text_preprocessing import remove_first_num
from text_preprocessing import remove_contractions
from text_preprocessing import clean_line

class TestPreprocessing(unittest.TestCase):
    
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

    def test_clean_line(self):
       self.assertEqual(clean_line("blah. blah? blah, blah . blah..."), "blah . blah ? blah , blah . blah . . .")
       self.assertEqual(clean_line("__silence__ yes, it's . &    £AND  maybe."), "__silence__ yes , it is . and maybe .")
       
if __name__ == "__main__":
    unittest.main()