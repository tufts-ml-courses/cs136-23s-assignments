'''
Summary
-------
Defines Vocabulary class, for managing the unique possible words in a text corpus

Examples
--------
>>> vocab = Vocabulary(["a", "b", "c"])
>>> print(vocab.size)
3
>>> vocab.get_word_id("a")
0
'''

import numpy as np
import pandas as pd
import os
import string

def load_lowercase_word_list_from_file(txt_path_list):
    punc_remover = str.maketrans('', '', 
        "0123456789" + string.punctuation.replace("'",""))
    word_list = list()
    for txt_path in txt_path_list:
        with open(txt_path, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.translate(punc_remover)
                for word in line.split():
                    word = str.strip(word).lower()
                    if len(word) > 0:
                        word_list.append(word)
    return word_list

class Vocabulary(object):
    """
    Vocabulary manager for a corpus

    Examples
    --------
    # Try direct from a list
    >>> vocab = Vocabulary(["a", "b", "c"])
    >>> print(vocab.size)
    3

    # Write each of the letters of the alphabet in order to a file
    # Then read it in to a vocab
    >>> with open("/tmp/a.txt", 'w') as f:
    ...     for i in range(26):
    ...         unused = f.write("%s " % str(chr(97+i)))
    >>> vocab = Vocabulary(["/tmp/a.txt"])
    >>> print(vocab.size)
    26
    >>> print(vocab.get_word_id("a"))
    0
    >>> print(vocab.get_word_id("m"))
    12
    >>> print(vocab.get_word_id("z"))
    25

    """

    def __init__(self, corpus_filepath_or_word_list):
        ''' Create vocabulary from a list of words or a list of paths to text files

        Args
        ----
        corpus_filepath_list : list of strings
                Each string should be a filepath to a plain-text file in UTF-8 format
                File should contain words separated by spaces

        Returns
        -------
        Instantiated object
        '''
        self.vocab_dict = dict()
        self.size = 0

        punc_remover = str.maketrans('', '', 
            "0123456789" + string.punctuation.replace("'",""))

        for fpath_or_word in corpus_filepath_or_word_list:
            if os.path.exists(fpath_or_word):
                fpath = fpath_or_word
                line_list = np.loadtxt(fpath, dtype=str, delimiter=' \t\n', ndmin=1)
                for line in line_list:
                    line = line.translate(punc_remover)
                    for word in line.split():
                        word = str.strip(word).lower()
                        if len(word) > 0 and word not in self.vocab_dict:
                            self.vocab_dict[word] = self.size
                            self.size += 1
            else:
                word = str.strip(fpath_or_word).translate(punc_remover)
                if len(word) > 0 and word not in self.vocab_dict:
                    self.vocab_dict[word] = self.size
                    self.size += 1

    def get_word_id(self, word):
        ''' Retrieve the integer id of the provided word in the corpus

        Returns
        -------
        w_id : int
                Value between 0 and vocab_size - 1, inclusive

        Raises
        ------
        KeyError, if the word is out of vocabulary
        '''
        if word not in self.vocab_dict:
            raise KeyError("Word %s not in the vocabulary" % word)
        return self.vocab_dict[word]
