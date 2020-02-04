# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - Tristan Bepler's code (https://github.com/tbepler/protein-sequence-embedding-iclr2019)

""" Biological sequence encoding functions """

import numpy as np


class Alphabet:
    """ biological sequence encoder """
    def __init__(self, chars, encoding, chars_rc=None, encoding_rc=None, missing=255):
        self.chars = np.frombuffer(chars, dtype=np.uint8)
        self.encoding = np.zeros(256, dtype=np.uint8) + missing
        self.encoding[self.chars] = encoding
        if chars_rc is not None:
            self.chars_rc = np.frombuffer(chars_rc, dtype=np.uint8)
            self.encoding_rc = np.zeros(256, dtype=np.uint8) + missing
            self.encoding_rc[self.chars_rc] = encoding_rc
        self.size = encoding.max() + 1

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return chr(self.chars[i])

    def encode(self, x, reverse_complement=False):
        """ encode a byte string into alphabet indices """
        if not reverse_complement:
            x = np.frombuffer(x, dtype=np.uint8)
            string = self.encoding[x]
        else:
            x = np.frombuffer(x, dtype=np.uint8)[::-1]
            string = self.encoding_rc[x]
        return string

    def decode(self, x, reverse_complement=False):
        """ decode index array, x, to byte string of this alphabet """
        if not reverse_complement:
            string = self.chars[x-1]
        else:
            string = self.chars_rc[x[::-1]-1]
        return string.tobytes()


class DNA(Alphabet):
    """ DNA sequence encoder """
    def __init__(self, unknown_nt=False):
        if unknown_nt: chars = b'ACGTN'; chars_rc =  b'TGCAN'
        else:          chars = b'ACGT';  chars_rc =  b'TGCA'
        encoding = np.arange(len(chars))
        encoding += 1                    # leave 0 for padding tokens
        encoding_rc = np.arange(len(chars_rc))
        encoding_rc += 1                 # leave 0 for padding tokens
        super(DNA, self).__init__(chars, encoding, chars_rc=chars_rc, encoding_rc=encoding_rc, missing=len(chars))


class RNA(Alphabet):
    """ RNA sequence encoder """
    def __init__(self, unknown_nt=False):
        if unknown_nt: chars = b'ACGUN'; chars_rc =  b'UGCAN'
        else:          chars = b'ACGU';  chars_rc =  b'UGCA'
        encoding = np.arange(len(chars))
        encoding += 1                    # leave 0 for padding tokens
        encoding_rc = np.arange(len(chars_rc))
        encoding_rc += 1                 # leave 0 for padding tokens
        super(RNA, self).__init__(chars, encoding, chars_rc=chars_rc, encoding_rc=encoding_rc, missing=len(chars))


class Protein(Alphabet):
    """ protein sequence encoder """
    def __init__(self):
        chars = b'ARNDCQEGHILKMFPSTWYVXOUBZ'
        encoding = np.arange(len(chars))
        encoding[21:] = [11, 4, 20, 20]  # encode 'OUBZ' as synonyms
        encoding += 1                    # leave 0 for padding tokens
        super(Protein, self).__init__(chars, encoding, missing=21)


# PLUS

class SecStr8(Alphabet):
    """ protein secondary structure encoder """
    def __init__(self):
        chars =  b'HBEGITS '
        encoding = np.arange(len(chars))
        super(SecStr8, self).__init__(chars, encoding, missing=255)
