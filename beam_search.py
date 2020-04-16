# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:15:49 2020

@author: jacqu

Utils and classes for beam search 
"""

class BeamSearchNode():
    def __init__(self, h, rnn_in, score, sequence):
        self.h = h
        self.rnn_in = rnn_in
        self.score = score
        self.sequence = sequence
        self.max_len = 60

    def __lt__(self, other):  # For x < y
        # Pour casser les cas d'égalité du score au hasard, on s'en fout un peu.
        # Eventuellement affiner en regardant les caractères de la séquence (pénaliser les cycles ?)
        return True