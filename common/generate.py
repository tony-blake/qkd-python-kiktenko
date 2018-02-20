# coding: utf-8

import numpy as np


def generate_key(length):
    """
    Generate random key of length 'length'
    """
    return np.random.randint(0, 2, (1, length))[0]


def generate_key_zeros(length):
    """
    Generate key with zeors only of length 'length'
    """
    return np.random.randint(0, 1, (1, length))[0]


def add_errors(a, error_prob):
    """
    Flip some values (1->0, 0->1) in 'a' with probability 'error_prob'
    """
    error_mask = np.random.choice(2, size=a.shape, p=[1.0-error_prob, error_prob])
    return np.where(error_mask, ~a+2, a)
