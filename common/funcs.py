#!/usr/bin/python
# coding: utf-8

import numpy as np


def h(x):
    """Binary entropy"""
    if x > 0:
        return -x * np.log2(x) - (1 - x) * np.log2(1 - x)
    elif x == 0:
        return 0
    else:
        print("Incorrect argument in binary entropy function")
