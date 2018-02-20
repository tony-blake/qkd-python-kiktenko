# coding: utf-8

import numpy as np


def array_from_file(file_path):
    try:
        with open(file_path, 'r') as arr_file:
            lines = arr_file.readlines()
            return np.array(map(lambda line: map(int, list(line[:-1])), lines))
    except IOError:
        raise ValueError("File not readable.")
    except ValueError:
        raise ValueError("File format incorrect.")


def array_to_file(array, file_path, delimeter=None):
    if delimeter is None:
        delimeter = ''
    with open(file_path, 'w+') as arr_file:
        arr_file.writelines(map(lambda line: delimeter.join(map(str, line)) + '\n', array))


def int_array_from_file(file_path):
    try:
        with open(file_path, 'r') as arr_file:
            lines = arr_file.readlines()
            return np.array(map(lambda line: int(line[:-1]), lines))
    except IOError:
        raise ValueError("File not readable.")
    except ValueError:
        raise ValueError("File format incorrect.")


def int_array_to_file(array, file_path):
    with open(file_path, 'w+') as arr_file:
        arr_file.writelines(map(lambda x: str(x) + '\n', array))
