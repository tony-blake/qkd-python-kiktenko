#!/usr/bin/python
# coding: utf-8

import sys
import os

import random
import numpy as np

sys.path.insert(0, os.path.abspath('..'))

from common.generate import generate_key
from common.files import array_to_file
from common.parseargs import ErrorPrintArgumentParser


GOST_KEY_SIZE = 256


def generate(message_size, unihash_block_size, unihash_bits_per_block):
    message = generate_key(message_size)
    gost_key = generate_key(GOST_KEY_SIZE)

    unihash_block_count = message_size / unihash_block_size + (message_size % unihash_block_size > 0)
    unihash_key_size = unihash_bits_per_block * unihash_block_count
    toeplitz_seed_size = unihash_block_size + unihash_bits_per_block - 1

    unihash_key = generate_key(unihash_key_size)
    toeplitz_seed = generate_key(toeplitz_seed_size)

    corrupt_index = random.randint(0, message_size)
    corrupted_message = np.copy(message)
    corrupted_message[corrupt_index] = 0 if corrupted_message[corrupt_index] == 1 else 1

    return message, corrupted_message, gost_key, unihash_key, toeplitz_seed


def main():
    parser = ErrorPrintArgumentParser(description='File generator for authorization.py.')

    parser.add_argument('-s', '--message-size', default=2*20, type=int,
                        help='Message size', metavar='message_size')
    parser.add_argument('-t', '--unihash-block-size', default=2**24, type=int,
                        help='Message block size in unihash', metavar='unihash_block_size')
    parser.add_argument('-u', '--unihash-bits-per-block', default=68, type=int,
                        help='Key bits per message block for unihash',
                        metavar='unihash_bits_per_block')
    args = parser.parse_args()

    message, corrupted_message, gost_key, \
    unihash_key, toeplitz_seed = generate(args.message_size, args.unihash_block_size,
                                          args.unihash_bits_per_block)

    array_to_file([message], 'msg.txt')
    array_to_file([corrupted_message], 'msg_cor.txt')
    array_to_file([gost_key], 'gost_key.txt')
    array_to_file([unihash_key], 'unihash_key.txt')
    array_to_file([toeplitz_seed], 'toeplitz_seed.txt')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        sys.stderr.write(u"{}\n{}\n".format(-1, exc))
        sys.exit(1)
