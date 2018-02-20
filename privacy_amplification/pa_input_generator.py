#!/usr/bin/python
# coding: utf-8

import sys
import os

sys.path.insert(0, os.path.abspath('..'))

from utils.generate import generate_key
from utils.files import array_to_file
from utils.parseargs import ErrorPrintArgumentParser


def generate(message_size, toeplitz_seed_length):
    key = generate_key(message_size)[0]
    toeplitz_seed = generate_key(toeplitz_seed_length)[0]
    return key, toeplitz_seed


def main():
    parser = ErrorPrintArgumentParser(description='File generator for authorization.py.')

    parser.add_argument('-s', '--message-size', default=2**20, type=int,
                        help='Message size', metavar='message_size')
    parser.add_argument('-t', '--toeplitz-seed-length', default=2**21, type=int,
                        help='Length of string used for Toeplitz matrix generation',
                        metavar='toeplitz_seed_length')
    args = parser.parse_args()

    key, toeplitz_seed = generate(args.message_size, args.toeplitz_seed_length)

    array_to_file([key], 'key.txt')
    array_to_file([toeplitz_seed], 'toeplitz_seed.txt')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        sys.stderr.write(u"{}\n{}\n".format(-1, exc))
        sys.exit(1)
