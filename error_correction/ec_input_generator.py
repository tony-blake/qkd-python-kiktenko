#!/usr/bin/python
# coding: utf-8

import sys
import os

sys.path.insert(0, os.path.abspath('..'))

from common.parseargs import ErrorPrintArgumentParser
from common.generate import generate_key
from common.generate import add_errors
from common.files import array_to_file


def generate(frame_len, block_count, error_probability):
    sifted_key_a = generate_key(frame_len * block_count)
    sifted_key_b = add_errors(sifted_key_a, error_probability)
    return sifted_key_a, sifted_key_b


def main():
    parser = ErrorPrintArgumentParser(description='File generator for error correction alghoritms.')

    parser.add_argument('-f', '--frame-len', default=1944, type=int,
                        help='Block size', metavar='block_size')
    parser.add_argument('-n', '--block-count', default=512, type=int,
                        help='Number of blocks', metavar='n')
    parser.add_argument('-e', '--error-probability', default=0.027, type=float,
                        help='Error probability for key generation', metavar='e')

    args = parser.parse_args()

    sifted_key_a, sifted_key_b, ver_ht_key = generate(
        args.frame_len, args.block_count, args.error_probability)

    array_to_file([sifted_key_a], 'sifted_key_a.txt')
    array_to_file([sifted_key_b], 'sifted_key_b.txt')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        sys.stderr.write(u"{}\n{}\n".format(-1, exc))
        sys.exit(1)
