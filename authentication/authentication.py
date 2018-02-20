#!/usr/bin/python
# coding: utf-8

import sys
import os
import numpy as np
from struct import Struct

sys.path.insert(0, os.path.abspath('..'))

from common.parseargs import ErrorPrintArgumentParser
from common.files import array_from_file
from common.files import array_to_file


MESSAGE_BLOCK_SIZE = 2 ** 24
KEY_BLOCK_SIZE = 68


def unihash_algo(message, toeplitz_seed_and_key):
    """
    UniHash algorhythm implementation
    message - binary message.
    key - binary key of length (m + n - 1) + w*n
          where w - message block count. Each block contains m = 1024 bits
                n - blocks count
    """
    m = MESSAGE_BLOCK_SIZE  # 1Mbit - Message block size
    n = KEY_BLOCK_SIZE  # Key bits per message block

    # Split message into blocks, filling with 0 if needed
    if not message.size % m == 0:
        M = np.hstack((message.astype(np.uint8),
                      np.zeros(m - (message.size % m), dtype=np.uint8)))
    else:
        M = np.array(message, dtype=np.uint8)
    M = M.reshape(M.size / m, m)
    w = M.shape[0]  # number of blocks in message

    # Check key size
    if toeplitz_seed_and_key.size != (m + n - 1 + w * n):
        raise Exception("Wrong key size, got {}, need {}".format(toeplitz_seed_and_key.size, m + n - 1 + w * n))

    S, key = toeplitz_seed_and_key[:m + n - 1], toeplitz_seed_and_key[m + n - 1:]
    R = key.reshape(key.size / n, n)
    w = R.shape[0]

    vertical = S[:n]
    horizontal = S[n - 1:]
    toeplitz_row_source = np.hstack((vertical[::-1], horizontal[1:]))
    result = np.zeros((w, n), dtype=int)
    for j in xrange(w):
        result[j][0] = (toeplitz_row_source[vertical.size-1:].dot(M[j]) % 2 + R[j][0]) % 2
        for i in xrange(1, vertical.size):
            result[j][i] = (toeplitz_row_source[vertical.size-1-i:-i].dot(M[j]) % 2 + R[j][i]) % 2
    result = np.ravel(result)

    return result


KEYSIZE = 32
BLOCKSIZE = 8

SEQ_MAC = (
    0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3, 4, 5, 6, 7,
    7, 6, 5, 4, 3, 2, 1, 0
)

SBOX = ((12, 4, 6, 2, 10, 5, 11, 9, 14, 8, 13, 7, 0, 3, 15, 1),
        (6, 8, 2, 3, 9, 10, 5, 12, 1, 14, 4, 7, 11, 13, 0, 15),
        (11, 3, 5, 8, 2, 15, 10, 13, 14, 1, 7, 4, 12, 9, 6, 0),
        (12, 8, 2, 1, 13, 4, 15, 6, 7, 0, 10, 5, 3, 14, 9, 11),
        (7, 15, 5, 10, 8, 1, 6, 13, 0, 9, 3, 14, 11, 4, 2, 12),
        (5, 13, 15, 6, 9, 2, 12, 10, 11, 7, 8, 1, 4, 3, 14, 0),
        (8, 14, 2, 5, 6, 9, 1, 12, 15, 4, 11, 0, 13, 10, 3, 7),
        (1, 7, 14, 13, 0, 5, 8, 3, 4, 15, 10, 6, 9, 12, 11, 2))


def _K(s, _in):
    """ S-box substitution
    :param s: S-box
    :param _in: 32-bit word
    :return: substituted 32-bit word
    """
    return (
        (s[0][(_in >> 0) & 0x0F] << 0) +
        (s[1][(_in >> 4) & 0x0F] << 4) +
        (s[2][(_in >> 8) & 0x0F] << 8) +
        (s[3][(_in >> 12) & 0x0F] << 12) +
        (s[4][(_in >> 16) & 0x0F] << 16) +
        (s[5][(_in >> 20) & 0x0F] << 20) +
        (s[6][(_in >> 24) & 0x0F] << 24) +
        (s[7][(_in >> 28) & 0x0F] << 28)
    )


def block2n(data):
    """ Convert block to 1 64 bit int
    """
    return Struct("<Q").unpack(data)[0]


def n2block(data):
    """ Convert 1 64 bit int to block
    """
    return Struct("<Q").pack(data)


def block2ns(data):
    """ Convert block to N1 and N2 integers
    """
    data = [ord(c) for c in data]
    return (
        data[0] | data[1] << 8 | data[2] << 16 | data[3] << 24,
        data[4] | data[5] << 8 | data[6] << 16 | data[7] << 24,
    )


def ns2block(ns):
    """ Convert N1 and N2 integers to 8-byte block
    """
    n1, n2 = ns
    return "".join(chr(i) for i in (
        (n2 >> 0) & 255, (n2 >> 8) & 255, (n2 >> 16) & 255, (n2 >> 24) & 255,
        (n1 >> 0) & 255, (n1 >> 8) & 255, (n1 >> 16) & 255, (n1 >> 24) & 255,
    ))


def addmod(x, y, mod=2 ** 32):
    """ Modulo adding of two integers
    """
    r = x + y
    return r if r < mod else r - mod


def strxor(a, b):
    """ XOR of two strings
    This function will process only shortest length of both strings,
    ignoring remaining one.
    """
    mlen = min(len(a), len(b))
    a, b, xor = bytearray(a), bytearray(b), bytearray(mlen)
    for i in xrange(mlen):
        xor[i] = a[i] ^ b[i]
    return str(xor)


def _shift11(x):
    """ 11-bit cyclic shift
    """
    return ((x << 11) & (2 ** 32 - 1)) | ((x >> (32 - 11)) & (2 ** 32 - 1))


def validate_key(key):
    if len(key) != KEYSIZE:
        raise ValueError("Invalid key size")


def validate_iv(iv):
    if len(iv) != BLOCKSIZE:
        raise ValueError("Invalid IV size")


def xcrypt(seq, sbox, key, ns):
    """ Perform full-round single-block operation
    :param seq: sequence of K_i S-box applying (either encrypt or decrypt)
    :param sbox: S-box parameters to use
    :type sbox: str, SBOXES'es key
    :param str key: 256-bit encryption key
    :param ns: N1 and N2 integers
    :type ns: (int, int)
    :return: resulting N1 and N2
    :rtype: (int, int)
    """
    s = sbox
    w = bytearray(key)
    x = [
        w[0 + i * 4] |
        w[1 + i * 4] << 8 |
        w[2 + i * 4] << 16 |
        w[3 + i * 4] << 24 for i in xrange(8)
    ]
    n1, n2 = ns
    for i in seq:
        n1, n2 = _shift11(_K(s, addmod(n1, x[i]))) ^ n2, n1
    return n1, n2


def _pad(data):
    """ _pad the data, to make it multiple of BLOCKSIZE
    :param str data: data to _pad
    :return: size of original data and its _padded version
    :rtype: (int, str)
    If data is already multiple of BLOCKSIZE, then nothing added.
    Data is _padded with zeros with leading one bit.
    """
    size = len(data)
    if size < BLOCKSIZE:
        pad_length = BLOCKSIZE - size
    elif size % BLOCKSIZE == 0:
        pad_length = 0
    else:
        pad_length = BLOCKSIZE - size % BLOCKSIZE

    if pad_length > 1:
        data = data + b"\x80" + b"\x00" * (pad_length - 1)
    elif pad_length == 1:
        data = data + b"\x80"

    return size, data


class MAC(object):
    """ MAC mode of operation
    >>> m = MAC(key)
    >>> m.update("some data")
    >>> m.update("another data")
    >>> m.digest()[:4].encode("hex")
    """

    def __init__(self, key, iv=8 * '\x00', sbox=SBOX):
        """
        :param key: authentication key
        :type key: str, 32 bytes
        :param iv: initialization vector
        :type iv: str, BLOCKSIZE length
        :param sbox: S-box parameters to use
        :type sbox: str, SBOXES'es key
        """
        validate_key(key)
        validate_iv(iv)
        self.key = key
        self.iv = iv
        self.sbox = sbox
        self.data = ""

    def update(self, data):
        """ Append data that has to be authenticated
        """
        self.data += data

    def digest(self):
        """ Get MAC tag of supplied data
        You have to provide at least single byte of data.
        """
        if not self.data:
            raise ValueError("No data processed")
        size, data = _pad(self.data)
        prev = block2ns(self.iv)

        r = ns2block(xcrypt(SEQ_MAC, self.sbox, self.key, prev)[::-1])
        r_bit_1 = (ord(r[0]) >> 7) & 1
        r_int = block2n(r)

        b = (0b11011 << 58)

        r_s = (r_int >> 1) & ~(1 << 63)
        if not r_bit_1:
            k1 = r_s
        else:
            k1 = r_s ^ b

        k1_bit_1 = k1 & 1
        k1_s = (k1 >> 1) & ~(1 << 63)
        if not k1_bit_1:
            k2 = k1_s
        else:
            k2 = k1_s ^ b

        if (size % BLOCKSIZE) == 0:
            k = k1
        else:
            k = k2

        for i in xrange(0, len(data), BLOCKSIZE):
            if i < (len(data) - BLOCKSIZE):
                prev = xcrypt(
                    SEQ_MAC, self.sbox, self.key, block2ns(strxor(
                        data[i:i + BLOCKSIZE],
                        ns2block(prev),
                    )),
                )[::-1]
            else:
                prev = xcrypt(
                    SEQ_MAC, self.sbox, self.key, block2ns(
                        strxor(
                            strxor(
                                data[i:i + BLOCKSIZE],
                                ns2block(prev)
                            ),
                            n2block(k)
                        )),
                )[::-1]
        return ns2block(prev)

    def __call__(self, data):
        self.data = data
        return self.digest()


def gost_algo(message, key):
    """
    GOST 34.13-2015 MAC algorhythm implementation
    message - binary messag.
    key - 256bit binary key
    """
    key_bytes = np.packbits(key).tobytes()
    msg_bytes = np.packbits(message).tobytes()

    gost_mac = MAC(key_bytes)
    digest_bytes = gost_mac(msg_bytes)

    return np.unpackbits(np.frombuffer(digest_bytes, dtype=np.uint8))


def get_mac(M, K, unihash):
    """
    Generate message auth code for given message and key
    Using UniHash algorhythm if unihash is True, else GOST MAC
    """
    return unihash_algo(M, K) if unihash else gost_algo(M, K)


def check_mac(M, K, unihash, MAC):
    """
    Check message auth code for given message, key and existing MAC
    Using UniHash algorhythm if unihash is True, else GOST MAC
    """
    return (get_mac(M, K, unihash) == MAC).all()


def main():
    parser = ErrorPrintArgumentParser(description='Authentication algorithm.')
    parser.add_argument('message_path', help='Path to file containing message')
    parser.add_argument('key_path', help='Path to file containing key for hashing algorithm')
    parser.add_argument('-g', '--get-hash', metavar='output_file_name',
                        help='Path to output file for hash. Perform hashing if specified')
    parser.add_argument('-c', '--check-hash', metavar='hash_file_name',
                        help='Path to file containing hash. Perform check if specified')
    parser.add_argument('-u', '--unihash', metavar='toeplitz_seed',
                        help='If specified, use unihash hor hashing, otherwise use gost. ' \
                             'File containing toeplitz matrix seed')
    args = parser.parse_args()

    if args.get_hash and args.check_hash:
        raise Exception("Only one of --get-hash and --check-hash must be provided")

    message = array_from_file(args.message_path)[0]
    key = array_from_file(args.key_path)[0]

    if not args.unihash and key.size != 256:
        raise ValueError("Incorrect key size for GOST algorithm")

    if args.unihash:
        toeplitz_seed = array_from_file(args.unihash)[0]
        key = np.hstack((toeplitz_seed, key))

    if args.get_hash:
        hash = get_mac(message, key, args.unihash)
        array_to_file([hash], args.get_hash)
        print 1
    elif args.check_hash:
        hash = array_from_file(args.check_hash)
        result = check_mac(message, key, args.unihash, hash)
        print int(result)
    else:
        raise Exception("One of --get-hash and --check-hash must be provided")


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        sys.stderr.write(u"{}\n{}\n".format(-1, exc))
        sys.exit(1)
