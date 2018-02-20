#!/usr/bin/python
# coding: utf-8

import sys
import os
import math
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.abspath('..'))

from common.funcs import h
from common.parseargs import ErrorPrintArgumentParser
from common.files import array_from_file
from common.files import array_to_file


def verify_input(mu, nu, lmbd):
    if nu >= mu:
        raise ValueError("Intensity of decoy state must be less then "
                         "intensity of the signal status photon beam")
    if lmbd >= nu / 2 or (lmbd + nu) >= mu:
        raise ValueError("Intensity of vacuum state is incorrect")


def get_key_length(n, Epa, r, t, mu, nu, lmbd, Nmu, nmu, Nnu, nnu, Nl, nl, emu):
    """
    :param: Epa   Required reliability level
    :param: Lmin  Minimal key length
    :param: r     Total syndromes length
    :param: t     Total hash-tag length
    :param: mu    Intensity of the signal status photon beam
    :param: nu    Intensity of decoy state
    :param: lmbd  Intensity of vacuum state
    :param: Nmu   Number of send singal states
    :param: nmu   Number of registered signal states
    :param: Nnu   Number of send decoy states
    :param: nnu   Number of registered decoy states
    :param: Nl    Number of send vacuum states
    :param: nl    Number of registered vacuum states
    :param: emu   Error rate in signal states
    :return:      Length of secret key
    """
    d = -norm.ppf(Epa ** 4 / 8)

    Qmu = nmu / Nmu
    Qnu = nnu / Nnu
    Ql = nl / Nl
    Qmuhat = Qmu + d * np.sqrt(Qmu * (1 - Qmu) / Nmu)
    QnuhatU = nnu / Nnu + d * np.sqrt(Qnu * (1 - Qnu) / Nnu)
    QnuhatL = nnu / Nnu - d * np.sqrt(Qnu * (1 - Qnu) / Nnu)
    QlhatU = nl / Nl + d * np.sqrt(Ql * (1 - Ql) / Nl)
    QlhatL = nl / Nl - d * np.sqrt(Ql * (1 - Ql) / Nl)

    Y0L = (nu * QlhatL * np.exp(lmbd) - lmbd * QnuhatU * np.exp(nu)) / (nu - lmbd)

    if Y0L < 0:
        Y0L = 0

    Q1hat = mu * np.exp(-mu) * (
        QnuhatL * np.exp(nu) - QlhatU * np.exp(lmbd) -
        (Qmuhat * np.exp(mu) - Y0L) * (nu * nu - lmbd * lmbd) / (mu * mu)) /\
        (nu * (1 - nu / mu) - lmbd * (1 - lmbd / mu))

    eta1hat = Q1hat / Qmuhat

    k1hat = eta1hat - d * np.sqrt(eta1hat * (1 - eta1hat) / n)

    Emuhat = emu + d * np.sqrt(emu * (1 - emu) / n)
    E1hat = (Emuhat * Qmuhat - Y0L * np.exp(-mu) / 2) / Q1hat

    e1hat = E1hat + d * np.sqrt(E1hat * (1 - E1hat) / n)
    if math.isnan(e1hat):
        return -1
    key_length = k1hat * n * (1 - h(e1hat)) - r - t - 5 * np.log2(1 / Epa)

    return int(key_length)


def ceil_to_power_of_two(x):
    return 1 << (x - 1).bit_length()


def toeplitz_permutation(K, l, S):
    """
    :param K: Verified key
    :param l: Secret key length
    :param S: Toeplitz matrix seed
    :return:  Secret key
    """
    vertical = S[:l]
    horizontal = S[l - 1:l + K.size - 1]

    desired_length = ceil_to_power_of_two(K.size * 2)
    toeplitz_seed_filler_len = desired_length - vertical.size - horizontal[::-1][:-1].size
    toeplitz_seed = np.hstack((vertical, np.zeros((toeplitz_seed_filler_len,)), horizontal[::-1][:-1])).astype(np.int)
    padded_key = np.hstack((K, np.zeros(desired_length - K.size, ))).astype(np.int)

    result = np.around(np.fft.ifft(np.fft.fft(toeplitz_seed) * np.fft.fft(padded_key)).real).astype(np.int) % 2

    return result[:l]


def processing(K, Epa, Lmin, r, t, mu, nu, lmbd, Nmu, nmu, Nnu, nnu, Nl, nl, emu, S):
    """
    :param: K     Verified key of length n
    :param: Epa   Required reliability level
    :param: Lmin  Minimal key length
    :param: r     Total syndromes length
    :param: t     Total hash-tag length
    :param: mu    Intensity of the signal status photon beam
    :param: nu    Intensity of decoy state
    :param: lmbd  Intensity of vacuum state
    :param: Nmu   Number of send singal states
    :param: nmu   Number of registered signal states
    :param: Nnu   Number of send decoy states
    :param: nnu   Number of registered decoy states
    :param: Nl    Number of send vacuum states
    :param: nl    Number of registered vacuum states
    :param: emu   Error rate in signal states
    :param: S     Toeplitz seed of length 2n
    :return:      Secret key
    """
    verify_input(mu, nu, lmbd)

    l = get_key_length(K.size, Epa, r, t, mu, nu, lmbd,
                       Nmu, nmu, Nnu, nnu, Nl, nl, emu)
    if l < Lmin:
        raise Exception("Key generation is not safe")

    if S.size < (l + K.size - 1):
        raise Exception("Key string length not enough to generate a Toeplitz matix")

    return toeplitz_permutation(K, l, S)


def main():
    parser = ErrorPrintArgumentParser(description='File generator for authorization.py.')
    parser.add_argument('--epa', default=1e-12, type=float, help='Required reliability level')
    parser.add_argument('--lmin', default=894, type=float, help='Minimal key length')
    parser.add_argument('-r', '--syndromes-len', default=314496.0, type=float, help='Total syndromes length')
    parser.add_argument('-t', '--hash-tags-len', default=50.0, type=float, help='Total hash-tag length')
    parser.add_argument('--mu', default=0.5, type=float, help='Intensity of the signal status photon beam')
    parser.add_argument('--nu', default=0.1, type=float, help='Intensity of decoy state')
    parser.add_argument('--lmbd', default=0.01, type=float, help='Intensity of vacuum state')
    parser.add_argument('--nmu-send', default=700000000.0, type=float, help='Number of send singal states')
    parser.add_argument('--nmu-reg', default=2032904.0, type=float, help='Number of registered signal states')
    parser.add_argument('--nnu-send', default=700000000.0, type=float, help='Number of send decoy states')
    parser.add_argument('--nnu-reg', default=476021.0, type=float, help='Number of registered decoy states')
    parser.add_argument('--nl-send', default=700000000.0, type=float, help='Number of send vacuum states')
    parser.add_argument('--nl-reg', default=125722.0, type=float, help='Number of registered vacuum states')
    parser.add_argument('--emu', default=0.0452813, type=float, help='Error rate in signal states')
    parser.add_argument('-k', '--key-path', default='key.txt', help='Path to file containing key')
    parser.add_argument('-s', '--s-path', default='toeplitz_seed.txt',
                        help='Path to file containing data for Toeplitz matrix generation')
    parser.add_argument('-o', '--output', default='result.txt',
                        help='Path to output file')
    args = parser.parse_args()

    K = array_from_file(args.key_path)[0]
    S = array_from_file(args.s_path)[0]

    result = processing(K, args.epa, args.lmin,
                        args.syndromes_len, args.hash_tags_len,
                        args.mu, args.nu, args.lmbd,
                        args.nmu_send, args.nmu_reg, args.nnu_send, args.nnu_reg, args.nl_send, args.nl_reg,
                        args.emu, S)

    array_to_file([result], args.output)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        sys.stderr.write(u"{}\n{}\n".format(-1, exc))
        sys.exit(1)
