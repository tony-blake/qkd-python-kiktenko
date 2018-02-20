#!/usr/bin/python
# coding: utf-8

import sys
import os
import numpy as np
from numpy import zeros, ceil, floor, copy, log, exp, mean, sign
import time
import random

from codes import codes_from_file

sys.path.insert(0, os.path.abspath('..'))

from common.funcs import h
from common.generate import generate_key
from common.generate import generate_key_zeros
from common.generate import add_errors
from common.parseargs import ErrorPrintArgumentParser
from common.files import array_from_file
from common.files import array_to_file
from common.files import int_array_to_file


def int2bin(v):
    return np.fromstring(b''.join(map(chr,
                                  map(int,
                                      bin(v)[2:]))),
                         dtype=np.uint8)


def bin2int(v):
    return int(np.array2string(v, separator='')[1:-1], base=2)


def get_hash(x, k):
    k = bin2int(k)

    p = 2**50 - 27

    x = np.hstack((np.zeros(49 - (x.size % 49), dtype=np.uint8),
                   x.ravel()))
    N = x.size / 49
    M = x.reshape((N, 49))
    M = np.hstack((np.zeros((M.shape[0], 1)),
                   M)).astype(np.int8)

    y = 1
    for i in range(N):
        M_i = bin2int(M[i])
        y = (y * k + M_i) % p

    res = int2bin(y)
    assert res.size <= 50
    if res.size == 50:
        return res
    else:
        return np.hstack(([0] * (50 - res.size), res))


def generate_sp(k_n, s_n, p_n, p_list=None):
    '''
    Generates positions of shortened 's_pos', punctured 'p_pos' and key 'k_pos' symbols together with values of shortened bits
    's_val'. Punctured symbols are token from 'p_list' if it's not None, if it is None then they are token from the whole key.
    '''
    n = k_n + s_n + p_n  # length of total key
    all_pos = range(int(n))  # array of all indices
    if p_list is None:
        punct_list = all_pos
    elif p_n <= len(p_list):
        punct_list = p_list
    else:
        punct_list = all_pos  # taking all postions if it not enough elements in p_list
    if p_n > len(punct_list):
        print 'Error with dimensions p_n=', p_n, 'but length of punct_list is', len(punct_list)
    p_pos = np.sort(random.sample(punct_list, p_n))
    all_pos1 = np.setdiff1d(all_pos, p_pos)
    s_pos = np.sort(random.sample(all_pos1, s_n))
    k_pos = np.setdiff1d(all_pos1, s_pos)
    return s_pos, p_pos, k_pos


def extend_sp(x, s_pos, p_pos, k_pos):
    '''
    Extend initial key x with shortened and punctured bits
    '''
    k_n = len(k_pos)
    s_n = len(s_pos)
    p_n = len(p_pos)
    if len(x) != len(k_pos):
        print "Error with dimensions in key and k_pos"
    n = k_n + s_n + p_n  # length of extended key
    x_ext = generate_key(n)
    if s_n > 0:
        x_ext[s_pos] = 0
    if p_n > 0:
        x_ext[p_pos] = generate_key(p_n)
    x_ext[k_pos] = x
    return x_ext


def find_qber(f, R):
    '''
    Finds qber for desired efficiency
    '''
    df_min = 1
    for qber in np.arange(0.001, 0.11, 0.001):
        f_cur = (1 - R) / h(qber)
        df = abs(f - f_cur)
        if df < df_min:
            qber_rslt = qber
            df_min = df
    return qber_rslt


def encode_syndrome(x, s_y_joins):
    """
    Encode vector 'x' with sparse matrix, characterized by 's_y_joins'
    """
    m = len(s_y_joins)
    s = generate_key_zeros(m)

    for k in range(m):
        s[k] = (sum(x[s_y_joins[k]]) % 2)
    return np.array(s)


def g_func_piecewise(x):
    """
    Approximation of log(1+exp(-x)) by linear interpolation between points
    """
    if x < 0.5:
        return -x / 2 + 0.7
    elif x < 1.6:
        return -x / 4 + 0.575
    elif x < 2.2:
        return -x / 8 + 0.375
    elif x < 3.2:
        return -x / 16 + 0.2375
    elif x < 4.4:
        return -x / 32 + 0.1375
    else:
        return 0


def g_func_table(x):
    """
    Aproximation of log(1+exp(-x)) by tabulated values
    """
    if x < 0.196:
        return 0.65
    elif x < 0.433:
        return 0.55
    elif x < 0.71:
        return 0.45
    elif x < 1.105:
        return 0.35
    elif x < 1.508:
        return 0.25
    elif x < 2.252:
        return 0.15
    elif x < 4.5:
        return 0.05
    else:
        return 0


def h_func_exact(x):
    return np.log(np.abs(np.exp(x) - 1))


def h_func(x):
    if x < -3:
        return 0
    elif x < -0.68:
        return -0.25 * x - 0.75
    elif x < -0.27:
        return -2 * x - 1.94
    elif x < 0:
        return -8 * x - 3.56
    elif x < 0.15:
        return 16 * x - 4
    elif x < 0.4:
        return 4 * x - 2.2
    elif x < 1.3:
        return 2 * x - 1.4
    else:
        return x - 0.1


def core_func(x, y, mode=1):
    '''
    Core function () for computation of LLRs.
    'x' and 'y' are arguments.
    'mode' is approximation method: 0 - piecewise, 1 - table, 2 - exact,
    '''
    if mode == 0:
        return np.sign(x) * np.sign(y) * min(abs(x), abs(y)) + g_func_piecewise(abs(x + y)) - g_func_piecewise(
            abs(x - y))  # piecewise
    elif mode == 1:
        return sign(x) * sign(y) * min(abs(x), abs(y)) + g_func_table(abs(x + y)) - g_func_table(abs(x - y))  # table
    else:
        return sign(x) * sign(y) * min(abs(x), abs(y)) + log(1 + exp(-abs(x + y))) - log(1 + exp(-abs(x - y)))  # exact


def decode_syndrome_minLLR(y, s, s_y_joins, y_s_joins, qber_est, s_pos, p_pos, k_pos, r_start=None, max_iter=300,
                           x=None, show=1, discl_n=20, n_iter_avg_window=5):
    """
    Decode vector 'y' according to syndrome 's'.
    Limit iterations to 'n_iter' (default 10)
    s_y_joins[i] (y_s_joins[i]) contains symbol(check) nodes connected to i-th check(symbol) node
    """
    if not qber_est < 0.5:  # Adequate QBER check
        raise ValueError('Aprior error probability must be less than 1/2')

    m = len(s_y_joins)
    n = len(y_s_joins)
    p_n = len(p_pos)
    s_n = len(s_pos)
    v_pos = list(set(p_pos) | set(k_pos))

    # Zeroing
    M = np.zeros((m, n))  # Array of messages from symbol nodes to check nodes
    sum_E_abs_mean_hist = []  # Array for mean values of LLRs
    n_iter = 0  # Iteration counter

    # Setting initial LLRs:
    if r_start is None:
        r = zeros(n)
        if s_n > 0:
            r[s_pos] = (1 - 2 * y[s_pos]) * 1000
        if p_n > 0:
            r[p_pos] = 0
        r[k_pos] = (1 - 2 * y[k_pos]) * np.log((1 - qber_est) / qber_est)
    else:
        r = r_start
        if s_n > 0:
            r[s_pos] = (1 - 2 * y[s_pos]) * 1000

    for j in xrange(m):  # Setting initial messages from symbol nodes to check nodes
        M[j, :] = r

    while n_iter < max_iter:  # Main cycle
        # Part 1: from check nodes to symbol nodes
        E = np.zeros((m, n))  # Array of messages from check nodes to symbol nodes
        for j in xrange(m):  # For all check nodes
            M_cur = M[j][s_y_joins[j]]
            M_cur_n = len(M_cur)  # All symbol nodes that are connected to current check node and their number
            n_zeros = list(M_cur).count(0.0)  # number of zero LLRs
            if n_zeros > 1:  # If check node is dead
                E[j, s_y_joins[j]] = np.zeros(M_cur_n)  # No messages
            elif n_zeros == 1:  # If current check node has one punctured symbol
                E_cur = np.zeros(M_cur_n)  # All messages are initializrd with zeros
                M_cur = list(M_cur)
                zero_ind = M_cur.index(0.0)
                M_cur.pop(zero_ind)  # Excluding zero message
                LS = M_cur[0]
                for k in range(1, M_cur_n - 1):  # Accumulation of the message
                    LS = core_func(LS, M_cur[k])
                E_cur[zero_ind] = LS
                E[j, s_y_joins[j]] = E_cur  # Filling with nonzero message
            elif n_zeros == 0:  # all messages are non zero
                LS = M_cur[0]
                for k in range(1, M_cur_n):
                    LS = core_func(LS, M_cur[k])
                E_cur = zeros(M_cur_n)
                for i1 in range(0, M_cur_n):
                    E[j][s_y_joins[j][i1]] = (1 - 2 * s[j]) * (
                        h_func(M_cur[i1] + LS) - h_func(M_cur[i1] - LS) - LS)  # Computation of messages

        # Part 2: from symbol nodes to check nodes
        sum_E = E.sum(axis=0) + r  # Array of sums of messages to symbol nodes (LLRs)
        z = (1 - np.sign(sum_E)) / 2  # Current decoded message

        if (s == encode_syndrome(z, s_y_joins)).all():  # If syndrome is correct
            if np.count_nonzero(z == x) != n and show > 1:
                print "Convergence error, error positions:"
                print '\n', np.nonzero((z + x) % 2)
            if show > 1:
                print 'Done in ', n_iter, 'iters, matched bits:', np.count_nonzero(z == x), '/', n
            return z, None, sum_E, n_iter
        if show > 2:
            print 'Matched bits:', np.count_nonzero(z == x), '/', n, 'Mean LLR magnitude:', mean(abs(sum_E[v_pos])), \
                  'Averaged mean LLR magnitude:', sum(sum_E_abs_mean_hist[max(0, n_iter - n_iter_avg_window):n_iter]) / (
                  min(n_iter, n_iter_avg_window) + 10 ** (-10))

        # Check for procedure stop

        sum_E_abs = list(abs(sum_E))
        sum_E_abs_mean_hist.append(mean(list(abs(sum_E[v_pos]))))

        if n_iter == n_iter_avg_window - 1:
            sum_E_mean_avg_old = mean(sum_E_abs_mean_hist)
        if n_iter >= n_iter_avg_window:
            sum_E_mean_avg_cur = sum_E_mean_avg_old + (sum_E_abs_mean_hist[n_iter] - sum_E_abs_mean_hist[
                n_iter - n_iter_avg_window]) / n_iter_avg_window
            if sum_E_mean_avg_cur <= sum_E_mean_avg_old:
                minLLR_inds = []
                maxLLR = max(sum_E_abs)
                for cnt in range(discl_n):
                    ind = sum_E_abs.index(min(sum_E_abs))
                    minLLR_inds.append(ind)
                    sum_E_abs[ind] += maxLLR
                return None, minLLR_inds, sum_E, n_iter
            else:
                sum_E_mean_avg_old = sum_E_mean_avg_cur

        # Calculating messages from symbol nodes to check nodes
        M = -E + sum_E

        n_iter += 1

    minLLR_inds = []
    maxLLR = max(sum_E_abs)
    for cnt in range(discl_n):
        ind = sum_E_abs.index(min(sum_E_abs))
        minLLR_inds.append(ind)
        sum_E_abs[ind] += maxLLR
    return None, minLLR_inds, sum_E, n_iter


def perform_ec(x, y, s_y_joins, y_s_joins, qber_est, s_n, p_n, punct_list=None, discl_n=20, show=0):
    n = len(y_s_joins)

    s_pos, p_pos, k_pos = generate_sp(n - s_n - p_n, s_n, p_n, p_list=punct_list)

    x_ext = extend_sp(x, s_pos, p_pos, k_pos)
    y_ext = extend_sp(y, s_pos, p_pos, k_pos)

    k_pos_in = copy(k_pos)  # For final exclusion

    s_x = encode_syndrome(x_ext, s_y_joins)
    s_y = encode_syndrome(y_ext, s_y_joins)

    s_d = (s_x + s_y) % 2
    key_sum = (x_ext + y_ext) % 2

    e_pat_in = generate_key_zeros(n)

    e_pat, minLLR_inds, sum_E, n_iter = decode_syndrome_minLLR(e_pat_in, s_d, s_y_joins, y_s_joins, qber_est, s_pos,
                                                               p_pos, k_pos, max_iter=300, x=key_sum, show=show,
                                                               discl_n=discl_n, n_iter_avg_window=5)

    add_info = 0
    com_iters = 0

    while e_pat is None:
        if show > 1:
            print 'Additional iteration', 'p_n', len(p_pos), 's_n', len(s_pos), 'k_n', len(k_pos), len(
                list(set(s_pos) & set(minLLR_inds)))
        e_pat_in[minLLR_inds] = (x_ext[minLLR_inds] + y_ext[minLLR_inds]) % 2
        s_pos = list(set(s_pos) | set(minLLR_inds))
        k_pos = list(set(k_pos) - set(minLLR_inds))
        if p_pos is not None:
            p_pos = list(set(p_pos) - set(minLLR_inds))
        e_pat, minLLR_inds, sum_E, n_iter = decode_syndrome_minLLR(e_pat_in, s_d, s_y_joins, y_s_joins, qber_est, s_pos,
                                                                   p_pos, k_pos, r_start=None, max_iter=300,
                                                                   x=key_sum, show=show, discl_n=discl_n,
                                                                   n_iter_avg_window=5)
        add_info += discl_n
        com_iters += 1

    x_dec = (x_ext[k_pos_in] + e_pat[k_pos_in]) % 2
    error_number = np.count_nonzero(e_pat[k_pos_in])

    ver_check = (x_dec == y).all()
    if not ver_check and show > 1:
        print "VERIFICATION ERROR"

    return add_info, com_iters, x_dec, ver_check, error_number


def get_sigma_pi(qber, f, R):
    pi = (f * h(qber) - 1 + R) / (f * h(qber) - 1)
    sigma = 1 - (1 - R) / f / h(qber)
    return max(0, sigma), max(0, pi)


def choose_sp(qber, f, R_range, n, d):
    '''
    Returns rate and numbers of shortened and punctured bits for estimated QBER 'qber', initial efficiency 'f',
    codes with rates 'R_range', frame length 'n', and number of total number shortend and punctured bits 'd'
    NOTE: d = 162 corresponds to n = 1944 codes.
    '''
    Rc = 1 - h(qber) * f
    for R in R_range:
        p_n = int(ceil((1 - R) * n - (1 - Rc) * (n - d)))
        s_n = int(d - p_n)
        if p_n >= 0 and s_n >= 0:
            return R, s_n, p_n


def test_ec(qber, R_range, codes, n, n_tries, f=1, show=1, discl_k=1):
    R, s_n, p_n = choose_sp(qber, f, R_range, n)
    k_n = n - s_n - p_n
    m = (1 - R) * n
    code_params = codes[(R, n)]
    s_y_joins = code_params['s_y_joins']
    y_s_joins = code_params['y_s_joins']
    punct_list = code_params['punct_list']
    p_n_max = len(punct_list)
    discl_n = int(round(n * (0.0280 - 0.02 * R) * discl_k))
    qber_est = qber
    f_rslt = []
    com_iters_rslt = []
    n_incor = 0

    print "QBER = ", qber, "R =", R, "s_n =", s_n, "p_n =", p_n, '(', p_n_max, ')', 'discl_n', discl_n

    t1 = time.time()
    for i in range(n_tries):
        print i,
        x = generate_key(n - s_n - p_n)
        y = add_errors(x, qber)
        add_info, com_iters, x_dec, ver_check, error_number = perform_ec(
            x, y, s_y_joins, y_s_joins, qber_est, s_n, p_n,
            punct_list=punct_list, discl_n=discl_n, show=show)
        f_cur = float(m - p_n + add_info) / (n - p_n - s_n) / h(qber)
        f_rslt.append(f_cur)
        com_iters_rslt.append(com_iters)
        if not ver_check:
            n_incor += 1
    t2 = time.time()
    print 'Throughput:', float(n_tries * k_n) / (t2 - t1), 'bit/s, Mean f:', np.mean(f_rslt), 'Mean com iters', np.mean(
        com_iters_rslt)
    return np.mean(f_rslt), np.std(f_rslt), np.mean(com_iters_rslt), np.std(
        com_iters_rslt), R, s_n, p_n, p_n_max, k_n, discl_n, float(n_incor) / n_tries


def process(frame_len, block_count, qber_est, discl_k, s_p_sum,
            a_sifted_key, b_sifted_key, codes, show=False):
    assert a_sifted_key.size == b_sifted_key.size

    stats = dict()
    show = 3 if show else 0
    R_range = [value['R'] for key, value in codes.items() if value['frame_len'] == frame_len]
    if len(R_range) == 0:
        raise ValueError("No available codes in codes file")

    start_time = time.time()

    R, s_n, p_n = choose_sp(qber_est, 1, R_range, frame_len, s_p_sum)

    code_params = codes[(R, frame_len)]
    s_y_joins = code_params['s_y_joins']
    y_s_joins = code_params['y_s_joins']
    punct_list = code_params['punct_list']

    discl_n = int(round(frame_len * (0.0280 - 0.02 * R) * discl_k))

    block_len = frame_len - s_n - p_n

    if block_count * block_len > a_sifted_key.size:
        raise ValueError("Sifted key is not enough for error correction, needed {}, got {}",
                         block_count * block_len, a_sifted_key.size)

    consumed_sifted_key_length = 0
    syndromes_length = 0
    a_error_numbers = []
    b_original_key = []
    a_b_dec = []
    for a_key_block, b_key_block in zip(a_sifted_key[:block_count * block_len].reshape((block_count, block_len)),
                                        b_sifted_key[:block_count * block_len].reshape((block_count, block_len))):
        consumed_sifted_key_length += a_key_block.size

        add_info, com_iters, a_b_dec_block, ver_check, error_number = perform_ec(
            a_key_block, b_key_block, s_y_joins, y_s_joins, qber_est, s_n, p_n,
            punct_list=punct_list, discl_n=discl_n, show=show)

        syndromes_length += (len(s_y_joins) - p_n + add_info)
        a_error_numbers.append(error_number)
        b_original_key.append(b_key_block)
        a_b_dec.append(a_b_dec_block.astype(np.int8))

    stats['syndromes_length'] = syndromes_length
    stats['consumed_sifted_key_length'] = consumed_sifted_key_length
    stats['correction_time'] = time.time() - start_time

    block_count = len(a_b_dec)

    start_time = time.time()

    # Посчитать hash-tag для всех блоков вместе
    ver_ht_key = generate_key(50)
    a_hash_tag = get_hash(np.ravel(a_b_dec), ver_ht_key)
    b_hash_tag = get_hash(np.ravel(b_original_key), ver_ht_key)
    a_hash_tags_length = a_hash_tag.size

    hash_check_result = (a_hash_tag == b_hash_tag).all()

    # Если хеш всего ключа совпал, то на обоих сторонах имеем верифицированные ключи
    if hash_check_result:
        a_decoding_mask = [0] * block_count

        stats['hash_tags_length'] = a_hash_tags_length
        stats['verification_time'] = time.time() - start_time
        return a_b_dec, b_original_key, a_decoding_mask, a_error_numbers, stats

    # Если хеш всего ключа не совпал, то на обоих сторонах считаем хеши каждого блока и сравниваем
    else:
        a_ver_key = []
        b_ver_key = []
        a_decoding_mask = []
        for a_b_dec_block, b_original_block in zip(a_b_dec, b_original_key):
            ver_ht_key = generate_key(50)
            a_hash_tag = get_hash(a_b_dec_block, ver_ht_key)
            b_hash_tag = get_hash(b_original_block, ver_ht_key)
            a_hash_tags_length += a_hash_tag.size
            if (a_hash_tag == b_hash_tag).all():
                a_ver_key.append(a_b_dec_block)
                b_ver_key.append(b_original_key)
                a_decoding_mask.append(0)
            else:
                a_decoding_mask.append(1)

        stats['hash_tags_length'] = a_hash_tags_length
        stats['verification_time'] = time.time() - start_time
        return a_ver_key, b_ver_key, a_decoding_mask, a_error_numbers, stats


def main():
    parser = ErrorPrintArgumentParser(description='Error correction alghoritm')
    parser.add_argument('frame_len', help='Frame length in LDPC code', type=int)
    parser.add_argument('block_count', help='Block count for processing', type=int)
    parser.add_argument('QBER_est', help='Estimated QBER', type=float)
    parser.add_argument('discl_k', help='Additional bit disclosure parameter', type=float)
    parser.add_argument('s_p_sum', help='Count of shortened and punctured bits', type=int)
    parser.add_argument('sifted_key_a', help='Path to file containing sifted key A')
    parser.add_argument('sifted_key_b', help='Path to file containing sifted key B')
    parser.add_argument('codes', help='Path to file containing verification matrixes')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable additional'
                                                                     'output')
    args = parser.parse_args()

    a_sifted_key = array_from_file(args.sifted_key_a)[0]
    b_sifted_key = array_from_file(args.sifted_key_b)[0]
    codes = codes_from_file(args.codes)
    verbose = args.verbose

    a_ver_key, b_ver_key, a_decoding_mask, a_error_numbers, stats = process(
        args.frame_len, args.block_count, args.QBER_est, args.discl_k, args.s_p_sum,
        a_sifted_key, b_sifted_key, codes, verbose)

    array_to_file(a_ver_key, 'ver_key_a.txt')
    array_to_file(b_ver_key, 'ver_key_b.txt')
    array_to_file([a_decoding_mask], 'decoding_mask_a.txt')
    int_array_to_file(a_error_numbers, 'errors_number_a.txt')

    statistic_fp = open('error_correction_stat.log', 'w+')

    statistic_file_format = "{:<34} {:>10}\n"

    syndromes_length = stats['syndromes_length']
    hash_tags_length = stats['hash_tags_length']
    consumed_sifted_key_length = stats['consumed_sifted_key_length']
    correction_time = stats['correction_time']
    verification_time = stats['verification_time']

    efficiency_parameter = (hash_tags_length + syndromes_length) / h(args.QBER_est) / consumed_sifted_key_length
    work_time = correction_time + verification_time / 2
    a_ver_key_array = np.ravel(a_ver_key)
    b_ver_key_array = np.ravel(b_ver_key)
    equal_bit_count = np.count_nonzero(a_ver_key_array == b_ver_key_array)

    statistic_fp.writelines([
        statistic_file_format.format(title, value) for title, value in [
            ("Consumed sifted key A length", consumed_sifted_key_length),
            ("Consumed sifted key B length", consumed_sifted_key_length),
            ("Syndromes length", syndromes_length),
            ("Hash tag length", hash_tags_length),
            ("Verified key A length", a_ver_key_array.size),
            ("Verified key B length", b_ver_key_array.size),
            ("Equal bits in verified key A and B", equal_bit_count),
            ("Efficiency parameter", "{:.5f}".format(efficiency_parameter)),
            ("Work time (in seconds)", "{:.3f}".format(work_time)),
            ("Throughput (in bit/sec)", "{:.3f}".format(equal_bit_count / work_time))
        ]
    ])

    statistic_fp.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        sys.stderr.write(u"{}\n{}\n".format(-1, exc))
        sys.exit(1)
