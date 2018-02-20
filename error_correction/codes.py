#!/usr/bin/python
# coding: utf-8

import sys
import os
import random
from scipy.sparse import dok_matrix as sparse_matrix
import numpy as np

sys.path.insert(0, os.path.abspath('..'))

from common.parseargs import ErrorPrintArgumentParser


def my_arg_max(x):
    """
    Like arg_max, but if two ocurrances found select random index, insted of
    first.
    """
    xmax = x[0]
    n = len(x)
    for i in range(0, n):
        if x[i] >= xmax:
            xmax = x[i]
    s = []
    for i in range(0, n):
        if x[i] == xmax:
            s.append(i)
    return random.choice(s)


def my_arg_min(x):
    """
    Like arg_min, but if two ocurrances found selects random index, insted of
    first.
    """
    xmin = x[0]
    n = len(x)
    for i in range(0, n):
        if x[i] <= xmin:
            xmin = x[i]
    s = []
    for i in range(0, n):
        if x[i] == xmin:
            s.append(i)
    return random.choice(s)


def highest_check_degree(O, fc):
    arg_max = my_arg_max(fc[O])
    return O[arg_max]


def lowest_check_degree(O, dc):
    arg_min = my_arg_min(dc[O])
    return O[arg_min]


def expand(O, E_c, E_s):
    data_dt = np.int32

    O_expanded = []
    for i in O.ravel():
        for j in E_c[i].ravel():
            O_expanded.extend(E_s[j])

    return np.unique(np.array(O_expanded, dtype=data_dt))


def find_N(O, j, E_c, E_s):
    N_cur = np.unique(E_s[j])
    N_prev = N_cur.copy()
    N_cur = expand(N_prev, E_c, E_s)
    while np.setxor1d(N_cur, N_prev, assume_unique=True).size > 0 and \
            not np.setdiff1d(O, N_cur, assume_unique=True).size == 0:
        N_cur, N_prev = expand(N_cur, E_c, E_s), N_cur
    res = np.setdiff1d(O, N_prev)
    return res


def PEG_construct(frame_len, syndrom_len, lam, ro):
    """
    Generate H-matrix.
    """
    data_dt = np.int32
    N_1 = np.round(frame_len * (np.sum(lam[1, :] / lam[0, :]) ** (-1)))

    ds = []
    for i in range(lam.shape[1]):
        ds.extend([int(lam[0, i])] * int(np.round(N_1 * (lam[1, i] / lam[0, i]))))

    if len(ds) < frame_len:
        # fill ds with lam[0,-1] to n elements
        ds.extend([int(lam[0, -1])] * (frame_len - len(ds)))

    while len(ds) > frame_len:
        ds.pop()
    ds.sort()
    ds = np.array(ds, dtype=data_dt)

    dc = []
    for i in range(ro.shape[1]):
        dc.extend([int(ro[0, i])] * int(np.round(N_1 * (ro[1, i] / ro[0, i]))))
    if len(dc) < syndrom_len:
        dc.extend([int(ro[0, -1])] * (syndrom_len - len(dc)))  # fill dc with ro[0,-1] to m elements
    while len(dc) > syndrom_len:
        dc.pop()  # substract last elements
    if not (len(ds) == frame_len and len(dc) == syndrom_len):
        print('some error with dimensions of the matrix', len(ds), len(dc))
    dc.sort()
    dc = np.array(dc, dtype=data_dt)
    fc = dc.copy()

    O_c = np.arange(syndrom_len, dtype=data_dt)
    O_c_free = O_c.copy()

    H = sparse_matrix((syndrom_len, frame_len), dtype=data_dt)

    E_s = []
    for i in range(frame_len):
        E_s.append(np.array([], dtype=data_dt))
    E_c = []
    for i in range(frame_len):
        E_c.append(np.array([], dtype=data_dt))

    for j in range(frame_len):
        if ds[j] == 2 and j < syndrom_len - 1:  # zig-zag pattern
            i = j
            E_s[j] = np.union1d(E_s[j], np.array([i], dtype=data_dt))
            E_c[i] = np.union1d(E_c[i], np.array([j], dtype=data_dt))
            H[i, j] = 1
            fc[i] = fc[i] - 1
            if fc[i] == 0:
                O_c_free = np.setdiff1d(O_c_free, np.array([i], dtype=data_dt))

            i = j + 1
            E_s[j] = np.union1d(E_s[j], np.array([i], dtype=data_dt))
            E_c[i] = np.union1d(E_c[i], np.array([j], dtype=data_dt))
            H[i, j] = 1
            fc[i] = fc[i] - 1
            if fc[i] == 0:
                O_c_free = np.setdiff1d(O_c_free, np.array([i], dtype=data_dt))
        else:
            # First node
            i = highest_check_degree(O_c, fc)
            E_s[j] = np.union1d(E_s[j], np.array([i], dtype=data_dt))
            E_c[i] = np.union1d(E_c[i], np.array([j], dtype=data_dt))
            H[i, j] = 1
            fc[i] = fc[i] - 1
            if fc[i] == 0:
                O_c_free = np.setdiff1d(O_c_free, np.array([i], dtype=data_dt))
            # All other nodes
            for k in range(1, ds[j]):
                O_c_tmp = find_N(O_c, j, E_c, E_s)
                O_c_tmp = np.setdiff1d(O_c_tmp, E_s[j], assume_unique=True)
                i = highest_check_degree(O_c_tmp, fc)

                E_s[j] = np.union1d(E_s[j], np.array([i], dtype=data_dt))
                E_c[i] = np.union1d(E_c[i], np.array([j], dtype=data_dt))
                H[i, j] = 1
                fc[i] = fc[i] - 1
                if fc[i] == 0:
                    O_c_free = np.setdiff1d(O_c_free, np.array([i], dtype=data_dt))

    return H


def get_joins(H, frame_len, syndrome_len):
    """
    Returns joins for check and symbol nodes in parity matrix H
    """
    if type(H) == np.ndarray:
        s_y_joins = [np.where(H[j, :] > 0)[1].astype(np.int32) for j in xrange(syndrome_len)]
        y_s_joins = [np.where(H[:, i] > 0)[0].astype(np.int32) for i in xrange(frame_len)]
    else:
        s_y_joins = [np.where(H[j, :].toarray() > 0)[1].astype(np.int32) for j in xrange(syndrome_len)]
        y_s_joins = [np.where(H[:, i].toarray() > 0)[0].astype(np.int32) for i in xrange(frame_len)]
    return s_y_joins, y_s_joins


def generate_joins(frame_len, R, lam, ro):
    syndrome_len = int(round(frame_len * (1 - R)))
    H = PEG_construct(frame_len, syndrome_len, lam, ro)
    s_y_joins, y_s_joins = get_joins(H, frame_len, syndrome_len)
    punct_list, p = get_unt_punct_list(s_y_joins, y_s_joins)
    return s_y_joins, y_s_joins, syndrome_len, punct_list


def get_N2(s_y_joins, y_s_joins):
    '''
    Returns neighbors of second order (2-neighbours) for each symbol node for parity matrix with 's_y_joins' and 'y_s_joins'
    Structure of N2list: ['node index',['indices of 2-neighbours'],'number of 2-neighbours']
    NOTE: 'indices of 2-neighbours' don't include 'node index'
    '''
    n = len(y_s_joins)
    N2_list = []
    for i in range(0, n):
        N2_list.append([i, [], 0])
        Omega = set([])  # storage for 2-neighbours
        for node in y_s_joins[i]:
            Omega = Omega | set(s_y_joins[node])
        Omega.remove(i)  # removing self index
        N2_list[i][1] = Omega
        N2_list[i][2] = len(Omega)
    return N2_list


def get_unt_punct_list(s_y_joins, y_s_joins):
    '''
    Obtain list for untained puncturing 'punct_list' and its size 'p' for parity matrix with 's_y_joins' and 'y_s_joins'
    '''
    p = 0  # storage for number of punctured bits
    n = len(y_s_joins)

    N2_list = get_N2(s_y_joins, y_s_joins)
    punct_list = []
    X = set(range(0, n))
    while len(X) > 0:
        # Step 1. Search for minmal number of 2-neighbours for nodes among X
        min_n = n
        for i in X:
            cur_n = len(N2_list[i][1] & X)
            if cur_n < min_n:
                min_n = cur_n
        # Step 2. Choose random node from with minimal number cand_list
        cand_list = []  # storage for candidates
        for i in X:
            cur_n = len(N2_list[i][1] & X)
            if cur_n == min_n:
                cand_list.append(N2_list[i])
        punct_s = random.choice(cand_list)  # random choise among candidates
        # Step 3. Add chosen symbol to list and remove its 2-nighbours from X
        punct_list.append(punct_s[0])
        X = X - set([punct_s[0]])
        X = X - punct_s[1]
        p += 1
    return punct_list, p


def codes_to_file(file_name, codes):
    with open(file_name, 'w+') as fp:
        for entry in codes:
            fp.write("{} {} {}\n".format(entry['R'], entry['frame_len'], entry['syndrome_len']))
            fp.writelines(map(lambda line: ' '.join(map(str, line)) + '\n', entry['s_y_joins']))
            fp.writelines(map(lambda line: ' '.join(map(str, line)) + '\n', entry['y_s_joins']))
            fp.write("{}\n".format(' '.join(map(str, entry['punct_list']))))
            fp.write("\n")


def lines_to_array(lines):
    return map(lambda x: map(int, x.split(' ')), lines)


def codes_from_file(file_path):
    with open(file_path, 'r') as fp:
        data = fp.read()
        lines = data.split('\n')

        start_read_index = 0
        result = dict()

        # 4 here because last entry must have at least 5 lines (1 for meta, 1 for
        # s_y_joins, 1 for y_s_joins and 2 blank lines
        while start_read_index < len(lines) - 4:
            R, frame_len, syndrome_len = lines[start_read_index].split(' ')
            R = float(R)
            frame_len = int(frame_len)
            syndrome_len = int(syndrome_len)

            begin_s_y_joins_index = start_read_index + 1
            end_s_y_joins_index = begin_s_y_joins_index + syndrome_len
            s_y_joins = lines_to_array(lines[begin_s_y_joins_index:end_s_y_joins_index])

            begin_y_s_joins_index = end_s_y_joins_index
            end_y_s_joins_index = begin_y_s_joins_index + frame_len
            y_s_joins = lines_to_array(lines[begin_y_s_joins_index:end_y_s_joins_index])

            begin_punct_list_index = end_y_s_joins_index
            end_puct_list_index = begin_punct_list_index + 1
            punct_list = lines_to_array(lines[begin_punct_list_index:end_puct_list_index])

            result[(R, frame_len)] = {
                'R': R,
                'frame_len': frame_len,
                'syndrome_len': syndrome_len,
                's_y_joins': s_y_joins,
                'y_s_joins': y_s_joins,
                'punct_list': punct_list[0]
            }

            start_read_index = end_puct_list_index + 1

        return result


"""
Array of params for construction on joints. Each entry is (R, lam, ro)
SOURCE: https://arxiv.org/abs/0901.2140
"""
PARAMS = (
    (
        0.9,
        np.array([[2,       3,       5,       9,       12,      21     ],
                [0.07689, 0.28096, 0.08933, 0.19620, 0.30631, 0.05031]]),
        np.array([[50,      51     ],
                [0.95025, 0.04974]])
    ),
    (
        0.85,
        np.array([[2,       3,       4,       5,        6,       7,       9,       21],
                  [0.04528, 0.20537, 0.05878, 0.094274, 0.08454, 0.01176, 0.05137, 0.50015]]),
        np.array([[41,      42     ],
                  [0.54204, 0.45795]])
    ),
    (
        0.8,
        np.array([[2,       3,       6,       7,       8,       17,       26],
                  [0.09420, 0.18088, 0.11972, 0.08550, 0.09816, 0.07194,  0.34960]]),
        np.array([[29,      30     ],
                  [0.58807, 0.41193]])
    ),
    (
        0.75,
        np.array([[2,       3,       4,       5,       6,       7,       28,      31],
                [0.10805,0.09511,  0.01449, 0.13764, 0.10667, 0.05288, 0.01107, 0.47408]]),
        np.array([[25,      26     ],
                  [0.74161, 0.25839]])
    ),
    (
        0.7,
        np.array([[2,       3,       6,       9,       12,       25,       46,       62,      65,      73],
                [0.05343, 0.29406, 0.00896, 0.15571,   0.12189,  0.19872,  0.09572,  0.02741, 0.04056, 0.00354]]),
        np.array([[20,      21     ],
                [0.76922, 0.2307720]])
    ),
    (
        0.65,
        np.array([[2,       3,       4,       5,       9,       13,      15,      21,      51],
                  [0.10451, 0.15652, 0.08057, 0.00056, 0.12151, 0.10485, 0.10719, 0.00771, 0.31656]]),
        np.array([[2,        15,        16,      21     ],
                  [0.000578, 0.06089,   0.47001, 0.46852]])
    ),
    (
        0.6,
        np.array([[2,       3,       8,       9,       26,       27,      46,      71],
                  [0.11040, 0.20804, 0.14163, 0.14858, 0.14438,  0.08909, 0.00748, 0.15038]]),
        np.array([[2,        10,       13,      18,      19],
                  [0.00036,  0.13063,  0.31068, 0.49341, 0.064915]])
    ),
    (
        0.55,
        np.array([[2,       3,       6,       15,      16,      18,      19,      31],
                  [0.16880, 0.20994, 0.18095, 0.03846, 0.02635, 0.23454, 0.05815, 0.0828]]),
        np.array([[10,      11     ],
                  [0.27631, 0.72369]])
    ),
        (
        0.5,
        np.array([[2,     3,       4,       5,       6,       8,        9,       10,      11,      14,      15,      17,      47,       49,       55,      56,      57,      58,      59,      66],
                [0.14438, 0.19026, 0.01836, 0.00233, 0.04697, 0.053943, 0.05590, 0.01290, 0.00162, 0.06159, 0.13115, 0.01481, 0.00879,  0.00650,  0.00210, 0.00099, 0.11178, 0.06238, 0.05094, 0.02230]]),
        np.array([[10,12,13,14],
                [0.47575,0.46847,0.02952,0.02626]])
    )
)


def generate(frame_len, file_name, callback=None):
    data_to_save = []
    for i, param in enumerate(PARAMS):
        R = param[0]
        print "Code generation with R = {} started".format(R)
        s_y_joins, y_s_joins, syndrome_len, punct_list = generate_joins(frame_len, R, param[1], param[2])
        dict_to_save = {
            'R': R,
            'frame_len': frame_len,
            'syndrome_len': syndrome_len,
            's_y_joins': map(lambda x: x.tolist(), s_y_joins),
            'y_s_joins': map(lambda x: x.tolist(), y_s_joins),
            'punct_list': punct_list
        }
        data_to_save.append(dict_to_save)
        print "Code generation with R = {} completed".format(R)

        if callback is not None:
            callback(i + 1, len(PARAMS))

    codes_to_file(file_name, data_to_save)


def main():
    parser = ErrorPrintArgumentParser(description='H-matrix generation for error correction alghoritm.')
    parser.add_argument('frame_len', help='Frame length in LDPC code.', type=int)

    args = parser.parse_args()
    file_name = 'codes_' + str(args.frame_len) + '.txt'

    generate(args.frame_len, file_name)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        sys.stderr.write(u"{}\n{}\n".format(-1, exc))
        sys.exit(1)
