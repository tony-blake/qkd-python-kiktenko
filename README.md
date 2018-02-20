# Post-processing algorithms for quantum key distribution system

## 1. General description
The repository contains four proof-of-principle realizations for algorithms applied for the post-processing procedure of sifted keys in quantum key distribution (QKD) 
setup designed in [Russian Quantum Center](http://www.rqc.ru/). All the algorithms are realized in Python 2.7.

### List of the algorithms
##### Error correction
Includes simulation of symmetric blind reconciliation together with verification using universal polynomial hashing.
##### Error estimation
Performs estimation of the quantum bit error rate (QBER) using the result of the error correction procedure.
##### Privacy amplification
Performs distillation of secure key according to BB84 decoy state protocol using Toeplitz hashing.
##### Authentication
Performs calculation and check of authentication hash-tag according to Toeplitz or GOST R 34.12-2015 hashing.

## 2. Folders' contents
Here we briefly describe all the .py script presented in the repository. Please, use  `-h` or `--help` for the detailed description of input and output data.

#### `authentication`
The folder contains `.py`  scripts related to the authentication procedure.
- `__init__.py`
    Auxiliary file.
- `au_input_generator.py`
    Performs genereration of the input data for `authentication.py` script.
- `authentication.py`
    Performs calculation of the authentication hash-tag using Toeplitz or GOST R 34.12-2015 hashing.

#### `common`
The folder containts `.py` files with auxiliary functions used by different the algorithms.
- `__init__.py`
    Auxiliary file.
- `files.py`
    Contains functions related to input/output with files.
- `funcs.py`
    Contains mathmatical functions.
- `generate.py`
    Contains function related to operations with keys.
- `parseargs.py`
    Contains functions related to argument processing.

#### `error_correction`
The folder contains `.py`  scripts related to the error correction procedure.
- `__init__.py`
    Auxiliary file.
- `codes.py`
    Performs generation of the parity check matrices with [improved progressive edge growing algorithm](http://ieeexplore.ieee.org/document/5606185/?arnumber=5606185) with [particular distribuition polynomials](http://ieeexplore.ieee.org/document/5205475/?arnumber=5205475). The set of code rates contains nine elements: {0.9, 0.85, ..., 0.5}.
The algortithm also generates a set of symbol postions for the [untainted puncturing technique](http://ieeexplore.ieee.org/document/6290312/)
- `ec_input_generator.py`
    Performs genereration of the input data for `error_correction.py` script (except the set of parity check matrices).
- `error_correction.py`
   Performs a simulation of the error correction with symmetric blind reconciliation together with verification based on universal polynomial hashing.

#### `error_estimation`
The folder contains `.py`  scripts related to the error estimation procedure.
- `__init__.py`
    Auxiliary file.
- `ee_input_generator.py`
    Performs generation of the input data for `error_estimation.py` script.
- `error_estimation.py`
    Performs estimation of the error in the quantum channel according to the result of error correction procedure.

#### `privacy_amplification`
The folder contains `.py`  scripts related to the privacy amplification procedure.
- `__init__.py`
    Auxiliary file.
- `pa_input_generator.py`
    Performs genereration of the input data for `privacy_amplification.py` script.
- `privacy_amplification.py`
    Performs generation of the secure key according to BB84 decoy state protocol and Toeplitz hashing.

### 3. Notes about storage of parity-check matrices
The storage of parity-check matrices is based on two variables: `s_y_joins` and `y_s_joins`. They contains positions of nonzero elements for each row and column correspondingly.
For example, for the matrix
```sh
H =
1 1 0 1
1 0 1 1
0 1 1 0
```
one has
```sh
s_y_joins = [[0,1,3], [0,2,3],[1,2]]
y_s_joins = [[0,1],[0,2], [1,2],[0,1]]
```
