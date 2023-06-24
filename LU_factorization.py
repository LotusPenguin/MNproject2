import copy
import time

import utils


def solve(setOfEquations, verbose=True):
    if verbose:
        print("Rozwiązywanie metodą faktoryzacji LU")
    t = time.time()

    # data initialization
    setOfEquations.x = [0 for _ in range(0, setOfEquations.N)]
    setOfEquations.res = [0 for _ in range(0, setOfEquations.N)]
    setOfEquations.norm_history = []

    # core algorithm
    LU_decomposition(setOfEquations, verbose=False)
    y = forward_substitution(setOfEquations.L, setOfEquations.b, setOfEquations.N)
    setOfEquations.x = backward_substitution(setOfEquations.U, y, setOfEquations.N)

    # calculating residuum norm
    setOfEquations.calculate_residuum_norm()

    # finishing
    t = time.time() - t
    if verbose:
        print(f"Czas obliczeń: {t} s")
        print(f"Norma z residuum: {setOfEquations.norm_history[0]}")
        print("")

    return t


def LU_decomposition(setOfEquations, verbose=True):
    if verbose:
        print("Faktoryzacja LU")

    setOfEquations.U = copy.deepcopy(setOfEquations.A)
    setOfEquations.L = utils.build_band_matrix(setOfEquations.N, 1)

    for i in range(0, setOfEquations.N):
        for j in range(i + 1, setOfEquations.N):
            setOfEquations.L[j][i] = setOfEquations.U[j][i] / setOfEquations.U[i][i]
            for k in range(i, setOfEquations.N):
                setOfEquations.U[j][k] = setOfEquations.U[j][k] - setOfEquations.L[j][i] * setOfEquations.U[i][k]


def forward_substitution(A, b, N):
    x = [0 for _ in range(N)]
    for i in range(0, N):
        temp = 0
        for j in range(0, i):
            temp += A[i][j] * x[j]
        x[i] = (b[i] - temp) / A[i][i]

    return x


def backward_substitution(A, b, N):
    x = [0 for _ in range(N)]
    for i in range(N - 1, -1, -1):
        temp = 0
        for j in range(N - 1, i, -1):
            temp += A[i][j] * x[j]
        x[i] = (b[i] - temp) / A[i][i]

    return x
