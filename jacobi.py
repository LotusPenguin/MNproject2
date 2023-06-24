import time
import copy


def solve(setOfEquations, stop=1e-6, verbose=True):
    if verbose:
        print("Rozwiązywanie metodą Jacobiego")
    t = time.time()

    iteration_count = 0
    xNew = [0 for _ in range(0, setOfEquations.N)]
    setOfEquations.x = [0 for _ in range(0, setOfEquations.N)]
    setOfEquations.res = [0 for _ in range(0, setOfEquations.N)]
    setOfEquations.residuum_vector = []

    # main loop
    while True:
        for i in range(setOfEquations.N):
            term1 = 0
            term2 = 0

            for j in range(0, i):
                term1 += setOfEquations.A[i][j] * setOfEquations.x[j]
            for j in range(i + 1, setOfEquations.N):
                term2 += setOfEquations.A[i][j] * setOfEquations.x[j]
            xNew[i] = (setOfEquations.b[i] - term1 - term2) / setOfEquations.A[i][i]

        setOfEquations.calculate_residuum_norm()
        setOfEquations.x = copy.deepcopy(xNew)
        iteration_count = iteration_count + 1

        # Sprawdzenie zbieżności normy residuum
        if iteration_count > 1 and setOfEquations.residuum_vector[-1] > setOfEquations.residuum_vector[-2]:
            if verbose:
                print("Metoda Jacobiego niezbieżna dla tego układu równań")
            break

        # Warunek końca obliczeń
        if setOfEquations.residuum_vector[-1] < stop:
            break

    t = time.time() - t
    if verbose:
        print(f"Czas obliczeń: {t} s")
        print(f"Liczba iteracji: {iteration_count}")
        print(f"Najniższa norma z residuum: {min(setOfEquations.residuum_vector)}")
        print("")

    return t
