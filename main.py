from math import sin
from SetOfEquations import *
import jacobi
import gauss_seidel
import LU_factorization
import utils

import matplotlib.pyplot as plt


def main():
    # Zadanie A
    # numer indeksu: 184663
    c = 6
    d = 3
    e = 6
    f = 4
    N = 9 * 100 + c * 10 + d
    setOfEquations = SetOfEquations(N)

    a1 = 5 + e
    a2 = a3 = -1
    setOfEquations.A = utils.build_diagonal_matrix(N, a1, 3, a2, a3)
    setOfEquations.b = [sin(n * (f + 1)) for n in range(N)]

    # Zadanie B
    task_b(setOfEquations)

    # Zadanie C
    task_c(setOfEquations)

    # Zadanie D
    LU_factorization.solve(setOfEquations)

    # Test faktoryzacji LU
    # testSet = SetOfEquations(4)
    # testSet.A = \
    #     [[2, 1, 1, 0],
    #      [4, 3, 3, 1],
    #      [8, 7, 9, 5],
    #      [6, 7, 9, 8]]

    # utils.print_matrix(testSet.A, testSet.N)
    # print("")

    # LU_factorization.build_LU_matrices(testSet)

    # utils.print_matrix(testSet.L, testSet.N)
    # print("")
    # utils.print_matrix(testSet.U, testSet.N)

    # Test rozwiązania metodą LU
    # testSet2 = SetOfEquations(2)
    # testSet2.A = \
    #     [[3, 2],
    #      [2, 1]]
    # testSet2.b = [4, 1]
#
    # LU_factorization.solve(testSet2)
    # utils.print_vector(testSet2.x, 2)


def task_b(setOfEquations):
    # Zadanie B, metoda Jacobiego
    jacobi.solve(setOfEquations, 1e-9)
    plt.plot(setOfEquations.residuum_vector, 'b')
    plt.yscale("log")
    plt.title("Wykres normy residuum (Zadanie B)")
    plt.xlabel("Numer iteracji")
    plt.ylabel("Wartość normy euklidesowej wektora rozwiązań")

    # Zadanie B, metoda Gaussa-Seidla
    gauss_seidel.solve(setOfEquations, 1e-9)
    plt.plot(setOfEquations.residuum_vector, 'r')
    plt.legend(["Metoda Jacobiego", "Metoda Gaussa-Seidla"])
    plt.show()

    # Zadanie B, porównanie czasu obliczeń
    iterations = 10
    time_jacobi = []
    time_gs = []
    for i in range(iterations):
        time_jacobi.append(jacobi.solve(setOfEquations, 1e-9, verbose=False))
        time_gs.append(gauss_seidel.solve(setOfEquations, 1e-9, verbose=False))

    plt.plot(time_jacobi, 'b')
    plt.plot(time_gs, 'r')
    plt.title(f"Wykres czasu trwania algorytmów, {iterations} iteracji")
    plt.ylabel("Czas [s]")
    plt.xlabel("Numer iteracji")
    plt.legend(["Metoda Jacobiego", "Metoda Gaussa-Seidla"])
    plt.show()

    avg_jacobi = 0
    avg_gs = 0
    for i in range(iterations):
        avg_jacobi += time_jacobi.pop()
        avg_gs += time_gs.pop()
    avg_jacobi /= iterations
    avg_gs /= iterations
    print(f"Średni czas obliczeń metodą Jacobiego, {iterations} iteracji: {avg_jacobi}")
    print(f"Średni czas obliczeń metodą Gaussa-Seidla, {iterations} iteracji: {avg_gs}")
    print("")


def task_c(setOfEquations):
    a1 = 3
    a2 = a3 = -1
    setOfEquations.A = utils.build_diagonal_matrix(setOfEquations.N, a1, 3, a2, a3)
    jacobi.solve(setOfEquations)
    plt.plot(setOfEquations.residuum_vector, 'bo')
    plt.yscale("log")
    plt.title("Wykres normy residuum, metoda Jacobiego (Zadanie C)")
    plt.xlabel("Numer iteracji")
    plt.ylabel("Wartość normy euklidesowej wektora rozwiązań")
    plt.show()

    gauss_seidel.solve(setOfEquations)
    plt.plot(setOfEquations.residuum_vector, 'ro')
    plt.yscale("log")
    plt.title("Wykres normy residuum, metoda Gaussa-Seidla (Zadanie C)")
    plt.xlabel("Numer iteracji")
    plt.ylabel("Wartość normy euklidesowej wektora rozwiązań")
    plt.show()


def task_e(setOfEquations):
    pass



if __name__ == '__main__':
    main()
