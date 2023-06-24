from math import sin
from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt

import LU_factorization
import gauss_seidel
import jacobi
import utils
from SetOfEquations import *


def main():
    # Zadanie A
    # numer indeksu: 184663
    c = 6
    d = 3
    N = 9 * 100 + c * 10 + d
    setOfEquations = SetOfEquations(N)
    task_a(setOfEquations)

    # Zadanie B
    task_b(setOfEquations)

    # Zadanie C
    task_c(setOfEquations)

    # Zadanie D
    LU_factorization.solve(setOfEquations)

    # Zadanie E
    task_e()

    # Dodatkowe testy poprawności na prostszych układach równań
    # validity_tests()


def task_a(setOfEquations):
    # numer indeksu: 184663
    e = 6
    f = 4

    a1 = 5 + e
    a2 = a3 = -1
    setOfEquations.A = utils.build_band_matrix(setOfEquations.N, a1, 3, a2, a3)
    setOfEquations.b = [sin(n * (f + 1)) for n in range(setOfEquations.N)]


def task_b(setOfEquations):
    # Zadanie B, metoda Jacobiego
    jacobi.solve(setOfEquations, 1e-9)
    plt.plot(range(1, len(setOfEquations.norm_history) + 1), setOfEquations.norm_history, 'b-o')
    plt.yscale("log")
    plt.title("Norma residuum w kolejnych iteracjach (Zadanie B)")
    plt.yticks([1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
    plt.xlabel("Numer iteracji")
    plt.ylabel("Wartość normy euklidesowej wektora rozwiązań")

    # Zadanie B, metoda Gaussa-Seidla
    gauss_seidel.solve(setOfEquations, 1e-9)
    plt.plot(range(1, len(setOfEquations.norm_history) + 1), setOfEquations.norm_history, 'r-o')
    plt.legend(["Metoda Jacobiego", "Metoda Gaussa-Seidla"])
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

    # Zadanie B, porównanie czasu obliczeń
    iterations = 100
    time_jacobi = []
    time_gs = []
    for i in range(iterations):
        time_jacobi.append(jacobi.solve(setOfEquations, 1e-9, verbose=False))
        time_gs.append(gauss_seidel.solve(setOfEquations, 1e-9, verbose=False))

    plt.plot(range(1, len(time_jacobi) + 1), time_jacobi, 'b-o')
    plt.plot(range(1, len(time_gs) + 1), time_gs, 'r-o')
    plt.title(f"Czas trwania algorytmów, {iterations} iteracji")
    plt.ylabel("Czas [s]")
    plt.xlabel("Numer iteracji")
    plt.legend(["Metoda Jacobiego", "Metoda Gaussa-Seidla"])
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
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
    setOfEquations.A = utils.build_band_matrix(setOfEquations.N, a1, 3, a2, a3)
    jacobi.solve(setOfEquations)
    plt.plot(range(1, len(setOfEquations.norm_history) + 1), setOfEquations.norm_history, 'b-o')
    plt.yscale("log")
    plt.title("Norma residuum w kolejnych iteracjach, metoda Jacobiego (Zadanie C)")
    plt.xlabel("Numer iteracji")
    plt.ylabel("Wartość normy euklidesowej wektora rozwiązań")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

    gauss_seidel.solve(setOfEquations)
    plt.plot(range(1, len(setOfEquations.norm_history) + 1), setOfEquations.norm_history, 'r-o')
    plt.yscale("log")
    plt.title("Norma residuum w kolejnych iteracjach, metoda Gaussa-Seidla (Zadanie C)")
    plt.xlabel("Numer iteracji")
    plt.ylabel("Wartość normy euklidesowej wektora rozwiązań")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()


def task_e():
    time_jacobi = []
    time_gs = []
    time_lu = []

    N = [100, 500, 1000, 2000, 3000]
    for n in N:
        setOfEquations = SetOfEquations(n)
        task_a(setOfEquations)
        print(f"N = {n}")
        time_jacobi.append(jacobi.solve(setOfEquations, verbose=False))
        print(f"Metoda Jacobiego: {time_jacobi[-1]} s")
        time_gs.append(gauss_seidel.solve(setOfEquations, verbose=False))
        print(f"Metoda Gaussa-Seidla: {time_gs[-1]} s")
        time_lu.append(LU_factorization.solve(setOfEquations, verbose=False))
        print(f"Metoda faktoryzacji LU: {time_lu[-1]} s")

    plt.plot(N, time_jacobi, 'b-o')
    plt.plot(N, time_gs, 'r-o')
    plt.plot(N, time_lu, 'g-o')
    plt.legend(["Metoda Jacobiego", "Metoda Gaussa-Seidla", "Metoda faktoryzacji LU"])
    plt.title("Czas trwania obliczeń w zależności od liczby niewiadomych")
    plt.xlabel("Liczba niewiadomych")
    plt.xticks(N)
    plt.ylabel("Czas trwania obliczeń [s]")
    plt.show()


def validity_tests():
    # Test faktoryzacji LU
    testSet = SetOfEquations(4)
    testSet.A = \
        [[2, 1, 1, 0],
         [4, 3, 3, 1],
         [8, 7, 9, 5],
         [6, 7, 9, 8]]

    utils.print_matrix(testSet.A, testSet.N)
    print("")

    LU_factorization.LU_decomposition(testSet)

    utils.print_matrix(testSet.L, testSet.N)
    print("")
    utils.print_matrix(testSet.U, testSet.N)
    # Testy poprawności, x = [0.8122, -0.6650]
    testSet2 = SetOfEquations(2)
    testSet2.A = \
        [[16, 3],
         [7, -11]]
    testSet2.b = [11, 13]
    # Test rozwiązania metodą Jacobiego
    jacobi.solve(testSet2)
    utils.print_vector(testSet2.x, 2)
    # Test rozwiązania metodą Gaussa-Seidla
    gauss_seidel.solve(testSet2)
    utils.print_vector(testSet2.x, 2)
    # Test rozwiązania metodą LU
    LU_factorization.solve(testSet2)
    utils.print_vector(testSet2.x, 2)
    # Test stabilności implementacji
    LU_factorization.solve(testSet2)
    utils.print_vector(testSet2.x, 2)

    jacobi.solve(testSet2)
    utils.print_vector(testSet2.x, 2)


if __name__ == '__main__':
    main()
