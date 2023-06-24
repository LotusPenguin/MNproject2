def print_matrix(A, N):
    for i in range(N):
        for j in range(N):
            print(A[i][j], end="\t")
        print("")


def print_vector(a, N):
    for i in range(N):
        print(a[i])
    print("")


def build_band_matrix(N, a1, diagonal_width=1, a2=None, a3=None):
    A = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                A[i][j] = a1

    if diagonal_width > 1:
        for i in range(N):
            for j in range(N):
                if i == j + 1 or i == j - 1:
                    A[i][j] = a2
                elif diagonal_width > 2 and (i == j + 2 or i == j - 2):
                    A[i][j] = a3
    return A
