import math

class SetOfEquations:
    def __init__(self, N):
        self.N = N
        self.A = None
        self.b = None
        self.x = None
        self.res = None
        self.norm_history = None
        self.L = None
        self.U = None

    def calculate_residuum_norm(self):
        # calculating residuum
        self.res = [0 for _ in range(0, self.N)]
        for i in range(0, self.N):
            self.res[i] = 0
            for j in range(0, self.N):
                self.res[i] += self.A[i][j] * self.x[j]
            self.res[i] -= self.b[i]

        # calculating norm
        temp = 0
        for i in range(0, self.N):
            temp += self.res[i] * self.res[i]
        self.norm_history.append(math.sqrt(temp))
