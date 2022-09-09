import numpy as np
from scipy.optimize import minimize
from utils import logit, rmse


class Hazard_model():
    def __init__(self, cure=False, forward = False, forward_step=0):
        self.forward_step = forward_step
        self.cure = cure
        self.forward = forward

    def fit(self, X, Cens, Times, delta, Z):
        if self.cure is True:
            init_b = [np.random.normal() for _ in range(len(Z[0]))]
            args = minimize(self.__log_like_cure, init_b, args=(delta, Z))
            self.b = args["x"]
            X_b = np.matmul(Z, self.b)
            print("estimated b : ", self.b)
            init_betas = [np.random.normal() for _ in range(len(X[0][0]))]  # True values [-1, -1.2, -0.65, -0.25, 1.55]
            print("Init betas :", init_betas)
            args = minimize(self.__log_like, init_betas, args=(X, Cens, Times, delta, Z))
            self.betas = args["x"]
            print("estimated betas : ", self.betas)
        elif self.forward is True:
            betas = [np.random.normal() for _ in range(len(X[0][0]))] # A changer !!! Il faut mettre ce truc en param !
            args = minimize(self.__log_like_duan, betas, args=(X, Cens, Times))
            print(args["x"])

    def __log_like_cure(self, b, *args):
        n = len(args[1])
        like = []
        X_b = np.matmul(args[1], b)
        pi_hat = [logit(X_b[i]) for i in range(n)]
        for i in range(n):
            like += [(1 - args[0][i]) * np.log((1 - pi_hat[i])) + (args[0][i]) * np.log(pi_hat[i])]
        return -np.sum(like)

    def __log_like_duan(self, betas, *args):
        intensities = np.clip(np.exp(np.matmul(args[0], betas).T), 10e-20, 10e20)
        n = len(intensities)
        like = []
        for i in range(n):
            int_intensities = -np.sum(intensities[i][:int(args[2][i])]) - intensities[i][int(args[2][i])] * (
                    float(args[2][i]) - int(args[2][i]))
            if args[1][i] == 1:
                like += [(1 - args[1][i]) * int_intensities]
            else:
                like += [(1 - args[1][i]) * int_intensities + (1 - args[1][i]) * np.log(intensities[i][int(args[2][i])])]
        return -np.sum(like)

    def __log_like(self, betas, *args):
        intensities = np.clip(np.exp(np.matmul(args[0], betas).T), 10e-20, 10e20)
        n = len(args[1])
        like = []
        for i in range(n):
            int_intensities = -np.sum(intensities[i][:int(args[2][i])]) - intensities[i][int(args[2][i])] * (
                    float(args[2][i]) - int(args[2][i]))
            if args[1][i] == 1:
                like += [(1 - args[3][i]) * int_intensities]
            else:
                like += [(1 - args[3][i]) * int_intensities + (1 - args[3][i]) * np.log(intensities[i][int(args[2][i])])]
        return -np.sum(like)