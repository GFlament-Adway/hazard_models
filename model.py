import numpy as np
from scipy.optimize import minimize

class Hazard_model():
    def __init__(self, forward_step):
        self.forward_step = forward_step

    def __log_like(self, betas, *args):
        intensities = np.clip(np.exp(np.matmul(args[0], betas).T), 10e-20, 10e20)
        n = len(args[1])
        like = []
        for i in range(n):
            int_intensities = -np.sum(intensities[i][:int(args[2][i])]) - intensities[i][int(args[2][i])] * (
                    float(args[2][i]) - int(args[2][i]))
            if args[1][i] == 1:
                like += [int_intensities]
            else:
                like += [int_intensities + np.log(intensities[i][int(args[2][i])])]
        return -np.sum(like)

    def fit(self, X, D, Y):
        init_betas = [0.2, -0.5, -0.25, -1.25, 0.55] #True values [-1.4, -0.4, 0, -1.9, 1.5]
        print("Init betas :", init_betas)
        args = minimize(self.__log_like, init_betas, args=(X, D, Y))

        print(args)
    #minimize(self.__log_like())