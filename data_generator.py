import numpy as np
from process import OU_process
from utils import logit


def get_data(n, T, **kwargs):
    """
    Generate data for T timesteps.

    X contains explainatory variables, 3d matrix of sizes : p*n*T
    betas contains true parameters to be estimated, list of size p
    Times containes the true duration until event, list of size n
    Y containes the frailty, represented as a list of size T

    :param T: Number of times
    :param kappa: Value close to estimation found by D. Duffie
    :param p: Number of parameters
    :param n: Sample size
    :return:
    """

    print(kwargs["kwargs"])
    OU = OU_process(kwargs["kwargs"]["real values"]["kappa"], burn=kwargs["kwargs"]["OU_burn"])
    Y = OU.get_OU(T)
    Y = [0 for _ in range(len(Y))]
    betas = kwargs["kwargs"]["real values"]["betas"]
    p = len(kwargs["kwargs"]["min_values_X"])
    b = kwargs["kwargs"]["real values"]["cure_b"]
    const_p = len(b)
    if kwargs["kwargs"]["const"] == 1:
        X = [[[1] + [np.random.uniform(kwargs["kwargs"]["min_values_X"][i], kwargs["kwargs"]["max_values_X"][i]) for i
                     in range(p - 1)] for _ in range(n)] for _ in range(T)]
    else:
        print("No constant")
        X = [[[np.random.uniform(kwargs["kwargs"]["min_values_X"][i], kwargs["kwargs"]["max_values_X"][i]) for i in
               range(p)] for _ in range(n)] for _ in range(T)]
        const_X = [[np.random.uniform(kwargs["kwargs"]["min_values_X_const"][i], kwargs["kwargs"]["max_values_X_const"][i]) for i in
               range(const_p)] for _ in range(n)]
        Z = [const_X for _ in range(T)]
        X = np.c_[X, Z]


    # betas = [np.random.choice([-1, 1], p = [0.8, 0.2])*np.random.normal(0.9, 0.2) for _ in range(p+1)]
    # data = [[np.sum([betas[j] * X[k][i][j] for j in range(p)]) for k in range(T)] for i in range(n)]

    if kwargs["kwargs"]["cure"] == 1:
        X_b = np.matmul(const_X, kwargs["kwargs"]["real values"]["cure_b"])
        pi = [logit(X_b[i]) for i in range(n)]
    else:
        pi = [0 for _ in range(len(X[0]))]
    eta = kwargs["kwargs"]["real values"]["eta"]  # As in D. Duffie
    intensities = [[np.exp(np.sum([betas[j] * X[k][i][j] for j in range(p + const_p)]) + eta * Y[k]) for k in range(T)] for i in
                   range(n)]

    t = np.arange(T)
    C = [np.random.exponential(kwargs["kwargs"]["Censure"]) for _ in range(n)]
    L = np.array([[np.sum(intensities[k][:i]) for k in range(n)] for i in t]).T
    Times = []
    Cens = []
    delta = [] #In a cure model, 0 if immune, 1 else.
    for i in range(n):
        if np.random.uniform() >= pi[i]:
            U = np.random.uniform(0, 1)
            value = [x for x in L[i] if x <= -np.log(1 - U)][-1]
            idx = np.where(L[i] == value)[0][0]
            time = ((-np.log(1 - U) - value) + intensities[i][idx] * idx) / intensities[i][idx]
            Times += [min(time, T - 1)]
            Cens += [1 if (time > C[i] or time > T) else 0]
            delta += [0]
        else:
            Times += [T - 1]
            Cens += [1]
            delta += [1]
    return X, Y, Times, Cens, betas, eta, pi, delta, const_X
