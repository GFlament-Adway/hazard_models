import numpy as np

def get_covariates(n,t, p):
    mu = [np.random.normal(1, 1) for _ in range(p)]
    X = [[[np.random.normal(loc = mu[i], scale = 1) for i in range(p)]  for _ in range(t) ] for _ in range(n)]
    return X

def kaplan_meier(Times, Cens, t):
    int_hat = []
    n = len(Times)
    for k in range(1, t):
        at_risk = np.sum([Times[i] >= k for i in range(n)])
        if at_risk > 0:
            int_hat += [1 - (np.sum([Times[i] == k if Cens[i]==1 else 0 for i in range(n)])/at_risk)]
        else:
            int_hat += [int_hat[-1]]
    KM = []
    for k in range(0, t):
        KM += [np.prod(int_hat[:k])]

    print(KM)
    return KM

def get_data(X, alpha):
    Times = []
    Cens = []
    n,t,p = len(X), len(X[0]), len(X[0][0])
    intensities = np.exp(np.matmul(X, alpha))
    Surv = np.exp([[-np.sum(intensities[i][:k]) for k in range(t)] for i in range(n)])
    U = [np.random.uniform(0, 1) for _ in range(n)] #Generate uniform sample to generate true event time
    C_times = [np.random.exponential(7) for _ in range(n)] #Generate censorship time
    for i in range(n):
        Y = np.min([k for k in range(t) if U[i] < 1 - Surv[i][k]] + [t]) #Compute event time from the univariate distribution previously sampled
        Times += [np.min([C_times[i], Y])] #Real observation.
        Cens += [1 if C_times[i] > Y else 0]
    return Times, Cens



if __name__ == "__main__":
    np.random.seed(555)
    n = 500
    t = 7
    p = 5
    alpha = [np.random.uniform(-1, 0) for _ in range(p)]
    print("Real params : ", alpha)
    X = get_covariates(n, t, p)
    Times, Cens = get_data(X, alpha)
    intensities = np.exp(np.matmul(X, alpha))
    Surv = np.exp([[-np.sum(intensities[i][:k]) for k in range(t)] for i in range(n)])

    #Estimation of the KM estimator
    KM = kaplan_meier(Times, Cens, t)

    import matplotlib.pyplot as plt
    print(np.mean(Cens))
    plt.plot(KM, label="KM estimator of the survival function on simulated times")
    plt.plot(np.array(Surv[0]).T, color="b", alpha=0.01, label="Real individuals survival function")
    plt.plot(np.array(Surv).T, color="b", alpha=0.01)
    plt.ylim(0,1)
    plt.legend()
    plt.show()