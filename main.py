from data_generator import get_data
from model import Hazard_model
from utils import load_params

import numpy as np

if __name__ == '__main__':
    params = load_params()
    X, Y, Times, Cens, betas, eta, pi, delta, Z = get_data(params["run"][0]["n_obs"], params["run"][0]["max_time"],
                                                        kwargs=params["run"][0])
    print("Mean censorship rate :", np.mean(Cens))
    print("Mean evt time : ", np.mean(Times))
    print("mean pi : ", np.mean(pi))
    hazard_model = Hazard_model(cure=False, forward=True)
    hazard_model.fit(X, Cens, Times, delta, Z)
