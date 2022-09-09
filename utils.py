import json
import numpy as np

def load_params(path="settings/settings.json"):
    """

    :param path: path to file that contains all the settings.
    :return: Dictionnary with all necessary settings.
    """
    with open(path) as json_file:
        data = json.load(json_file)

    return data

def logit(x):
    return np.clip(np.exp(x)/(1 + np.exp(x)), 0, 1)


def rmse(x,y):
    return np.mean([(x[i]-y[i])**2 for i in range(len(x))])