import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_vector(firast_nomber: [int, float], last_nomber: [int, float], step: [int, float] = 1 ) -> np.ndarray:
    ## нужно в отчете отметить его смешную работу с не целыми числами
    vector = np.arange(firast_nomber, last_nomber, step)
    return vector


def create_matrix(m: [int] , n: [int]) -> np.ndarray:
    matrix = np.random.rand(m, n)
    return matrix

def reshape_arr(arr: [list, np.ndarray], m: [int], n: [int]) -> np.ndarray:
    res = arr.reshape(m, n)
    return res
if __name__ == "__main__":
    v = create_vector(0, 10)
    m = create_matrix(3, 4)
    x = 1
    print(-x)