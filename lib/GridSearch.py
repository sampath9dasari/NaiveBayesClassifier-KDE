"""
Function to loop through available combinations of hyper parameters and
selecting the best performing parameter for a holdout set
"""
# from lib.NaiveBayesClassifier import NaiveBayesClassifier

import numpy as np
from itertools import product
import multiprocessing as mp
from sklearn.model_selection import train_test_split


def grid_scoring(row, x_sub, y_sub, x_val, y_val, multi_bw):
    bandwidth, kernel = row
    nbmodel = NaiveBayesClassifier(bandwidth=bandwidth, kernel=kernel, MultiBW=multi_bw)
    # Train the network
    nbmodel.fit(x_sub, y_sub)
    # Accuracy on the validation set
    val_accuracy = nbmodel.score(x_val, y_val)
    # Save results
    return row, val_accuracy, nbmodel


def grid_search(x_train, y_train, bw, kl, multi_bw=False):
    best_nb = None  # store the best model into this
    best_val = -1
    results = {}

    if multi_bw:
        bandwidths = list(product(bw, repeat=10))  # values to be tunned
    else:
        bandwidths = bw  # values to be tuned
    kernels = kl  # values to be tuned

    search_grid = [(x, y) for x in bandwidths for y in kernels]
    x_sub, x_val, y_sub, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=10)

    grid_pool = mp.Pool(mp.cpu_count())
    results_object = [grid_pool.apply_async(grid_scoring, args=(row, x_sub, y_sub, x_val, y_val, multi_bw)) \
                      for row in search_grid]

    results = np.array([r.get() for r in results_object])

    grid_pool.close()
    grid_pool.join()

    return results
