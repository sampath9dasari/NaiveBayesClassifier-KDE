from lib.classifier import NaiveBayesClassifier

import numpy as np
from itertools import product
import multiprocessing as mp
from sklearn.model_selection import train_test_split

"""
Functions to loop through available combinations of hyper parameters 
and selecting the best performing parameter based on a holdout set
"""


def grid_search(x_train, y_train, bw, kl, multi_bw=False):
    """
    Grid Search algorithm uses Multi Processing to perform faster parameter tuning

    :param x_train: Array of input data
    :param y_train: Array of output labels
    :param bw: An integer when we use a single bandwidth model
    :param bw: A list of integers when using a class-specific bandwidth model
    :param kl: String specifying the kernel to be used
    :param multi_bw: A boolean value specifying if we are using class-specific or single bandwidth models
    :return: A list of tuples containing - parameters used, accuracy on validation set, object of the model
    """

    if multi_bw:
        bandwidths = list(product(bw, repeat=10))  # values to be tuned
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


def grid_scoring(row, x_sub, y_sub, x_val, y_val, multi_bw):
    """
    Helper function to return the score of a model for given set of parameters

    :param row: A tuple containing parameters to be used
    :param x_sub: Array of input data to train model
    :param y_sub: Array of output labels to train model
    :param x_val: Array of input data to make predictions
    :param y_val: Array of output labels to calculate accuracy
    :param multi_bw: A boolean value specifying if we are using class-specific or single bandwidth models
    :return:
    """
    bandwidth, kernel = row
    nbmodel = NaiveBayesClassifier(bandwidth=bandwidth, kernel=kernel, multi_bw=multi_bw)
    # Train the network
    nbmodel.fit(x_sub, y_sub)
    # Accuracy on the validation set
    val_accuracy = nbmodel.score(x_val, y_val)
    # Save results
    return row, val_accuracy, nbmodel
