# Naive Bayes Classifier with KDE(Kernel Density Estimation) from scratch

In this work, we will implement a Naive Bayes Classifier that perform density estimation using Parzen windows.

* Using Kernel Density Estimation for Naive Bayes makes out model a lazy learner. 
* At training time, there's no processing done, except for memorizing the training data.
* At test time, every sample in test set is computed for density across every sample in training set.

## Features 

We will implement multiple kernels:
* Hypercube
* Radial

We will also implement 2 versions of model:
* Single bandwidth for a single-class/multi-class prediction problem.
* Class-specific multi bandwidths for a multi-class prediction problem. 

> At the end We will compare the results of different implementations of model with the ***sklearn - Gaussian Naive Bayes model***.


## Performance

Out model being a lazy learner has a very high time complexity. To increase the performance of the model we used Multi-processing pools at the time of Grid search to evaluate the model for different hyperparameters.
