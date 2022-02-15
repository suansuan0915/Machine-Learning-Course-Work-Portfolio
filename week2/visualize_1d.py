#!/usr/bin/env python

import assignment2 as a2
import numpy as np
import matplotlib.pyplot as plt
import polynomial_regression_1d as pr_1d
import polynomial_regression as pr


(countries, features, values) = a2.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
#x = a2.normalize_data(x)

N_TRAIN = 100;
DEGREE = 3
# Select a single feature.
x_train = x[0:N_TRAIN,:]
# x_train = x[0:N_TRAIN,10]
t_train = targets[0:N_TRAIN]
x_test = x[N_TRAIN:,:]
t_test = targets[N_TRAIN:]
x_test = np.asarray(x_test)
t_test = np.asarray(t_test)

# convert to numpy ndarray
x_train = np.asarray(x_train)
t_train = np.asarray(t_train)


# index when select any feature 8-15, numpy ndarrays inside.
x_train_cols = pr_1d.convert_format(x_train)  # features for training. 8 features (8 , 100)
x_test_cols = pr_1d.convert_format(x_test)

def polynomial_features_1col(x, order):
    features_with_1 = np.hstack([x ** i for i in range(0, order+1)])
    print(features_with_1.shape)  # (500, 4)
    print(features_with_1[:2])
    return features_with_1

def polynomial(values, coeffs):
    expanded = np.hstack([coeffs[i] * (values ** i) for i in range(0, len(coeffs) )])
    return np.sum(expanded, axis=-1)

def plot_polynomial(x, coeffs, color='red', label='polynomial', alpha=1.0):
    poly = polynomial(x, coeffs).reshape(-1, 1)
    print("poly shape:", poly.shape)
    sorted_pair = sorted(zip(x, poly))
    j, k = zip(*sorted_pair)
    plt.plot(j, k, color=color, linewidth=4, label=label, alpha=alpha)

def plot_learned(x, w):
    plt.figure(figsize=(10, 5))
    plot_polynomial(x, w)
    # plot_polynomial(x_train_cols[i], w)
    plt.title(f"Visualization of a regression estimate using random outputs, Feature {i+8}, {features[i+7]}")


# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate
# x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)


# TO DO:: Put your regression estimate here in place of x_ev.
for i in range(3, 6):
    x_ev = np.linspace(np.asscalar(min(x_train_cols[i])), np.asscalar(max(x_train_cols[i])), num=500)
    x_ev = x_ev.reshape(-1, 1)
    tr_features = pr.polynomial_features(x_train_cols[i], t_train, DEGREE)
    weights = pr.least_squares(tr_features, t_train)
    plot_learned(x_ev, weights)
    plt.scatter(x_train_cols[i], t_train, color='green')
    plt.scatter(x_test_cols[i], t_test, color='blue')
    # plot_regression(x_train_cols[i], t_train, i)
    # plt.scatter( x_train_cols[i], t_train, color='green')
    # plt.scatter(x_test_cols[i], t_test, color='blue')

plt.show()

# Evaluate regression on the linspace samples.
# y_ev = np.random.random_sample(x_ev.shape)
# y_ev = 100*np.sin(x_ev)

# plt.plot(x_ev,y_ev,'r.-')
# plt.plot(x_train,t_train,'bo')
# plt.title('A visualization of a regression estimate using random outputs')
# plt.show()
