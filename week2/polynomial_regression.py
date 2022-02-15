#!/usr/bin/env python

import assignment2 as a2
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a2.load_unicef_data()

# country with the highest child mortality rate in 1990 (under 5)
print(features)
U5MR1990_col = features.get_loc('Under-5 mortality rate (U5MR) 1990')  # 0
U5MR1990_val = values[:, U5MR1990_col]
U5MR1990_max = max(U5MR1990_val)
max1990_row = np.argmax(U5MR1990_val, axis=0)
max1990_country = countries[max1990_row][0][0]
# print(max_row)
# print(values[max1900_row, U5MR1990_col])
# print(max1990_country)
print('country with the highest child mortality rate in 1990: ', str(max1990_country))  # Niger
print('with a mortality rate (under 5): ', float(U5MR1990_max))  # 313.7

# country with the highest child mortality rate in 2011 (under 5)
U5MR2011_col = features.get_loc('Under-5 mortality rate (U5MR) 2011')  # 1
U5MR2011_val = values[:, U5MR2011_col]
# print(U5MR2011_val)
U5MR2011_max = max(U5MR2011_val)
max2011_row = np.argmax(U5MR2011_val, axis=0)
max2011_country = countries[max2011_row][0][0]
# print(max2011_row)
# print(values[max2011_row, U5MR2011_col])
print('country with the highest child mortality rate in 2011: ', str(max2011_country))  # Sierra Leone
print('with a mortality rate (under 5): ', float(U5MR2011_max))  # 185.3


def least_squares(x, y):
    # xTx = x.T.dot(x)
    # xTx_pinv = np.linalg.pinv(xTx)
    # w = xTx_pinv.dot(x.T.dot(y))
    w = np.linalg.pinv(x).dot(y)
    return w

def rms_loss(x, y, w):
    y_hat = x.dot(w)  # y hat: anticipated y (calculated from function)
    loss = np.sqrt(np.mean((y - y_hat) ** 2))
    return loss

def polynomial_features(x, t, order):
    t = np.arange(t.shape[0])
    features_no_1 = np.hstack([x ** i for i in range(1, order+1)])
    features_with_1 = np.column_stack((np.ones_like(t), features_no_1))
    return features_with_1


targets = values[:,1]  # (195, 1)
x = values[:,7:]  # (195, 33)
x_n = a2.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_train_norm = x_n[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
x_test_norm = x_n[N_TRAIN:,:]
# x_train_norm = a2.normalize_data(x[0:N_TRAIN,:])
# x_test_norm = a2.normalize_data(x[N_TRAIN:,:])
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


# convert to numpy ndarray
x_train = np.asarray(x_train)  # (100, 33)
t_train = np.asarray(t_train)  # (100, 1)
x_train_norm = np.asarray(x_train_norm)
x_test_norm = np.asarray(x_test_norm)
x_test = np.asarray(x_test)
t_test = np.asarray(t_test)
# print(type(x_train))  # np.matrix, not np.ndarray


''' 
Using none-normalized input 

'''
DEGREES = 9
train_losses = {}
test_losses = {}

for degree in range(1, DEGREES):
    features = polynomial_features(x_train, t_train, degree)
    w = least_squares(features, t_train)
    train_loss = rms_loss(features, t_train, w)
    test_features = polynomial_features(x_test, t_test, degree)
    test_loss = rms_loss(test_features, t_test, w)
    # print(features[:5])
    # print("degree: ", degree)
    # print("train_loss:", train_loss)
    train_losses[degree] = train_loss
    test_losses[degree] = test_loss
print(train_losses)
print(test_losses)

# Produce a plot of results.
plot1 = plt.figure(1)
plt.plot(list(train_losses.keys()), list(train_losses.values()))
plt.plot(list(test_losses.keys()), list(test_losses.values()))

plt.ylabel('RMS')
plt.semilogy()
plt.legend(['Training error', 'Test error'])
plt.title('Fit with polynomials, no regularization (non-normalized)')
plt.xlabel('Polynomial degree')


''' 
Using normalized input 

'''
train_losses_n = {}
test_losses_n = {}

for degree in range(1, DEGREES):
    tr_features = polynomial_features(x_train_norm, t_train, degree)
    te_features = polynomial_features(x_test_norm, t_test, degree)
    w = least_squares(tr_features, t_train)
    train_loss_n = rms_loss(tr_features, t_train, w)
    test_loss_n = rms_loss(te_features, t_test, w)
    print("degree: ", degree)
    print("train_loss_norm:", train_loss_n)
    train_losses_n[degree] = train_loss_n
    test_losses_n[degree] = test_loss_n
print(train_losses_n)
print(test_losses_n)

plot1 = plt.figure(2)
plt.plot(list(train_losses_n.keys()), list(train_losses_n.values()))
plt.plot(list(test_losses_n.keys()), list(test_losses_n.values()))

plt.ylabel('RMS')
plt.semilogy()
plt.legend(['Training error with Normalization', 'Test error with Normalization'])
plt.title('Fit with polynomials, no regularization (Normalized)')
plt.xlabel('Polynomial degree')
plt.show()