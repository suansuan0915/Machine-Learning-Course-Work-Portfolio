import matplotlib.pyplot as plt
import assignment2 as a2
import numpy as np
import polynomial_regression as pr

(countries, features, values) = a2.load_unicef_data()

targets = values[:,1]  # (195, 1)
x = values[:,7:]  # (195, 33)
# print(values.shape)  # (195, 40)
# print(features)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]  # (100,33)
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

# convert to numpy ndarray
x_train = np.asarray(x_train)
t_train = np.asarray(t_train)
x_test = np.asarray(x_test)
t_test = np.asarray(t_test)

FEATURE_LOWER_BOUND = 0
FEATURE_UPPER_BOUND = 15-8
ORDER = 3

'''

plot bar chart for feature 8 to 15 at degree 3 polynomial (no regularization)

'''

# change feature's format to column
def convert_format(feature_values):
    value_cols = []
    for j in range(FEATURE_LOWER_BOUND, FEATURE_UPPER_BOUND+1):
        lst = []
        for i in range(feature_values.shape[0]):
            lst.append([feature_values[i][j]])
        arr = np.array(lst)
        value_cols.append(arr)
    # value_cols = np.ndarray(value_cols)  # list of arrays
    value_cols = value_cols
    return value_cols


# index when select any feature 8-15, numpy ndarrays inside.
x_train_cols = convert_format(x_train)  # features for training
x_test_cols = convert_format(x_test)  # features for testing

# print(len(x_train_cols))  # 8 features
# print(x_train_cols[0].shape)  # (100,1) for 100 data points
# print(type(x_train_cols[0]))  # numpy ndarray

train_losses3 = {}
test_losses3 = {}

for i in range(len(x_train_cols)):
    tr_features = pr.polynomial_features(x_train_cols[i], t_train, ORDER)
    te_features = pr.polynomial_features(x_test_cols[i], t_test, ORDER)
    w = pr.least_squares(tr_features, t_train)
    train_loss = pr.rms_loss(tr_features, t_train, w)
    test_loss = pr.rms_loss(te_features, t_test, w)
    train_losses3[i+8-1] = train_loss
    test_losses3[i+8-1] = test_loss

print(train_losses3)  # len: 8
print(test_losses3)  # len: 8

# plot bar chart
labels = list(train_losses3.keys())
x = np.arange(8, 16)  # len(labels))
width = 0.4
fig, ax = plt.subplots()
ax.bar(x - width/2, list(train_losses3.values()), width=width, label='train')
ax.bar(x + width/2, list(test_losses3.values()), width=width, label='test')

plt.ylabel('RMS')
plt.title('Train & Test Losses with Feature 8-15 (no regularized)')
plt.legend(['Training error', 'Test error'])
plt.xlabel(f'features')
fig.tight_layout()
plt.show()





