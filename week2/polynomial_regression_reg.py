import assignment2 as a2
import numpy as np
import matplotlib.pyplot as plt
import polynomial_regression as pr

(countries, features, values) = a2.load_unicef_data()

targets = values[:,1]  # (195, 1)
x = values[:,7:]  # (195, 33)
x = a2.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

# convert to numpy ndarray
x_train_cross_n = np.asarray(x_train)  # (100, 33)
t_train = np.asarray(t_train)  # (100, 1)
x_test_cross_n = np.asarray(x_test)
t_test = np.asarray(t_test)

DEGREE = 2
FOLD = 10
LAMBDA = [0, 0.01, 0.1, 1, 10, 10**2, 10**3, 10**4, 10**5]
train_losses_reg = {}
validate_losses_reg = {}
avg_losses_lambda = {}

def find_w(x, y, lambda_val):
    xTx = x.T.dot(x)
    n = x.shape[1]
    identity = np.identity(n)
    lambda_times_identity = lambda_val * identity
    sum_terms = xTx + lambda_times_identity
    inv = np.linalg.inv(sum_terms)
    w = inv.dot(x.T.dot(y))
    return w

def rms_loss_reg(x, y, w, lambda_val):
    y_hat = x.dot(w)  # y hat: anticipated y (calculated from function)
    loss = np.sqrt(np.mean( ( (y - y_hat) ** 2) + (lambda_val * w.T.dot(w) ) ) )
    return loss

# group losses values for each lambda
def group_losses_per_lambda(lamb):
    losses_fold = []
    for fold in range(1, FOLD+1):
        loss_fold = validate_losses_reg[fold][lamb]
        losses_fold.append(loss_fold)
    # print('losses_fold', losses_fold)
    avg_loss_lambda = sum(losses_fold) / FOLD
    # print('avg_loss_lambda:', avg_loss_lambda)
    return avg_loss_lambda

# print(x_train_norm.shape[0])


''' using normalized features '''
for i in range(1, FOLD+1):
    # print('i:', i)
    train_losses_i = {}
    validate_losses_i = {}
    losses_lambda_i = {}

    length = x_train.shape[0]
    start = 10 * (i-1)
    end = start + 10

    x_train_cross1 = x_train_cross_n[0:start]
    x_train_cross2 = x_train_cross_n[end:length]
    x_train_cross_norm = np.concatenate((x_train_cross1, x_train_cross2), axis=0)  # (90, 33)
    x_validate_cross_norm = x_train_cross_n[start: end]  # (10,33)
    t_train_cross = np.concatenate((t_train[0:start], t_train[end:length]), axis=0)  # (90, 1)
    t_validate_cross = t_train[start: end]  # (10, 1)

    # normalize input x (both for training & validation)
    # x_train_cross_norm = a2.normalize_data(x_train_cross)
    # x_validate_cross_norm = a2.normalize_data(x_validate_cross)
    # print('shapes:')
    # print('x_train_cross:', x_train_cross.shape)
    # print('x_validate_cross:', x_validate_cross.shape)
    # print('t_train_cross:', t_train_cross.shape)
    # print('t_validate_cross:', t_validate_cross.shape)

    for lamb in LAMBDA:
        features_val = pr.polynomial_features(x_train_cross_norm, t_train_cross, DEGREE)
        w = find_w(features_val, t_train_cross, lamb)
        # print('w', w)
        train_loss_reg = rms_loss_reg(features_val, t_train_cross, w, lamb)
        train_losses_i[lamb] = train_loss_reg

        features_validate_val = pr.polynomial_features(x_validate_cross_norm, t_validate_cross, DEGREE)
        validate_loss_reg = pr.rms_loss(features_validate_val, t_validate_cross, w)
        validate_losses_i[lamb] = validate_loss_reg

    train_losses_reg[i] = train_losses_i
    print('train_losses_i: ', train_losses_i)
    validate_losses_reg[i] = validate_losses_i
    print('validate_losses_i: ', validate_losses_i)

print('train_losses_reg: ', train_losses_reg)
print('validate_losses_reg: ', validate_losses_reg)

for l in LAMBDA:
    avg_losses_lambda[l] = group_losses_per_lambda(l)
print('avg_losses_lambda:', avg_losses_lambda)


''' Plot '''

plt.axhline(avg_losses_lambda[0], color='c', lw=3)
plt.semilogx(list(avg_losses_lambda.keys()), list(avg_losses_lambda.values()), color='m', lw=3)
plt.ylabel('Average Losses')
plt.semilogy()
plt.legend(['Losses when not regularized (lambda=0)', 'Average Losses (Normalized & L2-Regularized'], fontsize='xx-small', loc='lower left')
plt.title('Fit with polynomials, with L2-regularization (Normalized, degree = 2)')
plt.xlabel('Lambda')
plt.show()
