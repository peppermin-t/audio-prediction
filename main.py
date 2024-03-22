import numpy as np
import matplotlib.pyplot as plt

np.random.seed(4)

amp_data = np.load('amp_data.npz')['amp_data']

# Q1.a
plt.clf()
plt.subplot(1, 2, 1)
plt.plot(amp_data, 'b-')
plt.subplot(1, 2, 2)
plt.hist(amp_data, bins=100)
plt.show()

# Q1.b
# Create dataset
col = 21
amp_data = np.reshape(amp_data[: - (len(amp_data) % col)], (-1, col))
amp_data = np.random.permutation(amp_data)

train_test_id = int(0.7 * len(amp_data))
test_val_id = int(0.15 * len(amp_data)) + train_test_id
train_data, val_data, test_data = np.split(amp_data, [train_test_id, test_val_id])
X_shuf_train, y_shuf_train = train_data[:, : 20], train_data[:, 20]
X_shuf_val, y_shuf_val = val_data[:, : 20], val_data[:, 20]
X_shuf_test, y_shuf_test = test_data[:, : 20], test_data[:, 20]

# Q2.a
# Curve fitting on a snippet of audio
tt = np.arange(0, 1, 0.05)
idx = np.random.randint(20)
xx = X_shuf_train[idx]  # (20, )
y = y_shuf_train[idx]  # (1)
X_linear = np.stack((np.ones(col - 1), tt), axis=1)  # (20, 2)
w_linear = np.linalg.lstsq(X_linear, xx, rcond=None)[0]  # rcond ?
y_linear = X_linear.dot(w_linear)
X_quartic = np.stack((np.ones(col - 1), tt, tt ** 2, tt ** 3, tt ** 4), axis=1)  # (20, 5)
w_quartic = np.linalg.lstsq(X_quartic, xx, rcond=None)[0]
y_quartic = X_quartic.dot(w_quartic)

plt.clf()
plt.plot(tt, xx, 'r+', label="training points")  # training points
plt.plot(1, y, 'g+', label="predicting point")
plt.plot(tt, y_linear, 'b-', label="linear fit")  # straight line fit
plt.plot(tt, y_quartic, 'p-', label="quartic fit")  # quartic fit
plt.legend()
plt.show()


# # Finding best param i, j (My own idea)
    # min_error = 100
    # best_param = None
    # for i in range(2, 9):  # order
    #     for j in range(i, 21):  # considering last j points
    #         tt = np.arange(100 - 5 * j, 100, 5) / 100
    #         # tt = np.arange(1 - 0.05 * j, 1, 0.05)  # Wrong??
    #         errors = []
    #         for idx in range(24543, 32344):
    #             xx = X_shuf_train[idx][20 - j:]
    #             y = y_shuf_train[idx]
    #             X = np.stack((tt ** order for order in range(i)), axis=1)
    #             w = np.linalg.lstsq(X, xx, rcond=None)[0]
    #             y_ = X.dot(w)
    #             errors.append((y_ - y) ** 2)
    #         mean_error = np.mean(errors)
    #         print(mean_error)
    #         if mean_error < min_error:
    #             best_param = (i, j)
    #             min_error = mean_error
    # print(best_param)
    # print(min_error)

# Q3.b
# Choosing a polynomial predictor based on performance
# Question a not solved
def Phi(C, K):
    tt = np.arange(100 - 5 * C, 100, 5) / 100
    X = np.stack((tt ** order for order in range(K)), axis=1)
    return X

def make_vv(C, K):
        X = Phi(C, K)
        v = np.linalg.lstsq(X.T, np.ones(K), rcond=None)[0]
        return v

C = 20
K = 2
yp_linear_v = make_vv(C, K).T.dot(xx)
yp_linear = np.ones(2).dot(w_linear)
print(np.isclose(yp_linear, yp_linear_v))

C = 20
K = 5
yp_quartic_v = make_vv(20, 5).T.dot(xx)
yp_quartic = np.ones(5).dot(w_quartic)
print(np.isclose(yp_quartic, yp_quartic_v))

# Q3.c
K_list = [i for i in range(2, 17)]
C_max = 20
min_error = int(1e6)
best_params = None

for K in K_list:
    for C in range(K, C_max + 1):  # (C shouldn't be less than K)
        yp = make_vv(C, K).T.dot(X_shuf_train[:, 20 - C:].T)
        error = np.mean((yp - y_shuf_train) ** 2)
        if error < min_error:
            min_error = error
            best_params = (C, K)
print(f"The best params are C = {best_params[0]}, K = {best_params[1]},")
print(f"where the square error is {min_error}")

C = 2
K = 2
yp_train = make_vv(C, K).T.dot(X_shuf_train[:, 20 - C:].T)
error_train = np.mean((yp_train - y_shuf_train) ** 2)
yp_val = make_vv(C, K).T.dot(X_shuf_val[:, 20 - C:].T)
error_val = np.mean((yp_val - y_shuf_val) ** 2)
yp_test = make_vv(C, K).T.dot(X_shuf_test[:, 20 - C:].T)
error_test = np.mean((yp_test - y_shuf_test) ** 2)
print(f"The training set error is: {error_train}.")
print(f"The validating set error is: {error_val}.")
print(f"The testing set error is: {error_test}.")

# Q4.a
c_train = None
mse_train = float('inf')

c_vali = None
mse_vali = float('inf')

for C in range(1, 21):
    v_C = np.linalg.lstsq(X_shuf_train[:, -C:], y_shuf_train, rcond=None)[0]
    y_pred_training = np.dot(X_shuf_train[:, -C:], v_C)
    mse_training = np.mean((y_shuf_train - y_pred_training)**2)
    y_pred_validation = np.dot(X_shuf_val[:, -C:], v_C)
    mse_validation = np.mean((y_shuf_val - y_pred_validation)**2)
    if mse_training < mse_train:
        c_train = C
        mse_train = mse_trainin
    if mse_validation < mse_vali:
        c_vali = C
        mse_vali = mse_validation

print(f"Best C on training set", c_train)
print(f"MSE on training set", mse_train)

print(f"Best C on validation set:", c_vali)
print(f"MSE on validation set", mse_vali)

# Q4.b
v_C = np.linalg.lstsq(X_shuf_test[:, -16:], y_shuf_test, rcond=None)[0]
y_pred_test = np.dot(X_shuf_test[:, -16:], v_C)
mse_test = np.mean((y_shuf_test - y_pred_test)**2)
print(f"MSE on test set", mse_test)

# Q4.c
v_C = np.linalg.lstsq(X_shuf_val[:, -16:], y_shuf_val, rcond=None)[0]
y_pred_val = np.dot(X_shuf_val[:, -16:], v_C)
residuals = y_shuf_val - y_pred_val
plt.hist(residuals, bins=100)
plt.show()
