import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Getting started
    amp_data = np.load('amp_data.npz')['amp_data']  # (33713280)
    
    # plt.clf()
    # plt.subplot(1, 2, 1)
    # plt.plot(amp_data, 'b-')
    # plt.subplot(1, 2, 2)
    # plt.hist(amp_data, bins=100)
    # plt.show()

    np.random.seed(4)  # shall I submit this?
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
    # print(X_shuf_train.shape)
    # print(y_shuf_train.shape)
    # print(X_shuf_val.shape)
    # print(y_shuf_val.shape)
    # print(X_shuf_test.shape)
    # print(y_shuf_test.shape)

    # # Curve fitting on a snippet of audio
    # # Do I need to demonstrate my plot on the 21st point?
    # tt = np.arange(0, 1, 0.05)
    # idx = np.random.randint(20)
    # xx = X_shuf_train[idx]  # (20, )
    # y = y_shuf_train[idx]  # (1)
    # X_linear = np.stack((np.ones(col - 1), tt), axis=1)  # (20, 2)
    # w_linear = np.linalg.lstsq(X_linear, xx, rcond=None)[0]  # rcond ?
    # y_linear = X_linear.dot(w_linear)
    # X_quartic = np.stack((np.ones(col - 1), tt, tt ** 2, tt ** 3, tt ** 4), axis=1)  # (20, 5)
    # w_quartic = np.linalg.lstsq(X_quartic, xx, rcond=None)[0]
    # y_quartic = X_quartic.dot(w_quartic)
    # plt.clf()
    # plt.plot(tt, xx, 'r+', label="training points")  # training points
    # plt.plot(1, y, 'g+', label="predicting point")
    # plt.plot(tt, y_linear, 'b-', label="linear fit")  # straight line fit
    # plt.plot(tt, y_quartic, 'p-', label="quartic fit")  # quartic fit
    # plt.legend()
    # plt.show()

    # # Finding best param i, j
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
