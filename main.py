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

    # Curve fitting on a snippet of audio

