import numpy as np
from keras.datasets import mnist


def mnist_process(x):
    x = x.astype(np.float32) / 255.0
    return x


def mnist_data():
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    return mnist_process(xtrain), mnist_process(xtest)

def prepare_mnist_with_anomalies():

    (x_train, x_trainLabels), (x_test, x_testLabels) = mnist.load_data()

    labels = x_trainLabels
    data = x_train


    k_ones = np.where(labels == 1)
    label_ones = labels[k_ones]
    data_ones = data[k_ones]

    k_sevens = np.where(labels == 7)
    label_sevens = labels[k_sevens]
    data_sevens = data[k_sevens]

    data_ones = data_ones[:5000]
    label_ones = label_ones[:5000]
    data_sevens = data_sevens[:10]
    label_sevens = label_sevens[:10]

    data = np.concatenate((data_ones, data_sevens), axis=0)
    label = np.concatenate((label_ones, label_sevens), axis=0)
    label[0:10] = 1
    label[5000:5010] = 7



    return [mnist_process(data_ones),label_ones, mnist_process(data_sevens),label_sevens]
