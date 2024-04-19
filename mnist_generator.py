from sklearn.datasets import load_digits
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.io
mnist = load_digits()

data_set = scipy.io.loadmat('MNIST_TrainSet_0to1_8x8pixel.mat')     #MNIST dataset in 8x8 resolution
data = data_set['number']
label_set = scipy.io.loadmat('MNIST_TrainSet_Label.mat')
label = label_set['label']
data = data.T
label = label.T
x1 = pd.DataFrame(data)
y1 = pd.DataFrame(label)

# x = pd.DataFrame(mnist.data)
# y = pd.DataFrame(mnist.target)

# import matplotlib.pyplot as plt
# image0 = x1.iloc[0].to_numpy().reshape(8,8)
# breakpoint()
# plt.imshow(image0)
# plt.show()
# breakpoint()
#x = x / 255.0

train_size = 0.7
x_train, x_test, y_train, y_test = train_test_split(x1, y1, train_size=train_size)
# breakpoint()
#breakpoint()
x_train.to_csv("mnist_8x8/train_X.csv", header=False, index=False)
x_test.to_csv("mnist_8x8/test_X.csv", header=False, index=False)
y_train.to_csv("mnist_8x8/train_Y.csv", header=False, index=False)
y_test.to_csv("mnist_8x8/test_Y.csv", header=False, index=False)