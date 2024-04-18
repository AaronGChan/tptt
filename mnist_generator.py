from sklearn.datasets import load_digits
import pandas as pd
from sklearn.model_selection import train_test_split
mnist = load_digits()

x = pd.DataFrame(mnist.data)
y = pd.DataFrame(mnist.target)

x = x[0:500]
y = y[0:500]
x = x / 255.0

train_size = 0.7
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)

# breakpoint()
breakpoint()
x_train.to_csv("mnist_8x8/train_X.csv", header=False, index=False)
x_test.to_csv("mnist_8x8/test_X.csv", header=False, index=False)
y_train.to_csv("mnist_8x8/train_Y.csv", header=False, index=False)
y_test.to_csv("mnist_8x8/test_Y.csv", header=False, index=False)