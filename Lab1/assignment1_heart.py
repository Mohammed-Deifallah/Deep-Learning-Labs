# %xmode Plain
# %pdb on
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

dataset = pd.read_csv("heart.csv")

X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75, random_state=0)
feature_len = train_X.shape[1]

model = Sequential()
model.add(Dense(1, input_dim=feature_len, activation='sigmoid', kernel_regularizer=l2(.001)))
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adadelta')
model.fit(train_X, train_y, verbose=1, batch_size=1, epochs=150)
score, accuracy = model.evaluate(test_X, test_y, batch_size=16, verbose=0)
print("Test fraction correct (NN-Score) = {:.2f}".format(score))
print("Test fraction correct (NN-Accuracy) = {:.2f}".format(accuracy))
