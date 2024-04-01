from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from pandas import read_csv
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

#Load Data
filename = "C:\Datasets\iris.csv"
dataset = read_csv(filename)
array = dataset.values
x = array[:,0:4].astype(float)
y = array[:,4]

#Encode class values as integer
encoder = LabelEncoder()
encoder.fit(y)
encoder_y = encoder.transform(y)

#Convert integer to dummy variables with one hot encoding
dummy_y = np_utils.to_categorical(encoder_y)

#Define baseline model
def baseline_model():
    model = Sequential()
    model.add(Dense(4, input_dim=4, bias_initializer='normal', activation='relu'))
    model.add(Dense(3, bias_initializer='normal', activation='relu'))

#Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=150, batch_size=5)
kfold = KFold(n_splits=10, shuffle=False, random_state=0)
result = cross_val_score(estimator, x, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (result.mean()*100, result.std()*100))
