from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy
import pandas as pd
from pandas import read_csv


filename = "C:\Datasets\pima-indians-diabetes.csv"
dataset = read_csv(filename)
array = dataset.values
x = array[:,0:8]
y = array[:,8]

#KFold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=False)

cvscores = []

for train, test in kfold.split(x, y):
    #Create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, bias_initializer='uniform', activation='relu'))
    model.add(Dense(8, bias_initializer='uniform', activation='relu'))
    model.add(Dense(1, bias_initializer='uniform', activation='sigmoid'))

#Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Evaluate model
scores = model.evaluate(x[test], y[test])

#Fit the model
model.fit(x[train], y[train], epochs=150, batch_size=10, verbose=0)
print('%s .2f%%' % model.metrics_names[1], scores[1]*100)
cvscores.append(scores[1]*100)


print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))