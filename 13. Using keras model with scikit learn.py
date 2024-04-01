from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
import numpy
from numpy import mean
from pandas import read_csv

#Function to create model for keras classifier
def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, bias_initializer='uniform', activation='relu'))
    model.add(Dense(8, bias_initializer='uniform', activation='relu'))
    model.add(Dense(1, bias_initializer='uniform', activation='sigmoid'))

#Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#Load Data
filename = "C:\Datasets\pima-indians-diabetes.csv"
dataset = read_csv(filename)
array = dataset.values
x = array[:,0:8]
y = array[:,8]

model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)

#Evaluate using 10 fold cross-validation
kfold = StratifiedKFold(n_splits=10, shuffle=True)
result = cross_val_score(model, x, y, cv=kfold)
print(result.mean)