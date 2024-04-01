from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import FitFailedWarning
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=FutureWarning)
simplefilter("ignore", category=FitFailedWarning)

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from sklearn.pipeline import Pipeline
import numpy as np
from pandas import read_csv
import time



filename = "C:\Datasets\pima-indians-diabetes.csv"
dataset = read_csv(filename)
array = dataset.values
x = array[:,0:8]
y = array[:,8]

#Split data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#Define Model
model = Sequential()
model.add(Dense(12, input_dim=8, bias_initializer='uniform', activation='relu'))
model.add(Dense(8, bias_initializer='uniform', activation='relu'))
model.add(Dense(1, bias_initializer='uniform', activation='sigmoid'))

#Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#Fit model
model.fit(x, y, epochs=150, validation_data=(x_test, y_test), batch_size=10, shuffle=True, verbose=1)

#Evaluate model
estimator = []
estimator.append(('Standardize', StandardScaler()))
estimator.append(('MLP', KerasClassifier(build_fn=model, epochs=50, batch_size=5)))
pipeline = Pipeline(estimator)


kfold = StratifiedKFold(n_splits=10, shuffle=False)
result = cross_val_score(pipeline, x, y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (result.mean()*100, result.std()*100))

#Save model
model.save("C:\Predictive models\pima-deep.h5")