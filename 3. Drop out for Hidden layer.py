from warnings import simplefilter
from sklearn.exceptions import FitFailedWarning
simplefilter("ignore", category=FitFailedWarning)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pandas import read_csv
import numpy as np

np.random.seed(7)


#Load model
filename = "C:\Datasets\sonar.csv"
dataset = read_csv(filename)
array = dataset.values
x = array[:,0:60]
y = array[:,60]

#Split data into test and training
x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.33, shuffle=False)

#Optimizer
sgd = SGD(learning_rate=0.01, momentum=0.9, decay=0.0, nesterov=False)

#Create model
model = Sequential()
model.add(Dropout(0.2))
model.add(Dense(60, input_dim=60, bias_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dense(30, bias_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dense(1, bias_initializer='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])


#Evaluate model
estimator = []
estimator.append(('Standardize', StandardScaler()))
estimator.append(('MLP', KerasClassifier(build_fn=model, epochs=50, batch_size=5)))
pipeline = Pipeline(estimator)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
result = cross_val_score(pipeline, x, y, cv=kfold)
print('Accuracy: %.2f%%  (%.2f%%)' % (result.mean()*100, result.std()*100))

#Fit model
model.fit(x, y, batch_size=5, epochs=50, validation_split=0.33, validation_data=(x_test,y_test))