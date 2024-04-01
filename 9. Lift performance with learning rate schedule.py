from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import FitFailedWarning
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=FutureWarning)
simplefilter("ignore", category=FitFailedWarning)

import  math
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from pandas import read_csv



np.random.seed(7)

filename = "C:\Datasets\ionosphere.csv"
dataset = read_csv(filename)
array = dataset.values
x = array[:,0:34].astype(float)
y = array[:,34]


#Encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

#Split data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#Define Model
model = Sequential()
model.add(Dense(34, input_dim=34, bias_initializer='uniform', activation='relu'))
model.add(Dense(17, bias_initializer='uniform', activation='relu'))
model.add(Dense(1, bias_initializer='uniform', activation='sigmoid'))

#Compile model
epochs = 50
learning_rate = 0.01
decay_rate = learning_rate/epochs
sgd = SGD(learning_rate=learning_rate, momentum=0.9, nesterov=False, decay=decay_rate)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Fit model
model.fit(x, y, epochs=50, validation_data=(x_test, y_test), batch_size=10, shuffle=True)

#Evaluate model
estimator = []
estimator.append(('Standardize', StandardScaler()))
estimator.append(('mlp', KerasClassifier(build_fn=model, epochs=50, batch_size=5)))
pipeline = Pipeline(estimator)


kfold = StratifiedKFold(n_splits=10, shuffle=False)
result = cross_val_score(pipeline, x, y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (result.mean()*100, result.std()*100))


#Learning Rate schedule callbacks
lrate = LearningRateScheduler(decay_rate)
callbacks_list = [lrate]


#Create Checkpoints
filepath = "C:\Predictive models\ionosphere.weights.best.hdf5"
checkpoint =  ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, save_weights_only=False, mode='max')
callbacks_list = [checkpoint]


#Fit model
model.fit(x, y, batch_size=5, epochs=50, validation_split=0.33, validation_data=(x_test,y_test), callbacks=callbacks_list)