from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas
from pandas import read_csv
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

#Load Data
filename = "C:\Datasets\sonar.csv"
dataset = read_csv(filename)
array = dataset.values
x = array[:,0:60].astype(float)
y = array[:,60]


#Define baseline model
model = Sequential()
model.add(Dense(60, input_dim=60, bias_initializer='normal', activation='relu'))
model.add(Dense(1, bias_initializer='normal', activation='sigmoid'))

#Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
return model


estimators = []
estimators.append(('standardsc', StandardScaler()))
estimators.append(('kerascl', KerasClassifier(build_fn=model, epochs=150, batch_size=5)))
pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=0)
result = cross_val_score(pipeline, x, y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (result.mean()*100, result.std()*100))