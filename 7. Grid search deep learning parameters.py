from keras.models import Sequential
from keras.layers import Dense
from pandas import read_csv
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy





filename = "C:\Datasets\pima-indians-diabetes.csv"
dataset = read_csv(filename)
array = dataset.values
x = array[:,0:8]
y = array[:,8]


#Function to create model for keras classifier
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model = KerasClassifier(build_fn=model)

#Grid Search
epochs = [50,100,150]
batches = [5,10,15]

param_grid = dict(batch_size=batches, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(x,y)

#Summarize result
print("Best: %f using %s" % grid_result.best_score_, grid_result.best_params_)
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(mean, stdev, param)