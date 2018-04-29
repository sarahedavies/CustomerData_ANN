# Bank Data Artificial Neural Net

# Data Preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
from sklearn.externals import joblib


# Importing dataset
df = pd.read_csv('Churn_Modelling.csv')
X = df.iloc[:, 3:13].values 
y = df.iloc[:, 13].values

# Encoding categorical data
#Encoding gender:
le_gender = preprocessing.LabelEncoder()
X[:, 2] = le_gender.fit_transform(X[:, 2])

# Encoding country: use one-hot encoding to avoid nonsensical averages
le_country = preprocessing.LabelEncoder()
X[:, 1] = le_country.fit_transform(X[:, 1])
ohe_country = preprocessing.OneHotEncoder(categorical_features = [1]) 
X = ohe_country.fit_transform(X).toarray()  
X = X[:, 1:]

# Splitting dataset to train and test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state=0)

# Feature scaling
sc = preprocessing.StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
scaler_filename = input('*Enter filename for scaler to be saved: ') + '.bin'
joblib.dump(sc, open(scaler_filename, 'wb'))


# Training the ANN


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Constructing the neural network
classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier_filename = input('*Enter filename for classifier architecture to be saved: ') + '.bin'
with open(classifier_filename, 'w') as file: file.write(classifier.to_json())


# Fitting ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
trained_filename = input('\n*Enter filename for trained model to be saved: ') + '.h5'
classifier.save(trained_filename)

# Making predictions and evaluating
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

