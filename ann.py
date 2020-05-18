#classification problem-predict
#which customers will leave the bank
#data preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,[-1]].values

#Encode the categorical variables
#since countries do not have any ordinal value, create dummy 
#variable using OneHotEncoder
labelEndcode1 = LabelEncoder()
X[:,1] = labelEndcode1.fit_transform(X[:,1])

labelEndcode2 = LabelEncoder()
X[:,2] = labelEndcode2.fit_transform(X[:,2])

#OneHot encode countries using ColumnTransformer
ct = ColumnTransformer ([('one_hot_encoder', OneHotEncoder(categories='auto'), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:,1:]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42)

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

classifier = Sequential()

#rectifier function for input layers
#sigmoid function for output layer
#for hidden layer (after the input layer)
#6 = (11+1)/2
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

#output layer 
#note:softmax for dependant variable with more than 2 categories
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#compiling the ANN using stochastic gradient descent
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,y_train,batch_size=10,epochs=100)

y_pred = classifier.predict(X_test)
#Now y_pred is in terms of true and false
y_pred = (y_pred > 0.5)

#Validate
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#test on a single row- create a new entry and use dataset to encode value for country and gender
#scale since test and train was scaled.
test_prediction = classifier.predict(scalar.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
test_prediction = (test_prediction > 0.5)

#use cross val score for further evaluation using kerasclassifier
def create_model():
    classifier = Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classfier = KerasClassifier(build_fn=create_model,epochs=100,batch_size=10)
results = cross_val_score(classifier,X_test,y_test,cv=10)

accuracy_mean = results.mean()
accuracy_varience = results.std()