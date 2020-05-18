#classification problem-predict
#which customers will leave the bank
#data preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42)

scalar = StandardScaler()
X_train = scalar.fit_transform(X_test)
X_test = scalar.transform(X_test)




