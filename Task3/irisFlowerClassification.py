import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor 
# Goal: Predict continuous numeric values
from sklearn.ensemble import RandomForestClassifier 
# Goal: Predict a category or class label
# Predicting whether an email is spam or not, or whether a flower is Setosa, Versicolor, or Virginica
from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score,mean_squared_error;
from sklearn.metrics import accuracy_score,classification_report

path="C:\\Users\\USER\\Downloads\\IRIS.xlsx"
data=pd.read_excel(path)

# print(data.head(5))

print(data.info())

# print(data.isnull().sum())

data["species" ]=data["species" ].str.replace("Iris-","",regex=False)

# print(data.head(5))



y=data["species"]

# y=pd.get_dummies(y,columns=["species"],drop_first=True)



data.drop("species",axis=1,inplace=True)

print(data.head(5))

x=data

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier()

model.fit(x_train,y_train)

pred=model.predict(x_test)

# print(pred)

# print("r2_score",r2_score(pred,y_test))
# print("mean squared error",mean_squared_error(pred,y_test))
# both will give error beacuse it use to calculate regression not for classification





print("accuracy_score",accuracy_score(y_test,pred))

print("classification report",classification_report(y_test,pred))


# classification report give :
# Precision:	Out of all predicted X class, how many were correct?
# Recall	Out of all actual X class, how many did we correctly predict?
# F1-score	Harmonic mean of precision and recall.
# Accuracy	Overall correct predictions / total predictions.


