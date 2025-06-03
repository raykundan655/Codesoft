import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=pd.read_excel("C:\\Users\\USER\\Downloads\\Titanic-Dataset.xlsx")
# print(data.head(5))
print(data.columns)

print(data.info())

print(data.isnull().sum())

avgAge=data["Age"].mean()

data["Age"].fillna(avgAge,inplace=True)

print(data.isnull().sum())

data.drop(["Ticket","Cabin","Name",],axis=1,inplace=True)

print(data.isnull().sum())

data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

print(data.isnull().sum())

# LogisticRegression do not work with non-numeric data. You must convert these to numeric format

x=data[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
       'Fare', 'Embarked']]
# pd.get_dummies() is a Pandas function used to convert categorical variables (like 'Sex' and 'Embarked')
#  into numerical dummy/indicator variables that machine learning models can understand.

x = pd.get_dummies(x, columns=['Sex', 'Embarked'], drop_first=True)

y=data['Survived']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression(max_iter=1000)
# The max_iter parameter stands for:
#  Maximum number of iterations for the solver (optimizer) to find the best weights.

model.fit(x_train,y_train)

pre=model.predict(x_test)

#  classification_report it give multiple matrix
#it include accuracy matrix,precision and 2 more
#  confusion_matrix

# print("Classification Report:\n", classification_report(y_test, pre))


accuracy = accuracy_score(y_test, pre)
# print("Accuracy:", accuracy) 




       





