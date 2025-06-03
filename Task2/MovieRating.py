
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
import seaborn as sns

path="C:\\Users\\USER\\Downloads\\IMDb Movies India.xlsx"
data=pd.read_excel(path)
# print(data.head(5))

print(data.columns)

print(data.info())

# print(data.isnull().sum())

# deleting row of table based on null value of rating 

data=data.dropna(subset=['Rating'])

y=data['Rating']

data.drop(['Name','Rating'],axis=1,inplace=True)

data["Duration"]=data["Duration"].str.replace("min","",regex=False)
data["Duration"]=pd.to_numeric(data["Duration"],errors="coerce")

data['Votes']=pd.to_numeric(data["Votes"],errors="coerce")
data["Year"]=pd.to_numeric(data["Year"],errors="coerce")

data["Duration"].fillna(data["Duration"].median(),inplace=True)

list=['Genre','Director',
       'Actor 1', 'Actor 2', 'Actor 3']

data[list]=data[list].fillna("other")

# print(data.isnull().sum())

x=data[['Year', 'Duration', 'Genre', 'Votes', 'Director', 'Actor 1', 'Actor 2',
       'Actor 3']]





# Linear regression
# It assumes a straight, simple relationship 
# (more votes → higher rating, longer duration → higher rating, etc.).
# It cannot handle complex patterns like 
# "If Director is Rajkumar Hirani AND Genre is Drama, THEN rating is 9."


# RandomForestRegressor
#best for Complex relationships (non-linear)
# only work with numbers


# map string into integer
x = pd.get_dummies(x, columns=['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], drop_first=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


model=RandomForestRegressor()

model.fit(x_train,y_train)


pred=model.predict(x_test)

print(pred)

print("R2_SCORE",r2_score(y_test,pred))
print("RMSE",mean_squared_error(y_test,pred))















