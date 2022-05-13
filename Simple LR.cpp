import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression


url='https://raw.githubusercontent.com/BhushanDhamankar/ML_Datasets/main/USA_Housing.csv'
df=pd.read_csv(url)
df.head()


df.drop('Address',axis=1,inplace=True)
df.head()


df.corr()


df.drop('Avg. Area Number of Bedrooms',axis=1,inplace=True)


X=df.drop('Price',axis=1)
Y=df['Price']


from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


LR=LinearRegression()
print("Linear Regression:-")
LR.fit(x_train,y_train)
pred=LR.predict(x_test)
print(mean_squared_error(y_test,pred))
print(mean_absolute_error(y_test,pred))
print(r2_score(y_test,pred))


LR.score(x_test,y_test)*100
