import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Salary_Data.csv')


df.head(10)


df.shape[0], df.shape[1]


df.columns


df.info()


df.describe()


X = df[['YearsExperience']].values
Y = df[['Salary']].values


plt.style.use('seaborn-darkgrid')
plt.scatter(X,Y)  


df.corr()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)


from sklearn.linear_model import LinearRegression
reg = LinearRegression()  #obj created
reg.fit(x_train, y_train)


y_pred = reg.predict(x_test)


reg.score(x_test, y_test)*100


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 


plt.figure()
plt.scatter(x_train, y_train, color = 'blue')
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, reg.predict(x_train), color = 'green')
plt.xticks(range(0,12))
plt.xlabel('Experience in Years')
plt.ylabel('Salary')


