import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score


url=''
df=pd.read_csv(url)
df.head()


df.shape


df.info()


df.describe()


X=df.drop('Outcome',axis=1)
Y=df['Outcome']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.3)


x_train.head()


y_test.head()


from sklearn.svm import SVC
classifier = SVC(kernel='linear')
classifier.fit(x_train,y_train)


y_pred = classifier.predict(x_test)


from sklearn.metrics import confusion_matrix,accuracy_score, precision_score
print(accuracy_score(y_test,y_pred)*100)
print(precision_score(y_test,y_pred)*100)


cm=confusion_matrix(y_test,y_pred)
print(cm)


import seaborn as sns
sns.heatmap(cm,annot= True)
