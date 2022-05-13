import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import datasets


iris=datasets.load_iris()
X=pd.DataFrame(iris.data)
Y=pd.DataFrame(iris.target)
  

data=pd.concat([X,Y],axis=1)
data.columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
data.head()


data.info()


data.describe()


X.head()


Y.head()


Y.value_counts()


X.shape


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
knn = KNeighborsClassifier(n_neighbors=5)


knn.fit(X_train, y_train)
pred=knn.predict(X_test)


cm=confusion_matrix(y_test,pred)
print(cm)


print(accuracy_score(y_test,pred)*100)


