import pandas as pd

dataset=pd.read_csv('Your_physical_condition_in_the_last_15_days.csv')

x=dataset.iloc[:,[0,2,3,4,5,6]].values
y=dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#SVM Algorithm
from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)

#K-Nearest Neighbors Algorithm
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
#classifier.fit(x_train,y_train)

predic=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,predic)

predic=classifier.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,predic)