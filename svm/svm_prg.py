import numpy as np
from sklearn import preprocessing,cross_validation,neighbors,svm
import pandas as pd
import pickle

df=pd.read_csv('breast-cancer-wisconsin.data')
#replacing the missing data
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)

#defining the labels and features
X=np.array(df.drop(['class'],1))
y=np.array(df['class'])

#creating the training and testing samples
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)

#defining the classifier
# clf=svm.SVC()
# clf.fit(X_train,y_train)
example_measures = np.array([4,2,1,1,1,2,3,2,1])
#if the length of the example_measures array is long then in reshape its x.reshape(len(example_measures),-1)
example_measures=example_measures.reshape(1,-1)

# with open("svm.pickle",'wb') as f:
#     pickle.dump(clf,f)
pickle_in=open("svm.pickle",'rb')
clf=pickle.load(pickle_in)
accuracy=clf.score(X_test,y_test)
print("accuracy:",accuracy)
prediction=clf.predict(example_measures)
print("prediction:",prediction)
