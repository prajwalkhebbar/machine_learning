import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
style.use('ggplot')
quandl.ApiConfig.api_key = 'fLxFhAe4-_DMQy-spe2Y'

df = quandl.get("WIKI/GOOGL")

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]

#highlow percentage
df['HL_PCT']=(df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100.0

#percentage change
df['PCT_changes']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_changes','Adj. Volume']]

# print(df.head())

forecast_col  = 'Adj. Close'
#filling the missing columns with values
df.fillna(-99999, inplace=True)
#basically figuring out by how many days ahead we want the forecast to be
#here 0.1 means that we want it 10 days to the future if the total no of days was 100
forecast_out = int(math.ceil(0.01*len(df)))
print("forecast_out days:",forecast_out)
#here we are shifting the col by the amount of days we want to predict,- meaning shifting the data up
df['label']  = df[forecast_col].shift(-forecast_out)

#creating the x and y features and lables
X=np.array(df.drop(['label'],1))
#scaling is getting the value bw 0 and 1 so that while calculating
#no features overwhelm each other and its faster to calculate if the values are in bw -1 and 1
X=preprocessing.scale(X)
#X_lately is the data we are predicting i.e last 1 or 10% of the data
X_lately=X[-forecast_out:]
X=X[:-forecast_out]

#remove the added data before
df.dropna(inplace=True)

y=np.array(df['label'])
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#building the classifier
# clf = LinearRegression(n_jobs=-1)
# clf.fit(X_train,y_train)

##in order to store the classifier
# with open('LinearRegression.pickle','wb') as f:
#     pickle.dump(clf,f)
pickle_in=open('LinearRegression.pickle','rb')
clf=pickle.load(pickle_in)
accuracy=clf.score(X_test,y_test)
# print("accuracy:",accuracy)
forecast_set=clf.predict(X_lately)
df['Forecast']  =  np.nan
# print(forecast_set,accuracy,forecast_out)

#assingnin dates to the predicted data
#getting the last date from the dataframe
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day=86400
next_unix=last_unix+one_day

for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
