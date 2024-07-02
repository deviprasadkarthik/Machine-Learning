import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn 
data_set=pd.read_csv(r"/Users/deviprasadkarthikp/Downloads/Dataset_master - Simple Linear Regression (Salar.csv")
X=data_set.iloc[:,:-1].values # splittng dependent and independent variable 
y=data_set.iloc[:,-1].values


# splitting train and test data
from sklearn.model_selection   import train_test_split
X_train,X_test,y_train,y_test=train_test_split (X,y,test_size=0.3,random_state=42)

#training the model

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)

# predicting
y_pred=reg.predict(X_test)

#visualizing  training set

plt.scatter(X_train,y_train)
plt.title("CTC VS YEARS OF EXPERIENCE")

plt.xlabel("EXPERIENCE")
plt.ylabel("CTC")
plt.plot(X_train,reg.predict(X_train),color='red')
plt.show()


#visualizing  testing set

plt.scatter(X_test,y_test)
plt.title("CTC VS YEARS OF EXPERIENCE")

plt.xlabel("EXPERIENCE")
plt.ylabel("CTC")
plt.plot(X_train,reg.predict(X_train),color='red')
plt.show()






