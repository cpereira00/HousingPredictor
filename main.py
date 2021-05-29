import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = np.loadtxt("cleanedDataSet.txt", delimiter=",")
#print(X)
y = data[:, 0] #prices aka ouputs
Xold = data[:,1:4] #feature matrix(array)
m = y.size #number of training examples

#split data into train and test sets
X_train, X_test, y_train,y_test = train_test_split(Xold,y,test_size=.20,random_state=42)
print(X_train)
print("*********")
print(X_test)
print("*********")
print(y_train)
print("*********")
print(y_test)



plt.plot(X_train[:,0],y_train,'o')
plt.title("Number of Bedrooms vs The Price of The House")
plt.xlabel("# of bedrooms")
plt.ylabel("Price of house")
plt.show()

plt.plot(X_train[:,1],y_train,'o')
plt.title("Number of Bathrooms vs The Price of The House")
plt.xlabel("# of Bathrooms")
plt.ylabel("Price of house")
plt.show()

plt.plot(X_train[:,2],y_train,'o')
plt.title("Size of House vs The Price of The House")
plt.xlabel("Size in sqft")
plt.ylabel("Price of house")
plt.show()

#print(m)
#num_rows, num_cols = X.shape #print(num_rows, num_cols)

thetazero = np.ones((m,1))

#feature normalizer

X = np.column_stack((thetazero,Xold))


alpha = .1
num_iters = 600
