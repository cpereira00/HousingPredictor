import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data = np.loadtxt("cleanedDataSet.txt", delimiter=",")
#print(X)
y = data[:, 0] #prices aka outputs
Xold = data[:,1:4] #feature matrix(array)
m = y.size #number of training examples

from sklearn.model_selection import train_test_split
#split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(Xold,y,test_size=.30,random_state=42)

# print(X_train) #[168,3]
# print("*********")
# print(X_test) #[42,3]
# print("*********")
# print(y_train)
# print("*********")
# print(y_test)

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


#feature scaling using standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train[:, 0:] = scaler.fit_transform(X_train[:, 0:])
X_test[:, 0:] = scaler.fit_transform(X_test[:, 0:])

# print(X_train)
# print(len(X_train),len(X_train[0]))
# print('------')
# print(len(X_test))


#train the model
reg = LinearRegression().fit(X_train, y_train)  #<-- needs fixing

print(y_test)

# Make predictions using the testing set
housing_y_pred = reg.predict(X_test)
print(housing_y_pred)
# the coefficients theta
print('Coefficients: \n', reg.coef_)  #<-- needs fixing

# calculate metrics
from sklearn.metrics import mean_squared_error, r2_score
# mean square error
print('Mean squared error: %.2f' % mean_squared_error(y_test, housing_y_pred))
# Coefficient of determination
print('Coefficient of determination: %.2f' % r2_score(y_test, housing_y_pred))
print('score: %.2f' % reg.score(X_test, y_test))

# plot outputs
plt.scatter(X_test[:, 2], y_test, color="black")
plt.plot(X_test, housing_y_pred, color= 'blue',linewidth=2) #<--fixing
plt.title("Housing Prediction")
plt.ylabel("Price of house")
plt.xticks()

plt.show()
