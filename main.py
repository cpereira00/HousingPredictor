import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import copy

data = np.loadtxt("cleanedDataSet.txt", delimiter=",")
y = data[:, 0] #prices aka outputs
Xold = data[:,1:4] #feature matrix(array)
m = y.size #number of training examples

from sklearn.model_selection import train_test_split
#split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(Xold,y,test_size=.30,random_state=42)

X_testOld = copy.deepcopy(X_test)
X_trainOld = copy.deepcopy(X_train)

plt.figure("Figure1-numberBedroomsVprice")
plt.plot(X_train[:,0],y_train,'o')
plt.title("Figure 1: Number of Bedrooms vs The Price of The House")
plt.xlabel("# of bedrooms")
plt.ylabel("Price of house")
plt.show()

plt.figure("Figure2-numberBathroomsVprice")
plt.plot(X_train[:,1],y_train,'o')
plt.title("Figure 2: Number of Bathrooms vs The Price of The House")
plt.xlabel("# of Bathrooms")
plt.ylabel("Price of house")
plt.show()

plt.figure("Figure3-sizeHouseVprice")
plt.plot(X_train[:,2],y_train,'o')
plt.title("Figure 3: Size of House vs The Price of The House")
plt.xlabel("Size in sqft")
plt.ylabel("Price of house")
plt.show()


#feature scaling using standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#train the model
reg = LinearRegression().fit(X_train, y_train)


# Make predictions using the testing set
housing_y_pred = reg.predict(X_test)
#print(housing_y_pred)

# the coefficients theta
print('Coefficients: \n', reg.coef_)

# calculate metrics
from sklearn.metrics import mean_squared_error, r2_score
# mean square error
print('Mean squared error: %.2f' % mean_squared_error(y_test, housing_y_pred))
# Coefficient of determination (score)
print('score: %.2f' % r2_score(y_test, housing_y_pred))
print('root mean square error: ',np.sqrt(mean_squared_error(y_test,housing_y_pred)))


# plot outputs
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure("figure4-predictionVtestSet")
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X_testOld[:, 0],X_testOld[:, 2],y_test,color='black', alpha=.7, s=5)
#ax.scatter(X_trainOld[:, 0],X_trainOld[:, 2],y_train,color='orange', alpha=.7, s=3)


ax.plot_trisurf(X_testOld[:, 0],X_testOld[:, 2], housing_y_pred, color= 'lightblue',linewidth=3)

plt.title("figure 4: Housing Prediction")
ax.set_xlabel("# of Bedrooms")
ax.set_ylabel("Size of House in sqft")
ax.set_zlabel("Price of House")


plt.show()
