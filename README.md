# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Python libraries (NumPy, pandas, scikit-learn) and load the dataset for linear regression.
2. Split the dataset into training and testing sets.
3. Create and train the Stochastic Gradient Descent (SGD) Regressor model using the training data.
4. Evaluate the model performance using test data and calculate metrics such as Mean Squared Error (MSE) or R² score.

## Program:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

#Load the dataset
data=pd.read_csv("CarPrice_Assignment.csv")
print(data.head())
print(data.info())

#Data preprocessing
#Dropping unnecessary columns and handling categorical variables
data=data.drop(['CarName','car_ID'],axis=1)
data=pd.get_dummies(data,drop_first=True)

#Splitting the data into features and target variable
x=data.drop('price',axis=1)
y=data['price']

#Standardizing the data
scaler=StandardScaler()
x=scaler.fit_transform(x)
y=scaler.fit_transform(np.array(y).reshape(-1,1))

#Splitting the dataset into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#Creating the SGD Regressor Model
sgd_model=SGDRegressor(max_iter=1000,tol=1e-3)

#Fitting the model on the training data
sgd_model.fit(x_train,y_train)

#Making predicitons
y_pred=sgd_model.predict(x_test)

#Evaluating model performance
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

#print evaluation metrics
print('Name: SHRIHARI M')
print('Reg. No: 212225230265')
print('Mean Squared Error:',mse)
print('Mean Absolute Error:',mae)
print('R-squared Error:',r2)

#print model coefficients
print("\nModel Coefficients:")
print("Coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)

#Visualizing actual vs predicted prices
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red')
plt.show()
```

## Output:
<img width="990" height="428" alt="Screenshot 2026-02-23 142739" src="https://github.com/user-attachments/assets/4abe2cf8-7f76-4e04-8373-c8d414b1c7d8" />
<img width="994" height="646" alt="Screenshot 2026-02-23 142817" src="https://github.com/user-attachments/assets/54163b8c-4606-4c35-a16e-a9c6b84855c7" />
<img width="1016" height="655" alt="Screenshot 2026-02-23 142853" src="https://github.com/user-attachments/assets/b47482c3-12d3-42a3-9a80-4040283f48c6" />
## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
