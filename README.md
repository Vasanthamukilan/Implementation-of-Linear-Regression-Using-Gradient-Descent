# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
   1. Import the required library and read the dataset.
   2. Write a function to generate the cost function.
   3. Perform iterations by following gradient descent steps with learning rate.
   4. Plot the Cost function using Gradient Descent and predict the value. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Vasanthamukilan M
RegisterNumber: 212222230167
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=100):
  x=np.c_[np.ones(len(x1)),x1]
  theta=np.zeros(x.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions=(x).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learning_rate*(1/len(x1))*x.T.dot(errors)
  return theta
data=pd.read_csv('50_Startups.csv',header=None)
x=(data.iloc[1:,:-2].values)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
x1_Scaled=scaler.fit_transform(x1)
y1_Scaled=scaler.fit_transform(y)

theta=linear_regression(x1_Scaled, y1_Scaled)

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
   i.) Dataset:
   
   ![image](https://github.com/shoaib3136/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/117919362/40c5c830-3e53-44ff-aecc-66f36c9a7087)

   ii.) Predicted value:
   
   ![image](https://github.com/shoaib3136/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/117919362/9e64546a-9fb0-495f-9678-cdd6cf6f9cc5)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
