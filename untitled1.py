import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
dataset= pd.read_csv ("Position_Salaries.csv")
x= dataset.iloc [ : ,  1] .values
y= dataset.iloc [ : , 2] .values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=1/4,random_state=0)


x=x.reshape(-1,1)
y=y.reshape(-1,1)



from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=4)
x_poly=poly.fit_transform(x)
lin_reg=LinearRegression()
lin_reg.fit(x_poly,y)
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(poly.fit_transform(x)),color='blue')
plt.show()














