# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score


# plt.show()

# %%
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
# %%


raw_df.head()
# %%
from sklearn.datasets import fetch_california_housing
housing=fetch_california_housing()
# %%
housing.keys()
# %%
## Lets check the description of the dataset
print(housing.DESCR)
#%%
dataset=pd.DataFrame(housing.data,columns=housing.feature_names)
dataset['Price']=housing.target

# %%
dataset.head()

# %%
dataset.info()

# %%
dataset.describe()

# %%
dataset.corr()

# %%
sns.pairplot(dataset)

# %%
plt.scatter(dataset['Population'],dataset['Price'])
plt.xlabel("Population")
plt.ylabel("Price")
# %%
sns.regplot(x="Population",y="Price",data=dataset)
# %%
sns.regplot(x="MedInc",y="Price",data=dataset)
# %%
## Independent and Dependent features

X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]
# %%
X.head()

# %%
y
# %%
##Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

# %%
## Standardize the dataset
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
# %%
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# %%
X_train
# %%
X_test
# %%
import pickle
pickle.dump(scaler,open('scaling.pkl','wb'))
# %%
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(X_train,y_train)

# %%
print(regression.coef_)

# %%
print(regression.intercept_)

# %%
### Prediction With Test Data
reg_pred=regression.predict(X_test)

## plot a scatter plot for the prediction
plt.scatter(y_test,reg_pred)

# %%
## Residuals
residuals=y_test-reg_pred
# %%
residuals
# %%
## Plot this residuals 

sns.displot(residuals,kind="kde")
# %%
## Scatter plot with respect to prediction and residuals
## uniform distribution
plt.scatter(reg_pred,residuals)
# %%
print(mean_absolute_error(y_test,reg_pred))
print(mean_squared_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))
# %%

score=r2_score(y_test,reg_pred)
print(score)
# %%
#display adjusted R-squared
1 - (1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
# %%
housing.data[0].reshape(1,-1)
# %%
##transformation of new data
scaler.transform(housing.data[0].reshape(1,-1))
# %%
regression.predict(scaler.transform(housing.data[0].reshape(1,-1)))
# %%
import pickle
# %%
pickle.dump(regression,open('regmodel.pkl','wb'))
pickled_model=pickle.load(open('regmodel.pkl','rb'))
## Prediction
pickled_model.predict(scaler.transform(housing.data[0].reshape(1,-1)))
# %%
