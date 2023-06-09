import tabular_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

''' It would be a good idea to split things into different functions and calling them later'''

df = pd.read_csv('airbnb-property-listings/listing.csv')
df_1= tabular_data.clean_tabular_data(raw_dataframe= df)
dataset =tabular_data.load_airbnb(df= df_1, label= 'Price_Night')

# setting random seed to obtain same predictions each time program is run, 
# given same input data and model

random_seed = 42
np.random.seed(random_seed)

X, y = dataset # assigning features(X) and labels (y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= random_seed) # splitting dataset

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

myModel = SGDRegressor(random_state= random_seed).fit(X_train, y_train)

y_train_pred = myModel.predict(X_train)
y_test_pred = myModel.predict(X_test)

print(y_train_pred.shape, y_test_pred.shape)
#y_pred = myModel.predict(X_test)

samples = len(y_train_pred)
plt.figure()
plt.scatter(np.arange(samples), y_train_pred, c='r', label='predictions')
plt.scatter(np.arange(samples), y_train, c='b', label='true labels', marker='x')
plt.legend()
plt.xlabel('Sample numbers')
plt.ylabel('Values')
plt.show()


rmse_train = mean_squared_error(y_train, y_train_pred, squared= False)
rmse_test = mean_squared_error(y_test, y_test_pred, squared= False)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print("RMSE (training set):", rmse_train)
print("RMSE (test set):", rmse_test)
print("R2 (training set):", r2_train)
print("R2 (test set):", r2_test)
# evaluating regression model performance 
# comparing the result of the prediction to what it should deliver (y_test)

#rmse = mean_squared_error(y_test, y_pred, squared= False) # Root mean squared error, rooted. 
# A value of 0 indicates perfect fit of the model to the data
#r2 = r2_score(y_test, y_pred) # R-squared (Coefficient of Determination)
# A value of 1 indicates a perfect fit of the model to the data, while below 0 can indicate poor model performance.



# plotting

def custom_tune_regression_model_hyperparameters(model_class):

    return










