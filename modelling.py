import tabular_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv('airbnb-property-listings/listing.csv')
df_1= tabular_data.clean_tabular_data(raw_dataframe= df)
dataset =tabular_data.load_airbnb(df= df_1, label= 'Price_Night')


X, y = dataset # assigning features(X) and labels (y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3) # splitting dataset

regressor_model = SGDRegressor() # model
regressor_model.fit(X_train,y_train)
y_pred = regressor_model.predict(X_test)

# metrics

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# plotting

plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.show()








