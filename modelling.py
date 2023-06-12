import tabular_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import statistics

''' It would be a good idea to split things into different functions and calling them later

        First function should be something like preparing the data and should prepare the dataset and scale the training data if necessary

        Second should do the work on trying out the SGDRegressor

        Third should be hyperparameter modelling

        Another function for plotting

'''

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

# using a standard scaler to scale input
scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

myModel = SGDRegressor(random_state= random_seed).fit(X_train, y_train)


y_train_pred = myModel.predict(X_train)
y_test_pred = myModel.predict(X_test)

print(y_train_pred.shape, y_test_pred.shape)




rmse_train = mean_squared_error(y_train, y_train_pred, squared= False)
rmse_test = mean_squared_error(y_test, y_test_pred, squared= False)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print("RMSE (training set):", rmse_train)
print("RMSE (test set):", rmse_test)
print("R2 (training set):", r2_train)
print("R2 (test set):", r2_test)


def plotting_prediction(y_pred, y_true, title=str):

    ''' Function to plot any model predictions'''

    samples = len(y_pred)
    plt.figure()
    plt.scatter(np.arange(samples), y_pred, c='r', label='predictions')
    plt.scatter(np.arange(samples), y_true, c='b', label='true labels', marker='x')
    plt.legend()
    plt.xlabel('Sample numbers')
    plt.ylabel('Values')
    plt.title(title)
    plt.show()

    return


def custom_tune_regression_model_hyperparameters(model_class: type, train_set, val_set, test_set, grid = dict):

 # Function to find best hyperparameters for the model
   

    X_train, y_train = train_set
    X_test, y_test = test_set
    X_val, y_val = val_set

    combinations = itertools.product(*grid.values()) 

    best_score = 0.0


    for combination in combinations:

        model = model_class(random_state = random_seed, **dict(zip(grid.keys(), combination))).fit(X_train, y_train)

        y_val_pred = model.predict(X_val)

        rmse_val = mean_squared_error(y_val, y_val_pred, squared= False)

        r2_val = r2_score(y_train, y_train_pred)


        if r2_val > best_score:
            best_score = r2_val
            best_hyperparameters = combination
        else:
            None


    return best_score, best_hyperparameters


parameter_grid = {

    'penalty': ['l1', 'l2', 'elasticnet', None],
    'loss' : ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'alpha' : [0.1, 0.01, 0.001, 0.0001],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'max_iter': [10000, 50000]

}

parameter_grid_test = {

    'penalty': ['l1', 'l2'],
    'loss' : ['squared_error', 'huber'],
    'alpha' : [0.1, 0.01],
    'learning_rate': ['constant', 'optimal']

}
    
# splitting test set into test and validation sets 
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size= 0.5, random_state= random_seed) 

train_set = (X_train, y_train)
test_set = (X_test, y_test)
val_set = (X_val, y_val)


best_scoring, best_parameters = custom_tune_regression_model_hyperparameters(model_class= SGDRegressor, train_set= train_set, val_set= val_set, test_set= test_set, grid= parameter_grid)

print('Best score is:', best_scoring)
print('Ideal hyperparameters for the model are:', best_parameters)




