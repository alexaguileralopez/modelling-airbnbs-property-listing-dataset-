import tabular_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import statistics
from joblib import dump, load
import json
import os


# setting random seed to obtain same predictions each time program is run, 
# given same input data and model

random_seed = 42
np.random.seed(random_seed)



df = pd.read_csv('airbnb-property-listings/listing.csv')
df_1= tabular_data.clean_tabular_data(raw_dataframe= df)
dataset =tabular_data.load_airbnb(df= df_1, label= 'Price_Night')



X, y = dataset # assigning features(X) and labels (y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= random_seed) # splitting dataset

# using a standard scaler to scale input
scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



def first_model(X_train = X_train, y_train= y_train, X_test=X_test, y_test=y_test):

    ''' Function to try the SGDRegressor using the data imported. Takes as arguments all of the labels and features and makes 
        a prediction for the testing set.
        It also measures rmse and r2 score for training and testing set and returns a dictionary including all these values.
    
    '''

    # creating model 
    myModel = SGDRegressor(random_state= random_seed)

    # training the model
    myModel.fit(X_train, y_train) 

    # making predictions
    y_train_pred = myModel.predict(X_train) 
    y_test_pred = myModel.predict(X_test)

    # calculating RMSE 
    rmse_train = mean_squared_error(y_train, y_train_pred, squared= False)
    rmse_test = mean_squared_error(y_test, y_test_pred, squared= False)

    # calculating R2 score
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    metrics= {
    "RMSE (training set)": rmse_train,
    "RMSE (test set)": rmse_test,
    "R2 score (training set)": r2_train,
    "R2 score (test set)": r2_test
    }

    return metrics


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

    ''' Function to find best hyperparameters for the model.
    An addition to the function should be to implement k-fold cross validation
 
    '''
   

    X_train, y_train = train_set
    X_test, y_test = test_set # not necessary for hyperparameter tuning
    X_val, y_val = val_set

    combinations = itertools.product(*grid.values()) 

    best_score = 0.0


    for combination in combinations:

        model = model_class(random_state = random_seed, **dict(zip(grid.keys(), combination))).fit(X_train, y_train)

        y_val_pred = model.predict(X_val)

        rmse_val = mean_squared_error(y_val, y_val_pred, squared= False)

        r2_val = r2_score(y_train, y_val_pred)


        if r2_val > best_score:
            best_score = r2_val
            best_hyperparameters = combination
        else:
            best_score = best_score
            best_hyperparameters = best_hyperparameters


    return best_score, best_hyperparameters

def tune_regression_model_hyperparameters(model_class: type, train_set, grid = dict):

    ''' Should eliminate test set and val set if not used.'''

    X_train, y_train = train_set


    model = model_class()

    grid_search = GridSearchCV(estimator= model, param_grid= grid, scoring= 'r2', cv= 5)

    grid_search.fit(X_train, y_train)

    # Retrieve the best hyperparameters
    best_params = grid_search.best_params_

    # Retrieve the best estimator
    best_estimator = grid_search.best_estimator_

    # Retrieve the best score
    best_score = grid_search.best_score_

    # Retrieve the detailed results
    cv_results = grid_search.cv_results_
 
    return best_params, best_estimator, best_score, cv_results

def tune_classification_model_hyperparameters(model_class: type, train_set, val_set, test_set, grid = dict):

    ''' function to find best hyperparameters for a classification model'''


    X_train, y_train = train_set
    X_test, y_test = test_set 
    X_val, y_val = val_set

    model = model_class()

    grid_search = GridSearchCV(estimator= model, param_grid= grid, scoring= 'accuracy', cv= 5)

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    cv_results = grid_search.cv_results_

    y_val_pred = best_estimator.predict(X_val)

    # Calculate performance metrics on the validation set
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, average= 'macro')
    recall = recall_score(y_val, y_val_pred, average= 'macro')
    f1 = f1_score(y_val, y_val_pred, average= 'macro')

    # Store the performance metrics in a dictionary
    performance_metrics = {
        'validation_accuracy': accuracy,
        'validation_precision': precision,
        'validation_recall': recall,
        'validation_f1': f1
    }


    return best_params, best_estimator, performance_metrics

def save_model(model, hyperparameters, metrics, folder):

    # Create the directory structure if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Save the model in the folder
    dump(model, folder + '/model.joblib')

    # Save the hyperparameters to hyperparameters.json
    with open(folder + '/hyperparameters.json', 'w') as hyperparameters_file:
        json.dump(hyperparameters, hyperparameters_file)

    # Save the metrics to metrics.json
    with open(folder + '/metrics.json', 'w') as metrics_file:
        json.dump(metrics, metrics_file)

    return 

def evaluate_all_models(): 

    ''' 
        Function to evaluate the performance of SGDRegressor, Decision Trees, 
        Random Forests, and Gradient Boosting algorithms on the dataset
    '''
    models = {'SGDRegressor': (SGDRegressor, parameter_grid_SGDRegressor), 
              'DecisionTreeRegressor': (DecisionTreeRegressor, parameter_grid_DecisionTree),
              'RandomForestRegressor':  (RandomForestRegressor, parameter_grid_RandomForest),
              'GradientBoostingRegressor': (GradientBoostingRegressor, parameter_grid_GradientBoost)
                                            
                                            }

    ''' Would be best to import a parameter grid for each of the models, so as a dict?'''

    for model_name in models:

        model_class, parameter_grid = models[model_name]

        # Get the model name
        model_name = model_class.__name__



        best_params, best_estimator, best_score, cv_results, = tune_regression_model_hyperparameters(model_class= model_class, 
                                                            train_set= train_set, grid= parameter_grid)
        
        # Define the folder path for saving the model
        folder_path = os.path.join('models', 'regression', model_name)
        save_model(model= best_estimator, hyperparameters= best_params, metrics= best_score, folder= folder_path)

    return

def find_best_model():


    ''' This function finds the best model saved in the respository. 
    Its purpose is to be used after all models have been evaluated and saved.
    '''

    folder_path = 'models/regression'  

    best_metrics = 0.0


    subfolder_names = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

    for folder_name in subfolder_names:
        file_path = os.path.join('models/regression', folder_name, 'metrics.json')

        with open(file_path, 'r') as file:
            model_metrics = float(file.read())
            
            print(model_metrics)

            if model_metrics >= best_metrics:
                best_metrics = model_metrics
                best_model = folder_name

            else: 
                best_metrics = best_metrics


    model = load(os.path.join('models/regression', best_model, 'model.joblib'))

    with open(os.path.join('models', 'regression', best_model, 'hyperparameters.json'), 'r') as json_file:
        # Load the JSON data
        hyperparameters = json.load(json_file)

    return model, hyperparameters, best_metrics # cv results are not saved in the files, so not used here



''' separation between functions and data manipulation'''


df = pd.read_csv('airbnb-property-listings/listing.csv')
df_1= tabular_data.clean_tabular_data(raw_dataframe= df)
dataset =tabular_data.load_airbnb(df= df_1, label= 'Price_Night')

X, y = dataset # assigning features(X) and labels (y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= random_seed) # splitting dataset

# using a standard scaler to scale input
scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# splitting test set into test and validation sets 
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size= 0.5, random_state= random_seed) 

train_set = (X_train, y_train)
test_set = (X_test, y_test)
val_set = (X_val, y_val)

parameter_grid_SGDRegressor = {

    'penalty': ['l1', 'l2', 'elasticnet', None],
    'loss' : ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'alpha' : [0.1, 0.01, 0.001, 0.0001],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'max_iter': [1000, 10000, 50000]

}

parameter_grid_DecisionTree = {

    
    'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter' : ['best', 'random'],
    'max_depth' : [3,5,7],
    'min_samples_split' : [2, 5, 10 ],
    'min_samples_leaf': [1, 3, 5],
    

}

parameter_grid_RandomForest = {

    'n_estimators' : [100,200,300],
    'max_depth' : [5,10],
    'min_samples_split' : [2,5,10],
    'min_samples_leaf' : [1,3,5],

    'criterion' : ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'] 
}

parameter_grid_GradientBoost = {

    'learning_rate': [0.01, 0.1, 0.5],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5],
    'min_samples_split' : [2, 5, 10],
    'min_samples_leaf' : [1, 3, 5],
   
    'loss' : ['absolute_error', 'quantile', 'squared_error', 'huber'] 
}

    
parameter_grid_DecisionTree_classifier = {

    'max_depth' : [3,5,7], # how many levels of splits and branches the tree can have
    'min_samples_split' : [2, 5, 10 ],
    'min_samples_leaf': [1, 3, 5],
    'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter' : ['best', 'random'],
    
}

parameter_grid_RandomForest_classifier = {}



#best_scoring, best_parameters = custom_tune_regression_model_hyperparameters(model_class= SGDRegressor, train_set= train_set, val_set= val_set, test_set= test_set, grid= parameter_grid)

''' best_params, best_estimator, best_score, cv_results, = tune_regression_model_hyperparameters(model_class= SGDRegressor, train_set= train_set, val_set= val_set, test_set= test_set, grid= parameter_grid)

print('Best score is:', best_score)
print('Ideal hyperparameters for the model are:', best_params)

save_model(model= best_estimator, hyperparameters= best_params, metrics= best_score, folder= 'models/regression/linear_regression') '''

if __name__ == "__main__":

    evaluate_all_models()
    best_model = find_best_model()


   









