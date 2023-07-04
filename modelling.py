import tabular_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
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
from regression_hyper import parameter_grid_RandomForest, parameter_grid_DecisionTree, parameter_grid_GradientBoost, parameter_grid_SGDRegressor


import itertools

class BaseModel():
    ''' This is a class that serves as the foundation for other model classes. It contains common methods 
    and attributes that can be shared among different models. The '__init__' method initializes the BaseModel
    with a configuration dictionary and sets the model type '''

    def __init__(self, model_type, test_size = 0.3):

        self.model = None
        
        self.model_type = model_type
        
        self.test_size = test_size

        pass

    def _create_model_instance(self) -> type: 
               
        if self.model_type == "RandomForest":
            model_instance = RandomForestRegressor()
            model_instance.estimators_ = []
        elif self.model_type == "DecisionTree":
            model_instance = DecisionTreeRegressor()
        elif self.model_type == "SGDRegressor":
            model_instance = SGDRegressor()
        elif self.model_type == "GradientBoost":
            model_instance = GradientBoostingRegressor()
        #else:
            #raise ValueError(f"Invalid model type: {self.model_type}")

        return model_instance

    def train_test_split(self, dataset, test_size):

        ''' Implement train-test split logic '''
        X, y = dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state= 0)

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state= 0)

        train_set = (X_train, y_train)
        test_set = (X_test, y_test)
        val_set = (X_val, y_val)
        
        return train_set, test_set, val_set


    def preprocess(self, X):
        
        # Perform common preprocessing steps
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        
        return X
    
    def fit(self, train_set):
        X_train, y_train = train_set

        # Perform any preprocessing steps specific to the BaseModel
        X_train = self.preprocess(X_train)

        # Create and fit the model instance
        self.model = self._create_model_instance()
        self.model.fit(X_train, y_train)

        return self.model

    

    def evaluate(self, test_set):
        # Implement the evaluation logic for the model

        X_test, y_test = test_set

        scaler = StandardScaler().fit(X_test) # scaling the data
        X_test = scaler.transform(X_test)

        y_pred = self.model.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print('R2 SCORE IS:', r2, '/n MSE is', mse)

        return r2

class RegressionModel(BaseModel):
    ''' Inherits from the BaseModel class and represents a specific regression model. It overrides the necessary
    methods such as fit, evaluate, and load_hyperparameters, with specific logic for regression models.'''

    def __init__(self, model_type=None, test_size=0.3):
        super().__init__(model_type, test_size)
        self.hyperparameters = None
        self.metrics = None
        self.best_estimator = None
       

    def load_hyperparameters(self, hyperparameters):
        # Implement logic to load the hyperparameters for the specific regression model

        if self.model is not None:
            self.model.set_params(**hyperparameters)
        
    def preprocess(self, X):
        # Perform regression-specific preprocessing steps
        X = super().preprocess(X)
       
        return X

    def fit(self, train_set):
        X_train, y_train = train_set

        # Perform regression-specific preprocessing steps
        X_train = self.preprocess(X_train)

        # Create and fit the regression model instance
        self.model = self._create_model_instance()

        if self.hyperparameters is not None:
            self.load_hyperparameters(self.hyperparameters)


        self.model.fit(X_train, y_train)

    
    def evaluate(self, test_set):
        X_test, y_test = test_set

        # Perform regression-specific preprocessing steps on test data
        X_test = self.preprocess(X_test)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate evaluation metrics specific to regression models
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.metrics = {
            "MSE": mse,
            "R2": r2
        }

        return r2


  ## not bound to the class or its instances, it behaves like a regular function
    def first_model(self, train_set, test_set) -> dict:
        '''Implement logic for first_model function'''
        base_model = BaseModel(model_type= self.model_type)
        myModel = base_model._create_model_instance()
        myModel = self.fit(train_set= train_set)


        X_train, y_train = train_set
        X_test, y_test = test_set

        # making predictions
        y_train_pred = myModel.predict(X_train) 
        y_test_pred = myModel.predict(X_test)

        # calculating RMSE 
        rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
        rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)

        # calculating R2 score
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        metrics = {
            "RMSE (training set)": rmse_train,
            "RMSE (test set)": rmse_test,
            "R2 score (training set)": r2_train,
            "R2 score (test set)": r2_test
        }

        return metrics
    
    def custom_tune_regression_hyperparameters(self, train_set, val_set, grid):
        X_train, y_train = train_set
        X_val, y_val = val_set

        combinations = itertools.product(*grid.values())
        best_score = 0.0
        best_hyperparameters = None

        for combination in combinations:
            hyperparameters = dict(zip(grid.keys(), combination))

            # Create a new instance of the model
            model = self._create_model_instance()

            # Set the hyperparameters
            model.set_params(**hyperparameters)

            # Train the model
            model.fit(X_train, y_train)

            # Evaluate on the validation set
            y_val_pred = model.predict(X_val)
            r2_val = r2_score(y_val, y_val_pred)

            if r2_val > best_score:
                best_score = r2_val
                best_hyperparameters = hyperparameters

        return best_score, best_hyperparameters
    
    
    def tune_regression_model_hyperparameters(self, train_set, grid):

        X_train, y_train = train_set

        model = self._create_model_instance()
        

        grid_search = GridSearchCV(estimator= model, param_grid= grid, scoring= 'r2', cv= 5)

        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_
        best_score = grid_search.best_score_
        cv_results = grid_search.cv_results_

        return best_params, best_estimator, best_score, cv_results

    
    def save_model(self, model, folder):
       
        os.makedirs(folder, exist_ok=True)
        dump(model.model, os.path.join(folder, 'model.joblib'))

        with open(os.path.join(folder, 'hyperparameters.json'), 'w') as hyperparameters_file:
            json.dump(model.hyperparameters, hyperparameters_file)
        
        with open(os.path.join(folder, 'metrics.json'), 'w') as metrics_file:
            json.dump(model.metrics, metrics_file)
    

    
class ModelsEvaluator:
    def __init__(self):
        self.models = {
            'SGDRegressor': parameter_grid_SGDRegressor,
            'DecisionTree': parameter_grid_DecisionTree,
            'RandomForest': parameter_grid_RandomForest,
            'GradientBoost': parameter_grid_GradientBoost
        }

    def evaluate_all_models(self, train_set, test_set):

        for model_name, parameter_grid in self.models.items():
            regression_model = RegressionModel(model_type=model_name)
            best_params, best_estimator, best_score, _ = regression_model.tune_regression_model_hyperparameters(
                train_set=train_set, grid=parameter_grid
            )

            regression_model.load_hyperparameters(best_params)
            regression_model.best_estimator = best_estimator
            regression_model.fit(train_set=train_set)
            r2_score = regression_model.evaluate(test_set=test_set)

            folder_path = os.path.join('models', 'regression', model_name)
            regression_model.save_model(folder=folder_path, model= regression_model)

    def save_model(self, model, folder):
       
        os.makedirs(folder, exist_ok=True)
        dump(model.model, os.path.join(folder, 'model.joblib'))

        with open(os.path.join(folder, 'hyperparameters.json'), 'w') as hyperparameters_file:
            json.dump(model.hyperparameters, hyperparameters_file)
        
        with open(os.path.join(folder, 'metrics.json'), 'w') as metrics_file:
            json.dump(model.metrics, metrics_file)

        

        


    
df = pd.read_csv('airbnb-property-listings/listing.csv')
df_1= tabular_data.clean_tabular_data(raw_dataframe= df)
dataset =tabular_data.load_airbnb(df= df_1, label= 'Price_Night')

X, y = dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 0)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, random_state= 0, test_size= 0.5)

train_set = (X_train, y_train)
test_set = (X_test, y_test)
val_set = (X_val, y_val)



# Instantiate the ModelsEvaluator class and evaluate all models
models_evaluator = ModelsEvaluator()
models_evaluator.evaluate_all_models(train_set, test_set)

'''
base_model = BaseModel(model_type="SGDRegressor")
train_set, test_set, val_set = base_model.train_test_split(dataset= dataset, test_size= 0.3)
base_model.fit(train_set= train_set)
base_model.evaluate(test_set= test_set)
'''

'''
regression_model = RegressionModel(model_type= 'SGDRegressor')
train_set, test_set, val_set = regression_model.train_test_split(dataset= dataset, test_size= 0.3)
grid = parameter_grid_SGDRegressor
best_score, best_hyperparameters = regression_model.custom_tune_regression_hyperparameters(train_set= train_set, val_set= val_set, grid= grid)



print(best_score, best_hyperparameters)
'''

'''

regression_model = RegressionModel(model_type="SGDRegressor")
train_set, test_set, val_set = regression_model.train_test_split(dataset=dataset, test_size= 0.3)

best_params, best_estimator, best_score, cv_results = regression_model.tune_regression_model_hyperparameters(train_set=train_set, grid= parameter_grid_SGDRegressor)


regression_model.hyperparameters = best_params
regression_model.best_estimator = best_estimator
regression_model.metrics = best_score

print('Best hyperparameters', regression_model.hyperparameters, 'Best metrics: ', regression_model.metrics)

'''

#regression_model.save_model(folder='models/regression/SGDRegressor')




