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
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import statistics
from joblib import dump, load
import json
import os
from classification_hyper import parameter_grid_DecisionTreeClassifier, parameter_grid_GradientBoostClassifier, parameter_grid_RandomForestClassifier, parameter_grid_LogisticRegressor
from regression_hyper import parameter_grid_DecisionTree, parameter_grid_GradientBoost, parameter_grid_RandomForest, parameter_grid_SGDRegressor
from modelling_v1 import BaseModel, RegressionModel


class BaseModel():

    def __init__(self, model_type, test_size = 0.3):

        self.model = None
        self.model_type = model_type
        self.test_size = test_size
        self.hyperparameters = None
        
        pass

    def _create_model_instance(self) -> type: 

        '''
        Building a model instance depending on model_type   
        '''

        if self.model_type == "RandomForestRegressor":
            model_instance = RandomForestRegressor()
            model_instance.estimators_ = []
        elif self.model_type == "RandomForestClassifier":
            model_instance = RandomForestClassifier()
        elif self.model_type == "DecisionTreeRegressor":
            model_instance = DecisionTreeRegressor()
        elif self.model_type == "DecisionTreeClassifier":
            model_instance = DecisionTreeClassifier()
        elif self.model_type == "SGDRegressor":
            model_instance = SGDRegressor()
        elif self.model_type == "LogisticRegression":
            model_instance = LogisticRegression()
        elif self.model_type == "GradientBoostRegressor":
            model_instance = GradientBoostingRegressor()
        elif self.model_type == "GradientBoostClassifier":
            model_instance = GradientBoostingClassifier()

        else:
            raise ValueError(f"Invalid model type: {self.model_type}")
        
        return model_instance
    
    def train_test_split(self, dataset, test_size):

        ''' Implementing train-test split logic '''
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

        '''Scaling data'''
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        return X
    
    def load_hyperparameters(self, hyperparameters):

        ''' Loads hyperparameters for the specific model'''

        if self.model is not None:
            self.model.set_params(**hyperparameters)
    
    def fit(self, train_set): 

        
        ''' Performs training logic for any model. First pre-processing, 
        creating model instance and training. '''
        
        X_train, y_train = train_set
        X_train = self.preprocess(X_train)
        self.model = self._create_model_instance()

        if self.hyperparameters is not None:
            self.load_hyperparameters(self.hyperparameters)

        self.model.fit(X_train, y_train)

        return self.model
    
    def evaluate(self, test_set, model_type):

        '''
        Performs evaluation of the model, with a two-case scenario of 
        regression and classification.
        '''

        X_test, y_test = test_set
        X_test = self.preprocess(X_test)
        y_pred = self.model.predict(X_test)

        if model_type == 'Regressor':

            '''Regression model evaluation. Makes predictions on the test data 
            after using fit method to train the model.'''

            mse = mean_squared_error(y_true= y_test, y_pred=y_pred)
            r2 = r2_score(y_true= y_test, y_pred=y_pred)

            self.metrics = {
                'y_pred': y_pred,
                'y_true' : y_test,
                'MSE' : mse,
                'R2' : r2

            }
            
            main_metric = r2
            
        
        elif model_type == 'Classifier': 

            '''Classification model evaluation. Makes predictions on the test data 
            after using fit method to train the model. Returns the F1 score, 
            which is what will be used to compare performance.'''

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average= 'macro')
            recall = recall_score(y_test, y_pred, average= 'macro')
            f1 = f1_score(y_test, y_pred, average = 'macro')

            self.metrics = {
                'y_pred' : y_pred,
                'y_true' : y_test,
                'Accuracy' : accuracy,
                'Precision' : precision,
                'Recall' : recall,
                'F1' : f1
            }

            main_metric = f1

        return main_metric
    
    def plot_results(self, model_type):

        ''' Method to plot the results of regression and classification models'''

        if model_type == 'Classifier':

            # Get the predicted and true labels from the metrics
            y_pred = self.metrics['y_pred']
            y_true = self.metrics['y_true']

            # Create a confusion matrix
            cm = confusion_matrix(y_true, y_pred) #confusion matrix

            # Plot the confusion matrix as a heatmap
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot= True, fmt='d', cmap= 'Blues', cbar= False)
            plt.xlabel('Predicted labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            plt.show()

            # Print other classification metrics
            print('Accuracy:', self.metrics['Accuracy'])
            print("Precision:", self.metrics['Precision'])
            print("Recall:", self.metrics['Recall'])
            print("F1 Score:", self.metrics['F1'])

        elif model_type == 'Regressor':

            # Get the predicted and true labels from the metrics
            y_pred = self.metrics['y_pred']
            y_true = self.metrics['y_true']

            # Create a scatter plot of true labels vs. predicted labels
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=y_true, y=y_pred)
            plt.xlabel("True Labels")
            plt.ylabel("Predicted Labels")
            plt.title("True vs. Predicted Labels")
            plt.show()

            # Create a scatter plot of residuals
            residuals = y_true - y_pred
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=y_pred, y=residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel("Predicted Labels")
            plt.ylabel("Residuals")
            plt.title("Residual Plot")
            plt.show()

        
    def save_model(self, folder):

        '''
        Method to save model, hyperparams and metrics in a folder locally.

        '''

        os.makedirs(folder, exist_ok= True)
        dump(self.model, os.path.join(folder, 'model.joblib'))
        
        with open(os.path.join(folder, 'hyperparameters.json'), 'w') as hyperparameters_file:
            json.dump(self.hyperparameters, hyperparameters_file)

        with open(os.path.join(folder, 'metrics.json'), 'w') as metrics_file:
            json.dump(self.metrics, metrics_file)
            
        

class ClassificationModel(BaseModel):

    ''' Inherits from the BaseModel class and represents a specific classification model. It overrides the necessary
     methods such as fit, evaluate, and load hyperparameters, with specific logic for classification models.'''

    def __init__(self, model_type, test_size=0.3):
        super().__init__(model_type, test_size)
        self.hyperparameters = None
        self.metrics = None
        self.best_estimator = None

    def load_hyperparameters(self, hyperparameters):

        super().load_hyperparameters(hyperparameters= hyperparameters)
    
    def fit(self, train_set):

        super().fit(train_set= train_set)

    def evaluate(self, test_set):

        f1 = super().evaluate(model_type= 'Classifier', test_set= test_set)
        return f1

      
    def tune_classification_model_hyperparameters(self, train_set, val_set, grid):

        '''
        Hyperparameter tunning using sklearn GridSearchCV. Method creates model instance according 
        to model type and performs grid search with the corresponding grid. 
        The best estimator found is then used on validation data, from which metrics are extracted and stored as best_metrics.

        Returns a tuple of best hyperparameters, the best model, the best metrics and cross validation results.

        It also stores the best parameters and best estimator as the 'self' of the instance.

        '''

        X_train, y_train = train_set
        X_val, y_val = val_set

        X_train = super().preprocess(X_train)
        X_val = super().preprocess(X_val)

        model = self._create_model_instance()
        grid_search = GridSearchCV(estimator= model, param_grid= grid, scoring= 'accuracy', cv= 5)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_
        best_score = grid_search.best_score_
        cv_results = grid_search.cv_results_

        # calculations on validation set
        y_val_pred = best_estimator.predict(X_val)

        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred, average= 'macro')
        recall = recall_score(y_val, y_val_pred, average= 'macro')
        f1 = f1_score(y_val, y_val_pred, average = 'macro')

        best_metrics = {
            'Accuracy' : accuracy,
            'Precision' : precision,
            'Recall' : recall,
            'F1' : f1
        }

        
        self.hyperparameters = best_params
        self.best_estimator = best_estimator

        return best_params, best_estimator, best_metrics, cv_results
    

    def save_model(self, folder):

        super().save_model(folder= folder)

class RegressionModel(BaseModel):

    ''' Inherits from the BaseModel class and represents a specific regression model. It overrides the necessary
     methods such as fit, evaluate, and load hyperparameters, with specific logic for regression models.'''
    
    def __init__(self, model_type, test_size=0.3):
        super().__init__(model_type, test_size)
        self.hyperparameters = None
        self.metrics = None
        self.best_estimator = None

    def load_hyperparameters(self, hyperparameters):
        super().load_hyperparameters(hyperparameters= hyperparameters)

    def fit(self, train_set):

        super().fit(train_set= train_set)

    def evaluate(self, test_set):

        r2 = super().evaluate(model_type= 'Regressor', test_set= test_set)
        return r2
    
    def first_model(self,train_set, test_set) -> dict:

        '''

        This method is not bound to the class or its instances, 
        it behaves like a regular function. It is a function that creates 
        a model from scratch, with the default hyperparameters, trains it 
        and delivers the results
        
        '''

        base_model = BaseModel(model_type=self.model_type)
        my_model = base_model._create_model_instance()
        my_model = self.fit(train_set= train_set)

        X_train, y_train = train_set
        X_test, y_test = test_set
        X_train = self.preprocess(X_train)
        X_test = self.preprocess(X_test)

        y_train_pred = my_model.predict(X_train)
        y_test_pred = my_model.predict(X_test)

        #calculating RMSE
        rmse_train = mean_squared_error(y_train, y_train_pred)
        rmse_test = mean_squared_error(y_test, y_test_pred)

        #calculating r2 score
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        metrics = {
            "RMSE (training set)": rmse_train,
            "RMSE (test set)": rmse_test,
            "R2 score (training set)": r2_train,
            "R2 score (test set)": r2_test
        }
        return metrics

    def custom_tune_regression_hyperparameters(self, train_set, val_set, grid)-> dict:
        
        '''
        
        This method performs a grid search on all the parameters of a given 
        model type and delivers the best results. The grid is built with itertools.product.
        It returns a dictionary of best model, hyperparams and r2 score

        '''
        X_train, y_train = train_set
        X_val, y_val = val_set

        X_train = self.preprocess(X_train)
        X_val = self.preprocess(X_val)


        combinations = itertools.product(*grid.values())
        best_score = 0.0
        best_hyperparameters = None
        best_model = None

        for combination in combinations:
            hyperparameters = dict(zip(grid.keys(), combination))

            model = self._create_model_instance()
            model.set_params(**hyperparameters)
            model.fit(X_train, y_train)

            
            y_val_pred = model.predict(X_val)
            r2_val = r2_score(y_val, y_val_pred)

            if r2_val > best_score:
                best_score = r2_val
                best_hyperparameters = hyperparameters
                best_model = model
        
        results = {
            'Best Model': best_model,
            'Best Hyperparameters' : best_hyperparameters,
            'Best r2 score' : best_score
        }

        return results
    
    def tune_regression_model_hyperparameters(self, train_set, grid):

        '''
        Hyperparameter tunning using sklearn GridSearchCV. Method creates model instance according 
        to model type and performs grid search with the corresponding grid. 
        Returns a tuple of best hyperparameters, the best model, the best R2 score and cross validation results.

        It also stores the best parameters and best estimator as the 'self' of the instance.

        '''

        X_train, y_train = train_set
        X_train = self.preprocess(X_train)
        model = self._create_model_instance()

        grid_search = GridSearchCV(estimator= model, param_grid= grid, scoring= 'r2', cv=5, refit= 'r2')
        grid_search.fit(X_train, y_train)

        best_params=grid_search.best_params_
        best_estimator=grid_search.best_estimator_
        best_score=grid_search.best_score_
        cv_results = grid_search.cv_results_

        self.hyperparameters = best_params
        self.best_estimator = best_estimator

        return best_params, best_estimator, best_score, cv_results
    
    def save_model(self, folder):

        super().save_model(folder= folder)



class ModelsEvaluator:

    '''
    This class is used to evaluate the performance of all models, and store the best-performing one. 
    It contains methods to evaluate models based on gridsearch and to find the best model amongst the ones built. 
    It also contains a class to find the best model stored locally, as the processing time for the method find_best_model() was really large.
    
    '''
    def __init__(self, model_type) -> None:

        self.best_models = {} # dictionary to store the best combination of hyperparams 
        self.model_type = model_type

        
        if model_type == 'Regression':
            
            self.models ={
            'SGDRegressor': parameter_grid_SGDRegressor,
            'DecisionTreeRegressor': parameter_grid_DecisionTree,
            'RandomForestRegressor': parameter_grid_RandomForest,
            'GradientBoostRegressor': parameter_grid_GradientBoost
            }

        elif model_type == 'Classification':
            self.models = {
            'LogisticRegression' : parameter_grid_LogisticRegressor,
            'DecisionTreeClassifier' : parameter_grid_DecisionTreeClassifier,
            'RandomForestClassifier' : parameter_grid_RandomForestClassifier,
            'GradientBoostClassifier' : parameter_grid_GradientBoostClassifier

            }

        
    
    def evaluate_all_models(self, train_set, val_set, test_set):

        '''
        
        Performs a grid search over all the possible models and stores the best estimator, 
        combination of hyperparameters and scores in self.best_models[model_name]
        
        '''
        
        for model_name, parameter_grid in self.models.items():
                
            if self.model_type == 'Classification':
                classification_model = ClassificationModel(model_type= model_name)
                best_params, best_estimator, best_score, _ = classification_model.tune_classification_model_hyperparameters(
                    train_set= train_set, val_set= val_set, grid= parameter_grid)
                
                classification_model.load_hyperparameters(best_params)
                classification_model.best_estimator = best_estimator
                classification_model.fit(train_set= train_set)
                f1 = classification_model.evaluate(test_set= test_set)

                self.best_models[model_name] = { 
                    'metrics' : {'F1' : f1},
                    'best_estimator' : best_estimator,
                    'best_params' : best_params,
                    'best_score' : best_score
                }

                folder_path = os.path.join('models', 'classification', model_name)
                classification_model.save_model(folder= folder_path)

            elif self.model_type == 'Regression':
                regression_model = RegressionModel(model_type= model_name)
                best_params, best_estimator, best_score, _ = regression_model.tune_regression_model_hyperparameters(
                    train_set= train_set, grid= parameter_grid
                )

                regression_model.load_hyperparameters(best_params)
                regression_model.best_estimator = best_estimator
                regression_model.fit(train_set= train_set)
                r2_score = regression_model.evaluate(test_set=test_set)

                self.best_models[model_name] = {
                    'metrics' : {'R2' : r2_score},
                    'best_estimator' : best_estimator,
                    'best_params' : best_params,
                    'best_score' : best_score
                }

                folder_path = os.path.join('models', 'regression', model_name)
                regression_model.save_model(folder=folder_path)

    def find_best_model(self):

        ''' 

        Finds best model after running evaluate_all_models. Iterates over the dictionary 
        self.best_models, which contains the best hyperparameters combination for each model type
        
        '''

        best_metrics = 0.0
        best_model_name = None

        for model_name, model_data in self.best_models.items():

            if self.model_type == 'Classification':
                f1 = model_data['metrics']['F1']
                if f1 >= best_metrics:
                    best_metrics = f1
                    best_model_name = model_name
            elif self.model_type == 'Regression':
                r2_score = model_data['metrics']['R2']
                if r2_score >= best_metrics:
                    best_metrics = r2_score
                    best_model_name = model_name
        
        best_model_data = self.best_models[best_model_name] 
        best_model = best_model_data['best_estimator']
        hyperparameters = best_model_data['best_params'] 

        return best_model, hyperparameters, best_metrics
    

    
    def find_best_model_local(self):

        ''' This method finds the best model stored in the local folders. 
        The previous method needs an instance of the class that has ran the
        'evaluate_all_models' method, which takes a really long time to run.'''

        if self.model_type == 'Classification':
            folder_path = 'models/classification'
        elif self.model_type == 'Regression':
            folder_path = 'models/regression'

        subfolder_names = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name)) and name != 'old']
        best_metrics = 0.0
        best_metrics_individual = 0.0
        

        for folder_name in subfolder_names:

            file_path = os.path.join(folder_path, folder_name, 'metrics.json')

            with open(file_path, 'r') as file:
                model_metrics = json.load(file)
                
                if self.model_type == 'Classification':
                    model_metrics_individual = model_metrics['F1']
                elif self.model_type == 'Regression':
                    if isinstance(model_metrics, dict):
                        model_metrics_individual = model_metrics['R2']
                    else:
                        model_metrics_individual = model_metrics

                if model_metrics_individual >= best_metrics_individual:
                    best_metrics = model_metrics
                    best_model = folder_name
                    best_metrics_individual = best_metrics_individual
                else: 
                    best_metrics = best_metrics
                
                
        
        model = load(os.path.join(folder_path, best_model, 'model.joblib'))

        with open(os.path.join(folder_path, best_model, 'hyperparameters.json'), 'r') as json_file:
            # Load the JSON data
            hyperparameters = json.load(json_file)
            
        return model, hyperparameters, best_metrics 
    
    



        

