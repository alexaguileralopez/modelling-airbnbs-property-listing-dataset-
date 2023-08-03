from modelling import BaseModel, ClassificationModel, RegressionModel, ModelsEvaluator
from classification_hyper import parameter_grid_DecisionTreeClassifier, parameter_grid_GradientBoostClassifier, parameter_grid_RandomForestClassifier, parameter_grid_LogisticRegressor
from regression_hyper import parameter_grid_DecisionTree, parameter_grid_GradientBoost, parameter_grid_RandomForest, parameter_grid_SGDRegressor
from tabular_data import clean_tabular_data, load_airbnb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


np.random.seed(0)

evaluator = ModelsEvaluator(model_type= 'Classification')
best_model = evaluator.find_best_model_local()

print('The best classification model found locally is:', best_model)



new_evaluator = ModelsEvaluator(model_type= 'Regression')
best_regressor = new_evaluator.find_best_model_local()

print('The best regression model found locally is:', best_regressor)