from modelling import BaseModel, ClassificationModel, RegressionModel, ModelsEvaluator
from classification_hyper import parameter_grid_DecisionTreeClassifier, parameter_grid_GradientBoostClassifier, parameter_grid_RandomForestClassifier, parameter_grid_LogisticRegressor
from regression_hyper import parameter_grid_DecisionTree, parameter_grid_GradientBoost, parameter_grid_RandomForest, parameter_grid_SGDRegressor
from tabular_data import clean_tabular_data, load_airbnb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


np.random.seed(0)
    
df = pd.read_csv('airbnb-property-listings/listing.csv')
df_1= clean_tabular_data(raw_dataframe= df)
dataset = load_airbnb(df= df_1, label= 'Category')

model = ClassificationModel(model_type= 'DecisionTreeClassifier')

train_set, test_set, val_set = model.train_test_split(dataset= dataset, test_size= 0.2)

model.fit(train_set= train_set)
print(model.evaluate(test_set= test_set))


best_params, best_estimator, best_metrics, cv_results = model.tune_classification_model_hyperparameters(train_set= train_set, val_set= val_set, grid= parameter_grid_DecisionTreeClassifier)
print(best_metrics)
print(model.best_estimator)
print(model.hyperparameters)


evaluator = ModelsEvaluator(model_type= 'Classification')
best_model = evaluator.find_best_model_local()

print(best_model)
