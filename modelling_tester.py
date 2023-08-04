#%%
'''
Importing the Classes and parameter grids from other files

'''
from modelling import BaseModel, RegressionModel, ClassificationModel, ModelsEvaluator
from tabular_data import clean_tabular_data, load_airbnb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from classification_hyper import parameter_grid_DecisionTreeClassifier, parameter_grid_GradientBoostClassifier, parameter_grid_RandomForestClassifier, parameter_grid_LogisticRegressor
from regression_hyper import parameter_grid_DecisionTree, parameter_grid_GradientBoost, parameter_grid_RandomForest, parameter_grid_SGDRegressor


#%% 

''' Defining the data that is going to be used for the regression model scenario'''
import pandas as pd

from modelling import BaseModel, RegressionModel, ClassificationModel
from tabular_data import clean_tabular_data, load_airbnb
from sklearn.model_selection import train_test_split

df = pd.read_csv('airbnb-property-listings/listing.csv')
df_1= clean_tabular_data(raw_dataframe= df)
dataset =load_airbnb(df= df_1, label= 'Category')

X, y = dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0) # validation slightly larger than test

train_set = (X_train, y_train)
test_set = (X_test, y_test)
val_set = (X_val, y_val)

best_model = BaseModel(model_type= 'GradientBoostClassifier')
# Step 2: Load the hyperparameters into the model
best_hyperparams = {'criterion': 'friedman_mse', 'learning_rate': 0.1, 'max_depth': 3, 'max_features': 'log2', 'min_samples_leaf': 3, 'min_samples_split': 10, 'n_estimators': 500}

best_model.load_hyperparameters(best_hyperparams)

best_model.fit(train_set)

f1 = best_model.evaluate(test_set, model_type= 'Classifier')

plot = best_model.plot_results(model_type= 'Classifier')


# %%
from neural_network import regression_NN, AirbnbNightlyPriceRegressionDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(0)
dataset = AirbnbNightlyPriceRegressionDataset()
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)


train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

model_config = {'hidden_layer_width': 32, 'depth': 3, 'dropout_rate': 0.5, 'lr': 0.01, 'optimiser': 'Adam'}
model = regression_NN(hidden_layer_width= model_config['hidden_layer_width'], depth= model_config['depth'], dropout_rate= model_config['dropout_rate'])
model.evaluate_model(testing_loader= test_loader, training_loader= train_loader, validation_loader= val_loader)
model.plot_results(test_loader)
# %%
import pandas as pd

from modelling import BaseModel, RegressionModel, ClassificationModel, ModelsEvaluator
from tabular_data import clean_tabular_data, load_airbnb
from sklearn.model_selection import train_test_split

df = pd.read_csv('airbnb-property-listings/listing.csv')
df_1= clean_tabular_data(raw_dataframe= df)
dataset =load_airbnb(df= df_1, label= 'bedrooms', Category= True)

X, y = dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0) # validation slightly larger than test

train_set = (X_train, y_train)
test_set = (X_test, y_test)
val_set = (X_val, y_val)


evaluator = ModelsEvaluator(model_type= 'Regression')
evaluator.evaluate_all_models(train_set= train_set, test_set= test_set, val_set= val_set)
best_model, hyperparameters, best_metrics = evaluator.find_best_model()

print('Best model:', best_model)
print('Best hyperparams:', hyperparameters)
print('Best metrics:', best_metrics)
# %%

''' Creating the best regression model and plotting its results'''

import pandas as pd

from modelling import BaseModel, RegressionModel, ClassificationModel
from tabular_data import clean_tabular_data, load_airbnb
from sklearn.model_selection import train_test_split

df = pd.read_csv('airbnb-property-listings/listing.csv')
df_1= clean_tabular_data(raw_dataframe= df)
dataset =load_airbnb(df= df_1, label= 'bedrooms', Category= True)

X, y = dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0) # validation slightly larger than test

train_set = (X_train, y_train)
test_set = (X_test, y_test)
val_set = (X_val, y_val)

model = BaseModel(model_type= 'RandomForestRegressor')
best_hyperparams = {'criterion': 'squared_error', "max_depth": 5, "min_samples_leaf": 1, "min_samples_split": 2, "n_estimators": 500}


model.load_hyperparameters(best_hyperparams)

model.fit(train_set)

r2 = model.evaluate(test_set, model_type= 'Regressor')



plot = model.plot_results(model_type= 'Regressor')
# %%
from neural_network import regression_NN, AirbnbNightlyPriceRegressionDataset, find_best_nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(0)
dataset = AirbnbNightlyPriceRegressionDataset()
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)


train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

best_model, best_metrics, best_hyperparameters = find_best_nn(val_loader= val_loader, train_loader= train_loader, test_loader= test_loader, 
                                                              folder_path='models/neural_networks/regression', n_configs= 5) 


best_model.train(train_data_loader= train_loader, val_data_loader= val_loader, epochs= 10, config= best_hyperparameters)
best_model.evaluate_model(training_loader= train_loader, validation_loader= val_loader, testing_loader= test_loader)
best_model.plot_results(data_loader= test_loader)
# %%
