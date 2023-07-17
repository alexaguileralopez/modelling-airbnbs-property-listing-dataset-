

parameter_grid_RandomForestClassifier = {
       
        "n_estimators" : [100, 200, 500],
        "max_depth" : [ 3, 5, 10],
        "min_samples_split" : [2, 5 , 10],
        "min_samples_leaf" : [1,3,5],
        "max_features" : ["sqrt", "log2"],
        "criterion" : ["gini"]

    }

parameter_grid_DecisionTreeClassifier = {
       
        "max_depth" : [3,5,10], 
        "min_samples_split" : [2, 5, 10 ],
        "min_samples_leaf": [1, 3, 5],
        "max_features" : ["sqrt", "log2"],
        "criterion" : ["gini"] 

    }

parameter_grid_GradientBoostClassifier = {
       
        "n_estimators" : [100, 200, 500],
        "learning_rate" : [0.1, 0.01],
        "max_depth" : [3,5,10], 
        "min_samples_split" : [2, 5, 10 ],
        "min_samples_leaf": [1, 3, 5],
        "max_features" : ["sqrt", "log2"],
        "criterion" : ["friedman_mse"]

    }
parameter_grid_LogisticRegressor = {
       
        "penalty" : ["l1", "l2"],
        "C" : [0.1, 1.0, 10.0],
        "solver" : ["liblinear", "saga"]


    }

