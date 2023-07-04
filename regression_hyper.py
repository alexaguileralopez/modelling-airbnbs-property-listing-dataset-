

parameter_grid_RandomForest = {

        "n_estimators" : [100, 200, 500],
        "max_depth" : [ 3, 5, 10],
        "min_samples_split" : [2, 5 , 10],
        "min_samples_leaf" : [1,3,5],
        "criterion" : ["squared_error", "absolute_error", "friedman_mse", "poisson"]
      

    }
parameter_grid_DecisionTree = {

        "splitter" : ["best", "random"], 
        "max_depth" : [3,5,10], 
        "min_samples_split" : [2, 5, 10 ],
        "min_samples_leaf": [1, 3, 5],
        "criterion" : ["squared_error", "friedman_mse", "absolute_error", "poisson"] 

    }
parameter_grid_GradientBoost = {

        "n_estimators" : [100, 200, 500],
        "learning_rate" : [0.1, 0.01],
        "max_depth" : [3,5,10], 
        "min_samples_split" : [2, 5, 10 ],
        "min_samples_leaf": [1, 3, 5],
        "max_features" : ["sqrt", "log2"],
        "criterion" : ["friedman_mse"]

    }
parameter_grid_SGDRegressor = {

        "penalty" : ["l1", "l2"],
        "loss" : ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
        "alpha" :  [0.1, 0.01, 0.001, 0.0001],
        "learning_rate" : ["constant", "optimal", "invscaling", "adaptive"],
        "max_iter" : [1000, 10000, 50000]


    }

