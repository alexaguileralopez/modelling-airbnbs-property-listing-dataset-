
[ 
    {
        "model_type" : "RandomForest",
        "n_estimators" : [100, 200, 500],
        "max_depth" : [ 3, 5, 10],
        "min_samples_split" : [2, 5 , 10],
        "min_samples_leaf" : [1,3,5],
        "max_features" : ["sqrt", "log2"],
        "criterion" : ["gini"]

    },
    {
        "model_type" : "DecisionTree",
        "max_depth" : [3,5,10], 
        "min_samples_split" : [2, 5, 10 ],
        "min_samples_leaf": [1, 3, 5],
        "max_features" : ["sqrt", "log2"],
        "criterion" : ["gini"] 

    },
    {
        "model_type" : "GradientBoost",
        "n_estimators" : [100, 200, 500],
        "learning_rate" : [0.1, 0.01],
        "max_depth" : [3,5,10], 
        "min_samples_split" : [2, 5, 10 ],
        "min_samples_leaf": [1, 3, 5],
        "max_features" : ["sqrt", "log2"],
        "criterion" : ["friedman_mse"]

    },
    {
        "model_type" : "LogisticRegressor",
        "penalty" : ["l1", "l2"],
        "C" : [0.1, 1.0, 10.0],
        "solver" : ["liblinear", "saga"]


    }

]
