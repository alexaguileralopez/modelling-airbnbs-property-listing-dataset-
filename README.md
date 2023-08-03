# MODELLING AIRBNB'S PROPERTY LISTING DATASET

This project consists in building a framework to systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team.

The information available are images of the properties and tabular data, which contains information such as ID, category, price, number of beds or different ratings for a large quantity of properties.
Here, the challenges to tackle are the way to find relationships between prices, number of bedrooms, ratings or property types to all of the other information stored in the dataset. And hence, to predict those values on unseen data, based on the relations extracted from the training data.

The project contains different files such as [tabular_data.py], [modelling.py], and [neural_network.py], which contain the main utilities of the project. In addition, there are other files like [classification_hyper.py] or [regression_hyper.py] that contain parameters that will be used in different models. The file [main.py] contains the logic flow of the program, including data processing, model training, evaluation and other computations.

## DATA PREPARATION

The first task to approach is preparing the data so it is suitable for modelling. Before building the framework, the dataset needs to be structured and clean. 
The file [tabular_data.py] contains a process of data cleaning that is wrapped up in a function called [clean_tabular_data(raw_dataframe)], that returns the processed data. 

This funciton contains nested functions that perform different data cleaning tasks: 
- Remove rows with missing ratings
- Combine property description strings (necessary as the raw file is a csv file)
- Set default feature values (in cases where there is no need to delete rows and default values can be added)

To use this data for modelling, it needs to be separated into features and labels. The colums containing text data are filtered out, because those will not be useful as features. However, different labels (price, property type, number of bedrooms) can be chosen. The function [load_airbnb(df,label)] is created to separate the clean dataset into a tuple containing a labels set and features.


## CREATING A BASE MODEL

The file [modelling.py] contains different classes containing a simple Base Model, from which a Regression and Classification versions are born. These models make use of the files [classification_hyper.py] and [regression_hyper.py], which contain different model parameters for different estimators such as trees or ensembles. 

A base model is built to set the base of the rest of the models (classification and regression) that will be used for this task. This models contains multiple methods that are common between the application of regression and classification models, such as fitting, evaluation, or creating a model instance. This model acts as a parent class.

The main method in this class is 
   def _create_model_instance(self) -> type:

    if self.model_type == "RandomForestRegressor":
        model_instance = RandomForestRegressor()
        model_instance.estimators_ = []
    elif self.model_type == "RandomForestClassifier":
        model_instance = RandomForestClassifier()
    elif self.model_type == "DecisionTreeRegressor":
        model_instance = DecisionTreeRegressor()
    elif self.model_type == "DecisionTreeClassifier":
        model_instance = DecisionTreeClassifier()
    ...

which returns the desired model instance type according to the string of the model that is specified:

    model = BaseModel(model_type= 'DecisionTreeClassifier')

The two child classes built are classification and regression models, those inherit methods such as __init__, load_hyperparameters, fit, evaluate. Some of them require different functioning, but the slight differences they posses are simple parameters that can be adjusted when building the model. For example, in the case of the evaluation method, it requires the model type:

    def evaluate(self, test_set):

        f1 = super().evaluate(model_type= 'Classifier', test_set= test_set)
        return f1

In this case, what is interesting to know in a classifier, when performing its evaluation, is the f1 score, so the base model method has to be adjusted to decide if it calculates the metrics for a regression model (r2, MSE) or for a classification model (accuracy, precision, recall, f1).

This is an efficient way of being able to build more model types without altering the base model. 

In addition, there is also a save_model method which saves the model, hyperparameters and metrics locally.


## CREATING A CLASSIFICATION MODEL

The ClassificationModel inherits from the BaseModel class. The differences with it is that the tunning of hyperparameters is unique to the classification model, as the scoring is different. 

    model = ClassificationModel(model_type= 'DecisionTreeClassifier')

The method 

    tune_classification_model_hyperparameters(self, train_set, val_set, grid)

makes use of the training set to perform a grid search (GridSearchCV) and finds the best hyperparameters for the classification model. What is then stored as the instance of the model itself is the best model chosen, with the best hyperparameters found. 

The file from which the hyperparameters are extracted is classification_hyper.py, which contains different dictionaries with lists of hyperparameters to try on the different models: Decision Tree Classifier, Random Forest Classifier, Gradient Boost Classifier and Logistic Regression.

## CREATING A REGRESSION MODEL

As the last class, the RegressionModel also inherits from the BaseModel class. In this case, there are some extra methods such as 

    def first_model(self,train_set, test_set) -> dict:

Which is a method that could be outside of the class, but it is used to create a model with default hyperparameters, train it, and deliver results.
On the other hand, there is a method to perform a 'manual' grid search of all hyperparameters in the hyperparameter grid.

    def custom_tune_regression_hyperparameters(self, train_set, val_set, grid)-> dict:

On the other side, it contains as well a method that makes use of grid search (scoring= r2) to define which are the most suitable hyperparameters for the model chosen.


## MODEL EVALUATION

To evaluate the performance of all models, a class called ModelsEvaluator is created. This class contains methods to evaluate all models based on gridsearchCV and to find the best model amongst the ones built (either regression or classification).

The method to evaluate all models performs a grid search over all possible models and stores the best estimator, combinaiton of hyperparameters and scores in an attribute called best_models of the class. 

To find the best out of the best models stored, a method called find_best_model is created, and this should be run after performing the general evaluation. This method looks into self.best_models.items() and compares the model data. Then, it choses the one with the best metrics and returns the resulting best model. 

As the evaluation of all models can be a long running process, a method to find the best model stored locally is also created. It does the same comparison but, instead of running the evaluation of all models, it can be run directly and will compare the metrics in the models that have been saved in the models directory. 

## RESULTS

For the regression problem, the model found as best performing is the SGDRegressor. The label chosen is 'Price_Night'.

    {'alpha': 0.0001, 'learning_rate': 'constant', 'loss': 'epsilon_insensitive', 'max_iter': 1000, 'penalty': 'l2'}

- R2 = 0.42
- MSE = 21262.03

[code_snippets/best_regression_scatter.png]

The scatter plot shows general positive correlation between the true labels and predicted values. As R2 is positive (0.42), it indicates that the model is capturing some of the variance in the data, and the predictions have a moderate correlation with the true labels.
However, the points do not form a tight cluster around the ideal regression line (45º), indicating that the model is not capturing all the variance in the data.

[code_snippets/best_regression_residual.png]

The residual plot shows a random scatter of points around the horizontal line y=0. That suggests that the model's predictions have some level of bias, but it is capturing some of the variance in the data.

Overall, the model's performance is moderate, with an R2 of 0.42 indicating that there is still room for improvement.

For the classification problem, the model found as best performing is the Gradient Boosting Classifier.

    {'criterion': 'friedman_mse', 'learning_rate': 0.1, 'max_depth': 3, 'max_features': 'log2', 'min_samples_leaf': 3, 'min_samples_split': 10, 'n_estimators': 500}, 

- Accuracy: 0.3614457831325301
- Precision: 0.3591074121956475
- Recall: 0.3554801894918174
- F1 Score: 0.34657117582723634

[code_snippets/best_classification_cm.png]

The confusion matrix shows that for the 5 labels in 'Category', the model found a strong relationship for labels 2 and 4. In the case of label 3, the amount of false positives. In the case of labels 0 and 1 the correlation was detected, but it was not strong enough to underline it severely. 

The results suggest that the chosen best model is not performing well on the given classification task. The low accuracy, precision, recall and F1 score indicate that the model is struggling to make accurate predictions for the positive class, and overall performance is not satisfactory. 

To improve the model's performance, exploring different hyperparameter settings can be explored, also feature engineering can be useful. 

## DEEP LEARNING APPROACH

Making use of the PyTorch library, the same challenges can be tackled using neural networks. The file [neural_network.py] contains the framework to train a custom built neural network to predict the Price Night or Number of Rooms on unseen data. 

In this file, there are two classes:
- A Dataset class, which is used to represent the data that will be used to train, validate and test. It acts as an input source for PyTorch's DataLoader. The name of the class is AirbnbNightlyPriceRegressionDataset(Dataset). 
- A Torch Neural Network Module class which is responsible for defining the structure and forward pass of the neural network. The name of the class is regression_NN(torch.nn.Module). 

The Dataset class is initialised so that it takes the raw data and uses the tabular data functions to clean it and load it as a pandas dataframe. Here, is where the label is set. The other methods that this class contains are ___getitem__ and __len__, which return a tuple of features and labels and the length of the data respectively. 

The model is built in the regression_NN class, which is responsible for defining the structure and forward pass of the neural network. The parameters that are initialised are the hidden layer width, depth and dropout rate of the model as hyperparameters, and the input size as 11 (according to the dataset size). The variable layers is also initialised as an empty list which will be appended later. 

The layers of the neural network are created with a for loop, determined by the depth parameter. 

    for i in range(depth):
        layers.append(torch.nn.Linear(input_size, hidden_layer_width))
        layers.append(torch.nn.BatchNorm1d(hidden_layer_width)) # Batch Normalisation
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout_rate)) # Dropout
        input_size = hidden_layer_width
    
    layers.append(torch.nn.Linear(input_size, 1))
    self.layers = torch.nn.Sequential(*layers)

This code creates a neural network model with 'depth' number of hidden layers, where each layer is a linear layer followed by Batch Normalisation, ReLU activation, and Dropout. The input size of each layer is equal to the number of neurons in the previous hidden layer ('hidden_layer_width'). The model's output is a single value, suitable for regression problems. Batch Normalisation and Dropout are later additions that have been made to the code, that will be discussed in the results section. 

Other methods have been included such as get_hyperparameters, and calculations of rmse loss and r2 (as in the regression model class outlined previously). The most relevant method is 'train'. This method takes in a training data loader, a validation data loader, number of epochs and a configuration dictionary. The 'train' method performs training using the specified data loaders and training configuration. It employs an optimiser chosen based on the configuration(e.g., SGD or Adam) and calculates the MSE loss during the training process. The training progress and validation loss are monitored and logged using 'SummaryWriter'.

The TensorBoard tool provides visualisation tools to observe the process of training and validation of the data. As seen on the graph below, the training loss becomes smaller as more iterations are performed. Meaning a better result could be expected if the dataset was more extense:

[code_snippets/training_scalar.png]

What can also be observed are the loss and rmse metrics for the validation data, which also descend steadily with increasing iterations of the loop. Ideally, these would drop to values between 1 and 0:

[code_snippets/val_loss_scalar.png]
[code_snippets/val_rmse_scalar.08.png]
 
The 'get_metrics' method calculates the average RMSE loss, R-squared, and interference latency across the entire dataset. It iterates through the data loader, computing predictions, RMSE loss, R-squared for each batch. The average metrics are then computed using the cumulative values. 

The save_model_metrics method saves the model, hyperparameters, and performance metrics in separate files within a folder. It takes care of handling existing folder names to avoid overwriting previous models.

Outside of the classes, there are additional functions:

The generate_nn_configs function generates a specified number of neural network configurations by randomly selecting hyperparameters like optimizer, learning rate, hidden layer width, depth, and dropout rate.

The find_best_nn function is used to sequentially train multiple models with different configurations and identify the best model based on validation R-squared. It saves the best model and its performance metrics in a designated folder called "best_model" to preserve the best-performing model.

## RESULTS

The best model found is 

- The hyperparameters.json contains:

    {'hidden_layer_width': 32, 'depth': 3, 'dropout_rate': 0.5, 'lr': 0.01, 'optimiser': 'Adam'}

- The metrics.json contains:

     {'training_duration': 0.3914968967437744, 'interference_latency': 1.157048236892884e-05, 'training_RMSE_loss': 3.304215749104818, 'training_R2': 0.0015443707141050062, 'validation_RMSE_loss': 4.263668777351093, 'validation_R2': 0.0037325275571722734, 'testing_RMSE_loss': 4.204518329666321, 'testing_R2': 0.0030592963638075865}

[code_snippets/nn_regression_scatter_test.png]


[code_snippets/nn_regression_residual_test.png]

The training duration and interference latency indicate that the model is training and performing quickly.

The training RMSE loss of 3.30 suggests that the model has some errors in its predictions on the training data. The training R2 score of 0.00154 indicates that the model performs worse than a simple horizontal line, suggesting that it is not effectively capturing the variance in the training data.

On the validation set, the RMSE loss (4.26) and R2 score (0.00373) both indicate worse performance compared to the training set. These results suggest that the model is not generalizing well to unseen data and may be overfitting to the training set.

The testing results also show a similar pattern with a RMSE loss of 4.20 and an R2 score of 0.00306, indicating poor generalization to new data.

Overall, the model's performance on both the training, validation, and testing sets is subpar, as indicated by the low R2 scores and relatively high RMSE values. This suggests that the model is not effectively capturing the underlying patterns in the data and may require further hyperparameter tuning, architecture adjustments, or data preprocessing to improve its performance. Additionally, investigating the quality of the data and potential feature engineering could also be beneficial to enhance the model's predictive capabilities.

## REUSING THE FRAMEWORK

Now, the label chosen is the integer number of bedrooms. The previous label used in the last section 'Category', is now used as another feature, even though it is not a numerical value. 





