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
 
The 'get_metrics' method calculates the average RMSE loss, R-squared, and interference latency across the entire dataset. It iterates through the data loader, computing predictions, RMSE loss, R-squared for each batch. The average metrics are then computed using the cumulative values. 

The save_model_metrics method saves the model, hyperparameters, and performance metrics in separate files within a folder. It takes care of handling existing folder names to avoid overwriting previous models.

Outside of the classes, there are additional functions:

The generate_nn_configs function generates a specified number of neural network configurations by randomly selecting hyperparameters like optimizer, learning rate, hidden layer width, depth, and dropout rate.

The find_best_nn function is used to sequentially train multiple models with different configurations and identify the best model based on validation R-squared. It saves the best model and its performance metrics in a designated folder called "best_model" to preserve the best-performing model.

## RESULTS






