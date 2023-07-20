# MODELLING AIRBNB'S PROPERTY LISTING DATASET

This project consists in building a framework to systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team.

The information available are images of the properties and tabular data, which contains information such as ID, category, price, number of beds or different ratings. 

## DATA PREPARATION

The first task to approach is preparing the data so it is suitable for modelling. 

Before building the framework, the dataset has to be structured and clean. 
Inside the files, there is a tabular dataset with the following columns:

- ID: Unique identifier for the listing
- Category: The category of the listing
- Title: The title of the listing
- Description: The description of the listing
- Amenities: The available amenities of the listing
- Location: The location of the listing
- guests: The number of guests that can be accommodated in the listing
- beds: The number of available beds in the listing
- bathrooms: The number of bathrooms in the listing
- Price_Night: The price per night of the listing
- Cleanliness_rate: The cleanliness rating of the listing
- Accuracy_rate: How accurate the description of the listing is, as reported by previous guests
- Location_rate: The rating of the location of the listing
- Check-in_rate: The rating of check-in process given by the host
- Value_rate: The rating of value given by the host
- amenities_count: The number of amenities in the listing
- url: The URL of the listing
- bedrooms: The number of bedrooms in the listing

The file [tabular_data.py] is created to manage that tabular data, and perform its cleaning process.

The function [remove_rows_with_missing_ratings(df)] performs this filtering in the ratings

    df = df[~df['Cleanliness_rating'].isna()]


The description column contains lists of strings that pandas does not recognise as such, instead it recognises them as strings. All of the lists begin with the same pattern, so a nested function is created with an if/else statement. If this condition is satisfied, the function ast.literal_eval is used to transform those strings into lists:

    if isinstance(row['Description'], str) and row['Description'].strip().startswith('['):
            row['Description']= ast.literal_eval(row['Description'])

            return row

However, there is one row where the elements from the description column are shifted one column to the right, so the element in description has to be deleted and all of the rest should be shifted from the 'Amenities' column to the left by one position:

    elif not isinstance(row['Description'], str) or not row['Description'].strip().startswith('[') and row['Amenities'].strip().startswith('['):
    row['Description'] = row['Amenities']
    row['Amenities'] = row['Location']
    row['Location'] = row['guests']
    row['guests'] = row['beds']
    row['beds'] = row['bathrooms']
    row['bathrooms'] = row['Price_Night']
    row['Price_Night'] = row['Cleanliness_rating']
    row['Cleanliness_rating'] = row['Accuracy_rating']
    row['Accuracy_rating'] = row['Communication_rating']
    row['Communication_rating'] = row['Location_rating']
    row['Location_rating'] = row['Check-in_rating']
    row['Check-in_rating'] = row['Value_rating']
    row['Value_rating'] = row['amenities_count']
    row['amenities_count'] = row['url']
    row['url'] = row['bedrooms']
    row['bedrooms'] = row[19]
    row[19] = np.nan

    row['Description']= ast.literal_eval(row['Description'])

Lastly, repeated string pieces are removed and the list is joined as a string, getting the description as a full text:

    df['Description'] = df['Description'].apply(lambda x: [item for item in x if item != ''])
...

    df['Description'] = df['Description'].apply(lambda x: ' '.join(x))

Some columns such as beds or guests have empty values that cannot be set to 0, therefore they are set to 1 with the function [set_default_feature_values(df)]

     df['guests'] = df['guests'].apply(lambda x: 1 if pd.isnull(x) else x)

All these functions are called from a function called [clean_tabular_data(raw_dataframe)]that returns the processed data.

In order to use this data for modelling, the data needs to be separated into features and labels. 

For now, the columns including text data are filtered out, and just the numeric tabular data is used, which is transformed to numpy arrays format to be suitable for modelling. 
[load_airbnb(df= pd.DataFrame , label=str)] is created to separate the dataset into two sets containing a label and features. 

## CREATING A BASE MODEL

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




