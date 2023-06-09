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

## CREATING A REGRESSION MODEL

For this task, it is useful to use the sklearn library, as it contains several models and functions that are used for machine learning modelling. A new file [modelling.py] is created.

The data is loaded from the previous script, and the label selected is 'Price_Night'. Therefore, the aim is to train and test different models into guessing the Price_Night according to the relations of the different features with the training labels. 

    dataset =tabular_data.load_airbnb(df= df_1, label= 'Price_Night')

    X, y = dataset

For testing purposes, the data is split 70/30 into training and testing:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3) # splitting dataset

The first model used is a Stochastic Gradient Descent Regressor, which supports different loss functions and penalties to fit linear regression models. This model is well suited for regression problems with a large number of training samples. 

    myModel = SGDRegressor().fit(X_train, y_train)

Predictions are made on the training and testing data:

    y_train_pred = myModel.predict(X_train)
    y_test_pred = myModel.predict(X_test)

In the first case, the model does not fill well the training data. The RMSE value for y_train_pred and y_train is 2015344407.52, which is super large. The R^2, similarly, is also very large and negative, indicating poor model performance.

- RMSE_train = 2015344407.52
- R^2_train = -209238121347827.88

[y_train_predictions](code_snippets/y_train_SGDR.png)

As expected from the last case, the model also does not fit well the testing data.

- RMSE_test = 2077415679.79
- R^2_test = -424609764864478.56

Those two values indicate very poor model performance.

[y_test_predicitons](code_snippets/y_test_SGDR.png)



