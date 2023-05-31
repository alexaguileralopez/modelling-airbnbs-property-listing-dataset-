# modelling-airbnbs-property-listing-dataset-

This project consists in building a framework to systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team.


## Milestone 3: Data preparation

Before building the framework, the dataset has to be structured and clean. 
Inside the listing.csv file, there is a tabular dataset with the following columns:

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
The first step is to define a function to remove the rows with missing ratings. This is done by simply keeping the rows in the dataframe that contain values in the ratings columns:

    df = df[~df['Cleanliness_rating'].isna()]
    df = df[~df['Accuracy_rating'].isna()]
    df = df[~df['Location_rating'].isna()]
    df = df[~df['Check-in_rating'].isna()]
    df = df[~df['Value_rating'].isna()]

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
    df['Description'] = df['Description'].apply(lambda x: [item for item in x if item != 'About this space'])
    df['Description'] = df['Description'].apply(lambda x: [item for item in x if item != 'The space'])
    df['Description'] = df['Description'].apply(lambda x: [item for item in x if item != 'License number'])
    df['Description'] = df['Description'].apply(lambda x: [item for item in x if item != 'Other things to note'])
    df['Description'] = df['Description'].apply(lambda x: [item for item in x if item != 'Guest acess'])

    df['Description'] = df['Description'].apply(lambda x: ' '.join(x))


The 'beds', 'guests', 'bathrooms', 'bedrooms' columns have empty values for some rows, so a function is defined to set a default value to 1 in those.

    df['guests'] = df['guests'].apply(lambda x: 1 if x== np.nan else x)
    df['beds'] = df['beds'].apply(lambda x: 1 if x== np.nan else x)
    df['bathrooms'] = df['bathrooms'].apply(lambda x: 1 if x== np.nan else x)
    df['bedrooms'] = df['bedrooms'].apply(lambda x: 1 if x== np.nan else x)

All these functions are called from a function called clean_tabular_data, that returns the processed data:

    def clean_tabular_data(raw_dataframe):

        #df = pd.read_csv(raw_dataframe)

        df = remove_rows_with_missing_ratings(raw_dataframe)
        df = combine_description_strings(df)
        df = set_default_feature_values(df)

        return df

To save that new clean dataframe, that last function is called and saved inside an if __name__ = '__main__'

    if __name__ == '__main__':
     
     df = pd.read_csv('listing.csv')
     df= clean_tabular_data(df)
     df.to_csv('clean_tabular_data.csv', index=False, mode = 'w' )


In order to get the data in the right format for testing, a function is created to divide the features and labels of the data in a tuple. The label is stored as a new variable and the dataframe drops the label:

    def load_airbnb(df= pd.DataFrame , label=str):
     
     labels = df[label]
     features = df.drop(label, axis=1)

     tuple_data = (features, labels)

     return tuple_data


