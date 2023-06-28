# file to manipulate tabular data
import pandas as pd
import ast
import numpy as np


def remove_rows_with_missing_ratings(df):

    ''' removing rows with missing values in the rating columns'''

    df = df[~df['Cleanliness_rating'].isna()]
    df = df[~df['Accuracy_rating'].isna()]
    df = df[~df['Communication_rating'].isna()]
    df = df[~df['Location_rating'].isna()]
    df = df[~df['Check-in_rating'].isna()]
    df = df[~df['Value_rating'].isna()]

    df = df[~df['Price_Night'].isna()] # if price does not exist, not useful


    return df

def combine_description_strings(df):

    '''function to convert description column to list and concatenate its elements 

    to form a text-like structure stored as a string'''


    # Removing missing descriptions
    df.dropna(subset=['Description'], inplace=True)

    # Convert description column to list if it satisfies the condition
    def replace_description(row):

        
        if isinstance(row['Description'], str) and row['Description'].strip().startswith('['):
             row['Description']= ast.literal_eval(row['Description'])

             return row
        
     
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
            
            return row
        
        else:
            return row
    
    df = df.apply(replace_description, axis=1)
    
    # Remove empty quotes and 'About this space' from list
    df['Description'] = df['Description'].apply(lambda x: [item for item in x if item != ''])
    df['Description'] = df['Description'].apply(lambda x: [item for item in x if item != 'About this space'])
    df['Description'] = df['Description'].apply(lambda x: [item for item in x if item != 'The space'])
    df['Description'] = df['Description'].apply(lambda x: [item for item in x if item != 'License number'])
    df['Description'] = df['Description'].apply(lambda x: [item for item in x if item != 'Other things to note'])
    df['Description'] = df['Description'].apply(lambda x: [item for item in x if item != 'Guest acess'])
    
    # Join list elements into a single string
    df['Description'] = df['Description'].apply(lambda x: ' '.join(x))
    #reset index
    df.reset_index(inplace= True)

    return df

    

def set_default_feature_values(df):

    '''replacing empty entries with value 1 '''


    df['guests'] = df['guests'].apply(lambda x: 1 if pd.isnull(x) else x)
    df['beds'] = df['beds'].apply(lambda x: 1 if pd.isnull(x) else x)
    df['bathrooms'] = df['bathrooms'].apply(lambda x: 1 if pd.isnull(x) else x)
    df['bedrooms'] = df['bedrooms'].apply(lambda x: 1 if pd.isnull(x) else x)


    return df

def clean_tabular_data(raw_dataframe):
        
        ''' contains all the functions defined earlier
          to clean the data step by step'''

        df = remove_rows_with_missing_ratings(raw_dataframe)
        df = combine_description_strings(df)
        df = set_default_feature_values(df)

        return df

def load_airbnb(df= pd.DataFrame , label=str):
     
     ''' Function used to split the data into labels and features. 
     It takes as arguments the dataframe where the data is to be extracted and 
     the name of the variable that is wanted as label '''
     
    # filter out columns with text data

     numerical_cols= ['guests', 'beds', 'bathrooms', 'Price_Night', 'Cleanliness_rating',
            'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 
            'Value_rating', 'amenities_count', 'bedrooms']
     
     #labels = df[label].to_numpy(dtype= 'float64')
     labels = df[label]

     if label in numerical_cols:
        features = df[numerical_cols].drop(label, axis=1)
     else:
        features = df[numerical_cols]
     #features = features.to_numpy(dtype='float64')

     #features = df[numerical_cols].to_numpy(dtype= 'float64').drop(label, axis = 1).reset_index(drop=True)

     #features = df1.drop(label, axis=1)
     
   
    
     
     

     #features = features.values

     tuple_data = (features, labels)

     return tuple_data




if __name__ == '__main__':
     
     ''' saving the clean tabular data in a csv file'''
     
     df = pd.read_csv('listing.csv')
     df= clean_tabular_data(df)
     df.to_csv('clean_tabular_data.csv', index=False, mode = 'w' )
