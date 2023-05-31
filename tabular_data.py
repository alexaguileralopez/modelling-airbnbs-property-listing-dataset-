# file to manipulate tabular data
import pandas as pd
import ast
import numpy as np

# function to remove rows with missing ratings
def remove_rows_with_missing_ratings(df):

    df = df[~df['Cleanliness_rating'].isna()]
    df = df[~df['Accuracy_rating'].isna()]
    df = df[~df['Location_rating'].isna()]
    df = df[~df['Check-in_rating'].isna()]
    df = df[~df['Value_rating'].isna()]

    return df

def combine_description_strings(df):
    # Removing missing descriptions
    df.dropna(subset=['Description'], inplace=True)

    # Convert description column to list if it satisfies the condition
    def replace_description(row):

        
        if isinstance(row['Description'], str) and row['Description'].strip().startswith('['):
             row['Description']= ast.literal_eval(row['Description'])

             return row
        
        #elif isinstance(next_column, str) and next_column.strip().startswith('['):
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

    return df

def set_default_feature_values(df):

    # replacing empty entries with value 1

    df['guests'] = df['guests'].apply(lambda x: 1 if x== np.nan else x)
    df['beds'] = df['beds'].apply(lambda x: 1 if x== np.nan else x)
    df['bathrooms'] = df['bathrooms'].apply(lambda x: 1 if x== np.nan else x)
    df['bedrooms'] = df['bedrooms'].apply(lambda x: 1 if x== np.nan else x)

    return df

def clean_tabular_data(raw_dataframe):

        #df = pd.read_csv(raw_dataframe)

        df = remove_rows_with_missing_ratings(raw_dataframe)
        df = combine_description_strings(df)
        df = set_default_feature_values(df)

        return df

def load_airbnb(df= pd.DataFrame , label=str):
     
     labels = df[label]
     features = df.drop(label, axis=1)

     tuple_data = (features, labels)

     return tuple_data

if __name__ == '__main__':
     
     df = pd.read_csv('listing.csv')
     df= clean_tabular_data(df)
     df.to_csv('clean_tabular_data.csv', index=False, mode = 'w' )
