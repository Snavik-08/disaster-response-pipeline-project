# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 20:11:48 2021

@author: snagowsv
"""

# import libraries
import pandas as pd
import sys
from sqlalchemy import create_engine


def load_data(messages_path, categories_path):
    
    """
    Loadinf and mergeing of data from two given paths.
    
    Parameters:
    messages_path: path to messages data as csv file
    categories_path: path to categories as csv file
    
    Returns:
    df: dataframe containing data of messages_path and categories_path merged
    
    """
    messages = pd.read_csv(messages_path)
    categories = pd.read_csv(categories_path)
    df = pd.merge(messages, categories, how="inner", on="id" )
    return(df)
    
    
def cleaning_data(df):
    
    """
    Cleaning and transforming data frame.
    
    Parameters:
    df: DataFrame to clean
    
    Returns:
    df: Cleaned DataFrame
    
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(pat=";", expand= True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    
    #apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = pd.Series(row).apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: str(x)[-1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    categories['related'] = categories['related'].replace(to_replace=2, value=1)

    # drop the original categories column from `df`
    df.drop("categories",  axis=1,  inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop(df[df.duplicated()].index, inplace= True)
    return(df)
    
def save_df(df, database_filepath):
    """
    Saves data in SQLite db
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')
    
    

    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = cleaning_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_df(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
    
    
        