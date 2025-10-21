import numpy as np  
import pandas as pd 

#A function to find all null values in a csv    

    
def find_nulls_in_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Find null values in the DataFrame
    nulls = df.isnull()
    
    # Get the count of null values in each column
    null_counts = df.isnull().sum()
    
    print(f'\n{null_counts}')

find_nulls_in_csv('SpotifyFeatures.csv')
