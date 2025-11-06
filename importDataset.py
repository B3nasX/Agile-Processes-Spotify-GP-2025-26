import pandas as pd

# File for importing the csv into a more advanced 2D matrix 
# data_set -> The dataset inside the 2D matrix.


def loadData():
    try:
        data_set = pd.read_csv("SpotifyFeatures.csv", skiprows=0)
        print("Data imported successfully:")
        return data_set
    
    except Exception as e:
        print("Error while importing CSV:")
        print(e)
    

if __name__ == "__main__":
    # Just to test
    data_set = loadData()
    print(data_set.head())
