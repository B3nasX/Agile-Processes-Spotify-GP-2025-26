import numpy as np

def calculate_mean():
    try:
        df = pd.read_csv("SpotifyFeatures.csv", skiprows=1)
        print("Data imported successfully:")
        print(df.head())

    except Exception as e:
        print("Error while importing CSV:")
        print(e)