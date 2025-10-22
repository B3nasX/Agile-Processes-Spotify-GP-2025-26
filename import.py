import pandas as pd

def load_data():
    try:
        df = pd.read_csv("SpotifyFeatures.csv", skiprows=1)
        print("Data imported successfully:")
        print(df.head())

    except Exception as e:
        print("Error while importing CSV:")
        print(e)
    

if __name__ == "__main__":
    load_data()