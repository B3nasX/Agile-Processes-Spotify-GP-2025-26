import sys
import os
# Going to upper directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importDataset

data_set = importDataset.loadData()

# A function to find all null values in a csv
# Argument is the data_set loaded from "importDataset.py"
def findNullsInCSV(data_set):
    # Find null values in the DataFrame
    nulls = data_set.isnull()
    # ^ Currently unused but that can change

    # Get the count of null values in each column
    null_counts = data_set.isnull().sum()
    return null_counts

if __name__ == "__main__":
    data_set_null_count = findNullsInCSV(data_set)
    print(data_set_null_count)