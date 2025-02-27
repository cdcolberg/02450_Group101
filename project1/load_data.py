import pandas as pd
import numpy as np

def load_data(data_dir = "./data/wdbc.csv"):
    "Loads a datafile...."

    # Pandas loads .data file
    data = pd.read_csv(data_dir)
    # Add columns
    _, n_cols = data.shape

    # We drop the ID column

    data = data.drop("ID", axis=1)
    target = data["Diagnosis"]
    data = data.drop("Diagnosis", axis=1)

    # Convert columns to floats
    data.astype(float)

    return data, target

if __name__ == "__main__":
    print(load_data())