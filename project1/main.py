from load_data import load_data
import importlib_resources
import numpy as np
import pandas as pd

data, target = load_data() # returns a pandas dataframe

n_rows, n_cols = data.shape
print(n_rows, n_cols)

data_values = data.values

# Making the data matrix X by indexing into data.
cols = range(0, 29)

X = data_values[:, cols]

# Extracting the attribute names from the header
attributeNames = np.asarray(data.columns[cols])

classNames = np.unique(target)

print(classNames)