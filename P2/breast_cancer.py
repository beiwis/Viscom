import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patheffects
from sklearn import datasets

# Load the dataset
boobs = datasets.load_breast_cancer()
x = boobs.data
feature_names = boobs.feature_names
print(f"feature_names: {feature_names}")
y = boobs.target
target_names = boobs.target_names
labels = list(range(len(target_names)))
print(f"\ntarget_names: {target_names} (labels: {labels})")

# Create a DataFrame
data = dict([(n, x[:,i]) for i, n in enumerate(feature_names)]+[("tumor class", target_names[y]), ("label", y)])
df = pd.DataFrame(data)
print(df)

# Shuffle the data
def shuffle_data(df: pd.DataFrame):
  """
  Shuffles a DataFrame rows randomly.
  """
  return df.sample(frac=1.0, random_state = 0).reset_index(drop = True)
  
df_shuffle = shuffle_data(df)



