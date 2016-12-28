# imports
import pandas as pd
import matplotlib.pyplot as plt
# this allows plots to appear directly in the notebook
%matplotlib inline

# read data into a DataFrame
data = pd.read_csv('data/train.csv', index_col=0)
data.head()