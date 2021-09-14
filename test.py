import pandas as pd
import numpy as np

data =  pd.read_csv('student-mat.csv', sep=';')

print(data.head())

data = data[['G1', 'G2', 'G3', 'failures']]
