import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

np.random.seed(42) # random seed to gain random number values

rows = 300 # for example, 300 data samples/values

#gamma(shape, scale, size)!!

rainfall = np.random.gamma(2.0, 10.0, rows) #produce rainfall values using gamma 
river_level = np.random.gamma(2.0, 0.7, rows) 
soil_moisture = np.random.gamma(20, 90, rows) 
elevation = np.random.gamma(5, 120, rows) 

risk = [] #array to store the risk being produced

