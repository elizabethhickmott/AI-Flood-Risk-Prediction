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

for r, level, soil in zip(rainfall, river_level, soil_moisture, elevation): 
    score = r*0.03 + level*0.8 + soil*0.01#through previous research, rainfall has some contribution. river level is super important and soil moisture is moderate, so these values are made up for now as a weighted score.

    if score < 2.2:
        risk.append(0) #low risk!
    elif score < 3.2:
        risk.append(1) #medium risk
    else:
        risk.append(2) #high risk.

data = pd.DataFrame({
    "rainfall_mm " : rainfall,
    "river_level_m ": river_level,
    "soil_moisture_pct ": soil_moisture,
    "elevation_m ": elevation,
    "flood_risk ": risk
})

data.head()