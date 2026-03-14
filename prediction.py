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

for r, level, soil in zip(rainfall, river_level, soil_moisture): 
    score = r*0.03 + level*0.8 + soil*0.01#through previous research, rainfall has some contribution. river level is super important and soil moisture is moderate, so these values are made up for now as a weighted score.

    if score < 17:
        risk.append(0) #low risk!
    elif score < 20:
        risk.append(1) #medium risk
    else:
        risk.append(2) #high risk.

data = pd.DataFrame({
    "rainfall_mm" : rainfall,
    "river_level_m": river_level,
    "soil_moisture_pct": soil_moisture,
    "elevation_m": elevation,
    "flood_risk": risk
})

data.head()


X = data[["rainfall_mm", "river_level_m", "soil_moisture_pct", "elevation_m"]]
y = data["flood_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)


importance = model.feature_importances_

plt.bar(X.columns, importance)
plt.title("Feature Importance for Flood Prediction")
plt.ylabel("Importance")
plt.xlabel("Environmental Features")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, predictions)

plt.title("Flood Risk Prediction Confusion Matrix")
plt.show()


example = [[35, 2.8, 75, 25]]

prediction = model.predict(example)

risk_labels = ["Low", "Medium", "High"]

print("Predicted Flood Risk:", risk_labels[prediction[0]])