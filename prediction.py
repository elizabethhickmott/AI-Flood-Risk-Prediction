import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

np.random.seed(42) # random seed to gain random number values

rows = 300 # for example, 300 data samples/values

#gamma(shape, scale, size)!!

rainfall = np.random.gamma(2.0, 10.0, rows) #produce rainfall values using gamma, mean around 20mm
river_level = np.random.gamma(2.0, 0.7, rows) # river level in metres, mean around 1.4
soil_moisture = np.random.gamma(2, 15, rows) # soil moisture as a %, mean  around30%
elevation = np.random.gamma(5, 120, rows) # elevation in metres, mean  around 600m


risk = [] #array to store the risk class (0= low, 1 = medium and 2 = high)

for r, level, soil in zip(rainfall, river_level, soil_moisture): 
    score = r*0.03 + level*0.8 + soil*0.01 #through previous research, rainfall has some contribution (hence coefficient chosen as 0.03); river level is super important (hence coefficient 0.8); and soil moisture is moderate (so coefficient is 0.01), so these values are made up for now as a weighted score.
    #I decided not to include elevation in this score as for the tim ebeing it will be a feature for the model to learn from

    #the three flood risk groups:
    if score < 2.2:
        risk.append(0) #low risk!
    elif score < 3.2:
        risk.append(1) #medium risk
    else:
        risk.append(2) #high risk.

#Builds the data frame by combining all features and the putting the risk label into single pandads
data = pd.DataFrame({
    "rainfall_mm" : rainfall,
    "river_level_m": river_level,
    "soil_moisture_pct": soil_moisture,
    "elevation_m": elevation,
    "flood_risk": risk #what the model should end up prediciting
})

data.head() # to see the first 5 rows 


X = data[["rainfall_mm", "river_level_m", "soil_moisture_pct", "elevation_m"]] #the input features
y = data["flood_risk"] #output/target

X_train, X_test, y_train, y_test = train_test_split( # 20% of data for testing and 80% for training
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42) #Initialising the random forest classifier as discussed in poster

model.fit(X_train, y_train) #training the model on the training data

predictions = model.predict(X_test) #generate the predictions on the unseen data -> the test data 

accuracy = accuracy_score(y_test, predictions) #compare the predicitions to the true labels

print("Model Accuracy:", accuracy) #Print the overall accuracy of the model, between 0.0-1.0


importance = model.feature_importances_ #shows how much each feature contributed

plt.bar(X.columns, importance) 
plt.title("Feature Importance for Flood Prediction")
plt.ylabel("Importance")
plt.xlabel("Environmental Features")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, predictions) #create a confusion matrix to show true values to the predictions.

plt.title("Flood Risk Prediction Confusion Matrix")
plt.show()

example = pd.DataFrame([[35, 2.8, 75, 25]], columns=X.columns) #hypothetical situation - can the model predict correctly?

prediction = model.predict(example) #returns an array e.g. [0], [1], or [2]

risk_labels = ["Low", "Medium", "High"] #changes the numeric value back to a readable value

print("Predicted Flood Risk:", risk_labels[prediction[0]]) #print the flood risk generated from the values 