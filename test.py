# SEA LEVEL PREDICTION

# DEVELOPED BY: 
# MOULISHANKAR M R

# IMPORTING MODULES
import numpy as np
import pandas as pd
from datetime import date
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# IMPORTING DATA
data = pd.read_csv("price.csv")

# PREPROCESSING DATA
x = []
initial_str = data["Date"][0]
initial = date(int(initial_str[-4:]),int(initial_str[3:5]),int(initial_str[:2]))
 
for i in range(len(data["Date"])):
    final_str = data["Date"][i]
    final = date(int(final_str[-4:]),int(final_str[3:5]),int(final_str[:2]))
    diff = (final - initial).days
    x.append(diff)

y = data["Price"].values

# RESHAPING THE DATA
x = np.reshape(x, (-1,1))
y = np.reshape(y, (-1,1))

# SCALING THE DATA
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y)

# LOADING THE TRAINED MODEL
model = load_model("model/model",custom_objects=None,compile=True)

# INPUT DATA
print("\nEnter the Time Period on when you want to explore the predicted Gold Rate !")
input_date = input("\nTime Period (DD-MM-YYYY) : ")

month = ["January","February","March","April","May","June","July","August","September","October","November","December"]

# PREPROCESSING INPUT DATA
x_str = date(int(input_date[-4:]),int(input_date[3:5]),int(input_date[:2]))
x_pred = (x_str - initial).days
x_pred = np.array(x_pred)
x_pred = np.reshape(x_pred, (-1,1))

# SCALING INPUT DATA
xpred_scaled = scaler_x.transform(x_pred)

# PREDICTING THE RESULTANT VALUE
ypred_scaled = model.predict(xpred_scaled)
y_pred = scaler_y.inverse_transform(ypred_scaled)

# DISPLAYING THE RESULTS
print(f"\n\n As per the prediction, the Gold Price on {input_date} might be -> ${round(float(y_pred),1)} per oz \n\n")