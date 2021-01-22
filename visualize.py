# INDIAN CURRENCY VALUE PREDICTION

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
