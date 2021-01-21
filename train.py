# SEA LEVEL PREDICTION

# DEVELOPED BY: 
# MOULISHANKAR M R

# IMPORTING MODULES
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import save_model 

# IMPORTING DATA
data = pd.read_csv("data/price.csv")

# PREPROCESSING DATA
data = pd.read_csv("price.csv")
x = []
initial_str = data["Date"][0]
initial = date(int(initial_str[-4:]),int(initial_str[3:5]),int(initial_str[:2]))

for i in range(len(data["Date"])):
    final_str = data["Date"][i]
    final = date(int(final_str[-4:]),int(final_str[3:5]),int(final_str[:2]))
    diff = (final - initial).days
    x.append(diff)

y = data["Price"].values

