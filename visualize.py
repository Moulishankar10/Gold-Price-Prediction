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

# SCALING THE DATA
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y)

# LOADING THE TRAINED MODEL
model = load_model("model/model",custom_objects=None,compile=True)

# PREDICTING THE MODEL
y_est = model.predict(x_scaled)
y_est = scaler_y.inverse_transform(y_est)


# VISUALISING THE MODEL PERFORMANCE
plt.plot(x,y, color = 'blue')
plt.plot(x,y_est, color = 'red')
plt.title('Gold Price Prediction - MODEL PERFORMANCE')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(['Actual Data', 'Predicted Data'], loc='upper left')
plt.show()
