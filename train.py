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

# SPLITTING THE TRAINING AND VALIDATION DATA
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1, random_state = 0)

# RESHAPING THE DATA
x_train = np.reshape(x_train, (-1,1))
x_val = np.reshape(x_val, (-1,1))
y_train = np.reshape(y_train, (-1,1))
y_val = np.reshape(y_val, (-1,1))

# SCALING THE DATA
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

xtrain_scaled = scaler_x.fit_transform(x_train)
ytrain_scaled = scaler_y.fit_transform(y_train)
xval_scaled = scaler_x.fit_transform(x_val)

# DESIGNING THE NEURAL NETWORK
model = Sequential()
model.add(Dense(2, input_dim = 1, activation = 'relu', kernel_initializer='normal'))
model.add(Dense(120, activation = 'relu'))
model.add(Dense(120, activation = 'relu'))
model.add(Dense(120, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

# TRAINING THE DATA
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse', 'mae', 'accuracy'])
history = model.fit(xtrain_scaled, ytrain_scaled, epochs = 150, batch_size = 40, validation_split = 0.1, verbose = 1)
print("\n\n ----- Model is trained successfully ! ----- \n\n")

# VISUALISING THE MODEL LOSS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MODEL LOSS')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()