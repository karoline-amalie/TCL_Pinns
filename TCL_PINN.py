import numpy as np
from matplotlib import pyplot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf

from CreateData import simulate_power_and_temp



seed_value = 1234
n_loads = 1000

# time parameters
res = 10             # time resolution in seconds 
Dt = res/3600        # time step of the model in secs

# note 900 s is 15 min - 300 s is 5 min 
 
# temperature parameters  

read_data = pd.read_csv('csv (1).csv', delimiter=';')
Ambient = read_data['Middeltemperatur'].values
Ambient_resampled = np.repeat(Ambient, 1/Dt)
delta = 1   # temperature dead band
T_set = 21  # the set temperature (the temperature we want)


### import data for neural network
x_data, y_data = simulate_power_and_temp(n_loads, seed_value, Ambient_resampled)

#### turn x_data, y_data into arrays
x_data = np.array(x_data)[500:]
y_data = np.array(y_data)[500:]
# =============================================================================
# print(y_data.size)
# print(np.zeros((len(y_data),1)).size)
# #### add zero column to y_data (output) since we want to train pinn_error to be (close to) zero
# y_true = np.array((y_data, np.zeros((len(y_data),1)))).reshape(-1,2) 
# print(y_true.shape)
# =============================================================================

y_true = np.column_stack((y_data, np.zeros(len(y_data))))
print(y_true.shape)

T_scale = np.max(x_data) - np.min(x_data)
P_scale = np.max(y_true) - np.min(y_true)


def NormalizeData(data):
    return (data ) / (np.max(data) - np.min(data))

norm_x_data = NormalizeData(x_data)
norm_y_true = NormalizeData(y_true)


print(norm_y_true[0,0]*P_scale)
print(y_data[0])

#### create training, test, validation dataset with new y_true
# define training and test set - 80% training set
x_tr, x_tv, y_tr, y_tv = train_test_split(norm_x_data, norm_y_true, train_size=0.8, random_state=seed_value, shuffle=True)

# define test and val set - 10% test and validation set
x_val, x_test, y_val, y_test = train_test_split(x_tv, y_tv, train_size=0.5, random_state=seed_value, shuffle=True)

pyplot.scatter(x_val,y_val[:,0])


##### PINN model
from PinnModel import PinnModel

weight1 = 1
weight2 = 1

PINN_model = PinnModel(weight1, weight2, P_scale, T_scale)


# train model / fit model
history = PINN_model.fit(
    x_tr,
    y_tr,
    #steps_per_epoch = steps_per_epoch,
    validation_data = (x_val, y_val),
    verbose=1, 
    epochs=500)

results = PINN_model.evaluate(x_test, y_test, verbose=1)



def plot_loss_PINN(history):
  pyplot.plot(history.history['loss'], label='loss')
  pyplot.plot(history.history['val_loss'], label='val_loss')
  pyplot.ylim([0,990])
  pyplot.xlabel('Epoch')
  pyplot.ylabel('Error')
  pyplot.legend()
  pyplot.grid(True)

# plot loss and val loss 
plot_loss_PINN(history)

# predictions (still normalized)

y_pred_normalized = PINN_model.predict(x_test)
v = range(len(x_test))

# plot over input (ambient temp) and output (power) predictions
pyplot.scatter(x_test, y_test[:,0])
pyplot.scatter(x_test, y_pred_normalized[0][:])


#print(y_pred_normalized[0][:].numpy())

# plot over predictions (power) over time 
pyplot.plot(v, y_test[:,0])
pyplot.plot(v, y_pred_normalized[0][:])























