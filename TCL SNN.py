import numpy as np
import math
from matplotlib import pyplot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
tfk = tf.keras
import tensorflow_probability as tfp
tfd = tfp.distributions
import keras_tuner as kt


# random seed
#seed_value = 42
#seed_value = 34
#seed_value = 68
seed_value = 1234
np.random.seed(seed_value)

# create sample of parameters 
R_dist = {}
C_dist = {}
P_rate_dist = {}


for i in range(1000):
    R_dist[i] = np.random.normal(24, 2) # C/kW
    C_dist[i] = np.random.normal(1.05, 0.1) # kWh/C
    P_rate_dist[i] = (1.6*R_dist[i]*C_dist[i])/25.2 + np.random.normal(0, 0.2) #kW    


# time parameters
res = 10             # time resolution in seconds 
Dt = res/3600        # time step of the model in secs

# note 900 s is 15 min - 300 s is 5 min 
 
# temperature parameters
read_data = pd.read_csv('cvs(1).csv',delimiter=';')  

Ambient = read_data['Middeltemperatur'].values
Ambient_resampled = np.repeat(Ambient, 1/Dt)
delta = 1   # temperature dead band
T_set = 21  # the set temperature (the temperature we want)

class TCL_system:
    def __init__(self, theta_set, delta, R, C, P_rate, Dt, on_off):
        self.theta_set = theta_set
        self.delta = delta
        self.R = R
        self.C = C
        self.P_rate = P_rate
        self.Dt = Dt
        self.on_off = on_off
        
    def update_function(self, state, theta_a):
            
        theta = state
            
        if state >= self.theta_set + self.delta:
            self.on_off = self.on_off*0
        elif state <= self.theta_set - self.delta:
            self.on_off = min(self.on_off + 1,1)
            
        state = math.exp((-self.Dt/(self.R*self.C)))*theta + (1-math.exp((-self.Dt/(self.R*self.C))))*(theta_a + self.on_off*self.R*self.P_rate)
            
        power = self.on_off*self.P_rate
            
        return state, power  
        
    def simulation(self, Amb_temp, state):
        power_list = []
        state_list = []
        for i in range(len(Amb_temp)):
            state, power = self.update_function(state, Amb_temp[i])
            power_list.append(power)
            state_list.append(state)
        
        return power_list, state_list


# empty dictionary to store data tcl simulation
tcl_instance = {}
power_dict = {}
state_dict ={}

n_loads = 1000   # number of loads (tcls)

# create n_loads tcls and simulate them
for i in range(n_loads):
    tcl_instance[i] = TCL_system(T_set, delta, R_dist[i], C_dist[i], P_rate_dist[i], Dt, 0)
    power_dict[i], state_dict[i] = tcl_instance[i].simulation(Amb_temp=Ambient_resampled, state=21)
# =============================================================================
# 
# 
# # graph one TCL simulation over 24 hours
# u = range(8000)
# fig, axs = pyplot.subplots(2)
# fig.suptitle('One TCL')
# axs[0].plot(u, power_dict[1][0:8000])
# axs[1].plot(u, state_dict[1][0:8000])
# #print(sum(power_dict[1]))
# 
# =============================================================================
        
# store all power data from loads in lists of lists
all_power = []
for i in range(n_loads):
    all_power.append(power_dict[i])
  
# sum power for all loads over each res (10s)
total_power = [sum(x) for x in zip(*all_power)]
print(np.mean(total_power))

# =============================================================================
# # sum of power with time step size 15 min
# mean_power_list_15min = []
# for i in range(0, len(total_power), 90):
#     interval = [x for x in total_power[i: i + 90]]
#     av_power = np.mean(interval)
#     mean_power_list_15min.append(av_power)
# 
# # ambient temperature with time step size 15 min
# mean_ambient_temp_list_15min = []
# for i in range(0, len(Ambient_resampled), 90):
#     interval = [x for x in Ambient_resampled[i: i + 90]]
#     av_ambient_temp = np.mean(interval)
#     mean_ambient_temp_list_15min.append(av_ambient_temp)
# 
# =============================================================================

# sum of power with time step size 5 min
mean_power_list_5min = []
for i in range(0, len(total_power), 30):
    interval = [x for x in total_power[i: i + 30]]
    av_power = np.mean(interval)
    mean_power_list_5min.append(av_power)

# ambient temperature with time step size 5 min
mean_ambient_temp_list_5min = []
for i in range(0, len(Ambient_resampled), 30):
    interval = [x for x in Ambient_resampled[i: i + 30]]
    av_ambient_temp = np.mean(interval)
    mean_ambient_temp_list_5min.append(av_ambient_temp)


# modeling our aggregated model
# average of parameters
C_av = 1.05     # since we know their Gaussian distribution 
R_av = 24

# Aggregated model in terms of power
norm_power = []
for i in range(len(mean_ambient_temp_list_5min)):
    norm_power.append((n_loads*(T_set-mean_ambient_temp_list_5min[i]))/R_av)
    
    

# compare aggregated model with simulations
time_5_min = np.arange(0, len(Ambient_resampled), 30).tolist()
pyplot.plot(time_5_min, norm_power)
pyplot.plot(time_5_min, mean_power_list_5min)

#### Standrad neural network 

## PINN
from PinnModel import PinnModel


# define input and output
x = np.array(mean_ambient_temp_list_5min) # input ambient temperature
y = np.array(mean_power_list_5min).reshape((-1,1)) 

################## make y 2-dim with a zero column!!!
# output power
y1 = np.column_stack((y, np.zeros((len(y),1))))
#pyplot.scatter(x, y[0])

# define training and test set - 80% training set
x_tr, x_tv, y_tr, y_tv = train_test_split(x, y1, train_size=0.8, random_state=seed_value, shuffle=True)

# define test and val set - 10% test and validation set
x_val, x_test, y_val, y_test = train_test_split(x_tv, y_tv, train_size=0.5, random_state=seed_value, shuffle=True)



def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

x = NormalizeData(x_tr)
y = NormalizeData(y_tr)
#y1 = np.array([y, np.zeros((len(y),1))]).reshape(-1,2)

x_test1 = NormalizeData(x_test)
y_test1 = NormalizeData(y_test)

x_val1 = NormalizeData(x_val)
y_val1 = NormalizeData(y_val)

# =============================================================================
# 
# 
# # SNN model
# model = tf.keras.Sequential([
#     tf.keras.layers.InputLayer(input_shape=(1, )),
#     tf.keras.layers.Dense(80, activation='relu'),
#     tf.keras.layers.Dense(50, activation='relu'),
#     tf.keras.layers.Dense(30, activation='relu'),
#     tf.keras.layers.Dense(10, activation='sigmoid'),
#     tf.keras.layers.Dense(1)
# ])
# 
# 
# batch_size = 50
# train_size = x.shape[0]
# steps_per_epoch = train_size/batch_size
# 
# 
# lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#   0.001,
#   decay_steps=steps_per_epoch*1000,
#   decay_rate=1,
#   staircase=False)
# 
# def get_optimizer():
#   return tf.keras.optimizers.Adam(lr_schedule)
# 
# model.compile(loss='mse', optimizer=get_optimizer(), metrics=['mse'])
# =============================================================================

weight1 = 1
weight2 = 0.5

model = PinnModel(weight1, weight2)

print(model.summary())

"from tensorflow https://www.tensorflow.org/tutorials/keras/regression#regression_using_a_dnn_and_a_single_input"
# train model / fit model
history = model.fit(
    x,
    y,
    #steps_per_epoch = steps_per_epoch,
    validation_data = (x_val, y_val),
    verbose=1, 
    epochs=500)

# =============================================================================
# # evaluate the model
# _, train_acc = model.evaluate(x, y[1], verbose=0)
# _, test_acc = model.evaluate(x_val1, y_val1[1], verbose=0)
# print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# 
# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# hist.tail()
# =============================================================================


def plot_loss_SNN(history):
  pyplot.plot(history.history['loss'], label='loss')
  pyplot.plot(history.history['val_loss'], label='val_loss')
  pyplot.ylim([0,0.075])
  pyplot.xlabel('Epoch')
  pyplot.ylabel('Error')
  pyplot.legend()
  pyplot.grid(True)

# plot loss and val loss 
plot_loss_SNN(history)

# predictions (still normalized)
y_pred_normalized = model.predict(x_test1)
v = range(len(x_test1))

print(y_test1[:,0].size)

# plot over input (ambient temp) and output (power) predictions
pyplot.scatter(x_test1, y_test1[:,0])
pyplot.scatter(x_test1, y_pred_normalized[0])
print(y_pred_normalized.numpy())
# plot over predictions (power) over time 
pyplot.plot(v, y_test1)
pyplot.plot(v, y_pred_normalized)

# revert back from normalization 
def revert(norm, data):
    return (np.max(data)-np.min(data))*norm+np.min(data)

mean = np.mean(y_test)
y_pred = revert(y_pred_normalized[0], y_test)

# plot of prediction 
#pyplot.subplot(311)
pyplot.title('Power 5 min intervals')
pyplot.plot(v, y_test, label='true')
pyplot.plot(v, y_pred, label='predection')
pyplot.hlines(y=mean, xmin=0, xmax=200, colors='aqua', linestyles='--', lw=2, label='mean')
pyplot.xlabel('time in s (step size 5 min)')
pyplot.ylabel('Power in kW')
pyplot.legend(loc='upper left')
pyplot.show()

