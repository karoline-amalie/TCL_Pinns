import numpy as np
import math
import pandas as pd

from CreateTCLSimulation import TCL_system

def simulate_power_and_temp(n_loads, seed_value, Amb_temp):
    
    """Simulate a 1000 loads TCLs:
        - create data for TCLs
        - simulate 1000 loads
        - outputs mean of total power and ambient temperature in 5min time step
    """
    
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

    delta = 1   # temperature dead band
    T_set = 21  # the set temperature (the temperature we want)
    
    # empty dictionary to store data tcl simulation
    tcl_instance = {}
    power_dict = {}
    state_dict ={}


    # create n_loads tcls and simulate them
    for i in range(n_loads):
        tcl_instance[i] = TCL_system(T_set, delta, R_dist[i], C_dist[i], P_rate_dist[i], Dt, 0)
        power_dict[i], state_dict[i] = tcl_instance[i].simulation(Amb_temp, state=21)
        
    # store all power data from loads in lists of lists
    all_power = []
    for i in range(n_loads):
        all_power.append(power_dict[i])
      
    # sum power for all loads over each res (10s)
    total_power = [sum(x) for x in zip(*all_power)]
    
    # sum of power with time step size 5 min
    mean_power_list_5min = []
    for i in range(0, len(total_power), 30):
        interval = [x for x in total_power[i: i + 30]]
        av_power = np.mean(interval)
        mean_power_list_5min.append(av_power)

    # ambient temperature with time step size 5 min
    mean_ambient_temp_list_5min = []
    for i in range(0, len(Amb_temp), 30):
        interval = [x for x in Amb_temp[i: i + 30]]
        av_ambient_temp = np.mean(interval)
        mean_ambient_temp_list_5min.append(av_ambient_temp)
        
        
    return mean_ambient_temp_list_5min, mean_power_list_5min