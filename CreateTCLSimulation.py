import math 


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
        
        