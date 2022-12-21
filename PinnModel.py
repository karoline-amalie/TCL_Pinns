#from PinnLayer import PinnLayer

import tensorflow as tf

class PinnModel(tf.keras.models.Model):
    
    def __init__(self, weight1, weight2, P_scale, T_scale):
        super(PinnModel, self).__init__()
        
        self.P_scale = P_scale
        self.T_scale = T_scale
        
        self.StandardModel = tf.keras.Sequential([
            #tf.keras.layers.InputLayer(input_shape=(1, )),
            tf.keras.layers.Dense(80, activation='relu'),
            tf.keras.layers.Dense(60, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(30, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
                
    
        loss_weights = [weight1, weight2]
        
        self.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss=tf.keras.losses.MeanSquaredError(),
                     loss_weights=loss_weights
                     )
        
        self.build(input_shape=(None, 1))
        
    def Get_Pinn_error(self, P_out, ambient_temp, R_av, T_set):
        #pinn_error = tf.reduce_sum(tf.square(P_out-(tf.multiply((1000/R_av),(T_set-ambient_temp)))))
        #pinn_error = tf.keras.backend.sum(tf.square(P_out-((1/self.P_scale)*tf.multiply((1000/R_av),(T_set-(self.T_scale*ambient_temp))))))
        
        T_set = tf.cast(T_set, tf.float32)
        
        pinn_eq = tf.math.scalar_mul((1000/(self.P_scale*R_av)), (tf.math.subtract(T_set,(tf.math.scalar_mul(self.T_scale,ambient_temp)))))
        
        pinn_error = tf.keras.backend.sum(tf.square(tf.math.subtract(P_out,pinn_eq)))
        
        
        tf.print(pinn_error)
        
        
        return pinn_error
    

    
    def call(self, inputs, training=None, mask=None):
        #xtest = inputs
        
        R_av = 24
        T_set = 21
            

        Power_output = self.StandardModel(inputs)    
        Pinn_error_output = self.Get_Pinn_error(Power_output, inputs, R_av, T_set)

        #print(Power_output)
        #tf.print(Power_output)
        
        #tf.print(Pinn_error_output)
        
        network_output = (Power_output, Pinn_error_output)
        #tf.print(network_output)
        
        #tf.print(network_output)
        
        return network_output
    

