from PinnLayer import PinnLayer

import tensorflow as tf

class PinnModel(tf.keras.models.Model):
    
    def __init__(self, weight1, weight2):
        super(PinnModel, self).__init__()
        
        
        self.StandardModel = tf.keras.Sequential([
            tf.keras.layers.Dense(80, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(30, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
                
    
        
        loss_weights = [weight1, weight2]
        
        self.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss=tf.keras.losses.mean_absolute_error
                     #loss_weights=loss_weights
                     )
        
        self.build(input_shape=(None , 1))
        
    def Get_Pinn_error(self, P_out, ambient_temp, R_av, T_set):
        pinn_error = tf.reduce_sum(tf.square(P_out-tf.multiply(1000/R_av,(T_set-ambient_temp))))
        
        return pinn_error
    

    
    def call(self, inputs, training=None, mask=None):
        xtest = inputs
        
        R_av = 24
        T_set = 21
            

        Power_output = self.StandardModel(xtest)    # output tensoflow object - converted with .numpy()
        Pinn_error_output = self.Get_Pinn_error(Power_output, inputs, R_av, T_set)

       
        
        network_output = tf.stack([Power_output, Pinn_error_output], axis=0) 
        tf.print(network_output)
        
        return network_output
    