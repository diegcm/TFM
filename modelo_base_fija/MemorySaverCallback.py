#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import os
from tensorflow.keras.callbacks import Callback

class MemorySaverCallback(Callback):  # Ahora hereda de Keras Callback
    def __init__(self, memory, save_dir, save_interval, initial_steps=0):
        super(MemorySaverCallback, self).__init__()
        self.memory = memory
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.total_steps = initial_steps
        os.makedirs(save_dir, exist_ok=True)

    def save_deques(self):
        # Verificar si se debe guardar la memoria según el intervalo de pasos
        if self.total_steps % self.save_interval == 0:
            # Obtener el diccionario con los deques (memorias)
            dictionary = self.memory.return_deque_dict()
            
            # Guardar el archivo con un nombre basado en los pasos totales
            filename = f'dict_{self.total_steps}.pkl'
            filepath = os.path.join(self.save_dir, filename)
            
            print(f"Saving deques at step {self.total_steps} to {filepath}")
            
            with open(filepath, 'wb') as f:
                pickle.dump(dictionary, f)

    def on_step_end(self, step, logs=None):
        # Incrementar el contador de pasos y guardar si es necesario
        self.total_steps += 1
        self.save_deques()