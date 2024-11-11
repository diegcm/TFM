#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import os
import pandas as pd
from tensorflow.keras.callbacks import Callback

class HistorySaverCallback(Callback):  # Ahora hereda de Keras Callback
    def __init__(self, history, save_dir, save_interval, initial_steps=0):
        super(HistorySaverCallback, self).__init__()
        self.save_dir = save_dir
        self.history = history
        self.save_interval = save_interval
        self.total_steps = initial_steps
        os.makedirs(save_dir, exist_ok=True)

    def save_history(self):
        # Verificar si se debe guardar la memoria según el intervalo de pasos
        if self.total_steps % self.save_interval == 0:
            df_historial = pd.DataFrame(self.history.history)
            df_historial.to_csv(os.path.join(self.save_dir, f'historial_entrenamiento_{self.total_steps}.csv'), index=False)
            

    def on_step_end(self, step, logs=None):
        # Incrementar el contador de pasos y guardar si es necesario
        self.total_steps += 1
        self.save_history()