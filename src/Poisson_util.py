import numpy as np
import time
from tensorflow import keras

class TimeHistory(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.times = []

  def on_epoch_begin(self, batch, logs={}):
    self.epoch_time_start = time.time()

  def on_epoch_end(self, batch, logs={}):
    self.times.append(time.time() - self.epoch_time_start)


def rmse(targets, predictions):
  return np.sqrt(np.mean((targets-predictions)**2))


def mae(targets, predictions):
  return np.mean(np.abs(targets-predictions))


def mape(targets, predictions):
  return np.mean(np.abs((targets - predictions) / targets)) * 100.0
