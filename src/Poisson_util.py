import numpy as np
import time
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd


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


def _getFigureTitle(eqId, archId):
  equationStrs = ['Homogeneous Poisson', 'Inhomogeneous Laplace', 'Inhomogeneous Poisson']
  architectureStrs = ['Simplified U-Net', 'Full U-Net + dropout', 'Full U-Net + batchnorm',]
  return 'Equation: {}, Architecture: {}'.format(equationStrs[eqId], architectureStrs[archId])


def getFileName(type, name, eqId, archId):
  return '{}_{}_eq-{}_arch-{}'.format(name, type, eqId, archId)


def plotPredictions(sP, nPred, bcAF, p, phat, name, eqId, archId):
  eP = sP + nPred
  errorP = np.abs(p[sP:eP,:,:,0] - phat[0:nPred,:,:,0])

  # Get statistics per frame
  maeLst, rmseLst, mapeLst = [], [], []
  for i in range(nPred):
    maeLst.append(mae(p[sP+i,:,:,0], phat[i,:,:,0]))
    rmseLst.append(rmse(p[sP+i,:,:,0], phat[i,:,:,0]))
    mapeLst.append(mape(p[sP+i,:,:,0], phat[i,:,:,0]))

  # Plots
  fig = plt.figure(figsize=(21, 1+2*nPred), dpi=120, constrained_layout=True)
  fig.suptitle(_getFigureTitle(eqId, archId), fontsize=16)

  # Min, max values, needed for colobar range
  minBc, maxBc     = np.min(bcAF[sP:eP,:,:,0]), np.max(bcAF[sP:eP,:,:,0])
  minA, maxA       = np.min(bcAF[sP:eP,:,:,1]), np.max(bcAF[sP:eP,:,:,1])
  minF, maxF       = np.min(bcAF[sP:eP,:,:,2]), np.max(bcAF[sP:eP,:,:,2])
  minP, maxP       = np.min(p[sP:eP,:,:,0]), np.max(p[sP:eP,:,:,0])
  minPHat, maxPHat = np.min(phat[0:nPred,:,:,0]), np.max(phat[0:nPred,:,:,0])
  minPErr, maxPErr = np.min(errorP[0:nPred,:,:]), np.max(errorP[0:nPred,:,:])

  nCols = 7
  for i in range(nPred):
    cnt = i*nCols
    iInput = sP+i

    ax = fig.add_subplot(nPred, nCols, cnt+1)
    plt.ylabel("Frame {}".format(sP+i))
    plt.title('bc')
    plt.imshow(bcAF[iInput,:,:,0], vmin=minBc, vmax=maxBc, origin='lower')
    plt.colorbar()

    ax = fig.add_subplot(nPred, nCols, cnt+2)
    plt.title('a')
    plt.imshow(bcAF[iInput,:,:,1], vmin=minA, vmax=maxA, origin='lower')
    plt.colorbar()

    ax = fig.add_subplot(nPred, nCols, cnt+3)
    plt.title('f')
    plt.imshow(bcAF[iInput,:,:,2], vmin=minF, vmax=maxF, origin='lower')
    plt.colorbar()

    ax = fig.add_subplot(nPred, nCols, cnt+4)
    plt.title('p')
    plt.imshow(p[iInput,:,:,0], vmin=minP, vmax=maxP, origin='lower')
    plt.colorbar()

    ax = fig.add_subplot(nPred, nCols, cnt+5)
    plt.title('phat')
    plt.imshow(phat[i,:,:,0], vmin=minP, vmax=maxP, origin='lower')
    plt.colorbar()

    ax = fig.add_subplot(nPred, nCols, cnt+6)
    plt.title('abs(p-phat)')
    plt.imshow(errorP[i,:,:], vmin=minPErr, vmax=maxPErr, origin='lower')
    plt.colorbar()

    ax = fig.add_subplot(nPred, nCols, cnt+7)
    ax.axis('off')
    plt.title('Stats')
    plt.text(0.2, 0.7, 'MAPE: %.4f %%' % mapeLst[i])
    plt.text(0.2, 0.5, 'RMSE: %.4f' % rmseLst[i])
    plt.text(0.2, 0.3, 'MAE: %.4f' % maeLst[i])

  fname = getFileName('predictions', name, eqId, archId)
  plt.savefig('../img/{}.png'.format(fname), bbox_inches='tight')
  plt.close(fig)


def plotLosses(model, epochStart, epochEnd, name, eqId, archId, plotLr=False):
  history = pd.read_csv(model + '.log', sep = ',', engine='python')
  hist    = history[epochStart:epochEnd]

  subplotTitles = ['MSE loss', 'MAE', 'Data loss', 'PDE loss']
  metrics       = [['loss', 'val_loss'], ['mae', 'val_mae'], ['data', 'val_data'], ['pde', 'val_pde']]

  nCols, nRows = 2, 2
  fig = plt.figure(figsize=(8*nCols, 4*nRows), dpi=120)
  fig.suptitle(_getFigureTitle(eqId, archId), fontsize=16)
  fig.subplots_adjust(hspace=.5)

  for i, names in enumerate(metrics):
    ax = fig.add_subplot(nRows, nCols, i+1)
    ax.set_yscale('log')
    for n in names:
      assert n in names
      plt.plot(hist[n])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    ax.title.set_text(subplotTitles[i])
    legend = metrics[i]
    if plotLr:
      plt.plot(hist['lr'], color='C3')
      legend.append('lr')
    plt.legend(legend, loc='upper right')

  fname = getFileName('losses', name, eqId, archId)
  plt.savefig('../img/{}.png'.format(fname), bbox_inches='tight')
  plt.close(fig)


def plotModel(net, name, eqId, archId):
  import visualkeras
  from PIL import ImageFont

  model = net.build_graph()
  model.summary()
  # Plot model graph with keras
  fname = getFileName('graph', name, eqId, archId)
  keras.utils.plot_model(model, dpi=96, show_shapes=True, show_layer_names=True,
                         to_file='../img/{}.png'.format(fname), expand_nested=False)

  # Plot model architecture with visualkeras
  font = ImageFont.truetype('Arial Unicode.ttf', 200)
  fname = getFileName('layers', name, eqId, archId)
  visualkeras.layered_view(model, to_file='../img/{}.png'.format(fname),
                           legend=True, font=font, spacing=80, scale_xy=180, max_xy=10000, scale_z=0.5, draw_volume=1)


def writeEpochTimes(name, eqId, archId, timeHistCB):
  avgTimeEpoch = sum(timeHistCB.times) / len(timeHistCB.times)
  print('Average time per epoch: {}'.format(avgTimeEpoch))

  fname = getFileName('epochtimes', name, eqId, archId)
  with open('{}.txt'.format(fname), 'w') as efile:
    for t in timeHistCB.times:
      efile.write(f'{t}\n')
    efile.write(f'Average: {avgTimeEpoch}\n')

