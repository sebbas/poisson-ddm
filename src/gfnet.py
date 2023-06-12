import sys, os
sys.path.append('../src')
import h5py as h5
import numpy as np
from tensorflow import keras
from datetime import datetime
import Poisson_model as PsnModel
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import time
from PIL import ImageFont

import tensorflow.keras.callbacks as KC

from Poisson_dataset import PoissonDataset
from Poisson_util import TimeHistory

keras.backend.set_floatx('float32')

parser = argparse.ArgumentParser()

# Regularizaer coefficients
parser.add_argument('-r', '--reg', type=float, nargs='*', default=None,\
                    help='l2 regularization')
parser.add_argument('-alpha', '--alpha', type=float, default=0.01, \
                    help='coefficient for data loss')
parser.add_argument('-a', '--activation', type=str, default='relu', \
                    help='activation function used in models layers')

# Epochs, checkpoints
parser.add_argument('-f', '--file', default='../data/psn_32_2000.h5', help='data file')
parser.add_argument('-name', '--name', default='psnNet', help='model name prefix')
parser.add_argument('-ie', '--initTrain', type=int, default=0, \
                    help='initial train epochs')
parser.add_argument('-e', '--nEpoch', default=2500, type=int, help='epochs')
parser.add_argument('-restart', '--restart', default=False, action='store_true',\
                    help='restart from checkpoint')
parser.add_argument('-ckpnt', '--checkpoint', default=None, help='checkpoint name')
parser.add_argument('-b', '--batchsize', default=200, type=int, help='batch size')
parser.add_argument('-t', '--tfdata', default=False, action='store_true',
                    help='use tf.data optimization')
parser.add_argument('-n', '--nSample',    type=int, default=40000, \
                    help = "number of samples")

# Plotting options
parser.add_argument('-v', '--visualize', default=False, action='store_true', \
                    help='enable model plotting functions')
parser.add_argument('-tbd', '--tboardDir', default='logs/', help='Tensorboard log directory')

# Learning rate
parser.add_argument('-lr0', '--lr0', type=float, default=5e-4, help='init leanring rate')
parser.add_argument('-lrmin', '--lrmin', type=float, default=1e-7, help='min leanring rate')
parser.add_argument('-p', '--patience', type=int, default=200, \
                    help='patience for reducing learning rate')

# Model architecture (symmetric U-Net by default)
parser.add_argument('-l', '--architecture', type=str, nargs='*', \
                    default=['input_{}_{}_{}_{}', \
                             'conv_3_64', \
                               'conv_3_128', \
                                 'maxpl_2_0', \
                                   'conv_3_256', \
                                     'maxpl_2_0', \
                                       'conv_3_512', \
                                         'conv_3_1024', \
                                         'tconv_3_512',\
                                       'tconv_3_256',\
                                     'bilinup_2', \
                                   'tconv_3_128',\
                                 'bilinup_2', \
                               'tconv_3_64',\
                             'tconv_3_1'],\
                    help='type and arguments of each layer')

args = parser.parse_args()

nSample = args.nSample
shape = (32, 32)
nChannel = 3 # bc, a, f
nDim = 2

# Insert values from args (input layer needs shape definition)
args.architecture[0] = args.architecture[0].format(shape[0], shape[1], nChannel, args.batchsize)
archStr   = '-'.join(args.architecture)
# Generate unreadable short architecture string (-> for shorter filenames)
tinyArch = []
for s in args.architecture:
  x = s.split('_')
  # Only keep first char of layer name + remove underscores
  tinyLayer = x[0][0] + ''.join(x[1:])
  tinyArch.append(tinyLayer)
tinyArch = ''.join(tinyArch)

usingTfData = args.tfdata
if usingTfData: print('Enabled tf.data optimization')

# Traing and validation split
nValid = int(nSample * 0.1)
nTrain = nSample - nValid
batchSize = args.batchsize
print('{} samples in training, {} in validation'.format(nTrain, nValid))
fname = '../data/psn_{}_{}.h5'.format(shape[0], nSample)

# Load external data
if not usingTfData:
  dFile = h5.File(fname, 'r')
  n        = dFile.attrs['nSample']
  assert nSample == n
  s        = dFile.attrs['shape']
  assert all(x == y for x, y in zip(shape, s))
  length   = dFile.attrs['length']
  aData    = np.array(dFile.get('a'))
  fData    = np.array(dFile.get('f'))
  ppData   = np.array(dFile.get('pp'))
  dFile.close()

  assert nSample == aData.shape[0] and nSample == fData.shape[0] and ppData.shape[0]

  # Construct bc array
  bcWidth     = 1
  # Times 2 because bc always on 2 sides in one dim
  pad         = bcWidth * 2
  sizeNoPad   = np.subtract(shape, pad) # Array size without padding
  # 0s on border of ones array
  onesWithPad = np.pad(np.ones(sizeNoPad), bcWidth)
  # Extra dim at beginning to match batchsize and at end to match channels
  onesExpand  = np.expand_dims(onesWithPad, axis=0)
  onesExpand  = np.expand_dims(onesExpand, axis=-1)
  # Repeat array 'nSample' times in 1st array dim
  bc          = np.repeat(onesExpand, nSample, axis=0)

  # Construct a, f, pp arrays - all need extra dim at end for channel
  a  = np.expand_dims(aData, axis=-1)
  f  = np.expand_dims(fData, axis=-1)
  pp = np.expand_dims(ppData, axis=-1)

  # Combine bc, a, f along channel dim
  bcAF = np.concatenate((bc, a, f), axis=-1)

# Create model
#with PsnModel.strategy.scope():
psnNet = PsnModel.PsnCnn(nCell=shape, operators=args.architecture, \
                         act=args.activation, last_act='linear', \
                         reg=args.reg, alpha=args.alpha)
psnNet.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr0))

# Create unique name based on args describing model
psnPrefix = args.name
psnSuffix = 'a-{}_e-{}_n-{}_b-{}_a-{}_p-{}_t-{}_l-{}' \
            .format(args.alpha, args.nEpoch, nSample, batchSize, args.activation, \
            args.patience, int(args.tfdata), tinyArch)
modelName = '{}_{}'.format(psnPrefix, psnSuffix)

# Plot model
if args.visualize:
  import visualkeras
  psnNet.build(input_shape=(args.batchsize, shape[1], shape[0], nChannel))
  model = psnNet.build_graph()
  model.summary()
  # Plot with keras
  keras.utils.plot_model(model, dpi=96, show_shapes=True, show_layer_names=True,
                         to_file='{}_modelplot_{}.png'.format(psnPrefix, psnSuffix),
                         expand_nested=False)
  # Also plot model architecture with visualkeras
  font = ImageFont.truetype('Arial Unicode.ttf', 34)
  visualkeras.layered_view(model, to_file='{}_visualkeras_{}.png'.format(psnPrefix, psnSuffix),
                           legend=True, font=font, spacing=80, scale_xy=20, scale_z=1)

# Callbacks
timeHistCB   = TimeHistory()
tboardCB     = KC.TensorBoard(log_dir=os.path.join(args.tboardDir, \
                                datetime.now().strftime("%Y%m%d-%H%M%S")), \
                              histogram_freq=1, \
                              profile_batch='500, 520')
checkpointCB = KC.ModelCheckpoint(filepath='./' + modelName + '/checkpoint', \
                                  monitor='val_loss', save_best_only=True,\
                                  save_weights_only=True, verbose=1)
reduceLrCB   = KC.ReduceLROnPlateau(monitor='loss', min_delta=0.01, \
                                    patience=args.patience, min_lr=args.lrmin)
csvLogCB     = keras.callbacks.CSVLogger(modelName + '.log', append=True)

psnCBs = [timeHistCB, tboardCB, checkpointCB, reduceLrCB, csvLogCB]

# Split samples into training, validation
if usingTfData:
  psnData   = PoissonDataset(fname=fname, nChannel=nChannel, shape=shape)
  psnData   = psnData.shuffle(buffer_size=2048)
  trainData = psnData.take(count=nTrain)
  validData = psnData.skip(count=nTrain)

#  numDatasets = 2
#  trainDatasetParallel = tf.data.Dataset.range(numDatasets) \
#      .interleave(lambda _: trainData, num_parallel_calls=tf.data.AUTOTUNE)
  trainDataset = trainData \
      .batch(batch_size=batchSize) \
      .cache() \
      .prefetch(buffer_size=tf.data.AUTOTUNE) \
      .repeat()

#  validDatasetParallel = tf.data.Dataset.range(numDatasets) \
#      .interleave(lambda _: validData, num_parallel_calls=tf.data.AUTOTUNE)
  validDataset = validData \
      .batch(batch_size=batchSize) \
      .cache() \
      .prefetch(buffer_size=tf.data.AUTOTUNE) \
      .repeat()
else:
  # Split into training / validation sets
  xTrain    = bcAF[:nTrain, ...]
  yTrain    = pp[:nTrain, ...]
  xValidate = bcAF[-nValid:, ...]
  yValidate = pp[-nValid:, ...]

# Training
startTrain = time.perf_counter()
if usingTfData:
  psnNet.fit(
      trainDataset,
      initial_epoch=args.initTrain,
      epochs=args.nEpoch,
      steps_per_epoch=nTrain//batchSize,
      callbacks=psnCBs,
      validation_data=validDataset,
      validation_steps=nValid//batchSize,
      verbose=True)
else:
  psnNet.fit(
      x=xTrain,
      y=yTrain,
      batch_size=batchSize,
      initial_epoch=args.initTrain,
      epochs=args.nEpoch,
      steps_per_epoch=nTrain//batchSize,
      callbacks=psnCBs,
      validation_data=(xValidate, yValidate),
      validation_steps=nValid//batchSize,
      verbose=True)
endTrain = time.perf_counter()
#print("fit() execution time in secs:", endTrain - startTrain)

# Evaluate callbacks
avgTimeEpoch = sum(timeHistCB.times) / len(timeHistCB.times)
print('{} samples ==> Average time per epoch: {}'.format(nSample, avgTimeEpoch))

with open('{}_epochtimes_{}.txt'.format(psnPrefix, psnSuffix), 'w') as f:
  for t in timeHistCB.times:
    f.write(f'{t}\n')
  f.write(f'Average: {avgTimeEpoch}\n')

