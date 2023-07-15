import sys, os
sys.path.append('../src')
import h5py as h5
import numpy as np
from tensorflow import keras
from datetime import datetime
import argparse
import tensorflow as tf
import time

import tensorflow.keras.callbacks as KC

import Poisson_dataset as PD
import Poisson_util as UT
import Poisson_model as PM

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
parser.add_argument('-tf', '--tfdata', default=False, action='store_true', \
                    help='use tf.data optimization')
parser.add_argument('-n', '--nSample', type=int, default=40000, \
                    help = "number of samples")
parser.add_argument('-eq', '--equation', type=int, default=2, \
                    help = "0: Homogeneous Poisson, 1: Inhomogeneous Laplace, 2: Inhomogeneous Poisson")
parser.add_argument('-t', '--train', type=int, default=1, \
                    help = "enable / disable training")
parser.add_argument('-p', '--predict', type=int, default=0, \
                    help = "enable / disable prediction by specifying number of predictions")

# Plotting options
parser.add_argument('-v', '--visualize', default=False, action='store_true', \
                    help='enable model plotting functions')
parser.add_argument('-tbd', '--tboardDir', default='logs/', help='Tensorboard log directory')

# Learning rate
parser.add_argument('-lr0', '--lr0', type=float, default=5e-4, help='init leanring rate')
parser.add_argument('-lrmin', '--lrmin', type=float, default=1e-7, help='min leanring rate')
parser.add_argument('-pa', '--patience', type=int, default=200, \
                    help='patience for reducing learning rate')
parser.add_argument('-lr',  '--restartLr', type=float, default=None,
                     help='learning rate to restart training')
parser.add_argument('-l', '--architecture', type=int, default=2, \
                    help='architecture id (0: simplified U-Net, 1: Full U-Net + dropout, 2: Full U-Net + batchnorm)')

args = parser.parse_args()

nSample    = args.nSample
archId     = args.architecture
activation = args.activation
name       = args.name
eqId       = args.equation

# Simplified U-Net
if archId == 0:
  architecture = ['input_{}_{}_{}_{}',             #  0
                  'conv_3_64_0',                   #  1
                    'conv_3_128_0',                #  2
                      'maxpl_2_0',                 #  3
                        'conv_3_256_0',            #  4
                          'maxpl_2_0',             #  5
                            'conv_3_512_0',        #  6
                              'conv_3_1024_0',     #  7
                              'tconv_3_512_0_1',   #  8
                            'tconv_3_256_0_1',     #  9
                          'bilinup_2',             # 10
                        'tconv_3_128_0_1',         # 11
                      'bilinup_2',                 # 12
                    'tconv_3_64_0_1',              # 13
                  'tconv_3_1_0_1']                 # 14

# Full U-Net with dropout
elif archId == 1:
  architecture = ['input_{}_{}_{}_{}',             #  0
                    'conv_3_64_1',                 #  1
                    'conv_3_64_1',                 #  2
                    'maxpl_2_0',                   #  3
                      'conv_3_128_1',              #  4
                      'conv_3_128_1',              #  5
                      'maxpl_2_0',                 #  6
                        'conv_3_256_1',            #  7
                        'conv_3_256_1',            #  8
                        'maxpl_2_0',               #  9
                          'conv_3_512_1',          # 10
                          'conv_3_512_1',          # 11
                          'maxpl_2_0',             # 12
                            'conv_3_1024_1',       # 13
                            'conv_3_1024_1',       # 14
                          'tconv_3_512_1_2',       # 15
                          'concat_15_11',          # 16
                          'dropout_3',             # 17
                          'conv_3_512_1',          # 18
                          'conv_3_512_1',          # 19
                        'tconv_3_256_1_2',         # 20
                        'concat_20_8',             # 21
                        'dropout_3',               # 22
                        'conv_3_256_1',            # 23
                        'conv_3_256_1',            # 24
                      'tconv_3_128_1_2',           # 25
                      'concat_25_5',               # 26
                      'dropout_3',                 # 27
                      'conv_3_128_1',              # 28
                      'conv_3_128_1',              # 29
                    'tconv_3_64_1_2',              # 30
                    'concat_30_2',                 # 31
                    'dropout_3',                   # 32
                    'conv_3_64_1',                 # 33
                    'conv_3_64_1',                 # 34
                  'conv_3_1_1']                    # 35

# Full U-Net with batchnorm
elif archId == 2:
  architecture = ['input_{}_{}_{}_{}',             #  0
                    'conv_3_64_1',                 #  1
                    'batchnorm',                   #  2
                    'leakyrelu',                   #  3
                    'conv_3_64_1',                 #  4
                    'batchnorm',                   #  5
                    'leakyrelu',                   #  6
                    'maxpl_2_0',                   #  7
                      'conv_3_128_1',              #  8
                      'batchnorm',                 #  9
                      'leakyrelu',                 # 10
                      'conv_3_128_1',              # 11
                      'batchnorm',                 # 12
                      'leakyrelu',                 # 13
                      'maxpl_2_0',                 # 14
                        'conv_3_256_1',            # 15
                        'batchnorm',               # 16
                        'leakyrelu',               # 17
                        'conv_3_256_1',            # 18
                        'batchnorm',               # 19
                        'leakyrelu',               # 20
                        'maxpl_2_0',               # 21
                          'conv_3_512_1',          # 22
                          'batchnorm',             # 23
                          'leakyrelu',             # 24
                          'conv_3_512_1',          # 25
                          'batchnorm',             # 26
                          'leakyrelu',             # 27
                          'maxpl_2_0',             # 28
                            'conv_3_1024_1',       # 29
                            'batchnorm',           # 30
                            'leakyrelu',           # 31
                            'conv_3_1024_1',       # 32
                            'batchnorm',           # 33
                            'leakyrelu',           # 34
                          'tconv_3_512_1_2',       # 35
                          'concat_35_27',          # 36
                          'conv_3_512_1',          # 37
                          'batchnorm',             # 38
                          'leakyrelu',             # 39
                          'conv_3_512_1',          # 40
                          'batchnorm',             # 41
                          'leakyrelu',             # 42
                        'tconv_3_256_1_2',         # 43
                        'concat_43_20',            # 44
                        'conv_3_256_1',            # 45
                        'batchnorm',               # 46
                        'leakyrelu',               # 47
                        'conv_3_256_1',            # 48
                        'batchnorm',               # 49
                        'leakyrelu',               # 50
                      'tconv_3_128_1_2',           # 51
                      'concat_51_13',              # 52
                      'conv_3_128_1',              # 53
                      'batchnorm',                 # 54
                      'leakyrelu',                 # 55
                      'conv_3_128_1',              # 56
                      'batchnorm',                 # 57
                      'leakyrelu',                 # 58
                    'tconv_3_64_1_2',              # 59
                    'concat_59_6',                 # 60
                    'conv_3_64_1',                 # 61
                    'batchnorm',                   # 62
                    'leakyrelu',                   # 63
                    'conv_3_64_1',                 # 64
                    'batchnorm',                   # 65
                    'leakyrelu',                   # 66
                  'conv_3_1_1']                    # 67

shape = (32, 32)
nx, ny = shape
nChannel = 3 # bc, a, f
nDim = 2
nEpoch = args.nEpoch
nPred = args.predict

# Insert values from args (input layer needs shape definition)
architecture[0] = architecture[0].format(shape[0], shape[1], nChannel, args.batchsize)
archStr   = '-'.join(architecture)
# Generate unreadable short architecture string (-> for shorter filenames)
tinyArch = []
for s in architecture:
  x = s.split('_')
  # Only keep first char of layer name + remove underscores
  tinyLayer = x[0][0] + ''.join(x[1:])
  tinyArch.append(tinyLayer)
tinyArch = ''.join(tinyArch)

usingTfData = args.tfdata
if usingTfData: print('Enabled tf.data optimization')

# Traing and validation split
nTrain = int(nSample * 0.9)
nValid = nSample - nTrain

batchsize = args.batchsize
print('{} samples in training, {} in validation'.format(nTrain, nValid))
fname = '../data/psn_{}_{}.h5'.format(shape[0], nSample)

# Create model
#with PM.strategy.scope():
psnNet = PM.PsnCnn(nCell=shape, operators=architecture, act=activation, last_act='linear', reg=args.reg, alpha=args.alpha)
psnNet.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr0))#, run_eagerly=1)

# Create unique name based on args describing model
psnPrefix = '{}'.format(name, eqId)
psnSuffix = 'a-{}_b-{}_a-{}_p-{}' \
            .format(args.alpha, batchsize, args.activation, args.patience)
modelName = '{}_{}'.format(psnPrefix, psnSuffix)

# Plot model
if args.visualize:
  psnNet.build(input_shape=(batchsize, shape[1], shape[0], nChannel))
  UT.plotModel(psnNet, name)

# Callbacks
timeHistCB   = UT.TimeHistory()
tboardCB     = KC.TensorBoard(log_dir=os.path.join(args.tboardDir, datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1, profile_batch='500, 520')
checkpointCB = KC.ModelCheckpoint(filepath='./' + modelName + '/checkpoint', monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
reduceLrCB   = KC.ReduceLROnPlateau(monitor='loss', min_delta=0.01, patience=args.patience, min_lr=args.lrmin)
csvLogCB     = keras.callbacks.CSVLogger(modelName + '.log', append=True)
psnCBs = [timeHistCB, tboardCB, checkpointCB, reduceLrCB, csvLogCB]

if args.restart:
  print('Restoring model {}'.format(modelName))
  ckpntName = modelName if args.checkpoint == None else args.checkpoint
  psnNet.load_weights(tf.train.latest_checkpoint(ckpntName))
  if args.restartLr != None:
    keras.backend.set_value(psnNet.optimizer.learning_rate, args.restartLr)

# Split samples into training, validation
if usingTfData:
  psnData   = PD.PoissonDataset(fname=fname, samples=nSample, nChannel=nChannel, shape=shape, mode=eqId)
  psnData   = psnData.shuffle(buffer_size=2048)
  trainData = psnData.take(count=nTrain)
  validData = psnData.skip(count=nTrain)

  trainDataset = trainData \
      .batch(batch_size=batchsize) \
      .cache() \
      .prefetch(buffer_size=tf.data.AUTOTUNE) \
      .repeat()
  validDataset = validData \
      .batch(batch_size=batchsize) \
      .cache() \
      .prefetch(buffer_size=tf.data.AUTOTUNE) \
      .repeat()
else:
  bcAF, p = PD.getBcAFP(fname, nSample, shape, eqId)
  # Split into training / validation sets
  xTrain    = bcAF[:nTrain, ...]
  yTrain    = p[:nTrain, ...]
  xValidate = bcAF[nTrain:, ...]
  yValidate = p[nTrain:, ...]

# Training
if args.train:
  startTrain = time.perf_counter()
  if usingTfData:
    psnNet.fit(
        trainDataset,
        initial_epoch=args.initTrain,
        epochs=nEpoch,
        steps_per_epoch=nTrain//batchsize,
        callbacks=psnCBs,
        validation_data=validDataset,
        validation_steps=nValid//batchsize,
        verbose=True)
  else:
    psnNet.fit(
        x=xTrain,
        y=yTrain,
        batch_size=batchsize,
        initial_epoch=args.initTrain,
        epochs=nEpoch,
        steps_per_epoch=nTrain//batchsize,
        callbacks=psnCBs,
        validation_data=(xValidate, yValidate),
        validation_steps=nValid//batchsize,
        verbose=True)
  endTrain = time.perf_counter()
  print("fit() execution time in secs:", endTrain - startTrain)

  # Evaluate callbacks
  avgTimeEpoch = sum(timeHistCB.times) / len(timeHistCB.times)
  print('{} samples ==> Average time per epoch: {}'.format(nSample, avgTimeEpoch))

  with open('{}_epochtimes_{}.txt'.format(psnPrefix, psnSuffix), 'w') as efile:
    for t in timeHistCB.times:
      efile.write(f'{t}\n')
    efile.write(f'Average: {avgTimeEpoch}\n')

# Predictions
if nPred > 0:
  if not os.path.exists(modelName):
    sys.exit('Model {} does not exist, exiting'.format(modelName))

  if not args.train:
    print('Restoring model {}'.format(modelName))
    psnNet.load_weights(tf.train.latest_checkpoint(modelName)).expect_partial()

  sP = 0 # Prediction start frame
  if usingTfData:
    phat = psnNet.predict(validDataset.take(count=nPred)) # Take from valid set for now
  else:
    phat = psnNet.predict(bcAF[sP:sP+nPred, ...])

  UT.plotPredictions(sP, nPred, bcAF, p, phat, name, eqId, archId)
  UT.plotLosses(modelName, 0, 500, name, eqId, archId)

