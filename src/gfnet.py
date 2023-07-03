import sys, os
sys.path.append('../src')
import h5py as h5
import numpy as np
from tensorflow import keras
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from PIL import ImageFont

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
parser.add_argument('-eq', '--modeEquation', type=int, default=0, \
                    help = "0: Poisson, 1: Laplace")
parser.add_argument('-t', '--train', type=int, default=1, \
                    help = "enable / disable training")
parser.add_argument('-p', '--predict', type=int, default=0, \
                    help = "enable / disable prediction by specifying number of predictions")
parser.add_argument('-ps', '--predictionSource', type=int, default=2, \
                    help = "0: predict on training data, 1: predict on validation data, 2: predict on test data")

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

'''
# Model architecture (simplified U-Net)
parser.add_argument('-l', '--architecture', type=str, nargs='*', \
                    default=['input_{}_{}_{}_{}', \
                             'conv_3_64_0', \
                               'conv_3_128_0', \
                                 'maxpl_2_0', \
                                   'conv_3_256_0', \
                                     'maxpl_2_0', \
                                       'conv_3_512_0', \
                                         'conv_3_1024_0', \
                                         'tconv_3_512_0_1',\
                                       'tconv_3_256_0_1',\
                                     'bilinup_2', \
                                   'tconv_3_128_0_1',\
                                 'bilinup_2', \
                               'tconv_3_64_0_1',\
                             'tconv_3_1_0_1'],\
                    help='type and arguments of each layer')
'''

# Model architecture (full U-Net)
parser.add_argument('-l', '--architecture', type=str, nargs='*', \
                    default=['input_{}_{}_{}_{}',         #  0
                               'conv_3_64_1',             #  1
                               'conv_3_64_1',             #  2
                               'maxpl_2_0',               #  3
                                 'conv_3_128_1',          #  4
                                 'conv_3_128_1',          #  5
                                 'maxpl_2_0',             #  6
                                   'conv_3_256_1',        #  7
                                   'conv_3_256_1',        #  8
                                   'maxpl_2_0',           #  9
                                     'conv_3_512_1',      # 10
                                     'conv_3_512_1',      # 11
                                     'maxpl_2_0',         # 12
                                       'conv_3_1024_1',   # 13
                                       'conv_3_1024_1',   # 14
                                     'tconv_3_512_1_2',   # 15
                                     'concat_15_11',      # 16
                                     'dropout_3',         # 17
                                     'conv_3_512_1',      # 18
                                     'conv_3_512_1',      # 19
                                   'tconv_3_256_1_2',     # 20
                                   'concat_20_8',         # 21
                                   'dropout_3',           # 22
                                   'conv_3_256_1',        # 23
                                   'conv_3_256_1',        # 24
                                 'tconv_3_128_1_2',       # 25
                                 'concat_25_5',           # 26
                                 'dropout_3',             # 27
                                 'conv_3_128_1',          # 28
                                 'conv_3_128_1',          # 29
                               'tconv_3_64_1_2',          # 30
                               'concat_30_2',             # 31
                               'dropout_3',               # 32
                               'conv_3_64_1',             # 33
                               'conv_3_64_1',             # 34
                             'conv_3_1_1'],               # 35
                    help='type and arguments of each layer')

args = parser.parse_args()

nSample = args.nSample
shape = (32, 32)
nx, ny = shape
nChannel = 3 # bc, a, f
nDim = 2
nEpoch = args.nEpoch
nPred = args.predict

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
nTrain = int(nSample * 0.8)
nValid = nSample - nTrain

batchSize = args.batchsize
print('{} samples in training, {} in validation'.format(nTrain, nValid))
fname = '../data/psn_{}_{}.h5'.format(shape[0], nSample)

# Create model
#with PM.strategy.scope():
psnNet = PM.PsnCnn(nCell=shape, operators=args.architecture, \
                   act=args.activation, last_act='linear', \
                   reg=args.reg, alpha=args.alpha)
psnNet.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr0))

# Create unique name based on args describing model
psnPrefix = '{}{}'.format(args.name, args.modeEquation)
psnSuffix = 'a-{}_e-{}_n-{}_b-{}_a-{}_p-{}' \
            .format(args.alpha, args.nEpoch, nSample, \
            batchSize, args.activation, args.patience)
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
timeHistCB   = UT.TimeHistory()
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

if args.restart:
  print('Restoring model {}'.format(modelName))
  ckpntName = modelName if args.checkpoint == None else args.checkpoint
  psnNet.load_weights(tf.train.latest_checkpoint(ckpntName))
  if args.restartLr != None:
    keras.backend.set_value(psnNet.optimizer.learning_rate, args.restartLr)

# Split samples into training, validation
if usingTfData:
  psnData   = PD.PoissonDataset(fname=fname, samples=nSample, nChannel=nChannel, shape=shape, mode=args.modeEquation)
  psnData   = psnData.shuffle(buffer_size=2048)
  trainData = psnData.take(count=nTrain)
  validData = psnData.skip(count=nTrain)

  trainDataset = trainData \
      .batch(batch_size=batchSize) \
      .cache() \
      .prefetch(buffer_size=tf.data.AUTOTUNE) \
      .repeat()
  validDataset = validData \
      .batch(batch_size=batchSize) \
      .cache() \
      .prefetch(buffer_size=tf.data.AUTOTUNE) \
      .repeat()
else:
  bcAF, p = PD.getBcAFP(fname, nSample, shape, args.modeEquation)
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
        epochs=nEpoch,
        steps_per_epoch=nTrain//batchSize,
        callbacks=psnCBs,
        validation_data=(xValidate, yValidate),
        validation_steps=nValid//batchSize,
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

  if args.predictionSource == 0:  # Predict on training data
    sP = 0
  elif args.predictionSource == 1: # Predict on validation data
    sP = nTrain
  elif args.predictionSource == 2: # Predict on test data
    sP = nTrain + nValid
  eP = sP + nPred

  if usingTfData:
    phat = psnNet.predict(validDataset.take(count=nPred)) # Take from valid set for now
  else:
    phat = psnNet.predict(bcAF[sP:eP, ...])

  # Get statistics per frame
  maeLst, rmseLst, mapeLst = [], [], []
  for i in range(nPred):
    maeLst.append(UT.mae(p[i,:,:,0], phat[i,:,:,0]))
    rmseLst.append(UT.rmse(p[i,:,:,0], phat[i,:,:,0]))
    mapeLst.append(UT.mape(p[i,:,:,0], phat[i,:,:,0]))

  # Plots
  fig = plt.figure(figsize=(21, 1+2*nPred), dpi=120, constrained_layout=True)
  fig.suptitle('PSN {}: bc, a, f, p, phat'.format(args.modeEquation), fontsize=16)

  # Min, max values, needed for colobar range
  minBc, maxBc     = np.min(bcAF[i,:,:,0]), np.max(bcAF[i,:,:,0])
  minA, maxA       = np.min(bcAF[i,:,:,1]), np.max(bcAF[i,:,:,1])
  minF, maxF       = np.min(bcAF[i,:,:,2]), np.max(bcAF[i,:,:,2])
  minP, maxP       = np.min(p[i,:,:,0]), np.max(p[i,:,:,0])
  minPHat, maxPHat = np.min(phat[i, ...]), np.max(phat[i, ...])

  nCols = 7
  for i in range(nPred):
    cnt = i*nCols

    ax = fig.add_subplot(nPred, nCols, cnt+1)
    plt.ylabel("Frame {}".format(sP+i))
    plt.title('bc')
    plt.imshow(bcAF[i,:,:,0], vmin=minBc, vmax=maxBc, origin='lower')
    plt.colorbar()

    ax = fig.add_subplot(nPred, nCols, cnt+2)
    plt.title('a')
    plt.imshow(bcAF[i,:,:,1], vmin=minA, vmax=maxA, origin='lower')
    plt.colorbar()

    ax = fig.add_subplot(nPred, nCols, cnt+3)
    plt.title('f')
    plt.imshow(bcAF[i,:,:,2], vmin=minF, vmax=maxF, origin='lower')
    plt.colorbar()

    ax = fig.add_subplot(nPred, nCols, cnt+4)
    plt.title('p')
    plt.imshow(p[i,:,:,0], vmin=minP, vmax=maxP, origin='lower')
    plt.colorbar()

    ax = fig.add_subplot(nPred, nCols, cnt+5)
    plt.title('phat')
    plt.imshow(phat[i, ...], vmin=minP, vmax=maxP, origin='lower')
    plt.colorbar()

    ax = fig.add_subplot(nPred, nCols, cnt+6)
    plt.title('phat (own colorbar)')
    plt.imshow(phat[i, ...], vmin=minPHat, vmax=maxPHat, origin='lower')
    plt.colorbar()

    ax = fig.add_subplot(nPred, nCols, cnt+7)
    ax.axis('off')
    plt.title('Stats')
    plt.text(0.2, 0.3, 'MAE: %.4f' % maeLst[i])
    plt.text(0.2, 0.5, 'RMSE: %.4f' % rmseLst[i])
    plt.text(0.2, 0.7, 'MAPE: %.4f' % mapeLst[i])

  plt.savefig('{}{}_ps-{}_bc_a_f_p_phat.png'.format(args.name, args.modeEquation, args.predictionSource), bbox_inches='tight')

