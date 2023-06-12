import tensorflow as tf
import argparse
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

import Poisson_model as PsnModel

parser = argparse.ArgumentParser()

# Model architecture
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

shape = (32, 32)
nChannel = 3
nDim = 2
batchsize = 800

args.architecture[0] = args.architecture[0].format(shape[0], shape[1], nChannel, batchsize)

# Adjust here: model name must match dir of model used for predictions
modelName = 'psnNet_a-2.0_e-800_n-40000_b-200_a-relu_p-200_t-0_l-i32323200c364c3128m20c3256m20c3512c31024t3512t3256b2t3128b2t364t31'

psnNet = PsnModel.PsnCnn(operators=args.architecture)
psnNet.load_weights(tf.train.latest_checkpoint(modelName)).expect_partial()

# Define how many samples to load from prediction dataset
predSample = 1000
fname = '../data/psn_{}_{}.h5'.format(shape[0], predSample)

# Load external data
dFile = h5.File(fname, 'r')
n        = dFile.attrs['nSample']
assert predSample == n
s        = dFile.attrs['shape']
assert all(x == y for x, y in zip(shape, s))
length   = dFile.attrs['length']
aData    = np.array(dFile.get('a'))
fData    = np.array(dFile.get('f'))
ppData   = np.array(dFile.get('pp'))
dFile.close()
print('Loaded {} samples from prediction dataset'.format(n))

assert predSample == aData.shape[0] and predSample == fData.shape[0] and predSample == ppData.shape[0]

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
# Repeat array 'predSample' times in 1st array dim
bc          = np.repeat(onesExpand, predSample, axis=0)

# Construct a, f, pp arrays - all need extra dim at end for channel
a  = np.expand_dims(aData, axis=-1)
f  = np.expand_dims(fData, axis=-1)
pp = np.expand_dims(ppData, axis=-1)

# Combine bc, a, f along channel dim
bcAF = np.concatenate((bc, a, f), axis=-1)

# Define prediction range (from frame, to frame)
rPred = (0, 1000)
nPred = rPred[1] - rPred[0]
yPred = []
yTrue = []
yError = []

relErrLst = []
maeLst = []
resLst = []

def compute_relative_error(yTrue, yPred, eps):
  nCell, rerr, res = 0, 0, 0
  for (i,j), value in np.ndenumerate(yTrue):
    if np.abs(yTrue[i,j]) > eps:
      nCell += 1
      residual = np.abs(yTrue[i,j] - yPred[i,j])
      rerr  += np.abs(residual / yTrue[i,j])
      res   += residual
  return rerr / nCell, res / nCell

# Make predictions
eps = 1e-2
for cnt, i in enumerate(range(rPred[0], rPred[1])):
  if (i % 100 == 0) and i > 0: print('Finished predicting {} samples'.format(i))
  sample = bcAF[i, ...]
  sampleExp = np.expand_dims(sample, axis=0)
  prediction = psnNet.predict(sampleExp)
  prediction = tf.squeeze(prediction)
  yPred.append(prediction)
  label = ppData[i, ...]
  yTrue.append(label)

  relErr, residuals = compute_relative_error(label, prediction, eps)
  relErrLst.append(relErr)
  resLst.append(residuals)

  mae = tf.keras.metrics.mean_absolute_error(label, prediction)
  maeLst.append(mae)

mRes = np.mean(resLst)
MRE = np.mean(relErrLst)
mMAE = np.mean(maeLst)
print(len(relErrLst))
print('MRE: {}, mean MAE: {}, mean Residual {}'.format(MRE, mMAE, mRes))

