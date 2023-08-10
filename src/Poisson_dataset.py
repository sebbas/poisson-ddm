import numpy as np
import h5py as h5
import tensorflow as tf
import itertools
from collections import defaultdict

def getBcAFP(fname, samples, shape, mode):
  dFile = h5.File(fname, 'r')
  nSample  = dFile.attrs['nSample']
  assert samples == nSample
  s        = dFile.attrs['shape']
  assert all(x == y for x, y in zip(shape, s))
  length   = dFile.attrs['length']
  aData    = np.array(dFile.get('a'))
  fData    = np.array(dFile.get('f'))
  ppData   = np.array(dFile.get('pp'))
  plData   = np.array(dFile.get('pl'))
  pBcData  = np.array(dFile.get('pBc'))
  dFile.close()

  assert nSample == aData.shape[0] and nSample == fData.shape[0] and ppData.shape[0]

  # Construct solution p
  if mode == 0: # Homogeneous Poisson
    p = np.expand_dims(ppData, axis=-1)
  elif mode == 1: # Inhomogeneous Laplace
    p = np.expand_dims(plData, axis=-1)
    fData *= 0.0 # For now, just set f to 0 and keep in channels
  elif mode == 2: # Inhomogeneous Poisson (i.e. Homogeneous Poisson + Inhomogeneous Laplace)
    pplData = ppData + plData
    p = np.expand_dims(pplData, axis=-1)

  # Construct solution, bc, a, and f arrays for Poisson / Laplace equation
  a  = np.expand_dims(aData, axis=-1)
  f  = np.expand_dims(fData, axis=-1)

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
  # Fill boundary with Dirichlet bc values (data from Laplace solve, ie f=0)
  if mode == 1 or mode == 2: # Inhomogeneous Poisson / Laplace
    nx, ny = shape
    bc[:,  0,  :, 0] = p[:,  0,  :, 0] # i- boundary
    bc[:,  :, -1, 0] = p[:,  :, -1, 0] # j+ boundary
    bc[:, -1,  :, 0] = p[:, -1,  :, 0] # i+ boundary
    bc[:,  :,  0, 0] = p[:,  :,  0, 0] # j- boundary
    # Average bc values to counterbalance overlap in corner cells
    if 0:
      bcCnt = np.ones_like(bc)
      corners = [[0,0], [nx-1,0], [0,ny-1], [nx-1,ny-1]]
      for x,y in corners:
        bcCnt[:,x,y,0] += 1
      bc /= bcCnt

  # Combine bc, a, f along channel dim
  bcAF = np.concatenate((bc, a, f), axis=-1)

  return bcAF, p


class PoissonDataset(tf.data.Dataset):

  ID = itertools.count() # Number of datasets generated
  NUM_EPOCHS = defaultdict(itertools.count) # Number of epochs done, per dataset

  def _generator(instance_idx, fname, samples, shape, mode):
    epoch_idx = next(PoissonDataset.NUM_EPOCHS[instance_idx])

    # Combine bc, a, f along channel dim
    bcAF, p = getBcAFP(fname, samples, shape, mode)

    for i in range(samples):
      yield bcAF[i, ...], p[i]


  def __new__(cls, fname, samples, nChannel=3, shape=(32,32), mode=2):
    tf.print('PoissonDataset {}'.format(PoissonDataset.ID))

    return tf.data.Dataset.from_generator(
      cls._generator,
      output_signature=(tf.TensorSpec(shape=(shape[0], shape[1], nChannel),
                                      dtype=tf.float32, name="bcAF"),
                        tf.TensorSpec(shape=(shape[0], shape[1], 1),
                                      dtype=tf.float32, name="p")),
      args=(next(cls.ID), fname, samples, shape, mode)
    )


