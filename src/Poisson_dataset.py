import numpy as np
import h5py as h5
import tensorflow as tf
import itertools
from collections import defaultdict

class PoissonDataset(tf.data.Dataset):

  ID = itertools.count() # Number of datasets generated
  NUM_EPOCHS = defaultdict(itertools.count) # Number of epochs done, per dataset

  def _generator(instance_idx, fname):
    epoch_idx = next(PoissonDataset.NUM_EPOCHS[instance_idx])

    dFile = h5.File(fname, 'r')
    nSample  = dFile.attrs['nSample']
    shape    = dFile.attrs['shape']
    length   = dFile.attrs['length']
    aData    = np.array(dFile.get('a'))
    fData    = np.array(dFile.get('f'))
    ppData   = np.array(dFile.get('pp'))
    dFile.close()

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

    for i in range(nSample):
      yield bcAF[i, ...], pp[i]


  def __new__(cls, fname, nChannel=3, shape=(32,32)):
    tf.print('PoissonDataset {}'.format(PoissonDataset.ID))

    return tf.data.Dataset.from_generator(
      cls._generator,
      output_signature=(tf.TensorSpec(shape=(shape[0], shape[1], nChannel),
                                      dtype=tf.float32, name="bcAF"),
                        tf.TensorSpec(shape=(shape[0], shape[1], 1),
                                      dtype=tf.float32, name="pp")),
      args=(next(cls.ID), fname)
    )


