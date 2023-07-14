import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL
import tensorflow.keras.regularizers as KR
import numpy as np

# -------------------- name of operators --------------------
# input_w_h_c_b - input layer,            shape (w, h), channels c, batchsize b
# conv_m_n_p    - convolution,            kernel size m, n filters, padding (0 == valid, 1 == same)
# tconv_m_n_p_s - transposed convolution, kernel size m, n filters, padding (0 == valid, 1 == same), s strides
# avgpl_m       - average pooling,        size (m, m)
# maxpl_m_p     - max pooling,            size (m, m), padding (0 == valid, 1 == same)
# bilinup_m     - up sampling (bilinear), size (m, m)
# nearup_m      - up sampling (nearest),  size (m, m)
# batchnorm     - batch normalization
# dropout_r     - dropout, r rate (r == 3 ~> rate=0.3)
# -----------------------------------------------------------

strategy = tf.distribute.MirroredStrategy()

class PsnCnn(keras.Model):
  def __init__(self, nCell=(32,32), nDim=2, operators=[], reg=None, \
               data_format='channels_last', act='relu', alpha=0.01, \
               last_act='linear', save_grad_stat=False, **kwargs):
    super(PsnCnn, self).__init__(**kwargs)
    # Check inputs
    assert len(operators) > 0
    assert nDim == 2 or nDim == 3
    if reg is not None: assert len(reg) == len(operators)
    # Save inputs
    self.nCell        = nCell
    self.nDim         = nDim
    self.alpha        = alpha
    self.act          = act
    self.lastAct      = last_act
    self.operators    = operators
    self.reg          = np.zeros(len(operators)) if reg==None else np.array(reg)
    self.saveGradStat = save_grad_stat
    assert data_format in ['channels_last', 'channels_first']
    self.dataForm     = data_format
    self.h            = 1 / max(nCell) # dx

    tf.print('PsnCNN for grid size {}x{} and with architecture:'.format(nCell[0], nCell[1]))

    # Input parameters will be filled with info from input layer (1st in arch string)
    self.inputLayer = None
    self.inputShape = None
    self.batchSize = None

    # Setup layers
    self.mlp = []
    for i, op in enumerate(self.operators):

      layerName, layerArgs = self._getLayerName(op)

      # Use custom activation function in last layer
      isLastLayer = (i == len(operators)-1)
      activation = self.lastAct if isLastLayer else self.act

      ## 1st layer must be an input layer
      if layerName == 'input':
        self.batchSize = layerArgs[3]
        hw = (layerArgs[1], layerArgs[0])
        c = (layerArgs[2],)
        self.inputShape = hw + c if data_format == 'channels_last' else c + hw
        self.inputLayer = ( KL.InputLayer( input_shape=self.inputShape,
                                           batch_size=self.batchSize) )
        tf.print('==> {} layer with input shape {}, batch size {} and data format \'{}\''\
                  .format(layerName, self.inputShape, self.batchSize, self.dataForm))
        continue

      assert i > 0, 'Invalid architecture, missing input layer'

      ## Convolution
      if layerName == 'conv':
        padding = 'valid' if layerArgs[2] == 0 else 'same'
        if self.nDim == 2:
          self.mlp.append( KL.Conv2D( filters=layerArgs[1],
                                      kernel_size=layerArgs[0],
                                      activation=activation,
                                      padding=padding,
                                      data_format=self.dataForm) )
        elif self.nDim == 3:
          self.mlp.append( KL.Conv3D( filters=layerArgs[1],
                                      kernel_size=layerArgs[0],
                                      padding=padding,
                                      activation=activation,
                                      data_format=self.dataForm) )
      ## Deconvolution
      elif layerName == 'tconv':
        padding = 'valid' if layerArgs[2] == 0 else 'same'
        if self.nDim == 2:
          self.mlp.append( KL.Conv2DTranspose( filters=layerArgs[1],
                                               kernel_size=layerArgs[0],
                                               padding=padding,
                                               strides=layerArgs[3],
                                               activation=activation,
                                               data_format=self.dataForm) )
        elif self.nDim == 3:
          self.mlp.append( KL.Conv3DTranspose( filters=layerArgs[1],
                                               kernel_size=layerArgs[0],
                                               padding=padding,
                                               strides=layerArgs[3],
                                               activation=activation,
                                               data_format=self.dataForm) )
      ## Pooling and upsampling
      elif layerName == 'avgpl':
        if self.nDim == 2:
          self.mlp.append( KL.AveragePooling2D( pool_size=layerArgs[0],
                                                data_format=self.dataForm) )
        elif self.nDim == 3:
          self.mlp.append( KL.AveragePooling3D( pool_size=layerArgs[0],
                                                data_format=self.dataForm) )
      elif layerName == 'maxpl':
        padding = 'valid' if layerArgs[1] == 0 else 'same'
        if self.nDim == 2:
          self.mlp.append( KL.MaxPooling2D( pool_size=layerArgs[0],
                                            data_format=self.dataForm,
                                            padding=padding) )
        elif self.nDim ==3:
          self.mlp.append( KL.MaxPooling3D( pool_size=layerArgs[0],
                                            data_format=self.dataForm,
                                            padding=padding) )
      elif layerName == 'bilinup':
        if self.nDim == 2:
          self.mlp.append( KL.UpSampling2D( size=layerArgs[0],
                                            interpolation='bilinear',
                                            data_format=self.dataForm) )
        if self.nDim == 3:
          self.mlp.append( KL.UpSampling3D( size=layerArgs[0],
                                            interpolation='bilinear',
                                            data_format=self.dataForm) )
      elif layerName == 'nearup':
        if self.nDim == 2:
          self.mlp.append( KL.UpSampling2D( size=layerArgs[0],
                                            interpolation='nearest',
                                            data_format=self.dataForm) )
        if self.nDim == 3:
          self.mlp.append( KL.UpSampling3D( size=layerArgs[0],
                                            interpolation='nearest',
                                            data_format=self.dataForm) )
      elif layerName == 'batchnorm':
        self.mlp.append( KL.BatchNormalization() )

      elif layerName == 'leakyrelu':
        self.mlp.append( KL.LeakyReLU() )

      elif layerName == 'relu':
        self.mlp.append( KL.ReLU() )

      elif layerName == 'concat':
        self.mlp.append( KL.Concatenate() )

      elif layerName == 'dropout':
        rate = layerArgs[0]
        assert rate >= 0 and rate <= 10
        rate *= 1e-1 # convert int input to [0,1] float
        self.mlp.append( KL.Dropout( rate=rate) )

      tf.print('==> {} layer with args {}'.format(layerName, layerArgs[0:]))

    # Dicts for metrics and statistics
    self.trainMetrics = {}
    self.validMetrics = {}
    # Construct metric names and add to train/valid dicts
    names = ['loss', 'data', 'pde']
    for key in names:
      self.trainMetrics[key] = keras.metrics.Mean(name='train_'+key)
      self.validMetrics[key] = keras.metrics.Mean(name='valid_'+key)
    self.trainMetrics['mae']  = keras.metrics.MeanAbsoluteError(name='train_mae')
    self.validMetrics['mae']  = keras.metrics.MeanAbsoluteError(name='valid_mae')

    ## Add metrics for layers' weights, if save_grad_stat is required
    ## i even for weights, odd for bias
    if self.saveGradStat:
      for i, op in enumerate(self.operators):
        if op.trainable: 
          names = ['dat_'+repr(i)+'w_avg', 'dat_'+repr(i)+'w_std',\
                   'dat_'+repr(i)+'b_avg', 'dat_'+repr(i)+'b_std',\
                   'pde_'+repr(i)+'w_avg', 'pde_'+repr(i)+'w_std',\
                   'pde_'+repr(i)+'b_avg', 'pde_'+repr(i)+'b_std']
          for name in names:
            self.trainMetrics[name] = keras.metrics.Mean(name='train '+name)

    # Dicts to save training and validation statistics
    self.trainStat = {}
    self.validStat = {}


  def _getLayerName(self, operator):
      strs = operator.split('_')
      layerName = strs[0]
      layerArgs = [int(s) for s in strs[1:]] # Convert layer args from string to int
      return layerName, layerArgs


  def call(self, inputs, training=False, withInputLayer=True):
    layers = [self.inputLayer(inputs)] if withInputLayer else [inputs]

    for i, (curLayer, op) in enumerate(zip(self.mlp, self.operators[1:])):
      layerName, layerArgs = self._getLayerName(op)

      if layerName == 'concat':
        p1, p2 = layerArgs[0], layerArgs[1] # indices in operator list of layers to concat
        nextLayer = curLayer([layers[p1], layers[p2]])
      else:
        prevLayer = layers[-1]
        if layerName == 'batchnorm':
          nextLayer = curLayer(prevLayer, training=training)
        else:
          nextLayer = curLayer(prevLayer)

      layers.append(nextLayer)
    return nextLayer


  def _compute_data_loss(self, true, pred):
    return keras.losses.mean_squared_error(true, pred)


  def _compute_pde_loss(self, bcAF, pPred):
    # Extract channels
    p = pPred[:,:,:,0]
    a = bcAF[:,:,:,1]
    f = bcAF[:,:,:,2]

    # Extract inner matrices (1 cell boundary)
    p_ij = p[:, 1:-1, 1:-1]
    a_ij = a[:, 1:-1, 1:-1]
    f_ij = f[:, 1:-1, 1:-1]
    # Extract inner matrices (1 cell shift plus/minus 1 in i,j)
    a_iP, a_jP = a[:, 2:  , 1:-1], a[:, 1:-1, 2:  ]
    a_iN, a_jN = a[:, 0:-2, 1:-1], a[:, 1:-1, 0:-2]
    p_iP, p_jP = p[:, 2:  , 1:-1], p[:, 1:-1, 2:  ]
    p_iN, p_jN = p[:, 0:-2, 1:-1], p[:, 1:-1, 0:-2]
    # Arithmetic average at cell interface:
    # 0 = [  [a_(i+1,j) +  a_(i,j)] * [ p_(i+1,j) - p_(i,j)]
    #      - [a_(i-1,j) +  a_(i,j)] * [-p_(i-1,j) + p_(i,j)]
    #      + [a_(i,j+1) +  a_(i,j)] * [ p_(i,j+1) - p_(i,j)]
    #      - [a_(i,j-1) +  a_(i,j)] * [-p_(i,j-1) + p_(i,j)] ] ^ 2
    pdePred = tf.square(  (a_iP + a_ij) * ( p_iP - p_ij)
                        - (a_iN + a_ij) * (-p_iN + p_ij)
                        + (a_jP + a_ij) * ( p_jP - p_ij)
                        - (a_jN + a_ij) * (-p_jN + p_ij)
                        - 2.0 * f_ij * self.h * self.h   )
    pdeTrue = 0.0

    # Compute the pde loss
    pdePred = tf.expand_dims(pdePred, axis=-1)
    pdeLoss = keras.losses.mean_squared_error(pdeTrue, pdePred)

    # Add padding to pdeLoss, ie match size of dataLoss
    paddings = tf.constant([[0, 0], [1, 1], [1, 1]])
    return tf.pad(pdeLoss, paddings)


  def train_step(self, data):
    bcAF = data[0]
    p = data[1]

    with tf.GradientTape() as tape:
      pPred = self(bcAF, training=True)
      # Compute data loss
      dataLoss = self._compute_data_loss(p, pPred)
      # Compute pde loss
      pdeLoss = self._compute_pde_loss(bcAF, pPred)
      # Compute total loss
      loss = dataLoss + self.alpha * pdeLoss

    # Compute gradients
    gradients = tape.gradient(loss, self.trainable_variables)
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    # Update metrics
    self.trainMetrics['loss'].update_state(loss)
    self.trainMetrics['data'].update_state(dataLoss)
    self.trainMetrics['pde'].update_state(pdeLoss)
    self.trainMetrics['mae'].update_state(p, pPred)

    # Return metrics in statistics dictionary
    for key in self.trainMetrics:
      self.trainStat[key] = self.trainMetrics[key].result()
    return self.trainStat


  def test_step(self, data):
    bcAF = data[0]
    p = data[1]

    pPred = self(bcAF, training=False)
    # Compute data loss
    dataLoss = self._compute_data_loss(p, pPred)
    # Compute pde loss
    pdeLoss = self._compute_pde_loss(bcAF, pPred)
    # Compute total loss
    loss = dataLoss + self.alpha * pdeLoss

    # Update metrics
    self.validMetrics['loss'].update_state(loss)
    self.validMetrics['data'].update_state(dataLoss)
    self.validMetrics['pde'].update_state(pdeLoss)
    self.validMetrics['mae'].update_state(p, pPred)

    # Return metrics in statistics dictionary
    for key in self.trainMetrics:
      self.validStat[key] = self.validMetrics[key].result()
    return self.validStat


  def reset_metrics(self):
    for key in self.trainMetrics:
      self.trainMetrics[key].reset_states()
    for key in self.validMetrics:
      self.validMetrics[key].reset_states()


  @property
  def metrics(self):
    return [self.trainMetrics[key] for key in self.trainMetrics] \
         + [self.validMetrics[key] for key in self.validMetrics]


  def build_graph(self):
    input = keras.Input(shape=self.inputShape, batch_size=self.batchSize)
    inLayer = self.inputLayer(input)
    outLayers = self.call(inLayer, training=False, withInputLayer=False)
    return keras.Model(inputs=inLayer, outputs=outLayers)

