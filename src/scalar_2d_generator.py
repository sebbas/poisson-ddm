import numpy as np
import scipy.stats.qmc as qmc
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt
import math
import copy

class ScalarGenerator2D:
  def __init__(self, size=(1.0, 1.0), nCell=(32, 32), nKnot=(4, 4)):
    assert len(size) == 2 and len(nCell) == 2 and len(nKnot) == 2
    assert size[0]/nCell[0] == size[1]/nCell[1]
    # input arguments
    self.nCell  = nCell
    self.size   = size
    self.nKnot  = nKnot

    # module constants
    self.lenScale = 1.0

    # knots info for GP
    xKnot = np.linspace(0.0, self.size[0], self.nKnot[0])
    yKnot = np.linspace(0.0, self.size[1], self.nKnot[1])
    xKnot, yKnot = np.meshgrid(xKnot, yKnot)
    # list of knot's (x, y)
    xyKnot = np.stack([xKnot.flatten(), yKnot.flatten()], axis=-1)
    # squared distance of each knot pair
    knotDistMat = distance.cdist(xyKnot, xyKnot, 'sqeuclidean')
    # knot's colvariance matrix
    knotCovMat    = np.exp(-knotDistMat / self.lenScale)
    self.knotCovMatInv = np.linalg.inv(knotCovMat)

    # setup the output scalar's coordinates and matrix for GP
    h      = self.size[0] / self.nCell[0]
    x      = [(i+0.5)*h for i in range(self.nCell[0])]
    y      = [(i+0.5)*h for i in range(self.nCell[1])]
    x, y   = np.meshgrid(x, y)
    # list of coordinates
    xy     = np.stack([x.flatten(), y.flatten()], axis=-1)
    # colvariance matrix for grid cells and knots
    self.covMat = distance.cdist(xy, xyKnot, 'sqeuclidean')
    self.covMat = np.exp(-self.covMat / self.lenScale)

    # values at the boundary cells' face centers
    xyBc   = np.zeros((2*np.sum(self.nCell), 2))
    nx, ny = self.nCell[0], self.nCell[1]
    # i- boundary
    xyBc[:nx, 0] = np.array([(j+0.5)*h for j in range(nx)])
    # j+ boundary
    xyBc[nx:nx+ny, 0] = size[0]
    xyBc[nx:nx+ny, 1] = np.array([(i+0.5)*h for i in range(ny)])
    # i+ boundary
    xyBc[nx+ny:2*nx+ny, 0] = np.flip(np.array([(j+0.5)*h for j in range(nx)]))
    xyBc[nx+ny:2*nx+ny, 1] = self.size[1]
    # j- boundary
    xyBc[2*nx+ny:, 1] = np.flip(np.array([(i+0.5)*h for i in range(ny)]))
    # colvariance matrix for boundary face centers and knots
    self.covMatBc = distance.cdist(xyBc, xyKnot, 'sqeuclidean')
    self.covMatBc = np.exp(-self.covMatBc / self.lenScale)



  def generate_scalar2d(self, nSample, valMin=0.0, valMax=1.0, outputBc=False,
                        strictMin=False, periodic=0):
    # create sobol sequence
    pow      = int(np.log2(self.nKnot[0]*self.nKnot[1]*nSample)) + 1
    sobolSeq = qmc.Sobol(d=1).random_base2(m=pow)
    sobolSeq = sobolSeq * (valMax - valMin) + valMin
    np.random.shuffle(sobolSeq)

    # allocate the scalars
    samples = np.zeros((nSample, self.nCell[1], self.nCell[0]))
    if outputBc:
      bcs = np.zeros((nSample, 2*np.sum(self.nCell)))

    # generate the scalar with GP
    # R.B.Gramacy P148, Eqn 5.2
    s, e = 0, 0
    for i in range(nSample):
      if periodic:
        period = 30
        A = 2*math.pi/period
        if i%period == 0:
          s, e = e, e + self.nKnot[0] * self.nKnot[1]
        knots = copy.deepcopy(sobolSeq[s:e])
        knots *= (math.sin(i/A-math.pi/2)+1) / 2
      else:
        s, e  = e, e + self.nKnot[0] * self.nKnot[1]
        knots = sobolSeq[s:e]
      # interpolate one scalar with GP
      sca            = np.matmul(self.covMat, np.dot(self.knotCovMatInv, knots))
      samples[i,...] = np.reshape(sca, self.nCell)
      if outputBc:
        bc           = np.matmul(self.covMatBc, np.dot(self.knotCovMatInv, knots))
        bcs[i,:]     = np.squeeze(bc)

    if strictMin:
      scalarMin = min(np.min(bcs), np.min(samples))
      samples   = samples - scalarMin + valMin
      bcs       = bcs     - scalarMin + valMin

    if outputBc:
      return samples, bcs

    return samples


class PeriodicScalarGenerator1D:
  def __init__(self, size=1.0, nCell=32, nKnot=4):
    self.size  = size
    self.nCell = nCell
    self.nKnot = nKnot

    h = size / nCell

    # module constants
    self.lenScale = 1.0

    # knots info for GP
    self.xKnot    = np.linspace(0.0, size, nKnot, endpoint=False)
    # extend xknot to left adn right for periodicity
    xKnotExtRight = self.xKnot + self.size
    xKnotExtLeft  = self.xKnot - self.size
    # combine xKnot
    self.xKnot    = np.concatenate([xKnotExtLeft, self.xKnot, xKnotExtRight], axis=-1)
    self.xKnot    = np.expand_dims(self.xKnot, axis=1)
    # print(self.xKnot)

    # squared distance of each knot pair
    knotDistMat = distance.cdist(self.xKnot, self.xKnot, 'sqeuclidean')
    # knot's colvariance matrix
    knotCovMat         = np.exp(-knotDistMat / self.lenScale)
    self.knotCovMatInv = np.linalg.inv(knotCovMat)

    # setup the output scalar's coordinates and matrix for GP
    h = self.size / self.nCell
    x = [(i+0.5)*h for i in range(-self.nCell, 2*self.nCell)]
    x = np.expand_dims(x, axis=1)
    # colvariance matrix for grid cells and knots
    self.covMat = distance.cdist(x, self.xKnot, 'sqeuclidean')
    self.covMat = np.exp(-self.covMat / self.lenScale)


  def generate_periodic_scalar(self, nSample, valMin=0.0, valMax=1.0, periodic=0):
    # create sobol sequence
    pow      = int(np.log2(self.nKnot*nSample)) + 1
    sobolSeq = qmc.Sobol(d=1).random_base2(m=pow)
    sobolSeq = sobolSeq * (valMax - valMin) + valMin
    np.random.shuffle(sobolSeq)

    # allocate the scalars
    samples = np.zeros((nSample, self.nCell))

    # generate the scalar with GP
    s, e  = 0, 0
    for i in range(nSample):
      if periodic:
        period = 30
        A = 2*math.pi/period
        if i%period == 0:
          s, e  = e, e + self.nKnot
        knots = copy.deepcopy(sobolSeq[s:e])
        knots *= (math.sin(i/A-math.pi/2)+1) / 2
      else:
        s, e  = e, e + self.nKnot
        knots = sobolSeq[s:e]
      knots         = np.concatenate([knots, knots, knots], axis=0)
      # interpolate one scalar with GP
      sca           = np.matmul(self.covMat, np.dot(self.knotCovMatInv, knots))
      samples[i,:]  = np.squeeze(sca[self.nCell : 2*self.nCell])

    return samples
