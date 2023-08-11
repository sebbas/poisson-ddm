import numpy as np
import h5py as h5
import argparse

from Poisson_solver import *
import Poisson_util as UT

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', default='../data/psn_32_20000.h5', help='data file')
parser.add_argument('-s', '--shape', type=int, default=32, help = 'size of sample')
parser.add_argument('-eq', '--equation', type=int, default=2, \
                    help = '0: Homogeneous Poisson, 1: Inhomogeneous Laplace, 2: Inhomogeneous Poisson')
parser.add_argument('-l', '--architecture', type=int, default=2, \
                    help='architecture id (0: simplified U-Net, 1: Full U-Net + dropout, 2: Full U-Net + batchnorm)')
parser.add_argument('-name', '--name', default='psnNet', help='model name prefix')
parser.add_argument('-n', '--nSample', type=int, default=20000, help = 'number of samples to solve')
args = parser.parse_args()

fname   = args.file
archId  = args.architecture
name    = args.name
eqId    = args.equation
nSample = args.nSample

# Read data from file
dFile = h5.File(fname, 'r')
#nSample  = dFile.attrs['nSample']
s        = dFile.attrs['shape']
length   = dFile.attrs['length']
aData    = np.array(dFile.get('a'))
fData    = np.array(dFile.get('f'))
ppData   = np.array(dFile.get('pp'))
plData   = np.array(dFile.get('pl'))
pBcData  = np.array(dFile.get('pBc'))
aBcData  = np.array(dFile.get('aBc'))
dFile.close()

psnSol = PoissonSolver2D(backend=2)
nx, ny = args.shape, args.shape
lx, ly = 1.0, 1.0
zeroBc = np.zeros(2*(nx+ny))
zeroF  = np.zeros((ny, nx))

# Solution arrays to write in to
plPyAmg = np.zeros((nSample, ny, nx))
ppPyAmg = np.zeros((nSample, ny, nx))
ppIPyAmg = np.zeros((nSample, ny, nx))

times = []

# Poisson, zero BC
timeTaken = 0
for s in range(nSample):
  ppPyAmg[s,...], solveTime = psnSol.solve((lx, ly), zeroBc, aData[s,...], fData[s,...], coefBc=aBcData[s,:])
  timeTaken += solveTime
times.append(timeTaken)
print('PyAMG Poisson (zero BC) execution time in secs: {}'.format(timeTaken))

# Laplace
timeTaken = 0
for s in range(nSample):
  plPyAmg[s,...], solveTime = psnSol.solve((lx, ly), pBcData[s,:], aData[s,...], zeroF, coefBc=aBcData[s,:])
  timeTaken += solveTime
times.append(timeTaken)
print('PyAMG Laplace execution time in secs: {}'.format(timeTaken))

# Inhomogeneous Poisson
timeTaken = 0
for s in range(nSample):
  ppIPyAmg[s,...], solveTime = psnSol.solve((lx, ly), pBcData[s,:], aData[s,...], fData[s,...], coefBc=aBcData[s,:])
  timeTaken += solveTime
times.append(timeTaken)
print('PyAMG Poisson (with BC) execution time in secs: {}'.format(timeTaken))

UT.writeTimes('pyamg', nSample, times)

# TODO: Compare and plot PyAMG solution with solution from data file


