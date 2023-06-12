import numpy as np
import h5py as h5
from scalar_2d_generator import PeriodicScalarGenerator1D, ScalarGenerator2D
from Poisson_solver import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filePrefix', type=str, default="psn", \
                    help = "prefix of the data file's name")
parser.add_argument('-n', '--nSample',    type=int, default=100, \
                    help = "number of samples")

args = parser.parse_args()

nx, ny = 32,  32
lx, ly = 1.0, 1.0
nSample= args.nSample
assert lx/nx == ly/ny

# hdf5 data file
dFile = h5.File(args.filePrefix + '_{}_{}.h5'.format(nx, args.nSample), 'w')
dFile.attrs['nSample']  = args.nSample
dFile.attrs['shape']    = np.array([nx, ny])
dFile.attrs['length']   = np.array([lx, ly])
# solution to the Poisson equation \nabla\cdot (a \nabla p) = f
ppData  = dFile.create_dataset('pp',  (nSample, ny, nx),    compression='gzip',
            compression_opts=9, dtype='float64', chunks=True)
# solution to the Laplace equation \nabla\cdot (a \nabla p) = 0, a is same as above
plData  = dFile.create_dataset('pl',  (nSample, ny, nx),    compression='gzip',
            compression_opts=9, dtype='float64', chunks=True)
# Dirichilet boundary condition
pBcData = dFile.create_dataset('pBc', (nSample, 2*(ny+nx)), compression='gzip',
            compression_opts=9, dtype='float64', chunks=True)
# coeficient a in the Poisson/Laplace equation
aData   = dFile.create_dataset('a',   (nSample, ny, nx),    compression='gzip',
            compression_opts=9, dtype='float64', chunks=True)
# right-hand-side f in the Poisson equation
fData   = dFile.create_dataset('f',   (nSample, ny, nx),    compression='gzip',
            compression_opts=9, dtype='float64', chunks=True)

# rhs, coefficients
scaGen2D = ScalarGenerator2D((lx, ly), (nx, ny))
f        = scaGen2D.generate_scalar2d(args.nSample, valMin=-10.0, valMax=10.0)
a, aBc   = scaGen2D.generate_scalar2d(args.nSample, valMin=0.1, valMax= 1.0,\
                                      outputBc=True, strictMin=True)
# solutions' bcs
bcGen1D = PeriodicScalarGenerator1D(size =2*(lx + ly), nCell=2*(nx + ny), nKnot=8)
pBc     = bcGen1D.generate_periodic_scalar(args.nSample, valMin=-1.0, valMax=1.0)

# generate solution
h      = lx / nx
pl     = np.zeros((nSample, ny, nx))
pp     = np.zeros((nSample, ny, nx))
psnSol = PoissonSolver2D()
zeroBc = np.zeros(2*(nx+ny))
zeroF  = np.zeros((ny, nx))
for s in range(nSample):

  pp[s,...] = psnSol.solve((lx, ly), zeroBc,   a[s,...], f[s,...], coefBc=aBc[s,:])
  pl[s,...] = psnSol.solve((lx, ly), pBc[s,:], a[s,...], zeroF,    coefBc=aBc[s,:])

  if (s+1) % 200 == 0:
    print("sample ", s+1)
    print("Laplace")
    psnSol.check_solution(h, pl[s,...], a[s,...], zeroF)
    print("Poisson, zero BC")
    psnSol.check_solution(h, pp[s,...], a[s,...], f[s,...])

plData[...]  = pl[...]
ppData[...]  = pp[...]
aData[...]   = a[...]
fData[...]   = f[...]
pBcData[...] = pBc[...]

dFile.close()