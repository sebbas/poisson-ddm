import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import scipy.sparse as sparse
import h5py as h5
from scalar_2d_generator import PeriodicScalarGenerator1D, ScalarGenerator2D
from Poisson_solver import *

#%%
nSample  = 10
nCell    = (32,  32)
size     = (1.0, 1.0)
scaGen2D = ScalarGenerator2D(size, nCell)
#%%
# rhs
rhsMin = -20.0
rhsMax =  20.0
rhss   = scaGen2D.generate_scalar2d(nSample, valMin=rhsMin, valMax=rhsMax)
# print(rhss.shape)
# fig, ax = plt.subplots(1, 2, figsize=(8,4))
# ax[0].imshow(rhss[0,:,:], origin='lower')
# ax[1].imshow(rhss[1,:,:], origin='lower')

#%%
# variable coefficient, alpha
# \nabla \cdot (\alpha\nabla p) = rhs
aMin =  0.1
aMax =  1.0
# az is used as the plural of a
az, aBcs = scaGen2D.generate_scalar2d(nSample, valMin=aMin, valMax=aMax,
                                      outputBc=True)
# print(az.shape)
# fig, ax = plt.subplots(1, 2, figsize=(8,4))
# ax[0].imshow(az[0,:,:], origin='lower')
# ax[1].imshow(az[1,:,:], origin='lower')

#%%
# bc
bcGen1D = PeriodicScalarGenerator1D(size =2*( size[0] +  size[1]),\
                                    nCell=2*(nCell[0] + nCell[1]),\
                                    nKnot=8)
bcMin   = -1.0
bcMax   =  1.0
bcs     = bcGen1D.generate_periodic_scalar(nSample, valMin=bcMin, valMax=bcMax)
# fig, ax = plt.subplots(2, 1, figsize=(8,8))
# h = size[0]/nCell[0]
# x = np.array([(i+0.5)*h for i in range(4*nCell[0])])
# l = 4 * size[0]
# x = np.concatenate([x-l, x, x+l])
# bc= np.concatenate([bcs[0,:,], bcs[0,:], bcs[0,:]])
# ax[0].plot(x,   bc)
# bc= np.concatenate([bcs[1,:,], bcs[1,:], bcs[1,:]])
# ax[1].plot(x,   bc)

#%%
# use previously generated rhs and coefficient to test the solver
h      = size[0] / nCell[0]
psnSol = PoissonSolver2D()
for s in range(nSample):
  x = psnSol.solve(size, bcs[s], az[s], rhss[s], coefBc=aBcs[s])
  psnSol.check_solution(h, x, az[s], rhss[s])

#%%
# test example with very simple cases
nx, ny = 32, 32
sz     = (1.0, 1.0)
h      = sz[0] / nx
psnSol = PoissonSolver2D()

# example 1
# p = sin(x)sin(y), a = cos(x)cos(y)
# f = cos(2x)sin^2(y) + sin^2(x)cos(2y)
# note that with
p, pBc, a, aBc, f = setup_psn2d_by_func(sz, (nx, ny), \
  lambda x, y: np.sin(x) * np.sin(y), \
  lambda x, y: np.cos(x) * np.cos(y), \
  lambda x, y: -np.sin(2*x)*np.sin(2*y), \
  outputCoefBc=True)
# solve and check solution
x = psnSol.solve(sz, pBc, a, f, coefBc=aBc)
print("Example 1, given rhs check solution")
psnSol.check_solution(h, x, a, f, exact=p)
print("Example 1, given solution check rhs")
ff = psnSol.compute_rhs(sz, p, pBc, a, coefBc=aBc)
psnSol.check_solution(h, p, a, ff)
print("error    L1, Linf: {:.4e} {:.4e}".format(
      np.mean(abs(f-ff)), np.amax(abs(f-ff))))
# plot the solution
# fig, ax = plt.subplots(1, 2, figsize=(9,4))
# im = ax[0].imshow(p, origin='lower', vmin=0.0, vmax=1.0)
# ax[0].set_title('exact')
# plt.colorbar(im, ax=ax[0], shrink=0.8)
# im = ax[1].imshow(x, origin='lower', vmin=0.0, vmax=1.0)
# ax[1].set_title('numerical')
# plt.colorbar(im, ax=ax[1], shrink=0.8)

# example 2
# p = sin(x)cos(y), a = sin(x)cos(y)
# f = cos(2x)cos^2(y) - sin^2(x)cos(2y)
p, pBc, a, aBc, f = setup_psn2d_by_func(sz, (nx, ny), \
  lambda x, y: np.sin(x) * np.cos(y), \
  lambda x, y: np.sin(x) * np.cos(y), \
  lambda x, y: np.cos(2*x)*np.cos(y)**2 - (np.sin(x)**2)*np.cos(2*y), \
  outputCoefBc=True)
# # solve and check solution
x = psnSol.solve(sz, pBc, a, f, coefBc=aBc)
print("Example 2, given rhs check solution")
psnSol.check_solution(h, x, a, f, exact=p)
print("Example 2, given solution check rhs")
ff = psnSol.compute_rhs(sz, p, pBc, a, coefBc=aBc)
psnSol.check_solution(h, p, a, ff)
print("error    L1, Linf: {:.4e} {:.4e}".format(
      np.mean(abs(f-ff)), np.amax(abs(f-ff))))
