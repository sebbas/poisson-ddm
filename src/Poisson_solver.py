import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import scipy.sparse as sparse
import time

class PoissonSolver2D:
  ''' Solve equation \nabla\cdot a\nabla p = b
  '''
  def __init__(self, backend=0):
    self.tolerance = 1e-03
    self.backend = backend

    if backend == 1: # pyAmg (CPU) - single core only!
      import pyamg

    if backend == 2: # pyAmgx (GPU)
      import pyamgx
      pyamgx.initialize()
      self.cfg = pyamgx.Config().create_from_dict({
        "config_version": 2,
        "solver": {
          "preconditioner": {
            "print_grid_stats": 1,
            "print_vis_data": 0,
            "solver": "AMG",
            "smoother": {
              "scope": "jacobi",
              "solver": "BLOCK_JACOBI",
              "monitor_residual": 0,
              "print_solve_stats": 0
            },
            "print_solve_stats": 0,
            "aggressive_levels": 2,
            "presweeps": 1,
            "interpolator": "D2",
            "max_iters": 1,
            "monitor_residual": 0,
            "store_res_history": 0,
            "scope": "amg",
            "max_levels": 100,
            "cycle": "V",
            "postsweeps": 1
          },
          "solver": "PCG",
          "print_solve_stats": 0,
          "obtain_timings": 1,
          "max_iters": 100,
          "monitor_residual": 1,
          "convergence": "ABSOLUTE",
          "scope": "main",
          "tolerance": self.tolerance,
          "norm": "L2"
        }
      })
      self.rsc = pyamgx.Resources().create_simple(cfg)
      self.pyamgxSolver = pyamgx.Solver().create(rsc, cfg)
      self.AGpu = pyamgx.Matrix().create(self.rsc)
      self.bGpu = pyamgx.Vector().create(self.rsc)
      self.xGpu = pyamgx.Vector().create(self.rsc)

    def __del__(self):
      if self.backend == 2:
        self.AGpu.destroy()
        self.xGpu.destroy()
        self.bGpu.destroy()
        self.pyamgxSolver.destroy()
        self.rsc.destroy()
        self.cfg.destroy()
        pyamgx.finalize()


  def _build_matrix(self, a, coefBc=None):
    # each grid cell corresponds to one row
    # inner cells have 5 nonzero entries, 1 diagonal entry for itself
    # and 4 entries for neighbor cells in stencil
    # cells on the edge drop one entry for that edge
    nx, ny   = a.shape[0], a.shape[1]
    nNonZero = 5*nx*ny - 2*(nx+ny)
    row      = np.zeros(nNonZero)
    col      = np.zeros(nNonZero)
    val      = np.zeros(nNonZero)
    iVal     = 0
    for i in range(ny):
      for j in range(nx):
        # coefficients on face centers (ip, im, jp, jm)
        # p - plus, m - minus
        if coefBc is not None:
          aip = _interp_face(a[i,  j],   a[i+1,j  ]) if i < ny-1 else coefBc[2*nx+ny-1-j]
          aim = _interp_face(a[i-1,j],   a[i,  j  ]) if i > 0    else coefBc[j]
          ajp = _interp_face(a[i,  j],   a[i,  j+1]) if j < nx-1 else coefBc[nx+i]
          ajm = _interp_face(a[i,  j-1], a[i,  j  ]) if j > 0    else coefBc[-i-1]
        # default diagal value
        diagVal = aip + aim + ajp + ajm
        # lower point in stencil
        if i == 0:
          diagVal = diagVal + aim
        else:
          row[iVal] =  i   *nx + j
          col[iVal] = (i-1)*nx + j
          val[iVal] = -aim
          iVal     += 1
        # upper point in stencil
        if i == ny - 1:
          diagVal = diagVal + aip
        else:
          row[iVal] =  i   *nx + j
          col[iVal] = (i+1)*nx + j
          val[iVal] = -aip
          iVal     += 1
        # left point in stencil
        if j == 0:
          diagVal = diagVal + ajm
        else:
          row[iVal] = i*nx + j
          col[iVal] = i*nx + j - 1
          val[iVal] = -ajm
          iVal     += 1
        # right point in stencil
        if j == nx - 1:
          diagVal = diagVal + ajp
        else:
          row[iVal] = i*nx + j
          col[iVal] = i*nx + j + 1
          val[iVal] = -ajp
          iVal     += 1
        # diagnal point
        row[iVal] = i*nx + j
        col[iVal] = i*nx + j
        val[iVal] = diagVal
        iVal     += 1

    assert iVal == nNonZero
    mat = sparse.csr_matrix((val, (row, col)), shape=(nx*ny, nx*ny))

    return mat


  def _build_rhs(self, h, bc, rhs, coefBc=None, coef=None):
    ny, nx = rhs.shape
    assert len(bc) == 2 * (ny + nx)

    b = -h*h * rhs.copy()
    if coefBc is not None:
      # i- boundary (i == 0)
      b[ 0, :] += 2 * coefBc[:nx] * bc[:nx]
      # j+ boundary
      b[ :,-1] += 2 * coefBc[nx:nx+ny] * bc[nx:nx+ny]
      # i+ boundary
      b[-1, :] += 2 * np.flip(coefBc[nx+ny:2*nx+ny] * bc[nx+ny:2*nx+ny])
      # j- boundary
      b[ :, 0] += 2 * np.flip(coefBc[2*nx+ny:] * bc[2*nx+ny:])

    b = np.reshape(b, (nx*ny))

    return b


  def solve(self, sz, bc, a, rhs, coefBc=None):
    assert a.shape == rhs.shape \
      and  len(bc) == 2*(a.shape[0] + a.shape[1])
    assert sz[0] / a.shape[0] == sz[1] / a.shape[1]

    h = sz[0] / a.shape[0]
    A = self._build_matrix(a, coefBc)
    b = self._build_rhs(h, bc, rhs, coefBc=coefBc, coef=a)
    if self.backend == 0:
      sTime = time.perf_counter()
      x = sparse.linalg.spsolve(A, b)
      eTime = time.perf_counter()

    elif self.backend == 1:
      sTime = time.perf_counter()
      x = pyamg.solve(A, b, verb=False, tol=self.tolerance)
      eTime = time.perf_counter()

    elif self.backend == 2:
      x = np.zeros_like(rhs.flatten())
      self.AGpu.upload_CSR(A)
      self.bGpu.upload(b)
      self.xGpu.upload(x)
      self.pyamgxSolver.setup(AGpu)

      sTime = time.perf_counter()
      solver.solve(bGpu, xGpu)
      eTime = time.perf_counter()

      self.xGpu.download(x)

    timeTaken = eTime - sTime
    x = np.reshape(x, a.shape)

    return x, timeTaken


  def compute_rhs(self, sz, p, bc, a, coefBc=None):
    assert sz[0]/p.shape[0] == sz[1]/p.shape[1]
    assert p.shape == a.shape and len(bc) == 2*np.sum(p.shape)

    nx, ny = p.shape[0], p.shape[1]
    h      = sz[0] / nx
    invhh  = 1.0/h/h
    f      = np.zeros(p.shape)

    for i in range(ny):
      for j in range(nx):
        # coefficients on cell (i,j)'s four face centers
        if coefBc is not None:
          aip = _interp_face(a[i,  j],   a[i+1,j  ]) if i < ny-1 else coefBc[2*nx+ny-1-j]
          aim = _interp_face(a[i-1,j],   a[i,  j  ]) if i > 0    else coefBc[j]
          ajp = _interp_face(a[i,  j],   a[i,  j+1]) if j < nx-1 else coefBc[nx+i]
          ajm = _interp_face(a[i,  j-1], a[i,  j  ]) if j > 0    else coefBc[-i-1]
        # neighbor cell values
        pip = p[i+1, j] if i < ny-1 else 2*bc[2*nx+ny-1-j] - p[i,j]
        pim = p[i-1, j] if i > 0    else 2*bc[j]           - p[i,j]
        pjp = p[i, j+1] if j < nx-1 else 2*bc[nx+i]        - p[i,j]
        pjm = p[i, j-1] if j > 0    else 2*bc[-i-1]        - p[i,j]
        # rhs value
        f[i,j] = aip*(pip - p[i,j]) - aim*(p[i,j] - pim) \
               + ajp*(pjp - p[i,j]) - ajm*(p[i,j] - pjm)
        f[i,j] = f[i,j] * invhh

    return f



  def check_solution(self, h, p, a, b, exact=None):
    '''
      check the residual and error (if exact solution presented)
    '''
    nx, ny = p.shape[0], p.shape[1]
    invhh  = 1.0/h/h

    avgRes, maxRes = 0.0, 0.0
    for i in range(1, nx-1):
      for j in range(1, ny-1):
        res = _interp_face(a[i,  j], a[i+1,j]) * (p[i+1,j] - p[i,  j]) \
            - _interp_face(a[i-1,j], a[i,  j]) * (p[i,  j] - p[i-1,j]) \
            + _interp_face(a[i,  j], a[i,j+1]) * (p[i,j+1] - p[i,  j]) \
            - _interp_face(a[i,j-1], a[i,  j]) * (p[i,  j] - p[i,j-1])
        res = abs(res * invhh - b[i,j])
        avgRes += res
        maxRes  = max(maxRes, res)
    print("residual L1, Linf: {:.4e} {:.4e}".format(avgRes, maxRes))

    if exact is not None:
      err = abs(p - exact)
      avgErr, maxErr = np.mean(err), np.amax(err)
      print("error    L1, Linf: {:.4e} {:.4e}".format(avgErr, maxErr))

    if exact is None:
      return avgRes, maxRes
    else:
      return avgRes, maxRes, avgErr, maxErr


def _interp_face(left, right):
  return 0.5 * (left + right)


def setup_psn2d_by_func(sz, nCell, f_p, f_a, f_b, outputCoefBc=False):
  assert len(sz) == len(nCell)
  nx, ny = nCell[0], nCell[1]
  h      = sz[0] / nx
  p      = np.zeros((ny, nx))
  a      = np.zeros((ny, nx))
  b      = np.zeros((ny, nx))
  pBc    = np.zeros(2*(nx + ny))
  aBc    = np.zeros(2*(nx + ny))
  # cell center
  for i in range(ny):
    for j in range(nx):
      x, y   = (j+0.5)*h, (i+0.5)*h
      p[i,j] = f_p(x, y)
      a[i,j] = f_a(x, y)
      b[i,j] = f_b(x, y)
  # bc, face center
  for j in range(nx):
    x      = (j + 0.5) * h
    pBc[j] = f_p(x, 0.0)
    aBc[j] = f_a(x, 0.0)
    pBc[2*nx+ny-1-j] = f_p(x, sz[1])
    aBc[2*nx+ny-1-j] = f_a(x, sz[1])
  for i in range(ny):
    y         = (i + 0.5) * h
    pBc[-i-1] = f_p(0.0, y)
    aBc[-i-1] = f_a(0.0, y)
    pBc[nx+i] = f_p(sz[0], y)
    aBc[nx+i] = f_a(sz[0], y)

  if outputCoefBc:
    return p, pBc, a, aBc, b
  else:
    return p, pBc, a, b

