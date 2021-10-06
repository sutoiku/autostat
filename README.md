# autostat

An implementation of the Automatic Statistician algorithm

## Problems

- why does adding restarts degrade performance so badly for PER? How is the model finding _better_ optima when it deviates from the residual-based starting point?
- product specs containing a sum operand have an extra scalar: $ s_1 \*A \* (s_2\*B + s_3 \* C)$ -- s_1 is redundant, should just be 1

### to do / roadmap

- catch errors and log instead of crashing when a spec fit crashes

- parallelization of tree search

  - catch GPU memory errors and respawn
  - can we autodetect GPU capacity and task memory usage somehow more granularly than as a proportion of a GPU?

- constraints seemingly not enforced on Dataset: 11-unemployment.mat in file:///home/bc/STOIC/time_series_forecasts/autostat/integration-test-reports/reports/report_2021-09-21_16%3A48%3A39.html for PER:

  - period_bounds=ConstraintBounds(lower=0.022619149694733665, upper=0.3596736412263738)
  - but
    - p=0.0197 at depth 1 PER\*RBF
    - p=0.0001 at depth 3 LIN+PER

- Change scaling / standardization method to avoid super small parameter values that lead to numerical instability for PERnc, possibly other kernels?

- gpytorch:

  - PERnc kernel
    - getting numerical errors. crashing with: `gpytorch.utils.errors.NotPSDError: Matrix not positive definite after repeatedly adding jitter up to 1.0e-06.`
    - on Mauna, PERnc doesn't seem to be updating in `scratch...`
      - are the derivatives on it's params meaningful?
      - is the kernel somehow frozen or being overwritten?
    - GPU memory usage!

- DECOMPOSITION

  - error per component

- Kernels

  - ADD CONSTANT KERNEL

    - make sure gpytorch is using ZeroMean with constant kernel option

  - priors / constraints on kernels:
    - some limit on length scale for smoothing kernels? -- smoothing kernels should NOT pick up non-stationarity in a signal, that should be captured by a non stationary (polynomial trend etc kernel) kernel
  - change point operator

  - Kernel SIMPLIFICATIONS:
    - no sum of LIN
    - no product of RBF
    - remove components with coef less than some value
    - prior on spacing of params for kernels of same type -- i.e., if RBF+RBF, length scales must differ by some amount (strong prior against close values), or if PER+PER, strong prior against similar periods

- cross validation / overfitting

- remove dependencies on gpytorch and sklearn (move to separate modules)

- server

- periodic time series

  - toeplitz matrix inversion for dense periodic series

### see:

- https://github.com/jamesrobertlloyd/gpss-research
- https://arxiv.org/pdf/1402.4304.pdf
- https://papers.nips.cc/paper/2016/file/3bbfdde8842a5c44a0323518eec97cbe-Paper.pdf#cite.duvenaud_grammars
  - hyperparam priors
  - GP search over structures
  - laplace approximation of model evidence
- https://arxiv.org/pdf/1302.4922.pdf

  - posterior decomposition: see note in appendix
    - sum of MV gaussians is MV gaussian: http://cs229.stanford.edu/section/more_on_gaussians.pdf

- block matrix inversion:
  - http://www.math.chalmers.se/~rootzen/highdimensional/blockmatrixinverse.pdf

### algorithm notes and tricks:
