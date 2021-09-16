# autostat

An implementation of the Automatic Statistician algorithm

## Problems

- why does adding restarts degrade performance so badly for PER? How is the model finding _better_ optima when it deviates from the residual-based starting point?
- product specs containing a sum operand have an extra scalar: $ s1 _ A _ (s2 _ B + s3 _ C)$ -- s1 is redundant, should just be 1

### to do / roadmap

- parallelization of tree search

- gpytorch:

  - add PERnc kernel
    - getting numerical errors. crashing with: `gpytorch.utils.errors.NotPSDError: Matrix not positive definite after repeatedly adding jitter up to 1.0e-06.`
    - on Mauna, PERnc doesn't seem to be updating in `scratch...`
      - are the derivatives on it's params meaningful?
      - is the kernel somehow frozen or being overwritten?

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

### implementation notes and tricks:
