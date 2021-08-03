# autostat

An implementation of the Automatic Statistician algorithm

### to do / roadmap

- Kernels

  - priors on kernels:
    - period should be less than 1x data length
    - some limit on length scale for smoothing kernels? -- smoothing kernels should NOT pick up non-stationarity in a signal, that should be captured by a non stationary (polynomial trend etc kernel) kernel
  - change point operator
  - non-stationary random walk-type kernels kernel?
  - matern vs RBF

- cross validation / overfitting

- decomposition

  - error per component

- parallelism
- reimplement gpytorch
- multiple data sets for integration test suite
- remove dependencies on gpytorch and sklearn (move to separate modules)

- server

### see:

- https://github.com/jamesrobertlloyd/gpss-research
- https://arxiv.org/pdf/1402.4304.pdf
