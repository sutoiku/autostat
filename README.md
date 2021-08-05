# autostat

An implementation of the Automatic Statistician algorithm

### to do / roadmap

- Kernels

  - priors / constraints on kernels:
    - some limit on length scale for smoothing kernels? -- smoothing kernels should NOT pick up non-stationarity in a signal, that should be captured by a non stationary (polynomial trend etc kernel) kernel
  - change point operator

- cross validation / overfitting

- decomposition

  - error per component

- parallelism
- reimplement gpytorch
- multiple data sets for integration test suite
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

#### constraints

- want to pass down constraints and initial guesses, both based on data
