# autostat

An implementation of the Automatic Statistician algorithm

### to do / roadmap

- Kernels

  - priors / constraints on kernels:
    - some limit on length scale for smoothing kernels? -- smoothing kernels should NOT pick up non-stationarity in a signal, that should be captured by a non stationary (polynomial trend etc kernel) kernel
  - change point operator

  - Kernel SIMPLIFICATION:
    - no sum of LIN
    - no product of RBF
    - remove components with coef less than some value
    - prior on spacing of params for kernels of same type -- i.e., if RBF+RBF, length scales must differ by some amount (strong prior against close values), or if PER+PER, strong prior against similar periods

- cross validation / overfitting

- decomposition

  - error per component

- parallelism
- reimplement gpytorch
- remove dependencies on gpytorch and sklearn (move to separate modules)

- server

- periodic time series

  - toeplitz matrix inversion for dense periodic series

- CLEAN UP
  - clean up implementation of kernelSpecTypes to use shared fit_count and spec_str implementations (unless overridden) by accessing field names and checking for defaults; add PER2 to kernels list

### see:

- https://github.com/jamesrobertlloyd/gpss-research
- https://arxiv.org/pdf/1402.4304.pdf
- https://papers.nips.cc/paper/2016/file/3bbfdde8842a5c44a0323518eec97cbe-Paper.pdf#cite.duvenaud_grammars
  - hyperparam priors
  - GP search over structures
  - laplace approximation of model evidence
