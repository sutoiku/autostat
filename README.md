# autostat

An implementation of the Automatic Statistician algorithm

## Problems

- why does adding restarts degrade performance so badly for PER? How is the model finding _better_ optima when it deviates from the residual-based starting point?

### to do / roadmap

- DECOMPOSITION

  - error per component
  - expand to additive form

    - once fitted, can construct new kernel matrices easily (independently of GP implementations if needed)

- Kernels

  - priors / constraints on kernels:
    - some limit on length scale for smoothing kernels? -- smoothing kernels should NOT pick up non-stationarity in a signal, that should be captured by a non stationary (polynomial trend etc kernel) kernel
  - change point operator

  - Kernel SIMPLIFICATIONS:
    - no sum of LIN
    - no product of RBF
    - remove components with coef less than some value
    - prior on spacing of params for kernels of same type -- i.e., if RBF+RBF, length scales must differ by some amount (strong prior against close values), or if PER+PER, strong prior against similar periods

- cross validation / overfitting

- parallelism
- reimplement gpytorch
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
