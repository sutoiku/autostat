# Autostat

An implementation of the Automatic Bayesian Covariance Discovery algorithm (part of the "Automatic Statistician" research program), with a handful of adjustments made to focus on

# How to use

```python
import autostat
from autostat.test_data.test_data_loader import load_matlab_test_data_by_file_num

# `file_num=2` loads the famous Mauna Loa CO2 data set
x,y = load_matlab_test_data_by_file_num(file_num=2)

abcd_model = autostat.ABCDModel(autostat.KernelSearchSettings(max_search_depth = 5, num_cpus = 8))

abcd_model.fit(x,y)

autostat.plot.plot_decomposition(abcd_model)


```

## API

# Development notes, to-dos, next steps, and open questions

## Development notes

- This implementation exposes a scikit-learn style OO API, but internally we make heavy use of `frozen` DataClasses, and have tried to adopt an immutable + functional style whenever possible.
- Though it somewhat goes against the flow of Python's usual duck-typing approach, this project makes extensive use of types a "multiple dispatch" style to achieve polymorphism.
  - Note that the `multipledispatch` module and other similar approaches to function overloading were not adopted because as of late 2021 they were not well supported by `typing` and/or Pylance, which is critical for developer productivity when using this style. The temporary solution was to use `if...elif...` blocks with `isinstance`, but the plan is to replace this with pattern matching blocks once Python 3.10 is more widely adopted and supported.

## Algorithm notes and tricks:

- This implementation uses a slightly different tree search procedure than that described in the original ABCD paper. In particular, rather than choosing the best performing model at each depth based on the BIC of that model when fitted on all available data, we hold out a sample of test data at the end of the time series, and we choose the model under which the test data has the highest posterior log probability. We do this for the following reasons:

  - In the original ABCD paper, overfitting is avoided by penalizing model complexity using a BIC score. Rather than use that approach, we choose to avoid overfitting by way of an empirical cross-validation-like method.
  - For this application for which this implementation was written, the quality of the decomposition of the signal throughout the dataset is important, but forecast plausibility is at least as important if not moreso. Thus it's important to somehow attach more weight to the observations at the end of the series. An analytic weighting scheme could be used, but the empirical CV-like approach is simple and produces satisfactory results.
  - Though a CV-like approach may appear a bit out of place in a largely Bayesian method, we note that the BIC score used in the original paper also seems to have been originally chosen as a matter of expedience rather than strictest rigour. (See e.g. https://dl.acm.org/doi/10.5555/3157382.3157422 for an approach that uses a more fully Bayesian method of selecting compositional models, though at the cost of additional complexity.)
  - Note that once the best model structure is chosen, the final model is refit using all available data.

- To find a suitable initialization for newly introduce period kernel components at search depth _n+1_, we run an FFT over the residuals of the upstream model at depth _n_ and extract the frequency with the largest weight. This typical results in better fits and convergence.
  - Note also that adding random initializations can degrade performance badly for PER. I'm not sure exactly what is going on, but somehow the model is finding _numerically better_ optima when it deviates from the residual-based starting point, however the fits are much less convincing visually.

## Open questions and problems

- product specs containing a sum operand have an extra scalar: $ s_1 \*A \* (s_2\*B + s_3 \* C)$ -- s_1 is redundant, should just be 1

## To-dos and next steps

### REFINE FITS

- cross validation / overfitting / HYPERPARAMETERS

  - are there hyperparameters that could be tuned across data sets to get better out-of-sample likelihood?
    - parameter constraints
    - kernel penalties
    - kernel scoring function (alternative to BIC?)
    - data rescaling parameters
    - noise component inflation
      - (keeping x% of data as test) Fit GPs using y% of data, then CV on (100-x-y)%, optimizing noise component to get best score. Model with best score given noise optimized on CV data is passed to next round
  - NEED OUT OF SAMPLE PREDICTION QUALITY MEASUREMENT
    - OVERALL quality comparison score that considers overall performance on all datasets

- Change scaling / standardization method to avoid super small parameter values that lead to numerical instability for PERnc, possibly other kernels? When everything is scaled into [-1, 1], data points can end up very close to each other, which can cause numerical issues especially with PERnc

- gpytorch:

  - PERnc kernel
    - getting numerical errors. crashing with: `gpytorch.utils.errors.NotPSDError: Matrix not positive definite after repeatedly adding jitter up to 1.0e-06.`
    - on Mauna, PERnc doesn't seem to be updating in `scratch...`
      - are the derivatives on it's params meaningful?
      - is the kernel somehow frozen or being overwritten?
    - GPU memory usage!

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

### CODE CLEANUP, OTHER IMPROVEMENTS

- When an individual model spec fit crashes for any reason, this should be caught and logged rather than crashing

- The Gpytorch and Sklearn backends used for GP fitting should probably move to separate modules to remove the direct dependency of this module on both of those -- it is perfectly possible to run this module with only Sklearn or Gpytorch, so there is no need to pull in both.

- For regular time series, look into Toeplitz matrix inversion / solvers. There exist fast (n \* log(n)^2) solvers for Toeplitz matrices that use cumulant matrix + FFT techniques.

- Parallelization of tree search on GPU needs improvement.

  - GPU parallelism is not really working for 1 GPU on one machine (i.e., it doesn't really work to spawn parallel GPU processes).

    - It is possible to spawn multiple GPU processes, but it seems to be the case that Ray expects GPUs to be fully available and not shared with desktop processes. This leads to OOM conditions when partitioning a GPU resource. Example: if you set a task to require 0.25 of a GPU, Ray will launch 4 tasks without checking the current free memory on the GPU

      - As of 2021-09, there was no obvious way to detect free GPU memory, nor to allocate tasks with more granularly than as a proportion of a GPU (e.g. by abosolute size)
      - It would be great to catch GPU memory errors and respawn -- NOT WORKING

    - It seems to be the case that Pytorch loads a ton of extra CUDA kernels, which eats a ton of memory. This happens once per process attempting to use the GPU, so the memory overhead is huge even with modest data. This prevents full utilization of the GPU.

# References and links

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
