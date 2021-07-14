# %%
# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn

from matplotlib import pyplot as plt
import numpy as np

from autostat.utils.mauna_data_loader import load_mauna_numpy

import autostat

from autostat.kernel_swaps import (
    other_base_kernels,
    sort_specs_by_type,
    kernel_str,
    KernelInitialValues,
)

from autostat.kernel_search import (
    kernel_search,
    score_kernel_spec,
    starting_kernel_specs,
)

from autostat.kernel_tree_types import (
    RBFKernelSpec as RBF,
    RQKernelSpec as RQ,
    LinearKernelSpec as LIN,
    PeriodicKernelSpec as PER,
    AdditiveKernelSpec as ADD,
    ProductKernelSpec as PROD,
    Dataset as Dataset,
    NpDataSet as NpDataSet,
)

from autostat.kernel_trees_sklearn import SklearnGPModel, build_kernel

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ExpSineSquared as sklearnPER,
    RBF as sklearnRBF,
)

train_x, test_x, train_y, test_y = load_mauna_numpy(rescale=True)
d = NpDataSet(train_x, train_y, test_x, test_y)
# %%
[build_kernel(k) for k in starting_kernel_specs()]
# %%
kernel_scores = kernel_search(d, SklearnGPModel, search_iterations=2)
# %%

PER(**{"period": 4, "length_scale": 5})
# %%
k = ADD([PROD([RBF()]), PROD([LIN()])])
print(repr(k))
print(k)
# %%

skk = build_kernel(k, top_level=True)
print(skk)
skk
# %%
skgp = GaussianProcessRegressor(kernel=skk, alpha=1e-8)

skgp.fit(d.train_x, d.train_y)

print(skgp.kernel)
print(skgp.kernel_)
skgp
# %%
m = SklearnGPModel(k, d, alpha=1e-8)

m.fit(d)
# m.gp.fit(d.train_x,d.train_y)
print(m.gp.kernel)
print(m.gp.kernel_)
# m.gp
# %%

y, l, u = m.predict(d.train_x)
plt.plot(d.train_x, d.train_y, "k.", markersize=1)
plt.plot(d.train_x, y)
plt.show()
plt.plot(d.train_y.flatten() - y)
print(k)
print(m.gp.kernel)
print(m.gp.kernel_)
# m.to_spec()
# %%
# k = sklearnPER() + sklearnRBF()
score_kernel_spec(k, d, SklearnGPModel)
# %%
# %%
best_kernel = min(kernel_scores.items(), key=lambda tup: tup[1])
# %%


k = ADD([PROD(RBF()), PROD(LIN())])

m = SklearnGPModel(k, d)

m.fit(d)
type(m.gp.kernel_)
m.gp.kernel_
y, l, u = m.predict(d.train_x)
resid = d.train_y.flatten() - y
plt.plot(d.train_x, resid)
plt.plot(d.train_x[:11], resid[:11])

d.train_x[11] - d.train_x[0]
# %%

# %%

from scipy.fft import fft, fftfreq, dct

N = len(resid)
yf = fft(resid)
T = 2 / N
xf = fftfreq(N, T)
plt.semilogx(1 / xf, np.abs(yf))

# yf2 = yf[yf>1]

# plt.plot(xf, yf2)
1 / xf[np.abs(yf) == max(np.abs(yf))]
# %%

from scipy.fft import fft, fftfreq

# Number of sample points
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N * T, N, endpoint=False)
y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x)
plt.plot(x, y)
plt.show()
yf = fft(y)
xf = fftfreq(N, T)

# plt.plot(xf, 2.0 / N * np.abs(yf))
plt.plot(xf, np.abs(yf))

indexes = list(range(n))
# sort indexes by frequency, lower -> higher
indexes.sort(key=lambda i: np.absolute(f[i]))

1 / xf[np.abs(yf) == max(np.abs(yf))]
xf[np.abs(yf) == max(np.abs(yf))]

# %%
import numpy

# %%


# %%


# %%
from sklearn.datasets import fetch_openml

# %%
fetch_openml(data_id=41744)

# %%
