# %%
# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn

from autostat.utils.mauna_data_loader import load_mauna_numpy


from autostat.kernel_search import kernel_search, find_best_kernel_and_predict

from autostat.kernel_tree_types import (
    NpDataSet as NpDataSet,
)

from autostat.kernel_trees_sklearn import SklearnGPModel


train_x, test_x, train_y, test_y = load_mauna_numpy(rescale=True)
d = NpDataSet(train_x, train_y, test_x, test_y)
# %%
kernel_scores = kernel_search(d, SklearnGPModel, search_iterations=2)
# %%
list(kernel_scores.keys())
# %%
y, l, u = find_best_kernel_and_predict(d, SklearnGPModel, search_iterations=2)

# %%
