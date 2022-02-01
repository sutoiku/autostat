from sklearn.preprocessing import MinMaxScaler

from .run_settings import KernelSearchSettings

from .kernel_search import kernel_search, get_best_kernel_info
from .dataset_adapters import Dataset
from .utils.data_prep import scale_split, end_block_split
from .utils.logger import Logger
from .test_data.test_data_loader import load_matlab_test_data_by_file_num


class ABCDModel:
    def __init__(self, search_settings: KernelSearchSettings, logger: Logger = None):
        self.search_settings = search_settings
        self.logger = logger

    def fit(self, x, y):

        train_x, test_x = end_block_split(x, self.search_settings.cv_split)
        train_y, test_y = end_block_split(y, self.search_settings.cv_split)

        x_scaler = MinMaxScaler((-1, 1))
        x_scaler.fit(train_x)
        train_x, test_x = x_scaler.transform(train_x), x_scaler.transform(test_x)
        self.x_scaler = x_scaler

        y_scaler = MinMaxScaler((-1, 1))
        y_scaler.fit(train_y)
        train_y, test_y = y_scaler.transform(train_y), y_scaler.transform(test_y)

        search_dataset = Dataset(train_x, train_y, test_x, test_y)

        self.search_settings = (
            self.search_settings.replace_kernel_proto_constraints_using_dataset(
                search_dataset
            )
        )
        self.best_kernel_info = get_best_kernel_info(
            kernel_search(
                search_dataset, run_settings=self.search_settings, logger=self.logger
            )
        )

        best_model_spec = self.best_kernel_info.model.to_spec()

        # FIXME: inelegant to use __class__ here
        self.model = self.best_kernel_info.model.__class__(
            best_model_spec,
            Dataset(
                train_x=x_scaler.transform(x),
                train_y=y_scaler.transform(y),
                test_x=None,
                test_y=None,
            ),
            run_settings=self.search_settings,
        )
        self.model.fit()

    def predict(self, x):
        return self.model.predict(self.x_scaler.transform(x))

    def list_components(self):
        ...

    def predict_component(self, component_num: int, x):
        ...
