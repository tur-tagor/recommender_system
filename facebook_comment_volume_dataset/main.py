import numpy as np
from scipy.io import arff
from sklearn import metrics, preprocessing
from tabulate import tabulate

from facebook_comment_volume_dataset.lr_gradient import gradient_descent


class FoldResult:
    def __init__(self):
        self._rmse_train = None
        self._rmse_test = None
        self._r2_train = None
        self._r2_test = None
        self._w = None

    @property
    def rmse_train(self) -> float:
        return self._rmse_train

    @property
    def rmse_test(self) -> float:
        return self._rmse_test

    @property
    def r2_train(self) -> float:
        return self._r2_train

    @property
    def r2_test(self) -> float:
        return self._r2_test

    @property
    def w(self) -> np.ndarray:
        return self._w

    def toNdArrayColumn(self) -> np.ndarray:
        row = np.concatenate(
            [np.array([self._rmse_test, self._rmse_train, self._r2_test, self._r2_train], dtype=np.double),
             self._w.flatten()])
        return row.reshape(row.shape[0], 1)

    @rmse_train.setter
    def rmse_train(self, value):
        self._rmse_train = value

    @rmse_test.setter
    def rmse_test(self, value):
        self._rmse_test = value

    @r2_test.setter
    def r2_test(self, value):
        self._r2_test = value

    @r2_train.setter
    def r2_train(self, value):
        self._r2_train = value

    @w.setter
    def w(self, value):
        self._w = value


def calc_regression(train: np.ndarray, method):
    X_train = train[:, :-1]
    y_train = train[:, -1:]
    return method(X_train, y_train)


def verify_regression(train: np.ndarray, test: np.ndarray, w) -> FoldResult:
    result = FoldResult()
    X_train = train[:, :-1]
    y_train = train[:, -1:]
    X_test = test[:, :-1]
    y_test = test[:, -1:]
    y_test_pred = X_test @ w
    y_train_pred = X_train @ w
    result.rmse_train = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
    result.rmse_test = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    result.r2_train = metrics.r2_score(y_train, y_train_pred)
    result.r2_test = metrics.r2_score(y_test, y_test_pred)
    result.w = w
    return result


def fit(train_dataset: np.ndarray, test_dataset: np.ndarray, method) -> FoldResult:
    norm_train_dataset = preprocessing.StandardScaler().fit_transform(train_dataset[:, :-1])
    train_dataset = np.c_[np.ones((train_dataset.shape[0], 1)), norm_train_dataset, train_dataset[:, -1:]]
    norm_test_dataset = preprocessing.StandardScaler().fit_transform(np.array(test_dataset[:, :-1]))
    test_dataset = np.c_[np.ones((test_dataset.shape[0], 1)), norm_test_dataset, test_dataset[:, -1:]]
    w = calc_regression(train_dataset, method)
    return verify_regression(train_dataset, test_dataset, w)


def convert_to_ndarray(d: dict):
    return np.column_stack([x.toNdArrayColumn() for x in d.values()])


def print_results(folded_results):
    label_y = np.concatenate([np.array(['-']),
                              np.array(['F%s' % (n) for n in folded_results.keys()]),
                              np.array(['mean', 'std'])])
    label_x = np.concatenate([np.array(['rmse_test', 'rmse_train', 'r2_test', 'r2_train']),
                              np.array(['w%s' % (n[0]) for n in enumerate(folded_results[0].w)])])
    s = convert_to_ndarray(folded_results)
    s = np.column_stack([label_x.reshape(label_x.shape[0], 1), s, np.mean(s, axis=1), np.std(s, axis=1)])
    np.set_printoptions(linewidth=200)
    print('| | F1 | F2 | F3 | F4 | F5 | mean | std |')
    print(tabulate(s, tablefmt='pipe'))


if __name__ == '__main__':
    # pseudoinverse_matrix
    # matrix_equation
    # gradient_descent(alpha=0.1, tolerance=0.1, max_iters=1000)
    method = gradient_descent(alpha=0.1, tolerance=0.1, max_iters=1000)
    dataset, meta = arff.loadarff('dataset/train/Features_Variant_1.arff')
    np.random.shuffle(dataset)
    folded_dataset = np.array_split(dataset, 5)
    folded_results = {}
    for i, test_dataset in enumerate(folded_dataset):
        print('fold = ', i)
        train_dataset = np.array(np.concatenate([x[1] for x in enumerate(folded_dataset) if x[0] != i]).tolist())
        test_dataset = np.array(test_dataset.tolist())
        folded_results[i] = fit(train_dataset, test_dataset, method)

    print_results(folded_results)
    print(123)
