import numpy as np
from scipy.io import arff
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt

from facebook_comment_volume_dataset.lr_gradient import gradient_descent
from facebook_comment_volume_dataset.lr_matrix import pseudoinverse_matrix, matrix_equation


def calc_regression(train: np.ndarray, method):
    X_train = train[:, :-1]
    y_train = train[:, -1:]
    return method(X_train, y_train)


def verify_regression(test: np.ndarray, w, step=1):
    X_test = test[:, :-1]
    y_test = test[:, -1:]
    y_pred = X_test @ w
    print(' ' * step, 'Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print(' ' * step, 'Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print(' ' * step, 'Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(' ' * step, 'R2 Error:', metrics.r2_score(y_test, y_pred))
    plt.plot(y_test, label='test')
    plt.plot(y_pred, label='pred')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    dataset, meta = arff.loadarff('dataset/train/Features_Variant_1.arff')
    np.random.shuffle(dataset)
    folded_dataset = np.array_split(dataset, 5)
    for i, test_dataset in enumerate(folded_dataset):
        train_dataset = np.array(np.concatenate([x[1] for x in enumerate(folded_dataset) if x[0] != i]).tolist())
        norm_train_dataset = preprocessing.StandardScaler().fit_transform(train_dataset[:, :-1])
        train_dataset = np.c_[np.ones((train_dataset.shape[0], 1)), norm_train_dataset, train_dataset[:, -1:]]
        w1 = calc_regression(train_dataset, pseudoinverse_matrix)
        w2 = calc_regression(train_dataset, matrix_equation)
        w3 = calc_regression(train_dataset, gradient_descent(alpha=0.1, tolerance=0.1, max_iters=1000))
        w3 = w3.reshape(len(w2), 1)
        print('fold = ', i)
        test_dataset = np.array(test_dataset.tolist())
        norm_test_dataset = preprocessing.StandardScaler().fit_transform(np.array(test_dataset[:, :-1]))
        test_dataset = np.c_[np.ones((test_dataset.shape[0], 1)), norm_test_dataset, test_dataset[:, -1:]]
        print(' w1', w1[0])
        verify_regression(test_dataset, w1)
        print(' w2', w2[0])
        verify_regression(test_dataset, w2)
        print(' w3', w3[0])
        verify_regression(test_dataset, w3)
