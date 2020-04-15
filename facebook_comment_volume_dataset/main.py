import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.io import arff
from sklearn import metrics, preprocessing, linear_model

from facebook_comment_volume_dataset.lr_gradient import gradient_descent
from facebook_comment_volume_dataset.lr_matrix import pseudoinverse_matrix, matrix_equation


def calc_regression(train: DataFrame, method):
    X_train = train.iloc[:, :-1].to_numpy()
    y_train = train.iloc[:, -1:].to_numpy()
    return method(X_train, y_train)


def verify_regression(test: DataFrame, w, step=1):
    X_test = test.iloc[:, :-1].to_numpy()
    y_test = test.iloc[:, -1:].to_numpy()
    y_pred = X_test @ w
    print(' ' * step, 'Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print(' ' * step, 'Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print(' ' * step, 'Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(' ' * step, 'R2 Error:', metrics.r2_score(y_test, y_pred))


if __name__ == '__main__':
    dataset, meta = arff.loadarff('dataset/train/Features_Variant_1.arff')
    np.random.shuffle(dataset)
    folded_dataset = np.array_split(dataset, 5)
    for i, train_dataset in enumerate(folded_dataset):
        # train_dataset = preprocessing.normalize(np.array(train_dataset.tolist(), dtype=np.dtype('f8'))[:, :-1])
        train_dataset = np.c_[preprocessing.normalize(np.array(train_dataset.tolist(), dtype=np.dtype('f8'))[:, :-1]), np.array(
            train_dataset.tolist(), dtype=np.dtype('f8'))[:, -1:]]
        df_train = pd.DataFrame(train_dataset)
        w = calc_regression(df_train, pseudoinverse_matrix)
        w1 = calc_regression(df_train, matrix_equation)
        # w2 = calc_regression(df_train, gradient_descent(alpha=1, tolerance=0.01, max_iters=10000))
        # w2 = w2.reshape(len(w2), 1)
        w3 = linear_model.LinearRegression().fit(df_train.iloc[:, :-1].to_numpy(), df_train.iloc[:, -1:].to_numpy()).coef_.T
        for j, test_dataset in enumerate(folded_dataset):
            print('train = ', i, 'test = ', j)
            df_test = pd.DataFrame(test_dataset)
            verify_regression(df_test, w)
        print()
        print(w)
