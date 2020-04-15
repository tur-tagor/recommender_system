import numpy as np
import pandas as pd
from scipy.io import arff


def pseudoinverse_matrix(X, y):
    # определяем псевдообратную матрицу
    inv = np.linalg.pinv(X.T @ X)
    # находим вектор весов
    return (inv @ X.T) @ y


def matrix_equation(X, y):
    a = X.T @ X
    b = X.T @ y
    return np.linalg.lstsq(a, b, rcond=None)[0]


if __name__ == '__main__':
    # dataset = pd.read_csv('dataset/train/Features_Variant_1.csv')
    dataset, meta = arff.loadarff('dataset/train/Features_Variant_1.arff')
    df = pd.DataFrame(dataset)
    x_np = df.iloc[:, :-1].to_numpy()
    y_np = df.iloc[:, -1:].to_numpy()
    ab_np = matrix_equation(x_np, y_np)
    # ab_np = pseudoinverse_matrix(x_np, y_np)
    print(ab_np)
