import numpy as np
import sys


def get_gradient(w, x, y):
    """
    Gradient function
    :param w:
    :param x:
    :param y:
    :return: gradient and mse
    """
    y_predict = x.dot(w).flatten()
    error = (y.flatten() - y_predict)
    mse = (1.0 / len(x)) * np.sum(np.power(error, 2))
    gradient = -(1.0 / len(x)) * error.dot(x)
    return gradient, mse


def gradient_descent(alpha: float, tolerance: float, max_iters: int = 1000):
    def gradient_descent_(X_train: np.ndarray, y_train: np.ndarray):
        # Perform gradient descent to learn model

        step = alpha
        # w = np.random.randn(X_train.shape[1])
        w = np.zeros(X_train.shape[1])

        # Perform Gradient Descent

        iterations = 1
        base_error = sys.float_info.max
        while True:
            gradient, error = get_gradient(w, X_train, y_train)
            new_w = w - step * gradient

            # Stopping Condition
            if np.sum(abs(new_w - w)) < tolerance:
                print("Converged.")
                break

            # Print error every 100 iterations
            if iterations % 100 == 0:
                print("Iteration: %d - Error: %.4f" % (iterations, error))
                if base_error > error:
                    base_error = error
                    step = step * 1.001
                else:
                    step = step * 0.999

            iterations += 1
            w = new_w

            if iterations == max_iters:
                break

        return w.reshape(w.shape[0], 1)

    return gradient_descent_
