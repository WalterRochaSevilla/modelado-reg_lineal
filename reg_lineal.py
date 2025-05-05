import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

class Regresion:
    def __init__(self, theta_init=None, include_bias=True, degree=2, selected_features=[0, 1, 3, 5]):
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.include_bias = include_bias
        self.theta = theta_init.reshape(-1, 1) if theta_init is not None else None
        self.__historial = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.selected_features = selected_features

    def _add_polynomial_features(self, X):
        X_poly = self.poly.fit_transform(X)
        return X_poly[:, self.selected_features]

    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_poly = self._add_polynomial_features(X)
        if self.include_bias:
            X_poly = np.concatenate((np.ones((X_poly.shape[0], 1)), X_poly), axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_poly, y.reshape(-1, 1), test_size=test_size, random_state=random_state
        )

        if self.theta is None:
            self.theta = np.zeros((self.X_train.shape[1], 1))

    def fit(self, X, y):
        assert X.shape[0] == y.shape[0], "Las dimensiones de X e y no coinciden"
        self.X_train = X
        self.y_train = y.reshape(-1, 1)
        if self.include_bias and not np.all(X[:, 0] == 1):
            self.X_train = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        if self.theta is None:
            self.theta = np.zeros((self.X_train.shape[1], 1))

    def get_j(self, X, y, theta):
        m = X.shape[0]
        error = X.dot(theta) - y
        return (1/(2*m)) * np.sum(error**2)

    def get_gradiente(self, X, y, theta, lambda_=0.0):
        m = X.shape[0]
        grad = (1/m) * X.T.dot(X.dot(theta) - y)
        grad[1:] += (lambda_/m) * theta[1:]  # Regularización sin bias
        return grad

    def descenso_gradiente(self, alpha=0.01, epsilon=1e-6, maxiter=1000, batch_size=32, lambda_=0.0):
        self.__historial = []
        theta = self.theta.copy()
        m = self.X_train.shape[0]

        for epoch in range(maxiter):
            indices = np.random.permutation(m)
            X_shuffled = self.X_train[indices]
            y_shuffled = self.y_train[indices]

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                grad = self.get_gradiente(X_batch, y_batch, theta, lambda_)
                theta -= alpha * grad

            costo = self.get_j(self.X_train, self.y_train, theta)
            self.__historial.append(costo)

            if len(self.__historial) > 1 and abs(self.__historial[-2] - self.__historial[-1]) < epsilon:
                logging.info(f"Converged at epoch {epoch}")
                break

        self.theta = theta

    def cross_validate(self, X, y, folds=5, alpha=0.01, maxiter=100, lambda_=0.0):
        kf = KFold(n_splits=folds)
        r2_scores = []
        X_poly = self._add_polynomial_features(X)

        if self.include_bias:
            X_poly = np.concatenate((np.ones((X_poly.shape[0], 1)), X_poly), axis=1)

        for train_idx, val_idx in kf.split(X_poly):
            X_train, X_val = X_poly[train_idx], X_poly[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            self.fit(X_train, y_train)
            self.descenso_gradiente(alpha=alpha, maxiter=maxiter, lambda_=lambda_)

            y_pred = X_val.dot(self.theta)
            r2 = self._calculate_r2(y_val.reshape(-1, 1), y_pred)
            r2_scores.append(r2)

        return np.mean(r2_scores)

    def score(self):
        y_pred = self.X_test.dot(self.theta)
        return self._calculate_r2(self.y_test, y_pred)

    def _calculate_r2(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / ss_tot)

    def get_residuales(self, X=None, y=None):
        if X is None or y is None:
            return self.y_test - self.X_test.dot(self.theta)

        X_poly = self._add_polynomial_features(X)
        if self.include_bias:
            X_poly = np.concatenate((np.ones((X_poly.shape[0], 1)), X_poly), axis=1)

        return y.reshape(-1, 1) - X_poly.dot(self.theta)

    def historial_costos(self):
        return self.__historial

    def graficar_costos(self):
        plt.plot(self.__historial)
        plt.xlabel("Iteraciones")
        plt.ylabel("Costo")
        plt.title("Curva de aprendizaje")
        plt.grid()
        plt.show()

    def graficar_residuales(self):
        residuales = self.get_residuales()
        plt.scatter(range(len(residuales)), residuales)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Índice")
        plt.ylabel("Residual")
        plt.title("Gráfico de residuales")
        plt.grid()
        plt.show()

    def save(self, path):
        np.save(path, self.theta)

    def load(self, path):
        self.theta = np.load(path)
