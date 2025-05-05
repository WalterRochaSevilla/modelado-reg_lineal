import numpy as np
from sklearn.model_selection import train_test_split

class Regresion:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.theta = None
        self.__historial = []

    def fit(self, X, y, test_size=0.2, random_state=42):
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Guardar subconjuntos
        self.X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
        self.X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)
        self.y_train = y_train.reshape(-1, 1)
        self.y_test = y_test.reshape(-1, 1)

        # Inicializar parámetros
        self.theta = np.zeros((self.X_train.shape[1], 1))

    def get_j(self, X, y, t):
        m = X.shape[0]
        t = t.reshape(-1, 1)
        h = X.dot(t)
        error = h - y
        j = (1 / (2 * m)) * np.sum(error ** 2)
        return j

    def get_gradiente(self, X, y, t):
        m = X.shape[0]
        t = t.reshape(-1, 1)
        h = X.dot(t)
        error = h - y
        grad = (1 / m) * X.T.dot(error)
        return grad.flatten()

    def descenso_gradiente(self, alpha=0.01, epsilon=1e-6, maxiter=1000):
        t = self.theta.flatten()
        i = 0
        self.__historial = [self.get_j(self.X_train, self.y_train, t)]

        while True:
            grad = self.get_gradiente(self.X_train, self.y_train, t)
            t = t - alpha * grad
            costo = self.get_j(self.X_train, self.y_train, t)
            self.__historial.append(costo)

            if abs(self.__historial[-2] - self.__historial[-1]) < epsilon:
                break
            if maxiter is not None and i >= maxiter:
                break
            i += 1

        self.theta = t.reshape(-1, 1)

    def score(self):
        y_pred = self.X_test.dot(self.theta)
        ss_res = np.sum((self.y_test - y_pred) ** 2)
        ss_tot = np.sum((self.y_test - np.mean(self.y_test)) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2

    def historial_costos(self):
        return self.__historial
