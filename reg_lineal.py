import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures

class Regresion:
    def __init__(self, theta_init=None, include_bias=True):
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.include_bias = include_bias
        self.theta = theta_init.reshape(-1, 1) if theta_init is not None else None
        self.__historial = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.selected_features = [0, 1, 3, 5]  # Índices para x0, x1, x1², x3²

    def _add_polynomial_features(self, X):
        """Añade términos polinómicos y selecciona características específicas"""
        X_poly = self.poly.fit_transform(X)
        return X_poly[:, self.selected_features]

    def fit(self, X, y, test_size=0.2, random_state=42):
        # Generar características polinómicas
        X_poly = self._add_polynomial_features(X)
        
        # Añadir columna de unos si se requiere bias
        if self.include_bias:
            X_poly = np.concatenate((np.ones((X_poly.shape[0], 1)), X_poly), axis=1)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_poly, y, test_size=test_size, random_state=random_state
        )

        # Guardar subconjuntos
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train.reshape(-1, 1)
        self.y_test = y_test.reshape(-1, 1)

        # Inicializar parámetros si no se proporcionaron
        if self.theta is None:
            self.theta = np.zeros((self.X_train.shape[1], 1))

    def get_j(self, X, y, theta):
        """Función de costo"""
        m = X.shape[0]
        error = X.dot(theta) - y
        return (1/(2*m)) * np.sum(error**2)

    def get_gradiente(self, X, y, theta):
        """Cálculo del gradiente"""
        m = X.shape[0]
        return (1/m) * X.T.dot(X.dot(theta) - y)

    def descenso_gradiente(self, alpha=0.01, epsilon=1e-6, maxiter=1000, batch_size=32):
        """Descenso de gradiente estocástico con mini-lotes"""
        self.__historial = []
        theta = self.theta.copy()
        m = self.X_train.shape[0]

        for epoch in range(maxiter):
            # Mezclar datos en cada época
            indices = np.random.permutation(m)
            X_shuffled = self.X_train[indices]
            y_shuffled = self.y_train[indices]

            # Procesar en mini-lotes
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                grad = self.get_gradiente(X_batch, y_batch, theta)
                theta -= alpha * grad

            # Calcular y guardar costo
            costo = self.get_j(self.X_train, self.y_train, theta)
            self.__historial.append(costo)

            # Criterio de parada
            if len(self.__historial) > 1 and abs(self.__historial[-2] - self.__historial[-1]) < epsilon:
                break

        self.theta = theta

    def cross_validate(self, X, y, folds=5, alpha=0.01, maxiter=100):
        """Validación cruzada de K-folds"""
        kf = KFold(n_splits=folds)
        r2_scores = []
        X_poly = self._add_polynomial_features(X)
        
        if self.include_bias:
            X_poly = np.concatenate((np.ones((X_poly.shape[0], 1)), X_poly), axis=1)

        for train_idx, val_idx in kf.split(X_poly):
            X_train, X_val = X_poly[train_idx], X_poly[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Entrenar modelo
            self.fit(X_train, y_train, test_size=0)
            self.descenso_gradiente(alpha=alpha, maxiter=maxiter)
            
            # Calcular R²
            y_pred = X_val.dot(self.theta)
            r2 = self._calculate_r2(y_val, y_pred)
            r2_scores.append(r2)
        
        return np.mean(r2_scores)

    def score(self):
        """Métrica R² para datos de prueba"""
        y_pred = self.X_test.dot(self.theta)
        return self._calculate_r2(self.y_test, y_pred)

    def _calculate_r2(self, y_true, y_pred):
        """Cálculo interno de R²"""
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / ss_tot)

    def get_residuales(self, X=None, y=None):
        """Obtener residuales"""
        if X is None or y is None:
            return self.y_test - self.X_test.dot(self.theta)
        
        X_poly = self._add_polynomial_features(X)
        if self.include_bias:
            X_poly = np.concatenate((np.ones((X_poly.shape[0], 1)), X_poly), axis=1)
            
        return y - X_poly.dot(self.theta)

    def historial_costos(self):
        return self.__historial