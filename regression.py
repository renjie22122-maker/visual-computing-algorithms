import numpy as np

def nsolve(X, y):
    X_aug = np.c_[np.ones(X.shape[0]), X]
    w = np.linalg.pinv(X_aug.T @ X_aug) @ (X_aug.T @ y)
    return w

def line_fit(X, y):
    w = nsolve(X, y)
    X_aug = np.c_[np.ones(X.shape[0]), X]
    y_pred = X_aug @ w
    l2_error = np.mean((y - y_pred) ** 2)
    return (w, l2_error)

def poly_fit(X, y):
    n_samples, n_features = X.shape
    poly_terms = []
    for i in range(n_features):
        poly_terms.append(X[:, i:i + 1])
        poly_terms.append(X[:, i:i + 1] ** 2)
        for j in range(i + 1, n_features):
            poly_terms.append((X[:, i] * X[:, j]).reshape(-1, 1))
    X_poly = np.hstack(poly_terms)
    X_poly = np.c_[np.ones(n_samples), X_poly]
    w = np.linalg.pinv(X_poly.T @ X_poly) @ (X_poly.T @ y)
    y_pred = X_poly @ w
    l2_error = np.mean((y - y_pred) ** 2)
    return (w, l2_error)
