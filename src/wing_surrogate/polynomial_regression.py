from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression


def build_model(degree=2):
    return Pipeline([
        ("polynomial_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("standard_scaler", StandardScaler()),
        ("linear_regression", LinearRegression()),
    ])
