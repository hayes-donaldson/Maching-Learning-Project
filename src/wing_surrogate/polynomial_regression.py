from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression


def build_model(degree=2):

    model = Pipeline([
        ("polynomial_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("standard_scaler", StandardScaler()),
        ("linear_regression", LinearRegression())
    ])

    return model