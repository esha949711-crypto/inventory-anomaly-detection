"""Preprocessing utilities shared by training and Flask inference."""
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from feature_engineering import FEATURE_COLUMNS


def build_preprocessor() -> Pipeline:
    # RobustScaler reduces the influence of extreme values without removing them.
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())])


def transform_features(df, preprocessor):
    return preprocessor.transform(df[FEATURE_COLUMNS])
