from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from text_preprocessor import TextPreprocessor
from typing import Union, Optional


def create_target_pipeline(
    target_col: str,
    anomaly_type: str,
    text_preprocessor: bool = True,
    label_encoder: bool = True
) -> Pipeline:
    """
    Create a preprocessing pipeline for the target column.

    Args:
        target_col (str): The name of the target column to preprocess.
        anomaly_type (str): The type of anomaly being processed, which determines 
                            specific preprocessing logic in the pipeline.
        text_preprocessor (bool, optional): Whether to include a text preprocessor 
                                            in the pipeline. Defaults to True.
        label_encoder (bool, optional): Whether to include a label encoder in the 
                                        pipeline. Defaults to True.

    Returns:
        Pipeline: A scikit-learn Pipeline object that applies the specified 
                  preprocessing steps to the target column.
    """
    pipe = []
    pipe.append(
        ('target_preprocessor', TargetColumnPreprocessor(target_col, anomaly_type)))
    if text_preprocessor:
        pipe.append(
            ('text_preprocessor', TextPreprocessor(columns=[target_col])))
    if label_encoder:
        pipe.append(
            ('label_encoder', DataFrameLabelEncoder(target_col=target_col)))
    return Pipeline(pipe)


class TargetColumnPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, target_col: str, anomaly_type: str) -> None:
        """
        Initialize the preprocessor with the target column and anomaly type.

        Args:
            target_col (str): The name of the target column.
            anomaly_type (str): The type of anomaly being processed.
        """
        self.target_col = target_col
        self.anomaly_type = anomaly_type

    def fit(self, X: Union[pd.DataFrame, pd.Series], y: Optional[pd.Series] = None) -> "TargetColumnPreprocessor":
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """
        Transform the input DataFrame or Series.

        Args:
            X (pd.DataFrame or pd.Series): The input data.

        Returns:
            pd.DataFrame or pd.Series: The transformed data.
        """
        if isinstance(X, pd.Series):
            X = X.to_frame(name=self.target_col)

        X = X.copy()

        if self.target_col in ["B_NUMER", "Z_SPEDITER", "IM_TK_HEAT"]:
            X[self.target_col] = X[self.target_col].map(
                {None: 0.0, ' ': 0.0, 'X': 1.0}).fillna(X[self.target_col])
            X[self.target_col] = pd.to_numeric(
                X[self.target_col], errors='coerce')

        elif self.target_col == "HEATING_TYPE":
            if "WITH_NULLS" in self.anomaly_type:
                X[self.target_col] = pd.to_numeric(
                    X[self.target_col], errors='coerce').fillna(0.0)
            else:
                X[self.target_col] = pd.to_numeric(
                    X[self.target_col], errors='coerce')

        elif self.target_col in ["EX_CO_TYP", "IM_CO_TYP", "T_CH_TYP", "HEATING_NOTE"]:
            X[self.target_col] = X[self.target_col].astype(str)

        elif self.target_col in ["TEMP_FROM", "TEMP_TO", "TEMPERATURE"]:
            X[self.target_col] = X[self.target_col].apply(lambda x: int(str(x).replace("°", "").replace("C", "").replace('+','').strip()) 
                    if pd.notnull(x) and str(x).replace("°", "").replace("C", "").replace('+','').strip().isdigit() 
                    else -1000 # np.nan # causes problems later, create dummy value that is far away from everything
                    )
            X[self.target_col] = pd.to_numeric(
                X[self.target_col], errors='coerce')

        elif self.target_col in ["IM_SP_REF", "EX_SP_REF"]:
            X[self.target_col] = X[self.target_col].astype(str)
            X[self.target_col] = X[self.target_col].apply(
                lambda x: 0 if pd.isna(x) or str(x).strip() == '' else 1)
            # X[self.target_col] = pd.to_numeric(
            #     X[self.target_col], errors='coerce')

        return X if isinstance(X, pd.DataFrame) else X[self.target_col]


class DataFrameLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, target_col: str) -> None:
        """
        Initialize the transformer with the target column.

        Args:
            target_col (str): The name of the target column to encode.
        """
        self.target_col = target_col
        self.label_encoder = LabelEncoder()

    def fit(self, X: Union[pd.DataFrame, pd.Series], y: Optional[pd.Series] = None) -> "DataFrameLabelEncoder":
        """
        Fit the LabelEncoder to the target column.

        Args:
            X (pd.DataFrame or pd.Series): The input data.
            y: Ignored.

        Returns:
            self
        """
        if isinstance(X, pd.Series):
            X = X.to_frame(name=self.target_col)

        if self.target_col in X.columns:
            self.label_encoder.fit(X[self.target_col].astype(str))
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """
        Transform the target column using the fitted LabelEncoder.

        Args:
            X (pd.DataFrame or pd.Series): The input data.

        Returns:
            pd.DataFrame or pd.Series: The transformed data.
        """
        if isinstance(X, pd.Series):
            X = X.to_frame(name=self.target_col)

        X = X.copy()
        if self.target_col in X.columns:
            X[self.target_col] = self.label_encoder.transform(
                X[self.target_col].astype(str))

        return X if isinstance(X, pd.DataFrame) else X[self.target_col]

    def inverse_transform(self, X: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """
        Inverse transform the target column to its original values.

        Args:
            X (pd.DataFrame or pd.Series): The input data.

        Returns:
            pd.DataFrame or pd.Series: The data with the target column inverse transformed.
        """
        if isinstance(X, pd.Series):
            X = X.to_frame(name=self.target_col)

        X = X.copy()
        if self.target_col in X.columns:
            X[self.target_col] = self.label_encoder.inverse_transform(
                X[self.target_col])

        return X if isinstance(X, pd.DataFrame) else X[self.target_col]
