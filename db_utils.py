from dotenv import load_dotenv
import oracledb
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.preprocessors.text_preprocessor import TextPreprocessor
from plot_utils import plot_target_distribution


def get_data_for_anomaly_type(cursor, anomaly_type, limit=100000):
    """
    Retrieve data for a specific anomaly type from the database.

    Args:
        cursor: The database cursor to execute queries.
        anomaly_type (str): The type of anomaly to retrieve data for (e.g., 'BHT', 'HEATING_TYPE').

    Returns:
        tuple: A tuple containing:
            - df (pd.DataFrame): The retrieved data as a pandas DataFrame.
            - X_cols (list): List of feature column names.
            - y_col (str): The target column name.
    """
    if anomaly_type == 'BHT':
        query = f"""
        SELECT
            *
        FROM BHT_TABLE
        """
        target_col = "BHT"
        group_keys = ["CLIENT", "SENDER"]
    elif anomaly_type == 'OTHER':
        ...
        # REDACTED

    # Execute the query and load data into a DataFrame
    df = pd.read_sql(query, con=cursor.connection)

    # Preprocess group keys
    # Moved to the TextPreprocessor class
    # for key in group_keys:
    #     df[key] = df[key].astype(str)

    return df, group_keys, target_col
