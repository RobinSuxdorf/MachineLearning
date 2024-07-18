from typing import Optional
import pandas as pd


def split_features_and_target(
    df: pd.DataFrame, target_column: Optional[str] = None
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Splits the DataFrame into features and target based on the specified target column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (Optional[str]): The column name to be used as the target. If None, the ast column will be used.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing the features DataFrame and the target Series.
    """
    if target_column is None:
        target_column = df.columns[-1]

    if target_column not in df.columns:
        raise ValueError("Target column does not exist in DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y
