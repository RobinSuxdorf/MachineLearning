import pytest
import pandas as pd

from preprocessing import split_features_and_target

def _generate_mock_data() -> pd.DataFrame:
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9]
    })

    return df

def _generate_expected_mock_results() -> tuple[pd.DataFrame, pd.Series]:
    expected_df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })

    expected_target = pd.Series([7, 8, 9])

    return expected_df, expected_target

def test_data_splitter_with_explicit_column() -> None:
    df = _generate_mock_data()

    expected_df, expected_target = _generate_expected_mock_results()

    X, y = split_features_and_target(df, "C")

    assert X.equals(expected_df)
    assert y.equals(expected_target)

def test_data_splitter_without_explicit_column() -> None:
    df = _generate_mock_data()

    expected_df, expected_target = _generate_expected_mock_results()

    X, y = split_features_and_target(df)

    assert X.equals(expected_df)
    assert y.equals(expected_target)

def test_data_splitter_without_existing_column() -> None:
    df = _generate_mock_data()

    with pytest.raises(ValueError, match=r"Target column does not exist in DataFrame."):
        split_features_and_target(df, "D")