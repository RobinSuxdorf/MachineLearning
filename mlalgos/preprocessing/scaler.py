import numpy as np
from typing import Optional
from mlalgos import ArrayLike
from mlalgos.helpers import check_array


class StandardScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        """
        Initializes the StandardScaler with options to center and scale the data.

        Args:
            with_mean (bool): If True, center the data before scaling. Defaults to True.
            with_std (bool): If True, scale the data to unit variance. Defaults to True.
        """
        self._with_mean = with_mean
        self._with_std = with_std

        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    @property
    def mean_(self) -> np.ndarray:
        """
        Returns the mean value for each feature after fitting.

        Returns:
            np.ndarray: The mean value for each feature.
        """
        if self._mean is None:
            raise AttributeError("mean_ is not set. You need to call 'fit' first.")
        return self._mean

    @property
    def std_(self) -> np.ndarray:
        """
        Returns the standard deviation for each feature after fitting.

        Returns:
            np.ndarray: The standard deviation for each feature.
        """
        if self._std is None:
            raise AttributeError("std_ is not set. You need to call 'fit' first.")
        return self._std

    def fit(self, X: ArrayLike) -> None:
        """
        Computes the mean and standard deviation for each feature in the dataset.

        Args:
            X (ArrayLike): The input data to fit, where each row represents a sample and each column represents a feature.
        """
        X = check_array(X)

        if self._with_mean:
            self._mean = np.mean(X, axis=0)

        if self._with_std:
            self._std = np.std(X, axis=0)

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transforms the input data using the mean and standard deviation computed during fitting.

        Args:
            X (ArrayLike): The data to transform, where each row represents a sample and each column represents a feature.

        Returns:
            np.ndarray: The transformed data.
        """
        X = check_array(X)

        if self._with_mean and self._mean is not None:
            X = X - self._mean

        if self._with_std and self._std is not None:
            X = X / self._std

        return X

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        """
        Fits the scaler to the data and then transforms it.

        Args:
            X (ArrayLike): The data to fit and transform, where each row represents a sample and each column represents a feature.

        Returns:
            np.ndarray: The transformed data.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        """
        Reverts the scaling applied by the transform method.

        Args:
            X (ArrayLike): The data to inverse transform, where each row represents a sample and each column represents a feature.

        Returns:
            np.ndarray: The data in its original form before scaling.
        """
        X = check_array(X)

        if self._with_std and self._std is not None:
            X = X * self._std

        if self._with_mean and self._mean is not None:
            X = X + self._mean

        return X


class MinMaxScaler:
    """
    Scales features to a given range.
    """

    def __init__(self, feature_range: tuple[float, float] = (0, 1)) -> None:
        """
        Initializes the MinMaxScaler with the given feature range.

        Args:
            feature_range (tuple[float, float]): Desired range of transformed data. Defaults to (0, 1).
        """
        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                "The first value of feature_range must be smaller than the second value."
            )

        self._min = feature_range[0]
        self._max = feature_range[1]
        self._data_min: Optional[np.ndarray] = None
        self._data_max: Optional[np.ndarray] = None
        self._scale: Optional[float] = None

    @property
    def min_(self) -> float:
        """
        Returns the minimum value of the desired feature range.

        Returns:
            float: The minimum value of the desired feature range.
        """
        return self._min

    @property
    def max_(self) -> float:
        """
        Returns the maximum value of the desired feature range.

        Returns:
            float: The maximum value of the desired feature range.
        """
        return self._max

    @property
    def data_min_(self) -> float:
        """
        Returns the per-feature minimum observed in the data after fitting.

        Returns:
            float: The per-feature minimum observed in the data.
        """
        if self._data_min is None:
            raise AttributeError("data_min_ is not set. You need to call 'fit' first.")
        return self._data_min

    @property
    def data_max_(self) -> float:
        """
        Returns the per-feature maximum observed in the data after fitting.

        Returns:
            float: The per-feature maximum observed in the data.
        """

        if self._data_max is None:
            raise AttributeError("data_max_ is not set. You need to call 'fit' first.")
        return self._data_max

    @property
    def scale_(self) -> float:
        """
        Returns the per-feature scaling factor used during transformation.

        Returns:
            float: The per-feature scaling factor.
        """
        if self._scale is None:
            raise AttributeError("scale_ is not set. You need to call 'fit' first.")
        return self._scale

    def fit(self, X: ArrayLike) -> None:
        """
        Computes the minimum and maximum to be used for later scaling.

        Args:
            X (ArrayLike): The input data to compute the per-feature minimum and maximum.
        """
        X = check_array(X)

        self._data_min = X.min(axis=0)
        self._data_max = X.max(axis=0)

        self._scale = (self._max - self._min) / (self.data_max_ - self.data_min_)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Scales the input data according to the feature range specified during initialization.

        Args:
            X (ArrayLike): The data to scale.

        Returns:
            np.ndarray: The transformed data.
        """
        X = check_array(X)

        return (X - self._data_min) * self._scale + self._min

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        """
        Fits the scaler to the data and then transforms it.

        Args:
            X (ArrayLike): The data to fit and transform.

        Returns:
            np.ndarray: The transformed data.
        """

        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        """
        Reverts the scaling applied by the transform method.

        Args:
            X (ArrayLike): The data to inverse transform.

        Returns:
            np.ndarray: The data in its original form before scaling.
        """
        X = check_array(X)

        return (X - self._min) / self._scale + self._data_min
